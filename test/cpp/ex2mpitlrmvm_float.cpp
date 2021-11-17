#include <string>
#include <vector>
#include <chrono>

#include <algorithm>
#include <mpi.h>
// include common component
#include <common/Common.h>
using namespace tlrmat;

// include tlrmvm component
#include <tlrmvm/Tlrmvm.h>
using namespace tlrmvm;
using namespace std;
int main (int argc, char ** argv){
    int originM;
    int originN;
    int nb;
    string acc;
    string datafolder;
    string problemname;
    string rankfile;
    string Ufile;
    string Vfile;
    vector<double> timestat;
    vector<double> bandstat;
    double bytesprocessed;
    size_t granksum;
    auto argparser = ArgsParser(argc, argv);
    originM = argparser.getint("M");
    originN = argparser.getint("N");
    nb = argparser.getint("nb");
    acc = argparser.getstring("errorthreshold");
    problemname = argparser.getstring("problemname");
    datafolder = argparser.getstring("datafolder");
    char rpath[100]; 
    sprintf(rpath, "%s/%s_Rmat_nb%d_acc%s.bin", datafolder.c_str(), problemname.c_str(), nb, acc.c_str());
    rankfile = string(rpath);
    TlrmvmConfig tlrmvmconfig(originM, originN, nb, datafolder, acc, problemname);
    Matrix<int> maskmat(tlrmvmconfig.Mtg, tlrmvmconfig.Ntg);
    MPI_Init(NULL, NULL);
    int rank,size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    maskmat.Fill(0);
    for(int i=0; i<tlrmvmconfig.Mtg; i++){
        for(int j=0; j<tlrmvmconfig.Ntg; j++){
            if (j % size == rank )
            maskmat.SetElem(i,j,1);
        }
    }
    tlrmvmconfig.Maskmat = maskmat;
    TlrmvmCPU<float> tlrmvmptr(tlrmvmconfig);
    double bytes = TLRMVMBytesProcessed<float>(tlrmvmptr.granksum, 
    tlrmvmptr.nb, tlrmvmptr.paddingM, tlrmvmptr.paddingN);
    tlrmvmptr.MemoryInit();
    for(int i=0; i<10000; i++){
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::steady_clock::now();
        tlrmvmptr.MVM();
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Reduce(tlrmvmptr.h_y, tlrmvmptr.h_yout, tlrmvmptr.originM, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        auto end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        timestat.push_back(elapsed_time*1e-6);
    }
    vector<double> mergetime(timestat.size(), 0);
    MPI_Allreduce(timestat.data(), mergetime.data(), timestat.size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if(rank == 0){
        FPPCMatrix seismicpcmat(datafolder, acc, nb, problemname, originM, originN);
        seismicpcmat.setX(tlrmvmptr.xmat);
        seismicpcmat.GetDense();
        Matrix<float> yv_pc = seismicpcmat.Phase1();
        auto hyv = Matrix<float>(tlrmvmptr.h_yv, tlrmvmptr.workmatgranksum, 1);
        // cout << " Phase 1 Correctness : " << hyv.allclose(yv_pc) << endl;
        Matrix<float> yu_pc = seismicpcmat.Phase2();
        auto hyu = Matrix<float>(tlrmvmptr.h_yu, tlrmvmptr.workmatgranksum, 1);
        // cout << " Phase 2 Correctness : " << hyu.allclose(yu_pc) << endl;
        Matrix<float> y_pc = seismicpcmat.Phase3();
        auto hy = Matrix<float>(tlrmvmptr.h_yout, tlrmvmptr.paddingM, 1);
        cout << " Check MPI Phase 3 Correctness : "<< hy.allclose(y_pc) << endl;
        std::sort(mergetime.begin(), mergetime.end());
        int N = mergetime.size();
        cout << "median " << mergetime[N / 2] * 1e6 << " us."<< endl;
        double bytes = TLRMVMBytesProcessed<float>(tlrmvmptr.granksum, tlrmvmptr.nb, originM, originN);
        cout << "Bandwidth " << bytes / mergetime[N/2] * 1e-9 << " GB/s" << endl;
    }
    tlrmvmptr.MemoryFree();
    MPI_Finalize();
    return 0;
}


