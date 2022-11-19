#include <string>
#include <vector>
#include <chrono>
#include <algorithm>
#include <mpi.h>
#include <common/Common.hpp>
#include <tlrmvm/Tlrmvm.hpp>
#define real float
using namespace std;
int main (int argc, char ** argv){
    int originM;
    int originN;
    int nb;
    int loopsize;
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
    loopsize = argparser.getint("loopsize");
    acc = argparser.getstring("threshold");
    problemname = argparser.getstring("problem");
    datafolder = argparser.getstring("datafolder");
    char rpath[300];
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
            if (j % size == rank)
            maskmat.SetElem(i,j,1);
        }
    }
    tlrmvmconfig.UpdateMaskmat(maskmat);
    TlrmvmCPU<real> tlrmvmptr(tlrmvmconfig);
    auto finalbuffer = new real[tlrmvmptr.config.paddingM];
    memset(finalbuffer, 0, sizeof(real) * tlrmvmptr.config.paddingM);
    tlrmvmptr.MemoryInit();
    auto curx = Matrix<real>(tlrmvmptr.config.originN, 1);
    curx.Fill(0.1);
    tlrmvmptr.setX(curx.RawPtr(), curx.Shape()[0]);
    for(int i=0; i<loopsize; i++){
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::steady_clock::now();
        tlrmvmptr.MVM();
        tlrmvmptr.CopyToFinalresults();
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Reduce(tlrmvmptr.finalresults,
                   finalbuffer, tlrmvmptr.config.paddingM,
                   MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        auto end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        timestat.push_back(elapsed_time*1e-6);
    }
    vector<double> mergetime(timestat.size(), 0);
    MPI_Allreduce(timestat.data(), mergetime.data(),
                  timestat.size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if(rank == 0){
#ifndef USE_NEC
        FPPCMatrix pcmat(datafolder, acc, nb, problemname, originM, originN);
        pcmat.setX(tlrmvmptr.xmat);
        pcmat.GetDense();
        Matrix<real> yv_pc = pcmat.Phase1();
        auto hyv = Matrix<real>(tlrmvmptr.p1ptrs.y, tlrmvmptr.config.workmatgranksum, 1);
//        cout << " Phase 1 Correctness : " << hyv.allclose(yv_pc) << endl;
        Matrix<real> yu_pc = pcmat.Phase2();
        auto hyu = Matrix<real>(tlrmvmptr.p3ptrs.x, tlrmvmptr.config.workmatgranksum, 1);
//        cout << " Phase 2 Correctness : " << hyu.allclose(yu_pc) << endl;
        Matrix<real> y_pc = pcmat.Phase3();
        auto hy = Matrix<real>(finalbuffer, tlrmvmptr.config.originM, 1);
        auto denseout = pcmat.GetDense() * tlrmvmptr.xmat;
        cout << " Check MPI Phase 3 Correctness : "<< hy.allclose(denseout) << endl;
#endif
        std::sort(mergetime.begin(), mergetime.end());
        int N = mergetime.size();
        cout << "median " << mergetime[N / 2] * 1e6 << " us."<< endl;
        double bytes = TLRMVMBytesProcessed<real>(tlrmvmptr.config.granksum,
                                                   tlrmvmptr.config.nb, originM, originN);
        cout << "U and V bases size: " << bytes * 1e-6 << " MB." << endl;
        cout << "Bandwidth " << bytes / mergetime[N/2] * 1e-9 << " GB/s" << endl;
    }
    tlrmvmptr.MemoryFree();
    delete[] finalbuffer;
    MPI_Finalize();
    return 0;
}
