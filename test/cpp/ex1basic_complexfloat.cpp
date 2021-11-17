#include <string>
#include <vector>
#include <chrono>
using namespace std;

// include common component
#include <common/Common.h>
using namespace tlrmat;

// include tlrmvm component
#include <tlrmvm/Tlrmvm.h>
using namespace tlrmvm;

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
    int mpirank = 0;
    TlrmvmConfig tlrmvmconfig(originM, originN, nb, datafolder, acc, problemname);
    Matrix<int> maskmat(tlrmvmconfig.Mtg, tlrmvmconfig.Ntg);
    maskmat.Fill(1);
    tlrmvmconfig.Maskmat = maskmat;
    TlrmvmCPU<complex<float>> tlrmvmptr(tlrmvmconfig);
    double bytes = TLRMVMBytesProcessed<complex<float>>(tlrmvmptr.granksum, 
    tlrmvmptr.nb, tlrmvmptr.paddingM, tlrmvmptr.paddingN);
    tlrmvmptr.MemoryInit();
    for(int i=0; i<1; i++){
        auto start = std::chrono::steady_clock::now();
        tlrmvmptr.MVM();
        auto end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        timestat.push_back(elapsed_time*1e-6);
    }
    
    CFPPCMatrix seismicpcmat(datafolder, acc, nb, problemname, originM, originN);
    seismicpcmat.setX(tlrmvmptr.xmat);
    seismicpcmat.GetDense();
    Matrix<complex<float>> yv_pc = seismicpcmat.Phase1();
    auto hyv = Matrix<complex<float>>(tlrmvmptr.h_yv, tlrmvmptr.workmatgranksum, 1);
    cout << " Phase 1 Correctness : " << hyv.allclose(yv_pc) << endl;
    Matrix<complex<float>> yu_pc = seismicpcmat.Phase2();
    auto hyu = Matrix<complex<float>>(tlrmvmptr.h_yu, tlrmvmptr.workmatgranksum, 1);
    cout << " Phase 2 Correctness : " << hyu.allclose(yu_pc) << endl;
    Matrix<complex<float>> y_pc = seismicpcmat.Phase3();
    auto hy = Matrix<complex<float>>(tlrmvmptr.h_y, tlrmvmptr.paddingM, 1);
    cout << " Phase 3 Correctness : "<< hy.allclose(y_pc) << endl;
    tlrmvmptr.MemoryFree();
    return 0;
}


