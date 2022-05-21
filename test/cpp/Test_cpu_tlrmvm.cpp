#include <string>
#include <vector>
#include <chrono>
using namespace std;

// include common component
#include <common/Common.hpp>

// include tlrmvm component
#include <tlrmvm/Tlrmvm.hpp>

int main (int argc, char ** argv){
    auto argparser = ArgsParser(argc, argv);
    auto originM = argparser.getint("M");
    auto originN = argparser.getint("N");
    auto nb = argparser.getint("nb");
    auto threshold = argparser.getstring("threshold");
    auto problem = argparser.getstring("problem");
    auto datafolder = argparser.getstring("datafolder");
    TlrmvmConfig tlrmvmconfig(originM, originN, nb, datafolder, threshold, problem);
    Matrix<int> maskmat(tlrmvmconfig.Mtg, tlrmvmconfig.Ntg);
    maskmat.Fill(1);
    tlrmvmconfig.Maskmat = maskmat;
    TlrmvmCPU<complex<float>> tlrmvmptr(tlrmvmconfig);
    tlrmvmptr.MemoryInit();
    auto xmat = tlrmvmptr.xmat;
    tlrmvmptr.Phase1();
    tlrmvmptr.Phase2();
    tlrmvmptr.Phase3();
    CFPPCMatrix seismicpcmat(datafolder, threshold, nb, problem, originM, originN);
    seismicpcmat.setX(tlrmvmptr.xmat);
    auto densemat = seismicpcmat.GetDense();
    Matrix<complex<float>> yv_pc = seismicpcmat.Phase1();
    auto hyv = Matrix<complex<float>>(tlrmvmptr.p1ptrs.y, tlrmvmptr.config.workmatgranksum, 1);
    cout << "====================================================" << endl;
    cout << "Test TLR-MVM Implementation. " << endl;
    cout << " Phase 1 Correctness : " << hyv.allclose(yv_pc) << endl;
    Matrix<complex<float>> yu_pc = seismicpcmat.Phase2();
    auto hyu = Matrix<complex<float>>(tlrmvmptr.p3ptrs.x, tlrmvmptr.config.workmatgranksum, 1);
    cout << " Phase 2 Correctness : " << hyu.allclose(yu_pc) << endl;
    Matrix<complex<float>> y_pc = seismicpcmat.Phase3();
    auto hy = Matrix<complex<float>>(tlrmvmptr.p3ptrs.y, tlrmvmptr.config.paddingM, 1);
    cout << " Phase 3 Correctness : "<< hy.allclose(y_pc) << endl;
    auto denseout = densemat * xmat;
    cout << "dense results vs tlrmvm results " << hy.allclose(denseout) << endl;
    cout << "====================================================" << endl;
    tlrmvmptr.MemoryFree();
    return 0;
}


