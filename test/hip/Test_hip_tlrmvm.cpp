#include <iostream>
#include <unistd.h>
#include <memory.h>
#include "common/Common.hpp"
#include "tlrmvm/Tlrmvm.hpp"

using namespace hiptlrmvm;

int main (int argc, char ** argv){
    auto argparser = ArgsParser(argc, argv);
    auto originM = argparser.getint("M");
    auto originN = argparser.getint("N");
    auto nb = argparser.getint("nb");
    auto threshold = argparser.getstring("threshold");
    auto problem = argparser.getstring("problem");
    auto datafolder = argparser.getstring("datafolder");
    auto streams = argparser.getint("streams");
    TlrmvmConfig tlrmvmconfig(originM, originN, nb, datafolder, threshold, problem);
    /********************************
     * cuda instance
     ********************************/
    Tlrmvmhip<complex<float>, hipComplex> cudaptr(tlrmvmconfig);
    cudaptr.StreamInit(streams);
    cudaptr.MemoryInit();

    // do the computation and send results back to cpu instance.
    cudaptr.Phase1();
    cudaptr.Phase2();
    cudaptr.Phase3();
    cudaptr.CopyBackResults();

    CFPPCMatrix seismicpcmat(datafolder, threshold, nb, problem, originM, originN);
    auto tlrmvmcpu = cudaptr.tlrmvmcpu;
    seismicpcmat.setX(cudaptr.tlrmvmcpu->xmat);
    auto densemat = seismicpcmat.GetDense();
    Matrix<complex<float>> yv_pc = seismicpcmat.Phase1();
    auto hyv = Matrix<complex<float>>(tlrmvmcpu->p1ptrs.y, tlrmvmcpu->config.workmatgranksum, 1);
    cout << "====================================================" << endl;
    cout << "Test TLR-MVM CUDA Implementation. " << endl;
    cout << " Phase 1 Correctness : " << hyv.allclose(yv_pc) << endl;
    Matrix<complex<float>> yu_pc = seismicpcmat.Phase2();
    auto hyu = Matrix<complex<float>>(tlrmvmcpu->p3ptrs.x, tlrmvmcpu->config.workmatgranksum, 1);
    cout << " Phase 2 Correctness : " << hyu.allclose(yu_pc) << endl;
    Matrix<complex<float>> y_pc = seismicpcmat.Phase3();
    auto hy = Matrix<complex<float>>(tlrmvmcpu->p3ptrs.y, tlrmvmcpu->config.paddingM, 1);
    cout << " Phase 3 Correctness : "<< hy.allclose(y_pc) << endl;
    auto denseout = densemat * cudaptr.tlrmvmcpu->xmat;
    cout << "dense results vs tlrmvm results " << hy.allclose(denseout) << endl;
    cout << "====================================================" << endl;
    cudaptr.MemoryFree();
    return 0;
}
