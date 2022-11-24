//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

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
    Tlrmvmhip<complex<float>, hipComplex> cudatlrmvmptr(tlrmvmconfig);
    cudatlrmvmptr.StreamInit(streams);
    cudatlrmvmptr.MemoryInit();
    cudatlrmvmptr.SetTransposeConjugate(false, false);
    cudatlrmvmptr.TryConjugateXvec();

    // do the computation and send results back to cpu instance.
    cudatlrmvmptr.MVM();

    cudatlrmvmptr.TryConjugateResults();
    cudatlrmvmptr.CopyBackResults();

    CFPPCMatrix seismicpcmat(datafolder, threshold, nb, problem, originM, originN);
    auto densemat = seismicpcmat.GetDense();
    auto hy = Matrix<complex<float>>(cudatlrmvmptr.tlrmvmcpu->finalresults, cudatlrmvmptr.tlrmvmcpu->config.originM, 1);
    auto denseout = densemat * cudatlrmvmptr.tlrmvmcpu->xmat;
    cout << "====================================================" << endl;
    cout << "Test TLR-MVM conjugate single call Implementation. " << endl;
    cout << "dense results vs tlrmvm results " << hy.allclose(denseout) << endl;
    cout << "====================================================" << endl;
    cudatlrmvmptr.MemoryFree();
    return 0;
}
