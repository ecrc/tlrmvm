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
    auto ranksize = argparser.getint("ranksize");
    auto loopsize = argparser.getint("loopsize");

    // rank size should be smaller than nb.
    TlrmvmConfig tlrmvmconfig(originM, originN, nb, ranksize);
    /********************************
     * cuda instance
     ********************************/
    TlrmvmhipConstRank<complex<float>, hipComplex> cudatlrmvmptr(tlrmvmconfig);
    cudatlrmvmptr.StreamInit(0);
    cudatlrmvmptr.MemoryInit();
    cudatlrmvmptr.SetTransposeConjugate(false, false);
    cudatlrmvmptr.TryConjugateXvec();

    // time
    hipEvent_t start;
    hipEvent_t stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    vector<double> rawtime;
    float milliseconds = 0;

    for(int i=0; i<loopsize; i++){
        hipEventRecord(start);
        // do the computation and send results back to cpu instance.
        cudatlrmvmptr.MVM();
        hipEventRecord(stop);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&milliseconds, start, stop);
        rawtime.push_back(milliseconds * 1e-3);
    }

    cudatlrmvmptr.TryConjugateResults();
    cudatlrmvmptr.CopyBackResults();

    std::sort(rawtime.begin(), rawtime.end());
    int nruns = rawtime.size();
    cout << "Median Time " << rawtime[nruns/2] * 1e6  << " us."<< endl;
    double bytes = TLRMVMBytesProcessed<complex<float>>(cudatlrmvmptr.config.granksum, nb,
            originM, originN);
    cout << "Bandwidth: " << bytes / rawtime[nruns/2] * 1e-9 << " GB/s." << endl;

    cudatlrmvmptr.MemoryFree();
    return 0;
}
