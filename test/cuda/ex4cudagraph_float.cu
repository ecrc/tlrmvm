#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <memory.h>
#include <cuda.h>
#include <mpi.h>

#include <complex>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h> // if you need CUBLAS v2, include before magma.h

#include "common/AppUtil.h"

// include common component
#include <common/Common.h>
using namespace tlrmat;

// include tlrmvm component
#include <tlrmvm/Tlrmvm.h>
using namespace tlrmvm;
using namespace cudatlrmvm;
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
    auto argparser = ArgsParser(argc, argv);
    originM = argparser.getint("M");
    originN = argparser.getint("N");
    nb = argparser.getint("nb");
    acc = argparser.getstring("errorthreshold");
    problemname = argparser.getstring("problemname");
    datafolder = argparser.getstring("datafolder");
    int streamsize = argparser.getint("streamsize");
    int loopsize = argparser.getint("loopsize");
    char rpath[100]; 
    sprintf(rpath, "%s/%s_Rmat_nb%d_acc%s.bin", datafolder.c_str(), problemname.c_str(), nb, acc.c_str());
    rankfile = string(rpath);
    TlrmvmConfig tlrmvmconfig(originM, originN, nb, datafolder, acc, problemname);
    Matrix<int> maskmat(tlrmvmconfig.Mtg, tlrmvmconfig.Ntg);
    maskmat.Fill(1);
    tlrmvmconfig.Maskmat = maskmat;
    TlrmvmGPU<float, float> tlrmvmptr(tlrmvmconfig);
    tlrmvmptr.MemoryInit();
    tlrmvmptr.StreamInit(streamsize);
    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    vector<double> rawtime;
    for(int i=0; i<loopsize; i++){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        tlrmvmptr.MVM();
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        rawtime.push_back(milliseconds*1e-3);
    }
    tlrmvmptr.CopyBackResults();
    cudaDeviceSynchronize();
    FPPCMatrix pcmat(datafolder, acc, nb, problemname, originM, originN);
    pcmat.setX(tlrmvmptr.xmat);
    pcmat.GetDense();
    Matrix<float> yv_pc = pcmat.Phase1();
    auto hyv = Matrix<float>(tlrmvmptr.h_yv, tlrmvmptr.workmatgranksum, 1);
    cout << " Phase 1 Correctness : " << hyv.allclose(yv_pc) << endl;
    Matrix<float> yu_pc = pcmat.Phase2();
    auto hyu = Matrix<float>(tlrmvmptr.h_yu, tlrmvmptr.workmatgranksum, 1);
    cout << " Phase 2 Correctness : " << hyu.allclose(yu_pc) << endl;
    Matrix<float> y_pc = pcmat.Phase3();
    auto hy = Matrix<float>(tlrmvmptr.h_y, tlrmvmptr.originM, 1);
    cout << " Phase 3 Correctness : "<< hy.allclose(y_pc) << endl;
    tlrmvmptr.MemoryFree();
    tlrmvmptr.StreamDestroy();
    std::sort(rawtime.begin(), rawtime.end());
    int nruns = rawtime.size();
    cout << "Median Time " << rawtime[nruns/2] * 1e6  << " us."<< endl;
    double bytes = TLRMVMBytesProcessed<float>(tlrmvmptr.granksum, nb, originM, originN);
    cout << "Bandwidth: " << bytes / rawtime[nruns/2] * 1e-9 << " GB/s." << endl;
    return 0;
}


