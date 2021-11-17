#include <stdio.h>
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
    double bytesprocessed;
    size_t granksum;
    auto argparser = ArgsParser(argc, argv);
    originM = argparser.getint("M");
    originN = argparser.getint("N");
    nb = argparser.getint("nb");
    acc = argparser.getstring("errorthreshold");
    problemname = argparser.getstring("problemname");
    datafolder = argparser.getstring("datafolder");
    int streamsize = argparser.getint("streamsize");
    char rpath[100]; 
    sprintf(rpath, "%s/%s_Rmat_nb%d_acc%s.bin", datafolder.c_str(), problemname.c_str(), nb, acc.c_str());
    rankfile = string(rpath);
    int mpirank = 0;
    TlrmvmConfig tlrmvmconfig(originM, originN, nb, datafolder, acc, problemname);
    Matrix<int> maskmat(tlrmvmconfig.Mtg, tlrmvmconfig.Ntg);
    maskmat.Fill(1);
    tlrmvmconfig.Maskmat = maskmat;
    TlrmvmGPU<complex<float>, cuComplex> tlrmvmptr(tlrmvmconfig);
    tlrmvmptr.MemoryInit();
    tlrmvmptr.StreamInit(streamsize);
    cudaDeviceSynchronize();
    tlrmvmptr.Phase1();
    tlrmvmptr.Phase2();
    tlrmvmptr.Phase3();
    cudaDeviceSynchronize();
    tlrmvmptr.CopyBackResults();
    cudaDeviceSynchronize();
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
    auto hy = Matrix<complex<float>>(tlrmvmptr.h_y, tlrmvmptr.originM, 1);
    cout << " Phase 3 Correctness : "<< hy.allclose(y_pc) << endl;
    tlrmvmptr.MemoryFree();
    tlrmvmptr.StreamDestroy();
    
    return 0;
}


