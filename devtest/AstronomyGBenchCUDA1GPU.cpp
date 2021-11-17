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
#include <nccl.h>
#include <complex>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h> // if you need CUBLAS v2, include before magma.h


#include "common/Common.h"
#include "common/AppUtil.h"
#include "tlrmvm/Tlrmvm.h"
#include "tlrmvm/Tlrmvmcuda.h"
#include "benchmark/benchmark.h"

using namespace std;
using namespace tlrmat;
using namespace cudatlrmat;
using namespace benchmark;


class AstronomyFixture : public benchmark::Fixture {
public:
  void SetUp(const ::benchmark::State& state) {

  }

  void TearDown(const ::benchmark::State& state) {
      
  }
};

BENCHMARK_F(AstronomyFixture, FooTest)(benchmark::State& st) {
   for (auto _ : st) {

  }
}



void BM_TLRMVM_StreamedSGEMV(benchmark::State &state){

    // load data for astronomy
    float *DataAv, *DataAu;
    int *DataR;
    string datafolder = "/datawaha/ecrc/hongy0a/astronomy/mavis/output/";
    string acc = "0.0001";
    string id = "000";
    int nb = 128;
    int originM = 4802;
    int originN = 19078;
    int paddingM = ( originM / nb + (originM % nb != 0) ) * nb;
    int paddingN = ( originN / nb + (originN % nb != 0) ) * nb;
    int Mtglobal = paddingM / nb;
    int Ntglobal = paddingN / nb;
    ReadAstronomyBinary(datafolder, &DataR, Mtglobal * Ntglobal, acc, nb, id);
    Matrix<int> Rmat(DataR, Mtglobal, Ntglobal);
    size_t totalsum = Rmat.Sum();
    ReadAstronomyBinary(datafolder+"/V", &DataAv, totalsum * nb, acc ,nb, id);
    ReadAstronomyBinary(datafolder+"/U", &DataAu, totalsum * nb, acc ,nb, id);

    vector<int> colrsum = Rmat.ColSum();
    float *hAv, *hx, *hyv;
    float **hAvbp, **hxbp, **hyvbp;
    float val = 1.0;
    vector<size_t> AvMs,AvKs,AvNs;
    for(int i=0; i<Ntglobal; i++){
        AvMs.push_back( (size_t)colrsum[i] );
        AvKs.push_back(nb);
        AvNs.push_back(1);
    }
    GetHostMemoryBatched(&hAv, &hx, &hyv, 
    &hAvbp, &hxbp, &hyvbp, AvMs, AvKs, AvNs, val);
    size_t Avtotalelems, xtotalelems, yvtotalelems;
    CaluclateTotalElements(AvMs, AvKs, AvNs, Avtotalelems, xtotalelems, yvtotalelems);
    memcpy(hAv, DataAv, sizeof(float) * Avtotalelems);
    delete[] DataAv;

    float *Av, *x, *yv;
    float **Avbatchpointer, **xbatchpointer, **yvbatchpointer;
    GetDeviceMemoryBatched(&Av, &x, &yv, 
    &Avbatchpointer, &xbatchpointer, &yvbatchpointer, AvMs, AvKs, AvNs);
    float **Avbatchpointer_h,**xbatchpointer_h, **yvbatchpointer_h;
    Avbatchpointer_h = new float*[Ntglobal];
    xbatchpointer_h = new float*[Ntglobal];
    yvbatchpointer_h = new float*[Ntglobal];
    CUDACHECK( cudaMemcpy(Avbatchpointer_h, Avbatchpointer, 
    Ntglobal * sizeof(float*), cudaMemcpyDefault) );
    CUDACHECK( cudaMemcpy(xbatchpointer_h, xbatchpointer, 
    Ntglobal * sizeof(float*), cudaMemcpyDefault) );
    CUDACHECK( cudaMemcpy(yvbatchpointer_h, yvbatchpointer, 
    Ntglobal * sizeof(float*), cudaMemcpyDefault) );
    

    vector<int> rowrsum = Rmat.RowSum();
    float *hAu, *hyu, *hy;
    float **hAubp, **hyubp, **hybp;
    vector<size_t> AuMs,AuKs,AuNs;
    for(int i=0; i<Mtglobal; i++){
        AuMs.push_back(nb);
        AuKs.push_back( (size_t)rowrsum[i] );
        AuNs.push_back(1);
    }
    GetHostMemoryBatched(&hAu, &hyu, &hy, 
    &hAubp, &hyubp, &hybp, AuMs, AuKs, AuNs, val);
    size_t Autotalelems, yutotalelems, ytotalelems;
    CaluclateTotalElements(AuMs, AuKs, AuNs, Autotalelems, yutotalelems, ytotalelems);
    memcpy(hAu, DataAu, sizeof(float) * Autotalelems);
    delete[] DataAu;

    float *Au, *yu, *y;
    float **Aubatchpointer, **yubatchpointer, **ybatchpointer;
    GetDeviceMemoryBatched(&Au, &yu, &y, 
    &Aubatchpointer, &yubatchpointer, &ybatchpointer, AuMs, AuKs, AuNs);
    float **Aubatchpointer_h,**yubatchpointer_h, **ybatchpointer_h;
    Aubatchpointer_h = new float*[Mtglobal];
    yubatchpointer_h = new float*[Mtglobal];
    ybatchpointer_h = new float*[Mtglobal];
    CUDACHECK( cudaMemcpy(Aubatchpointer_h, Aubatchpointer, 
    Mtglobal * sizeof(float*), cudaMemcpyDefault) );
    CUDACHECK( cudaMemcpy(yubatchpointer_h, yubatchpointer, 
    Mtglobal * sizeof(float*), cudaMemcpyDefault) );
    CUDACHECK( cudaMemcpy(ybatchpointer_h, ybatchpointer, 
    Mtglobal * sizeof(float*), cudaMemcpyDefault) );
    
    CopyDataB2HD((float*)Av, (float*)x, (float*)yv, hAv, hx, hyv, AvMs, AvKs, AvNs);
    CopyDataB2HD((float*)Au, (float*)yu, (float*)y, hAu, hyu, hy, AuMs, AuKs, AuNs);

    Matrix<int> Rtransmat = Rmat.Transpose();
    vector<int> Rprefix = PrefixSum(Rmat.RawPtr(), Rmat.Row() * Rmat.Col());
    vector<int> Rtransprefix = PrefixSum(Rtransmat.RawPtr(), Rtransmat.Row() * Rtransmat.Col());
    unsigned long int* offsetinyu;
    GetDeviceMemory(&offsetinyu, (size_t)totalsum);
    unsigned long int* offsetinyu_h = new unsigned long int[totalsum];
    unsigned long int prefix = 0;
    for(int i=0; i<Ntglobal; i++){
        for(int j=0; j<Mtglobal; j++){
            for(int k=0; k<Rmat.GetElem(j,i); k++){
                offsetinyu_h[Rtransprefix[i + j * Ntglobal]+k] = prefix; 
                prefix++;
            }
        }
    }
    CUDACHECK(cudaMemcpy(offsetinyu, offsetinyu_h, sizeof(int) * totalsum, cudaMemcpyDefault));

    float alpha, beta;
    alpha = 1.0;
    beta = 0.0;
    size_t streamsize = (Ntglobal > Mtglobal ? Ntglobal : Mtglobal);


    /**************************
     * CUBLAS CALL 
     * ************************/
    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreate(&streamptr[i]);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);

    // timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    vector<double> rawtime;
    
    for(auto st: state){
        cudaEventRecord(start);

        for(int i=0; i<Ntglobal; i++){
            cublasSetStream_v2(cublashandleptr[i], streamptr[i]);
            // CUBLASCHECK(
            cublasgemv(cublashandleptr[i], CUBLAS_OP_N,
            AvMs[i], AvKs[i],
            &alpha, (const float*)Avbatchpointer_h[i], AvMs[i], 
            (const float*)xbatchpointer_h[i], 1, &beta, 
            yvbatchpointer_h[i], 1);
            // );
        }
        cudaDeviceSynchronize();
        cudatlrmvm::phase2dirver(yu, yv, totalsum, offsetinyu);
        cudaDeviceSynchronize();
        for(int i=0; i<Mtglobal; i++){
            cublasSetStream_v2(cublashandleptr[i], streamptr[i]);
            // CUBLASCHECK(
            cublasgemv(cublashandleptr[i], CUBLAS_OP_N,
            AuMs[i], AuKs[i],
            &alpha, (const float*)Aubatchpointer_h[i], AuMs[i], 
            (const float*)yubatchpointer_h[i], 1, &beta, 
            ybatchpointer_h[i], 1);
            // );
        }

        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
    }
    unsigned long int phase1 = totalsum*nb + originN + totalsum;
    unsigned long int shuffle = 2 * totalsum;
    unsigned long int phase2 = totalsum*nb + totalsum + originM;
    double bytes = sizeof(float) * (phase1 + shuffle + phase2) * (double)state.iterations();
    cout << "Total bytes is " << bytes * 1e-6 << endl;
    state.counters["BandWidth"] =
    Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
    for(int i=0; i<streamsize; i++) cudaStreamDestroy(streamptr[i]);
    for(int i=0; i<streamsize; i++) cublasDestroy_v2(cublashandleptr[i]);
    delete[] streamptr;
    delete[] cublashandleptr;

    FreeHostMemoryBatched(hAv, hx, hyv, hAvbp, hxbp, hyvbp);
    FreeHostMemoryBatched(hAu, hyu, hy, hAubp, hyubp, hybp);
    FreeDeviceMemoryBatched(Av,x,yv,Avbatchpointer, xbatchpointer,yvbatchpointer);
    FreeDeviceMemoryBatched(Au,yu,y,Aubatchpointer, yubatchpointer,ybatchpointer);
    delete[] offsetinyu_h;
    FreeDeviceMemory(offsetinyu);
    delete[] Avbatchpointer_h; delete[] xbatchpointer_h; delete[] yvbatchpointer_h;
    delete[] Aubatchpointer_h; delete[] yubatchpointer_h; delete[] ybatchpointer_h;

}

BENCHMARK(BM_TLRMVM_StreamedSGEMV)
->Unit(benchmark::kMicrosecond)
->Iterations(1000)->Repetitions(1)->UseManualTime();


BENCHMARK_MAIN();