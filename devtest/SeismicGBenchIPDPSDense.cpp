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
#include "benchmark/benchmark.h"

using namespace std;
using namespace tlrmat;
using namespace cudatlrmat;
using namespace benchmark;
using namespace cudatlrmvm;







// void BM_Dense_SGEMM(benchmark::State &state){
//     // configuration
//     size_t m = state.range(0);
//     size_t k = state.range(1);
//     size_t n = 1;
//     size_t f_sz = sizeof(float);
//     float *hA, *hx, *hy;
//     hA = new float[m * k];
//     hx = new float[k];
//     hy = new float[m];
//     float val = (float)1.0;
//     Init(hA, m * k, val);
//     Init(hx, k, val);
//     Init(hy, m, val);
//     float *A, *x, *y;
//     CUDACHECK(cudaMalloc(&A, m * k * f_sz));
//     CUDACHECK(cudaMalloc(&x, k * f_sz));
//     CUDACHECK(cudaMalloc(&y, m * f_sz));
//     CUDACHECK(cudaMemcpy(A, hA, m * k * f_sz, cudaMemcpyDefault));
//     CUDACHECK(cudaMemcpy(x, hx, k * f_sz, cudaMemcpyDefault));
//     CUDACHECK(cudaMemcpy(y, hy, m * f_sz, cudaMemcpyDefault));
//     // timer
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     // create cublas handle
//     cublasHandle_t handle;
//     cublasCreate(&handle);
//     cudaStream_t stream;
//     CUDACHECK(cudaStreamCreate(&stream));
//     float alpha, beta;
    
//     vector<double> rawtime;
//     for(auto st : state){
//         cudaEventRecord(start);
//         cudaDeviceSynchronize();
//         CUBLASCHECK(cublasgemv(handle, CUBLAS_OP_N, m, k, &alpha, A, m, x, 1, &beta, y, 1));
//         cudaDeviceSynchronize();
//         cudaEventRecord(stop);
//         float milliseconds = 0;
//         cudaEventSynchronize(stop);
//         cudaEventElapsedTime(&milliseconds, start, stop);
//         state.SetIterationTime(milliseconds*1e-3);
//         rawtime.push_back(milliseconds*1e-3);
//     }
//     double bytes = sizeof(float) * (m * k + m + k) * (double)state.iterations();
//     state.counters["BandWidth"] =
//     Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
//     cublasDestroy(handle);
//     CUDACHECK(cudaFree(A));
//     CUDACHECK(cudaFree(x));
//     CUDACHECK(cudaFree(y));
//     delete[] hA;
//     delete[] hx;
//     delete[] hy;
// }

// BENCHMARK(BM_Dense_SGEMM)->Args({9801, 9801})
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(5000)->Repetitions(1)->UseManualTime();



// void BM_Dense_SGEMV(benchmark::State &state){
//     // configuration
//     size_t m = state.range(0);
//     size_t k = state.range(1);
//     size_t n = 1;
//     size_t f_sz = sizeof(float);
//     float *hA, *hx, *hy;
//     hA = new float[m * k];
//     hx = new float[k];
//     hy = new float[m];
//     float val = (float)1.0;
//     Init(hA, m * k, val);
//     Init(hx, k, val);
//     Init(hy, m, val);
//     float *A, *x, *y;
//     CUDACHECK(cudaMalloc(&A, m * k * f_sz));
//     CUDACHECK(cudaMalloc(&x, k * f_sz));
//     CUDACHECK(cudaMalloc(&y, m * f_sz));
//     CUDACHECK(cudaMemcpy(A, hA, m * k * f_sz, cudaMemcpyDefault));
//     CUDACHECK(cudaMemcpy(x, hx, k * f_sz, cudaMemcpyDefault));
//     CUDACHECK(cudaMemcpy(y, hy, m * f_sz, cudaMemcpyDefault));
//     // timer
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     // create cublas handle
//     cublasHandle_t handle;
//     cublasCreate(&handle);
//     cudaStream_t stream;
//     CUDACHECK(cudaStreamCreate(&stream));
//     float alpha, beta;
    
//     vector<double> rawtime;
//     for(auto st : state){
//         cudaEventRecord(start);
//         cudaDeviceSynchronize();
//         CUBLASCHECK(cublasgemv(handle, CUBLAS_OP_N, m, k, &alpha, A, m, x, 1, &beta, y, 1));
//         cudaDeviceSynchronize();
//         cudaEventRecord(stop);
//         float milliseconds = 0;
//         cudaEventSynchronize(stop);
//         cudaEventElapsedTime(&milliseconds, start, stop);
//         state.SetIterationTime(milliseconds*1e-3);
//         rawtime.push_back(milliseconds*1e-3);
//     }
//     double bytes = sizeof(float) * (m * k + m + k) * (double)state.iterations();
//     state.counters["BandWidth"] =
//     Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
//     cublasDestroy(handle);
//     CUDACHECK(cudaFree(A));
//     CUDACHECK(cudaFree(x));
//     CUDACHECK(cudaFree(y));
//     delete[] hA;
//     delete[] hx;
//     delete[] hy;
// }

// BENCHMARK(BM_Dense_SGEMV)->Args({9801, 9801})
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(5000)->Repetitions(1)->UseManualTime();


void BM_Dense_CGEMV(benchmark::State &state){
    // configuration
    size_t m = state.range(0);
    size_t k = state.range(1);
    size_t n = 1;
    size_t f_sz = sizeof(float);
    size_t c_sz = sizeof(cuComplex);
    complex<float> *hA, *hx, *hy;
    hA = new complex<float>[m * k];
    hx = new complex<float>[k];
    hy = new complex<float>[m];
    complex<float> val = complex<float>(1.0, 0.0);
    Init(hA, m * k, val);
    Init(hx, k, val);
    Init(hy, m, val);
    cuComplex *A, *x, *y;
    CUDACHECK(cudaMalloc(&A, m * k * c_sz));
    CUDACHECK(cudaMalloc(&x, k * c_sz));
    CUDACHECK(cudaMalloc(&y, m * c_sz));
    CUDACHECK(cudaMemcpy(A, hA, m * k * c_sz, cudaMemcpyDefault));
    CUDACHECK(cudaMemcpy(x, hx, k * c_sz, cudaMemcpyDefault));
    CUDACHECK(cudaMemcpy(y, hy, m * c_sz, cudaMemcpyDefault));
    // timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // create cublas handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));
    cuComplex alpha, beta;
    alpha.x = 1.0;
    alpha.y = 0.0;
    beta.x = 0.0;
    beta.y = 0.0;
    vector<double> rawtime;
    for(auto st : state){
        cudaEventRecord(start);
        cudaDeviceSynchronize();
        CUBLASCHECK(cublasgemv(handle, CUBLAS_OP_N, m, k, &alpha, A, m, x, 1, &beta, y, 1));
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
    }
    double bytes = sizeof(complex<float>) * (m * k + m + k) * (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
    cublasDestroy(handle);
    CUDACHECK(cudaFree(A));
    CUDACHECK(cudaFree(x));
    CUDACHECK(cudaFree(y));
    delete[] hA;
    delete[] hx;
    delete[] hy;
}

// void Dense_CGEMV_CustomArguments(benchmark::internal::Benchmark* b) {
//     b->Args({9801,9801});
// }
// BENCHMARK(BM_Dense_CGEMV)->Apply(Dense_CGEMV_CustomArguments)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(5000)->Repetitions(1)->UseManualTime();

// BENCHMARK(BM_Dense_CGEMV)->Args({9801, 9801})
// ->Unit(benchmark::kMicrosecond)
// ->Repetitions(1)->UseManualTime();


#define dtype complex<float>

void BM_Dense_CGEMVTLRMVM(benchmark::State &state){
    // configuration
    size_t nb = state.range(0);
    dtype *hAv, *hx, *hyv;
    dtype **hAvbp, **hxbp, **hyvbp;
    dtype val = dtype(0.0,0.0);
    cuComplex alpha;
    cuComplex beta;
    alpha.x = 1.0;
    alpha.y = 0.0;
    beta.x = 0.0;
    beta.y = 0.0;
    vector<size_t> AvMs,AvKs,AvNs;
    size_t Ntglobal = 1024*10 / nb;
    size_t maxrowsize = state.range(1);
    size_t M = maxrowsize;
    size_t N = 1;
    size_t K = nb;
    for(int i=0; i<Ntglobal; i++){
        AvMs.push_back( M );
        AvKs.push_back( K );
        AvNs.push_back( N );
    }
    GetHostMemoryBatched(&hAv, &hx, &hyv, 
    &hAvbp, &hxbp, &hyvbp, AvMs, AvKs, AvNs, val);
    cuComplex *Av, *x, *yv;
    cuComplex **Avbatchpointer, **xbatchpointer, **yvbatchpointer;
    size_t Avtotalelems, xtotalelems, yvtotalelems;
    CaluclateTotalElements(AvMs, AvKs, AvNs, Avtotalelems, xtotalelems, yvtotalelems);
    // memcpy(hAv, DataAv, sizeof(dtype) * Avtotalelems);
    for(size_t i=0; i<Avtotalelems; i++){
        hAv[i] = 1.0;
    }
    for(size_t i=0; i<xtotalelems; i++){
        hx[i] = 1.0;
    }
    GetDeviceMemoryBatched(&Av, &x, &yv, 
    &Avbatchpointer, &xbatchpointer, &yvbatchpointer, AvMs, AvKs, AvNs);
    CopyDataB2HD((dtype*)Av, (dtype*)x, (dtype*)yv, hAv, hx, hyv, AvMs, AvKs, AvNs);
    dtype yvback;
    CopyDataB2HD(&yvback, (dtype*)yv, 1);
    cublasHandle_t cublashandleptr;
    cublasCreate(&cublashandleptr);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    vector<double> rawtime;
    for(auto st : state){
        cudaEventRecord(start);
        cudaDeviceSynchronize();

        CUBLASCHECK(cublasgemmbatched( cublashandleptr, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        M, N, K, &alpha, (const cuComplex**)Avbatchpointer, maxrowsize, 
        (const cuComplex**)xbatchpointer, nb, &beta, 
        yvbatchpointer, maxrowsize, Ntglobal));

        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
    }
    complex<float> hhyv;
    CopyDataB2HD(&hhyv, (complex<float>*)yv, 1);
    double bytes = sizeof(complex<float>) * ( Ntglobal * (M * K + K * 1 + M * 1) ) * (double)state.iterations();
    // double flops = Ntglobal * ( M * K * N ) * (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
    // state.counters["Flops/s"] =
    // Counter(static_cast<double>(flops), Counter::kIsRate, Counter::kIs1000);
    cublasDestroy(cublashandleptr);
    cudaFree(Avbatchpointer);
    cudaFree(xbatchpointer);
    cudaFree(yvbatchpointer);
    cudaFree(Av);
    cudaFree(x);
    cudaFree(yv);
    delete[] hAv;
    delete[] hx;
    delete[] hyv;
    delete[] hAvbp;
    delete[] hxbp;
    delete[] hyvbp;
}



void BM_Dense_CGEMVEXTLRMVM(benchmark::State &state){
    // configuration
    size_t nb = state.range(0);
    dtype *hAv, *hx, *hyv;
    dtype **hAvbp, **hxbp, **hyvbp;
    dtype val = dtype(0.0,0.0);
    cuComplex alpha;
    cuComplex beta;
    alpha.x = 1.0;
    alpha.y = 0.0;
    beta.x = 0.0;
    beta.y = 0.0;
    vector<size_t> AvMs,AvKs,AvNs;
    size_t Ntglobal = 10240 / nb;
    size_t maxrowsize = state.range(1);
    size_t M = maxrowsize;
    size_t N = 1;
    size_t K = nb;
    for(int i=0; i<Ntglobal; i++){
        AvMs.push_back( M );
        AvKs.push_back( K );
        AvNs.push_back( N );
    }
    GetHostMemoryBatched(&hAv, &hx, &hyv, 
    &hAvbp, &hxbp, &hyvbp, AvMs, AvKs, AvNs, val);
    cuComplex *Av, *x, *yv;
    cuComplex **Avbatchpointer, **xbatchpointer, **yvbatchpointer;
    size_t Avtotalelems, xtotalelems, yvtotalelems;
    CaluclateTotalElements(AvMs, AvKs, AvNs, Avtotalelems, xtotalelems, yvtotalelems);
    // memcpy(hAv, DataAv, sizeof(dtype) * Avtotalelems);
    for(size_t i=0; i<Avtotalelems; i++){
        hAv[i] = 1.0;
    }
    for(size_t i=0; i<xtotalelems; i++){
        hx[i] = 1.0;
    }
    GetDeviceMemoryBatched(&Av, &x, &yv, 
    &Avbatchpointer, &xbatchpointer, &yvbatchpointer, AvMs, AvKs, AvNs);
    CopyDataB2HD((dtype*)Av, (dtype*)x, (dtype*)yv, hAv, hx, hyv, AvMs, AvKs, AvNs);
    dtype yvback;
    CopyDataB2HD(&yvback, (dtype*)yv, 1);
    cublasHandle_t cublashandleptr;
    cublasCreate(&cublashandleptr);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    vector<double> rawtime;

    for(auto st : state){
        cudaEventRecord(start);
        cudaDeviceSynchronize();

        CUBLASCHECK(cublasGemmBatchedEx( cublashandleptr, 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        (int)M, (int)N, (int)K, 
        (const void*)(&alpha), 
        (const void**)Avbatchpointer, CUDA_C_32F, (int)maxrowsize, 
        (const void**)xbatchpointer, CUDA_C_32F, (int)nb, 
        (const void*)(&beta), 
        (void **)(yvbatchpointer), CUDA_C_32F, 
        (int)maxrowsize, (int)Ntglobal, 
        CUBLAS_COMPUTE_32F_FAST_16BF, 
        CUBLAS_GEMM_DEFAULT));
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
    }
    complex<float> hhyv;
    CopyDataB2HD(&hhyv, (complex<float>*)yv, 1);
    double bytes = sizeof(complex<float>) * ( Ntglobal * (M * K + K * N + M * N) ) * (double)state.iterations();
    // double flops = Ntglobal * ( M * K * N ) * (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
    // state.counters["Flops/s"] =
    // Counter(static_cast<double>(flops), Counter::kIsRate, Counter::kIs1000);
    cublasDestroy(cublashandleptr);
    cudaFree(Avbatchpointer);
    cudaFree(xbatchpointer);
    cudaFree(yvbatchpointer);
    cudaFree(Av);
    cudaFree(x);
    cudaFree(yv);
    delete[] hAv;
    delete[] hx;
    delete[] hyv;
    delete[] hAvbp;
    delete[] hxbp;
    delete[] hyvbp;
}



// BENCHMARK(BM_Dense_CGEMVTLRMVM)
// ->Unit(benchmark::kMicrosecond)
// ->ArgsProduct({{256},
// benchmark::CreateRange(512, 4096, 2),
// {1,2,32,64,128,256}
// })
// ->Repetitions(1)->UseManualTime();

// BENCHMARK(BM_Dense_CGEMVEXTLRMVM)
// ->Unit(benchmark::kMicrosecond)
// ->ArgsProduct({{256},
// benchmark::CreateRange(512, 4096, 2),
// {1,2,32,64,128,256}
// })
// ->Repetitions(1)->UseManualTime();




// BENCHMARK(BM_Dense_CGEMVTLRMVM)
// ->Unit(benchmark::kMicrosecond)
// ->ArgsProduct({{256,512},
// benchmark::CreateRange(512, 4096, /*multi=*/2),
// // benchmark::CreateDenseRange(512, 4096, 128),
// })
// ->Repetitions(1)->UseManualTime();

// BENCHMARK(BM_Dense_CGEMVEXTLRMVM)
// ->Unit(benchmark::kMicrosecond)
// ->ArgsProduct({{256,512},
// benchmark::CreateDenseRange(512, 4096, 128),
// {1,2,4}})
// ->Repetitions(1)->UseManualTime();


void getYvbatchedpointer(half ** tmpyv, half *** yvpointer, vector<size_t> AvMs, vector<size_t> AvNs){
    half * yv;
    size_t ttval = 0;
    for(int i=0; i<AvMs.size(); i++){
        ttval += AvMs[i] * AvNs[i];
    }
    CUDACHECK(cudaMalloc(&yv, sizeof(half) * ttval));
    half ** yvptr;
    CUDACHECK(cudaMalloc(&yvptr, sizeof(half*) * AvMs.size()));
    half **yvhost;
    CUDACHECK(cudaMallocHost(&yvhost, sizeof(half*) * AvMs.size()));
    yvhost[0] = yv;
    for(int i=1; i<AvMs.size(); i++){
        yvhost[i] = yvhost[i-1] + AvMs[i-1] * AvNs[i-1];
    }
    CUDACHECK(cudaMemcpy(yvptr, yvhost, sizeof(half*) * AvMs.size(),cudaMemcpyDefault));
    tmpyv[0] = yv;
    yvpointer[0] = yvptr;
}


// void BM_Dense_CGEMVTLRMVMHALF(benchmark::State &state){
//     // configuration
//     size_t nb = state.range(0);

//     // get host pointer 
//     half *hAv_real, *hx_real, *hyv_real;
//     half *hAv_imag, *hx_imag, *hyv_imag;
//     half val = 1.0;
//     half alpha = 1.0;
//     half beta = 0.0;
//     vector<size_t> AvMs,AvKs,AvNs;
//     size_t Ntglobal = 10240 / nb;
//     size_t maxrowsize = state.range(1);
//     size_t Avtotalelems(0), xtotalelems(0), yvtotalelems(0);
//     size_t M = maxrowsize;
//     size_t N = 1;
//     size_t K = nb;
//     for(int i=0; i<Ntglobal; i++){
//         AvMs.push_back( M );
//         AvKs.push_back( K );
//         AvNs.push_back( N );
//         Avtotalelems += M * K;
//         xtotalelems += K * N;
//         yvtotalelems += M * N;
//     }
//     size_t hfsize = sizeof(half);
//     CUDACHECK(cudaMallocHost(&hAv_real, hfsize * Avtotalelems));
//     CUDACHECK(cudaMallocHost(&hx_real, hfsize * xtotalelems));
//     CUDACHECK(cudaMallocHost(&hyv_real, hfsize * yvtotalelems));
//     CUDACHECK(cudaMallocHost(&hAv_imag, hfsize * Avtotalelems));
//     CUDACHECK(cudaMallocHost(&hx_imag, hfsize * xtotalelems));
//     CUDACHECK(cudaMallocHost(&hyv_imag, hfsize * yvtotalelems));
//     // init data
//     for(size_t i=0; i<Avtotalelems; i++){
//         hAv_real[i] = 1.0;
//         hAv_imag[i] = 1.0;
//     }
//     for(size_t i=0; i<xtotalelems; i++){
//         hx_real[i] = 1.0;
//         hx_imag[i] = 1.0;
//     }
//     // get device pointer 
//     half *Av_real, *x_real, *yv_real;
//     half *Av_imag, *x_imag, *yv_imag;
//     half **Avbatchpointer_real, **xbatchpointer_real, **yvbatchpointer_real;
//     half **Avbatchpointer_imag, **xbatchpointer_imag, **yvbatchpointer_imag;
//     GetDeviceMemoryBatched(&Av_real, &x_real, &yv_real, &Avbatchpointer_real,
//     &xbatchpointer_real, &yvbatchpointer_real, AvMs, AvKs, AvNs);
//     GetDeviceMemoryBatched(&Av_imag, &x_imag, &yv_imag, &Avbatchpointer_imag,
//     &xbatchpointer_imag, &yvbatchpointer_imag, AvMs, AvKs, AvNs);
//     CopyDataB2HD(Av_real, hAv_real, Avtotalelems);
//     CopyDataB2HD(Av_imag, hAv_imag, Avtotalelems);
//     CopyDataB2HD(x_real, hx_real, xtotalelems);
//     CopyDataB2HD(x_imag, hx_imag, xtotalelems);

//     size_t STREAMSIZE = 4;
//     cudaStream_t * streamptr = new cudaStream_t[STREAMSIZE];
//     cublasHandle_t * cublashandleptr = new cublasHandle_t[STREAMSIZE];
//     for(int i=0; i<STREAMSIZE; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
//     for(int i=0; i<STREAMSIZE; i++) cublasCreate_v2(&cublashandleptr[i]);
//     for(int i=0; i<STREAMSIZE; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     cudaEvent_t *events;
//     cudaEvent_t event_start;
//     cudaEvent_t event_phase2finish;
//     CUDACHECK(cudaEventCreate(&event_start));
//     CUDACHECK(cudaEventCreate(&event_phase2finish));
//     events = new cudaEvent_t[STREAMSIZE];
//     for(int i=0; i<STREAMSIZE; i++) CUDACHECK(cudaEventCreate(&events[i]));


//     vector<double> rawtime;
//     // cout << maxrowsize << endl;
//     half *tmpyv_Areal_Bimag, **tmpyv_Areal_Bimag_batched;
//     half *tmpyv_Aimag_Breal, **tmpyv_Aimag_Breal_batched;
//     getYvbatchedpointer(&tmpyv_Areal_Bimag, &tmpyv_Areal_Bimag_batched, AvMs, AvNs);
//     getYvbatchedpointer(&tmpyv_Aimag_Breal, &tmpyv_Aimag_Breal_batched, AvMs, AvNs);
    
//     for(auto st : state){
//         cudaEventRecord(start);
//         cudaDeviceSynchronize();
//         CUBLASCHECK(cublasGemmBatchedEx( cublashandleptr[0], 
//         CUBLAS_OP_N, CUBLAS_OP_N, 
//         M, N, K, &alpha, (const void**)Avbatchpointer_real, CUDA_R_16F,
//         maxrowsize, 
//         (const void**)xbatchpointer_real, CUDA_R_16F, nb, &beta, 
//         (void **)yvbatchpointer_real, CUDA_R_16F, maxrowsize, Ntglobal,
//         CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
        
        

//         CUBLASCHECK(cublasGemmBatchedEx( cublashandleptr[1], 
//         CUBLAS_OP_N, CUBLAS_OP_N, 
//         M, N, K, &alpha, (const void**)Avbatchpointer_imag, CUDA_R_16F,
//         maxrowsize, 
//         (const void**)xbatchpointer_imag, CUDA_R_16F, nb, &beta, 
//         (void **)yvbatchpointer_imag, CUDA_R_16F, maxrowsize, Ntglobal,
//         CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));

//         cudaEventRecord(events[1], streamptr[1]);

//         CUBLASCHECK(cublasGemmBatchedEx( cublashandleptr[2], 
//         CUBLAS_OP_N, CUBLAS_OP_N, 
//         M, N, K, &alpha, (const void**)Avbatchpointer_real, CUDA_R_16F,
//         maxrowsize, 
//         (const void**)xbatchpointer_imag, CUDA_R_16F, nb, &beta, 
//         (void **)tmpyv_Areal_Bimag_batched, CUDA_R_16F, maxrowsize, Ntglobal,
//         CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));

        
//         CUBLASCHECK(cublasGemmBatchedEx( cublashandleptr[3], 
//         CUBLAS_OP_N, CUBLAS_OP_N, 
//         M, N, K, &alpha, (const void**)Avbatchpointer_imag, CUDA_R_16F,
//         maxrowsize, 
//         (const void**)xbatchpointer_real, CUDA_R_16F, nb, &beta, 
//         (void **)tmpyv_Aimag_Breal_batched, CUDA_R_16F, maxrowsize, Ntglobal,
//         CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));

//         cudaEventRecord(events[3], streamptr[3]);

//         // cudaStreamSynchronize(streamptr[0]);
//         // cudaStreamSynchronize(streamptr[1]);
//         cudaStreamWaitEvent(streamptr[0], events[1]);
//         phase1fuse4ops_real(yv_real, yv_imag, maxrowsize * Ntglobal, streamptr[0]);
//         cudaStreamWaitEvent(streamptr[2], events[3]);
//         phase1fuse4ops_imag(tmpyv_Areal_Bimag, tmpyv_Aimag_Breal, maxrowsize*Ntglobal, streamptr[2]);
//         cudaStreamSynchronize(streamptr[0]);
//         cudaStreamSynchronize(streamptr[2]);
        
//         cudaEventRecord(stop);
//         float milliseconds = 0;
//         cudaEventSynchronize(stop);
//         cudaEventElapsedTime(&milliseconds, start, stop);
//         state.SetIterationTime(milliseconds*1e-3);
//         rawtime.push_back(milliseconds*1e-3);
//     }
//     half hhyv_real;
//     half hhyv_imag;
//     CopyDataB2HD(&hhyv_real, &yv_real[1], 1);
//     float yvfreal = hhyv_real;
//     // cout << yvfreal << endl;
//     CopyDataB2HD(&hhyv_imag, &tmpyv_Areal_Bimag[1], 1);
//     float yvimag = hhyv_imag;
//     // cout << yvimag << endl;
//     for(int i=0; i<STREAMSIZE; i++) cudaStreamDestroy(streamptr[i]);
//     for(int i=0; i<STREAMSIZE; i++) cublasDestroy(cublashandleptr[i]);
//     double bytes = sizeof(half) * ( Ntglobal * (M * K + K*N + M*K) ) * (double)state.iterations();
//     state.counters["BandWidth"] =
//     Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);

// }


// BENCHMARK(BM_Dense_CGEMVTLRMVM)
// ->Unit(benchmark::kMicrosecond)
// ->ArgsProduct({{512},
// benchmark::CreateRange(512, 4096, /*multi=*/2),
// // benchmark::CreateDenseRange(512, 4096, 128),
// })
// ->Repetitions(1)->UseManualTime();


// BENCHMARK(BM_Dense_CGEMVTLRMVMHALF)
// ->Unit(benchmark::kMicrosecond)
// // ->Iterations(50)
// ->Args({512,4096})
// // ->ArgsProduct({{256,512},
// // benchmark::CreateRange(512, 4096, /*multi=*/2),
// // })
// ->Iterations(50)
// ->Repetitions(1)->UseManualTime();







void BM_Dense_CGEMVTLRMVMHALFTWOGEMV(benchmark::State &state){
    // configuration
    size_t nb = state.range(0);

    // get host pointer 
    half *hAv_real, *hx_copy1, *hmiddle1;
    half *hAv_imag, *hx_copy2, *hmiddle2;
    half val = 1.0;
    half alpha = 1.0;
    half beta = 0.0;
    vector<size_t> AvMs,AvKs,AvNs;
    size_t Ntglobal = 10240 / nb;
    size_t maxrowsize = state.range(1);
    size_t Avtotalelems(0), xtotalelems(0), yvtotalelems(0);
    size_t M = maxrowsize;
    size_t N = 8;
    size_t K = nb;
    for(int i=0; i<Ntglobal; i++){
        AvMs.push_back( M );
        AvKs.push_back( K );
        AvNs.push_back( N );
        Avtotalelems += M * K;
        xtotalelems += K * N;
        yvtotalelems += M * N;
    }
    size_t hfsize = sizeof(half);
    CUDACHECK(cudaMallocHost(&hAv_real, hfsize * Avtotalelems));
    CUDACHECK(cudaMallocHost(&hx_copy1, hfsize * xtotalelems));
    CUDACHECK(cudaMallocHost(&hmiddle1, hfsize * yvtotalelems));
    CUDACHECK(cudaMallocHost(&hAv_imag, hfsize * Avtotalelems));
    CUDACHECK(cudaMallocHost(&hx_copy2, hfsize * xtotalelems));
    CUDACHECK(cudaMallocHost(&hmiddle2, hfsize * yvtotalelems));
    for(size_t i=0; i<Avtotalelems; i++){
        hAv_real[i] = (half)1.0;
        hAv_imag[i] = (half)1.0;
    }
    // cout << hAv_real[0] << endl;
    for(size_t i=0; i<xtotalelems; i++){
        hx_copy1[i] = (half)1.0;
        hx_copy2[i] = (half)1.0;
    }
    
    // cout << " xtotal " << xtotalelems << endl;
    // get device pointer 
    half *Av_real, *x_copy1, *middle1;
    half *Av_imag, *x_copy2, *middle2;
    half **Avbatchpointer_real, **xbatchpointer_real, **middle1batched;
    half **Avbatchpointer_imag, **xbatchpointer_imag, **middle2batched;
    GetDeviceMemoryBatched(&Av_real, &x_copy1, &middle1, &Avbatchpointer_real,
    &xbatchpointer_real, &middle1batched, AvMs, AvKs, AvNs);
    GetDeviceMemoryBatched(&Av_imag, &x_copy2, &middle2, &Avbatchpointer_imag,
    &xbatchpointer_imag, &middle2batched, AvMs, AvKs, AvNs);
    CopyDataB2HD(Av_real, hAv_real, Avtotalelems);
    CopyDataB2HD(Av_imag, hAv_imag, Avtotalelems);
    CopyDataB2HD(x_copy1, hx_copy1, xtotalelems);
    CopyDataB2HD(x_copy2, hx_copy2, xtotalelems);
    half databack;
    CopyDataB2HD(&databack, Av_real, 1);
    size_t STREAMSIZE = 2;
    cudaStream_t * streamptr = new cudaStream_t[STREAMSIZE];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[STREAMSIZE];
    for(int i=0; i<STREAMSIZE; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<STREAMSIZE; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<STREAMSIZE; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEvent_t *events;
    cudaEvent_t event_start;
    cudaEvent_t event_phase2finish;
    CUDACHECK(cudaEventCreate(&event_start));
    CUDACHECK(cudaEventCreate(&event_phase2finish));
    events = new cudaEvent_t[STREAMSIZE];
    for(int i=0; i<STREAMSIZE; i++) CUDACHECK(cudaEventCreate(&events[i]));


    vector<double> rawtime;
    // half *tmpyv_Areal_Bimag, **tmpyv_Areal_Bimag_batched;
    // half *tmpyv_Aimag_Breal, **tmpyv_Aimag_Breal_batched;
    // getYvbatchedpointer(&tmpyv_Areal_Bimag, &tmpyv_Areal_Bimag_batched, AvMs, AvNs);
    // getYvbatchedpointer(&tmpyv_Aimag_Breal, &tmpyv_Aimag_Breal_batched, AvMs, AvNs);
 
    for(auto st : state){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        CUBLASCHECK(cublasGemmBatchedEx( cublashandleptr[0], 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        M, N, K, &alpha, (const void**)Avbatchpointer_real, CUDA_R_16F,
        M, (const void**)xbatchpointer_real, CUDA_R_16F, K, &beta, 
        (void **)middle1batched, CUDA_R_16F, M, Ntglobal,
        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));        
        CUBLASCHECK(cublasGemmBatchedEx( cublashandleptr[1], 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        M, N, K, &alpha, (const void**)Avbatchpointer_imag, CUDA_R_16F,
        M, (const void**)xbatchpointer_imag, CUDA_R_16F, K, &beta, 
        (void **)middle2batched, CUDA_R_16F, M, Ntglobal,
        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
        // cudaDeviceSynchronize();
        cudaEventRecord(events[1], streamptr[1]);
        cudaStreamWaitEvent(streamptr[0], events[1]);
        phase1twogemv(middle1, middle2, maxrowsize  * Ntglobal, streamptr[0]);
        cudaStreamSynchronize(streamptr[0]);
        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
    }
    float fval;
    half *tmp_real;
    half *tmp_imag;
    GetHostHalfMemory(&tmp_real, Ntglobal * maxrowsize);
    GetHostHalfMemory(&tmp_imag, Ntglobal * maxrowsize);
    CopyDataB2HD(tmp_real, middle1, Ntglobal * maxrowsize);
    CopyDataB2HD(tmp_imag, middle2, Ntglobal * maxrowsize);
    // CopyDataB2HD(&fval, tmpyv_Areal_Bimag, 1);
    // float yvfreal = hhyv_real[0];
    // cout << fval << endl;
    // CopyDataB2HD(hhyv_imag, middle2, Ntglobal * maxrowsize);
    for(int i=0; i<Ntglobal * maxrowsize; i++){
        float val1 = tmp_real[i];
        float val2 = tmp_imag[i];
        if(val1 != 0.0) printf("%f ",val1);
        if(val2 != nb*2.0) printf("%f ", val2);
        break;
    }
    // float yvimag = hhyv_imag;
    // cout << yvimag << endl;
    for(int i=0; i<STREAMSIZE; i++) cudaStreamDestroy(streamptr[i]);
    for(int i=0; i<STREAMSIZE; i++) cublasDestroy(cublashandleptr[i]);
    // double bytes = sizeof(half) * ( Ntglobal * (M * K + K*N + M*K) ) * (double)state.iterations();
    // state.counters["BandWidth"] =
    // Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);

}

// BENCHMARK(BM_Dense_CGEMVTLRMVMHALFTWOGEMV)
// ->Unit(benchmark::kMicrosecond)
// // ->Iterations(50)
// // ->Args({256,4096})
// ->ArgsProduct({{256,512},
// // benchmark::CreateRange(512, 4096, /*multi=*/2),
// benchmark::CreateDenseRange(512,3000,256)
// })
// // ->Iterations(20)
// ->Repetitions(1)->UseManualTime();





void BM_Dense_CGEMVTLRMVMHALFTWOGEMVINT8(benchmark::State &state){
    // configuration
    size_t nb = state.range(0);

    // get host pointer 
    half *hAv_real, *hx_copy1, *hmiddle1;
    half *hAv_imag, *hx_copy2, *hmiddle2;
    half val = 1.0;
    half alpha = 1.0;
    half beta = 0.0;
    vector<size_t> AvMs,AvKs,AvNs;
    size_t Ntglobal = 10240 / nb;
    size_t maxrowsize = state.range(1);
    size_t Avtotalelems(0), xtotalelems(0), yvtotalelems(0);
    size_t M = maxrowsize;
    size_t N = 2;
    size_t K = nb;
    for(int i=0; i<Ntglobal; i++){
        AvMs.push_back( M );
        AvKs.push_back( K );
        AvNs.push_back( N );
        Avtotalelems += M * K;
        xtotalelems += K * N;
        yvtotalelems += M * N;
    }
    size_t hfsize = sizeof(half);
    CUDACHECK(cudaMallocHost(&hAv_real, hfsize * Avtotalelems));
    CUDACHECK(cudaMallocHost(&hx_copy1, hfsize * xtotalelems));
    CUDACHECK(cudaMallocHost(&hmiddle1, hfsize * yvtotalelems));
    CUDACHECK(cudaMallocHost(&hAv_imag, hfsize * Avtotalelems));
    CUDACHECK(cudaMallocHost(&hx_copy2, hfsize * xtotalelems));
    CUDACHECK(cudaMallocHost(&hmiddle2, hfsize * yvtotalelems));
    for(size_t i=0; i<Avtotalelems; i++){
        hAv_real[i] = 1.0;
        hAv_imag[i] = 1.0;
    }
    for(size_t i=0; i<2*xtotalelems; i++){
        hx_copy1[i] = 1.0;
        hx_copy2[i] = 1.0;
    }
    // get device pointer 
    half *Av_real, *x_copy1, *middle1;
    half *Av_imag, *x_copy2, *middle2;
    half **Avbatchpointer_real, **xbatchpointer_real, **yvbatchpointer_real;
    half **Avbatchpointer_imag, **xbatchpointer_imag, **yvbatchpointer_imag;
    GetDeviceMemoryBatched(&Av_real, &x_copy1, &middle1, &Avbatchpointer_real,
    &xbatchpointer_real, &yvbatchpointer_real, AvMs, AvKs, AvNs);
    GetDeviceMemoryBatched(&Av_imag, &x_copy2, &middle2, &Avbatchpointer_imag,
    &xbatchpointer_imag, &yvbatchpointer_imag, AvMs, AvKs, AvNs);
    CopyDataB2HD(Av_real, hAv_real, Avtotalelems);
    CopyDataB2HD(Av_imag, hAv_imag, Avtotalelems);
    CopyDataB2HD(x_copy1, middle1, xtotalelems);
    CopyDataB2HD(x_copy2, middle2, xtotalelems);

    size_t STREAMSIZE = 2;
    cudaStream_t * streamptr = new cudaStream_t[STREAMSIZE];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[STREAMSIZE];
    for(int i=0; i<STREAMSIZE; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<STREAMSIZE; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<STREAMSIZE; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEvent_t *events;
    cudaEvent_t event_start;
    cudaEvent_t event_phase2finish;
    CUDACHECK(cudaEventCreate(&event_start));
    CUDACHECK(cudaEventCreate(&event_phase2finish));
    events = new cudaEvent_t[STREAMSIZE];
    for(int i=0; i<STREAMSIZE; i++) CUDACHECK(cudaEventCreate(&events[i]));


    vector<double> rawtime;
    // cout << maxrowsize << endl;
    half *tmpyv_Areal_Bimag, **tmpyv_Areal_Bimag_batched;
    half *tmpyv_Aimag_Breal, **tmpyv_Aimag_Breal_batched;
    getYvbatchedpointer(&tmpyv_Areal_Bimag, &tmpyv_Areal_Bimag_batched, AvMs, AvNs);
    getYvbatchedpointer(&tmpyv_Aimag_Breal, &tmpyv_Aimag_Breal_batched, AvMs, AvNs);
    
    for(auto st : state){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        CUBLASCHECK(cublasGemmBatchedEx( cublashandleptr[0], 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        M, N, K, &alpha, (const void**)Avbatchpointer_real, CUDA_R_16F,
        maxrowsize, 
        (const void**)xbatchpointer_real, CUDA_R_16F, nb, &beta, 
        (void **)yvbatchpointer_real, CUDA_R_16F, maxrowsize, Ntglobal,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));        
        CUBLASCHECK(cublasGemmBatchedEx( cublashandleptr[1], 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        M, N, K, &alpha, (const void**)Avbatchpointer_imag, CUDA_R_16F,
        maxrowsize, 
        (const void**)xbatchpointer_imag, CUDA_R_16F, nb, &beta, 
        (void **)yvbatchpointer_imag, CUDA_R_16F, maxrowsize, Ntglobal,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        cudaEventRecord(events[1], streamptr[1]);
        cudaStreamWaitEvent(streamptr[0], events[1]);
        phase1twogemv(middle1, middle2, maxrowsize * Ntglobal, streamptr[0]);
        cudaStreamSynchronize(streamptr[0]);
        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
    }
    half hhyv_real;
    half hhyv_imag;
    CopyDataB2HD(&hhyv_real, &middle1[1], 1);
    float yvfreal = hhyv_real;
    // cout << yvfreal << endl;
    CopyDataB2HD(&hhyv_imag, &middle2[1], 1);
    float yvimag = hhyv_imag;
    // cout << yvimag << endl;
    for(int i=0; i<STREAMSIZE; i++) cudaStreamDestroy(streamptr[i]);
    for(int i=0; i<STREAMSIZE; i++) cublasDestroy(cublashandleptr[i]);
    // double bytes = sizeof(half) * ( Ntglobal * (M * K + K*N + M*K) ) * (double)state.iterations();
    // state.counters["BandWidth"] =
    // Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);

}

// BENCHMARK(BM_Dense_CGEMVTLRMVMHALFTWOGEMVINT8)
// ->Unit(benchmark::kMicrosecond)
// // ->Iterations(50)
// ->Args({256,4096})
// // ->ArgsProduct({{256,512},
// // benchmark::CreateRange(512, 4096, /*multi=*/2),
// // })
// // ->Iterations(5)
// ->Repetitions(1)->UseManualTime();


void BM_Dense_INT8GEMM(benchmark::State &state){
    // configuration
    size_t nb = state.range(0);
    // get host pointer 
    int8_t *hAv_real, *hx_copy1, *hmiddle1;
    int8_t *hAv_imag, *hx_copy2, *hmiddle2;
    int8_t val = 1.0;
    int8_t alpha = 1.0;
    int8_t beta = 0.0;
    vector<size_t> AvMs,AvKs,AvNs;
    size_t Ntglobal = 10240 / nb;
    size_t maxrowsize = state.range(1);
    size_t Avtotalelems(0), xtotalelems(0), yvtotalelems(0);
    size_t M = maxrowsize;
    size_t N = 2;
    size_t K = nb;
    for(int i=0; i<Ntglobal; i++){
        AvMs.push_back( M );
        AvKs.push_back( K );
        AvNs.push_back( N );
        Avtotalelems += M * K;
        xtotalelems += K * N;
        yvtotalelems += M * N;
    }
    size_t hfsize = sizeof(int8_t);
    CUDACHECK(cudaMallocHost(&hAv_real, hfsize * Avtotalelems));
    CUDACHECK(cudaMallocHost(&hx_copy1, hfsize * xtotalelems));
    CUDACHECK(cudaMallocHost(&hmiddle1, hfsize * yvtotalelems));
    CUDACHECK(cudaMallocHost(&hAv_imag, hfsize * Avtotalelems));
    CUDACHECK(cudaMallocHost(&hx_copy2, hfsize * xtotalelems));
    CUDACHECK(cudaMallocHost(&hmiddle2, hfsize * yvtotalelems));
    for(size_t i=0; i<Avtotalelems; i++){
        hAv_real[i] = 1.0;
        hAv_imag[i] = 1.0;
    }
    for(size_t i=0; i<2*xtotalelems; i++){
        hx_copy1[i] = 1.0;
        hx_copy2[i] = 1.0;
    }
    // get device pointer 
    int8_t *Av_real, *x_copy1, *middle1;
    int8_t *Av_imag, *x_copy2, *middle2;
    int8_t **Avbatchpointer_real, **xbatchpointer_real, **yvbatchpointer_real;
    int8_t **Avbatchpointer_imag, **xbatchpointer_imag, **yvbatchpointer_imag;
    GetDeviceMemoryBatched(&Av_real, &x_copy1, &middle1, &Avbatchpointer_real,
    &xbatchpointer_real, &yvbatchpointer_real, AvMs, AvKs, AvNs);
    GetDeviceMemoryBatched(&Av_imag, &x_copy2, &middle2, &Avbatchpointer_imag,
    &xbatchpointer_imag, &yvbatchpointer_imag, AvMs, AvKs, AvNs);
    CopyDataB2HD(Av_real, hAv_real, Avtotalelems);
    CopyDataB2HD(Av_imag, hAv_imag, Avtotalelems);
    CopyDataB2HD(x_copy1, middle1, xtotalelems);
    CopyDataB2HD(x_copy2, middle2, xtotalelems);

    size_t STREAMSIZE = 2;
    cudaStream_t * streamptr = new cudaStream_t[STREAMSIZE];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[STREAMSIZE];
    for(int i=0; i<STREAMSIZE; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<STREAMSIZE; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<STREAMSIZE; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEvent_t *events;
    cudaEvent_t event_start;
    cudaEvent_t event_phase2finish;
    CUDACHECK(cudaEventCreate(&event_start));
    CUDACHECK(cudaEventCreate(&event_phase2finish));
    events = new cudaEvent_t[STREAMSIZE];
    for(int i=0; i<STREAMSIZE; i++) CUDACHECK(cudaEventCreate(&events[i]));

    vector<double> rawtime;
    int8_t *tmpyv_Areal_Bimag, **tmpyv_Areal_Bimag_batched;
    int8_t *tmpyv_Aimag_Breal, **tmpyv_Aimag_Breal_batched;

    for(auto st : state){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        CUBLASCHECK(cublasGemmBatchedEx( cublashandleptr[0], 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        M, N, K, &alpha, (const void**)Avbatchpointer_real, CUDA_C_8I,
        maxrowsize, 
        (const void**)xbatchpointer_real, CUDA_C_8I, nb, &beta, 
        (void **)yvbatchpointer_real, CUDA_C_32F, maxrowsize, Ntglobal,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        // CUBLASCHECK(cublasGemmBatchedEx( cublashandleptr[1], 
        // CUBLAS_OP_N, CUBLAS_OP_N, 
        // M, N, K, &alpha, (const void**)Avbatchpointer_imag, CUDA_R_16F,
        // maxrowsize, 
        // (const void**)xbatchpointer_imag, CUDA_R_16F, nb, &beta, 
        // (void **)yvbatchpointer_imag, CUDA_R_16F, maxrowsize, Ntglobal,
        // CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        cudaEventRecord(events[1], streamptr[1]);
        cudaStreamWaitEvent(streamptr[0], events[1]);
        // phase1twogemv(middle1, middle2, maxrowsize * Ntglobal, streamptr[0]);
        cudaStreamSynchronize(streamptr[0]);
        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
    }
    float hhyv_real;
    float hhyv_imag;
    // CopyDataB2HD(&hhyv_real, &middle1[1], 1);
    float yvfreal = hhyv_real;
    // cout << yvfreal << endl;
    // CopyDataB2HD(&hhyv_imag, &middle2[1], 1);
    float yvimag = hhyv_imag;
    // cout << yvimag << endl;
    // for(int i=0; i<STREAMSIZE; i++) cudaStreamDestroy(streamptr[i]);
    // for(int i=0; i<STREAMSIZE; i++) cublasDestroy(cublashandleptr[i]);
    // double bytes = sizeof(half) * ( Ntglobal * (M * K + K*N + M*K) ) * (double)state.iterations();
    // state.counters["BandWidth"] =
    // Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
}

BENCHMARK(BM_Dense_INT8GEMM)
->Unit(benchmark::kMicrosecond)
// ->Iterations(50)
->Args({256,4096})
// ->ArgsProduct({{256,512},
// benchmark::CreateRange(512, 4096, /*multi=*/2),
// })
->Iterations(1)
->Repetitions(1)->UseManualTime();


void BM_Dense_INT8Padding(benchmark::State &state){
    // configuration
    size_t nb = state.range(0);
    // get host pointer 
    int8_t *hAv_real, *hx_copy1, *hmiddle1;
    int8_t *hAv_imag, *hx_copy2, *hmiddle2;
    int8_t val = 1.0;
    int8_t alpha = 1.0;
    int8_t beta = 0.0;
    vector<size_t> AvMs,AvKs,AvNs;
    size_t Ntglobal = 10240 / nb;
    size_t maxrowsize = state.range(1);
    size_t Avtotalelems(0), xtotalelems(0), yvtotalelems(0);
    size_t M = maxrowsize;
    size_t N = 2;
    size_t K = nb;
    for(int i=0; i<Ntglobal; i++){
        AvMs.push_back( M );
        AvKs.push_back( K );
        AvNs.push_back( N );
        Avtotalelems += M * K;
        xtotalelems += K * N;
        yvtotalelems += M * N;
    }
    size_t hfsize = sizeof(int8_t);
    CUDACHECK(cudaMallocHost(&hAv_real, hfsize * Avtotalelems));
    CUDACHECK(cudaMallocHost(&hx_copy1, hfsize * xtotalelems));
    CUDACHECK(cudaMallocHost(&hmiddle1, hfsize * yvtotalelems));
    CUDACHECK(cudaMallocHost(&hAv_imag, hfsize * Avtotalelems));
    CUDACHECK(cudaMallocHost(&hx_copy2, hfsize * xtotalelems));
    CUDACHECK(cudaMallocHost(&hmiddle2, hfsize * yvtotalelems));
    for(size_t i=0; i<Avtotalelems; i++){
        hAv_real[i] = 1.0;
        hAv_imag[i] = 1.0;
    }
    for(size_t i=0; i<2*xtotalelems; i++){
        hx_copy1[i] = 1.0;
        hx_copy2[i] = 1.0;
    }
    // get device pointer 
    int8_t *Av_real, *x_copy1, *middle1;
    int8_t *Av_imag, *x_copy2, *middle2;
    int8_t **Avbatchpointer_real, **xbatchpointer_real, **yvbatchpointer_real;
    int8_t **Avbatchpointer_imag, **xbatchpointer_imag, **yvbatchpointer_imag;
    GetDeviceMemoryBatched(&Av_real, &x_copy1, &middle1, &Avbatchpointer_real,
    &xbatchpointer_real, &yvbatchpointer_real, AvMs, AvKs, AvNs);
    GetDeviceMemoryBatched(&Av_imag, &x_copy2, &middle2, &Avbatchpointer_imag,
    &xbatchpointer_imag, &yvbatchpointer_imag, AvMs, AvKs, AvNs);
    CopyDataB2HD(Av_real, hAv_real, Avtotalelems);
    CopyDataB2HD(Av_imag, hAv_imag, Avtotalelems);
    CopyDataB2HD(x_copy1, middle1, xtotalelems);
    CopyDataB2HD(x_copy2, middle2, xtotalelems);

    size_t STREAMSIZE = 2;
    cudaStream_t * streamptr = new cudaStream_t[STREAMSIZE];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[STREAMSIZE];
    for(int i=0; i<STREAMSIZE; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<STREAMSIZE; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<STREAMSIZE; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEvent_t *events;
    cudaEvent_t event_start;
    cudaEvent_t event_phase2finish;
    CUDACHECK(cudaEventCreate(&event_start));
    CUDACHECK(cudaEventCreate(&event_phase2finish));
    events = new cudaEvent_t[STREAMSIZE];
    for(int i=0; i<STREAMSIZE; i++) CUDACHECK(cudaEventCreate(&events[i]));

    vector<double> rawtime;
    int8_t *tmpyv_Areal_Bimag, **tmpyv_Areal_Bimag_batched;
    int8_t *tmpyv_Aimag_Breal, **tmpyv_Aimag_Breal_batched;

    for(auto st : state){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        CUBLASCHECK(cublasGemmBatchedEx( cublashandleptr[0], 
        CUBLAS_OP_N, CUBLAS_OP_N, 
        M, N, K, &alpha, (const void**)Avbatchpointer_real, CUDA_C_8I,
        maxrowsize, 
        (const void**)xbatchpointer_real, CUDA_C_8I, nb, &beta, 
        (void **)yvbatchpointer_real, CUDA_C_32F, maxrowsize, Ntglobal,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        // CUBLASCHECK(cublasGemmBatchedEx( cublashandleptr[1], 
        // CUBLAS_OP_N, CUBLAS_OP_N, 
        // M, N, K, &alpha, (const void**)Avbatchpointer_imag, CUDA_R_16F,
        // maxrowsize, 
        // (const void**)xbatchpointer_imag, CUDA_R_16F, nb, &beta, 
        // (void **)yvbatchpointer_imag, CUDA_R_16F, maxrowsize, Ntglobal,
        // CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        cudaEventRecord(events[1], streamptr[1]);
        cudaStreamWaitEvent(streamptr[0], events[1]);
        // phase1twogemv(middle1, middle2, maxrowsize * Ntglobal, streamptr[0]);
        cudaStreamSynchronize(streamptr[0]);
        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
    }
    float hhyv_real;
    float hhyv_imag;
    // CopyDataB2HD(&hhyv_real, &middle1[1], 1);
    float yvfreal = hhyv_real;
    // cout << yvfreal << endl;
    // CopyDataB2HD(&hhyv_imag, &middle2[1], 1);
    float yvimag = hhyv_imag;
    // cout << yvimag << endl;
    // for(int i=0; i<STREAMSIZE; i++) cudaStreamDestroy(streamptr[i]);
    // for(int i=0; i<STREAMSIZE; i++) cublasDestroy(cublashandleptr[i]);
    // double bytes = sizeof(half) * ( Ntglobal * (M * K + K*N + M*K) ) * (double)state.iterations();
    // state.counters["BandWidth"] =
    // Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
}


// void BM_Dense_CGEMVTLRMVMHALFGraph(benchmark::State &state){
//     // configuration
//     size_t nb = state.range(0);

//     // get host pointer 
//     half *hAv_real, *hx_real, *hyv_real;
//     half *hAv_imag, *hx_imag, *hyv_imag;
//     half val = 1.0;
//     half alpha = 1.0;
//     half beta = 0.0;
//     vector<size_t> AvMs,AvKs,AvNs;
//     size_t Ntglobal = 40;
//     size_t maxrowsize = state.range(1);
//     size_t Avtotalelems(0), xtotalelems(0), yvtotalelems(0);
//     size_t M = maxrowsize;
//     size_t N = 1;
//     size_t K = nb;
//     for(int i=0; i<Ntglobal; i++){
//         AvMs.push_back( M );
//         AvKs.push_back( K );
//         AvNs.push_back( N );
//         Avtotalelems += M * K;
//         xtotalelems += K * N;
//         yvtotalelems += M * N;
//     }
//     size_t hfsize = sizeof(half);
//     CUDACHECK(cudaMallocHost(&hAv_real, hfsize * Avtotalelems));
//     CUDACHECK(cudaMallocHost(&hx_real, hfsize * xtotalelems));
//     CUDACHECK(cudaMallocHost(&hyv_real, hfsize * yvtotalelems));
//     CUDACHECK(cudaMallocHost(&hAv_imag, hfsize * Avtotalelems));
//     CUDACHECK(cudaMallocHost(&hx_imag, hfsize * xtotalelems));
//     CUDACHECK(cudaMallocHost(&hyv_imag, hfsize * yvtotalelems));
//     // init data
//     for(size_t i=0; i<Avtotalelems; i++){
//         hAv_real[i] = 1.0;
//         hAv_imag[i] = 1.0;
//     }
//     for(size_t i=0; i<xtotalelems; i++){
//         hx_real[i] = 1.0;
//         hx_imag[i] = 1.0;
//     }
//     // get device pointer 
//     half *Av_real, *x_real, *yv_real;
//     half *Av_imag, *x_imag, *yv_imag;
//     half **Avbatchpointer_real, **xbatchpointer_real, **yvbatchpointer_real;
//     half **Avbatchpointer_imag, **xbatchpointer_imag, **yvbatchpointer_imag;
//     GetDeviceMemoryBatched(&Av_real, &x_real, &yv_real, &Avbatchpointer_real,
//     &xbatchpointer_real, &yvbatchpointer_real, AvMs, AvKs, AvNs);
//     GetDeviceMemoryBatched(&Av_imag, &x_imag, &yv_imag, &Avbatchpointer_imag,
//     &xbatchpointer_imag, &yvbatchpointer_imag, AvMs, AvKs, AvNs);
//     CopyDataB2HD(Av_real, hAv_real, Avtotalelems);
//     CopyDataB2HD(Av_imag, hAv_imag, Avtotalelems);
//     CopyDataB2HD(x_real, hx_real, xtotalelems);
//     CopyDataB2HD(x_imag, hx_imag, xtotalelems);

//     size_t STREAMSIZE = 4;
//     cudaStream_t * streamptr = new cudaStream_t[STREAMSIZE];
//     cublasHandle_t * cublashandleptr = new cublasHandle_t[STREAMSIZE];
//     for(int i=0; i<STREAMSIZE; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
//     for(int i=0; i<STREAMSIZE; i++) cublasCreate_v2(&cublashandleptr[i]);
//     for(int i=0; i<STREAMSIZE; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     // cuda graph
//     bool graphCreated=false;
//     cudaGraph_t graph;
//     cudaGraphExec_t instance;
//     cudaEvent_t *events;
//     cudaEvent_t event_start;
//     cudaEvent_t event_phase2finish;
//     CUDACHECK(cudaEventCreate(&event_start));
//     CUDACHECK(cudaEventCreate(&event_phase2finish));
//     events = new cudaEvent_t[STREAMSIZE];
//     for(int i=0; i<STREAMSIZE; i++) CUDACHECK(cudaEventCreate(&events[i]));

//     vector<double> rawtime;
//     // cout << maxrowsize << endl;
//     half *tmpyv_Areal_Bimag, **tmpyv_Areal_Bimag_batched;
//     half *tmpyv_Aimag_Breal, **tmpyv_Aimag_Breal_batched;
//     getYvbatchedpointer(&tmpyv_Areal_Bimag, &tmpyv_Areal_Bimag_batched, AvMs, AvNs);
//     getYvbatchedpointer(&tmpyv_Aimag_Breal, &tmpyv_Aimag_Breal_batched, AvMs, AvNs);
    
//     for(auto st : state){
//         cudaEventRecord(start);
//         cudaDeviceSynchronize();

//         if(!graphCreated){
//             cudaStreamBeginCapture(streamptr[0],cudaStreamCaptureModeGlobal);
//             cudaEventRecord(event_start, streamptr[0]);
            
//             cudaStreamWaitEvent(streamptr[1], event_start);
//             cudaStreamWaitEvent(streamptr[2], event_start);
//             cudaStreamWaitEvent(streamptr[3], event_start);

//             CUBLASCHECK(cublasGemmBatchedEx( cublashandleptr[0], 
//             CUBLAS_OP_N, CUBLAS_OP_N, 
//             M, N, K, &alpha, (const void**)Avbatchpointer_real, CUDA_R_16F,
//             maxrowsize, 
//             (const void**)xbatchpointer_real, CUDA_R_16F, nb, &beta, 
//             (void **)yvbatchpointer_real, CUDA_R_16F, maxrowsize, Ntglobal,
//             CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
//             CUBLASCHECK(cublasGemmBatchedEx( cublashandleptr[1], 
//             CUBLAS_OP_N, CUBLAS_OP_N, 
//             M, N, K, &alpha, (const void**)Avbatchpointer_imag, CUDA_R_16F,
//             maxrowsize, 
//             (const void**)xbatchpointer_imag, CUDA_R_16F, nb, &beta, 
//             (void **)yvbatchpointer_imag, CUDA_R_16F, maxrowsize, Ntglobal,
//             CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
//             // CUBLASCHECK(cublasGemmBatchedEx( cublashandleptr[2], 
//             // CUBLAS_OP_N, CUBLAS_OP_N, 
//             // M, N, K, &alpha, (const void**)Avbatchpointer_real, CUDA_R_16F,
//             // maxrowsize, 
//             // (const void**)xbatchpointer_imag, CUDA_R_16F, nb, &beta, 
//             // (void **)tmpyv_Areal_Bimag_batched, CUDA_R_16F, maxrowsize, Ntglobal,
//             // CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
//             // CUBLASCHECK(cublasGemmBatchedEx( cublashandleptr[3], 
//             // CUBLAS_OP_N, CUBLAS_OP_N, 
//             // M, N, K, &alpha, (const void**)Avbatchpointer_imag, CUDA_R_16F,
//             // maxrowsize, 
//             // (const void**)xbatchpointer_real, CUDA_R_16F, nb, &beta, 
//             // (void **)tmpyv_Aimag_Breal_batched, CUDA_R_16F, maxrowsize, Ntglobal,
//             // CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));

//             cudaEventRecord(events[1], streamptr[1]);
//             cudaStreamWaitEvent(streamptr[0], events[1]);
//             phase1fuse4ops_real(yv_real, yv_imag, maxrowsize * Ntglobal, streamptr[0]);
//             cudaEventRecord(events[3], streamptr[3]);
//             cudaStreamWaitEvent(streamptr[2], events[3]);
//             phase1fuse4ops_imag(tmpyv_Areal_Bimag, tmpyv_Aimag_Breal, maxrowsize*Ntglobal, streamptr[2]);
//             cudaEventRecord(events[2], streamptr[2]);
//             cudaStreamWaitEvent(streamptr[0], events[2]);    
//             cudaStreamEndCapture(streamptr[0], &graph);
//             cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
//             graphCreated=true;
//         }
//         cudaGraphLaunch(instance, streamptr[0]);
//         cudaStreamSynchronize(streamptr[0]);
//         cudaEventRecord(stop);
//         float milliseconds = 0;
//         cudaEventSynchronize(stop);
//         cudaEventElapsedTime(&milliseconds, start, stop);
//         state.SetIterationTime(milliseconds*1e-3);
//         rawtime.push_back(milliseconds*1e-3);
//     }
//     complex<float> hhyv;
//     for(int i=0; i<STREAMSIZE; i++) cudaStreamDestroy(streamptr[i]);
//     for(int i=0; i<STREAMSIZE; i++) cublasDestroy(cublashandleptr[i]);
//     double bytes = sizeof(half) * ( Ntglobal * (M * K + K*N + M*K) ) * (double)state.iterations();
//     state.counters["BandWidth"] =
//     Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);

// }

// BENCHMARK(BM_Dense_CGEMVTLRMVMHALFGraph)
// ->Unit(benchmark::kMicrosecond)
// // ->Args({512,4096})
// ->ArgsProduct({{512},
// benchmark::CreateRange(512, 4096, /*multi=*/2),
// })
// ->Iterations(50)
// ->Repetitions(1)->UseManualTime();

// void BM_Dense_FP16GEMV(benchmark::State &state){
//     // configuration
//     size_t m = state.range(0);
//     size_t k = state.range(1);
//     size_t n = 1;
//     size_t f_sz = sizeof(float);
//     size_t c_sz = sizeof(cuComplex);
//     complex<float> *hA, *hx, *hy;
//     hA = new complex<float>[m * k];
//     hx = new complex<float>[k];
//     hy = new complex<float>[m];
//     complex<float> val = complex<float>(1.0, 0.0);
//     Init(hA, m * k, val);
//     Init(hx, k, val);
//     Init(hy, m, val);
//     cuComplex *A, *x, *y;
//     CUDACHECK(cudaMalloc(&A, m * k * c_sz));
//     CUDACHECK(cudaMalloc(&x, k * c_sz));
//     CUDACHECK(cudaMalloc(&y, m * c_sz));
//     CUDACHECK(cudaMemcpy(A, hA, m * k * c_sz, cudaMemcpyDefault));
//     CUDACHECK(cudaMemcpy(x, hx, k * c_sz, cudaMemcpyDefault));
//     CUDACHECK(cudaMemcpy(y, hy, m * c_sz, cudaMemcpyDefault));
//     // timer
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     // create cublas handle
//     cublasHandle_t handle;
//     cublasCreate(&handle);
//     cudaStream_t stream;
//     CUDACHECK(cudaStreamCreate(&stream));
//     cuComplex alpha, beta;
//     alpha.x = 1.0;
//     alpha.y = 0.0;
//     beta.x = 0.0;
//     beta.y = 0.0;
//     vector<double> rawtime;
//     for(auto st : state){
//         cudaEventRecord(start);
//         cudaDeviceSynchronize();
//         CUBLASCHECK(cublasgemv(handle, CUBLAS_OP_N, m, k, &alpha, A, m, x, 1, &beta, y, 1));
//         cudaDeviceSynchronize();
//         cudaEventRecord(stop);
//         float milliseconds = 0;
//         cudaEventSynchronize(stop);
//         cudaEventElapsedTime(&milliseconds, start, stop);
//         state.SetIterationTime(milliseconds*1e-3);
//         rawtime.push_back(milliseconds*1e-3);
//     }
//     double bytes = sizeof(complex<float>) * (m * k + m + k) * (double)state.iterations();
//     state.counters["BandWidth"] =
//     Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
//     cublasDestroy(handle);
//     CUDACHECK(cudaFree(A));
//     CUDACHECK(cudaFree(x));
//     CUDACHECK(cudaFree(y));
//     delete[] hA;
//     delete[] hx;
//     delete[] hy;
// }

// BENCHMARK(BM_Dense_FP16GEMV)->Args({7500, 10000})
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(5000)->Repetitions(1)->UseManualTime();

// BENCHMARK(BM_Dense_FP16GEMV)->Args({75, 10000})
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(5000)->Repetitions(1)->UseManualTime();

// BENCHMARK(BM_Dense_FP16GEMV)->Args({7500, 100})
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(5000)->Repetitions(1)->UseManualTime();

// BENCHMARK(BM_Dense_FP16GEMV)->Args({75*50, 10000})
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(5000)->Repetitions(1)->UseManualTime();

// BENCHMARK(BM_Dense_FP16GEMV)->Args({7500, 100*50})
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(5000)->Repetitions(1)->UseManualTime();





// void BM_Dense_SGEMV(benchmark::State &state){

//     // configuration
//     size_t m = state.range(0);
//     size_t n = state.range(1);
//     size_t f_sz = sizeof(float);
//     size_t float_sz = sizeof(float);
//     int k = 1;
//     float *hA, *hx, *hy;
//     hA = new float[m * n];
//     hx = new float[n];
//     hy = new float[m];
//     float val = 1.0;
//     Init(hA, m * n, val);
//     Init(hx, n, val);
//     Init(hy, m, val);
//     float *A, *x, *y;
//     CUDACHECK(cudaMalloc(&A, m * n * float_sz));
//     CUDACHECK(cudaMalloc(&x, n * float_sz));
//     CUDACHECK(cudaMalloc(&y, m * float_sz));
//     CUDACHECK(cudaMemcpy(A, hA, m * n * float_sz, cudaMemcpyDefault));
//     CUDACHECK(cudaMemcpy(x, hx, n * float_sz, cudaMemcpyDefault));
//     CUDACHECK(cudaMemcpy(y, hy, m * float_sz, cudaMemcpyDefault));
//     // timer
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     // create cublas handle
//     cublasHandle_t handle;
//     cublasCreate(&handle);
//     cudaStream_t stream;
//     CUDACHECK(cudaStreamCreate(&stream));
//     float alpha, beta;
//     alpha = 1.0;
//     beta = 0.0;
//     vector<double> rawtime;
//     for(auto st : state){
//         cudaEventRecord(start);
//         cudaDeviceSynchronize();
//         CUBLASCHECK(cublasgemv(handle, CUBLAS_OP_N, m, n, &alpha, A, m, x, 1, &beta, y, 1));
//         cudaDeviceSynchronize();
//         cudaEventRecord(stop);
//         float milliseconds = 0;
//         cudaEventSynchronize(stop);
//         cudaEventElapsedTime(&milliseconds, start, stop);
//         state.SetIterationTime(milliseconds*1e-3);
//         rawtime.push_back(milliseconds*1e-3);
//     }
//     double bytes = sizeof(float) * (m * n + m + n) * (double)state.iterations();
//     state.counters["BandWidth"] =
//     Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
//     cublasDestroy(handle);
//     CUDACHECK(cudaFree(A));
//     CUDACHECK(cudaFree(x));
//     CUDACHECK(cudaFree(y));
//     delete[] hA;
//     delete[] hx;
//     delete[] hy;

// }

// BENCHMARK(BM_Dense_SGEMV)->Args({2500, 10000})
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(5000)->Repetitions(1)->UseManualTime();

// void BM_Dense_StreamCGEMV(benchmark::State &state){
//     vector<size_t> Ms,Ks,Ns;
//     int batchsize = state.range(0);
//     int nb = state.range(1);
//     int L = state.range(2);
//     for(int i=0; i<batchsize; i++){
//         Ms.push_back( rand() % L );
//         Ks.push_back(nb);
//         Ns.push_back(1);
//     }
//     complex<float> *hA, *hB, *hC, **hAbp, **hBbp, **hCbp, val;
//     val = complex<float>(1.0,0.0);

//     GetHostMemoryBatched(&hA, &hB, &hC, 
//     &hAbp, &hBbp, &hCbp, Ms, Ks, Ns, val);

//     cuComplex *A, *B, *C;
//     cuComplex **Abatchpointer, **Bbatchpointer, **Cbatchpointer;

//     GetDeviceMemoryBatched(&A, &B, &C, 
//     &Abatchpointer, &Bbatchpointer, &Cbatchpointer, Ms, Ks, Ns);

//     cuComplex **Abatchpointer_h,**Bbatchpointer_h, **Cbatchpointer_h;
//     Abatchpointer_h = new cuComplex*[batchsize];
//     Bbatchpointer_h = new cuComplex*[batchsize];
//     Cbatchpointer_h = new cuComplex*[batchsize];
//     CUDACHECK( cudaMemcpy(Abatchpointer_h, Abatchpointer, 
//     batchsize * sizeof(cuComplex*), cudaMemcpyDefault) );
//     CUDACHECK( cudaMemcpy(Bbatchpointer_h, Bbatchpointer, 
//     batchsize * sizeof(cuComplex*), cudaMemcpyDefault) );
//     CUDACHECK( cudaMemcpy(Cbatchpointer_h, Cbatchpointer, 
//     batchsize * sizeof(cuComplex*), cudaMemcpyDefault) );

//     CopyDataB2HD((complex<float>*)A, (complex<float>*)B, (complex<float>*)C, 
//     hA, hB, hC, Ms, Ks, Ns);

//     cuComplex alpha, beta;
//     alpha.x = 1.0; 
//     alpha.y = beta.x = beta.y = 0;
//     /**************************
//      * CUBLAS CALL 
//      * ************************/
    
//     cudaStream_t * streamptr = new cudaStream_t[batchsize];
//     cublasHandle_t * cublashandleptr = new cublasHandle_t[batchsize];
//     for(int i=0; i<batchsize; i++) cudaStreamCreate(&streamptr[i]);
//     for(int i=0; i<batchsize; i++) cublasCreate_v2(&cublashandleptr[i]);

//     // timer
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     vector<double> rawtime;

//     for(auto st : state){
//         cudaEventRecord(start);
//         cudaDeviceSynchronize();

//         for(int i=0; i<batchsize; i++){
//             cublasSetStream_v2(cublashandleptr[i], streamptr[i]);
//             CUBLASCHECK(
//             cublasgemv(cublashandleptr[i], CUBLAS_OP_N,
//             Ms[i], Ks[i],
//             &alpha, (const cuComplex*)Abatchpointer_h[i], Ms[i], 
//             (const cuComplex*)Bbatchpointer_h[i], 1, &beta, 
//             Cbatchpointer_h[i], 1);
//             );
//         }

//         cudaDeviceSynchronize();
//         cudaEventRecord(stop);
//         float milliseconds = 0;
//         cudaEventSynchronize(stop);
//         cudaEventElapsedTime(&milliseconds, start, stop);
//         state.SetIterationTime(milliseconds*1e-3);
//         rawtime.push_back(milliseconds*1e-3);
//     }
//     size_t Atotalelems, Btotalelems, Ctotalelems;
//     tlrmat::CaluclateTotalElements(Ms, Ks, Ns, Atotalelems, Btotalelems, Ctotalelems);
//     double bytes = sizeof(float) * (Atotalelems + Btotalelems + Ctotalelems) * (double)state.iterations();
//     state.counters["BandWidth"] =
//     Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);

//     for(int i=0; i<batchsize; i++) cudaStreamDestroy(streamptr[i]);
//     for(int i=0; i<batchsize; i++) cublasDestroy_v2(cublashandleptr[i]);
//     delete[] streamptr;
//     delete[] cublashandleptr;
    
//     FreeDeviceMemoryBatched(A,B,C,Abatchpointer, Bbatchpointer,
//     Cbatchpointer);
//     FreeHostMemoryBatched(hA, hB, hC, hAbp, hBbp, hCbp);
//     delete[] Abatchpointer_h;
//     delete[] Bbatchpointer_h;
//     delete[] Cbatchpointer_h;
// }

// // BENCHMARK(BM_Dense_StreamCGEMV)->Args({40, 256, 2000})
// // ->Unit(benchmark::kMicrosecond)
// // ->Iterations(5000)->Repetitions(1)->UseManualTime();



// void BM_Dense_StreamSGEMV(benchmark::State &state){
//     vector<size_t> Ms,Ks,Ns;
//     int batchsize = state.range(0);
//     int nb = state.range(1);
//     int L = state.range(2);
//     for(int i=0; i<batchsize; i++){
//         Ms.push_back( rand() % L );
//         Ks.push_back(nb);
//         Ns.push_back(1);
//     }
//     float *hA, *hB, *hC, **hAbp, **hBbp, **hCbp, val;
//     val = 1.0;

//     GetHostMemoryBatched(&hA, &hB, &hC, 
//     &hAbp, &hBbp, &hCbp, Ms, Ks, Ns, val);

//     float *A, *B, *C;
//     float **Abatchpointer, **Bbatchpointer, **Cbatchpointer;

//     GetDeviceMemoryBatched(&A, &B, &C, 
//     &Abatchpointer, &Bbatchpointer, &Cbatchpointer, Ms, Ks, Ns);

//     float **Abatchpointer_h,**Bbatchpointer_h, **Cbatchpointer_h;
//     Abatchpointer_h = new float*[batchsize];
//     Bbatchpointer_h = new float*[batchsize];
//     Cbatchpointer_h = new float*[batchsize];
//     CUDACHECK( cudaMemcpy(Abatchpointer_h, Abatchpointer, 
//     batchsize * sizeof(float*), cudaMemcpyDefault) );
//     CUDACHECK( cudaMemcpy(Bbatchpointer_h, Bbatchpointer, 
//     batchsize * sizeof(float*), cudaMemcpyDefault) );
//     CUDACHECK( cudaMemcpy(Cbatchpointer_h, Cbatchpointer, 
//     batchsize * sizeof(float*), cudaMemcpyDefault) );

//     CopyDataB2HD((float*)A, (float*)B, (float*)C, 
//     hA, hB, hC, Ms, Ks, Ns);

//     float alpha, beta;
//     alpha = 1.0;
//     beta = 0.0;
//     /**************************
//      * CUBLAS CALL 
//      * ************************/
    
//     cudaStream_t * streamptr = new cudaStream_t[batchsize];
//     cublasHandle_t * cublashandleptr = new cublasHandle_t[batchsize];
//     for(int i=0; i<batchsize; i++) cudaStreamCreate(&streamptr[i]);
//     for(int i=0; i<batchsize; i++) cublasCreate_v2(&cublashandleptr[i]);

//     // timer
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     vector<double> rawtime;

//     for(auto st : state){
//         cudaEventRecord(start);
//         cudaDeviceSynchronize();

//         for(int i=0; i<batchsize; i++){
//             cublasSetStream_v2(cublashandleptr[i], streamptr[i]);
//             CUBLASCHECK(
//             cublasgemv(cublashandleptr[i], CUBLAS_OP_N,
//             Ms[i], Ks[i],
//             &alpha, (const float*)Abatchpointer_h[i], Ms[i], 
//             (const float*)Bbatchpointer_h[i], 1, &beta, 
//             Cbatchpointer_h[i], 1);
//             );
//         }

//         cudaDeviceSynchronize();
//         cudaEventRecord(stop);
//         float milliseconds = 0;
//         cudaEventSynchronize(stop);
//         cudaEventElapsedTime(&milliseconds, start, stop);
//         state.SetIterationTime(milliseconds*1e-3);
//         rawtime.push_back(milliseconds*1e-3);
//     }
//     size_t Atotalelems, Btotalelems, Ctotalelems;
//     tlrmat::CaluclateTotalElements(Ms, Ks, Ns, Atotalelems, Btotalelems, Ctotalelems);
//     double bytes = sizeof(float) * (Atotalelems + Btotalelems + Ctotalelems) * (double)state.iterations();
//     state.counters["BandWidth"] =
//     Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);

//     for(int i=0; i<batchsize; i++) cudaStreamDestroy(streamptr[i]);
//     for(int i=0; i<batchsize; i++) cublasDestroy_v2(cublashandleptr[i]);
//     delete[] streamptr;
//     delete[] cublashandleptr;
    
//     FreeDeviceMemoryBatched(A,B,C,Abatchpointer, Bbatchpointer,
//     Cbatchpointer);
//     FreeHostMemoryBatched(hA, hB, hC, hAbp, hBbp, hCbp);
//     delete[] Abatchpointer_h;
//     delete[] Bbatchpointer_h;
//     delete[] Cbatchpointer_h;
// }

// BENCHMARK(BM_Dense_StreamSGEMV)->Args({40, 256, 2000})
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(10)->Repetitions(1)->UseManualTime();


BENCHMARK_MAIN();