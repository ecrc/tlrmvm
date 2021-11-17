#include <stdio.h>
#include <string>
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
#include <vector>
#include <complex>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h> // if you need CUBLAS v2, include before magma.h
using std::complex;
using std::string;
using std::cout;
using std::endl;
using std::vector;

void getalpabeta(float &alpha, float &beta){
    alpha = 1.0;
    beta = 0.0;
}

void getalpabeta(complex<float> &alpha, complex<float> &beta){
    alpha = complex<float>(1.0,0.0);
    beta = complex<float>(0.0,0.0);
}

void getalpabeta(cuComplex &alpha, cuComplex &beta){
    alpha.x = 1.0; alpha.y = 0.0;
    beta.x = 1.0; beta.y = 0.0;
}

void initdata(float * a, float *x, int m, int n){
    for(int i=0; i<m * n; i++) a[i] = (float)1.0;
    for(int i=0; i<n; i++) x[i] = (float)1.0;
}

void initdata(complex<float> * a, complex<float> *x, int m, int n){
    for(int i=0; i<m * n; i++) a[i] = complex<float>(1.0,1.0);
    for(int i=0; i<n; i++) x[i] = complex<float>(1.0,1.0);
}

void gemv(cublasHandle_t handle, int m, int n, float *a, float *x, float *y, float alpha, float beta){
    cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, (const float *)a, 
    m, (const float *)x,  1, &beta, y, 1);
}

void gemv(cublasHandle_t handle, int m, int n, cuComplex *a, cuComplex *x, cuComplex *y, cuComplex alpha, cuComplex beta){
    cublasCgemv(handle, CUBLAS_OP_N, m, n, &alpha, (const cuComplex *)a,
    m, (const cuComplex *)x,  1, &beta, y, 1);
}


template<typename hosttype, typename devicetype>
void run_gemv_example(int M, int N, int loopsize){
    cublasHandle_t handle;
    cublasCreate(&handle);
    hosttype *a, *x;
    int sizea = M * N;
    int sizex = N;
    int sizey = M;
    a = new hosttype[sizea];
    x = new hosttype[sizex];
    initdata(a,x,M,N);
    devicetype *da, *dx, *dy;
    devicetype alpha, beta;
    getalpabeta(alpha, beta);
    cudaMalloc(&da, sizeof(devicetype) * sizea);
    cudaMalloc(&dx, sizeof(devicetype) * sizex);
    cudaMalloc(&dy, sizeof(devicetype) * sizey);
    cudaMemcpy(da, a, sizeof(devicetype) * sizea, cudaMemcpyDefault);
    cudaMemcpy(dx, x, sizeof(devicetype) * sizex,cudaMemcpyDefault);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    vector<double> rawtime;
    for(int i=0; i<loopsize; i++){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        gemv(handle, M, N, da, dx, dy, alpha, beta);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        rawtime.push_back(milliseconds*1e-3);
    }
    std::sort(rawtime.begin(),rawtime.end());
    double mdtime = rawtime[rawtime.size() / 2];
    cout << "median time " << mdtime *1e6 << " us. " << endl;
    double bytes = sizeof(hosttype) * (M * N + M + N);
    double bd = bytes / mdtime * 1e-9;
    cout << "bandwidth " << bd << " GB/s " << endl;
}



int main (int argc, char ** argv){
    int M;
    int N;
    int loopsize;
    string runtype;
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    loopsize = atoi(argv[3]);
    runtype = string(argv[4]);
    if(runtype == "float"){
        run_gemv_example<float,float>(M, N, loopsize);
    }else{
        run_gemv_example<complex<float>, cuComplex>(M, N, loopsize);
    }
    
    
    return 0;
}


