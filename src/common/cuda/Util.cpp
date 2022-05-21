#include "Util.hpp"

#include "../cpu/Util.hpp"

#include <cassert>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h> // if you need CUBLAS v2, include before magma.h
#include <cuComplex.h>

template<typename T>
void GetcuHostMemory(T **A, size_t m){
    size_t elesize = sizeof(T);
    CUDACHECK(cudaMallocHost(A, elesize * m));
}
template void GetcuHostMemory<int8_t>(int8_t **, size_t);
template void GetcuHostMemory<int8_t*>(int8_t* **, size_t);

template void GetcuHostMemory<half>(half **, size_t);
template void GetcuHostMemory<half*>(half* **, size_t);

template void GetcuHostMemory<float>(float **, size_t);
template void GetcuHostMemory<float*>(float* **, size_t);

template void GetcuHostMemory<double>(double **, size_t);
template void GetcuHostMemory<double*>(double* **, size_t);

template void GetcuHostMemory<cuInt8Complex>(cuInt8Complex **, size_t);
template void GetcuHostMemory<cuInt8Complex*>(cuInt8Complex* **, size_t);

template void GetcuHostMemory<cuHalfComplex>(cuHalfComplex **, size_t);
template void GetcuHostMemory<cuHalfComplex*>(cuHalfComplex* **, size_t);
template void GetcuHostMemory<cuHalfComplex**>(cuHalfComplex** **, size_t);

template void GetcuHostMemory<cubfComplex>(cubfComplex **, size_t);
template void GetcuHostMemory<cubfComplex*>(cubfComplex* **, size_t);
template void GetcuHostMemory<cubfComplex**>(cubfComplex** **, size_t);


template void GetcuHostMemory<cuComplex>(cuComplex **, size_t);
template void GetcuHostMemory<cuComplex*>(cuComplex* **, size_t);

template void GetcuHostMemory<cuDoubleComplex>(cuDoubleComplex **, size_t);
template void GetcuHostMemory<cuDoubleComplex*>(cuDoubleComplex* **, size_t);



template<typename T>
void FreecuHostMemory(T *A){
    CUDACHECK(cudaFreeHost(A));
}
template void FreecuHostMemory<int8_t>(int8_t *);
template void FreecuHostMemory<int8_t*>(int8_t* *);

template void FreecuHostMemory<half>(half *);
template void FreecuHostMemory<half*>(half* *);

template void FreecuHostMemory<float>(float *);
template void FreecuHostMemory<float*>(float* *);

template void FreecuHostMemory<double>(double *);
template void FreecuHostMemory<double*>(double* *);

template void FreecuHostMemory<cuInt8Complex>(cuInt8Complex *);
template void FreecuHostMemory<cuInt8Complex*>(cuInt8Complex* *);

template void FreecuHostMemory<cuHalfComplex>(cuHalfComplex *);
template void FreecuHostMemory<cuHalfComplex*>(cuHalfComplex* *);

template void FreecuHostMemory<cubfComplex>(cubfComplex *);
template void FreecuHostMemory<cubfComplex*>(cubfComplex* *);

template void FreecuHostMemory<cuComplex>(cuComplex *);
template void FreecuHostMemory<cuComplex*>(cuComplex* *);

template void FreecuHostMemory<cuDoubleComplex>(cuDoubleComplex *);
template void FreecuHostMemory<cuDoubleComplex*>(cuDoubleComplex* *);



template<typename T>
void GetDeviceMemory(T **A, size_t m){
    size_t elesize = sizeof(T);
    CUDACHECK(cudaMalloc(&A[0], m * elesize));
}

template void GetDeviceMemory<unsigned long int>(unsigned long int**, size_t);
template void GetDeviceMemory<int>(int**, size_t);
template void GetDeviceMemory<int8_t>(int8_t**, size_t);
template void GetDeviceMemory<int8_t*>(int8_t***, size_t);
template void GetDeviceMemory<int8_t**>(int8_t****, size_t);
template void GetDeviceMemory<float>(float**, size_t);
template void GetDeviceMemory<float*>(float***, size_t);
template void GetDeviceMemory<float**>(float****, size_t);
template void GetDeviceMemory<double>(double**, size_t);
template void GetDeviceMemory<double*>(double***, size_t);
template void GetDeviceMemory<double**>(double****, size_t);
template void GetDeviceMemory<complex<float>>(complex<float>**, size_t);
template void GetDeviceMemory<complex<float>*>(complex<float>***, size_t);
template void GetDeviceMemory<complex<float>**>(complex<float>****, size_t);
template void GetDeviceMemory<cuComplex>(cuComplex**, size_t);
template void GetDeviceMemory<cuComplex*>(cuComplex***, size_t);
template void GetDeviceMemory<cuComplex**>(cuComplex****, size_t);
template void GetDeviceMemory<cuDoubleComplex>(cuDoubleComplex**, size_t);
template void GetDeviceMemory<cuDoubleComplex*>(cuDoubleComplex* **, size_t);
template void GetDeviceMemory<cuDoubleComplex**>(cuDoubleComplex** **, size_t);
template void GetDeviceMemory<half>(half**, size_t);
template void GetDeviceMemory<half*>(half***, size_t);
template void GetDeviceMemory<half**>(half****, size_t);
template void GetDeviceMemory<half2>(half2**, size_t);
template void GetDeviceMemory<cuHalfComplex*>(cuHalfComplex***, size_t);
template void GetDeviceMemory<cuHalfComplex>(cuHalfComplex**, size_t);
template void GetDeviceMemory<cubfComplex*>(cubfComplex***, size_t);
template void GetDeviceMemory<cubfComplex>(cubfComplex**, size_t);
template void GetDeviceMemory<cuInt8Complex*>(cuInt8Complex* **, size_t);
template void GetDeviceMemory<cuInt8Complex>(cuInt8Complex**, size_t);

template<typename T>
void FreeDeviceMemory(T *A){
    CUDACHECK(cudaFree(A));
}

template void FreeDeviceMemory<unsigned long int>(unsigned long int*);
template void FreeDeviceMemory<int>(int*);
template void FreeDeviceMemory<int8_t>(int8_t*);
template void FreeDeviceMemory<int8_t*>(int8_t**);
template void FreeDeviceMemory<int8_t**>(int8_t***);
template void FreeDeviceMemory<float>(float*);
template void FreeDeviceMemory<float*>(float**);
template void FreeDeviceMemory<float**>(float***);
template void FreeDeviceMemory<double>(double*);
template void FreeDeviceMemory<double*>(double**);
template void FreeDeviceMemory<double**>(double***);
template void FreeDeviceMemory<cuComplex>(cuComplex*);
template void FreeDeviceMemory<cuComplex*>(cuComplex**);
template void FreeDeviceMemory<cuComplex**>(cuComplex***);
template void FreeDeviceMemory<cuDoubleComplex>(cuDoubleComplex*);
template void FreeDeviceMemory<cuDoubleComplex*>(cuDoubleComplex* *);
template void FreeDeviceMemory<cuDoubleComplex**>(cuDoubleComplex** *);
template void FreeDeviceMemory<half>(half*);
template void FreeDeviceMemory<half*>(half**);
template void FreeDeviceMemory<half**>(half***);
template void FreeDeviceMemory<cuHalfComplex>(cuHalfComplex*);
template void FreeDeviceMemory<cuHalfComplex*>(cuHalfComplex**);
template void FreeDeviceMemory<cuHalfComplex**>(cuHalfComplex***);
template void FreeDeviceMemory<cubfComplex>(cubfComplex*);
template void FreeDeviceMemory<cubfComplex*>(cubfComplex**);
template void FreeDeviceMemory<cubfComplex**>(cubfComplex***);
template void FreeDeviceMemory<cuInt8Complex>(cuInt8Complex*);
template void FreeDeviceMemory<cuInt8Complex*>(cuInt8Complex**);
template void FreeDeviceMemory<cuInt8Complex**>(cuInt8Complex***);


template<typename T>
void GetDeviceMemory(T **A, T **B, T **C, size_t m, size_t k, size_t n){
    size_t elesize = sizeof(T);
    CUDACHECK(cudaMalloc(&A[0], m * k * elesize));
    CUDACHECK(cudaMalloc(&B[0], k * n * elesize));
    CUDACHECK(cudaMalloc(&C[0], m * n * elesize));
}

template void GetDeviceMemory<int8_t>(int8_t**, int8_t**, int8_t**,
size_t, size_t, size_t);
template void GetDeviceMemory<float>(float**, float**, float**,
size_t, size_t, size_t);
template void GetDeviceMemory<cuComplex>(cuComplex**, cuComplex**, cuComplex**,
size_t, size_t, size_t);


template<typename T>
void FreeDeviceMemory(T *A, T *B, T *C){
    CUDACHECK(cudaFree(A));
    CUDACHECK(cudaFree(B));
    CUDACHECK(cudaFree(C));
}

template void FreeDeviceMemory(float *A, float *B, float *C);
template void FreeDeviceMemory(cuComplex *A, cuComplex *B, cuComplex *C);

template<typename T>
void FreeDeviceMemoryBatched(T *A, T *B, T *C, 
T **Abatchpointer, T **Bbatchpointer, T **Cbatchpointer){
    FreeDeviceMemory(A,B,C);
    CUDACHECK(cudaFree(Abatchpointer));
    CUDACHECK(cudaFree(Bbatchpointer));
    CUDACHECK(cudaFree(Cbatchpointer));
}
template void FreeDeviceMemoryBatched(float *A, float *B, float *C, 
float **Abatchpointer, float **Bbatchpointer, float **Cbatchpointer);
template void FreeDeviceMemoryBatched(cuComplex *A, cuComplex *B, cuComplex *C, 
cuComplex **Abatchpointer, cuComplex **Bbatchpointer, cuComplex **Cbatchpointer);


template<typename T>
void CopyDataB2HD(T *dstA, T *dstB, T *dstC,
T *srcA, T *srcB, T *srcC, vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns){
    size_t Atotalelems, Btotalelems, Ctotalelems;
    CaluclateTotalElements(Ms, Ks, Ns, Atotalelems, Btotalelems, Ctotalelems);
    CUDACHECK( cudaMemcpy((void*)dstA, (void*)srcA, Atotalelems * sizeof(T), cudaMemcpyDefault) );
    CUDACHECK( cudaMemcpy((void*)dstB, (void*)srcB, Btotalelems * sizeof(T), cudaMemcpyDefault) );
    CUDACHECK( cudaMemcpy((void*)dstC, (void*)srcC, Ctotalelems * sizeof(T), cudaMemcpyDefault) );    
}

template void CopyDataB2HD(float *dstA, float *dstB, float *dstC,
float *srcA, float *srcB, float *srcC, vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns);
template void CopyDataB2HD(complex<float> *dstA, complex<float> *dstB, complex<float> *dstC,
complex<float> *srcA, complex<float> *srcB, complex<float> *srcC, vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns);
template void CopyDataB2HD(int8_t *dstA, int8_t *dstB, int8_t *dstC,
int8_t *srcA, int8_t *srcB, int8_t *srcC, vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns);


template<typename T>
void CopyDataB2HD(T *dstA, T *srcA, size_t length){
    CUDACHECK( cudaMemcpy((void*)dstA, (void*)srcA, length * sizeof(T), cudaMemcpyDefault) );
}

template void CopyDataB2HD<int8_t>(int8_t *dstA, int8_t *srcA, size_t length);
template void CopyDataB2HD<int8_t*>(int8_t* *dstA, int8_t* *srcA, size_t length);
template void CopyDataB2HD<int>(int *dstA, int *srcA, size_t length);
template void CopyDataB2HD<unsigned long int>(unsigned long int *dstA, unsigned long int *srcA, size_t length);
template void CopyDataB2HD<float>(float *dstA, float *srcA, size_t length);
template void CopyDataB2HD<float*>(float* *dstA, float* *srcA, size_t length);
template void CopyDataB2HD<complex<float>>(complex<float> *dstA, complex<float> *srcA, size_t length);
template void CopyDataB2HD<complex<float>*>(complex<float> **dstA, complex<float> **srcA, size_t length);
template void CopyDataB2HD<double>(double *dstA, double *srcA, size_t length);
template void CopyDataB2HD<double*>(double **dstA, double **srcA, size_t length);
template void CopyDataB2HD<complex<double>>(complex<double> *dstA, complex<double> *srcA, size_t length);
template void CopyDataB2HD<complex<double>*>(complex<double> **dstA, complex<double> **srcA, size_t length);
template void CopyDataB2HD<half>(half *dstA, half *srcA, size_t length);
template void CopyDataB2HD<half*>(half* *dstA, half* *srcA, size_t length);
template void CopyDataB2HD<cuComplex>(cuComplex *dstA,cuComplex *srcA, size_t length);
template void CopyDataB2HD<cuHalfComplex>(cuHalfComplex *dstA,cuHalfComplex *srcA, size_t length);
template void CopyDataB2HD<cubfComplex>(cubfComplex *dstA,cubfComplex *srcA, size_t length);
template void CopyDataB2HD<cuInt8Complex>(cuInt8Complex *dstA,cuInt8Complex *srcA, size_t length);


void CopyDataB2HD(cuHalfComplex * dstA, complex<float> *srcA, size_t length){
    auto hAhalfComplex = new cuHalfComplex[length];
    for(int i=0; i<length; i++) hAhalfComplex[i] = cuHalfComplex(srcA[i].real(), srcA[i].imag());
    CopyDataB2HD(dstA, hAhalfComplex, length);
    delete[] hAhalfComplex;
}

void CopyDataB2HD(cubfComplex * dstA, complex<float> *srcA, size_t length){
    auto hAhalfComplex = new cubfComplex[length];
    for(int i=0; i<length; i++) hAhalfComplex[i] = cubfComplex(srcA[i].real(), srcA[i].imag());
    CopyDataB2HD(dstA, hAhalfComplex, length);
    delete[] hAhalfComplex;
}

void CopyDataB2HD(complex<float> *dstA, cuHalfComplex *srcA, size_t length){
    auto hAhalfComplex = new cuHalfComplex[length];
    CopyDataB2HD(hAhalfComplex, srcA, length);
    for(int i=0; i<length; i++) dstA[i] = complex<float>((float)hAhalfComplex[i].x, (float)hAhalfComplex[i].y);
    delete[] hAhalfComplex;
}

void CopyDataB2HD(complex<float> *dstA, cubfComplex *srcA, size_t length){
    auto hAhalfComplex = new cubfComplex[length];
    CopyDataB2HD(hAhalfComplex, srcA, length);
    for(int i=0; i<length; i++) dstA[i] = complex<float>((float)hAhalfComplex[i].x, (float)hAhalfComplex[i].y);
    delete[] hAhalfComplex;
}


void CopyDataB2HD(complex<float> *dstA, cuInt8Complex *srcA, size_t length){
    auto hAhalfComplex = new cuInt8Complex[length];
    CopyDataB2HD(hAhalfComplex, srcA, length);
    for(int i=0; i<length; i++) dstA[i] = complex<float>((float)hAhalfComplex[i].x, (float)hAhalfComplex[i].y);
    delete[] hAhalfComplex;
}

void CopyDataB2HD(cuInt8Complex * dstA, cuComplex & maxinfo, complex<float> *srcA, size_t length){
    auto mx = abs(srcA[0].real());
    auto my = abs(srcA[0].imag());
    for(size_t i=0; i<length; i++) {
        mx = max(mx, abs(srcA[i].real()));
        my = max(my, abs(srcA[i].imag()));
    }
    maxinfo.x = mx;
    maxinfo.y = my;
    if(maxinfo.x < 1e-14) maxinfo.x = 0;
    if(maxinfo.y < 1e-14) maxinfo.y = 0;
    auto hAI8Complex = new cuInt8Complex[length];
    for(size_t i=0; i<length; i++) {
        if(maxinfo.x > 1e-14){
            hAI8Complex[i].x = (int8_t)(srcA[i].real() / maxinfo.x * 125.0);
        }else{
            hAI8Complex[i].x = 0;
            hAI8Complex[i].x = (int8_t)(srcA[i].real() / maxinfo.x * 125.0);
        }
        if(maxinfo.y > 1e-14){
            hAI8Complex[i].y = (int8_t)(srcA[i].imag() / maxinfo.y * 125.0);
        }else{
            hAI8Complex[i].y = 0;
            hAI8Complex[i].y = (int8_t)(srcA[i].imag() / maxinfo.y * 125.0);
        }
    }
    CopyDataB2HD(dstA, hAI8Complex, length);
    delete[] hAI8Complex;
}

void CopyDataB2HD(cuInt8Complex * dstA, cuComplex & maxinfo, cuHalfComplex *srcA, size_t length){
    auto mx = abs((float)srcA[0].x);
    auto my = abs((float)srcA[0].y);
    for(size_t i=0; i<length; i++) {
        mx = max(mx, abs((float)srcA[i].x));
        my = max(my, abs((float)srcA[i].y));
    }
    maxinfo.x = mx;
    maxinfo.y = my;
    if(maxinfo.x < 1e-14) maxinfo.x = 0;
    if(maxinfo.y < 1e-14) maxinfo.y = 0;
    auto hAI8Complex = new cuInt8Complex[length];
    for(size_t i=0; i<length; i++) {
        if(maxinfo.x > 1e-14){
            hAI8Complex[i].x = (int8_t)((float)srcA[i].x / maxinfo.x * 125.0);
        }else{
            hAI8Complex[i].x = 0;
            hAI8Complex[i].x = (int8_t)((float)srcA[i].x / maxinfo.x * 125.0);
        }
        if(maxinfo.y > 1e-14){
            hAI8Complex[i].y = (int8_t)((float)srcA[i].y / maxinfo.y * 125.0);
        }else{
            hAI8Complex[i].y = 0;
            hAI8Complex[i].y = (int8_t)((float)srcA[i].y / maxinfo.y * 125.0);
        }
    }
    CopyDataB2HD(dstA, hAI8Complex, length);
    delete[] hAI8Complex;
}

void CopyDataB2HD(cuComplex *dstA, complex<float> *srcA, size_t length){
    CopyDataB2HD(reinterpret_cast<complex<float>*>(dstA), srcA, length);
}

void CopyDataB2HD(complex<float> *dstA, cuComplex *srcA, size_t length){
    CopyDataB2HD((dstA), reinterpret_cast<complex<float>*>(srcA), length);
}


size_t ceil32(size_t val){
    return (val / 32 + 1) * 32;
}

template<typename T>
size_t TLRMVMBytesProcessed_cuda(size_t granksum, size_t nb, size_t M, size_t N){
    // phase 1
    unsigned long int phase1 = granksum*nb + N + granksum;
    // phase 2
    unsigned long int shuffle = 2 * granksum;
    // phase 3
    unsigned long int phase2 = granksum*nb + granksum + M;
    return sizeof(T) * (phase1 + shuffle + phase2);
}
template size_t TLRMVMBytesProcessed_cuda<cuComplex>(size_t, size_t, size_t, size_t);
template size_t TLRMVMBytesProcessed_cuda<cuHalfComplex>(size_t, size_t, size_t, size_t);


void init_alpha_beta(cuComplex &alpha, cuComplex &beta){
    alpha.x = (float)1.0;
    alpha.y = beta.x = beta.y = (float)0.0;
}

void init_alpha_beta(cuDoubleComplex &alpha, cuDoubleComplex &beta){
    alpha.x = (float)1.0;
    alpha.y = beta.x = beta.y = (double)0.0;
}


