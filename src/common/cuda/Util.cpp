#include "common/Common.h"
#include "Util.h"
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h> // if you need CUBLAS v2, include before magma.h

namespace cudatlrmat
{


template<typename T>
void Fillval(T * dptr, T val, size_t len){
    for(size_t i=0; i<len; i++) dptr[i] = val;
}

template void Fillval(float *, float, size_t);
template void Fillval(half *, half, size_t);
template void Fillval(int8_t *, int8_t, size_t);
template void Fillval(complex<float> *, complex<float>, size_t);


// template<typename T>
// void GetHostHalfMemory(T **A, size_t m){
//     size_t elesize = sizeof(T);
//     CUDACHECK(cudaMallocHost(A, m * elesize));
// }
// template void GetHostHalfMemory<half>(half **, size_t);
// template void GetHostHalfMemory<half*>(half* **, size_t);
// template void GetHostHalfMemory<half2>(half2 **, size_t);
// template void GetHostHalfMemory<half2*>(half2* **, size_t);

template<typename T>
void GetcuHostMemory(T **A, size_t m){
    size_t elesize = sizeof(T);
    CUDACHECK(cudaMallocHost(A, elesize * m));
}
template void GetcuHostMemory<half>(half **, size_t);
template void GetcuHostMemory<float>(float **, size_t);
template void GetcuHostMemory<double>(double **, size_t);
template void GetcuHostMemory<half*>(half* **, size_t);
template void GetcuHostMemory<float*>(float* **, size_t);
template void GetcuHostMemory<double*>(double* **, size_t);
template void GetcuHostMemory<cuComplex>(cuComplex **, size_t);
template void GetcuHostMemory<cuComplex*>(cuComplex* **, size_t);
template void GetcuHostMemory<cuDoubleComplex>(cuDoubleComplex **, size_t);
template void GetcuHostMemory<cuDoubleComplex*>(cuDoubleComplex* **, size_t);


template<typename T>
void FreecuHostMemory(T *A){
    CUDACHECK(cudaFreeHost(A));
}
template void FreecuHostMemory<half>(half *);
template void FreecuHostMemory<float>(float *);
template void FreecuHostMemory<double>(double *);
template void FreecuHostMemory<half*>(half* *);
template void FreecuHostMemory<float*>(float* *);
template void FreecuHostMemory<double*>(double* *);
template void FreecuHostMemory<cuComplex>(cuComplex *);
template void FreecuHostMemory<cuComplex*>(cuComplex* *);
template void FreecuHostMemory<cuDoubleComplex>(cuDoubleComplex *);
template void FreecuHostMemory<cuDoubleComplex*>(cuDoubleComplex* *);

// template<typename T>
// void FreeHostHalfMemory(T *A){
//     CUDACHECK(cudaFreeHost(A));
// }
// template void FreeHostHalfMemory<half>(half *);
// template void FreeHostHalfMemory<half*>(half* *);
// template void FreeHostHalfMemory<half2>(half2 *);
// template void FreeHostHalfMemory<half2*>(half2* *);

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
template void GetDeviceMemory<cuDoubleComplex>(cuDoubleComplex**, size_t);
template void GetDeviceMemory<cuDoubleComplex*>(cuDoubleComplex* **, size_t);
template void GetDeviceMemory<cuDoubleComplex**>(cuDoubleComplex** **, size_t);
template void GetDeviceMemory<half>(half**, size_t);
template void GetDeviceMemory<half*>(half***, size_t);
template void GetDeviceMemory<half**>(half****, size_t);
template void GetDeviceMemory<half2>(half2**, size_t);


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


void CalculateBatchedMemoryLayout(cuComplex *A, cuComplex *B, cuComplex *C,
cuComplex **Abatchpointer, cuComplex **Bbatchpointer, cuComplex **Cbatchpointer, 
vector<size_t> Ms,vector<size_t> Ks,vector<size_t> Ns){
    complex<float> *hA, *hB, *hC;
    complex<float> **hAbp, **hBbp, **hCbp;
    size_t batchsize = Ms.size();
    hA = (complex<float>*)A; 
    hB = (complex<float>*)B; 
    hC = (complex<float>*)C;
    hAbp = new complex<float>*[batchsize];
    hBbp = new complex<float>*[batchsize];
    hCbp = new complex<float>*[batchsize];
    tlrmat::CalculateBatchedMemoryLayout(hA, hB, hC, hAbp, hBbp, hCbp, Ms, Ks, Ns);
    CUDACHECK(cudaMemcpy(Abatchpointer, hAbp, batchsize * sizeof(cuComplex*), cudaMemcpyDefault));
    CUDACHECK(cudaMemcpy(Bbatchpointer, hBbp, batchsize * sizeof(cuComplex*), cudaMemcpyDefault));
    CUDACHECK(cudaMemcpy(Cbatchpointer, hCbp, batchsize * sizeof(cuComplex*), cudaMemcpyDefault));
    delete[] hAbp;
    delete[] hBbp;
    delete[] hCbp;
}

void CalculateBatchedMemoryLayout(float *A, float *B, float *C,
float **Abatchpointer, float **Bbatchpointer, float **Cbatchpointer, 
vector<size_t> Ms,vector<size_t> Ks,vector<size_t> Ns){
    float *hA, *hB, *hC;
    float **hAbp, **hBbp, **hCbp;
    size_t batchsize = Ms.size();
    hA = (float*)A; 
    hB = (float*)B; 
    hC = (float*)C;
    hAbp = new float*[batchsize];
    hBbp = new float*[batchsize];
    hCbp = new float*[batchsize];
    tlrmat::CalculateBatchedMemoryLayout(hA, hB, hC, hAbp, hBbp, hCbp, Ms, Ks, Ns);
    CUDACHECK(cudaMemcpy(Abatchpointer, hAbp, batchsize * sizeof(float*), cudaMemcpyDefault));
    CUDACHECK(cudaMemcpy(Bbatchpointer, hBbp, batchsize * sizeof(float*), cudaMemcpyDefault));
    CUDACHECK(cudaMemcpy(Cbatchpointer, hCbp, batchsize * sizeof(float*), cudaMemcpyDefault));
    delete[] hAbp;
    delete[] hBbp;
    delete[] hCbp;
}

void CalculateBatchedMemoryLayout(half *A, half *B, half *C,
half **Abatchpointer, half **Bbatchpointer, half **Cbatchpointer, 
vector<size_t> Ms,vector<size_t> Ks,vector<size_t> Ns){
    half **hAbp, **hBbp, **hCbp;
    size_t batchsize = Ms.size();
    hAbp = new half*[batchsize];
    hBbp = new half*[batchsize];
    hCbp = new half*[batchsize];
    hAbp[0] = A;
    hBbp[0] = B;
    hCbp[0] = C;
    for(int i=1; i<Ms.size(); i++){
        hAbp[i] = hAbp[i-1] + Ms[i-1] * Ks[i-1];
        hBbp[i] = hBbp[i-1] + Ks[i-1] * Ns[i-1];
        hCbp[i] = hCbp[i-1] + Ms[i-1] * Ns[i-1];
    }
    CUDACHECK(cudaMemcpy(Abatchpointer, hAbp, batchsize * sizeof(half*), cudaMemcpyDefault));
    CUDACHECK(cudaMemcpy(Bbatchpointer, hBbp, batchsize * sizeof(half*), cudaMemcpyDefault));
    CUDACHECK(cudaMemcpy(Cbatchpointer, hCbp, batchsize * sizeof(half*), cudaMemcpyDefault));
    delete[] hAbp;
    delete[] hBbp;
    delete[] hCbp;
}

void CalculateBatchedMemoryLayout(int8_t *A, int8_t *B, int8_t *C,
int8_t **Abatchpointer, int8_t **Bbatchpointer, int8_t **Cbatchpointer,
vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns){
    int8_t **hAbp, **hBbp, **hCbp;
    size_t batchsize = Ms.size();
    hAbp = new int8_t*[batchsize];
    hBbp = new int8_t*[batchsize];
    hCbp = new int8_t*[batchsize];
    Abatchpointer[0] = A;
    Bbatchpointer[0] = B;
    Cbatchpointer[0] = C;
    for(int i=1; i<Ms.size(); i++){
        hAbp[i] = hAbp[i-1] + Ms[i-1] * Ks[i-1];
        hBbp[i] = hBbp[i-1] + Ks[i-1] * Ns[i-1];
        hCbp[i] = hCbp[i-1] + Ms[i-1] * Ns[i-1];
    }
    CUDACHECK(cudaMemcpy(Abatchpointer, hAbp, batchsize * sizeof(int8_t*), cudaMemcpyDefault));
    CUDACHECK(cudaMemcpy(Bbatchpointer, hBbp, batchsize * sizeof(int8_t*), cudaMemcpyDefault));
    CUDACHECK(cudaMemcpy(Cbatchpointer, hCbp, batchsize * sizeof(int8_t*), cudaMemcpyDefault));
    delete[] hAbp;
    delete[] hBbp;
    delete[] hCbp;
}

template<typename T>
void GetDeviceMemoryBatched(T **A, T **B, T **C, 
T ***Abatchpointer, T ***Bbatchpointer, T ***Cbatchpointer,
vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns){
    T *tmpA, *tmpB, *tmpC;
    T **tmpAbp, **tmpBbp, **tmpCbp;
    size_t elemsize = sizeof(T);
    size_t Atotalelems, Btotalelems, Ctotalelems;
    tlrmat::CaluclateTotalElements(Ms, Ks, Ns, Atotalelems, Btotalelems, Ctotalelems);
    // allocate memory space for A
    CUDACHECK(cudaMalloc(&tmpA, Atotalelems * elemsize));
    // allocate memory space for B
    CUDACHECK(cudaMalloc(&tmpB, Btotalelems * elemsize));
    // allocate memory space for C
    CUDACHECK(cudaMalloc(&tmpC, Ctotalelems * elemsize));
    assert(Ms.size() == Ks.size() && Ks.size() == Ns.size());
    size_t batchsize = Ms.size();
    CUDACHECK(cudaMalloc(&tmpAbp, batchsize * sizeof(T*)));
    CUDACHECK(cudaMalloc(&tmpBbp, batchsize * sizeof(T*)));
    CUDACHECK(cudaMalloc(&tmpCbp, batchsize * sizeof(T*)));
    CalculateBatchedMemoryLayout(tmpA, tmpB, tmpC, tmpAbp, tmpBbp, tmpCbp, Ms, Ks, Ns);
    A[0] = tmpA;
    B[0] = tmpB;
    C[0] = tmpC;
    Abatchpointer[0] = tmpAbp;
    Bbatchpointer[0] = tmpBbp;
    Cbatchpointer[0] = tmpCbp;
}

template void GetDeviceMemoryBatched(float **A, float **B, float **C, 
float ***Abatchpointer, float ***Bbatchpointer, float ***Cbatchpointer,
vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns);

template void GetDeviceMemoryBatched(cuComplex **A, cuComplex **B, cuComplex **C, 
cuComplex ***Abatchpointer, cuComplex ***Bbatchpointer, cuComplex ***Cbatchpointer,
vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns);

template void GetDeviceMemoryBatched(half **A, half **B, half **C, 
half ***Abatchpointer, half ***Bbatchpointer, half ***Cbatchpointer,
vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns);

template void GetDeviceMemoryBatched(int8_t **A, int8_t **B, int8_t **C, 
int8_t ***Abatchpointer, int8_t ***Bbatchpointer, int8_t ***Cbatchpointer,
vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns);

template<typename T>
void FreeDeviceMemory(T *A){
    CUDACHECK(cudaFree(A));
}
template void FreeDeviceMemory(unsigned long int *A);
template void FreeDeviceMemory(int *A);
template void FreeDeviceMemory(int8_t *A);
template void FreeDeviceMemory(float *A);
template void FreeDeviceMemory(double *A);
template void FreeDeviceMemory(cuDoubleComplex *A);
template void FreeDeviceMemory(cuComplex *A);
template void FreeDeviceMemory(half *A);
template void FreeDeviceMemory(half2 *A);

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
    tlrmat::CaluclateTotalElements(Ms, Ks, Ns, Atotalelems, Btotalelems, Ctotalelems);
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
template void CopyDataB2HD<complex<double>>(complex<double> *dstA, complex<double> *srcA, size_t length);
template void CopyDataB2HD<half>(half *dstA, half *srcA, size_t length);
template void CopyDataB2HD<half*>(half* *dstA, half* *srcA, size_t length);

void CopyDataB2HD(cuComplex *dstA, complex<float> *srcA, size_t length){
    CopyDataB2HD(reinterpret_cast<complex<float>*>(dstA), srcA, length);
}

void CopyDataB2HD(complex<float> *dstA, cuComplex *srcA, size_t length){
    CopyDataB2HD((dstA), reinterpret_cast<complex<float>*>(srcA), length);
}


template<typename T>
void CopyDataAsyncB2HD(T *dstA, T *srcA, size_t length, cudaStream_t stream){
    CUDACHECK( cudaMemcpyAsync((void*)dstA, (void*)srcA, length * sizeof(T), cudaMemcpyDefault, stream) );
}

template void CopyDataAsyncB2HD<int8_t>(int8_t *dstA, int8_t *srcA, size_t length,cudaStream_t stream);
template void CopyDataAsyncB2HD<int>(int *dstA, int *srcA, size_t length,cudaStream_t stream);
template void CopyDataAsyncB2HD<unsigned long int>(unsigned long int *dstA, unsigned long int *srcA, size_t length,cudaStream_t stream);
template void CopyDataAsyncB2HD<float>(float *dstA, float *srcA, size_t length,cudaStream_t stream);
template void CopyDataAsyncB2HD<float*>(float* *dstA, float* *srcA, size_t length,cudaStream_t stream);
template void CopyDataAsyncB2HD<complex<float>>(complex<float> *dstA, complex<float> *srcA, size_t length,cudaStream_t stream);
template void CopyDataAsyncB2HD<complex<float>*>(complex<float> **dstA, complex<float> **srcA, size_t length,cudaStream_t stream);
template void CopyDataAsyncB2HD<double>(double *dstA, double *srcA, size_t length,cudaStream_t stream);
template void CopyDataAsyncB2HD<complex<double>>(complex<double> *dstA, complex<double> *srcA, size_t length,cudaStream_t stream);
template void CopyDataAsyncB2HD<half>(half *dstA, half *srcA, size_t length,cudaStream_t stream);
template void CopyDataAsyncB2HD<half*>(half* *dstA, half* *srcA, size_t length,cudaStream_t stream);



size_t ceil32(size_t val){
    return (val / 32 + 1) * 32;
}

// void CopyDataB2HD(float *dstA, float *dstB, float *dstC,
// float *srcA, float *srcB, float *srcC, vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns){
//     size_t Atotalelems, Btotalelems, Ctotalelems;
//     tlrmat::CaluclateTotalElements(Ms, Ks, Ns, Atotalelems, Btotalelems, Ctotalelems);
//     cout << Atotalelems << ", " << Btotalelems << ", " << Ctotalelems << endl;
//     CUDACHECK( cudaMemcpy(dstA, srcA, Atotalelems * sizeof(float), cudaMemcpyDefault) );
//     CUDACHECK( cudaMemcpy(dstA, srcA, Btotalelems * sizeof(float), cudaMemcpyDefault) );
//     CUDACHECK( cudaMemcpy(dstA, srcA, Ctotalelems * sizeof(float), cudaMemcpyDefault) );
// }

// void CopyDataB2HD(complex<float> *dstA, complex<float> *dstB, complex<float> *dstC,
// cuComplex *srcA, cuComplex *srcB, cuComplex *srcC, 
// vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns){
//     size_t Atotalelems, Btotalelems, Ctotalelems;
//     tlrmat::CaluclateTotalElements(Ms, Ks, Ns, Atotalelems, Btotalelems, Ctotalelems);
//     CUDACHECK( cudaMemcpy(dstA, srcA, Atotalelems * sizeof(cuComplex), cudaMemcpyDefault) );
//     CUDACHECK( cudaMemcpy(dstA, srcA, Btotalelems * sizeof(cuComplex), cudaMemcpyDefault) );
//     CUDACHECK( cudaMemcpy(dstA, srcA, Ctotalelems * sizeof(cuComplex), cudaMemcpyDefault) );
// }

// void CopyDataB2HD(cuComplex *dstA, cuComplex *dstB, cuComplex *dstC,
// complex<float> *srcA, complex<float> *srcB, complex<float> *srcC, 
// vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns){
//     size_t Atotalelems, Btotalelems, Ctotalelems;
//     tlrmat::CaluclateTotalElements(Ms, Ks, Ns, Atotalelems, Btotalelems, Ctotalelems);
//     CUDACHECK( cudaMemcpy(dstA, srcA, Atotalelems * sizeof(cuComplex), cudaMemcpyDefault) );
//     CUDACHECK( cudaMemcpy(dstA, srcA, Btotalelems * sizeof(cuComplex), cudaMemcpyDefault) );
//     CUDACHECK( cudaMemcpy(dstA, srcA, Ctotalelems * sizeof(cuComplex), cudaMemcpyDefault) );
// }

} // namespace cudatlrmat
