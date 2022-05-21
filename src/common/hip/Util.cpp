//
// Created by Yuxi Hong on 28/02/2022.
//
#include "../cpu/Util.hpp"
#include "Util.hpp"

template<typename T>
void GethipHostMemory(T **A, size_t m){
    size_t elesize = sizeof(T);
    T * tmpA = new T[elesize * m];
    A[0] = tmpA;
}
//template void GethipHostMemory<half>(half **, size_t);
template void GethipHostMemory<float>(float **, size_t);
template void GethipHostMemory<double>(double **, size_t);
//template void GethipHostMemory<half*>(half* **, size_t);
template void GethipHostMemory<float*>(float* **, size_t);
template void GethipHostMemory<double*>(double* **, size_t);
template void GethipHostMemory<hipComplex>(hipComplex **, size_t);
template void GethipHostMemory<hipComplex*>(hipComplex* **, size_t);
template void GethipHostMemory<hipDoubleComplex>(hipDoubleComplex **, size_t);
//template void GethipHostMemory<cudatlrmvm::cuHalfComplex>(cudatlrmvm::cuHalfComplex **, size_t);
template void GethipHostMemory<hipDoubleComplex*>(hipDoubleComplex* **, size_t);
//template void GethipHostMemory<cudatlrmvm::cuHalfComplex*>(cudatlrmvm::cuHalfComplex* **, size_t);

template<typename T>
void FreehipHostMemory(T *A){
    delete[] A;
}
template void FreehipHostMemory<half>(half *);
template void FreehipHostMemory<float>(float *);
template void FreehipHostMemory<double>(double *);
//template void FreehipHostMemory<half*>(half* *);
template void FreehipHostMemory<float*>(float* *);
template void FreehipHostMemory<double*>(double* *);
template void FreehipHostMemory<hipComplex>(hipComplex *);
template void FreehipHostMemory<hipComplex*>(hipComplex* *);
template void FreehipHostMemory<hipDoubleComplex>(hipDoubleComplex *);
template void FreehipHostMemory<hipDoubleComplex*>(hipDoubleComplex* *);



template<typename T>
void GetDeviceMemory(T **A, size_t m){
    size_t elesize = sizeof(T);
    HIPCHECK(hipMalloc(&A[0], m * elesize));
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
template void GetDeviceMemory<std::complex<float>>(std::complex<float>**, size_t);
template void GetDeviceMemory<std::complex<float>*>(std::complex<float>***, size_t);
template void GetDeviceMemory<std::complex<float>**>(std::complex<float>****, size_t);
template void GetDeviceMemory<hipComplex>(hipComplex**, size_t);
template void GetDeviceMemory<hipComplex*>(hipComplex* **, size_t);
template void GetDeviceMemory<hipComplex**>(hipComplex** **, size_t);
template void GetDeviceMemory<hipDoubleComplex>(hipDoubleComplex**, size_t);
template void GetDeviceMemory<hipDoubleComplex*>(hipDoubleComplex* **, size_t);
template void GetDeviceMemory<hipDoubleComplex**>(hipDoubleComplex** **, size_t);
template void GetDeviceMemory<half>(half**, size_t);
template void GetDeviceMemory<half*>(half***, size_t);
template void GetDeviceMemory<half**>(half****, size_t);
template void GetDeviceMemory<half2>(half2**, size_t);
//template void GetDeviceMemory<cudatlrmvm::cuHalfComplex*>(cudatlrmvm::cuHalfComplex***, size_t);
//template void GetDeviceMemory<cudatlrmvm::cuHalfComplex>(cudatlrmvm::cuHalfComplex**, size_t);


template<typename T>
void GetDeviceMemory(T **A, T **B, T **C, size_t m, size_t k, size_t n){
    size_t elesize = sizeof(T);
    HIPCHECK(hipMalloc(&A[0], m * k * elesize));
    HIPCHECK(hipMalloc(&B[0], k * n * elesize));
    HIPCHECK(hipMalloc(&C[0], m * n * elesize));
}

template void GetDeviceMemory<int8_t>(int8_t**, int8_t**, int8_t**,
                                      size_t, size_t, size_t);
template void GetDeviceMemory<float>(float**, float**, float**,
                                     size_t, size_t, size_t);
template void GetDeviceMemory<hipComplex>(hipComplex**, hipComplex**, hipComplex**,
                                         size_t, size_t, size_t);


template<typename T>
void FreeDeviceMemory(T *A){
    HIPCHECK(hipFree(A));
}
template void FreeDeviceMemory(unsigned long int *A);
template void FreeDeviceMemory(int *A);
template void FreeDeviceMemory(int8_t *A);
template void FreeDeviceMemory(float *A);
template void FreeDeviceMemory(float **A);
template void FreeDeviceMemory(double *A);
template void FreeDeviceMemory(double **A);
template void FreeDeviceMemory(hipDoubleComplex *A);
template void FreeDeviceMemory(hipDoubleComplex **A);
template void FreeDeviceMemory(hipComplex *A);
template void FreeDeviceMemory(hipComplex **A);
template void FreeDeviceMemory(half *A);
template void FreeDeviceMemory(half2 *A);

template<typename T>
void FreeDeviceMemory(T *A, T *B, T *C){
    HIPCHECK(hipFree(A));
    HIPCHECK(hipFree(B));
    HIPCHECK(hipFree(C));
}

template void FreeDeviceMemory(float *A, float *B, float *C);
template void FreeDeviceMemory(hipComplex *A, hipComplex *B, hipComplex *C);

template<typename T>
void FreeDeviceMemoryBatched(T *A, T *B, T *C,
                             T **Abatchpointer, T **Bbatchpointer, T **Cbatchpointer){
    FreeDeviceMemory(A,B,C);
    HIPCHECK(hipFree(Abatchpointer));
    HIPCHECK(hipFree(Bbatchpointer));
    HIPCHECK(hipFree(Cbatchpointer));
}
template void FreeDeviceMemoryBatched(float *A, float *B, float *C,
                                      float **Abatchpointer, float **Bbatchpointer, float **Cbatchpointer);
template void FreeDeviceMemoryBatched(hipComplex *A, hipComplex *B, hipComplex *C,
                                      hipComplex **Abatchpointer, hipComplex **Bbatchpointer, hipComplex **Cbatchpointer);


template<typename T>
void CopyDataB2HD(T *dstA, T *dstB, T *dstC,
                  T *srcA, T *srcB, T *srcC, std::vector<size_t> Ms, std::vector<size_t> Ks, std::vector<size_t> Ns){
    size_t Atotalelems, Btotalelems, Ctotalelems;
    CaluclateTotalElements(Ms, Ks, Ns, Atotalelems, Btotalelems, Ctotalelems);
    HIPCHECK( hipMemcpy((void*)dstA, (void*)srcA, Atotalelems * sizeof(T), hipMemcpyDefault) );
    HIPCHECK( hipMemcpy((void*)dstB, (void*)srcB, Btotalelems * sizeof(T), hipMemcpyDefault) );
    HIPCHECK( hipMemcpy((void*)dstC, (void*)srcC, Ctotalelems * sizeof(T), hipMemcpyDefault) );
}

template void CopyDataB2HD(float *dstA, float *dstB, float *dstC,
                           float *srcA, float *srcB, float *srcC, std::vector<size_t> Ms, std::vector<size_t> Ks, std::vector<size_t> Ns);
template void CopyDataB2HD(std::complex<float> *dstA, std::complex<float> *dstB, std::complex<float> *dstC,
                           std::complex<float> *srcA, std::complex<float> *srcB, std::complex<float> *srcC, std::vector<size_t> Ms, std::vector<size_t> Ks, std::vector<size_t> Ns);
template void CopyDataB2HD(int8_t *dstA, int8_t *dstB, int8_t *dstC,
                           int8_t *srcA, int8_t *srcB, int8_t *srcC, std::vector<size_t> Ms, std::vector<size_t> Ks, std::vector<size_t> Ns);


template<typename T>
void CopyDataB2HD(T *dstA, T *srcA, size_t length){
    HIPCHECK( hipMemcpy((void*)dstA, (void*)srcA, length * sizeof(T), hipMemcpyDefault) );
}

template void CopyDataB2HD<int8_t>(int8_t *dstA, int8_t *srcA, size_t length);
template void CopyDataB2HD<int8_t*>(int8_t* *dstA, int8_t* *srcA, size_t length);
template void CopyDataB2HD<int>(int *dstA, int *srcA, size_t length);
template void CopyDataB2HD<unsigned long int>(unsigned long int *dstA, unsigned long int *srcA, size_t length);
template void CopyDataB2HD<float>(float *dstA, float *srcA, size_t length);
template void CopyDataB2HD<float*>(float* *dstA, float* *srcA, size_t length);
template void CopyDataB2HD<std::complex<float>>(std::complex<float> *dstA, std::complex<float> *srcA, size_t length);
template void CopyDataB2HD<std::complex<float>*>(std::complex<float> **dstA, std::complex<float> **srcA, size_t length);
template void CopyDataB2HD<double>(double *dstA, double *srcA, size_t length);
template void CopyDataB2HD<double*>(double **dstA, double **srcA, size_t length);
template void CopyDataB2HD<std::complex<double>>(std::complex<double> *dstA, std::complex<double> *srcA, size_t length);
template void CopyDataB2HD<std::complex<double>*>(std::complex<double> **dstA, std::complex<double> **srcA, size_t length);
template void CopyDataB2HD<half>(half *dstA, half *srcA, size_t length);
template void CopyDataB2HD<half*>(half* *dstA, half* *srcA, size_t length);
//template void CopyDataB2HD<cudatlrmvm::cuHalfComplex>(cudatlrmvm::cuHalfComplex *dstA,
//                                                      cudatlrmvm::cuHalfComplex *srcA, size_t length);

void CopyDataB2HD(hipComplex *dstA, std::complex<float> *srcA, size_t length){
    CopyDataB2HD(reinterpret_cast<std::complex<float>*>(dstA), srcA, length);
}

void CopyDataB2HD(std::complex<float> *dstA, hipComplex *srcA, size_t length){
CopyDataB2HD((dstA), reinterpret_cast<std::complex<float>*>(srcA), length);
}


template<typename T>
void CopyDataAsyncB2HD(T *dstA, T *srcA, size_t length, hipStream_t stream){
    HIPCHECK( hipMemcpyAsync((void*)dstA, (void*)srcA, length * sizeof(T), hipMemcpyDefault, stream) );
}

template void CopyDataAsyncB2HD<int8_t>(int8_t *dstA, int8_t *srcA, size_t length,hipStream_t stream);
template void CopyDataAsyncB2HD<int>(int *dstA, int *srcA, size_t length,hipStream_t stream);
template void CopyDataAsyncB2HD<unsigned long int>(unsigned long int *dstA, unsigned long int *srcA, size_t length,hipStream_t stream);
template void CopyDataAsyncB2HD<float>(float *dstA, float *srcA, size_t length,hipStream_t stream);
template void CopyDataAsyncB2HD<float*>(float* *dstA, float* *srcA, size_t length,hipStream_t stream);
template void CopyDataAsyncB2HD<std::complex<float>>(std::complex<float> *dstA, std::complex<float> *srcA, size_t length,hipStream_t stream);
template void CopyDataAsyncB2HD<std::complex<float>*>(std::complex<float> **dstA, std::complex<float> **srcA, size_t length,hipStream_t stream);
template void CopyDataAsyncB2HD<double>(double *dstA, double *srcA, size_t length,hipStream_t stream);
template void CopyDataAsyncB2HD<std::complex<double>>(std::complex<double> *dstA, std::complex<double> *srcA, size_t length,hipStream_t stream);
template void CopyDataAsyncB2HD<half>(half *dstA, half *srcA, size_t length,hipStream_t stream);
template void CopyDataAsyncB2HD<half*>(half* *dstA, half* *srcA, size_t length,hipStream_t stream);

void init_alpha_beta(hipComplex &alpha, hipComplex &beta){
    alpha.x = (float)1.0;
    alpha.y = beta.x = beta.y = (float)0.0;
}

void init_alpha_beta(hipDoubleComplex &alpha, hipDoubleComplex &beta){
    alpha.x = (float)1.0;
    alpha.y = beta.x = beta.y = (double)0.0;
}
