//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

//
// Created by Yuxi Hong on 28/02/2022.
//

#ifndef HIP_UTIL_H
#define HIP_UTIL_H

#include <unistd.h>
#include <vector>
#include <complex>
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <hip/hip_fp16.h>
#include <hipblas.h>

using namespace std;

// cuBLAS API errors
static const char *_cudaGetErrorEnum(hipblasStatus_t error)
{
    switch (error)
    {
        case HIPBLAS_STATUS_SUCCESS:
            return "HIPBLAS_STATUS_SUCCESS";

        case HIPBLAS_STATUS_NOT_INITIALIZED:
            return "HIPBLAS_STATUS_NOT_INITIALIZED";

        case HIPBLAS_STATUS_ALLOC_FAILED:
            return "HIPBLAS_STATUS_ALLOC_FAILED";

        case HIPBLAS_STATUS_INVALID_VALUE:
            return "HIPBLAS_STATUS_INVALID_VALUE";

        case HIPBLAS_STATUS_ARCH_MISMATCH:
            return "HIPBLAS_STATUS_ARCH_MISMATCH";

        case HIPBLAS_STATUS_MAPPING_ERROR:
            return "HIPBLAS_STATUS_MAPPING_ERROR";

        case HIPBLAS_STATUS_EXECUTION_FAILED:
            return "HIPBLAS_STATUS_EXECUTION_FAILED";

        case HIPBLAS_STATUS_INTERNAL_ERROR:
            return "HIPBLAS_STATUS_INTERNAL_ERROR";

        case HIPBLAS_STATUS_NOT_SUPPORTED:
            return "HIPBLAS_STATUS_NOT_SUPPORTED";

        case HIPBLAS_STATUS_HANDLE_IS_NULLPTR:
            return "HIPBLAS_STATUS_HANDLE_IS_NULLPTR";

        case HIPBLAS_STATUS_INVALID_ENUM:
            return "HIPBLAS_STATUS_INVALID_ENUM";

        case HIPBLAS_STATUS_UNKNOWN:
            return "HIPBLAS_STATUS_UNKNOWN";
    }

    return "<unknown>";
}

#define HIPCHECK(cmd) do {                                 \
        hipError_t e = cmd;                                    \
        if( e != hipSuccess ) {                                \
            printf("Failed: HIP error %s:%d '%s'\n",           \
            __FILE__,__LINE__,hipGetErrorString(e));           \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while(0)

#define HIPBLASCHECK(cmd) do {                         \
   hipblasStatus_t e = cmd;                              \
   if( e != HIPBLAS_STATUS_SUCCESS ) {                          \
     printf("Failed: Cublas error %s:%d '%s'\n",             \
         __FILE__,__LINE__, _cudaGetErrorEnum(e));   \
     exit(EXIT_FAILURE);                             \
   }                                                 \
 } while(0)

struct hipHalfComplex{
    half x;
    half y;
    __host__ __device__ __forceinline__ hipHalfComplex(float a, float b):x(__float2half(a)),y(__float2half(b)){}
    __host__ __device__ __forceinline__ hipHalfComplex(half a, half b):x(a),y(b){}
    __host__ __device__ __forceinline__ hipHalfComplex():x(0.0),y(0.0){}


    __device__ __forceinline__ hipHalfComplex operator*(const hipHalfComplex &a) const{
        return {x*a.x - y*a.y, x*a.y + y*a.x};
    }
    __device__ __forceinline__ hipHalfComplex operator+(const hipHalfComplex &a) const{
        return {x + a.x, y + a.y};
    }
};


struct hipInt8Complex{
    int8_t x;
    int8_t y;
    __host__ __device__ __forceinline__ hipInt8Complex(int8_t a, int8_t b):x(a),y(b){}
    __host__ __device__ __forceinline__ hipInt8Complex():x(0),y(0){}
};


template<typename T>
void GethipHostMemory(T **A, size_t m);

template<typename T>
void FreehipHostMemory(T *A);

template<typename T>
void GetDeviceMemory(T **A, size_t m);

template<typename T>
void GetDeviceMemory(T **A, T **B, T **C, size_t m, size_t k, size_t n);

template<typename T>
void FreeDeviceMemory(T *A, T *x, T *y);

template<typename T>
void FreeDeviceMemory(T *A);

template<typename T>
void CopyDataB2HD(T *dstA, T *dstB, T *dstC,
T *srcA, T *srcB, T *srcC, vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns);
template<typename T>
void CopyDataB2HD(T *dstA, T *srcA, size_t length);
void CopyDataB2HD(hipComplex *dstA, complex<float> *srcA, size_t length);
void CopyDataB2HD(complex<float> *dstA, hipComplex *srcA, size_t length);
void CopyDataB2HD(hipHalfComplex * dstA, complex<float> *srcA, size_t length);
void CopyDataB2HD(complex<float> *dstA, hipHalfComplex *srcA, size_t length);
//void CopyDataB2HD(complex<float> *dstA, cubfComplex *srcA, size_t length);
//void CopyDataB2HD(cubfComplex *dstA,complex<float> * srcA, size_t length);
void CopyDataB2HD(hipInt8Complex * dstA, hipComplex & maxinfo, complex<float> *srcA, size_t length);
void CopyDataB2HD(hipInt8Complex * dstA, hipComplex & maxinfo, hipHalfComplex *srcA, size_t length);
void CopyDataB2HD(complex<float> *dstA, hipInt8Complex *srcA, size_t length);

template<typename T>
void CopyDataAsyncB2HD(T *dstA, T *srcA, size_t length, hipStream_t stream=0);


void init_alpha_beta(hipComplex &alpha, hipComplex &beta);

void init_alpha_beta(hipDoubleComplex &alpha, hipDoubleComplex &beta);


#endif //HIP_UTIL_H
