#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <unistd.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cuda.h>
#include <cstdint>
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_bf16.h>

using namespace std;

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#endif

#define TESTING_CHECK( err )                                                 \
    do {                                                                     \
        magma_int_t err_ = (err);                                            \
        if ( err_ != 0 ) {                                                   \
            fprintf( stderr, "Error: %s\nfailed at %s:%d: error %lld: %s\n", \
                     #err, __FILE__, __LINE__,                               \
                     (long long) err_, magma_strerror(err_) );               \
            exit(1);                                                         \
        }                                                                    \
    } while( 0 )

    #define CUDACHECK(cmd) do {                                 \
        cudaError_t e = cmd;                                    \
        if( e != cudaSuccess ) {                                \
            printf("Failed: Cuda error %s:%d '%s'\n",           \
            __FILE__,__LINE__,cudaGetErrorString(e));           \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while(0)

    #define CUBLASCHECK(cmd) do {                               \
        cublasStatus_t e = cmd;                                 \
        if( e != CUBLAS_STATUS_SUCCESS ) {                      \
        printf("Failed: Cublas error %s:%d '%s'\n",             \
            __FILE__,__LINE__, _cudaGetErrorEnum(e));           \
            exit(EXIT_FAILURE);                                 \
        }                                                       \
    } while(0)
 
    #define NCCLCHECK(cmd) do {                                 \
        ncclResult_t r = cmd;                                   \
        if (r!= ncclSuccess) {                                  \
        printf("Failed, NCCL error %s:%d '%s'\n",               \
        __FILE__,__LINE__,ncclGetErrorString(r));               \
        exit(EXIT_FAILURE);                                     \
        }                                                       \
    } while(0)

    #define NCCLCHECK(cmd) do {                                 \
        ncclResult_t r = cmd;                                   \
        if (r!= ncclSuccess) {                                  \
         printf("Failed, NCCL error %s:%d '%s'\n",              \
             __FILE__,__LINE__,ncclGetErrorString(r));          \
         exit(EXIT_FAILURE);                                    \
        }                                                       \
    } while(0)

struct cuHalfComplex{
    half x;
    half y;
    __host__ __device__ __forceinline__ cuHalfComplex(float a, float b):x(__float2half(a)),y(__float2half(b)){}
    __host__ __device__ __forceinline__ cuHalfComplex(half a, half b):x(a),y(b){}
    __host__ __device__ __forceinline__ cuHalfComplex():x(0.0),y(0.0){}


#if __CUDA_ARCH__ >= 800
    __device__ __forceinline__ cuHalfComplex operator*(const cuHalfComplex &a) const{
        return {__hsub(__hmul(x,a.x),__hmul(y,a.y)),__hadd(__hmul(x,a.y),__hmul(y,a.x))};
    }
    __device__ __forceinline__ cuHalfComplex operator+(const cuHalfComplex &a) const{
        return {__hadd(x, a.x), __hadd(y,a.y)};
    }
#else
    __device__ __forceinline__ cuHalfComplex operator*(const cuHalfComplex &a) const{
        return {x*a.x - y*a.y, x*a.y + y*a.x};
    }
    __device__ __forceinline__ cuHalfComplex operator+(const cuHalfComplex &a) const{
        return {x + a.x, y + a.y};
    }
#endif
};

struct cubfComplex{
    nv_bfloat16 x;
    nv_bfloat16 y;
    __host__ __device__ __forceinline__ cubfComplex(float a, float b){
        x = __float2bfloat16(a);
        y = __float2bfloat16(b);
    }
    __host__ __device__ __forceinline__ cubfComplex(nv_bfloat16 a, nv_bfloat16 b){
        x = (a);
        y = (b);
    }
    __host__ __device__ __forceinline__ cubfComplex():x(0.0),y(0.0){}
#if __CUDA_ARCH__ >= 800
    __device__ __forceinline__ cubfComplex operator*(const cubfComplex &a) const{
        return {__hsub(__hmul(x,a.x),__hmul(y,a.y)),__hadd(__hmul(x,a.y),__hmul(y,a.x))};
    }
    __device__ __forceinline__ cubfComplex operator+(const cubfComplex &a) const{
        return {__hadd(x, a.x), __hadd(y,a.y)};
    }
#else
    __device__ __forceinline__ cubfComplex operator*(const cubfComplex &a) const{
        return {x*a.x - y*a.y, x*a.y + y*a.x};
    }
    __device__ __forceinline__ cubfComplex operator+(const cubfComplex &a) const{
        return {x + a.x, y + a.y};
    }
#endif
};


struct cuInt8Complex{
    int8_t x;
    int8_t y;
    __host__ __device__ __forceinline__ cuInt8Complex(int8_t a, int8_t b):x(a),y(b){}
    __host__ __device__ __forceinline__ cuInt8Complex():x(0),y(0){}
};


template<typename T>
void GetcuHostMemory(T **A, size_t m);

template<typename T>
void FreecuHostMemory(T *A);

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
void CopyDataB2HD(cuComplex *dstA, complex<float> *srcA, size_t length);
void CopyDataB2HD(complex<float> *dstA, cuComplex *srcA, size_t length);
void CopyDataB2HD(cuHalfComplex * dstA, complex<float> *srcA, size_t length);
void CopyDataB2HD(complex<float> *dstA, cuHalfComplex *srcA, size_t length);
void CopyDataB2HD(complex<float> *dstA, cubfComplex *srcA, size_t length);
void CopyDataB2HD(cubfComplex *dstA,complex<float> * srcA, size_t length);
void CopyDataB2HD(cuInt8Complex * dstA, cuComplex & maxinfo, complex<float> *srcA, size_t length);
void CopyDataB2HD(cuInt8Complex * dstA, cuComplex & maxinfo, cuHalfComplex *srcA, size_t length);
void CopyDataB2HD(complex<float> *dstA, cuInt8Complex *srcA, size_t length);

template<typename T>
void CopyDataAsyncB2HD(T *dstA, T *srcA, size_t length, cudaStream_t stream=0);

size_t ceil32(size_t val);

template<typename T>
size_t TLRMVMBytesProcessed_cuda(size_t granksum, size_t nb, size_t M, size_t N);

void init_alpha_beta(cuComplex &alpha, cuComplex &beta);

void init_alpha_beta(cuDoubleComplex &alpha, cuDoubleComplex &beta);



#endif // CUDA_UTIL_H