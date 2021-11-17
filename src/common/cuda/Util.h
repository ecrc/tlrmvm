#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <unistd.h>
#include <vector>
#include <cuda.h>
#include <cublas_v2.h>

using namespace std;
namespace cudatlrmat {

template<typename T>
void Fillval(T * dptr, T val, size_t len);


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

 #define CUDACHECK(cmd) do {                         \
   cudaError_t e = cmd;                              \
   if( e != cudaSuccess ) {                          \
     printf("Failed: Cuda error %s:%d '%s'\n",             \
         __FILE__,__LINE__,cudaGetErrorString(e));   \
     exit(EXIT_FAILURE);                             \
   }                                                 \
 } while(0)
 
  #define CUBLASCHECK(cmd) do {                         \
   cublasStatus_t e = cmd;                              \
   if( e != CUBLAS_STATUS_SUCCESS ) {                          \
     printf("Failed: Cublas error %s:%d '%s'\n",             \
         __FILE__,__LINE__, _cudaGetErrorEnum(e));   \
     exit(EXIT_FAILURE);                             \
   }                                                 \
 } while(0)
 
 #define NCCLCHECK(cmd) do {                         \
   ncclResult_t r = cmd;                             \
   if (r!= ncclSuccess) {                            \
     printf("Failed, NCCL error %s:%d '%s'\n",             \
         __FILE__,__LINE__,ncclGetErrorString(r));   \
     exit(EXIT_FAILURE);                             \
   }                                                 \
 } while(0)
 
 
 #define NCCLCHECK(cmd) do {                         \
   ncclResult_t r = cmd;                             \
   if (r!= ncclSuccess) {                            \
     printf("Failed, NCCL error %s:%d '%s'\n",             \
         __FILE__,__LINE__,ncclGetErrorString(r));   \
     exit(EXIT_FAILURE);                             \
   }                                                 \
 } while(0)

// template<typename T>
// void GetHostHalfMemory(T **A, size_t m);

template<typename T>
void GetcuHostMemory(T **A, size_t m);

template<typename T>
void FreecuHostMemory(T *A);

// template<typename T>
// void FreeHostHalfMemory(T *A);

template<typename T>
void GetDeviceMemory(T **A, size_t m);

template<typename T>
void GetDeviceMemory(T **A, T **B, T **C, size_t m, size_t k, size_t n);

template<typename T>
void GetDeviceMemoryBatched(T **A, T **B, T **C, 
T ***Abatchpointer, T ***Bbatchpointer, T ***Cbatchpointer,
vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns);

template<typename T>
void FreeDeviceMemory(T *A, T *x, T *y);

template<typename T>
void FreeDeviceMemory(T *A);

template<typename T>
void FreeDeviceMemoryBatched(T *A, T *B, T *C, 
T **Abatchpointer, T **Bbatchpointer, T **Cbatchpointer);

template<typename T>
void CopyDataB2HD(T *dstA, T *dstB, T *dstC,
T *srcA, T *srcB, T *srcC, vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns);


template<typename T>
void CopyDataB2HD(T *dstA, T *srcA, size_t length);

void CopyDataB2HD(cuComplex *dstA, complex<float> *srcA, size_t length);

void CopyDataB2HD(complex<float> *dstA, cuComplex *srcA, size_t length);

template<typename T>
void CopyDataAsyncB2HD(T *dstA, T *srcA, size_t length, cudaStream_t stream=0);


size_t ceil32(size_t val);

// void CopyDataB2HD(float *dstA, float *dstB, float *dstC,
// float *srcA, float *srcB, float *srcC, vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns);

// void CopyDataB2HD(complex<float> *dstA, complex<float> *dstB, complex<float> *dstC,
// cuComplex *srcA, cuComplex *srcB, cuComplex *srcC, vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns);

// void CopyDataB2HD(cuComplex *dstA, cuComplex *dstB, cuComplex *dstC,
// complex<float> *srcA, complex<float> *srcB, complex<float> *srcC, 
// vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns);


} // cudatlrmat

#endif // CUDA_UTIL_H