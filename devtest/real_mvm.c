/**
 * @copyright (c) 2020- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <memory.h>

#include "util.h"

#ifdef USE_INTEL
#include <mkl.h>
#endif 

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

#ifdef USE_NEC
#include <cblas.h>
#endif

#ifdef USE_AMD_BLIS
#include <blis.h>
#endif 

#ifdef USE_AMD_OPENBLAS
#include <cblas.h>
#endif

#ifdef USE_HIP

#endif


#ifdef USE_NVIDIA

void checkcudaerror(cudaError_t st){
    if(st != cudaSuccess){
        printf("cuda status failed. \n");
        exit(0);
    }
}
void checkcudastatus(cublasStatus_t st){
    if(st != CUBLAS_STATUS_SUCCESS){
        printf("cuda status failed. \n");
        exit(0);
    }
}
#endif 

int main(int argc, const char* argv[])
{
    /**
     * @brief read input and init pointers
     * 
     */

    real_t *A, *y, *ynaive, *x;
    int m, n, nruns;
    int warmup = 100;
    real_t alpha, beta;
    alpha = 1.0;
    beta = 0.0;
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    nruns = atoi(argv[3]);
    printf("Dense MVM Info: \n");
    printf("1) m : %d n: %d nruns: %d\n\n", m, n, nruns);

/**
 * @brief 
 * allocate memory
 */

#ifdef USE_INTEL
    A = (real_t *)mkl_malloc( (long)m*(long)n*sizeof( real_t ), 128 );
    x = (real_t *)mkl_malloc( (long)n*sizeof( real_t ), 128 );
    y = (real_t *)mkl_malloc( (long)m*sizeof( real_t ), 128 );
    ynaive = (real_t *)mkl_malloc( (long)m*sizeof( real_t ), 128 );
#endif 


#if defined(USE_NEC) || defined(USE_AMD)
    A = (real_t *)aligned_alloc(128, (long)m*(long)n*sizeof( real_t ));
    x = (real_t *)aligned_alloc(128, (long)n*sizeof( real_t ));
    y = (real_t *)aligned_alloc(128, (long)m*sizeof( real_t ));
    ynaive = (real_t *)aligned_alloc(128, (long)m*sizeof( real_t ));
#endif 

#ifdef USE_NVIDIA 
    cudaSetDevice(0);
    // malloc host 
    A = (real_t *)malloc( (long)m*(long)n*sizeof( real_t ));
    x = (real_t *)malloc( (long)n*sizeof( real_t ));
    y = (real_t *)malloc( (long)m*sizeof( real_t ));
    ynaive = (real_t *)malloc( (long)m*sizeof( real_t ));
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    real_t *d_A, *d_y, *d_x;
    // malloc device
    cudaStat = cudaMalloc ((void**)&d_A, (long)m*(long)n*sizeof(real_t));
    checkcudaerror(cudaStat);
    cudaStat = cudaMalloc ((void**)&d_x, (long)n*sizeof(real_t));
    checkcudaerror(cudaStat);
    cudaStat = cudaMalloc ((void**)&d_y, (long)m*sizeof(real_t));
    checkcudaerror(cudaStat);
    stat = cublasCreate(&handle);
    checkcudastatus(stat);
#endif 

    Initdata(A, x, y, m, n);
    char * info = malloc(sizeof(char) * 100);
    sprintf(info, "DenseMVM");
    double st, et;
/**
 * @brief mvm various interface.
 */
#if defined(USE_INTEL) || defined(USE_NEC) || defined(USE_AMD)
for(int nr=0; nr<nruns+warmup; nr++){
    st = gettime();
#ifdef USE_DOUBLE
    cblas_dgemv(CblasColMajor, CblasNoTrans, m, n, alpha, A, m, x, 1, beta, y, 1);
#else 
    cblas_sgemv(CblasColMajor, CblasNoTrans, m, n, alpha, A, m, x, 1, beta, y, 1);
#endif 
    et = gettime();
    if(nr < warmup) continue;
    perfoutput(et-st, m, n, info, A);
}
#endif 

#ifdef USE_NVIDIA
    cudaMemcpy(d_A, A, (long)m*(long)n * sizeof(real_t), cudaMemcpyDefault);
    cudaMemcpy(d_x, x, (long)n * sizeof(real_t), cudaMemcpyDefault);
    cudaMemcpy(d_y, y, (long)m * sizeof(real_t), cudaMemcpyDefault);

for(int nr=0; nr<nruns+warmup; nr++){
    cudaDeviceSynchronize();
    st = gettime();
#ifdef USE_DOUBLE
    cublasDgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_A, m, d_x, 1, &beta, d_y, 1);
#else 
    cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_A, m, d_x, 1, &beta, d_y, 1);
#endif 
    cudaDeviceSynchronize();
    et = gettime();
    if(nr < warmup) continue;
    perfoutput(et-st, m, n, info, A);
}
#endif

    free(info);
/**
 * @brief free space and write to file
 * 
 */
#ifdef USE_INTEL
    mkl_free(A);
    mkl_free(x);
    mkl_free(y);
#endif

#if defined(USE_NEC) || defined(USE_AMD)
    free(A);
    free(x);
    free(y);
#endif

#ifdef USE_NVIDIA
    free(A);
    free(x);
    free(y);
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
#endif 

    return 0;
}
