//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#include <cuda.h>
#include <cublas_v2.h> // if you need CUBLAS v2, include before magma.h
#include <cuComplex.h>


// overload cublas gemv Interface

cublasStatus_t cublasgemv(cublasHandle_t handle, cublasOperation_t trans,
int m, int n,
const float *alpha,
const float *A, int lda,
const float *x, int incx,
const float *beta,
float *y, int incy){
    return cublasSgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublasgemv(cublasHandle_t handle, cublasOperation_t trans,
int m, int n,
const double *alpha,
const double *A, int lda,
const double *x, int incx,
const double *beta,
double *y, int incy){
    return cublasDgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublasgemv(cublasHandle_t handle, cublasOperation_t trans,
int m, int n,
const cuComplex *alpha,
const cuComplex *A, int lda,
const cuComplex *x, int incx,
const cuComplex *beta,
cuComplex *y, int incy){
    return cublasCgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublasgemv(cublasHandle_t handle, cublasOperation_t trans,
int m, int n,
const cuDoubleComplex *alpha,
const cuDoubleComplex *A, int lda,
const cuDoubleComplex *x, int incx,
const cuDoubleComplex *beta,
cuDoubleComplex *y, int incy){
    return cublasZgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}


cublasStatus_t cublasgemm(cublasHandle_t handle,
cublasOperation_t transa, cublasOperation_t transb,
int m, int n, int k,const float *alpha,
const float *A, int lda,
const float *B, int ldb,
const float *beta,
float *C, int ldc){
    return cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublasgemm(cublasHandle_t handle,
cublasOperation_t transa, cublasOperation_t transb,
int m, int n, int k,const double *alpha,
const double *A, int lda,
const double *B, int ldb,
const double *beta,
double *C, int ldc){
    return cublasDgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t cublasgemm(cublasHandle_t handle,
cublasOperation_t transa, cublasOperation_t transb,
int m, int n, int k,const cuComplex *alpha,
const cuComplex *A, int lda,
const cuComplex *B, int ldb,
const cuComplex *beta,
cuComplex *C, int ldc){
    return cublasCgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
cublasStatus_t cublasgemm(cublasHandle_t handle,
cublasOperation_t transa, cublasOperation_t transb,
int m, int n, int k,const cuDoubleComplex *alpha,
const cuDoubleComplex *A, int lda,
const cuDoubleComplex *B, int ldb,
const cuDoubleComplex *beta,
cuDoubleComplex *C, int ldc){
    return cublasZgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
cublasStatus_t cublasgemm(cublasHandle_t handle,
cublasOperation_t transa, cublasOperation_t transb,
int m, int n, int k, const __half *alpha,
const __half *A, int lda,
const __half *B, int ldb,
const __half *beta,
__half *C, int ldc){
    return cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


// cublas gemmex interface
//
//cublasStatus_t cublasgemmex(cublasHandle_t handle,
//cublasOperation_t transa,
//cublasOperation_t transb,
//int m, int n, int k, const void *alpha, const void *A,
//cudaDataType_t Atype, int lda, const void *B,
//cudaDataType_t Btype, int ldb, const void *beta, void *C,
//cudaDataType_t Ctype, int ldc, cublasComputeType_t computeType,
//cublasGemmAlgo_t algo){
//    return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb,
//    beta, C, Ctype, ldc, computeType, algo);
//}
//
//cublasStatus_t cublassgemmex(cublasHandle_t handle,
//cublasOperation_t transa, cublasOperation_t transb,
//int m, int n, int k, const float *alpha,
//const void *A, cudaDataType_t  Atype, int lda,
//const void *B, cudaDataType_t  Btype, int ldb,
//const float *beta, void *C, cudaDataType_t  Ctype, int ldc){
//    return cublasSgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
//}
//
//cublasStatus_t cublascgemmex(cublasHandle_t handle,
//cublasOperation_t transa, cublasOperation_t transb,
//int m, int n, int k, const cuComplex *alpha,
//const void *A, cudaDataType_t  Atype, int lda,
//const void *B, cudaDataType_t  Btype, int ldb,
//const cuComplex *beta, void *C, cudaDataType_t  Ctype, int ldc){
//    return cublasCgemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);
//}


// overload cublas gemmbatched Interface

cublasStatus_t cublasgemmbatched(cublasHandle_t handle,
cublasOperation_t transa,
cublasOperation_t transb,
int m, int n, int k,
const __half *alpha,
const __half *Aarray[], int lda,
const __half *Barray[], int ldb,
const __half *beta,
__half *Carray[], int ldc,
int batchCount){
    return cublasHgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, 
    Barray, ldb, beta, Carray, ldc, batchCount);
}

cublasStatus_t cublasgemmbatched(cublasHandle_t handle,
cublasOperation_t transa,
cublasOperation_t transb,
int m, int n, int k,
const float *alpha,
const float *Aarray[], int lda,
const float *Barray[], int ldb,
const float *beta,
float *Carray[], int ldc,
int batchCount){
    return cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, 
    Barray, ldb, beta, Carray, ldc, batchCount);
}

cublasStatus_t cublasgemmbatched(cublasHandle_t handle,
cublasOperation_t transa,
cublasOperation_t transb,
int m, int n, int k,
const double *alpha,
const double *Aarray[], int lda,
const double *Barray[], int ldb,
const double *beta,
double *Carray[], int ldc,
int batchCount){
    return cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, 
    Barray, ldb, beta, Carray, ldc, batchCount);
}

cublasStatus_t cublasgemmbatched(cublasHandle_t handle,
cublasOperation_t transa,
cublasOperation_t transb,
int m, int n, int k,
const cuComplex *alpha,
const cuComplex **Aarray, int lda,
const cuComplex **Barray, int ldb,
const cuComplex *beta,
cuComplex **Carray, int ldc,
int batchCount){
    return cublasCgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, 
    Barray, ldb, beta, Carray, ldc, batchCount);
}

cublasStatus_t cublasgemmbatched(cublasHandle_t handle,
cublasOperation_t transa,
cublasOperation_t transb,
int m, int n, int k,
const cuDoubleComplex *alpha,
const cuDoubleComplex *Aarray[], int lda,
const cuDoubleComplex *Barray[], int ldb,
const cuDoubleComplex *beta,
cuDoubleComplex *Carray[], int ldc,
int batchCount){
    return cublasZgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, 
    Barray, ldb, beta, Carray, ldc, batchCount);
}




