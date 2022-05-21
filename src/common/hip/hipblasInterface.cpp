//
// Created by Yuxi Hong on 28/02/2022.
//

#include "hipblasInterface.hpp"

hipblasStatus_t hipblasgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                          int m, int n,
                          const float *alpha,
                          const float *A, int lda,
                          const float *x, int incx,
                          const float *beta,
                          float *y, int incy){
    return hipblasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}


hipblasStatus_t hipblasgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                          int m, int n,
                          const double *alpha,
                          const double *A, int lda,
                          const double *x, int incx,
                          const double *beta,
                          double *y, int incy){
    return hipblasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

hipblasStatus_t hipblasgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                          int m, int n,
                          const hipComplex *alpha,
                          const hipComplex *A, int lda,
                          const hipComplex *x, int incx,
                          const hipComplex *beta,
                          hipComplex *y, int incy){
    return hipblasCgemv(handle, trans, m, n,
                        reinterpret_cast<const hipblasComplex*>(alpha),
                        reinterpret_cast<const hipblasComplex*>(A), lda,
                        reinterpret_cast<const hipblasComplex*>(x), incx,
                        reinterpret_cast<const hipblasComplex*>(beta),
                        reinterpret_cast<hipblasComplex*>(y), incy);
}

hipblasStatus_t hipblasgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                          int m, int n,
                          const hipDoubleComplex *alpha,
                          const hipDoubleComplex *A, int lda,
                          const hipDoubleComplex *x, int incx,
                          const hipDoubleComplex *beta,
                          hipDoubleComplex *y, int incy){
    return hipblasZgemv(handle, trans, m, n,
                        reinterpret_cast<const hipblasDoubleComplex*>(alpha),
                        reinterpret_cast<const hipblasDoubleComplex*>(A), lda,
                        reinterpret_cast<const hipblasDoubleComplex*>(x), incx,
                        reinterpret_cast<const hipblasDoubleComplex*>(beta),
                        reinterpret_cast<hipblasDoubleComplex*>(y), incy);
}


hipblasStatus_t hipblasgemm(hipblasHandle_t handle,
                          hipblasOperation_t transa, hipblasOperation_t transb,
                          int m, int n, int k,const float *alpha,
                          const float *A, int lda,
                          const float *B, int ldb,
                          const float *beta,
                          float *C, int ldc){
    return hipblasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

hipblasStatus_t hipblasgemm(hipblasHandle_t handle,
                          hipblasOperation_t transa, hipblasOperation_t transb,
                          int m, int n, int k,const double *alpha,
                          const double *A, int lda,
                          const double *B, int ldb,
                          const double *beta,
                          double *C, int ldc){
    return hipblasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

hipblasStatus_t hipblasgemm(hipblasHandle_t handle,
                          hipblasOperation_t transa, hipblasOperation_t transb,
                          int m, int n, int k,const hipComplex *alpha,
                          const hipComplex *A, int lda,
                          const hipComplex *B, int ldb,
                          const hipComplex *beta,
                          hipComplex *C, int ldc){
    return hipblasCgemm(handle, transa, transb, m, n, k,
                        reinterpret_cast<const hipblasComplex*>(alpha),
                        reinterpret_cast<const hipblasComplex*>(A), lda,
                        reinterpret_cast<const hipblasComplex*>(B), ldb,
                        reinterpret_cast<const hipblasComplex*>(beta),
                        reinterpret_cast<hipblasComplex*>(C), ldc);
}
hipblasStatus_t hipblasgemm(hipblasHandle_t handle,
                          hipblasOperation_t transa, hipblasOperation_t transb,
                          int m, int n, int k,const hipDoubleComplex *alpha,
                          const hipDoubleComplex *A, int lda,
                          const hipDoubleComplex *B, int ldb,
                          const hipDoubleComplex *beta,
                          hipDoubleComplex *C, int ldc){
    return hipblasZgemm(handle, transa, transb, m, n, k,
                        reinterpret_cast<const hipblasDoubleComplex*>(alpha),
                        reinterpret_cast<const hipblasDoubleComplex*>(A), lda,
                        reinterpret_cast<const hipblasDoubleComplex*>(B), ldb,
                        reinterpret_cast<const hipblasDoubleComplex*>(beta),
                        reinterpret_cast<hipblasDoubleComplex*>(C), ldc);
}
hipblasStatus_t hipblasgemm(hipblasHandle_t handle,
                          hipblasOperation_t transa, hipblasOperation_t transb,
                          int m, int n, int k, const half *alpha,
                          const half *A, int lda,
                          const half *B, int ldb,
                          const half *beta,
                          half *C, int ldc){
    return hipblasHgemm(handle, transa, transb, m, n, k,
                        reinterpret_cast<const hipblasHalf *>(alpha),
                        reinterpret_cast<const hipblasHalf *>(A), lda,
                        reinterpret_cast<const hipblasHalf *>(B), ldb,
                        reinterpret_cast<const hipblasHalf *>(beta),
                        reinterpret_cast<hipblasHalf *>(C), ldc);
}




hipblasStatus_t hipblasgemmbatched(hipblasHandle_t handle,
                                 hipblasOperation_t transa,
                                 hipblasOperation_t transb,
                                 int m, int n, int k,
                                 const __half *alpha,
                                 const __half *Aarray[], int lda,
                                 const __half *Barray[], int ldb,
                                 const __half *beta,
                                 __half *Carray[], int ldc,
                                 int batchCount){
    return hipblasHgemmBatched(handle, transa, transb, m, n, k,
                               reinterpret_cast<const hipblasHalf *>(alpha),
                               reinterpret_cast<const hipblasHalf **>(Aarray), lda,
                               reinterpret_cast<const hipblasHalf **>(Barray), ldb,
                               reinterpret_cast<const hipblasHalf *>(beta),
                               reinterpret_cast<hipblasHalf **>(Carray), ldc, batchCount);
}

hipblasStatus_t hipblasgemmbatched(hipblasHandle_t handle,
                                 hipblasOperation_t transa,
                                 hipblasOperation_t transb,
                                 int m, int n, int k,
                                 const float *alpha,
                                 const float *Aarray[], int lda,
                                 const float *Barray[], int ldb,
                                 const float *beta,
                                 float *Carray[], int ldc,
                                 int batchCount){
    return hipblasSgemmBatched(handle, transa, transb, m, n, k,
                               reinterpret_cast<const float *>(alpha),
                               reinterpret_cast<const float **>(Aarray), lda,
                               reinterpret_cast<const float **>(Barray), ldb,
                               reinterpret_cast<const float *>(beta),
                               reinterpret_cast<float **>(Carray), ldc, batchCount);
}

hipblasStatus_t hipblasgemmbatched(hipblasHandle_t handle,
                                 hipblasOperation_t transa,
                                 hipblasOperation_t transb,
                                 int m, int n, int k,
                                 const double *alpha,
                                 const double *Aarray[], int lda,
                                 const double *Barray[], int ldb,
                                 const double *beta,
                                 double *Carray[], int ldc,
                                 int batchCount){
    return hipblasDgemmBatched(handle, transa, transb, m, n, k,
                               reinterpret_cast<const double *>(alpha),
                               reinterpret_cast<const double **>(Aarray), lda,
                               reinterpret_cast<const double **>(Barray), ldb,
                               reinterpret_cast<const double *>(beta),
                               reinterpret_cast<double **>(Carray), ldc, batchCount);
}

hipblasStatus_t hipblasgemmbatched(hipblasHandle_t handle,
                                 hipblasOperation_t transa,
                                 hipblasOperation_t transb,
                                 int m, int n, int k,
                                 const hipComplex *alpha,
                                 const hipComplex **Aarray, int lda,
                                 const hipComplex **Barray, int ldb,
                                 const hipComplex *beta,
                                 hipComplex **Carray, int ldc,
                                 int batchCount){
    return hipblasCgemmBatched(handle, transa, transb, m, n, k,
                               reinterpret_cast<const hipblasComplex *>(alpha),
                               reinterpret_cast<const hipblasComplex **>(Aarray), lda,
                               reinterpret_cast<const hipblasComplex **>(Barray), ldb,
                               reinterpret_cast<const hipblasComplex *>(beta),
                               reinterpret_cast<hipblasComplex **>(Carray), ldc, batchCount);
}

hipblasStatus_t hipblasgemmbatched(hipblasHandle_t handle,
                                 hipblasOperation_t transa,
                                 hipblasOperation_t transb,
                                 int m, int n, int k,
                                 const hipDoubleComplex *alpha,
                                 const hipDoubleComplex *Aarray[], int lda,
                                 const hipDoubleComplex *Barray[], int ldb,
                                 const hipDoubleComplex *beta,
                                 hipDoubleComplex *Carray[], int ldc,
                                 int batchCount){
    return hipblasZgemmBatched(handle, transa, transb, m, n, k,
                               reinterpret_cast<const hipblasDoubleComplex *>(alpha),
                               reinterpret_cast<const hipblasDoubleComplex **>(Aarray), lda,
                               reinterpret_cast<const hipblasDoubleComplex **>(Barray), ldb,
                               reinterpret_cast<const hipblasDoubleComplex *>(beta),
                               reinterpret_cast<hipblasDoubleComplex **>(Carray), ldc, batchCount);
}




