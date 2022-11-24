//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

//
// Created by Yuxi Hong on 28/02/2022.
//

#ifndef HIP_HIPBLAS_INTERFACE_H
#define HIP_HIPBLAS_INTERFACE_H

#include <hipblas.h>
#include <hip/hip_complex.h>
#include <hip/hip_fp16.h>


hipblasStatus_t hipblasgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                          int m, int n,
                          const float *alpha,
                          const float *A, int lda,
                          const float *x, int incx,
                          const float *beta,
                          float *y, int incy);


hipblasStatus_t hipblasgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                            int m, int n,
                            const double *alpha,
                            const double *A, int lda,
                            const double *x, int incx,
                            const double *beta,
                            double *y, int incy);

hipblasStatus_t hipblasgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                            int m, int n,
                            const hipComplex *alpha,
                            const hipComplex *A, int lda,
                            const hipComplex *x, int incx,
                            const hipComplex *beta,
                            hipComplex *y, int incy);

hipblasStatus_t hipblasgemv(hipblasHandle_t handle, hipblasOperation_t trans,
                            int m, int n,
                            const hipDoubleComplex *alpha,
                            const hipDoubleComplex *A, int lda,
                            const hipDoubleComplex *x, int incx,
                            const hipDoubleComplex *beta,
                            hipDoubleComplex *y, int incy);


hipblasStatus_t hipblasgemm(hipblasHandle_t handle,
                            hipblasOperation_t transa, hipblasOperation_t transb,
                            int m, int n, int k,
                            const float           *alpha,
                            const float           *A, int lda,
                            const float           *B, int ldb,
                            const float           *beta,
                            float           *C, int ldc);
hipblasStatus_t hipblasgemm(hipblasHandle_t handle,
                            hipblasOperation_t transa, hipblasOperation_t transb,
                            int m, int n, int k,
                            const double          *alpha,
                            const double          *A, int lda,
                            const double          *B, int ldb,
                            const double          *beta,
                            double          *C, int ldc);
hipblasStatus_t hipblasgemm(hipblasHandle_t handle,
                            hipblasOperation_t transa, hipblasOperation_t transb,
                            int m, int n, int k,
                            const hipComplex       *alpha,
                            const hipComplex       *A, int lda,
                            const hipComplex       *B, int ldb,
                            const hipComplex       *beta,
                            hipComplex       *C, int ldc);
hipblasStatus_t hipblasgemm(hipblasHandle_t handle,
                            hipblasOperation_t transa, hipblasOperation_t transb,
                            int m, int n, int k,
                            const hipDoubleComplex *alpha,
                            const hipDoubleComplex *A, int lda,
                            const hipDoubleComplex *B, int ldb,
                            const hipDoubleComplex *beta,
                            hipDoubleComplex *C, int ldc);

hipblasStatus_t hipblasgemm(hipblasHandle_t handle,
                            hipblasOperation_t transa, hipblasOperation_t transb,
                            int m, int n, int k, const half *alpha,
                            const half *A, int lda,
                            const half *B, int ldb,
                            const half *beta,
                            half *C, int ldc);

hipblasStatus_t hipblasgemmbatched(hipblasHandle_t handle,
                                 hipblasOperation_t transa,
                                 hipblasOperation_t transb,
                                 int m, int n, int k,
                                 const __half *alpha,
                                 const __half *Aarray[], int lda,
                                 const __half *Barray[], int ldb,
                                 const __half *beta,
                                 __half *Carray[], int ldc,
                                 int batchCount);

hipblasStatus_t hipblasgemmbatched(hipblasHandle_t handle,
                                 hipblasOperation_t transa,
                                 hipblasOperation_t transb,
                                 int m, int n, int k,
                                 const float *alpha,
                                 const float *Aarray[], int lda,
                                 const float *Barray[], int ldb,
                                 const float *beta,
                                 float *Carray[], int ldc,
                                 int batchCount);

hipblasStatus_t hipblasgemmbatched(hipblasHandle_t handle,
                                 hipblasOperation_t transa,
                                 hipblasOperation_t transb,
                                 int m, int n, int k,
                                 const double *alpha,
                                 const double *Aarray[], int lda,
                                 const double *Barray[], int ldb,
                                 const double *beta,
                                 double *Carray[], int ldc,
                                 int batchCount);

hipblasStatus_t hipblasgemmbatched(hipblasHandle_t handle,
                                 hipblasOperation_t transa,
                                 hipblasOperation_t transb,
                                 int m, int n, int k,
                                 const hipComplex *alpha,
                                 const hipComplex **Aarray, int lda,
                                 const hipComplex **Barray, int ldb,
                                 const hipComplex *beta,
                                 hipComplex **Carray, int ldc,
                                 int batchCount);

hipblasStatus_t hipblasgemmbatched(hipblasHandle_t handle,
                                 hipblasOperation_t transa,
                                 hipblasOperation_t transb,
                                 int m, int n, int k,
                                 const hipDoubleComplex *alpha,
                                 const hipDoubleComplex *Aarray[], int lda,
                                 const hipDoubleComplex *Barray[], int ldb,
                                 const hipDoubleComplex *beta,
                                 hipDoubleComplex *Carray[], int ldc,
                                 int batchCount);

#endif //HIP_HIPBLAS_INTERFACE_H
