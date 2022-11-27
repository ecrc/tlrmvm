//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#ifndef OPENBLAS_INTERFACE_H
#define OPENBLAS_INTERFACE_H

#include <complex>

#if defined(USE_OPENBLAS) || defined(USE_COMPILER_BLAS)
#include <cblas.h>

/*******************
 * GEMV
 * ******************/
void cblasgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans,
int m, int n,
float alpha, const float *A, int lda, const float *x, int incx,
float beta, float *y, int incy);
void cblasgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans,
int m, int n, double alpha, const double *A, int lda,
const double *x, int incx, double beta, double *y, int incy);
void cblasgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans,
int m, int n, std::complex<float> alpha, const std::complex<float> *A, int lda,
const std::complex<float> *x, int incx,
std::complex<float> beta, std::complex<float> *y, int incy);
void cblasgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans,
int m, int n, std::complex<double> alpha, const std::complex<double> *A, int lda,
const std::complex<double> *x, int incx,
std::complex<double> beta, std::complex<double> *y, int incy);

/*******************
 * GEMM
 * ******************/
void cblasgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE trans_a,
CBLAS_TRANSPOSE trans_b,int m, int n, int k,
float alpha, const float *A, int lda,
const float *B, int ldb, float beta, float *C, int ldc);
void cblasgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE trans_a,
CBLAS_TRANSPOSE trans_b,int m, int n, int k,
double alpha, const double *A, int lda,
const double *B, int ldb, double beta, double *C, int ldc);
void cblasgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE trans_a,
CBLAS_TRANSPOSE trans_b,int m, int n, int k, std::complex<float> alpha,
const std::complex<float> *A, int lda,
const std::complex<float> *B, int ldb, std::complex<float> beta, std::complex<float> *C, int ldc);
void cblasgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE trans_a,
CBLAS_TRANSPOSE trans_b,int m, int n, int k, std::complex<double> alpha,
const std::complex<double> *A, int lda,
const std::complex<double> *B, int ldb, std::complex<double> beta, std::complex<double> *C, int ldc);

#ifdef USE_FUJITSU
void cblasgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE trans_a,
CBLAS_TRANSPOSE trans_b,int m, int n, int k, __fp16 alpha,
const __fp16 *A, int lda,
const __fp16 *B, int ldb, __fp16 beta, __fp16 *C, int ldc);
#endif

#endif

#endif // OPENBLASINTERFACE_H

