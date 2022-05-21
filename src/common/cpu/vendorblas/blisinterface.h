#ifndef BLIS_INTERFACE_H
#define BLIS_INTERFACE_H

#include <complex>

#ifdef USE_BLIS
#include <blis.h>


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


#endif 

#endif // BLISINTERFACE_H

