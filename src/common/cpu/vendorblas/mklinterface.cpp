#if defined(USE_MKL)
#include "mklinterface.h"

namespace tlrmat
{

void cblasgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE trans, 
int m, int n, float alpha, const float *A, int lda, const float *x, int incx, float beta, float *y, int incy){
    cblas_sgemv(order, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
void cblasgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE trans, 
int m, int n, double alpha, const double *A, int lda, const double *x, int incx, double beta, double *y, int incy){
    cblas_dgemv(order, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
void cblasgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE trans, 
int m, int n, complex<float> alpha, const complex<float> *A, int lda, 
const complex<float> *x, int incx, complex<float> beta, complex<float> *y, int incy){
    cblas_cgemv(order, trans, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}
void cblasgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE trans, 
int m, int n, complex<double> alpha, const complex<double> *A, int lda, 
const complex<double> *x, int incx, complex<double> beta, complex<double> *y, int incy){
    cblas_zgemv(order, trans, m, n, &alpha, A, lda, x, incx, &beta, y, incy);
}


void cblasgemm(const CBLAS_ORDER order, const CBLAS_TRANSPOSE trans_a, 
const CBLAS_TRANSPOSE trans_b,int m, int n, int k, 
float alpha, const float *A, int lda, 
const float *B, int ldb, float beta, float *C, int ldc){
    cblas_sgemm(order, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void cblasgemm(const CBLAS_ORDER order, const CBLAS_TRANSPOSE trans_a, 
const CBLAS_TRANSPOSE trans_b,int m, int n, int k, 
double alpha, const double *A, int lda, 
const double *B, int ldb, double beta, double *C, int ldc){
    cblas_dgemm(order, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void cblasgemm(const CBLAS_ORDER order, const CBLAS_TRANSPOSE trans_a, 
const CBLAS_TRANSPOSE trans_b,int m, int n, int k, complex<float> alpha, 
const complex<float> *A, int lda, 
const complex<float> *B, int ldb, complex<float> beta, complex<float> *C, int ldc){
    cblas_cgemm(order, trans_a, trans_b, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

void cblasgemm(const CBLAS_ORDER order, const CBLAS_TRANSPOSE trans_a, 
const CBLAS_TRANSPOSE trans_b,int m, int n, int k, complex<double> alpha, 
const complex<double> *A, int lda, 
const complex<double> *B, int ldb, complex<double> beta, complex<double> *C, int ldc){
    cblas_zgemm(order, trans_a, trans_b, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

} // namespace tlrmat

#endif
