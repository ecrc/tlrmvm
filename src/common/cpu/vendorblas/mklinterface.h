#ifndef MKL_INTERFACE_H
#define MKL_INTERFACE_H

#include <complex>

#ifdef USE_MKL

#include <mkl.h>

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
               CBLAS_TRANSPOSE trans_b, int m, int n, int k,
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

#ifdef SPLITRI

void cblasgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE trans_a,
               CBLAS_TRANSPOSE trans_b,int m, int n, int k, float alpha,
               const MKL_BF16 *A, int lda,
               const MKL_BF16 *B, int ldb, float beta, float *C, int ldc);



union conv_union_bf16{
    float float_part;
    MKL_BF16 int_part[2];
};

union conv_union_f16{
    MKL_F16  raw;
    struct {
        unsigned int frac : 10;
        unsigned int exp  :  5;
        unsigned int sign :  1;
    } bits;
};

union conv_union_f32{
    float raw;
    struct {
        unsigned int frac : 23;
        unsigned int exp  :  8;
        unsigned int sign :  1;
    } bits;
};

// convert function
MKL_F16 mkl_f2h(float x);
float mkl_h2f(MKL_F16 x);
float mkl_b2f(MKL_BF16 src);
MKL_BF16 mkl_f2b(float src);

#endif

#endif

#endif // MKL_INTERFACE_H

