#ifdef USE_MKL
#include "mklinterface.h"

void cblasgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans,
               int m, int n, float alpha, const float *A, int lda,
               const float *x, int incx, float beta, float *y, int incy){
    cblas_sgemv(order, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
void cblasgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans,
               int m, int n, double alpha, const double *A, int lda,
               const double *x, int incx, double beta, double *y, int incy){
    cblas_dgemv(order, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
void cblasgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans,
               int m, int n, std::complex<float> alpha, const std::complex<float> *A, int lda,
               const std::complex<float> *x, int incx, std::complex<float> beta, std::complex<float> *y, int incy){
    cblas_cgemv(order, trans, m, n, &alpha, A,
                lda, x, incx, &beta, y, incy);
}
void cblasgemv(CBLAS_ORDER order, CBLAS_TRANSPOSE trans,
               int m, int n, std::complex<double> alpha, const std::complex<double> *A, int lda,
               const std::complex<double> *x, int incx, std::complex<double> beta, std::complex<double> *y, int incy){
    cblas_zgemv(order, trans, m, n, &alpha, A,
                lda, x, incx, &beta, y, incy);
}

void cblasgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE trans_a,
               CBLAS_TRANSPOSE trans_b, int m, int n, int k,
               float alpha, const float *A, int lda,
               const float *B, int ldb, float beta, float *C, int ldc){
    cblas_sgemm(order, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void cblasgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE trans_a,
               CBLAS_TRANSPOSE trans_b,int m, int n, int k,
               double alpha, const double *A, int lda,
               const double *B, int ldb, double beta, double *C, int ldc){
    cblas_dgemm(order, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void cblasgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE trans_a,
               CBLAS_TRANSPOSE trans_b,int m, int n, int k, std::complex<float> alpha,
               const std::complex<float> *A, int lda, const std::complex<float> *B, int ldb,
               std::complex<float> beta, std::complex<float> *C, int ldc){
    cblas_cgemm(order, trans_a, trans_b,
                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

void cblasgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE trans_a,
               CBLAS_TRANSPOSE trans_b,int m, int n, int k, std::complex<double> alpha,
               const std::complex<double> *A, int lda,
               const std::complex<double> *B, int ldb, std::complex<double> beta,
               std::complex<double> *C, int ldc){
    cblas_zgemm(order, trans_a, trans_b,
                m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

#ifdef SPLITRI

void cblasgemm(CBLAS_ORDER order, CBLAS_TRANSPOSE trans_a,
               CBLAS_TRANSPOSE trans_b,int m, int n, int k, float alpha,
               const MKL_BF16 *A, int lda,
               const MKL_BF16 *B, int ldb, float beta, float *C, int ldc){
    cblas_gemm_bf16bf16f32(order, trans_a, trans_b, m, n, k,
                           alpha,A, lda, B, ldb, beta, C, ldc);
}




MKL_F16 mkl_f2h(float x) {
    conv_union_f32 src;
    conv_union_f16 dst;

    src.raw  = x;
    dst.raw  = 0;
    dst.bits.sign = src.bits.sign;

    if (src.bits.exp == 0x0ffU) {
        dst.bits.exp  = 0x01fU;
        dst.bits.frac = (src.bits.frac >> 13);
        if (src.bits.frac > 0) dst.bits.frac |= 0x200U;
    } else if (src.bits.exp >= 0x08fU) {
        dst.bits.exp  = 0x01fU;
        dst.bits.frac = 0x000U;
    } else if (src.bits.exp >= 0x071U){
        dst.bits.exp  = src.bits.exp + ((1 << 4) - (1 << 7));
        dst.bits.frac = (src.bits.frac >> 13);
    } else if (src.bits.exp >= 0x067U){
        dst.bits.exp  = 0x000;
        if (src.bits.frac > 0) {
            dst.bits.frac = (((1U << 23) | src.bits.frac) >> 14);
        } else {
            dst.bits.frac = 1;
        }
    }

    return dst.raw;
}

float mkl_h2f(MKL_F16 x) {
    conv_union_f16 src;
    conv_union_f32 dst;

    src.raw  = x;
    dst.raw  = 0;
    dst.bits.sign = src.bits.sign;

    if (src.bits.exp == 0x01fU) {
        dst.bits.exp  = 0xffU;
        if (src.bits.frac > 0) {
            dst.bits.frac = ((src.bits.frac | 0x200U) << 13);
        }
    } else if (src.bits.exp > 0x00U) {
        dst.bits.exp  = src.bits.exp + ((1 << 7) - (1 << 4));
        dst.bits.frac = (src.bits.frac << 13);
    } else {
        unsigned int v = (src.bits.frac << 13);

        if (v > 0) {
            dst.bits.exp = 0x71;
            while ((v & 0x800000UL) == 0) {
                dst.bits.exp --;
                v <<= 1;
            }
            dst.bits.frac = v;
        }
    }

    return dst.raw;
}

float mkl_b2f(MKL_BF16 src) {
    conv_union_bf16 conv;
    conv.int_part[0] = 0;
    conv.int_part[1] = src;
    return conv.float_part;
}

MKL_BF16 mkl_f2b(float src) {
    conv_union_bf16 conv;
    conv.float_part = src;
    return conv.int_part[1];
}

#endif

#endif
