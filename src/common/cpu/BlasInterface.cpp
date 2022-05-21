#include "BlasInterface.hpp"

void gemm(const int *A, const int *B, int *C, int m, int n, int k){
    float alpha(1.0), beta(1.0);
    float *fA, *fB, *fC;
    fA = new float[m*k];
    fB = new float[k*n];
    fC = new float[m*n];
    for(size_t i=0; i<m*k; i++) fA[i] = (float)A[i];
    for(size_t i=0; i<k*n; i++) fB[i] = (float)B[i];
    for(size_t i=0; i<m*n; i++) fC[i] = (float)C[i];
    cblasgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
              alpha, fA, m, fB, k, beta, fC, m);
    for(size_t i=0; i<m*n; i++) C[i] = (int)fC[i];
    delete[] fA;
    delete[] fB;
    delete[] fC;
}

void gemm(const size_t *A, const size_t *B, size_t *C, int m, int n, int k){
    float alpha(1.0), beta(1.0);
    float *fA, *fB, *fC;
    fA = new float[m*k];
    fB = new float[k*n];
    fC = new float[m*n];
    for(size_t i=0; i<m*k; i++) fA[i] = (float)A[i];
    for(size_t i=0; i<k*n; i++) fB[i] = (float)B[i];
    for(size_t i=0; i<m*n; i++) fC[i] = (float)C[i];
    cblasgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
              alpha, fA, m, fB, k, beta, fC, m);
    for(size_t i=0; i<m*n; i++) C[i] = (size_t)fC[i];
    delete[] fA;
    delete[] fB;
    delete[] fC;
}

void gemm(const float *A, const float *B, float *C, int m, int n, int k){
    float alpha(1.0), beta(1.0);
    cblasgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
              alpha, A, m, B, k, beta, C, m);
}

void gemm(const double *A, const double *B, double *C, int m, int n, int k){
    double alpha(1.0), beta(1.0);
    cblasgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
              alpha, A, m, B, k, beta, C, m);
}

void gemm(const std::complex<float> *A, const std::complex<float> *B, std::complex<float> *C, int m, int n, int k){
    std::complex<float> alpha(1.0,0.0), beta(1.0,0.0);
    cblasgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
              alpha, A, m, B, k, beta, C, m);
}

void gemm(const std::complex<double> *A, const std::complex<double> *B, std::complex<double> *C, int m, int n, int k){
    std::complex<double> alpha(1.0,0.0), beta(1.0,0.0);
    cblasgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k,
              alpha, A, m, B, k, beta, C, m);
}

