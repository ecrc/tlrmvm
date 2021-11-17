#include "BlasInterface.h"

namespace tlrmat
{

void gemm(const int *A, const int *B, int *C, int m, int n, int k){
    float alpha(1.0), beta(1.0);
    float *fA, *fB, *fC;
    fA = new float[m*k];
    fB = new float[k*n];
    fC = new float[m*n];
    #pragma omp parallel for 
    for(size_t i=0; i<m*k; i++) fA[i] = (float)A[i];
    #pragma omp parallel for 
    for(size_t i=0; i<k*n; i++) fB[i] = (float)B[i];
    #pragma omp parallel for 
    for(size_t i=0; i<m*n; i++) fC[i] = (float)C[i];
    cblasgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 
    alpha, fA, m, fB, k, beta, fC, m);
    #pragma omp parallel for 
    for(size_t i=0; i<m*n; i++) C[i] = (int)fC[i];    
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

void gemm(const complex<float> *A, const complex<float> *B, complex<float> *C, int m, int n, int k){
    complex<float> alpha(1.0,0.0), beta(1.0,0.0);
    cblasgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, m, B, k, beta, C, m);
}

void gemm(const complex<double> *A, const complex<double> *B, complex<double> *C, int m, int n, int k){
    complex<double> alpha(1.0,0.0), beta(1.0,0.0);
    cblasgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, m, B, k, beta, C, m);
}


} // namespace tlrmat

