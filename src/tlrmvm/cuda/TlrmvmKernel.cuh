#ifndef TLRMVMKERNEL_CUH
#define TLRMVMKERNEL_CUH

#include <iostream>
#include <complex>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h> // if you need CUBLAS v2, include before magma.h
#include <cuda_fp16.h>

using namespace std;

namespace cudatlrmvm{

void MergeyfinalBasicDriver(float ** y, int vectorlength, int streamsize, cudaStream_t cudastream=0);


void phase2dirver(float * yu, float *yv, 
unsigned long int vlength, unsigned long int* offsetinyu, cudaStream_t stream = 0);

void phase2gpuconstrankreshuffledriver(float * y_in, float * y_out, 
int Aurows, int Aucols,
int Avrows, int Avcols,
int m, int n, int nt, int mt, int nb, int size);

void fp16gemv(complex<float> * mat, int m, int n, complex<float> * x, complex<float> *y);

// __global__ void phase1(float *Av,float *x, float *yv, 
// int *coloffset, int nb, size_t totallda);

void phase1driver(float * Av, float * x, float * yv, 
int * coloffset, int * collda, int nb, size_t Ntlocal, size_t granksum);


void phase1fuse4ops_real(half * Arealreal, half * Bimagimag, size_t len, cudaStream_t stream);

void phase1fuse4ops_imag(half * Arealimag, half * Bimagreal, size_t len, cudaStream_t stream);

void phase1twogemv(half * Arealimag, half * Bimagreal, size_t len, cudaStream_t stream);




void merge_half_realimag(half *real, half *imag, size_t len, cudaStream_t stream);

void merge_float_realimag(float *real, float *imag, cuComplex * complexout, size_t len, cudaStream_t stream);


// normal phase 2
template<typename T>
void phase2_nosplit(const T *yv, const size_t * phase2mapping, 
    T * yu, size_t len, cudaStream_t stream);

// FLOAT
void merge_float_2floatout_realimag(const float *real, const float *imag, 
const int* colrank, const int ntg, 
float *realout, float *imagout, size_t len, cudaStream_t stream);

void phase2(const float *real, const float *imag, const int *colrank, const int ntg,
const size_t * phase2mapping, float * yuout, size_t len, cudaStream_t stream);

void phase2(const half *real, const half *imag, const int *colrank, const int ntg,
    const size_t * phase2mapping, half * yuout, size_t len, cudaStream_t stream);

void phase3_merge(const float *rr_ri, const float *ir_ii,
const int nb, cuComplex *finaly, size_t len, cudaStream_t stream);




// HALF
void merge_half_2halfout_realimag(const half *rr_ri, const half *ir_ii, const int * colrank, const int ntg,
half *realout, half *imagout,  size_t len, cudaStream_t stream);
void phase2_half(const half *real, const half *imag, const int *colrank, const int ntg,
const size_t * phase2mapping, half * yuout, size_t len, cudaStream_t stream);
void phase3_merge_half(const half *rr_ri, const half *ir_ii,
const int nb, cuComplex *finaly, size_t len, cudaStream_t stream);

// INT
void merge_int_2intout_realimag(
const int *rr_ri, const int *ir_ii, const int * colrank, 
const int * colrank_withpadding, const int ntg,
float *realout, float *imagout,  size_t len,  size_t originlen,  cudaStream_t stream);

} // namespace



#endif 


