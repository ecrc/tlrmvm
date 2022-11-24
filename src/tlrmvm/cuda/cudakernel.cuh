//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#ifndef TLRMVMKERNEL_CUH
#define TLRMVMKERNEL_CUH

#include <iostream>
#include <complex>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h> // if you need CUBLAS v2, include before magma.h
#include <cuda_fp16.h>
using namespace std;

#include "../../common/cuda/Util.hpp"

namespace cudatlrmvm{

    // normal phase 2
    template<typename T>
    void phase2_nosplit(const T *yv, const size_t * phase2mapping, T * yu, size_t len, cudaStream_t stream);

    void phase2_Int8_driver(const cuHalfComplex *yv, const size_t * phase2mapping,
                            cuHalfComplex * yu, size_t launchlen, size_t originlen,
                            size_t * xelems_device,
                            size_t * xelemsoffset_device,
                            cuComplex *p3xreductionbuffer_device,
                            cuComplex *xmax_device,
                            cuInt8Complex * output,
                            int batchcount,
                            cudaStream_t stream);

    void phase2_getmaxx_driver(cuHalfComplex * p3x,cuComplex * middlebuffer,
                               cuComplex * maxinfo, int batchidx,
                               cuInt8Complex * output, size_t totallength, cudaStream_t stream);
    // in-place conjugate convert
    template<typename T>
    void ConjugateDriver(T *Invec, size_t length, cudaStream_t stream);

    void Cgemv_Phase1_driver(cuComplex *A, cuComplex *x, cuComplex *y, int M, int N, cudaStream_t stream);
    void Cgemv_Phase1_driver(float* A, float *x, float *y, int M, int N, cudaStream_t stream);
    void Cgemv_Phase1_driver(double* A, double *x, double *y, int M, int N, cudaStream_t stream);
    void Cgemv_Phase1_driver(cuDoubleComplex* A, cuDoubleComplex *x, cuDoubleComplex *y, int M, int N, cudaStream_t stream);
    void Cgemv_Phase3_driver(cuComplex* A, cuComplex *x, cuComplex *y, int M, int N, cudaStream_t stream);
    void Cgemv_Phase3_driver(float* A, float *x, float *y, int M, int N, cudaStream_t stream);
    void Cgemv_Phase3_driver(double* A, double *x, double *y, int M, int N, cudaStream_t stream);
    void Cgemv_Phase3_driver(cuDoubleComplex* A, cuDoubleComplex *x, cuDoubleComplex *y, int M, int N, cudaStream_t stream);

    void Hgemv_Phase1_driver(cuHalfComplex *A, cuHalfComplex *x, cuHalfComplex *y, int M, int N, cudaStream_t stream);
    void Hgemv_Phase1_driver(cubfComplex* A, cubfComplex *x, cubfComplex *y, int M, int N, cudaStream_t stream);
    void Hgemv_Phase1_driver(float* A, float *x, float *y, int M, int N, cudaStream_t stream);
    void Hgemv_Phase1_driver(double* A, double *x, double *y, int M, int N, cudaStream_t stream);
    void Hgemv_Phase1_driver(cuDoubleComplex* A, cuDoubleComplex *x, cuDoubleComplex *y, int M, int N, cudaStream_t stream);
    void Hgemv_Phase1_Transpose_driver(cuHalfComplex *A, cuHalfComplex *x, cuHalfComplex *y, int M, int N, cudaStream_t stream);
    void Hgemv_Phase1_Transpose_driver(cubfComplex *A, cubfComplex *x, cubfComplex *y, int M, int N, cudaStream_t stream);
    void Hgemv_Phase1_Transpose_driver(float *A, float *x, float *y, int M, int N, cudaStream_t stream);
    void Hgemv_Phase1_Transpose_driver(double *A, double *x, double *y, int M, int N, cudaStream_t stream);
    void Hgemv_Phase1_Transpose_driver(cuDoubleComplex *A, cuDoubleComplex *x, cuDoubleComplex *y, int M, int N, cudaStream_t stream);
    void Hgemv_Phase3_driver(cuHalfComplex* A, cuHalfComplex *x, cuHalfComplex *y, int M, int N, cudaStream_t stream);
    void Hgemv_Phase3_driver(cubfComplex* A, cubfComplex *x, cubfComplex *y, int M, int N, cudaStream_t stream);
    void Hgemv_Phase3_driver(float* A, float *x, float *y, int M, int N, cudaStream_t stream);
    void Hgemv_Phase3_driver(double* A, double *x, double *y, int M, int N, cudaStream_t stream);
    void Hgemv_Phase3_driver(cuDoubleComplex* A, cuDoubleComplex *x, cuDoubleComplex *y, int M, int N, cudaStream_t stream);
    void Hgemv_Phase3_Transpose_driver(cuHalfComplex* A, cuHalfComplex *x, cuHalfComplex *y, int M, int N, cudaStream_t stream);
    void Hgemv_Phase3_Transpose_driver(cubfComplex* A, cubfComplex *x, cubfComplex *y, int M, int N, cudaStream_t stream);
    void Hgemv_Phase3_Transpose_driver(float* A, float *x, float *y, int M, int N, cudaStream_t stream);
    void Hgemv_Phase3_Transpose_driver(double* A, double *x, double *y, int M, int N, cudaStream_t stream);
    void Hgemv_Phase3_Transpose_driver(cuDoubleComplex* A, cuDoubleComplex *x, cuDoubleComplex *y, int M, int N, cudaStream_t stream);

    void Phase3UpcastingDriver(cuHalfComplex * outputhalf, cuInt8Complex * inputint8,
                               cuComplex maxinfo, int elemoffset, int elems, int parti, cudaStream_t stream);

    void Igemv_Phase1_driver(cuInt8Complex* A, cuInt8Complex *x,cuComplex maxA, cuComplex maxx,
                             cuHalfComplex *y, int M, int N, cudaStream_t stream);
    void Igemv_Phase1_Transpose_driver(cuInt8Complex* A, cuInt8Complex *x,cuComplex maxA, cuComplex maxx,
                             cuHalfComplex *y, int M, int N, cudaStream_t stream);
    void Igemv_Phase3_driver(cuInt8Complex* A, cuHalfComplex *x,cuComplex maxA,
                             cuHalfComplex *y, int M, int N, cudaStream_t stream);

    void New_Igemv_Phase3_driver(cuInt8Complex* A, cuInt8Complex *x,cuComplex * maxA, cuComplex * maxx,
                             cuHalfComplex *y, int M, int N, int batchidx, cudaStream_t stream);
} // namespace



#endif 


