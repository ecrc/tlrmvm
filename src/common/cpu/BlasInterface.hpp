//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#ifndef BLASINTERFACE_H
#define BLASINTERFACE_H

#include <complex>


/******************************************************************************//**
 * blasinterface.h:
 * This file offers a general interface to different vendor blas library and 
 * A simplified gemm interface.
 *******************************************************************************/


#if defined(USE_OPENBLAS) || defined(USE_COMPILER_BLAS)
#include "vendorblas/openblasinterface.h"
#endif 

#ifdef USE_MKL
#include "vendorblas/mklinterface.h"
#endif

#ifdef USE_BLIS
#include "vendorblas/blisinterface.h"
#endif

void gemm(const size_t *A, const size_t *B, size_t *C, int m, int n, int k);
void gemm(const int *A, const int *B, int *C, int m, int n, int k);
void gemm(const float *A, const float *x, float *y, int m, int n, int k);
void gemm(const double *A, const double *x, double *y, int m, int n, int k);
void gemm(const std::complex<float> *A, const std::complex<float> *x, std::complex<float> *y, int m, int n, int k);
void gemm(const std::complex<double> *A, const std::complex<double> *x, std::complex<double> *y, int m, int n, int k);



#endif // BLAS_INTERFACE


