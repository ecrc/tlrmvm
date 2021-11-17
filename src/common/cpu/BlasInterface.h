#ifndef BLASINTERFACE_H
#define BLASINTERFACE_H

#include <complex>
using std::complex;


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

namespace tlrmat
{

void gemm(const int *A, const int *B, int *C, int m, int n, int k);
void gemm(const float *A, const float *x, float *y, int m, int n, int k);
void gemm(const double *A, const double *x, double *y, int m, int n, int k);
void gemm(const complex<float> *A, const complex<float> *x, complex<float> *y, int m, int n, int k);
void gemm(const complex<double> *A, const complex<double> *x, complex<double> *y, int m, int n, int k);



} // namespace tlrmat



#endif // BLASINTERFACE


