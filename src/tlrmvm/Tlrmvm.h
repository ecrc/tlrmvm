#ifndef TLRMVM_H
#define TLRMVM_H


#include "cpu/PCMatrix.h"
#include "cpu/TlrmvmCPU.h"

#ifdef USE_DPCPP 
#include "cpu/TlrmvmDPCPP.h"
#endif

#ifdef USE_CUDA
#include "common/cuda/cublasInterface.h"
#include "common/cuda/Util.h"
#include "cuda/TlrmvmGPU.h"
#include "cuda/TlrmvmKernel.cuh"
#endif

#endif



