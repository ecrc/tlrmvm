// @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                     All rights reserved.

#ifndef TLRMVM_H
#define TLRMVM_H


#include "cpu/PCMatrix.hpp"
#include "cpu/TlrmvmCPU.hpp"

#ifdef SPLITRI
#include "cpu/TlrmvmCPUSplitRI.hpp"
#endif

#ifdef USE_DPCPP 
#include "cpu/TlrmvmDPCPP.h"
#endif

#ifdef USE_CUDA
#include "../common/Common.hpp"
#include "../common/cuda/Util.hpp"
#include "cuda/Tlrmvmcuda.hpp"
#include "cuda/BatchTlrmvmcuda.hpp"
#include "cuda/BatchTlrmvmcudaFP16.hpp"
#include "cuda/BatchTlrmvmcudaINT8.hpp"

#include "cuda/tlrmvmcudautil.hpp"
#include "cuda/TlrmvmcudaConstRank.hpp"
#include "cuda/TlrmvmMPfp16.hpp"
#include "cuda/TlrmvmMPint8.hpp"
#ifdef USE_A100_MP_TimeDecomp
#include "cuda/BatchTlrmvmcuda_p1.hpp"
#include "cuda/BatchTlrmvmcuda_p1p2.hpp"
#include "cuda/BatchTlrmvmcudaFP16_p1.hpp"
#include "cuda/BatchTlrmvmcudaFP16_p1p2.hpp"
#include "cuda/BatchTlrmvmcudaINT8_p1.hpp"
#include "cuda/BatchTlrmvmcudaINT8_p1p2.hpp"
#endif
#endif


#ifdef USE_HIP
#include "../common/Common.hpp"
#include "../common/hip/Util.hpp"
#include "hip/Tlrmvmhip.hpp"
#include "hip/BatchTlrmvmhip.hpp"
#include "hip/TlrmvmhipConstRank.hpp"
#endif

#endif



