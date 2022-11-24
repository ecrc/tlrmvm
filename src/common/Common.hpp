// @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                     All rights reserved.

#pragma once

#include "cpu/BlasInterface.hpp"
#include "cpu/Util.hpp"
#include "cpu/Matrix.hpp"

#ifdef USE_CUDA
#include "cuda/cublasInterface.hpp"
#include "cuda/Util.hpp"
#endif

#ifdef USE_HIP
#include "hip/hipblasInterface.hpp"
#include "hip/Util.hpp"
#endif



