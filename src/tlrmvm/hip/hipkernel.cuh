//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#pragma once

#include <iostream>
#include <complex>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>


namespace hiptlrmvm{

    // normal phase 2
    template<typename T>
    void phase2_nosplit(const T *yv, const size_t * phase2mapping, T * yu, size_t len, hipStream_t stream);

    // in-place conjugate convert
    template<typename T>
    void ConjugateDriver(T *Invec, size_t length, hipStream_t stream);



} // namespace
