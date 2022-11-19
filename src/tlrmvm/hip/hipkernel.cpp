//
// Created by Yuxi Hong on 28/02/2022.
//

#include "../../common/hip/Util.hpp"
#include "hipkernel.cuh"


namespace hiptlrmvm {

    template<typename T>
    __global__ void phase2_nosplit_kernel(const T * __restrict__ yv,
                                          const size_t * __restrict__ phase2mapping, T * __restrict__ yu, size_t len)
    {
        size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
        if (thread_x < len){
            yu[phase2mapping[thread_x]] = yv[thread_x];
        }
    }

    template<typename T>
    void phase2_nosplit(const T *yv, const size_t * phase2mapping, T * yu, size_t len, hipStream_t stream){
        int dimx = 512;
        int griddimx = (len+dimx-1) / dimx;
        phase2_nosplit_kernel<<<griddimx, dimx, 0, stream>>>(yv, phase2mapping, yu, len);
        HIPCHECK(hipGetLastError());
    }

    template void phase2_nosplit<float>(const float*, const size_t *, float *, size_t, hipStream_t);
    template void phase2_nosplit<double>(const double*, const size_t *, double *, size_t, hipStream_t);
    template void phase2_nosplit<hipDoubleComplex>(const hipDoubleComplex*, const size_t *,
            hipDoubleComplex *, size_t, hipStream_t);
    template void phase2_nosplit<hipComplex>(const hipComplex*, const size_t *, hipComplex *, size_t, hipStream_t);
//    template void phase2_nosplit<cuHalfComplex>(const cuHalfComplex*, const size_t *, cuHalfComplex *, size_t, hipStream_t);


    __forceinline__ __device__ float conj(float Invec){
        return Invec;
    }
    __forceinline__ __device__ double conj(double Invec){
        return Invec;
    }
    __forceinline__ __device__ hipComplex conj(hipComplex Invec){
        return {Invec.x, -Invec.y};
    }
    __forceinline__ __device__ hipDoubleComplex conj(hipDoubleComplex Invec){
        return {Invec.x, -Invec.y};
    }

    template<typename T>
    __global__ void ConjugateKernel(T *Invec, size_t length)
    {
        size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
        if (thread_x < length){
            Invec[thread_x] = conj(Invec[thread_x]);
        }
    }

    template<typename T>
    void ConjugateDriver(T *Invec, size_t length, hipStream_t stream){
        int dimx = 512;
        int griddimx = (length+dimx-1) / dimx;
        ConjugateKernel<<<griddimx, dimx, 0, stream>>>(Invec, length);
        HIPCHECK(hipGetLastError());
    }
    template void ConjugateDriver<float>(float *Invec, size_t length, hipStream_t stream);
    template void ConjugateDriver<double>(double *Invec, size_t length, hipStream_t stream);

    template void ConjugateDriver<hipComplex>(hipComplex *Invec, size_t length, hipStream_t stream);
    template void ConjugateDriver<hipDoubleComplex>(hipDoubleComplex *Invec, size_t length, hipStream_t stream);


} // namespace

