//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#include <cuda.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include "cudakernel.cuh"

namespace cg = cooperative_groups;

using namespace std;
namespace cudatlrmvm {

//    template <unsigned int WarpSize>
//    __device__ __forceinline__ nv_bfloat16 warpReduceSum(nv_bfloat16 sum) {
//        if (WarpSize >= 32)sum = sum + __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
//        if (WarpSize >= 16)sum = sum + __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
//        if (WarpSize >= 8)sum = sum + __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
//        if (WarpSize >= 4)sum = sum + __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
//        if (WarpSize >= 2)sum = sum + __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
//        return sum;
//    }
//
//    template <unsigned int WarpSize>
//    __device__ __forceinline__ half warpReduceSum(half sum) {
//        if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
//        if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
//        if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
//        if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
//        if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
//        return sum;
//    }
    template <unsigned int WarpSize>
    __device__ __forceinline__ float warpReduceSum(float sum) {
        if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
        if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
        if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
        if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
        if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
        return sum;
    }
    template <unsigned int WarpSize>
    __device__ __forceinline__ int warpReduceSumInt(int sum) {
        if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
        if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
        if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
        if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
        if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
        return sum;
    }
    template <unsigned int WarpSize>
    __device__ __forceinline__ float warpReduceMax(float maxval) {
        if (WarpSize >= 32)maxval = max(maxval,__shfl_down_sync(0xffffffff, maxval, 16));
        if (WarpSize >= 16)maxval = max(maxval,__shfl_down_sync(0xffffffff, maxval, 8));
        if (WarpSize >= 8)maxval = max(maxval,__shfl_down_sync(0xffffffff, maxval, 4));
        if (WarpSize >= 4)maxval = max(maxval,__shfl_down_sync(0xffffffff, maxval, 2));
        if (WarpSize >= 2)maxval = max(maxval,__shfl_down_sync(0xffffffff, maxval, 1));
        return maxval;
    }

    template<typename T>
    __global__ void phase2_nosplit_kernel(const T * __restrict__ yv,
                                          const size_t * __restrict__ phase2mapping,
                                          T * __restrict__ yu, size_t len)
    {
        size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
        if (thread_x < len){
//            yu[thread_x] = yv[phase2mapping[thread_x]];
            yu[phase2mapping[thread_x]] = yv[thread_x];
        }
    }

    template<typename T>
    void phase2_nosplit(const T *yv, const size_t * phase2mapping,
    T * yu, size_t len, cudaStream_t stream){
        if(len == 0) return;
        int dimx = 512;
        int griddimx = (len+dimx-1) / dimx;
        phase2_nosplit_kernel<<<griddimx, dimx, 0, stream>>>(yv, phase2mapping, yu, len);
        CUDACHECK(cudaGetLastError());
    }

    template void phase2_nosplit<float>(const float*, const size_t *, float *, size_t, cudaStream_t);
    template void phase2_nosplit<double>(const double*, const size_t *, double *, size_t, cudaStream_t);
    template void phase2_nosplit<cuDoubleComplex>(const cuDoubleComplex*, const size_t *, cuDoubleComplex *, size_t, cudaStream_t);
    template void phase2_nosplit<cuComplex>(const cuComplex*, const size_t *, cuComplex *, size_t, cudaStream_t);
    template void phase2_nosplit<cuHalfComplex>(const cuHalfComplex*, const size_t *, cuHalfComplex *, size_t, cudaStream_t);
    template void phase2_nosplit<cubfComplex>(const cubfComplex*, const size_t *, cubfComplex *, size_t, cudaStream_t);


    template<int STRIDESIZE>
    __global__ void phase2_Int8_kernel(const cuHalfComplex * __restrict__ yv,
                                       const size_t * __restrict__ phase2mapping,
                                       cuHalfComplex * __restrict__ yu,
                                       size_t originlen,
                                       const size_t * __restrict__ xelems_device,
                                       const size_t * __restrict__ xelemsoffset_device,
                                       cuComplex *p3xreductionbuffer_device, const int batchcount)
    {
        size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
        static __shared__ cuComplex sm[4];
        cuHalfComplex tmpval = cuHalfComplex(0.0,0.0);
        int batchidx = 0;
        int curblocksize = (((int)xelems_device[batchidx]+STRIDESIZE-1)/STRIDESIZE)*STRIDESIZE;
        int blockview_offset = curblocksize;
        int residual = 0;
        while(batchidx < batchcount && thread_x >= blockview_offset){
            residual += curblocksize - xelems_device[batchidx];
            batchidx++;
            curblocksize = (((int)xelems_device[batchidx]+STRIDESIZE-1)/STRIDESIZE)*STRIDESIZE;
            blockview_offset += curblocksize;
        }
        int originview_rank = thread_x - residual;
        blockview_offset -= curblocksize;
        int localbatch_rank = thread_x - blockview_offset;
        if(localbatch_rank < xelems_device[batchidx] && originview_rank < originlen) {
            tmpval = yv[phase2mapping[originview_rank]];
        }

        cuComplex regval;
        regval.x = abs((float)tmpval.x);
        regval.y = abs((float)tmpval.y);

        regval.x = warpReduceMax<32>(regval.x);
        regval.y = warpReduceMax<32>(regval.y);

        int laneId = threadIdx.x % 32;
        int warpId = threadIdx.x / 32;
        if(laneId == 0) sm[warpId]=regval;
        __syncthreads();
        if(threadIdx.x < 4) regval = sm[laneId];
        if(warpId == 0){
            regval.x = warpReduceMax<4>(regval.x);
            regval.y = warpReduceMax<4>(regval.y);
        }
        if(threadIdx.x == 0) p3xreductionbuffer_device[batchidx * 128 + localbatch_rank/128] = regval;
    }

    __global__ void phase2_Int8_maxsave_kernel(const cuHalfComplex * __restrict__ yv,
                                        const size_t * __restrict__ phase2mapping,
                                        cuHalfComplex * __restrict__ yu,
                                        size_t originlen,
                                        const size_t * __restrict__ xelems_device,
                                        const size_t * __restrict__ xelemsoffset_device,
                                        cuComplex *p3xreductionbuffer_device,
                                        cuComplex *xmax_device,
                                        const int batchcount)
    {
        cuComplex regval = p3xreductionbuffer_device[blockIdx.x * 128 + threadIdx.x];
        static __shared__ cuComplex warpsum[4];
        regval.x = warpReduceMax<32>(regval.x);
        regval.y = warpReduceMax<32>(regval.y);
        const int laneId = threadIdx.x % 32;
        const int warpId = threadIdx.x / 32;
        if(laneId == 0) warpsum[warpId] = regval;
        __syncthreads();
        if(threadIdx.x < 4) regval = warpsum[laneId];
        if (warpId == 0) {
            regval.x = warpReduceMax<4>(regval.x);
            regval.y = warpReduceMax<4>(regval.y);
        }
        if (threadIdx.x == 0) {xmax_device[blockIdx.x] = regval;}
    }

    template<int STRIDESIZE>
    __global__ void convert_Int8_kernel(const cuHalfComplex * __restrict__ yv,
                                       const size_t * __restrict__ phase2mapping,
                                       cuHalfComplex * __restrict__ yu,
                                       size_t originlen,
                                       const size_t * __restrict__ xelems_device,
                                       const size_t * __restrict__ xelemsoffset_device,
                                       cuComplex *p3xreductionbuffer_device,
                                       cuComplex *xmax_device,
                                       cuInt8Complex * output,
                                       const int batchcount)
    {
        size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
        int batchidx = 0;
        int curblocksize = (((int)xelems_device[batchidx]+STRIDESIZE-1)/STRIDESIZE)*STRIDESIZE;
        int blockview_offset = curblocksize;
        int residual = 0;
        while(batchidx < batchcount && thread_x >= blockview_offset){
            residual += curblocksize - xelems_device[batchidx];
            batchidx++;
            curblocksize = (((int)xelems_device[batchidx]+STRIDESIZE-1)/STRIDESIZE)*STRIDESIZE;
            blockview_offset += curblocksize;
        }
        int originview_rank = thread_x - residual;
        blockview_offset -= curblocksize;
        int localbatch_rank = thread_x - blockview_offset;
        cuComplex maxinfo = xmax_device[batchidx];
        if(localbatch_rank < xelems_device[batchidx] && originview_rank < originlen) {
            cuHalfComplex halfval = yv[phase2mapping[originview_rank]];
            auto real = (int8_t)((float)halfval.x / maxinfo.x * 125.0);
            auto imag = (int8_t)((float)halfval.y / maxinfo.y * 125.0);
            output[originview_rank] = cuInt8Complex(real,imag);
        }
    }

    void phase2_Int8_driver(const cuHalfComplex *yv, const size_t * phase2mapping,
                            cuHalfComplex * yu, size_t launchlen, size_t originlen,
                            size_t * xelems_device,
                            size_t * xelemsoffset_device,
                            cuComplex *p3xreductionbuffer_device,
                            cuComplex *xmax_device,
                            cuInt8Complex * output,
                            int batchcount,
                            cudaStream_t stream)
    {
        if(originlen == 0) return;
        int dimx = 128;
        int griddimx = (launchlen+dimx-1) / dimx;
        phase2_Int8_kernel<128><<<griddimx, 128, 32, stream>>>(yv, phase2mapping,yu,
                                                            originlen,
                                                            xelems_device,
                                                            xelemsoffset_device,
                                                            p3xreductionbuffer_device,
                                                            batchcount);

        phase2_Int8_maxsave_kernel<<<batchcount, 128, 32, stream>>>(yv, phase2mapping,yu,
                                                                    originlen,
                                                                    xelems_device,
                                                                    xelemsoffset_device,
                                                                    p3xreductionbuffer_device,
                                                                    xmax_device,
                                                                    batchcount);
        convert_Int8_kernel<128><<<griddimx, dimx, 0, stream>>>(yv, phase2mapping,yu,
                                                                 originlen,
                                                                 xelems_device,
                                                                 xelemsoffset_device,
                                                                 p3xreductionbuffer_device,
                                                                 xmax_device,
                                                                 output,
                                                                 batchcount);
    }

    __forceinline__ __device__ float conj(float Invec){
        return Invec;
    }
    __forceinline__ __device__ double conj(double Invec){
        return Invec;
    }
    __forceinline__ __device__ cuComplex conj(cuComplex Invec){
        return {Invec.x, -Invec.y};
    }
    __forceinline__ __device__ cuDoubleComplex conj(cuDoubleComplex Invec){
        return {Invec.x, -Invec.y};
    }

#if __CUDA_ARCH__ >= 800
    __forceinline__ __device__ cuHalfComplex conj(cuHalfComplex Invec){
        return {Invec.x, __hneg(Invec.y)};
    }
    __forceinline__ __device__ cubfComplex conj(cubfComplex Invec){
        return {Invec.x, __hneg(Invec.y)};
    }
#else
    __forceinline__ __device__ cuHalfComplex conj(cuHalfComplex Invec){
        return {Invec.x, -Invec.y};
    }
    __forceinline__ __device__ cubfComplex conj(cubfComplex Invec){
        return {__bfloat162float(Invec.x), -__bfloat162float(Invec.y)};
    }
#endif
    template<typename T>
    __global__ void ConjugateKernel(T *Invec, size_t length)
    {
        size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
        if (thread_x < length){
            Invec[thread_x] = conj(Invec[thread_x]);
        }
    }

    template<typename T>
    void ConjugateDriver(T *Invec, size_t length, cudaStream_t stream){
        int dimx = 512;
        int griddimx = (length+dimx-1) / dimx;
        ConjugateKernel<<<griddimx, dimx, 0, stream>>>(Invec, length);
        CUDACHECK(cudaGetLastError());
    }
    template void ConjugateDriver<float>(float *Invec, size_t length, cudaStream_t stream);
    template void ConjugateDriver<double>(double *Invec, size_t length, cudaStream_t stream);
    template void ConjugateDriver<cuComplex>(cuComplex *Invec, size_t length, cudaStream_t stream);
    template void ConjugateDriver<cuDoubleComplex>(cuDoubleComplex *Invec, size_t length, cudaStream_t stream);
    template void ConjugateDriver<cuHalfComplex>(cuHalfComplex *Invec, size_t length, cudaStream_t stream);
    template void ConjugateDriver<cubfComplex>(cubfComplex *Invec, size_t length, cudaStream_t stream);


    __global__ void Cgemv_v1(
            cuComplex * __restrict__ A,
            cuComplex * __restrict__ x,
            cuComplex * __restrict__ y,
            const int M,
            const int N) {
        // Block index
        int by = blockIdx.y;
        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        const int warp_size=32;
        int laneId= tx % warp_size;
        int current_row = (blockDim.y * by + ty);

        if(current_row < M){
            cuComplex res1;
            res1.x = 0;
            res1.y = 0;
            int kIteration = (N/warp_size);
            if(kIteration==0) kIteration=1;
            A = &A[current_row];
            #pragma unroll
            for(int i=0; i< kIteration; i++){
                int current_col_vec = (i*warp_size + laneId) * M;
                cuComplex current_val= A[current_col_vec];
                cuComplex current_x = x[i * 32 + laneId];
                res1.x += current_val.x*current_x.x - current_val.y*current_x.y;
                res1.y += current_val.x*current_x.y + current_val.y*current_x.x;
            }
            res1.x = warpReduceSum<warp_size>(res1.x);
            res1.y = warpReduceSum<warp_size>(res1.y);
            if(laneId==0) {
                y[current_row]=res1;
            }
        }
    }

    __global__ void Cgemv_v2(
            cuComplex * __restrict__ A,
            cuComplex * __restrict__ x,
            cuComplex * __restrict__ y,
            const int M,
            const int N) {
        // Block index
        int by = blockIdx.y;
        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        const int warp_size=32;
        int laneId = tx % warp_size;
        int current_row = (blockDim.y * by + ty) * 2;

        if(current_row < M){
            float4 res1;
            res1.x = res1.y = res1.z = res1.w = 0;
            int kIteration = (N/warp_size);
            if(kIteration==0) kIteration=1;
            for(int i=0; i< kIteration; i++){
                int current_col_vec = (i*warp_size + laneId) * M / 2;
                float4 current_val= reinterpret_cast<float4 *>(A)[current_col_vec + current_row/2];
                cuComplex current_x = x[i * warp_size + laneId];
                res1.x += current_val.x*current_x.x - current_val.y*current_x.y;
                res1.y += current_val.x*current_x.y + current_val.y*current_x.x;
                res1.z += current_val.z*current_x.x - current_val.w*current_x.y;
                res1.w += current_val.z*current_x.y + current_val.w*current_x.x;
            }
            res1.x = warpReduceSum<warp_size>(res1.x);
            res1.y = warpReduceSum<warp_size>(res1.y);
            res1.z = warpReduceSum<warp_size>(res1.z);
            res1.w = warpReduceSum<warp_size>(res1.w);
            if(laneId==0) reinterpret_cast<float4 *>(y)[current_row/2]=res1;
        }
    }

    __global__ void Cgemv_v2_Phase3(
            cuComplex * __restrict__ A,
            cuComplex * __restrict__ x,
            cuComplex * __restrict__ y,
            const int M,
            const int N) {
        // Block index
        int by = blockIdx.y;
        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        const int warp_size=32;
        int laneId = tx % warp_size;
        int current_row = (blockDim.y * by + ty) * 2;

        if(current_row < M){
            float4 res1;
            res1.x = res1.y = res1.z = res1.w = 0;
            int kIteration = (N/warp_size);
//            if(kIteration==0) kIteration=1;
            if(N%warp_size!=0) kIteration++;
            for(int i=0; i< kIteration; i++){
                int current_col_vec = (i*warp_size + laneId) * M / 2;
                if(i*warp_size + laneId < N){
                    float4 current_val= reinterpret_cast<float4 *>(A)[current_col_vec + current_row/2];
                    cuComplex current_x = x[i * warp_size + laneId];
                    res1.x += current_val.x*current_x.x - current_val.y*current_x.y;
                    res1.y += current_val.x*current_x.y + current_val.y*current_x.x;
                    res1.z += current_val.z*current_x.x - current_val.w*current_x.y;
                    res1.w += current_val.z*current_x.y + current_val.w*current_x.x;
                }
            }
            res1.x = warpReduceSum<warp_size>(res1.x);
            res1.y = warpReduceSum<warp_size>(res1.y);
            res1.z = warpReduceSum<warp_size>(res1.z);
            res1.w = warpReduceSum<warp_size>(res1.w);
            if(laneId==0) reinterpret_cast<float4 *>(y)[current_row/2]=res1;
        }
    }

    void Cgemv_Phase1_driver(cuComplex* A, cuComplex *x, cuComplex *y, int M, int N, cudaStream_t stream){
        int val = M / 8;
        if(M%8 != 0) val++;
        dim3 dimGrid(1,val);
        dim3 dimBlock(32,4);
        Cgemv_v2<<<dimGrid,dimBlock,0,stream>>>(A,x,y,M,N);
        cudaDeviceSynchronize();
        CUDACHECK(cudaGetLastError());
    }
    void Cgemv_Phase1_driver(float* A, float *x, float *y, int M, int N, cudaStream_t stream){}
    void Cgemv_Phase1_driver(double* A, double *x, double *y, int M, int N, cudaStream_t stream){}
    void Cgemv_Phase1_driver(cuDoubleComplex* A, cuDoubleComplex *x, cuDoubleComplex *y, int M, int N, cudaStream_t stream){}

    void Cgemv_Phase3_driver(cuComplex* A, cuComplex *x, cuComplex *y, int M, int N, cudaStream_t stream){
        int val = M / 8;
        if(M%8 != 0) val++;
        dim3 dimGrid(1,val);
        dim3 dimBlock(32,4);
        Cgemv_v2_Phase3<<<dimGrid,dimBlock,0,stream>>>(A,x,y,M,N);
        cudaDeviceSynchronize();
        CUDACHECK(cudaGetLastError());
    }
    void Cgemv_Phase3_driver(float* A, float *x, float *y, int M, int N, cudaStream_t stream){}
    void Cgemv_Phase3_driver(double* A, double *x, double *y, int M, int N, cudaStream_t stream){}
    void Cgemv_Phase3_driver(cuDoubleComplex* A, cuDoubleComplex *x, cuDoubleComplex *y, int M, int N, cudaStream_t stream){}

    __global__ void Hgemv_v1(
            cuHalfComplex * __restrict__ A,
            cuHalfComplex * __restrict__ x,
            cuHalfComplex * __restrict__ y,
            const int M,
            const int N) {
        // Block index
        int by = blockIdx.y;
        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        const int warp_size=32;
        int laneId= tx % warp_size;
        int current_row = blockDim.y * by + ty;
        if(current_row < M){
            cuHalfComplex results;
            results = cuHalfComplex(0.0,0.0);
            int kIteration = (N/warp_size);
            if(N % warp_size != 0) kIteration++;
            for(int i=0; i<kIteration; i++){
                if((i * warp_size+laneId) < N){
                    cuHalfComplex current_val = A[(i * warp_size+laneId) * M + current_row];
                    cuHalfComplex current_x = x[i*warp_size + laneId];
                    results = results + current_val * current_x;
                }
            }
            results.x = warpReduceSum<warp_size>(results.x);
            results.y = warpReduceSum<warp_size>(results.y);
            if(laneId==0) y[current_row] = results;
        }
    }

    // only for float 16 and bf 16
    template<typename T>
    __global__ void Hgemv_v2(
            T * __restrict__ A,
            T * __restrict__ x,
            T * __restrict__ y,
            const int M,
            const int N) {
        // Block index
        int by = blockIdx.y;
        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        const int warp_size=32;
        int laneId= tx % warp_size;
        int current_row = (blockDim.y * by + ty) * 4;

        if(current_row < M){
            T results[4];
            results[0] = results[1] = results[2] = results[3] = T(0.0,0.0);
            int kIteration = ( N / (warp_size*4) );
            if(kIteration==0) kIteration=1;
            for(int i=0; i<kIteration; i++){
                float4 f4x = reinterpret_cast<float4*>(x)[i*warp_size + laneId];
                auto *out_x = reinterpret_cast<T*>(&f4x);
                #pragma unroll
                for(int col=0; col<4; col++){
                    float4 current_val = reinterpret_cast<float4 *>(A)[ ((i*warp_size+laneId)*4+col) * M * 4 / 16 + current_row/4];
                    auto *current_A = reinterpret_cast<T*>(&current_val);
                    T current_x = out_x[col];

                    #pragma unroll
                    for(int row=0; row<4; row++){
                        results[row] = results[row] + current_A[row] * current_x;
                        if(current_row == 0 && laneId == 0 && row == 0 && col == 0 && i == 0){
                        }
                    }
                }
            }
            #pragma unroll
            for(int row=0; row<4; row++){
                results[row].x = warpReduceSum<warp_size>(results[row].x);
                results[row].y = warpReduceSum<warp_size>(results[row].y);
            }

            if(laneId==0) reinterpret_cast<float4 *>(y)[current_row/4] = *reinterpret_cast<float4 *>(&results);
        }
    }


//    __global__ void Hgemv_v2(
//            cuHalfComplex * __restrict__ A,
//            cuHalfComplex * __restrict__ x,
//            cuHalfComplex * __restrict__ y,
//            const int M,
//            const int N) {
//        // Block index
//        int by = blockIdx.y;
//        // Thread index
//        int tx = threadIdx.x;
//        int ty = threadIdx.y;
//        const int warp_size=32;
//        int laneId= tx % warp_size;
//        int current_row = (blockDim.y * by + ty) * 4;
//
//        if(current_row < M){
//            cuComplex results[4];
//            cuComplex zero;zero.x = zero.y = 0;
//            results[0] = results[1] = results[2] = results[3] = zero;
//            int kIteration = ( N / (warp_size*4) );
//            if(kIteration==0) kIteration=1;
//            for(int i=0; i<kIteration; i++){
//                float4 f4x = reinterpret_cast<float4*>(x)[i*warp_size + laneId];
//                auto *out_x = reinterpret_cast<cuHalfComplex*>(&f4x);
//#pragma unroll
//                for(int col=0; col<4; col++){
//                    float4 current_val = reinterpret_cast<float4 *>(A)[ ((i*warp_size+laneId)*4+col) * M * 4 / 16 + current_row/4];
//                    auto *current_A = reinterpret_cast<cuHalfComplex*>(&current_val);
//                    cuHalfComplex current_x = out_x[col];
//#pragma unroll
//                    for(int row=0; row<4; row++){
//                        results[row].x = results[row].x + (float)current_A[row].x * (float)current_x.x - (float)current_A[row].y * (float)current_x.y;
//                        results[row].y = results[row].y + (float)current_A[row].x * (float)current_x.y + (float)current_A[row].y * (float)current_x.x;
//                    }
//                }
//            }
//#pragma unroll
//            for(int row=0; row<4; row++){
//                results[row].x = warpReduceSum<warp_size>(results[row].x);
//                results[row].y = warpReduceSum<warp_size>(results[row].y);
//            }
//            if(laneId==0) {
//                cuHalfComplex halfresults[4];
//                for(int i=0; i<4; i++){
//                    halfresults[i] = cuHalfComplex(results[i].x, results[i].y);
//                }
//                reinterpret_cast<float4 *>(y)[current_row/4] = *reinterpret_cast<float4 *>(&halfresults);
//            }
//        }
//    }

    template<typename T>
    __global__ void Hgemv_T_v2(
            T * __restrict__ A,
            T * __restrict__ x,
            T * __restrict__ y,
            const int M,
            const int N) {
        // Block index
        int by = blockIdx.y;
        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        const int warp_size=32;
        int laneId= tx % warp_size;
        int current_row = (blockDim.y * by + ty) * 4;

        if(current_row < M){
            T results[4];
            results[0] = results[1] = results[2] = results[3] = T(0.0,0.0);
            int kIteration = ( N / (warp_size*4) );
            if(kIteration==0) kIteration=1;
            for(int i=0; i<kIteration; i++){
                float4 f4x = reinterpret_cast<float4*>(x)[i*warp_size + laneId];
                auto *out_x = reinterpret_cast<T*>(&f4x);
                #pragma unroll
                for(int row=0; row<4; row++){
                    float4 current_val = reinterpret_cast<float4 *>(A)[(current_row + row)* N * 4 / 16 + i*warp_size+laneId];
                    auto *current_A = reinterpret_cast<T*>(&current_val);
                    #pragma unroll
                    for(int col=0; col<4; col++){
                        T current_x = out_x[col];
                        results[row] = results[row] + current_A[col] * current_x;
                    }
                }
            }
            #pragma unroll
            for(int row=0; row<4; row++){
                results[row].x = warpReduceSum<warp_size>(results[row].x);
                results[row].y = warpReduceSum<warp_size>(results[row].y);
            }
            if(laneId==0) reinterpret_cast<float4 *>(y)[current_row/4] = *reinterpret_cast<float4 *>(&results);
        }
    }
    template<typename T>
    __global__ void Hgemv_v3(
            T * __restrict__ A,
            T * __restrict__ x,
            T * __restrict__ y,
            const int M,
            const int N) {
        // Block index
        int by = blockIdx.y;
        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        const int warp_size=32;
        int laneId= tx % warp_size;
        int current_row = (blockDim.y * by + ty) * 4;

        if(current_row < M){
            T results[4];
            results[0] = results[1] = results[2] = results[3] = T(0.0,0.0);
            int kIteration = ( N / (warp_size*4) );
            if(N % (warp_size * 4) != 0) kIteration++;
            for(int i=0; i<kIteration; i++){
                if((i*warp_size + laneId)*4 < N){
                    float4 f4x = reinterpret_cast<float4*>(x)[i*warp_size + laneId];
                    auto *out_x = reinterpret_cast<T*>(&f4x);
                    for(int col=0; col<4; col++){
                        if(((i*warp_size+laneId)*4+col) < N){
                            float4 current_val = reinterpret_cast<float4 *>(A)[ ((i*warp_size+laneId)*4+col) * M * 4 / 16 + current_row/4];
                            auto current_A = reinterpret_cast<T*>(&current_val);
                            T current_x = out_x[col];
                            for(int row=0; row<4; row++){
                                results[row] = results[row] + current_A[row] * current_x;
                            }
                        }
                    }
                }
            }
            for(int row=0; row<4; row++){
                results[row].x = warpReduceSum<warp_size>(results[row].x);
                results[row].y = warpReduceSum<warp_size>(results[row].y);
            }
            if(laneId==0) reinterpret_cast<float4 *>(y)[current_row/4] = *reinterpret_cast<float4 *>(&results);
        }
    }

    template<typename T>
    __global__ void Hgemv_T_v3(
            T * __restrict__ A,
            T * __restrict__ x,
            T * __restrict__ y,
            const int M,
            const int N) {
        // Block index
        int by = blockIdx.y;
        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        const int warp_size=32;
        int laneId= tx % warp_size;
        int current_row = (blockDim.y * by + ty) * 4;

        if(current_row < M){
            T results[4];
            results[0] = results[1] = results[2] = results[3] = T(0.0,0.0);
            int kIteration = ( N / (warp_size*4) );
            if(N % (warp_size * 4) != 0) kIteration++;
            for(int i=0; i<kIteration; i++){
                if((i*warp_size + laneId)*4 < N){
                    float4 f4x = reinterpret_cast<float4*>(x)[i*warp_size + laneId];
                    auto *out_x = reinterpret_cast<T*>(&f4x);
#pragma unroll
                    for(int row=0; row<4; row++){
                        float4 current_val = reinterpret_cast<float4 *>(A)[(current_row + row) * N * 4 / 16 + i*warp_size+laneId];
                        auto current_A = reinterpret_cast<T*>(&current_val);
#pragma unroll
                        for(int col=0; col<4; col++){
                            T current_x = out_x[col];
                            results[row] = results[row] + current_A[col] * current_x;
                        }
                    }
                }
            }
#pragma unroll
            for(int row=0; row<4; row++){
                results[row].x = warpReduceSum<warp_size>(results[row].x);
                results[row].y = warpReduceSum<warp_size>(results[row].y);
            }
            if(laneId==0) reinterpret_cast<float4 *>(y)[current_row/4] = *reinterpret_cast<float4 *>(&results);
        }
    }


    void Hgemv_Phase1_driver(cuHalfComplex* A, cuHalfComplex *x, cuHalfComplex *y, int M, int N, cudaStream_t stream){
        int val = M / 16;
        if(M%16 != 0) val++;
        dim3 dimGrid(1,val);
        dim3 dimBlock(32,4);
        Hgemv_v2<cuHalfComplex><<<dimGrid,dimBlock,0,stream>>>(A,x,y,M,N);

//        int val = M / 4;
//        if(M%4 != 0) val++;
//        dim3 dimGrid(1,val);
//        dim3 dimBlock(32,4);
//        Hgemv_v1<<<dimGrid,dimBlock,0,stream>>>(A,x,y,M,N);
        CUDACHECK(cudaGetLastError());
    }
    void Hgemv_Phase1_driver(cubfComplex* A, cubfComplex *x, cubfComplex *y, int M, int N, cudaStream_t stream){
        int val = M / 16;
        if(M%16 != 0) val++;
        dim3 dimGrid(1,val);
        dim3 dimBlock(32,4);
        Hgemv_v2<cubfComplex><<<dimGrid,dimBlock,0,stream>>>(A,x,y,M,N);
    }
    void Hgemv_Phase1_driver(float* A, float *x, float *y, int M, int N, cudaStream_t stream){}
    void Hgemv_Phase1_driver(double* A, double *x, double *y, int M, int N, cudaStream_t stream){}
    void Hgemv_Phase1_driver(cuDoubleComplex* A, cuDoubleComplex *x, cuDoubleComplex *y, int M, int N, cudaStream_t stream){}

    void Hgemv_Phase1_Transpose_driver(cuHalfComplex *A, cuHalfComplex *x, cuHalfComplex *y, int M, int N, cudaStream_t stream)
    {
        int val = M / 16;
        if(M%16 != 0) val++;
        dim3 dimGrid(1,val);
        dim3 dimBlock(32,4);
        Hgemv_T_v2<cuHalfComplex><<<dimGrid,dimBlock,0,stream>>>(A,x,y,M,N);
        CUDACHECK(cudaGetLastError());
    }
    void Hgemv_Phase1_Transpose_driver(cubfComplex *A, cubfComplex *x, cubfComplex *y, int M, int N, cudaStream_t stream){
        int val = M / 16;
        if(M%16 != 0) val++;
        dim3 dimGrid(1,val);
        dim3 dimBlock(32,4);
        Hgemv_T_v2<cubfComplex><<<dimGrid,dimBlock,0,stream>>>(A,x,y,M,N);
        CUDACHECK(cudaGetLastError());
    }
    void Hgemv_Phase1_Transpose_driver(float* A, float *x, float *y, int M, int N, cudaStream_t stream){}
    void Hgemv_Phase1_Transpose_driver(double* A, double *x, double *y, int M, int N, cudaStream_t stream){}
    void Hgemv_Phase1_Transpose_driver(cuDoubleComplex* A, cuDoubleComplex *x, cuDoubleComplex *y, int M, int N, cudaStream_t stream){}


    void Hgemv_Phase3_driver(cuHalfComplex* A, cuHalfComplex *x, cuHalfComplex *y, int M, int N, cudaStream_t stream){
        int val = M / 16;
        if(M%16 != 0) val++;
        dim3 dimGrid(1,val);
        dim3 dimBlock(32,4);
        Hgemv_v3<cuHalfComplex><<<dimGrid,dimBlock,0,stream>>>(A,x,y,M,N);
//        int val = M / 4;
//        if(M%4 != 0) val++;
//        dim3 dimGrid(1,val);
//        dim3 dimBlock(32,4);
//        Hgemv_v1<<<dimGrid,dimBlock,0,stream>>>(A,x,y,M,N);
        CUDACHECK(cudaGetLastError());
    }

    void Hgemv_Phase3_driver(cubfComplex* A, cubfComplex *x, cubfComplex *y, int M, int N, cudaStream_t stream){
        int val = M / 16;
        if(M%16 != 0) val++;
        dim3 dimGrid(1,val);
        dim3 dimBlock(32,4);
        Hgemv_v3<cubfComplex><<<dimGrid,dimBlock,0,stream>>>(A,x,y,M,N);
//        int val = M / 4;
//        if(M%4 != 0) val++;
//        dim3 dimGrid(1,val);
//        dim3 dimBlock(32,4);
//        Hgemv_v1<<<dimGrid,dimBlock,0,stream>>>(A,x,y,M,N);
        CUDACHECK(cudaGetLastError());
    }
    void Hgemv_Phase3_driver(float* A, float *x, float *y, int M, int N, cudaStream_t stream){}
    void Hgemv_Phase3_driver(double* A, double *x, double *y, int M, int N, cudaStream_t stream){}
    void Hgemv_Phase3_driver(cuDoubleComplex* A, cuDoubleComplex *x, cuDoubleComplex *y, int M, int N, cudaStream_t stream){}

    void Hgemv_Phase3_Transpose_driver(cuHalfComplex* A, cuHalfComplex *x, cuHalfComplex *y, int M, int N, cudaStream_t stream){
        int val = M / 16;
        if(M%16 != 0) val++;
        dim3 dimGrid(1,val);
        dim3 dimBlock(32,4);
        Hgemv_T_v3<cuHalfComplex><<<dimGrid,dimBlock,0,stream>>>(A,x,y,M,N);
        CUDACHECK(cudaGetLastError());
    }
    void Hgemv_Phase3_Transpose_driver(cubfComplex* A, cubfComplex *x, cubfComplex *y, int M, int N, cudaStream_t stream){
        int val = M / 16;
        if(M%16 != 0) val++;
        dim3 dimGrid(1,val);
        dim3 dimBlock(32,4);
        Hgemv_T_v3<cubfComplex><<<dimGrid,dimBlock,0,stream>>>(A,x,y,M,N);
        CUDACHECK(cudaGetLastError());
    }
    void Hgemv_Phase3_Transpose_driver(float* A, float *x, float *y, int M, int N, cudaStream_t stream){}
    void Hgemv_Phase3_Transpose_driver(double* A, double *x, double *y, int M, int N, cudaStream_t stream){}
    void Hgemv_Phase3_Transpose_driver(cuDoubleComplex* A, cuDoubleComplex *x, cuDoubleComplex *y, int M, int N, cudaStream_t stream){}



//    __global__ void phase2_getmaxx_kernel_V2(cuHalfComplex *p3x, cuComplex *middlebuffer,size_t totallength){
//        unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
//        tid *= 4;
//        cuComplex regval[4];
//        if(tid < totallength) {
//            cuHalfComplex tmpval[4];
//            *reinterpret_cast<float4*>(tmpval) = *reinterpret_cast<float4*>(&p3x[tid*4]);
//#pragma unroll
//            for(int i=0; i<4; i++) {
//                regval[i].x = abs(float(tmpval[i].x));
//                regval[i].y = abs(float(tmpval[i].y));
//            }
//        }
//        const int laneId = threadIdx.x % 32;
//        const int warpId = threadIdx.x / 32;
//        static __shared__ cuComplex warpsum[16];
//        for(int i=0; i<4; i++){
//            regval[i].x = warpReduceMax<32>(regval[i].x);
//            regval[i].y = warpReduceMax<32>(regval[i].y);
//            if(laneId == 0) warpsum[warpId + i * 4] = regval[i];
//        }
//        __syncthreads();
//        for(int i=0; i<4; i++){
//            if(threadIdx.x < 4) regval[i] = warpsum[laneId + i * 4];
//            __syncthreads();
//            if (warpId == 0) {
//                regval[i].x = warpReduceMax<4>(regval[i].x);
//                regval[i].y = warpReduceMax<4>(regval[i].y);
//            }
//        }
//        if(threadIdx.x == 0){
//            reinterpret_cast<float4*>(middlebuffer)[blockIdx.x*2] = reinterpret_cast<float4*>(regval)[0];
//            reinterpret_cast<float4*>(middlebuffer)[blockIdx.x*2+1] = reinterpret_cast<float4*>(regval)[1];
//        }
//    }

    __global__ void phase2_getmaxx_kernel(cuHalfComplex *p3x, cuComplex *middlebuffer,size_t totallength)
    {
        unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
        cuComplex regval; regval.x = regval.y = 0;
        if(tid < totallength) {
            cuHalfComplex tmpval = p3x[tid];
            regval.x = abs((float)tmpval.x); regval.y = abs((float)tmpval.y);
        }
        static __shared__ cuComplex warpsum[4];
        regval.x = warpReduceMax<32>(regval.x);
        regval.y = warpReduceMax<32>(regval.y);

        const int laneId = threadIdx.x % 32;
        const int warpId = threadIdx.x / 32;
        if(laneId == 0) warpsum[warpId] = regval;
        __syncthreads();
        if(threadIdx.x < 4) regval = warpsum[laneId];
        __syncthreads();
        if (warpId == 0) {
            regval.x = warpReduceMax<4>(regval.x);
            regval.y = warpReduceMax<4>(regval.y);
        }
        if(threadIdx.x == 0) middlebuffer[blockIdx.x] = regval;
    }

    __global__ void phase2_maxxsave_kernel(cuComplex *middlebuffer, cuComplex *finalbuffer,
                                           int batchidx, size_t totallength){
        unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
        cuComplex regval; regval.x = regval.y = 0;
        if(tid < totallength) regval = middlebuffer[tid];
        static __shared__ cuComplex warpsum[4];
        regval.x = warpReduceMax<32>(regval.x);
        regval.y = warpReduceMax<32>(regval.y);

        const int laneId = threadIdx.x % 32;
        const int warpId = threadIdx.x / 32;
        if(laneId == 0) warpsum[warpId] = regval;
        __syncthreads();
        if(threadIdx.x < 4) regval = warpsum[laneId];
        __syncthreads();
        if (warpId == 0) {
            regval.x = warpReduceMax<4>(regval.x);
            regval.y = warpReduceMax<4>(regval.y);
        }
        if (threadIdx.x == 0) finalbuffer[batchidx] = regval;
    }


    __global__ void phase2_xtransform_kernel(cuHalfComplex * p3x, cuComplex * maxinfo, int batchidx,
                                             cuInt8Complex * output, size_t totallength){
        // Block index
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        if(tid < totallength){
            auto real = (int8_t)((float)p3x[tid].x / maxinfo[batchidx].x * 125.0);
            auto imag = (int8_t)((float)p3x[tid].y / maxinfo[batchidx].y * 125.0);
            output[tid] = cuInt8Complex(real, imag);
        }
    }

    __global__ void phase2_xtransform_kernel_V2(cuHalfComplex * p3x, cuComplex * maxinfo, int batchidx,
                                             cuInt8Complex * output, size_t totallength){
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        if(tid * 8 < totallength){
            cuHalfComplex halfbuffer[8];
            cuInt8Complex intbuffer[8];
            reinterpret_cast<float4*>(halfbuffer)[0] = reinterpret_cast<float4*>(p3x)[tid*2];
            reinterpret_cast<float4*>(halfbuffer)[1] = reinterpret_cast<float4*>(p3x)[tid*2+1];
#pragma unroll
            for(int i=0; i<8; i++){
                auto real = (int8_t)((float)halfbuffer[i].x / maxinfo[batchidx].x * 125.0);
                auto imag = (int8_t)((float)halfbuffer[i].y / maxinfo[batchidx].y * 125.0);
                intbuffer[i] = cuInt8Complex(real, imag);
            }
            reinterpret_cast<float4*>(output)[tid] = *reinterpret_cast<float4*>(intbuffer);
        }
    }


    void phase2_getmaxx_driver(cuHalfComplex * p3x,cuComplex * middlebuffer,
                               cuComplex * maxinfo, int batchidx,
                               cuInt8Complex * output, size_t totallength, cudaStream_t stream){
        int val = (totallength + 127) / 128;

        dim3 dimGrid(val);
        dim3 dimBlock(128);
        phase2_getmaxx_kernel<<<dimGrid,dimBlock,128,stream>>>(p3x, middlebuffer, totallength);
        dimGrid = dim3(val);
        phase2_maxxsave_kernel<<<1,128,32,stream>>>(middlebuffer, maxinfo, batchidx, totallength);
//        int v1 = (val + 7) / 8;
//        dimGrid = dim3(v1);
        phase2_xtransform_kernel<<<dimGrid,dimBlock,0,stream>>>(p3x, maxinfo, batchidx, output, totallength);
    }


    __global__ void half2singlekernel(const cuHalfComplex * cuhalfvec, cuComplex * cucomplexvec, size_t length){
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < length){
            cucomplexvec[idx].x = (float)cuhalfvec[idx].x;
            cucomplexvec[idx].y = (float)cuhalfvec[idx].y;
        }
    }


    __device__ __forceinline__ cuHalfComplex convertI82half(int* midres, float4 f4){
        return cuHalfComplex(midres[0] * f4.x / 15625.0 - midres[3] * f4.w / 15625.0 ,
                             midres[1] * f4.y / 15625.0 + midres[2] * f4.z / 15625.0);
    }

    __device__ __forceinline__ cuHalfComplex convertlonglong2half(long long* midres, float4 f4){
        return cuHalfComplex(midres[0] * f4.x / 15625.0 - midres[3] * f4.w / 15625.0 ,
                             midres[1] * f4.y / 15625.0 + midres[2] * f4.z / 15625.0);
    }


//    __global__ void Igemv_phase1_v1(
//            cuInt8Complex * __restrict__ A, cuInt8Complex * __restrict__ x, cuComplex maxA, cuComplex maxx,
//            cuHalfComplex * __restrict__ y,const int M,const int N) {
//        // Block index
//        int by = blockIdx.y;
//        // Thread index
//        int tx = threadIdx.x;
//        int ty = threadIdx.y;
//        const int warp_size=32;
//        int laneId= tx % warp_size;
//        int current_row = (blockDim.y * by + ty) * 8;
//        if(current_row < M){
//            int midres[8][2];
//            for(int i=0; i<8; i++){
//                for(int j=0; j<2; j++){
//                    midres[i][j] = 0;
//                }
//            }
//            float4 f4x = reinterpret_cast<float4*>(x)[laneId];
//            auto out_x = reinterpret_cast<cuInt8Complex*>(&f4x);
//#pragma unroll
//            for(int col=0; col<8; col++){
//                float4 current_val = reinterpret_cast<float4 *>(A)[ (laneId*8+col) * M * 2 / 16 + current_row/8];
//                auto current_A = reinterpret_cast<cuInt8Complex*>(&current_val);
//                cuInt8Complex current_x = out_x[col];
//#pragma unroll
//                for(int row=0; row<8; row++){
//                    midres[row][0] += (int)current_A[row].x * (int)current_x.x - (int)current_A[row].y * (int)current_x.y;
//                    midres[row][1] += (int)current_A[row].x * (int)current_x.y + (int)current_A[row].y * (int)current_x.x;
//                }
//            }
//#pragma unroll
//            for(int row=0; row<8; row++){
//                midres[row][0] = warpReduceSumInt<warp_size>(midres[row][0]);
//                midres[row][1] = warpReduceSumInt<warp_size>(midres[row][1]);
////                midres[row][2] = warpReduceSumInt<warp_size>(midres[row][2]);
////                midres[row][3] = warpReduceSumInt<warp_size>(midres[row][3]);
//            }
//
//            if(laneId==0) {
//                cuHalfComplex results[4];
//                float4 f4 = {maxA.x * maxx.x,maxA.x * maxx.y,maxA.y * maxx.x,maxA.y * maxx.y};
//                results[0] = convertI82half(midres[0], f4);
//                results[1] = convertI82half(midres[1], f4);
//                results[2] = convertI82half(midres[2], f4);
//                results[3] = convertI82half(midres[3], f4);
//                reinterpret_cast<float4 *>(y)[current_row/4] = *reinterpret_cast<float4 *>(&results);
//                results[0] = convertI82half(midres[4], f4);
//                results[1] = convertI82half(midres[5], f4);
//                results[2] = convertI82half(midres[6], f4);
//                results[3] = convertI82half(midres[7], f4);
//                reinterpret_cast<float4 *>(y)[current_row/4+1] = *reinterpret_cast<float4 *>(&results);
//            }
//        }
//    }


    __global__ void Igemv_phase1_v1(
            cuInt8Complex * __restrict__ A, cuInt8Complex * __restrict__ x, cuComplex maxA, cuComplex maxx,
            cuHalfComplex * __restrict__ y,const int M,const int N) {
        // Block index
        int by = blockIdx.y;
        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        const int warp_size=32;
        int laneId= tx % warp_size;
        int current_row = (blockDim.y * by + ty) * 8;
        if(current_row < M){
            int midres[8][4];
            for(int i=0; i<8; i++){
                for(int j=0; j<4; j++){
                    midres[i][j] = 0;
                }
            }
            float4 f4x = reinterpret_cast<float4*>(x)[laneId];
            auto out_x = reinterpret_cast<cuInt8Complex*>(&f4x);
#pragma unroll
            for(int col=0; col<8; col++){
                float4 current_val = reinterpret_cast<float4 *>(A)[ (laneId*8+col) * M * 2 / 16 + current_row/8];
                auto current_A = reinterpret_cast<cuInt8Complex*>(&current_val);
                cuInt8Complex current_x = out_x[col];
#pragma unroll
                for(int row=0; row<8; row++){
                    midres[row][0] += (int)current_A[row].x * (int)current_x.x;
                    midres[row][1] += (int)current_A[row].x * (int)current_x.y;
                    midres[row][2] += (int)current_A[row].y * (int)current_x.x;
                    midres[row][3] += (int)current_A[row].y * (int)current_x.y;
                }
            }
#pragma unroll
            for(int row=0; row<8; row++){
                midres[row][0] = warpReduceSumInt<warp_size>(midres[row][0]);
                midres[row][1] = warpReduceSumInt<warp_size>(midres[row][1]);
                midres[row][2] = warpReduceSumInt<warp_size>(midres[row][2]);
                midres[row][3] = warpReduceSumInt<warp_size>(midres[row][3]);
            }
//            if(laneId==0) {
//                cuHalfComplex results[4];
//                float4 f4 = {maxA.x * maxx.x,maxA.x * maxx.y,maxA.y * maxx.x,maxA.y * maxx.y};
//                results[0] = convertI82half(midres[0], f4);
//                results[1] = convertI82half(midres[1], f4);
//                results[2] = convertI82half(midres[2], f4);
//                results[3] = convertI82half(midres[3], f4);
//                reinterpret_cast<float4 *>(y)[current_row/4] = *reinterpret_cast<float4 *>(&results);
//                results[0] = convertI82half(midres[4], f4);
//                results[1] = convertI82half(midres[5], f4);
//                results[2] = convertI82half(midres[6], f4);
//                results[3] = convertI82half(midres[7], f4);
//                if(current_row == 0 && laneId == 0 && tx/warp_size == 0) printf(" 0 %d %d \n", midres[0][0], midres[0][1]);
//                reinterpret_cast<float4 *>(y)[current_row/4+1] = *reinterpret_cast<float4 *>(&results);
//            }
            __shared__ int sm[4][8][4];
            if(laneId == 0){
                for(int i=0; i<8; i++){
                    for(int j=0; j<4; j++){
                        sm[threadIdx.y][i][j] = midres[i][j];
                    }
                }
            }
            __syncthreads();
            if(laneId < 8) {
                cuHalfComplex fres;
                float4 f4 = {maxA.x * maxx.x,maxA.x * maxx.y,maxA.y * maxx.x,maxA.y * maxx.y};
                fres = convertI82half(sm[threadIdx.y][laneId], f4);
                y[current_row + laneId] = fres;
            }
        }
    }
//
//    __global__ void Igemv_phase1_v1(
//            cuInt8Complex * __restrict__ A, cuInt8Complex * __restrict__ x, cuComplex maxA, cuComplex maxx,
//            cuHalfComplex * __restrict__ y,const int M,const int N) {
//        // Block index
//        int by = blockIdx.y;
//        // Thread index
//        int tx = threadIdx.x;
//        int ty = threadIdx.y;
//        const int warp_size=32;
//        int laneId= tx % warp_size;
//        int current_row = (blockDim.y * by + ty) * 8;
//        if(current_row < M){
//            int midres[8][4];
//            for(int i=0; i<8; i++){
//                for(int j=0; j<4; j++){
//                    midres[i][j] = 0;
//                }
//            }
//            float4 f4x = reinterpret_cast<float4*>(x)[laneId];
//            auto out_x = reinterpret_cast<cuInt8Complex*>(&f4x);
//#pragma unroll
//            for(int col=0; col<8; col++){
//                float4 current_val = reinterpret_cast<float4 *>(A)[ (laneId*8+col) * M * 2 / 16 + current_row/8];
//                auto current_A = reinterpret_cast<cuInt8Complex*>(&current_val);
//                cuInt8Complex current_x = out_x[col];
//#pragma unroll
//                for(int row=0; row<8; row++){
//                    midres[row][0] += (int)current_A[row].x * (int)current_x.x;
//                    midres[row][1] += (int)current_A[row].x * (int)current_x.y;
//                    midres[row][2] += (int)current_A[row].y * (int)current_x.x;
//                    midres[row][3] += (int)current_A[row].y * (int)current_x.y;
//                }
//            }
//#pragma unroll
//            for(int row=0; row<8; row++){
//                midres[row][0] = warpReduceSumInt<warp_size>(midres[row][0]);
//                midres[row][1] = warpReduceSumInt<warp_size>(midres[row][1]);
//                midres[row][2] = warpReduceSumInt<warp_size>(midres[row][2]);
//                midres[row][3] = warpReduceSumInt<warp_size>(midres[row][3]);
//            }
//
//            if(laneId==0) {
//                cuHalfComplex results[4];
//                float4 f4 = {maxA.x * maxx.x,maxA.x * maxx.y,maxA.y * maxx.x,maxA.y * maxx.y};
//                results[0] = convertI82half(midres[0], f4);
//                results[1] = convertI82half(midres[1], f4);
//                results[2] = convertI82half(midres[2], f4);
//                results[3] = convertI82half(midres[3], f4);
//                reinterpret_cast<float4 *>(y)[current_row/4] = *reinterpret_cast<float4 *>(&results);
//                results[0] = convertI82half(midres[4], f4);
//                results[1] = convertI82half(midres[5], f4);
//                results[2] = convertI82half(midres[6], f4);
//                results[3] = convertI82half(midres[7], f4);
//                reinterpret_cast<float4 *>(y)[current_row/4+1] = *reinterpret_cast<float4 *>(&results);
//            }
//        }
//    }
//
    __global__ void Igemv_phase1_T_v1(
            cuInt8Complex * __restrict__ A, cuInt8Complex * __restrict__ x, cuComplex maxA, cuComplex maxx,
            cuHalfComplex * __restrict__ y,const int M,const int N) {
        // Block index
        int by = blockIdx.y;
        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        const int warp_size=32;
        int laneId= tx % warp_size;
        int current_row = (blockDim.y * by + ty) * 8;
        if(current_row < M){
            int midres[8][4];
            for(int i=0; i<8; i++){
                for(int j=0; j<4; j++){
                    midres[i][j] = 0;
                }
            }
            float4 f4x = reinterpret_cast<float4*>(x)[laneId];
            auto out_x = reinterpret_cast<cuInt8Complex*>(&f4x);
#pragma unroll
            for(int row=0; row<8; row++){
                float4 current_val = reinterpret_cast<float4 *>(A)[(current_row + row) * N * 2 / 16 + laneId];
                auto current_A = reinterpret_cast<cuInt8Complex*>(&current_val);
#pragma unroll
                for(int col=0; col<8; col++){
                    cuInt8Complex current_x = out_x[col];
                    midres[row][0] += (int)current_A[col].x * (int)current_x.x;
                    midres[row][1] += (int)current_A[col].x * (int)current_x.y;
                    midres[row][2] += (int)current_A[col].y * (int)current_x.x;
                    midres[row][3] += (int)current_A[col].y * (int)current_x.y;
                }
            }
#pragma unroll
            for(int row=0; row<8; row++){
                midres[row][0] = warpReduceSumInt<warp_size>(midres[row][0]);
                midres[row][1] = warpReduceSumInt<warp_size>(midres[row][1]);
                midres[row][2] = warpReduceSumInt<warp_size>(midres[row][2]);
                midres[row][3] = warpReduceSumInt<warp_size>(midres[row][3]);
            }

            if(laneId==0) {
                cuHalfComplex results[4];
                float4 f4 = {maxA.x * maxx.x,maxA.x * maxx.y,maxA.y * maxx.x, maxA.y * maxx.y};
                results[0] = convertI82half(midres[0], f4);
                results[1] = convertI82half(midres[1], f4);
                results[2] = convertI82half(midres[2], f4);
                results[3] = convertI82half(midres[3], f4);
                reinterpret_cast<float4 *>(y)[current_row/4] = *reinterpret_cast<float4 *>(&results);
                results[0] = convertI82half(midres[4], f4);
                results[1] = convertI82half(midres[5], f4);
                results[2] = convertI82half(midres[6], f4);
                results[3] = convertI82half(midres[7], f4);
                reinterpret_cast<float4 *>(y)[current_row/4+1] = *reinterpret_cast<float4 *>(&results);
            }
        }
    }



    void Igemv_Phase1_driver(cuInt8Complex* A, cuInt8Complex *x,cuComplex maxA, cuComplex maxx,
                             cuHalfComplex *y, int M, int N, cudaStream_t stream){
//        int val = M / 32;
//        if(M%32 != 0) val++;
//        dim3 dimGrid(1,val);
//        dim3 dimBlock(32,4);
//        Igemv_phase1_v1<<<dimGrid,dimBlock,0,stream>>>(A,x,maxA,maxx,y,M,N);
//        CUDACHECK(cudaGetLastError());

        int val = M / 32;
        if(M%32 != 0) val++;
        dim3 dimGrid(1,val);
        dim3 dimBlock(32,4);
        Igemv_phase1_v1<<<dimGrid,dimBlock,512,stream>>>(A,x,maxA,maxx,y,M,N);
        CUDACHECK(cudaGetLastError());
    }


    void Igemv_Phase1_Transpose_driver(cuInt8Complex* A, cuInt8Complex *x,cuComplex maxA, cuComplex maxx,
                             cuHalfComplex *y, int M, int N, cudaStream_t stream){
        int val = M / 32;
        if(M%32 != 0) val++;
        dim3 dimGrid(1,val);
        dim3 dimBlock(32,4);
        Igemv_phase1_T_v1<<<dimGrid,dimBlock,0,stream>>>(A,x,maxA,maxx,y,M,N);
        CUDACHECK(cudaGetLastError());
    }


    __device__ __forceinline__ cuComplex midcompute(cuInt8Complex A, cuComplex maxA, cuHalfComplex x){
        cuComplex Acomplex,xcomplex,res;
        Acomplex.x = (float)A.x * maxA.x / 125.0;
        Acomplex.y = (float)A.y * maxA.y / 125.0;
        xcomplex.x = (float)x.x;
        xcomplex.y = (float)x.y;
        res.x = Acomplex.x * xcomplex.x - Acomplex.y * xcomplex.y;
        res.y = Acomplex.x * xcomplex.y + Acomplex.y * xcomplex.x;
        return res;
    }

    __global__ void Igemv_Phase3_v1(
            cuInt8Complex * __restrict__ A, cuHalfComplex * __restrict__ x, cuComplex maxA,
            cuHalfComplex * __restrict__ y,const int M,const int N) {
        // Block index
        int by = blockIdx.y;
        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        const int warp_size=32;
        int laneId= tx % warp_size;
        int current_row = (blockDim.y * by + ty) * 8;
        if(current_row < M){
            cuComplex midres[8];
#pragma unroll
            for(int i=0; i<8; i++){
                midres[i].x = 0.0; midres[i].y = 0.0;
            }
            int kIteration = (N / (warp_size * 8));
            if(N % (warp_size * 8) != 0) kIteration++;

            for(int i=0; i<kIteration; i++){
                if((i*warp_size + laneId) * 8 < N){
                    int xoffset = i*warp_size + laneId;
                    auto f4x = reinterpret_cast<float4*>(x)[xoffset * 2];
                    auto out_x = reinterpret_cast<cuHalfComplex*>(&f4x);
#pragma unroll
                    for(int col=0; col<4; col++){
                        float4 current_val = reinterpret_cast<float4 *>(A)
                                [((i*warp_size+laneId)*8+col) * M * 2 / 16 + current_row/8];
                        auto *current_A = reinterpret_cast<cuInt8Complex*>(&current_val);
                        cuHalfComplex current_x = out_x[col];
#pragma unroll
                        for(int row=0; row<8; row++){
                            auto m = midcompute(current_A[row], maxA, current_x);
                            midres[row].x += m.x;
                            midres[row].y += m.y;
                        }
                    }
                    f4x = reinterpret_cast<float4*>(x)[xoffset * 2 + 1];
                    out_x = reinterpret_cast<cuHalfComplex*>(&f4x);
#pragma unroll
                    for(int col=4; col<8; col++){
                        float4 current_val = reinterpret_cast<float4 *>(A)
                        [((i*warp_size+laneId)*8+col) * M * 2 / 16 + current_row/8];
                        auto *current_A = reinterpret_cast<cuInt8Complex*>(&current_val);
                        cuHalfComplex current_x = out_x[col-4];
#pragma unroll
                        for(int row=0; row<8; row++){
                            auto m = midcompute(current_A[row], maxA, current_x);
                            midres[row].x += m.x;
                            midres[row].y += m.y;
                        }
                    }
                }
            }
#pragma unroll
            for(int row=0; row<8; row++){
                midres[row].x = warpReduceSum<warp_size>(midres[row].x);
                midres[row].y = warpReduceSum<warp_size>(midres[row].y);
            }

            if(laneId==0) {
                cuHalfComplex results[4];
                results[0] = cuHalfComplex(midres[0].x, midres[0].y);
                results[1] = cuHalfComplex(midres[1].x, midres[1].y);
                results[2] = cuHalfComplex(midres[2].x, midres[2].y);
                results[3] = cuHalfComplex(midres[3].x, midres[3].y);
                reinterpret_cast<float4 *>(y)[current_row/4] = *reinterpret_cast<float4*>(&results);
                results[0] = cuHalfComplex(midres[4].x, midres[4].y);
                results[1] = cuHalfComplex(midres[5].x, midres[5].y);
                results[2] = cuHalfComplex(midres[6].x, midres[6].y);
                results[3] = cuHalfComplex(midres[7].x, midres[7].y);
                reinterpret_cast<float4 *>(y)[current_row/4+1] = *reinterpret_cast<float4*>(&results);
            }
        }
    }

    // use cuhalfcomplex as internal computation
    __global__ void Igemv_Phase3_v2(
            cuInt8Complex * __restrict__ A, cuHalfComplex * __restrict__ x, cuComplex maxA,
            cuHalfComplex * __restrict__ y,const int M,const int N) {
        // Block index
        int by = blockIdx.y;
        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        const int warp_size=32;
        int laneId= tx % warp_size;
        int current_row = (blockDim.y * by + ty) * 8;
        if(current_row < M){
            cuHalfComplex midres[8];
#pragma unroll
            for(int i=0; i<8; i++){
                midres[i].x = 0.0; midres[i].y = 0.0;
            }
            int kIteration = (N / (warp_size * 8));
            if(N % (warp_size * 8) != 0) kIteration++;

            for(int i=0; i<kIteration; i++){
                if((i*warp_size + laneId) * 8 < N){
                    int xoffset = i*warp_size + laneId;
                    auto f4x = reinterpret_cast<float4*>(x)[xoffset * 2];
                    auto out_x = reinterpret_cast<cuHalfComplex*>(&f4x);
#pragma unroll
                    for(int col=0; col<4; col++){
                        float4 current_val = reinterpret_cast<float4 *>(A)
                        [((i*warp_size+laneId)*8+col) * M * 2 / 16 + current_row/8];
                        auto *current_A = reinterpret_cast<cuInt8Complex*>(&current_val);
                        cuHalfComplex current_x = out_x[col];
#pragma unroll
                        for(int row=0; row<8; row++){
                            auto m = midcompute(current_A[row], maxA, current_x);
                            midres[row].x += m.x;
                            midres[row].y += m.y;
                        }
                    }
                    f4x = reinterpret_cast<float4*>(x)[xoffset * 2 + 1];
                    out_x = reinterpret_cast<cuHalfComplex*>(&f4x);
#pragma unroll
                    for(int col=4; col<8; col++){
                        float4 current_val = reinterpret_cast<float4 *>(A)
                        [((i*warp_size+laneId)*8+col) * M * 2 / 16 + current_row/8];
                        auto *current_A = reinterpret_cast<cuInt8Complex*>(&current_val);
                        cuHalfComplex current_x = out_x[col-4];
#pragma unroll
                        for(int row=0; row<8; row++){
                            auto m = midcompute(current_A[row], maxA, current_x);
                            midres[row].x += m.x;
                            midres[row].y += m.y;
                        }
                    }
                }
            }
#pragma unroll
            for(int row=0; row<8; row++){
                midres[row].x = warpReduceSum<warp_size>(midres[row].x);
                midres[row].y = warpReduceSum<warp_size>(midres[row].y);
            }

            if(laneId==0) {
                reinterpret_cast<float4 *>(y)[current_row/4] = *reinterpret_cast<float4*>(&midres);
                reinterpret_cast<float4 *>(y)[current_row/4+1] = *reinterpret_cast<float4*>(&midres[4]);
            }
        }
    }

    // use cuhalfcomplex as internal computation and 4 elements
    __global__ void Igemv_Phase3_v3(
            cuInt8Complex * __restrict__ A, cuHalfComplex * __restrict__ x, cuComplex maxA,
            cuHalfComplex * __restrict__ y,const int M,const int N) {
        // Block index
        int by = blockIdx.y;
        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        const int warp_size=32;
        int laneId= tx % warp_size;
        int current_row = (blockDim.y * by + ty) * 4;
        if(current_row < M){
            cuHalfComplex midres[4];
#pragma unroll
            for(int i=0; i<4; i++){
                midres[i].x = 0.0; midres[i].y = 0.0;
            }
            int kIteration = (N / (warp_size * 4));
            if(N % (warp_size * 4) != 0) kIteration++;

            for(int i=0; i<kIteration; i++){
                if((i*warp_size + laneId) * 4 < N){
                    auto f4x = reinterpret_cast<float4*>(x)[i*warp_size + laneId];
                    auto out_x = reinterpret_cast<cuHalfComplex*>(&f4x);
#pragma unroll
                    for(int col=0; col<4; col++){
                        float2 current_val = reinterpret_cast<float2 *>(A)
                        [((i*warp_size+laneId)*4+col) * M * 2 / 8 + current_row/4];
                        auto *current_A = reinterpret_cast<cuInt8Complex*>(&current_val);
                        cuHalfComplex current_x = out_x[col];
#pragma unroll
                        for(int row=0; row<4; row++){
                            auto m = midcompute(current_A[row], maxA, current_x);
                            midres[row].x += m.x;
                            midres[row].y += m.y;
                        }
                    }
                }
            }
#pragma unroll
            for(int row=0; row<4; row++){
                midres[row].x = warpReduceSum<warp_size>(midres[row].x);
                midres[row].y = warpReduceSum<warp_size>(midres[row].y);
            }

            if(laneId==0) {
                reinterpret_cast<float4 *>(y)[current_row/4] = *reinterpret_cast<float4*>(&midres);
            }
        }
    }



    __global__ void New_Igemv_Phase3_v1(
            cuInt8Complex * __restrict__ A, cuInt8Complex * __restrict__ x, cuComplex * maxAvec, cuComplex * maxxvec,
            cuHalfComplex * __restrict__ y,const int M,const int N,const int batchidx) {
        // Block index
        int by = blockIdx.y;
        // Thread index
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        const int warp_size=32;
        int laneId= tx % warp_size;
        int current_row = (blockDim.y * by + ty) * 8;
        cuComplex maxA = maxAvec[batchidx];
        cuComplex maxx = maxxvec[batchidx];
        if(current_row < M){
            int midres[8][4];
            for(int i=0; i<8; i++){
                for(int j=0; j<4; j++){
                    midres[i][j] = 0;
                }
            }
            int kIteration = (N / (warp_size * 8));
            if(N % (warp_size * 8) != 0) kIteration++;

            for(int i=0; i<kIteration; i++){
                if((i*warp_size + laneId) * 8 < N){
                    int xoffset = i*warp_size + laneId;
                    auto f4x = reinterpret_cast<float4*>(x)[xoffset];
                    auto out_x = reinterpret_cast<cuInt8Complex*>(&f4x);

#pragma unroll
                    for(int col=0; col<8; col++){
                        float4 current_val = reinterpret_cast<float4 *>(A)[((i*warp_size+laneId)*8+col) * M * 2 / 16 + current_row/8];
                        auto *current_A = reinterpret_cast<cuInt8Complex*>(&current_val);
                        cuInt8Complex current_x = out_x[col];
//                        if(laneId == 0 && tx == 0 && ty == 0 && current_row == 0){
//                            printf("get value %d %d %d %d\n",(int)current_A[0].x,(int)current_A[0].x, (int)current_x.x, (int)current_x.y);
//                        }
#pragma unroll
                        for(int row=0; row<8; row++){
                            midres[row][0] += (int)current_A[row].x * (int)current_x.x;
                            midres[row][1] += (int)current_A[row].x * (int)current_x.y;
                            midres[row][2] += (int)current_A[row].y * (int)current_x.x;
                            midres[row][3] += (int)current_A[row].y * (int)current_x.y;
//                            if(laneId == 0 && tx == 0 && ty == 0 && by == 0 && current_row == 0){
//                                printf("col %d %d %d %d\n",col, (int)midres[row][0], (int)current_A[row].x, (int)current_x.x);
//                            }
                        }
                    }
                }
            }
#pragma unroll
            for(int row=0; row<8; row++){
                midres[row][0] = warpReduceSumInt<warp_size>(midres[row][0]);
                midres[row][1] = warpReduceSumInt<warp_size>(midres[row][1]);
                midres[row][2] = warpReduceSumInt<warp_size>(midres[row][2]);
                midres[row][3] = warpReduceSumInt<warp_size>(midres[row][3]);
            }

//            if(laneId==0) {
//                cuHalfComplex results[4];
//                float4 f4 = {maxA.x * maxx.x,maxA.x * maxx.y,maxA.y * maxx.x, maxA.y * maxx.y};
//                results[0] = convertI82half(midres[0], f4);
//                results[1] = convertI82half(midres[1], f4);
//                results[2] = convertI82half(midres[2], f4);
//                results[3] = convertI82half(midres[3], f4);
//                reinterpret_cast<float4 *>(y)[current_row/4] = *reinterpret_cast<float4 *>(&results);
//                results[0] = convertI82half(midres[4], f4);
//                results[1] = convertI82half(midres[5], f4);
//                results[2] = convertI82half(midres[6], f4);
//                results[3] = convertI82half(midres[7], f4);
//                reinterpret_cast<float4 *>(y)[current_row/4+1] = *reinterpret_cast<float4 *>(&results);
//            }
            __shared__ int sm[4][8][4];
            if(laneId == 0){
                for(int i=0; i<8; i++){
                    for(int j=0; j<4; j++){
                        sm[threadIdx.y][i][j] = midres[i][j];
                    }
                }
            }
            __syncthreads();
            if(laneId < 8) {
                cuHalfComplex fres;
                float4 f4 = {maxA.x * maxx.x,maxA.x * maxx.y,maxA.y * maxx.x,maxA.y * maxx.y};
                fres = convertI82half(sm[threadIdx.y][laneId], f4);
                y[current_row + laneId] = fres;
            }
        }
    }

    void New_Igemv_Phase3_driver(cuInt8Complex* A, cuInt8Complex *x,cuComplex * maxA, cuComplex * maxx,
                                 cuHalfComplex *y, int M, int N, int batchidx, cudaStream_t stream){
        int val = M / 32;
        if(M%32 != 0) val++;
        dim3 dimGrid(1,val);
        dim3 dimBlock(32,4);
        New_Igemv_Phase3_v1<<<dimGrid,dimBlock,512,stream>>>(A,x,maxA,maxx,y,M,N,batchidx);
        CUDACHECK(cudaGetLastError()); // around 900 us
    }


    void Igemv_Phase3_driver(cuInt8Complex* A, cuHalfComplex *x,cuComplex maxA,
                             cuHalfComplex *y, int M, int N, cudaStream_t stream){
//        int val = M / 32;
//        if(M%32 != 0) val++;
//        dim3 dimGrid(1,val);
//        dim3 dimBlock(32,4);
//        Igemv_Phase3_v1<<<dimGrid,dimBlock,0,stream>>>(A,x,maxA,y,M,N);
//        CUDACHECK(cudaGetLastError()); // around 900 us


//        int val = M / 32;
//        if(M%32 != 0) val++;
//        dim3 dimGrid(1,val);
//        dim3 dimBlock(32,4);
//        Igemv_Phase3_v2<<<dimGrid,dimBlock,0,stream>>>(A,x,maxA,y,M,N);
//        CUDACHECK(cudaGetLastError()); // around 600 us

//        int val = M / 16;
//        if(M%16 != 0) val++;
//        dim3 dimGrid(1,val);
//        dim3 dimBlock(32,4);
//        Igemv_Phase3_v3<<<dimGrid,dimBlock,0,stream>>>(A,x,maxA,y,M,N);
//        CUDACHECK(cudaGetLastError());
    }

    __global__ void Phase3Upcasting(cuHalfComplex * outputhalf, cuInt8Complex * inputint8,
                                    cuComplex maxinfo, size_t elemoffset, size_t elems, int parti)
    {
        size_t tx = blockIdx.x * blockDim.x + threadIdx.x;
        if(tx < elems){
            size_t elemidx = elemoffset + tx;
            outputhalf[elemidx].x =
                    (half)((float)(inputint8[elemidx].x) * maxinfo.x / 125.0);
            outputhalf[elemidx].y =
                    (half)((float)(inputint8[elemidx].y) * maxinfo.y / 125.0);
        }
    }

    __global__ void Phase3UpcastingV2(cuHalfComplex * outputhalf, cuInt8Complex * inputint8,
                                    cuComplex maxinfo, size_t elemoffset, size_t elems, int parti)
    {
        size_t tx = (blockIdx.x * blockDim.x + threadIdx.x)*8;
        if(tx < elems){
            size_t elemidx = elemoffset + tx;
            auto i8_8 = reinterpret_cast<float4*>(inputint8)[elemidx/8];
            auto i8ptr = reinterpret_cast<cuInt8Complex*>(&i8_8);
            cuHalfComplex midhalf4[8];
            for(int i=0; i<8; i++){
                midhalf4[i].x = (half)((float)(i8ptr[i].x) * maxinfo.x / 125.0);
                midhalf4[i].y = (half)((float)(i8ptr[i].y) * maxinfo.y / 125.0);
            }
            reinterpret_cast<float4*>(outputhalf)[elemidx/4] = reinterpret_cast<float4*>(midhalf4)[0];
            reinterpret_cast<float4*>(outputhalf)[elemidx/4+1] = reinterpret_cast<float4*>(midhalf4)[1];
        }
    }

    void Phase3UpcastingDriver(cuHalfComplex * outputhalf, cuInt8Complex * inputint8,
                         cuComplex maxinfo, int elemoffset, int elems, int parti, cudaStream_t stream){
//        int val = elems / 256;
//        if(elems % 256 != 0) val++;
//        dim3 dimGrid(val);
//        dim3 dimBlock(256);
//        Phase3Upcasting<<<dimGrid,dimBlock,0,stream>>>(outputhalf, inputint8, maxinfo, elemoffset, elems, parti);
//        CUDACHECK(cudaGetLastError());

        int val = elems / 256 / 8;
        if(elems % 256 != 0) val++;
        dim3 dimGrid(val);
        dim3 dimBlock(256);
        Phase3UpcastingV2<<<dimGrid,dimBlock,0,stream>>>(outputhalf, inputint8, maxinfo, elemoffset, elems, parti);
        CUDACHECK(cudaGetLastError());
    }



} // namespace cudatlrmvm
