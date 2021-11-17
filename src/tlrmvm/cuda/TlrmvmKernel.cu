#include <cuda.h>
#include <cublas_v2.h>
#include <stdio.h>
#include "TlrmvmKernel.cuh"
#include "common/cuda/Util.h"
#include <iostream>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
using namespace std;
namespace cudatlrmvm {



__global__ void MergeyfinalBasic(float **y, int vectorlength, int streamsize){
    int lx = threadIdx.x + blockDim.x * blockIdx.x;
    if(lx < vectorlength){
        for(int i=1; i<streamsize; i++){
            y[0][lx] += y[i][lx];
        }
    }
}

void MergeyfinalBasicDriver(float ** y, int vectorlength, int streamsize, cudaStream_t cudastream){
    int thread_x = 256;
    int grid_x = (vectorlength+255) / 256;
    MergeyfinalBasic<<<grid_x, thread_x, 0, cudastream>>>(y, vectorlength,streamsize);
    CUDACHECK(cudaGetLastError());
}
    

__global__ void phase2(float * __restrict__ yu, float * __restrict__ yv, unsigned long int vlength, unsigned long int* __restrict__ offsetinyu){
    unsigned long int lx = blockIdx.x * blockDim.x + threadIdx.x;
    if(lx < vlength){
        unsigned long int pos = offsetinyu[lx];
        yu[lx] = yv[pos]; 
    }
}


void phase2dirver(float * yu, float *yv, 
    unsigned long int vlength, unsigned long int* offsetinyu, cudaStream_t stream){
    int thread_x = 256;
    phase2<<<(vlength+thread_x -1)/thread_x, thread_x,0, stream>>>(yu, yv, vlength, offsetinyu);
    CUDACHECK(cudaGetLastError());
}



__global__ void phase2gpuconstrankreshuffle(float*  __restrict__ y_in, float*  __restrict__ y_out, 
  int Aurows, int Aucols,
  int Avrows, int Avcols,
  int m, int n, int nt, int mt, int nb, int size){

  unsigned int threadidx = threadIdx.x + blockDim.x * blockIdx.x;
  unsigned int threadidy = threadIdx.y + blockDim.y * blockIdx.y;
  if(threadidx >= nt || threadidy >= Avrows) return;

  unsigned int out_col = threadidy / (nb/4) ;
  unsigned int out_row = threadidx;
  unsigned int k = threadidy % (nb/4);
  y_out[out_col* Aucols + out_row * (nb/4) + k] = 
  y_in[threadidx * Avrows + threadidy];
}

void phase2gpuconstrankreshuffledriver(float* y_in, float* y_out, 
int Aurows, int Aucols,
int Avrows, int Avcols,
int m, int n, int nt, int mt, int nb, int size){
    // int thread_x = 128;
    // int thread_y = 1;
    // int nbx =  nt / thread_x + (nt % thread_x != 0);
    // int nby =  Avrows;
    // dim3 dimBlock(thread_x, thread_y);
    // dim3 dimGrid(nbx, nby);
    // phase2gpuconstrankreshuffle<<<dimGrid,dimBlock>>>
    // (y_in, y_out, Aurows, Aucols, Avrows, Avcols, m, n, nt, mt, nb, size);
}

#define FULL_MASK 0xffffffff


// __global__ void phase1(
// const float  * __restrict__ Av, const float * __restrict__ x, float * __restrict__ yv, 
// const int * __restrict__ coloffset, const int * __restrict__ collda, 
//     const int nb, const size_t Ntlocal, const size_t granksum){
//         size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
//         size_t thread_y = blockDim.y * blockIdx.y + threadIdx.y;
//         __shared__ int sm[256 + 80];
//         float accval = 0;
//         if(thread_x < Ntlocal) {
//             float* sm;
//             sm[thread_x] = coloffset[thread_x];
//             sm[thread_x+Ntlocal] = collda[thread_x];
//         }
//         // int cnt = 0;
//         // while(cnt < Ntlocal-1){
//         //     if(thread_y < sm[cnt+1]) break;
//         //     cnt++;
//         // }
//         // if(blockIdx.y == 0){
//         //     for(int i=0; i < nb/blockDim.x; i++){
//         //         int offset = (i*blockDim.x + thread_x);
//         //         sm[80 + offset] = x[cnt*nb+offset];
//         //     }
//         // }
//         __syncthreads();
//         if(thread_y < granksum){
//             int cnt = 0;
//             while(cnt < Ntlocal-1){
//                 if(thread_y < coloffset[cnt+1]) break;
//                 cnt++;
//             }
//             int rowoffset = thread_y - coloffset[cnt];
//             #pragma unroll
//             for(int i=0; i<nb / blockDim.x; i++){
//                 int offset = (i*blockDim.x+thread_x);
//                 accval += Av[coloffset[cnt]*nb + rowoffset + offset * collda[cnt]] * x[cnt*nb + offset];
//             }
//         }
//         // if(thread_y < granksum){
//         //     // int cnt = 0;
//         //     // while(cnt < Ntlocal-1){
//         //     //     if(thread_y < sm[cnt+1]) break;
//         //     //     cnt++;
//         //     // }
//         //     int rowoffset = thread_y - sm[cnt];
//         //     #pragma unroll
//         //     for(int i=0; i<nb / blockDim.x; i++){
//         //         int offset = (i*blockDim.x+thread_x);
//         //         accval += Av[sm[cnt]*nb + rowoffset + offset * sm[Ntlocal+cnt]] * sm[80 + offset];
//         //     }
//         // }
//         __syncwarp();
//         if(thread_y < granksum){
//             #pragma unroll
//             for (int offset = 16; offset > 0; offset /= 2)
//             accval += __shfl_down_sync(FULL_MASK, accval, offset);
//         }
//         __syncwarp();
//         if(thread_x == 0) yv[thread_y] = accval;
// }





// void phase1driver(float * Av, float * x, float * yv, 
// int * coloffset, int * collda, int nb, size_t Ntlocal, size_t granksum){
//     int dimx = 32;
//     int dimy = 2;
//     int griddimy = granksum / dimy ;
//     dim3 block(dimx, dimy);
//     dim3 grid(1, griddimy);
//     phase1<<<grid,block, (80 + 256) * sizeof(int)>>>(Av, x, yv, coloffset, collda, nb, Ntlocal, granksum);
//     CUDACHECK(cudaGetLastError());
// }



__global__ void phase1fuse4ops_real_kernel(half * __restrict__ y1, const half * __restrict__ y2, const size_t len)
{
    size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_x < len){
        half tmp = __hsub(y1[thread_x] , y2[thread_x]);
        y1[thread_x] = tmp;
    }
    
}

__global__ void phase1fuse4ops_imag_kernel(half * __restrict__ y1, const half * __restrict__ y2, const size_t len)
{
    size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_x < len){
        half tmp = __hadd(y1[thread_x] , y2[thread_x]);
        y1[thread_x] = tmp;
    }
}

__global__ void phase1twogemv_kernel(half * y1, half * y2, const size_t len)
{
    size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_x < len){
        // float val = y2[0];
        // if(thread_x == 0) printf("%f ", val);
        half real = __hsub(y1[thread_x] , y2[thread_x+len]);
        half imag = __hadd(y1[thread_x+len], y2[thread_x]);
        y1[thread_x] = real;
        y2[thread_x] = imag;

        // y1[thread_x] = (half)1.0;
        // y2[thread_x] = (half)1.0;
    }
}

__global__ void merge_half_realimag_kernel(half * y1, half * y2, const size_t len)
{
    size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_x < len){
        half real = __hsub(y1[thread_x] , y2[thread_x+len]);
        half imag = __hadd(y1[thread_x+len], y2[thread_x]);
        y1[thread_x] = real;
        y2[thread_x] = imag;
    }
}

__global__ void merge_float_realimag_kernel(const float * y1, const float * y2, cuComplex * yout, const size_t len)
{
    size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_x < len){
        float real = y1[thread_x] - y2[thread_x + len];
        float imag = y1[thread_x+len] + y2[thread_x];
        yout[thread_x].x = real;
        yout[thread_x].y = imag;
        // y1[thread_x] = real;
        // y2[thread_x] = imag;
    }
}




/************************
* TLRMVM Utility
**************************/

// Float


__global__ void merge_float_2floatout_realimag_kernel(
    const float * __restrict__ rr_ri, const float * __restrict__ ir_ii, 
    const int * __restrict__ colrank, const int ntg, 
    float * __restrict__ realout, float * __restrict__ imagout, 
    size_t len)
{
    size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_x < len){
        int i = 0;
        size_t prefix = 0;
        size_t currank = 0;
        for(; i<ntg; i++){
            currank = colrank[i];
            if(prefix + currank > thread_x) break;
            prefix += currank;
        }
        size_t localoffset = thread_x - prefix;
        float curreal = rr_ri[2*prefix + localoffset] - ir_ii[2*prefix + currank + localoffset];
        float curimag = rr_ri[2*prefix + currank + localoffset] + ir_ii[2*prefix + localoffset];
        realout[thread_x] = curreal;
        imagout[thread_x] = curimag;
    }
}


__global__ void phase2_kernel(
    const float * __restrict__ real, const float * __restrict__ imag, 
    const int * __restrict__ colrank, const int ntg, const size_t * __restrict__ phase2mapping, 
    float * __restrict__ yuout, size_t len)
{
    size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_x < len){
        int i = 0;
        size_t prefix = 0;
        size_t currank = 0;
        for(; i<ntg; i++){
            currank = colrank[i];
            if(prefix + currank > thread_x) break;
            prefix += currank;
        }
        size_t localoffset = thread_x - prefix;
        float curreal = real[2*prefix + localoffset] - imag[2*prefix + currank + localoffset];
        float curimag = real[2*prefix + currank + localoffset] + imag[2*prefix + localoffset];
        // realout[thread_x] = curreal;
        // imagout[thread_x] = curimag;
        yuout[phase2mapping[thread_x]] = curreal;
        yuout[phase2mapping[thread_x+len]] = curimag;
    }
}

template<typename T>
__global__ void phase2_nosplit_kernel(const T * __restrict__ yv,
    const size_t * __restrict__ phase2mapping, T * __restrict__ yu, size_t len)
{
    size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_x < len){
        yu[phase2mapping[thread_x]] = yv[thread_x];
    }
}

// __global__ void phase3_merge_kernel(
//     const float * __restrict__ rr_ri, const float * __restrict__ ir_ii, 
//     const int nb, cuComplex * __restrict__ finaly, size_t len)
// {
//     size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
//     if (thread_x < len){
//         int i = 0;
//         size_t prefix = (thread_x / nb) * 2 * nb;
//         size_t currank = 0;
//         size_t localoffset = thread_x % nb;
//         float curreal = rr_ri[prefix + localoffset] - ir_ii[prefix + nb + localoffset];
//         float curimag = rr_ri[prefix + nb + localoffset] + ir_ii[prefix + localoffset];
//         finaly[thread_x].x = curreal;
//         finaly[thread_x].y = curimag;
//     }
// }

// DRIVER
void merge_float_2floatout_realimag(
const float *rr_ri, const float *ir_ii, const int * colrank, const int ntg,
float *realout, float *imagout,  size_t len, cudaStream_t stream){
int dimx = 512;
int griddimx = (len+dimx-1) / dimx;
merge_float_2floatout_realimag_kernel<<<griddimx, dimx, 0, stream>>>(rr_ri, ir_ii, colrank, ntg, realout, imagout, len);
CUDACHECK(cudaGetLastError());
}


template<typename T>
void phase2_nosplit(const T *yv, const size_t * phase2mapping, 
T * yu, size_t len, cudaStream_t stream){
    int dimx = 512;
    int griddimx = (len+dimx-1) / dimx;
    phase2_nosplit_kernel<<<griddimx, dimx, 0, stream>>>(yv, phase2mapping, yu, len);
    CUDACHECK(cudaGetLastError());
}

template void phase2_nosplit<float>(const float*, const size_t *, float *, size_t, cudaStream_t);
template void phase2_nosplit<double>(const double*, const size_t *, double *, size_t, cudaStream_t);
template void phase2_nosplit<cuDoubleComplex>
(const cuDoubleComplex*, const size_t *, cuDoubleComplex *, size_t, cudaStream_t);
template void phase2_nosplit<cuComplex>
(const cuComplex*, const size_t *, cuComplex *, size_t, cudaStream_t);

// void phase2_complex(const cuComplex *yv, const size_t * phase2mapping, 
//     cuComplex * yu, size_t len, cudaStream_t stream){
//         int dimx = 512;
//         int griddimx = (len+dimx-1) / dimx;
//         phase2_complex_kernel<<<griddimx, dimx, 0, stream>>>(yv, phase2mapping, yu, len);
//         CUDACHECK(cudaGetLastError());
//     }

void phase2(const float *real, const float *imag, const int *colrank, const int ntg,
const size_t * phase2mapping, float * yuout, size_t len, cudaStream_t stream){
    int dimx = 512;
    int griddimx = (len+dimx-1) / dimx;
    phase2_kernel<<<griddimx, dimx, 0, stream>>>(real, imag, colrank, ntg, phase2mapping, yuout, len);
    CUDACHECK(cudaGetLastError());
}
    
// void phase3_merge(const float *rr_ri, const float *ir_ii,
// const int nb, cuComplex *finaly, size_t len, cudaStream_t stream){
//     int dimx = 512;
//     int griddimx = (len+dimx-1) / dimx;
//     phase3_merge_kernel<<<griddimx, dimx, 0, stream>>>(rr_ri, ir_ii, nb, finaly, len);
//     CUDACHECK(cudaGetLastError());
// }




// Half

__global__ void merge_half_2halfout_realimag_kernel(
    const half * __restrict__ rr_ri, const half * __restrict__ ir_ii, 
    const int * __restrict__ colrank, const int ntg, 
    half * __restrict__ realout, half * __restrict__ imagout, 
    size_t len)
{
    size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_x < len){
        int i = 0;
        size_t prefix = 0;
        size_t currank = 0;
        for(; i<ntg; i++){
            currank = colrank[i];
            if(prefix + currank > thread_x) break;
            prefix += currank;
        }
        size_t localoffset = thread_x - prefix;
        half curreal = __hsub( rr_ri[2*prefix + localoffset] , ir_ii[2*prefix + currank + localoffset] );
        half curimag = __hadd( rr_ri[2*prefix + currank + localoffset] , ir_ii[2*prefix + localoffset] );
        realout[thread_x] = curreal;
        imagout[thread_x] = curimag;
    }
}

__global__ void phase2_half_kernel(
    const half * __restrict__ real, const half * __restrict__ imag, 
    const int * __restrict__ colrank, const int ntg, const size_t * __restrict__ phase2mapping, 
    half * __restrict__ yuout, size_t len)
{
    size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_x < len){
        int i = 0;
        size_t prefix = 0;
        size_t currank = 0;
        for(; i<ntg; i++){
            currank = colrank[i];
            if(prefix + currank > thread_x) break;
            prefix += currank;
        }
        size_t localoffset = thread_x - prefix;
        half curreal = __hsub( real[2*prefix + localoffset] , imag[2*prefix + currank + localoffset] );
        half curimag = __hadd( real[2*prefix + currank + localoffset] , imag[2*prefix + localoffset] );
        yuout[phase2mapping[thread_x]] = curreal;
        yuout[phase2mapping[thread_x+len]] = curimag;
        
    }
}

__global__ void phase3_merge_half_kernel(
    const half * __restrict__ rr_ri, const half * __restrict__ ir_ii, 
    const int nb, cuComplex * __restrict__ finaly, size_t len)
{
    size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_x < len){
        size_t prefix = (thread_x / nb) * 2 * nb;
        size_t localoffset = thread_x % nb;
        half curreal = __hsub( rr_ri[prefix + localoffset] , ir_ii[prefix + nb + localoffset] );
        half curimag = __hadd( rr_ri[prefix + nb + localoffset] , ir_ii[prefix + localoffset] );
        finaly[thread_x].x = __half2float(curreal);
        finaly[thread_x].y = __half2float(curimag);
        float tmp = __half2float(curreal);
    }
}

void merge_half_2halfout_realimag(
const half *rr_ri, const half *ir_ii, const int * colrank, const int ntg,
half *realout, half *imagout,  size_t len, cudaStream_t stream){
    int dimx = 512;
    int griddimx = (len+dimx-1) / dimx;
    merge_half_2halfout_realimag_kernel<<<griddimx, dimx, 0, stream>>>
    (rr_ri, ir_ii, colrank, ntg, realout, imagout, len);
    CUDACHECK(cudaGetLastError());
}


void phase2_half(const half *real, const half *imag, const int *colrank, const int ntg,
const size_t * phase2mapping, half * yuout, size_t len, cudaStream_t stream){
    int dimx = 512;
    int griddimx = (len+dimx-1) / dimx;
    phase2_half_kernel<<<griddimx, dimx, 0, stream>>>(real, imag, colrank, ntg, phase2mapping, yuout, len);
    CUDACHECK(cudaGetLastError());
}

void phase3_merge_half(const half *rr_ri, const half *ir_ii,
const int nb, cuComplex *finaly, size_t len, cudaStream_t stream){
    int dimx = 512;
    int griddimx = (len+dimx-1) / dimx;
    phase3_merge_half_kernel<<<griddimx, dimx, 0, stream>>>(rr_ri, ir_ii, nb, finaly, len);
    CUDACHECK(cudaGetLastError());
}
    





void phase1fuse4ops_real(half * Arealreal, half * Bimagimag, size_t len, cudaStream_t stream)
{
    int dimx = 512;
    int griddimx = len / dimx;
    phase1fuse4ops_real_kernel<<<griddimx, dimx, 0, stream>>>(Arealreal, Bimagimag, len);
    CUDACHECK(cudaGetLastError());
}

void phase1fuse4ops_imag(half * Arealimag, half * Bimagreal, size_t len, cudaStream_t stream)
{
    int dimx = 512;
    int griddimx = len / dimx;
    phase1fuse4ops_imag_kernel<<<griddimx, dimx, 0, stream>>>(Arealimag, Bimagreal, len);
    CUDACHECK(cudaGetLastError());
}

void phase1twogemv(half * Arealimag, half * Bimagreal, size_t len, cudaStream_t stream)
{
    int dimx = 512;
    int griddimx = len / dimx;
    phase1twogemv_kernel<<<griddimx, dimx, 0, stream>>>(Arealimag, Bimagreal, len);
    CUDACHECK(cudaGetLastError());
}


void merge_half_realimag(half *real, half *imag, size_t len, cudaStream_t stream){
    int dimx = 512;
    int griddimx = (len+dimx-1) / dimx;
    merge_half_realimag_kernel<<<griddimx, dimx, 0, stream>>>(real, imag, len);
    CUDACHECK(cudaGetLastError());
}

void merge_float_realimag(float *real, float *imag, cuComplex * complexout, size_t len, cudaStream_t stream){
    int dimx = 512;
    int griddimx = (len+dimx-1) / dimx;
    merge_float_realimag_kernel<<<griddimx, dimx, 0, stream>>>(real, imag, complexout, len);
    CUDACHECK(cudaGetLastError());
}









__global__ void merge_int_2intout_realimag_kernel(
    const int * __restrict__ rr_ri, const int * __restrict__ ir_ii, 
    const int * __restrict__ colrank, 
    const int * __restrict__ colrank_withpadding, 
    const int ntg,
    float * __restrict__ realout, float * __restrict__ imagout, size_t len, size_t originlen)
{
    size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_x < len){
        int i = 0;
        size_t prefix = 0;
        size_t currank_withpadding = 0;
        size_t leftoffset = 0;
        size_t currank;
        for(; i<ntg; i++){
            currank_withpadding = colrank_withpadding[i];
            currank = colrank[i];
            leftoffset += currank_withpadding - currank;
            if(prefix + currank_withpadding > thread_x) break;
            prefix += currank_withpadding;
        }

        size_t localoffset = thread_x - prefix;
        if(localoffset < currank){
            // float curreal = rr_ri[2*prefix + localoffset] - ir_ii[2*prefix + currank_withpadding + localoffset];
            // float curimag = rr_ri[2*prefix + currank_withpadding + localoffset] + ir_ii[2*prefix + localoffset];
            realout[thread_x] = 1.0;
            imagout[thread_x] = 1.0;
        }
    }
}

void merge_int_2intout_realimag(
const int *rr_ri, const int *ir_ii, const int * colrank, 
const int * colrank_withpadding,
const int ntg, float *realout, float *imagout,  size_t len, size_t originlen,  cudaStream_t stream){
    int dimx = 128;
    int griddimx = (len+dimx-1) / dimx;
    merge_int_2intout_realimag_kernel<<<griddimx, dimx, 0, stream>>>
    (rr_ri, ir_ii, colrank, colrank_withpadding, ntg, realout, imagout, len, originlen);
    CUDACHECK(cudaGetLastError());
}
    


// __global__ void merge_int_2intout_realimag_kernel(
//     const int * __restrict__ rr_ri, const int * __restrict__ ir_ii, 
//     const int * __restrict__ colrank, 
//     const int * __restrict__ colrank_withpadding, 
//     const int ntg, const float * xtilemax, const float *Avtilemax, const float * 
//     float * __restrict__ realout, float * __restrict__ imagout, size_t len, size_t originlen)
// {
//     size_t thread_x = blockDim.x * blockIdx.x + threadIdx.x;
//     if (thread_x < len){
//         int i = 0;
//         size_t prefix = 0;
//         size_t currank_withpadding = 0;
//         size_t leftoffset = 0;
//         size_t currank;
//         for(; i<ntg; i++){
//             currank_withpadding = colrank_withpadding[i];
//             currank = colrank[i];
//             leftoffset += currank_withpadding - currank;
//             if(prefix + currank_withpadding > thread_x) break;
//             prefix += currank_withpadding;
//         }
        
//         size_t localoffset = thread_x - prefix;
//         if(localoffset < currank){
//             float curreal = rr_ri[2*prefix + localoffset] - ir_ii[2*prefix + currank_withpadding + localoffset];
//             float curimag = rr_ri[2*prefix + currank_withpadding + localoffset] + ir_ii[2*prefix + localoffset];
//             realout[phase2mapping[thread_x - leftoffset]] = curreal;
//             imagout[phase2mapping[thread_x - leftoffset + originlen]] = curimag;
//         }
//     }
// }



} // namespace cudatlrmvm

