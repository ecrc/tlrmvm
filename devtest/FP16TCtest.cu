#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda.h>

#define CUDACHECK(cmd) do {                         \
cudaError_t e = cmd;                              \
if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
}                                                 \
} while(0)

#define CUBLASCHECK(cmd) do {                         \
cublasStatus_t e = cmd;                              \
if( e != CUBLAS_STATUS_SUCCESS ) {                          \
    printf("Failed: Cublas error %s:%d '%s'\n",             \
        __FILE__,__LINE__, _cudaGetErrorEnum(e));   \
    exit(EXIT_FAILURE);                             \
}                                                 \
} while(0)

#include <stdio.h>  

static const char *_cudaGetErrorEnum(cublasStatus_t error) {
    switch (error) {
      case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
  
      case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
  
      case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
  
      case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
  
      case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
  
      case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
  
      case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
  
      case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
  
      case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
  
      case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
    }
  
    return "<unknown>";
  }
// void GetHostMemoryBatched( 
// half** A, half** B, half** C, 
// long int AvMs[], long int AvKs[], long int AvNs[], long int size){
//     half *tmpA, *tmpB, *tmpC;
//     half **tmpAb, **tmpBb, **tmpCc;
//     long int Asize(0), Bsize(0), Csize(0);
//     for(long int i=0; i<size; i++){
//         Asize += AvMs[i] * AvKs[i];
//         Bsize += AvKs[i] * AvNs[i];
//         Csize += AvMs[i] * AvNs[i];
//     }
//     CUDACHECK(cudaMalloc(&tmpA, sizeof(half) * Asize));
//     CUDACHECK(cudaMalloc(&tmpB, sizeof(half) * Bsize));
//     CUDACHECK(cudaMalloc(&tmpC, sizeof(half) * Csize));
    
// }

void GetDeviceMemoryBatched( 
half** A, half** B, half** C, 
half *** Abatchedptr, half*** Bbatchedptr, 
half*** Cbatchedptr, 
long int AvMs[], long int AvKs[], long int AvNs[], long int size){
    half *tmpA, *tmpB, *tmpC;
    half **tmpAa, **tmpBb, **tmpCc;
    long int Asize(0), Bsize(0), Csize(0);
    for(long int i=0; i<size; i++){
        Asize += AvMs[i] * AvKs[i];
        Bsize += AvKs[i] * AvNs[i];
        Csize += AvMs[i] * AvNs[i];
    }
    printf("%ld %ld %ld \n", Asize, Bsize, Csize);
    CUDACHECK(cudaMalloc(&tmpA, sizeof(half) * Asize));
    CUDACHECK(cudaMalloc(&tmpB, sizeof(half) * Bsize));
    CUDACHECK(cudaMalloc(&tmpC, sizeof(half) * Csize));

    CUDACHECK(cudaMalloc(&tmpAa, sizeof(half*) * size));
    CUDACHECK(cudaMalloc(&tmpBb, sizeof(half*) * size));
    CUDACHECK(cudaMalloc(&tmpCc, sizeof(half*) * size));
    half **htmpA, **htmpB, **htmpC;
    CUDACHECK(cudaMallocHost(&htmpA, sizeof(half*) * size));
    CUDACHECK(cudaMallocHost(&htmpB, sizeof(half*) * size));
    CUDACHECK(cudaMallocHost(&htmpC, sizeof(half*) * size));
    htmpA[0] = tmpA;
    htmpB[0] = tmpB;
    htmpC[0] = tmpC;

    for(int i=1; i<size; i++){
        htmpA[i] = htmpA[i-1] + AvMs[i-1] * AvKs[i-1];
        htmpB[i] = htmpB[i-1] + AvKs[i-1] * AvNs[i-1];
        htmpC[i] = htmpC[i-1] + AvMs[i-1] * AvNs[i-1];
    }
    CUDACHECK( cudaMemcpy( tmpAa, htmpA, sizeof(half*) * size, cudaMemcpyDefault) );
    CUDACHECK( cudaMemcpy( tmpBb, htmpB, sizeof(half*) * size, cudaMemcpyDefault) );
    CUDACHECK( cudaMemcpy( tmpCc, htmpC, sizeof(half*) * size, cudaMemcpyDefault) );
    A[0] = tmpA;
    B[0] = tmpB;
    C[0] = tmpC;
    Abatchedptr[0] = tmpAa;
    Bbatchedptr[0] = tmpBb;
    Cbatchedptr[0] = tmpCc;

}

int main (){


    // configuration
    long int nb = 256;

    // get host pointer 
    half *hAv, *hx;
    half val = 1.0;
    half alpha = 1.0;
    half beta = 0.0;
 
    long int Ntglobal = 40;
    long int maxrowsize = 2048;
    long int Avtotalelems(0), xtotalelems(0), yvtotalelems(0);
    long int M = maxrowsize;
    long int N = 2;
    long int K = nb;
    long int AvMs[40];
    long int AvKs[40];
    long int AvNs[40];
    M = 2048;
    K = 256;
    N = 2;
    for(int i=0; i<Ntglobal; i++){
        AvMs[i] = ( M );
        AvKs[i] = ( K );
        AvNs[i] = ( N );
        Avtotalelems += M * K;
        xtotalelems += K * N;
        yvtotalelems += M * N;
    }
    long int hfsize = sizeof(half);
    CUDACHECK(cudaMallocHost(&hAv, hfsize * Avtotalelems));
    CUDACHECK(cudaMallocHost(&hx, hfsize * xtotalelems));
    half one = 1.0;
    for(long int i=0; i< Avtotalelems; i++){
        hAv[i] = one;
    }
    for(long int i=0; i< xtotalelems; i++){
        hx[i] = one;
    }

    // get device pointer 
    half *Av, *x, *yout;
    half **Avbp, **xbp, **youtbp;
    GetDeviceMemoryBatched(&Av, &x, &yout, &Avbp, &xbp, &youtbp, AvMs, AvKs, AvNs, 40);
    cublasHandle_t handle;
    cublasCreate(&handle);
    CUDACHECK( cudaMemcpy( Av, hAv, sizeof(half) * Avtotalelems, cudaMemcpyDefault) );
    CUDACHECK( cudaMemcpy( x, hx, sizeof(half) * xtotalelems, cudaMemcpyDefault) );

    CUBLASCHECK(cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
    M, N, K, &alpha, (const void**)Avbp, CUDA_R_16F,
    M, (const void**)xbp, CUDA_R_16F, K, &beta, 
    (void **)youtbp, CUDA_R_16F, M, Ntglobal,
    CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));

    cublasDestroy(handle);
    half fval;
    float ffval = fval;
    for(int i=0; i<10; i++){
        CUDACHECK( cudaMemcpy( &fval, Av, sizeof(half), cudaMemcpyDefault) );
        ffval = fval;
        printf("%f \n", ffval);
    }
    for(int i=0; i<10; i++){
        CUDACHECK( cudaMemcpy( &fval, x, sizeof(half), cudaMemcpyDefault) );
        ffval = fval;
        printf("%f \n", ffval);
    }
    for(int i=0; i<10; i++){
        CUDACHECK( cudaMemcpy( &fval, yout, sizeof(half), cudaMemcpyDefault) );
        ffval = fval;
        printf("%f \n", ffval);
    }

    // GetDeviceMemoryBatched(&Av_imag, &x_copy2, &middle2, &Avbatchpointer_imag,
    // &xbatchpointer_imag, &yvbatchpointer_imag, AvMs, AvKs, AvNs);
    // CopyDataB2HD(Av_real, hAv_real, Avtotalelems);
    // CopyDataB2HD(Av_imag, hAv_imag, Avtotalelems);
    // CopyDataB2HD(x_copy1, middle1, xtotalelems);
    // CopyDataB2HD(x_copy2, middle2, xtotalelems);

    // long int STREAMSIZE = 2;
    // cudaStream_t * streamptr = new cudaStream_t[STREAMSIZE];
    // cublasHandle_t * cublashandleptr = new cublasHandle_t[STREAMSIZE];
    // for(int i=0; i<STREAMSIZE; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    // for(int i=0; i<STREAMSIZE; i++) cublasCreate_v2(&cublashandleptr[i]);
    // for(int i=0; i<STREAMSIZE; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEvent_t *events;
    // cudaEvent_t event_start;
    // cudaEvent_t event_phase2finish;
    // CUDACHECK(cudaEventCreate(&event_start));
    // CUDACHECK(cudaEventCreate(&event_phase2finish));
    // events = new cudaEvent_t[STREAMSIZE];
    // for(int i=0; i<STREAMSIZE; i++) CUDACHECK(cudaEventCreate(&events[i]));


    // vector<double> rawtime;
    // // cout << maxrowsize << endl;
    // half *tmpyv_Areal_Bimag, **tmpyv_Areal_Bimag_batched;
    // half *tmpyv_Aimag_Breal, **tmpyv_Aimag_Breal_batched;
    // getYvbatchedpointer(&tmpyv_Areal_Bimag, &tmpyv_Areal_Bimag_batched, AvMs, AvNs);
    // getYvbatchedpointer(&tmpyv_Aimag_Breal, &tmpyv_Aimag_Breal_batched, AvMs, AvNs);

    // for(auto st : state){
    //     cudaDeviceSynchronize();
    //     cudaEventRecord(start);
    //     CUBLASCHECK(cublasGemmBatchedEx( cublashandleptr[0], 
    //     CUBLAS_OP_N, CUBLAS_OP_N, 
    //     M, N, K, &alpha, (const void**)Avbatchpointer_real, CUDA_R_16F,
    //     maxrowsize, 
    //     (const void**)xbatchpointer_real, CUDA_R_16F, nb, &beta, 
    //     (void **)yvbatchpointer_real, CUDA_R_16F, maxrowsize, Ntglobal,
    //     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));        
    //     CUBLASCHECK(cublasGemmBatchedEx( cublashandleptr[1], 
    //     CUBLAS_OP_N, CUBLAS_OP_N, 
    //     M, N, K, &alpha, (const void**)Avbatchpointer_imag, CUDA_R_16F,
    //     maxrowsize, 
    //     (const void**)xbatchpointer_imag, CUDA_R_16F, nb, &beta, 
    //     (void **)yvbatchpointer_imag, CUDA_R_16F, maxrowsize, Ntglobal,
    //     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    //     cudaEventRecord(events[1], streamptr[1]);
    //     cudaStreamWaitEvent(streamptr[0], events[1]);
    //     phase1twogemv(middle1, middle2, maxrowsize * Ntglobal, streamptr[0]);
    //     cudaStreamSynchronize(streamptr[0]);
    //     cudaEventRecord(stop);
    //     float milliseconds = 0;
    //     cudaEventSynchronize(stop);
    //     cudaEventElapsedTime(&milliseconds, start, stop);
    //     state.SetIterationTime(milliseconds*1e-3);
    //     rawtime.push_back(milliseconds*1e-3);
    // }
    // half hhyv_real;
    // half hhyv_imag;
    // CopyDataB2HD(&hhyv_real, &middle1[1], 1);
    // float yvfreal = hhyv_real;
    // // cout << yvfreal << endl;
    // CopyDataB2HD(&hhyv_imag, &middle2[1], 1);
    // float yvimag = hhyv_imag;
    // // cout << yvimag << endl;
    // for(int i=0; i<STREAMSIZE; i++) cudaStreamDestroy(streamptr[i]);
    // for(int i=0; i<STREAMSIZE; i++) cublasDestroy(cublashandleptr[i]);
    // // double bytes = sizeof(half) * ( Ntglobal * (M * K + K*N + M*K) ) * (double)state.iterations();
    // // state.counters["BandWidth"] =
    // // Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);


}