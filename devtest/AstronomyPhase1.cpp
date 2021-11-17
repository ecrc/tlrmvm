#include <algorithm>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <memory.h>
#include <cuda.h>
#include <complex>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h> // if you need CUBLAS v2, include before magma.h
#include "nvToolsExt.h"

using namespace std;


#ifdef USE_NVTX
#include "nvToolsExt.h"

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif


/**
 * @brief Just for Phase 1!!!!!!!!!!!!!!
 * 
 */

#define TESTING_CHECK( err )                                                 \
    do {                                                                     \
        magma_int_t err_ = (err);                                            \
        if ( err_ != 0 ) {                                                   \
            fprintf( stderr, "Error: %s\nfailed at %s:%d: error %lld: %s\n", \
                     #err, __FILE__, __LINE__,                               \
                     (long long) err_, magma_strerror(err_) );               \
            exit(1);                                                         \
        }                                                                    \
    } while( 0 )

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
 
 #define NCCLCHECK(cmd) do {                         \
   ncclResult_t r = cmd;                             \
   if (r!= ncclSuccess) {                            \
     printf("Failed, NCCL error %s:%d '%s'\n",             \
         __FILE__,__LINE__,ncclGetErrorString(r));   \
     exit(EXIT_FAILURE);                             \
   }                                                 \
 } while(0)
 
 
 #define NCCLCHECK(cmd) do {                         \
   ncclResult_t r = cmd;                             \
   if (r!= ncclSuccess) {                            \
     printf("Failed, NCCL error %s:%d '%s'\n",             \
         __FILE__,__LINE__,ncclGetErrorString(r));   \
     exit(EXIT_FAILURE);                             \
   }                                                 \
 } while(0)



void readbinary(int **ptr, int length, string abspath){
    int * tmp = new int[length];
    FILE *f = fopen(abspath.c_str(), "rb");
    int ret = fread(tmp, sizeof(int), length, f); 
    assert(ret == length);
    fclose(f);
    ptr[0] = tmp;
}

void readbinary(float **ptr, int length, string abspath){
    float * tmp = new float[length];
    FILE *f = fopen(abspath.c_str(), "rb");
    int ret = fread(tmp, sizeof(float), length, f); 
    assert(ret == length);
    fclose(f);
    ptr[0] = tmp;
}

void init(float *ptr, size_t length){
    for(size_t i=0; i<length; i++) ptr[i] = 1.0;
}

int main(int argc, char *argv[]){
    float **hA, **hB, **hC;
    float **A, **B, **C;
    int *Rmat;
    string datafolder = "/datawaha/ecrc/hongy0a/astronomy/mavis/output/";
    string acc = "0.0001";
    string id = "000";
    int nb = 128;
    int m = 4802;
    int n = 19078;
    int pm = ( m / nb + (m % nb != 0) ) * nb;
    int pn = ( n / nb + (n % nb != 0) ) * nb;
    int nt = pn / nb;
    int mt = pm / nb;
    hA = new float*[nt];
    hB = new float*[nt];
    hC = new float*[nt];
    char filename[200];
    sprintf(filename, "R_id%s_nb%d_acc%s.bin", id.c_str(), nb, acc.c_str());
    readbinary(&Rmat, nt*mt, datafolder + "/" + string(filename));
    size_t granksum = 0;
    vector<size_t> colsum(nt,0);
    for(int i=0; i<nt; i++){
        for(int j=0; j<mt; j++){
            colsum[i] += Rmat[j + i * mt];
        }
    }

    size_t glength = granksum * nb;
    for(int i=0; i<nt; i++){ 
        hA[i] = new float[colsum[i] * nb];
        init(hA[i], colsum[i] * nb);
        hB[i] = new float[nb];
        init(hB[i], nb);
        hC[i] = new float[colsum[i]];
        init(hC[i], colsum[i]);
    }

    A = new float*[nt];
    B = new float*[nt];
    C = new float*[nt];

    for(int i=0; i<nt; i++){
        
        CUDACHECK(cudaMalloc(&A[i], colsum[i] * nb * sizeof(float)));
        CUDACHECK(cudaMemcpy(A[i], hA[i], 
        colsum[i] * nb * sizeof(float), cudaMemcpyDefault));
        
        CUDACHECK(cudaMalloc(&B[i], nb * sizeof(float)));
        CUDACHECK(cudaMemcpy(B[i], hB[i], 
        nb * sizeof(float), cudaMemcpyDefault));
        
        
        CUDACHECK(cudaMalloc(&C[i], colsum[i] * sizeof(float)));
        CUDACHECK(cudaMemcpy(C[i], hC[i], 
        colsum[i] * sizeof(float), cudaMemcpyDefault));

    }

    float alpha(1.0),beta(0.0);
    if(argc < 3){
        cout << "Usage ./bin streamsize loopsize" << endl;
        exit(0);
    }
    size_t streamsize = atol(argv[1]);
    size_t loopcount = atol(argv[2]);
    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreate(&streamptr[i]);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream(cublashandleptr[i], streamptr[i]);
    cout << "Workload of Phase1" << endl;
    for(int i=0; i<nt; i++){
        printf("GEMV %d M : %lu N : %d \n", i, colsum[i], nb);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    /********************
     * Correctness check for phase 1
     * *******************/
    for(int i=0; i<streamsize; i++) cudaStreamSynchronize(streamptr[i]);
#ifdef USE_NVTX
    PUSH_RANGE("Correctness",1)
#endif 
    for(int i=0; i<nt; i++){
        cublasSgemv(cublashandleptr[i%streamsize], CUBLAS_OP_N, colsum[i], nb, &alpha, 
        A[i], colsum[i], B[i], 1, &beta, C[i], 1);
    }
    for(int i=0; i<streamsize; i++) cudaStreamSynchronize(streamptr[i]);
    
    for(int i=0; i<nt; i++){
        float * checky = new float[colsum[i]];
        CUDACHECK(cudaMemcpy(checky, C[i], colsum[i] * sizeof(float), cudaMemcpyDefault));
        for(int j=0; j<colsum[i]; j++) assert(checky[j] == nb);
        delete[] checky;
    }    
#ifdef USE_NVTX
    POP_RANGE
#endif 
    vector<double> timestat;
    for(int loopi=0; loopi<loopcount; loopi++){
        for(int i=0; i<streamsize; i++) cudaStreamSynchronize(streamptr[i]);
#ifdef USE_NVTX
        PUSH_RANGE("Phase1",loopi)
#endif 
        cudaEventRecord(start);
        for(int i=0; i<nt; i++){
            cublasSgemv(cublashandleptr[i%streamsize], CUBLAS_OP_N, colsum[i], nb, &alpha, 
            A[i], colsum[i], B[i], 1, &beta, C[i], 1);
        }
        for(int i=0; i<streamsize; i++) cudaStreamSynchronize(streamptr[i]);
        cudaEventRecord(stop);
#ifdef USE_NVTX
        POP_RANGE
#endif 
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        timestat.push_back(milliseconds*1e-3);
    }
    std::sort(timestat.begin(), timestat.end());   
    cout << "phase 1 Test" << endl;
    cout << "We are using " << streamsize <<" streams, running "<< timestat.size() << " times, median time is " << timestat[timestat.size() / 2] * 1e6 << " us." << endl;


    for(int i=0; i<nt; i++){
        delete[] hA[i];
        delete[] hB[i];
        delete[] hC[i];
        CUDACHECK(cudaFree(A[i]));
        CUDACHECK(cudaFree(B[i]));
        CUDACHECK(cudaFree(C[i]));
    }
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] hA;
    delete[] hB;
    delete[] hC;

    return 0;
}