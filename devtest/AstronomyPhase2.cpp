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

int main(int argc, char*argv[]){
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
    hA = new float*[mt];
    hB = new float*[mt];
    hC = new float*[mt];
    char filename[200];
    sprintf(filename, "R_id%s_nb%d_acc%s.bin", id.c_str(), nb, acc.c_str());
    readbinary(&Rmat, nt*mt, datafolder + "/" + string(filename));
    size_t granksum = 0;
    vector<size_t> rowsum(mt,0);
    for(int i=0; i<nt; i++){
        for(int j=0; j<mt; j++){
            rowsum[j] += Rmat[j + i * mt];
        }
    }

    for(int i=0; i<mt; i++){ 
        hA[i] = new float[rowsum[i] * nb];
        init(hA[i], rowsum[i] * nb);
        hB[i] = new float[rowsum[i]];
        init(hB[i], rowsum[i]);
        hC[i] = new float[nb];
        init(hC[i], nb);
        
    }

    A = new float*[mt];
    B = new float*[mt];
    C = new float*[mt];

    for(int i=0; i<mt; i++){
        
        CUDACHECK(cudaMalloc(&A[i], rowsum[i] * nb * sizeof(float)));
        CUDACHECK(cudaMemcpy(A[i], hA[i], 
        rowsum[i] * nb * sizeof(float), cudaMemcpyDefault));

        CUDACHECK(cudaMalloc(&B[i], rowsum[i] * sizeof(float)));
        CUDACHECK(cudaMemcpy(B[i], hB[i], 
        rowsum[i] * sizeof(float), cudaMemcpyDefault));

        CUDACHECK(cudaMalloc(&C[i], nb * sizeof(float)));
        CUDACHECK(cudaMemcpy(C[i], hC[i], 
        nb * sizeof(float), cudaMemcpyDefault));

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
    cout << "Workload of Phase2" << endl;
    for(int i=0; i<mt; i++){
        printf("GEMV %d M : %d N : %lu \n", i, nb, rowsum[i]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /********************
     * Correctness check for phase 1
     * *******************/
    for(int i=0; i<streamsize; i++) cudaStreamSynchronize(streamptr[i]);
    PUSH_RANGE("Correctness",1)
    for(int i=0; i<mt; i++){
        cublasSgemv(cublashandleptr[i%streamsize], CUBLAS_OP_N, nb, rowsum[i], &alpha, 
        A[i], nb, B[i], 1, &beta, C[i], 1);
    }
    for(int i=0; i<streamsize; i++) cudaStreamSynchronize(streamptr[i]);
    
    for(int i=0; i<mt; i++){
        float * checky = new float[nb];    
        CUDACHECK(cudaMemcpy(checky, C[i], nb*sizeof(float), cudaMemcpyDefault));
        for(int j=0; j<nb; j++){
            assert(checky[j] == rowsum[i]);
        }
        delete[] checky;
    }
    vector<double> timestat;
    POP_RANGE
    for(int loopi=0; loopi<loopcount; loopi++){
        for(int i=0; i<streamsize; i++) cudaStreamSynchronize(streamptr[i]);
        PUSH_RANGE("Phase2",loopi)
        cudaEventRecord(start);
        for(int i=0; i<mt; i++){
            cublasSgemv(cublashandleptr[i%streamsize], CUBLAS_OP_N, nb, rowsum[i], &alpha, 
            A[i], nb, B[i], 1, &beta, C[i], 1);
        }
        for(int i=0; i<streamsize; i++) cudaStreamSynchronize(streamptr[i]);
        cudaEventRecord(stop);
        POP_RANGE
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        timestat.push_back(milliseconds*1e-3);
    }
    std::sort(timestat.begin(), timestat.end());   
    cout << "phase 2 Test" << endl;
    cout << "We are using " << streamsize <<" streams, running "<< timestat.size() << " times, median time is " << timestat[timestat.size() / 2] * 1e6 << " us." << endl;

    for(int i=0; i<mt; i++){
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