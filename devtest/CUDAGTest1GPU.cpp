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
#include <mpi.h>
#include <nccl.h>
#include <complex>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h> // if you need CUBLAS v2, include before magma.h

#include "common/Common.h"
#include "common/AppUtil.h"
#include "tlrmvm/Tlrmvm.h"
#include "tlrmvm/Tlrmvmcuda.h"
#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace std;
using namespace tlrmat;
using namespace cudatlrmat;
using ::testing::Pointwise;
using ::testing::NanSensitiveFloatNear;


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

vector<string> g_command_line_arg_vec;

TEST(UTIL, CUDAMallocCUDAFree){
    cuComplex *A, *B, *C;
    GetDeviceMemory(&A,&B,&C,100,100,100);
    FreeDeviceMemory(A,B,C);
}

TEST(UTIL, BatchedMallocFree){

    vector<size_t> Ms,Ks,Ns;
    int batchsize = 100;
    for(int i=0; i<batchsize; i++){
        Ms.push_back(128);
        Ks.push_back(256);
        Ns.push_back(32);
    }
    complex<float> *hA, *hB, *hC, **hAbp, **hBbp, **hCbp, val;
    val = complex<float>(1.0,0.0);
    GetHostMemoryBatched(&hA, &hB, &hC, 
    &hAbp, &hBbp, &hCbp, Ms, Ks, Ns, val);
    cuComplex *A, *B, *C;
    cuComplex **Abatchpointer, **Bbatchpointer, **Cbatchpointer;
    GetDeviceMemoryBatched(&A, &B, &C, 
    &Abatchpointer, &Bbatchpointer, &Cbatchpointer, Ms, Ks, Ns);
    CUDACHECK( cudaMemcpy(A, hA, 1 * sizeof(cuComplex), cudaMemcpyDefault) );
    CopyDataB2HD((complex<float>*)A, (complex<float>*)B, (complex<float>*)C, 
    hA, hB, hC, Ms, Ks, Ns);
    cuComplex alpha, beta;
    alpha.x = 1.0; 
    alpha.y = beta.x = beta.y = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    FreeDeviceMemoryBatched(A,B,C,Abatchpointer, Bbatchpointer,
    Cbatchpointer);
    cublasDestroy(handle);
}


void get1gemvpointer(size_t m, size_t n, float **A_out, float **B_out, float **C_out){
    float *hA, *hB, *hC, val;
    val = 1.0;
    GetHostMemory(&hA, m * n);
    GetHostMemory(&hB, n * 1);
    GetHostMemory(&hC, m * 1);

    float *A, *B, *C, alpha(1.0), beta(0.0);

    GetDeviceMemory(&A, m * n);
    GetDeviceMemory(&B, n * 1);
    GetDeviceMemory(&C, m * 1);

    Init(hA, m * n, (float)0.37);
    Init(hB, 1 * n, (float)0.37);
    Init(hC, m * 1, (float)0.37);

    CopyDataB2HD(A, hA, m*n);
    CopyDataB2HD(B, hB, 1*n);
    CopyDataB2HD(C, hC, m*1);

    A_out[0] = A;
    B_out[0] = B;
    C_out[0] = C;

    FreeHostMemory(hA);
    FreeHostMemory(hB);
    FreeHostMemory(hC);

}


TEST(CUDATLRMVM, RefGEMV){
    size_t m1 = 5000;
    size_t n1 = 20000;
    float *A1, *B1, *C1;
    get1gemvpointer(m1, n1, &A1, &B1, &C1);

    float alpha(1.0),beta(0.0);
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    double timestat[500];
    // double bd = (m1 * n1 + m1 + n1) * sizeof(float) / (milliseconds * 1e-3) * 1e-9;
    for(int i=0; i<500; i++){
        // PUSH_RANGE("OneGEMV",i)
        cudaEventRecord(start);
        
        CUBLASCHECK(cublasgemv(handle, CUBLAS_OP_N, m1, n1, &alpha, 
        A1, m1, B1, 1, &beta, C1, 1));
        
        cudaEventRecord(stop);
        // POP_RANGE
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if(i < 100) continue;
        timestat[i-100] = (double) milliseconds * 1e-3;
        // cout << "time per loop " << milliseconds << ", bandwidth " << bd <<  endl;
    }
    double avgtime(0.0), avgbd(0.0);
    for(int i=0; i<400; i++){
        avgtime += timestat[i];
        avgbd  += (m1 * n1 + m1 + n1) * sizeof(float) / (timestat[i]) * 1e-9;
    }
    avgtime /= 400.0;
    avgbd /= 400.0;
    cout << "avg time (us) " << avgtime*1e6 << ", " << " avg bd (GB/s) " << avgbd << endl;
    FreeDeviceMemory(A1);
    FreeDeviceMemory(B1);
    FreeDeviceMemory(C1);
}

TEST(CUDATLRMVM, RefTwoGEMV){
    size_t m1 = 400;
    size_t n1 = 10000;
    float *A1, *B1, *C1;
    get1gemvpointer(m1, n1, &A1, &B1, &C1);

    size_t m2 = 800;
    size_t n2 = 10000;
    float *A2, *B2, *C2;
    get1gemvpointer(m2, n2, &A2, &B2, &C2);

    float alpha(1.0),beta(0.0);
    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(int i=0; i<50; i++){
        cudaEventRecord(start);
        PUSH_RANGE("TWOGEMV",i)
        CUBLASCHECK(cublasgemv(handle, CUBLAS_OP_N, m1, n1, &alpha, 
        A1, m1, B1, 1, &beta, C1, 1));
        CUBLASCHECK(cublasgemv(handle, CUBLAS_OP_N, m2, n2, &alpha, 
        A2, m2, B2, 1, &beta, C2, 1));
        POP_RANGE
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cout << "time per loop " << milliseconds << endl;
    }
    
    FreeDeviceMemory(A1);
    FreeDeviceMemory(B1);
    FreeDeviceMemory(C1);
    FreeDeviceMemory(A2);
    FreeDeviceMemory(B2);
    FreeDeviceMemory(C2);
}


TEST(CUDATLRMVM, RefTwoGEMVStream){
    // vector<size_t> m = {400, 800};
    // vector<size_t> n = {10000,10000};

    vector<size_t> m = {700, 128};
    vector<size_t> n = {700, 128};
    

    float **A, **B, **C;
    A = new float*[m.size()];
    B = new float*[m.size()];
    C = new float*[m.size()];
    for(int i=0; i<m.size(); i++){
        get1gemvpointer(m[i], n[i], &A[i], &B[i], &C[i]);
    }

    float alpha(1.0),beta(0.0);
    size_t streamsize = 2;
    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreate(&streamptr[i]);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for(int loopi=0; loopi<50; loopi++){
        for(int i=0; i<streamsize; i++) cudaStreamSynchronize(streamptr[i]);
        PUSH_RANGE("TWOGEMV",loopi)
        cudaEventRecord(start);
        for(int i=0; i<m.size(); i++){
            cublasSetStream_v2(cublashandleptr[i%streamsize], streamptr[i%streamsize]);
            // CUBLASCHECK(
                cublasgemv(cublashandleptr[i%streamsize], CUBLAS_OP_N, m[i], n[i], &alpha, 
            A[i], m[i], B[i], 1, &beta, C[i], 1);
            // );
        }
        for(int i=0; i<streamsize; i++) cudaStreamSynchronize(streamptr[i]);
        cudaEventRecord(stop);
        POP_RANGE
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cout << "time per loop " << milliseconds << endl;
    }

    for(int i=0; i<streamsize; i++) cudaStreamDestroy(streamptr[i]);
    for(int i=0; i<streamsize; i++) cublasDestroy_v2(cublashandleptr[i]);
    delete[] streamptr;
    delete[] cublashandleptr;

    for(int i=0; i<m.size(); i++){
        FreeDeviceMemory(A[i]);
        FreeDeviceMemory(B[i]);
        FreeDeviceMemory(C[i]);    
    }
}

TEST(CUDATLRMVM, RefTwoGEMVStreamGraph){

    vector<size_t> m = {400, 800};
    vector<size_t> n = {10000, 10000};
    

    float **A, **B, **C;
    A = new float*[m.size()];
    B = new float*[m.size()];
    C = new float*[m.size()];
    for(int i=0; i<m.size(); i++){
        get1gemvpointer(m[i], n[i], &A[i], &B[i], &C[i]);
    }

    float alpha(1.0),beta(0.0);
    size_t streamsize = 2;
    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) 
    cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    bool graphCreated=false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaEvent_t event1, event2;
    CUDACHECK(cudaEventCreate(&event1));
    CUDACHECK(cudaEventCreate(&event2));

    for(int loopi=0; loopi<50; loopi++){
        cudaEventRecord(start);
        PUSH_RANGE("TWOGEMV",loopi)
        if(!graphCreated){
            cudaStreamBeginCapture(streamptr[0], cudaStreamCaptureModeGlobal);
            cudaEventRecord(event1, streamptr[0]);
            cudaStreamWaitEvent(streamptr[1], event1);
            CUBLASCHECK(cublasSetStream_v2(cublashandleptr[0], streamptr[0]));
            CUBLASCHECK(cublasgemv(cublashandleptr[0], CUBLAS_OP_N, m[0], n[0], &alpha, 
            A[0], m[0], B[0], 1, &beta, C[0], 1));
            CUBLASCHECK(cublasSetStream_v2(cublashandleptr[1], streamptr[1]));
            CUBLASCHECK(cublasgemv(cublashandleptr[1], CUBLAS_OP_N, m[1], n[1], &alpha, 
            A[1], m[1], B[1], 1, &beta, C[1], 1));
            cudaEventRecord(event2, streamptr[1]);
            cudaStreamWaitEvent(streamptr[0], event2);

            // for(int i=0; i<m.size(); i++){
            //     CUBLASCHECK(cublasSetStream_v2(cublashandleptr[i%streamsize], streamptr[i%streamsize]));
            //     CUBLASCHECK(cublasgemv(cublashandleptr[i%streamsize], CUBLAS_OP_N, m[i], n[i], &alpha, 
            //     A[i], m[i], B[i], 1, &beta, C[i], 1));
            // }
            cudaStreamEndCapture(streamptr[0], &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated=true;
        }
        cudaGraphLaunch(instance, streamptr[0]);
        cudaStreamSynchronize(streamptr[0]);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        POP_RANGE
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cout << "time per loop " << milliseconds << endl;
    }

    for(int i=0; i<streamsize; i++) cudaStreamDestroy(streamptr[i]);
    for(int i=0; i<streamsize; i++) cublasDestroy_v2(cublashandleptr[i]);
    delete[] streamptr;
    delete[] cublashandleptr;

    for(int i=0; i<m.size(); i++){
        FreeDeviceMemory(A[i]);
        FreeDeviceMemory(B[i]);
        FreeDeviceMemory(C[i]);    
    }
}


TEST(CUDATLRMVM, BatchedSGEMM){

    vector<size_t> Ms,Ks,Ns;
    int batchsize = 100;
    for(int i=0; i<batchsize; i++){
        Ms.push_back(128);
        Ks.push_back(256);
        Ns.push_back(32);
    }

    float *hA, *hB, *hC, **hAbp, **hBbp, **hCbp, val;
    val = 1.0;

    GetHostMemoryBatched(&hA, &hB, &hC, 
    &hAbp, &hBbp, &hCbp, Ms, Ks, Ns, val);

    float *A, *B, *C;
    float **Abatchpointer, **Bbatchpointer, **Cbatchpointer;

    GetDeviceMemoryBatched(&A, &B, &C, 
    &Abatchpointer, &Bbatchpointer, &Cbatchpointer, Ms, Ks, Ns);

    CUDACHECK( cudaMemcpy(A, hA, 1 * sizeof(float), cudaMemcpyDefault) );
    CopyDataB2HD((float*)A, (float*)B, (float*)C, hA, hB, hC, Ms, Ks, Ns);

    float alpha, beta;
    alpha = 1.0;
    beta = 0.0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    float * tmpC = C;
    tmpC = tmpC + Ms[0] * Ns[0];
    CUBLASCHECK(cublasgemmbatched(handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, Ms[0], Ns[0], Ks[0],
    &alpha, (const float **)Abatchpointer, Ms[0], 
    (const float **)Bbatchpointer, Ks[0], &beta, 
    Cbatchpointer, Ms[0], batchsize));

    size_t Atotalelems, Btotalelems, Ctotalelems;
    tlrmat::CaluclateTotalElements(Ms, Ks, Ns, Atotalelems, Btotalelems, Ctotalelems);
    float *checkC = new float[Ctotalelems];
    CopyDataB2HD(hA, hB, checkC, (float*)A,
    (float*)B, (float*)C, Ms, Ks, Ns);
    for(int i=0; i<Ctotalelems; i++){
        ASSERT_EQ(checkC[i], Ks[0]);
    }
    FreeDeviceMemoryBatched(A,B,C,Abatchpointer, Bbatchpointer,
    Cbatchpointer);
    cublasDestroy(handle);
    delete[] checkC;
}


TEST(CUDATLRMVM, BatchedCGEMM){
    vector<size_t> Ms,Ks,Ns;
    int batchsize = 100;
    for(int i=0; i<batchsize; i++){
        Ms.push_back(128);
        Ks.push_back(256);
        Ns.push_back(32);
    }

    complex<float> *hA, *hB, *hC, **hAbp, **hBbp, **hCbp, val;
    val = complex<float>(1.0,0.0);

    GetHostMemoryBatched(&hA, &hB, &hC, 
    &hAbp, &hBbp, &hCbp, Ms, Ks, Ns, val);

    cuComplex *A, *B, *C;
    cuComplex **Abatchpointer, **Bbatchpointer, **Cbatchpointer;

    GetDeviceMemoryBatched(&A, &B, &C, 
    &Abatchpointer, &Bbatchpointer, &Cbatchpointer, Ms, Ks, Ns);

    CUDACHECK( cudaMemcpy(A, hA, 1 * sizeof(cuComplex), cudaMemcpyDefault) );
    CopyDataB2HD((complex<float>*)A, (complex<float>*)B, (complex<float>*)C, 
    hA, hB, hC, Ms, Ks, Ns);

    cuComplex alpha, beta;
    alpha.x = 1.0; 
    alpha.y = beta.x = beta.y = 0;
    cublasHandle_t handle;
    cublasCreate(&handle);
    cuComplex * tmpC = C;
    tmpC = tmpC + Ms[0] * Ns[0];
    CUBLASCHECK(cublasgemmbatched(handle, 
    CUBLAS_OP_N, CUBLAS_OP_N, Ms[0], Ns[0], Ks[0],
    &alpha, (const cuComplex **)Abatchpointer, Ms[0], 
    (const cuComplex **)Bbatchpointer, Ks[0], &beta, 
    Cbatchpointer, Ms[0], batchsize));

    size_t Atotalelems, Btotalelems, Ctotalelems;
    tlrmat::CaluclateTotalElements(Ms, Ks, Ns, Atotalelems, Btotalelems, Ctotalelems);
    complex<float> *checkC = new complex<float>[Ctotalelems];
    CopyDataB2HD(hA, hB, checkC, (complex<float>*)A,
    (complex<float>*)B, (complex<float>*)C, Ms, Ks, Ns);
    for(int i=0; i<Ctotalelems; i++){
        ASSERT_EQ(checkC[i].real(), Ks[0]);
    }
    FreeDeviceMemoryBatched(A,B,C,Abatchpointer, Bbatchpointer,
    Cbatchpointer);
    cublasDestroy(handle);
    delete[] checkC;
}


TEST(CUDATLRMVM, SimpleStreamGEMV){
    vector<size_t> Ms,Ks,Ns;
    int batchsize = 12;
    for(int i=0; i<batchsize; i++){
        Ms.push_back(32);
        Ks.push_back(32);
        Ns.push_back(1);
    }
    complex<float> *hA, *hB, *hC, **hAbp, **hBbp, **hCbp, val;
    val = complex<float>(1.0,0.0);

    GetHostMemoryBatched(&hA, &hB, &hC, 
    &hAbp, &hBbp, &hCbp, Ms, Ks, Ns, val);

    cuComplex *A, *B, *C;
    cuComplex **Abatchpointer, **Bbatchpointer, **Cbatchpointer;

    GetDeviceMemoryBatched(&A, &B, &C, 
    &Abatchpointer, &Bbatchpointer, &Cbatchpointer, Ms, Ks, Ns);

    cuComplex **Abatchpointer_h,**Bbatchpointer_h, **Cbatchpointer_h;
    Abatchpointer_h = new cuComplex*[batchsize];
    Bbatchpointer_h = new cuComplex*[batchsize];
    Cbatchpointer_h = new cuComplex*[batchsize];
    CUDACHECK( cudaMemcpy(Abatchpointer_h, Abatchpointer, 
    batchsize * sizeof(cuComplex*), cudaMemcpyDefault) );
    CUDACHECK( cudaMemcpy(Bbatchpointer_h, Bbatchpointer, 
    batchsize * sizeof(cuComplex*), cudaMemcpyDefault) );
    CUDACHECK( cudaMemcpy(Cbatchpointer_h, Cbatchpointer, 
    batchsize * sizeof(cuComplex*), cudaMemcpyDefault) );

    CopyDataB2HD((complex<float>*)A, (complex<float>*)B, (complex<float>*)C, 
    hA, hB, hC, Ms, Ks, Ns);

    cuComplex alpha, beta;
    alpha.x = 1.0; 
    alpha.y = beta.x = beta.y = 0;
    /**************************
     * CUBLAS CALL 
     * ************************/
    
    cudaStream_t * streamptr = new cudaStream_t[batchsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[batchsize];
    for(int i=0; i<batchsize; i++) cudaStreamCreate(&streamptr[i]);
    for(int i=0; i<batchsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    // #pragma omp parallel for 
    for(int i=0; i<batchsize; i++){
        cublasSetStream_v2(cublashandleptr[i], streamptr[i]);
        CUBLASCHECK(
        cublasgemv(cublashandleptr[i], CUBLAS_OP_N,
        Ms[i], Ks[i],
        &alpha, (const cuComplex*)Abatchpointer_h[i], Ms[i], 
        (const cuComplex*)Bbatchpointer_h[i], 1, &beta, 
        Cbatchpointer_h[i], 1);
        );
    }

    for(int i=0; i<batchsize; i++) cudaStreamDestroy(streamptr[i]);
    for(int i=0; i<batchsize; i++) cublasDestroy_v2(cublashandleptr[i]);
    delete[] streamptr;
    delete[] cublashandleptr;

    size_t Atotalelems, Btotalelems, Ctotalelems;
    tlrmat::CaluclateTotalElements(Ms, Ks, Ns, Atotalelems, Btotalelems, Ctotalelems);
    complex<float> *checkC = new complex<float>[Ctotalelems];
    CopyDataB2HD(hA, hB, checkC, (complex<float>*)A,
    (complex<float>*)B, (complex<float>*)C, Ms, Ks, Ns);
    for(int i=0; i<Ctotalelems; i++){
        ASSERT_EQ(checkC[i].real(), Ks[0]);
    }
    delete[] checkC;

    FreeDeviceMemoryBatched(A,B,C,Abatchpointer, Bbatchpointer,
    Cbatchpointer);
    FreeHostMemoryBatched(hA, hB, hC, hAbp, hBbp, hCbp);
    delete[] Abatchpointer_h;
    delete[] Bbatchpointer_h;
    delete[] Cbatchpointer_h;
}


TEST(CUDATLRMVM, GraphGEMV){
    vector<size_t> Ms,Ks,Ns;
    int batchsize = 200;
    for(int i=0; i<batchsize; i++){
        Ms.push_back(32);
        Ks.push_back(32);
        Ns.push_back(1);
    }
    complex<float> *hA, *hB, *hC, **hAbp, **hBbp, **hCbp, val;
    val = complex<float>(1.0,0.0);

    GetHostMemoryBatched(&hA, &hB, &hC, 
    &hAbp, &hBbp, &hCbp, Ms, Ks, Ns, val);

    cuComplex *A, *B, *C;
    cuComplex **Abatchpointer, **Bbatchpointer, **Cbatchpointer;

    GetDeviceMemoryBatched(&A, &B, &C, 
    &Abatchpointer, &Bbatchpointer, &Cbatchpointer, Ms, Ks, Ns);

    cuComplex **Abatchpointer_h,**Bbatchpointer_h, **Cbatchpointer_h;
    Abatchpointer_h = new cuComplex*[batchsize];
    Bbatchpointer_h = new cuComplex*[batchsize];
    Cbatchpointer_h = new cuComplex*[batchsize];
    CUDACHECK( cudaMemcpy(Abatchpointer_h, Abatchpointer, 
    batchsize * sizeof(cuComplex*), cudaMemcpyDefault) );
    CUDACHECK( cudaMemcpy(Bbatchpointer_h, Bbatchpointer, 
    batchsize * sizeof(cuComplex*), cudaMemcpyDefault) );
    CUDACHECK( cudaMemcpy(Cbatchpointer_h, Cbatchpointer, 
    batchsize * sizeof(cuComplex*), cudaMemcpyDefault) );

    CopyDataB2HD((complex<float>*)A, (complex<float>*)B, (complex<float>*)C, 
    hA, hB, hC, Ms, Ks, Ns);

    cuComplex alpha, beta;
    alpha.x = 1.0; 
    alpha.y = beta.x = beta.y = 0;

    bool graphCreated=false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    int NSTEP = 5;
    cudaStream_t stream;
    cublasHandle_t handle;
    cudaStreamCreate(&stream);
    cublasCreate_v2(&handle);
    for(int istep=0; istep<NSTEP; istep++){
    if(!graphCreated){
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
        for(int i=0; i<batchsize; i++){
            cublasSetStream_v2(handle, stream);
            cublasgemv(handle, CUBLAS_OP_N,
            Ms[i], Ks[i],
            &alpha, (const cuComplex*)Abatchpointer_h[i], Ms[i], 
            (const cuComplex*)Bbatchpointer_h[i], 1, &beta, 
            Cbatchpointer_h[i], 1);
        }
        cudaStreamEndCapture(stream, &graph);
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
        graphCreated=true;
    }
    cudaGraphLaunch(instance, stream);
    cudaStreamSynchronize(stream);
    }

    size_t Atotalelems, Btotalelems, Ctotalelems;
    tlrmat::CaluclateTotalElements(Ms, Ks, Ns, Atotalelems, Btotalelems, Ctotalelems);
    complex<float> *checkC = new complex<float>[Ctotalelems];
    CopyDataB2HD(hA, hB, checkC, (complex<float>*)A,
    (complex<float>*)B, (complex<float>*)C, Ms, Ks, Ns);
    for(int i=0; i<Ctotalelems; i++){
        ASSERT_EQ(checkC[i].real(), Ks[0]);
    }
    delete[] checkC;

    FreeDeviceMemoryBatched(A,B,C,Abatchpointer, Bbatchpointer,
    Cbatchpointer);
    FreeHostMemoryBatched(hA, hB, hC, hAbp, hBbp, hCbp);
    delete[] Abatchpointer_h;
    delete[] Bbatchpointer_h;
    delete[] Cbatchpointer_h;
}


TEST(CUDATLRMVM, GraphStreamGEMV){
    vector<size_t> Ms,Ks,Ns;
    int batchsize = 200;
    for(int i=0; i<batchsize; i++){
        Ms.push_back(32);
        Ks.push_back(32);
        Ns.push_back(1);
    }
    complex<float> *hA, *hB, *hC, **hAbp, **hBbp, **hCbp, val;
    val = complex<float>(1.0,0.0);

    GetHostMemoryBatched(&hA, &hB, &hC, 
    &hAbp, &hBbp, &hCbp, Ms, Ks, Ns, val);

    cuComplex *A, *B, *C;
    cuComplex **Abatchpointer, **Bbatchpointer, **Cbatchpointer;

    GetDeviceMemoryBatched(&A, &B, &C, 
    &Abatchpointer, &Bbatchpointer, &Cbatchpointer, Ms, Ks, Ns);

    cuComplex **Abatchpointer_h,**Bbatchpointer_h, **Cbatchpointer_h;
    Abatchpointer_h = new cuComplex*[batchsize];
    Bbatchpointer_h = new cuComplex*[batchsize];
    Cbatchpointer_h = new cuComplex*[batchsize];
    CUDACHECK( cudaMemcpy(Abatchpointer_h, Abatchpointer, 
    batchsize * sizeof(cuComplex*), cudaMemcpyDefault) );
    CUDACHECK( cudaMemcpy(Bbatchpointer_h, Bbatchpointer, 
    batchsize * sizeof(cuComplex*), cudaMemcpyDefault) );
    CUDACHECK( cudaMemcpy(Cbatchpointer_h, Cbatchpointer, 
    batchsize * sizeof(cuComplex*), cudaMemcpyDefault) );

    CopyDataB2HD((complex<float>*)A, (complex<float>*)B, (complex<float>*)C, 
    hA, hB, hC, Ms, Ks, Ns);

    cuComplex alpha, beta;
    alpha.x = 1.0; 
    alpha.y = beta.x = beta.y = 0;

    
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    int NSTEP = 5;
    int STREAMSIZE = 2;
    cudaStream_t * streamptr = new cudaStream_t[STREAMSIZE];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[STREAMSIZE];
    for(int i=0; i<STREAMSIZE; i++) cudaStreamCreate(&streamptr[i]);
    for(int i=0; i<STREAMSIZE; i++) cublasCreate_v2(&cublashandleptr[i]);
    bool *graphCreated;
    graphCreated = new bool[STREAMSIZE];
    for(int i=0; i<STREAMSIZE; i++) graphCreated[i] = false;

    for(int sid=0; sid < STREAMSIZE; sid++){
        if(!graphCreated[sid]){
            cudaStreamBeginCapture(streamptr[sid], cudaStreamCaptureModeGlobal);
            cublasSetStream_v2(cublashandleptr[sid], streamptr[sid]);
            for(int i=0; i<batchsize; i++){
                cublasgemv(cublashandleptr[sid], CUBLAS_OP_N,
                Ms[i], Ks[i],
                &alpha, (const cuComplex*)Abatchpointer_h[i], Ms[i], 
                (const cuComplex*)Bbatchpointer_h[i], 1, &beta, 
                Cbatchpointer_h[i], 1);
            }
            cudaStreamEndCapture(streamptr[sid], &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated[sid]=true;
        }
    }

    for(int istep=0; istep<NSTEP; istep++){
        #pragma omp parallel for 
        for(int sid=0; sid<STREAMSIZE; sid++){
            cudaGraphLaunch(instance, streamptr[sid]);
        }
        for(int sid=0; sid < STREAMSIZE; sid++)
        cudaStreamSynchronize(streamptr[sid]);
    }

    

    size_t Atotalelems, Btotalelems, Ctotalelems;
    tlrmat::CaluclateTotalElements(Ms, Ks, Ns, Atotalelems, Btotalelems, Ctotalelems);
    complex<float> *checkC = new complex<float>[Ctotalelems];
    CopyDataB2HD(hA, hB, checkC, (complex<float>*)A,
    (complex<float>*)B, (complex<float>*)C, Ms, Ks, Ns);
    for(int i=0; i<Ctotalelems; i++){
        ASSERT_EQ(checkC[i].real(), Ks[0]);
    }
    delete[] checkC;

    FreeDeviceMemoryBatched(A,B,C,Abatchpointer, Bbatchpointer,
    Cbatchpointer);
    FreeHostMemoryBatched(hA, hB, hC, hAbp, hBbp, hCbp);
    delete[] Abatchpointer_h;
    delete[] Bbatchpointer_h;
    delete[] Cbatchpointer_h;
}





class MyTestEnvironment : public testing::Environment {
 public:
  explicit MyTestEnvironment(const vector<string> &command_line_arg) {
    g_command_line_arg_vec = command_line_arg;
  }
};

int main(int argc, char **argv) {
  vector<string> command_line_arg_vec;
  testing::InitGoogleTest(&argc, argv);
  for(int i=0; i<argc-1; i++){
      char tmp[200];
      sprintf(tmp, "%s", argv[i+1]);
      command_line_arg_vec.push_back(string(tmp));
  }
  testing::AddGlobalTestEnvironment(new MyTestEnvironment(command_line_arg_vec));
  return RUN_ALL_TESTS();
}