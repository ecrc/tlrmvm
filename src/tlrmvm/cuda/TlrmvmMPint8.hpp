//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#pragma once
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h> // if you need CUBLAS v2, include before magma.h

#include "cudakernel.cuh"
#include "../../common/Common.hpp"
#include "../cpu/TlrmvmCPU.hpp"
#include "../../common/cuda/Util.hpp"
#include "Tlrmvmcuda.hpp"
#include "tlrmvmcudautil.hpp"

namespace cudatlrmvm
{

    struct CUDAI8Phase1Pointers{
        CUDAI8Phase1Pointers();
        size_t Acnt;
        size_t Xcnt;
        size_t Ycnt;
        vector<size_t> Ms;
        vector<size_t> Ks;
        vector<size_t> Ns;
        vector<cuComplex> maxA;
        vector<cuComplex> maxx;
        cuInt8Complex *A;
        cuInt8Complex *x;
        cuHalfComplex *y;
        cuInt8Complex **Abp;
        cuInt8Complex **xbp;
        cuHalfComplex **ybp;
    };

    struct CUDAI8Phase3Pointers{
        CUDAI8Phase3Pointers();
        size_t Acnt;
        size_t Xcnt;
        size_t Ycnt;
        vector<size_t> Ms;
        vector<size_t> Ks;
        vector<size_t> Ns;
        cuHalfComplex *A;
        cuHalfComplex *x;
        cuHalfComplex *y;
        cuHalfComplex **Abp;
        cuHalfComplex **xbp;
        cuHalfComplex **ybp;
    };


    class TlrmvmMPint8
    {
    public:
        explicit TlrmvmMPint8(TlrmvmConfig tlrmvmconfig);
        TlrmvmMPint8();
        void UpdateConfig(TlrmvmConfig &tlrmvmConfig);
        void SetTransposeConjugate(bool transpose, bool conjugate);
        void StreamInit(int streamsize);
        void StreamDestroy();
        void MemoryInit();
        void MemoryFree();
        void Phase1();
        void Phase1Transpose();
        void Phase1GetMembuffer();
        void AllocatePhase1Buffer();
        void Phase1CopyData();
        void Phase1GetMembufferTranspose();
        void AllocatePhase1BufferTranspose();
        void Phase1CopyDataTranspose();
        void Phase2();
        void Phase2Transpose();
        void Phase2Prepare();
        void Phase2PrepareTranspose();
        void Phase3();
        void Phase3Transpose();
        void Phase3GetMembuffer();
        void AllocatePhase3Buffer();
        void Phase3CopyData();
        void Phase3GetMembufferTranspose();
        void AllocatePhase3BufferTranspose();
        void Phase3CopyDataTranspose();
        void MVM();
        void MVMTranspose();
        void MVMNoTranspose();
        void MVMGraph();
        void MVMGraphTranspose();
        void MVMGraphNoTranspose();
        void setX(complex<float> * xvector, size_t xlength);
        // seperate 2 functions.
        void TryConjugateXvec();
        void TryConjugateResults();
        void CopyBackResults();

        bool transpose;
        bool conjugate;
        // cpu instance
        TlrmvmConfig config;
        shared_ptr<TlrmvmCPU<complex<float>>> tlrmvmcpu;
        // GPU resources
        cudaStream_t * streamptr;
        cublasHandle_t * cublashandleptr;
        int stream_size;
        cudaGraph_t graph;
        bool graphCreated;
        cudaGraphExec_t instance;
        cudaEvent_t *events;
        cudaEvent_t event_start;
        cudaEvent_t event_phase2finish;

        cuComplex alpha;
        cuComplex beta;

        size_t *d_phase2mapping;
        size_t *d_phase2mapping_transpose;
        // gpu pointers, fp16 data buffer
        CUDAI8Phase1Pointers cudap1ptrs;
//        CUDAPhasePointers<int8_t> cudap1transptrs;
        CUDAI8Phase3Pointers cudap3ptrs;
//        CUDAPhasePointers<int8_t> cudap3transptrs;

    };

} //



