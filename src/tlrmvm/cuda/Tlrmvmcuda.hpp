//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#pragma once

#include "../../common/Common.hpp"
#include "../cpu/TlrmvmCPU.hpp"
#include "../../common/cuda/Util.hpp"
#include "tlrmvmcudautil.hpp"
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h> // if you need CUBLAS v2, include before magma.hw
#include <memory>

namespace cudatlrmvm
{
    template<typename T>
    struct CUDAPhasePointers{
        CUDAPhasePointers();
        size_t Acnt;
        size_t Xcnt;
        size_t Ycnt;
        vector<size_t> Ms;
        vector<size_t> Ks;
        vector<size_t> Ns;
        T *A;
        T *x;
        T *y;
        T ** ysplits;
        T **Abp;
        T **xbp;
        T **ybp;
        T *** ysplitsbp;
    };

    template<typename SrcType, typename DestType>
    void PhasePointersCopyNonPointers(CUDAPhasePointers<DestType> &dest, const PhasePointers<SrcType> &src);

    void I8PhasePointersCopyNonPointers(CUDAI8basesPointers& dest, const PhasePointers<complex<float>>& src);
    // Tlrmvm cuda is only responsible for cuda memory ops.
    // Any host memory related ops should go to CPU instance.
    template<typename HostType, typename DeviceType>
    class Tlrmvmcuda
    {
        public:
        explicit Tlrmvmcuda(TlrmvmConfig tlrmvmconfig);
        Tlrmvmcuda();
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
        void setX(HostType * xvector, size_t xlength);
        // seperate 2 functions.
        void TryConjugateXvec();
        void TryConjugateResults();
        void CopyBackResults();

        bool transpose;
        bool conjugate;
        // cpu instance
        TlrmvmConfig config;
        shared_ptr<TlrmvmCPU<HostType>> tlrmvmcpu;
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

        DeviceType alpha;
        DeviceType beta;

        // gpu pointers
        CUDAPhasePointers<DeviceType> cudap1ptrs;
        CUDAPhasePointers<DeviceType> cudap1transptrs;
        size_t *d_phase2mapping;
        size_t *d_phase2mapping_transpose;
        CUDAPhasePointers<DeviceType> cudap3ptrs;
        CUDAPhasePointers<DeviceType> cudap3transptrs;
    };




} // 



