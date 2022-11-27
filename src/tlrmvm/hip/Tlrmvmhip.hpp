//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#pragma once

#include "../../common/Common.hpp"
#include "../cpu/TlrmvmCPU.hpp"
#include "../../common/hip/Util.hpp"
#include <cassert>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas.h>
#include <memory>

namespace hiptlrmvm
{
    template<typename T>
    struct HIPPhasePointers{
        HIPPhasePointers();
        size_t Acnt;
        size_t Xcnt;
        size_t Ycnt;
        vector<size_t> Ms;
        vector<size_t> Ks;
        vector<size_t> Ns;
        T *A;
        T *x;
        T *y;
        T **Abp;
        T **xbp;
        T **ybp;
    };

    template<typename SrcType, typename DestType>
    void PhasePointersCopyNonPointers(HIPPhasePointers<DestType> &dest, const PhasePointers<SrcType> &src);

    // Tlrmvm cuda is only responsible for cuda memory ops.
    // Any host memory related ops should go to CPU instance.
    template<typename HostType, typename DeviceType>
    class Tlrmvmhip
    {
    public:
        explicit Tlrmvmhip(TlrmvmConfig tlrmvmconfig);
        Tlrmvmhip();
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
        hipStream_t * streamptr;
        hipblasHandle_t * cublashandleptr;
        int stream_size;
        hipGraph_t graph;
        bool graphCreated;
        hipGraphExec_t instance;
        hipEvent_t *events;
        hipEvent_t event_start;
        hipEvent_t event_phase2finish;

        DeviceType alpha;
        DeviceType beta;

        // gpu pointers
        HIPPhasePointers<DeviceType> cudap1ptrs;
        HIPPhasePointers<DeviceType> cudap1transptrs;
        size_t *d_phase2mapping;
        size_t *d_phase2mapping_transpose;
        HIPPhasePointers<DeviceType> cudap3ptrs;
        HIPPhasePointers<DeviceType> cudap3transptrs;
    };




} //



