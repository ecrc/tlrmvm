#pragma once

#include "Tlrmvmcuda.hpp"


namespace cudatlrmvm {


    template<typename HostType, typename DeviceType>
    class TlrmvmcudaConstRank
    {
    public:
        explicit TlrmvmcudaConstRank(TlrmvmConfig tlrmvmconfig);
        TlrmvmcudaConstRank();
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
        cudaStream_t stream;
        cublasHandle_t cublashandle;

        DeviceType alpha;
        DeviceType beta;

        // gpu pointers
        CUDAPhasePointers<DeviceType> cudap1ptrs;
        CUDAPhasePointers<DeviceType> cudap1transptrs;
        size_t *d_phase2mapping;
        size_t *d_phase2mapping_transpose;
        CUDAPhasePointers<DeviceType> cudap3ptrs;
        CUDAPhasePointers<DeviceType> cudap3transptrs;

        DeviceType **d_p1Aptrs;
        DeviceType **d_p1xptrs;
        DeviceType **d_p1yptrs;
        DeviceType **d_p3Aptrs;
        DeviceType **d_p3xptrs;
        DeviceType **d_p3yptrs;

        DeviceType **d_p1transAptrs;
        DeviceType **d_p1transxptrs;
        DeviceType **d_p1transyptrs;
        DeviceType **d_p3transAptrs;
        DeviceType **d_p3transxptrs;
        DeviceType **d_p3transyptrs;
    };
}