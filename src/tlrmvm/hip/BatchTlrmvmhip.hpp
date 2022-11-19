#pragma once

#include <vector>
using std::vector;


#include "../cpu/TlrmvmCPU.hpp"
#include "Tlrmvmhip.hpp"
#include "tlrmvmhiputil.hpp"
#include <memory>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace hiptlrmvm
{
    template<typename HostType, typename DeviceType>
    class BatchTlrmvmhip
    {
    public:
        explicit BatchTlrmvmhip(vector<TlrmvmConfig> tlrmvmconfigvec);
        BatchTlrmvmhip();
        void StreamInit(int streamsize);
        void StreamDestroy();
        void MemoryInit();
        void MemoryFree();
        void Phase1();
        void Phase1Transpose();
        void Phase1Prepare();
        void Phase1PrepareTranspose();
        void Phase2();
        void Phase2Transpose();
        void Phase2Prepare();
        void Phase2PrepareTranspose();
        void Phase3();
        void Phase3Transpose();
        void Phase3Prepare();
        void Phase3PrepareTranspose();
        void MVM_SingleGraph();
        void MVM_SingleGraphTranspose();
        void MVM_SingleGraphNoTranspose();
        void MVM_MultiGraph();
        void MVM_MultiGraphTranspose();
        void MVM_MultiGraphNoTranspose();

        // seperate 2 functions.
        void SetTransposeConjugate(bool transpose, bool conjugate);
        void setX(HostType * xvector, size_t xlength);
        void TryConjugateXvec();
        void TryConjugateResults();
        void CopyBackResults();

        bool transpose;
        bool conjugate;

        int batchsize;
        // cpu instance
        vector<TlrmvmConfig> config_vec;
        vector<std::shared_ptr<TlrmvmCPU<HostType>>> cpuinstvec;

        // GPU resources
        hipStream_t * streamptr;
        hipblasHandle_t * cublashandleptr;
        int stream_size;

        MultiGraph multigraph;
        MultiGraph transposemultigraph;
        SingleGraph singlegraph;
        SingleGraph transposesinglegraph;

        DeviceType alpha;
        DeviceType beta;
        // gpu pointers
        vector<HIPPhasePointers<DeviceType>> cudap1ptrs_vec;
        vector<HIPPhasePointers<DeviceType>> cudap1transptrs_vec;
        size_t * *d_phase2mapping_vec;
        size_t * *d_phase2mapping_transpose_vec;
        vector<HIPPhasePointers<DeviceType>> cudap3ptrs_vec;
        vector<HIPPhasePointers<DeviceType>> cudap3transptrs_vec;
        vector<HostType> finalresults;
    };
}

