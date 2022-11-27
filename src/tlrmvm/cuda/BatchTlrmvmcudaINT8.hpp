//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#pragma once

#include <vector>
#include <memory>
using std::vector;


#include "../cpu/TlrmvmCPU.hpp"
#include "Tlrmvmcuda.hpp"
#include "tlrmvmcudautil.hpp"
#include "TlrmvmMPint8.hpp"
#include <memory>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace cudatlrmvm
{

    template<typename HostType, typename DeviceType>
    class BatchTlrmvmcudaINT8
    {
    public:
        explicit BatchTlrmvmcudaINT8(vector<TlrmvmConfig> tlrmvmconfigvec);
        BatchTlrmvmcudaINT8();
        void StreamInit(int streamsize);
        void StreamDestroy();
        void MemoryInit();
        void MemoryFree();
        void Phase1();
        void Phase1Transpose();
        void Phase1Prepare();
        void Phase2();
        void Phase2Transpose();
        void Phase2Prepare();
        void Phase2PrepareTranspose();
        void Phase3();
        void Phase3Transpose();
        void Phase3Prepare();
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
        cudaStream_t * streamptr;
        cublasHandle_t * cublashandleptr;
        int stream_size;

        MultiGraph multigraph;
        MultiGraph transposemultigraph;
        SingleGraph singlegraph;
        SingleGraph transposesinglegraph;

//        minmax_unary_op unary_op;
//        minmax_binary_op binary_op;
        cuHalfComplex init;
        // These 3 pointers are actual data host.
        // the length size is whole frequency number.
        // You will use these pointers to store data.
        vector<CUDAI8basesPointers> Ubases;
        vector<CUDAI8basesPointers> Vbases;
        vector<CUDAI8XPointers> xinput;
        vector<cuHalfComplex*> P3Ahalfbuffer;
        vector<cuHalfComplex*> P3xhalfbuffer;
        vector<CUDAI8XPointers> p3xint8;
        vector<cuHalfComplex**> P3maxxinfo;
        vector<cuComplex*> p3xreductionbuffer;
//        cuComplex* p3xreductionbuffer_device;
        CBMaxInfo cbmaxinfo;
        size_t * *d_phase2mapping_vec;
        size_t * *d_phase2mapping2_vec;
        size_t * *d_phase2mapping_transpose_vec;
        vector<HostType> finalresults;
    };
}

