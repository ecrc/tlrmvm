//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

//
// Created by Yuxi Hong on 02/03/2022.
//

#include "../../common/Common.hpp"
#include "../cpu/TlrmvmCPU.hpp"
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include "BatchTlrmvmcuda.hpp"
#include "BatchTlrmvmcudaFP16.hpp"
#include "cudakernel.cuh"
#include <chrono>

namespace cudatlrmvm
{
    template<typename HostType, typename DeviceType>
    BatchTlrmvmcudaFP16<HostType, DeviceType>::BatchTlrmvmcudaFP16(vector<TlrmvmConfig> tlrmvmconfigvec)
    :config_vec(tlrmvmconfigvec),batchsize(tlrmvmconfigvec.size())
    {
        cout << "calling Batch TlrmvmcudaFP16" << endl;
#ifdef USE_MPI
        int initflag;
        MPI_Initialized(&initflag);
        if(initflag == 1){
            int rank;
            int size;
            MPI_Comm_rank(MPI_COMM_WORLD,&rank);
            MPI_Comm_size(MPI_COMM_WORLD,&size);
            if(rank == 0)
                cout << "we are in mpi environment:" << endl;
            int totaldevcount = 0;
            CUDACHECK(cudaGetDeviceCount(&totaldevcount));
            if(totaldevcount < size){
                if(rank == 0)
                    cout << "not enough cards, in debug mode, set all to 0." << endl;
                CUDACHECK(cudaSetDevice(0));
            }else{
                if(rank == 0)
                    cout << "we have enough cards, set to different cards." << endl;
                CUDACHECK(cudaSetDevice(rank%8));
            }
        }
#endif
        cpuinstvec.resize(tlrmvmconfigvec.size());
        for(int i=0; i<tlrmvmconfigvec.size(); i++)
            cpuinstvec[i] = std::move(make_shared<TlrmvmCPU<HostType>>(tlrmvmconfigvec[i]));
        finalresults.resize(tlrmvmconfigvec.size() * tlrmvmconfigvec[0].originM);
    }

    template<typename HostType, typename DeviceType>
    BatchTlrmvmcudaFP16<HostType, DeviceType>::BatchTlrmvmcudaFP16(){}
    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::StreamInit(int streamsize){
        this->stream_size = streamsize;
        streamptr = new cudaStream_t[streamsize];
        cublashandleptr = new cublasHandle_t[streamsize];
        for(int i=0; i<streamsize; i++)
            cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
        for(int i=0; i<streamsize; i++)
            cublasCreate(&cublashandleptr[i]);
        for(int i=0; i<streamsize; i++)
            cublasSetStream(cublashandleptr[i], streamptr[i]);
        multigraph.StreamInit(batchsize, streamsize);
        transposemultigraph.StreamInit(batchsize, streamsize);
        singlegraph.StreamInit(streamsize);
        transposesinglegraph.StreamInit(streamsize);
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::MemoryInit(){
        cudap1ptrs_vec.resize(batchsize);
        cudap3ptrs_vec.resize(batchsize);
        cudap1transptrs_vec.resize(batchsize);
        cudap3transptrs_vec.resize(batchsize);
        auto start = std::chrono::steady_clock::now();
#pragma omp parallel for default(none)
        for(int bi=0; bi<batchsize; bi++){
            cpuinstvec[bi]->MemoryInit();
            PhasePointersCopyNonPointers<HostType, DeviceType>
                    (cudap1ptrs_vec[bi],cpuinstvec[bi]->p1ptrs);
            PhasePointersCopyNonPointers<HostType, DeviceType>
                    (cudap3ptrs_vec[bi],cpuinstvec[bi]->p3ptrs);
            PhasePointersCopyNonPointers<HostType, DeviceType>(
                    cudap1transptrs_vec[bi],cpuinstvec[bi]->p1transptrs);
            PhasePointersCopyNonPointers<HostType, DeviceType>(
                    cudap3transptrs_vec[bi],cpuinstvec[bi]->p3transptrs);
        }
        auto end = std::chrono::steady_clock::now();
        auto elapse_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
#ifdef USE_MPI
        int initflag;
        MPI_Initialized(&initflag);
        if(initflag == 1){
            int rank;
            int size;
            MPI_Comm_rank(MPI_COMM_WORLD,&rank);
            MPI_Comm_size(MPI_COMM_WORLD,&size);
            MPI_Barrier(MPI_COMM_WORLD);
            auto recv_buffer = elapse_time;
            MPI_Allreduce(&elapse_time, &recv_buffer, 1,
                          MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
            if(rank == 0) cout << "Reading data buffer takes time is " << recv_buffer * 1e-6 << " seconds."<< endl;
        }
#else
        cout << "Reading data buffer takes time is " << elapse_time * 1e-6 << endl;
#endif
        Phase1Prepare();
        Phase2Prepare();
        Phase3Prepare();
        // transpose
        Phase1PrepareTranspose();
        Phase2PrepareTranspose();
        Phase3PrepareTranspose();
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType, DeviceType>::SetTransposeConjugate(bool transpose, bool conjugate){
        this->transpose = transpose;
        this->conjugate = conjugate;
        for(auto &x : cpuinstvec) x->SetTransposeConjugate(transpose, conjugate);
    }


    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::setX(HostType * xvector, size_t xlength){
        int offset = 0;
        assert(xlength == config_vec[0].originN * batchsize);
        for(int i=0; i<cpuinstvec.size(); i++){
            cpuinstvec[i]->setX(xvector + offset , config_vec[i].originN);
            offset += config_vec[i].originN;
        }
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::TryConjugateXvec() {
        for(int bi=0; bi<batchsize; bi++){
            // no transpose logic
            cpuinstvec[bi]->TryConjugateXvec();
            CopyDataB2HD(cudap1ptrs_vec[bi].x, cpuinstvec[bi]->p1ptrs.x, cpuinstvec[bi]->xmat.Shape()[0]);
        }
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::Phase1(){
        cudaDeviceSynchronize();
        for(int bi=0; bi < batchsize; bi++){
            for(int i=0; i<config_vec[bi].Ntg; i++){
                if(cudap1ptrs_vec[bi].Ms[i] != 0){
                    Hgemv_Phase1_driver(cudap1ptrs_vec[bi].Abp[i], cudap1ptrs_vec[bi].xbp[i], cudap1ptrs_vec[bi].ybp[i],
                                        cudap1ptrs_vec[bi].Ms[i], cudap1ptrs_vec[bi].Ks[i], streamptr[i % stream_size]);
                }
            }
        }
        cudaDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::Phase1Transpose(){
        cudaDeviceSynchronize();
        for(int bi=0; bi < batchsize; bi++){
            for(int i=0; i<config_vec[bi].Ntg; i++){
                if(cudap1transptrs_vec[bi].Ms[i] != 0){
                    Hgemv_Phase1_Transpose_driver(cudap1transptrs_vec[bi].Abp[i], cudap1transptrs_vec[bi].xbp[i],
                                                  cudap1transptrs_vec[bi].ybp[i],
                               cudap1transptrs_vec[bi].Ms[i], cudap1transptrs_vec[bi].Ks[i],
                                                  streamptr[i % stream_size]);
                }
            }
        }
        cudaDeviceSynchronize();
    }
    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::Phase1Prepare() {

        for(int bi=0; bi<batchsize; bi++){
            int curbatch = cudap1ptrs_vec[bi].Ms.size();
            GetcuHostMemory(&cudap1ptrs_vec[bi].Abp, curbatch);
            GetcuHostMemory(&cudap1ptrs_vec[bi].xbp, curbatch);
            GetcuHostMemory(&cudap1ptrs_vec[bi].ybp, curbatch);
        }
        for(int bi=0; bi<batchsize; bi++){
            GetDeviceMemory(&cudap1ptrs_vec[bi].A, cudap1ptrs_vec[bi].Acnt);
            GetDeviceMemory(&cudap1ptrs_vec[bi].x, cudap1ptrs_vec[bi].Xcnt);
            GetDeviceMemory(&cudap1ptrs_vec[bi].y, cudap1ptrs_vec[bi].Ycnt);
            cudap1ptrs_vec[bi].Abp[0] = cudap1ptrs_vec[bi].A;
            cudap1ptrs_vec[bi].xbp[0] = cudap1ptrs_vec[bi].x;
            cudap1ptrs_vec[bi].ybp[0] = cudap1ptrs_vec[bi].y;
        }
        for(int bi=0; bi<batchsize; bi++){
            auto AvMs = cudap1ptrs_vec[bi].Ms;
            auto AvNs = cudap1ptrs_vec[bi].Ns;
            auto AvKs = cudap1ptrs_vec[bi].Ks;
            for(int i=1; i<config_vec[bi].Ntg; i++){
                size_t AvMK = AvMs[i-1] * AvKs[i-1];
                size_t AvKN = AvKs[i-1] * AvNs[i-1];
                size_t AvMN = AvMs[i-1] * AvNs[i-1];
                cudap1ptrs_vec[bi].Abp[i] =cudap1ptrs_vec[bi].Abp[i-1] + AvMK;
                cudap1ptrs_vec[bi].xbp[i] = cudap1ptrs_vec[bi].xbp[i-1] + AvKN;
                cudap1ptrs_vec[bi].ybp[i] = cudap1ptrs_vec[bi].ybp[i-1] + AvMN;
            }
            // load phase1 A,x to GPU
            CopyDataB2HD(cudap1ptrs_vec[bi].A, cpuinstvec[bi]->p1ptrs.A, cpuinstvec[bi]->p1ptrs.Acnt);
            CopyDataB2HD(cudap1ptrs_vec[bi].x, cpuinstvec[bi]->p1ptrs.x, cpuinstvec[bi]->p1ptrs.Xcnt);
        }

    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::Phase1PrepareTranspose() {
        for(int bi=0; bi < batchsize; bi++){
            int curbatch = cudap1ptrs_vec[bi].Ms.size();
            GetcuHostMemory(&cudap1transptrs_vec[bi].Abp, curbatch);
            GetcuHostMemory(&cudap1transptrs_vec[bi].xbp, curbatch);
            GetcuHostMemory(&cudap1transptrs_vec[bi].ybp, curbatch);
        }
        for(int bi=0; bi<batchsize; bi++){
            cudap1transptrs_vec[bi].A = cudap3ptrs_vec[bi].A;
            cudap1transptrs_vec[bi].x = cudap1ptrs_vec[bi].x;
            GetDeviceMemory(&cudap1transptrs_vec[bi].y, cudap1transptrs_vec[bi].Ycnt);
            cudap1transptrs_vec[bi].Abp[0] = cudap3ptrs_vec[bi].A; // use phase 3, U bases
            cudap1transptrs_vec[bi].xbp[0] = cudap1ptrs_vec[bi].x; // use phase 1, x
            cudap1transptrs_vec[bi].ybp[0] = cudap1transptrs_vec[bi].y; // create a new buffer
        }
        for(int bi=0; bi<batchsize; bi++){
            for(int i=1; i<cudap1transptrs_vec[bi].Ms.size(); i++){
                size_t AvMK = cudap1transptrs_vec[bi].Ms[i-1] * cudap1transptrs_vec[bi].Ks[i-1];
                size_t AvKN = cudap1transptrs_vec[bi].Ks[i-1] * cudap1transptrs_vec[bi].Ns[i-1];
                size_t AvMN = cudap1transptrs_vec[bi].Ms[i-1] * cudap1transptrs_vec[bi].Ns[i-1];
                cudap1transptrs_vec[bi].Abp[i] = cudap1transptrs_vec[bi].Abp[i-1] + AvMK;
                cudap1transptrs_vec[bi].xbp[i] = cudap1transptrs_vec[bi].xbp[i-1] + AvKN;
                cudap1transptrs_vec[bi].ybp[i] = cudap1transptrs_vec[bi].ybp[i-1] + AvMN;
            }
        }
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::Phase2(){
        cudaDeviceSynchronize();
        for(int bi=0; bi<batchsize; bi++){
            phase2_nosplit<DeviceType>(cudap1ptrs_vec[bi].y, d_phase2mapping_vec[bi],
                                       cudap3ptrs_vec[bi].x,
                                       config_vec[bi].granksum, streamptr[bi % stream_size]);
        }
        cudaDeviceSynchronize();
    }
    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::Phase2Transpose(){
        cudaDeviceSynchronize();
        for(int bi=0; bi<batchsize; bi++){
            phase2_nosplit<DeviceType>(cudap1transptrs_vec[bi].y, d_phase2mapping_transpose_vec[bi],
                                       cudap3transptrs_vec[bi].x, config_vec[bi].granksum,
                                       streamptr[bi%stream_size]);
        }
        cudaDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::Phase2Prepare(){
        d_phase2mapping_vec = new size_t*[batchsize];
        for(int bi=0; bi<batchsize; bi++){
            GetDeviceMemory(&d_phase2mapping_vec[bi], cpuinstvec[bi]->h_phase2mapping.size());
            CopyDataB2HD(d_phase2mapping_vec[bi], cpuinstvec[bi]->h_phase2mapping.data(),
                         cpuinstvec[bi]->h_phase2mapping.size());
        }
    }
    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::Phase2PrepareTranspose(){
        d_phase2mapping_transpose_vec = new size_t*[batchsize];
        for(int bi=0; bi<batchsize; bi++){
            GetDeviceMemory(&d_phase2mapping_transpose_vec[bi], cpuinstvec[bi]->h_phase2mappingTranspose.size());
            CopyDataB2HD(d_phase2mapping_transpose_vec[bi], cpuinstvec[bi]->h_phase2mappingTranspose.data(),
                         cpuinstvec[bi]->h_phase2mappingTranspose.size());
        }
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::Phase3(){
        cudaDeviceSynchronize();
        for(int bi=0; bi<batchsize; bi++){
            for(int i=0; i<config_vec[bi].Mtg; i++){
                Hgemv_Phase3_driver(cudap3ptrs_vec[bi].Abp[i],cudap3ptrs_vec[bi].xbp[i], cudap3ptrs_vec[bi].ybp[i],
                                    cudap3ptrs_vec[bi].Ms[i], cudap3ptrs_vec[bi].Ks[i], streamptr[i % stream_size]);
            }
        }
        cudaDeviceSynchronize();
        CUDACHECK(cudaGetLastError());
    }
    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::Phase3Transpose(){
        cudaDeviceSynchronize();
        for(int bi=0; bi<batchsize; bi++){
            for(int i=0; i<config_vec[bi].Mtg; i++){
                Hgemv_Phase3_Transpose_driver(cudap3transptrs_vec[bi].Abp[i], cudap3transptrs_vec[bi].xbp[i],
                                              cudap3transptrs_vec[bi].ybp[i],
                                              cudap3transptrs_vec[bi].Ms[i], cudap3transptrs_vec[bi].Ks[i],
                                              streamptr[i % stream_size]);
            }
        }
        cudaDeviceSynchronize();
    }
    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::Phase3Prepare() {
        for(int bi=0; bi<batchsize; bi++){
            int curbatch = cudap3ptrs_vec[bi].Ms.size();
            GetcuHostMemory(&cudap3ptrs_vec[bi].Abp, curbatch);
            GetcuHostMemory(&cudap3ptrs_vec[bi].xbp, curbatch);
            GetcuHostMemory(&cudap3ptrs_vec[bi].ybp, curbatch);
            // indev
            GetcuHostMemory(&cudap3ptrs_vec[bi].ysplitsbp, 4);
            GetcuHostMemory(&cudap3ptrs_vec[bi].ysplitsbp[0], curbatch);
            GetcuHostMemory(&cudap3ptrs_vec[bi].ysplitsbp[1], curbatch);
            GetcuHostMemory(&cudap3ptrs_vec[bi].ysplitsbp[2], curbatch);
            GetcuHostMemory(&cudap3ptrs_vec[bi].ysplitsbp[3], curbatch);
        }
        for(int bi=0; bi<batchsize; bi++){
            GetDeviceMemory(&cudap3ptrs_vec[bi].A, cudap3ptrs_vec[bi].Acnt);
            GetDeviceMemory(&cudap3ptrs_vec[bi].x, cudap3ptrs_vec[bi].Xcnt);
            GetDeviceMemory(&cudap3ptrs_vec[bi].y, cudap3ptrs_vec[bi].Ycnt);
            // indev
            GetcuHostMemory(&cudap3ptrs_vec[bi].ysplits, 4); // 4 pointers still on host
            GetDeviceMemory(&cudap3ptrs_vec[bi].ysplits[0], cudap3ptrs_vec[bi].Ycnt); // then each mem is on device
            GetDeviceMemory(&cudap3ptrs_vec[bi].ysplits[1], cudap3ptrs_vec[bi].Ycnt);
            GetDeviceMemory(&cudap3ptrs_vec[bi].ysplits[2], cudap3ptrs_vec[bi].Ycnt);
            GetDeviceMemory(&cudap3ptrs_vec[bi].ysplits[3], cudap3ptrs_vec[bi].Ycnt);
            cudap3ptrs_vec[bi].Abp[0] = cudap3ptrs_vec[bi].A;
            cudap3ptrs_vec[bi].xbp[0] = cudap3ptrs_vec[bi].x;
            cudap3ptrs_vec[bi].ybp[0] = cudap3ptrs_vec[bi].y;
        }
        for(int bi=0; bi<batchsize; bi++){
            auto AuMs = cudap3ptrs_vec[bi].Ms;
            auto AuNs = cudap3ptrs_vec[bi].Ns;
            auto AuKs = cudap3ptrs_vec[bi].Ks;
            for(int i=1; i<cpuinstvec[bi]->config.Mtg; i++){
                size_t AuMK = AuMs[i-1] * AuKs[i-1];
                size_t AuKN = AuKs[i-1] * AuNs[i-1];
                size_t AuMN = AuMs[i-1] * AuNs[i-1];
                cudap3ptrs_vec[bi].Abp[i] = cudap3ptrs_vec[bi].Abp[i-1] + AuMK;
                cudap3ptrs_vec[bi].xbp[i] = cudap3ptrs_vec[bi].xbp[i-1] + AuKN;
                cudap3ptrs_vec[bi].ybp[i] = cudap3ptrs_vec[bi].ybp[i-1] + AuMN;
            }
            // load phase 3 A to GPU
            CopyDataB2HD(cudap3ptrs_vec[bi].A, cpuinstvec[bi]->p3ptrs.A, cudap3ptrs_vec[bi].Acnt);
        }
    }
    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::Phase3PrepareTranspose() {
        for(int bi=0; bi<batchsize; bi++){
            int curbatch = cudap3transptrs_vec[bi].Ms.size();
            GetcuHostMemory(&cudap3transptrs_vec[bi].Abp, curbatch);
            GetcuHostMemory(&cudap3transptrs_vec[bi].xbp, curbatch);
            GetcuHostMemory(&cudap3transptrs_vec[bi].ybp, curbatch);
        }
        for(int bi=0; bi<batchsize; bi++){
            cudap3transptrs_vec[bi].A = cudap1ptrs_vec[bi].A;
            cudap3transptrs_vec[bi].x = cudap3ptrs_vec[bi].x;
            GetDeviceMemory(&cudap3transptrs_vec[bi].y, cudap3transptrs_vec[bi].Ycnt);
            cudap3transptrs_vec[bi].Abp[0] = cudap1ptrs_vec[bi].A; // use phase 1, V bases
            cudap3transptrs_vec[bi].xbp[0] = cudap3ptrs_vec[bi].x; // use phase 3, x
            cudap3transptrs_vec[bi].ybp[0] = cudap3transptrs_vec[bi].y; // create a new buffer
        }
        for(int bi=0; bi<batchsize; bi++){
            for(int i=1; i<cudap3transptrs_vec[bi].Ms.size(); i++){
                size_t AvMK = cudap3transptrs_vec[bi].Ms[i-1] * cudap3transptrs_vec[bi].Ks[i-1];
                size_t AvKN = cudap3transptrs_vec[bi].Ks[i-1] * cudap3transptrs_vec[bi].Ns[i-1];
                size_t AvMN = cudap3transptrs_vec[bi].Ms[i-1] * cudap3transptrs_vec[bi].Ns[i-1];
                cudap3transptrs_vec[bi].Abp[i] = cudap3transptrs_vec[bi].Abp[i-1] + AvMK;
                cudap3transptrs_vec[bi].xbp[i] = cudap3transptrs_vec[bi].xbp[i-1] + AvKN;
                cudap3transptrs_vec[bi].ybp[i] = cudap3transptrs_vec[bi].ybp[i-1] + AvMN;
            }
            // no need to copy data.
        }
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::MVM_MultiGraph(){
        if(transpose){
            MVM_MultiGraphTranspose();
        }else{
            MVM_MultiGraphNoTranspose();
        }
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::MVM_MultiGraphTranspose(){
        auto & graphCreated = transposemultigraph.graphCreated;
        auto & event_start = transposemultigraph.event_start;
        auto & events = transposemultigraph.events;
        auto & graph = transposemultigraph.graph;
        auto & instance = transposemultigraph.instance;
        for(int bi=0; bi<batchsize; bi++){
            if(!graphCreated[bi]){
                cudaStreamBeginCapture(streamptr[0],cudaStreamCaptureModeGlobal);
                cudaEventRecord(event_start[bi], streamptr[0]);
                for(int streami=1; streami<stream_size; streami++){
                    cudaStreamWaitEvent(streamptr[streami], event_start[bi]);
                }
                // phase 1 transpose
                for(int i=0; i<config_vec[bi].Ntg; i++){
                    if(cudap1transptrs_vec[bi].Ms[i] != 0){
                        Hgemv_Phase1_Transpose_driver(cudap1transptrs_vec[bi].Abp[i], cudap1transptrs_vec[bi].xbp[i],
                                            cudap1transptrs_vec[bi].ybp[i],
                                            cudap1transptrs_vec[bi].Ms[i], cudap1transptrs_vec[bi].Ks[i],
                                            streamptr[i % stream_size]);
                    }
                }
                for(int streami=1; streami < stream_size; streami++){
                    cudaEventRecord(events[bi][streami], streamptr[streami]);
                }
                for(int streami=1; streami < stream_size; streami++){
                    cudaStreamWaitEvent(streamptr[0], events[bi][streami]);
                }
                // phase 2 transpose
                phase2_nosplit<DeviceType>(cudap1transptrs_vec[bi].y,
                                           d_phase2mapping_transpose_vec[bi],
                                           cudap3transptrs_vec[bi].x,
                                           config_vec[bi].granksum, streamptr[0]);
                cudaEventRecord(events[bi][0], streamptr[0]);
                for(int streami=1; streami < stream_size; streami++){
                    cudaStreamWaitEvent(streamptr[streami], events[bi][0]);
                }
                // phase 3 transpose
                for(int i=0; i<config_vec[bi].Mtg; i++){
                    if(cudap3transptrs_vec[bi].Ms[i] != 0){
                        Hgemv_Phase3_Transpose_driver(cudap3transptrs_vec[bi].Abp[i], cudap3transptrs_vec[bi].xbp[i],
                                            cudap3transptrs_vec[bi].ybp[i],
                                            cudap3transptrs_vec[bi].Ms[i], cudap3transptrs_vec[bi].Ks[i],
                                            streamptr[i % stream_size]);
                    }
                }
                // final merge
                for(int streami=1; streami < stream_size; streami++){
                    cudaEventRecord(events[bi][stream_size + streami], streamptr[streami]);
                }
                for(int streami=1; streami < stream_size; streami++){
                    cudaStreamWaitEvent(streamptr[0], events[bi][stream_size + streami]);
                }
                cudaStreamEndCapture(streamptr[0], &graph[bi]);
                cudaGraphInstantiate(&instance[bi],graph[bi], nullptr, nullptr, 0);
                graphCreated[bi] = true;
            }
            cudaGraphLaunch(instance[bi], streamptr[0]);
            cudaStreamSynchronize(streamptr[0]);
        }
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::MVM_MultiGraphNoTranspose()
    {
        auto & graphCreated = multigraph.graphCreated;
        auto & event_start = multigraph.event_start;
        auto & events = multigraph.events;
        auto & graph = multigraph.graph;
        auto & instance = multigraph.instance;
        for(int bi=0; bi<batchsize; bi++){
            if(!graphCreated[bi]){
                cudaStreamBeginCapture(streamptr[0],cudaStreamCaptureModeGlobal);
                cudaEventRecord(event_start[bi], streamptr[0]);
                for(int streami=1; streami<stream_size; streami++){
                    cudaStreamWaitEvent(streamptr[streami], event_start[bi]);
                }
                // phase 1
                for(int i=0; i<config_vec[bi].Ntg; i++){
                    if(cudap1ptrs_vec[bi].Ms[i] != 0){
                        Hgemv_Phase1_driver(cudap1ptrs_vec[bi].Abp[i], cudap1ptrs_vec[bi].xbp[i], cudap1ptrs_vec[bi].ybp[i],
                                            cudap1ptrs_vec[bi].Ms[i], cudap1ptrs_vec[bi].Ks[i], streamptr[i % stream_size]);
                    }
                }
                for(int streami=1; streami < stream_size; streami++){
                    cudaEventRecord(events[bi][streami], streamptr[streami]);
                }
                for(int streami=1; streami < stream_size; streami++){
                    cudaStreamWaitEvent(streamptr[0], events[bi][streami]);
                }
                // phase 2
                phase2_nosplit<DeviceType>(cudap1ptrs_vec[bi].y,
                                           d_phase2mapping_vec[bi], cudap3ptrs_vec[bi].x,
                                           config_vec[bi].granksum, streamptr[0]);
                cudaEventRecord(events[bi][0], streamptr[0]);
                for(int streami=1; streami < stream_size; streami++){
                    cudaStreamWaitEvent(streamptr[streami], events[bi][0]);
                }
                // phase 3
                for(int i=0; i<config_vec[bi].Mtg; i++){
                    Hgemv_Phase3_driver(cudap3ptrs_vec[bi].Abp[i],cudap3ptrs_vec[bi].xbp[i], cudap3ptrs_vec[bi].ybp[i],
                                        cudap3ptrs_vec[bi].Ms[i], cudap3ptrs_vec[bi].Ks[i], streamptr[i % stream_size]);
                }
                // final merge
                for(int streami=1; streami < stream_size; streami++){
                    cudaEventRecord(events[bi][stream_size + streami], streamptr[streami]);
                }
                for(int streami=1; streami < stream_size; streami++){
                    cudaStreamWaitEvent(streamptr[0], events[bi][stream_size + streami]);
                }
                cudaStreamEndCapture(streamptr[0], &graph[bi]);
                cudaGraphInstantiate(&instance[bi],
                                     graph[bi], nullptr, nullptr, 0);
                graphCreated[bi] = true;
            }
            cudaGraphLaunch(instance[bi], streamptr[0]);
            cudaStreamSynchronize(streamptr[0]);
        }
    }



    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::MVM_SingleGraph()
    {
        if(transpose){
            MVM_SingleGraphTranspose();
        }else{
            MVM_SingleGraphNoTranspose();
        }
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::MVM_SingleGraphTranspose()
    {
        auto & graphCreated = transposesinglegraph.graphCreated;
        auto & event_start = transposesinglegraph.event_start;
        auto & events = transposesinglegraph.events;
        auto & graph = transposesinglegraph.graph;
        auto & instance = transposesinglegraph.instance;
        if(!graphCreated){
            cudaStreamBeginCapture(streamptr[0],cudaStreamCaptureModeGlobal);
            transposesinglegraph.syncotherstreams(event_start, streamptr, stream_size);
            for(int bi=0; bi<batchsize; bi++){
                // phase 1 transpose
                for(int i=0; i<config_vec[bi].Ntg; i++){
                    if(cudap1transptrs_vec[bi].Ms[i] != 0){
                        Hgemv_Phase1_Transpose_driver(cudap1transptrs_vec[bi].Abp[i], cudap1transptrs_vec[bi].xbp[i],
                                                      cudap1transptrs_vec[bi].ybp[i],
                                                      cudap1transptrs_vec[bi].Ms[i], cudap1transptrs_vec[bi].Ks[i],
                                                      streamptr[i % stream_size]);
                    }
                }
            }
            // phase 1 synchronization
            transposesinglegraph.syncallstreams(events, streamptr, stream_size);
            for(int bi=0; bi<batchsize; bi++){
                // phase 2 transpose
                phase2_nosplit<DeviceType>(cudap1transptrs_vec[bi].y,
                                           d_phase2mapping_transpose_vec[bi],
                                           cudap3transptrs_vec[bi].x,
                                           config_vec[bi].granksum, streamptr[0]);
            }
            // phase 2 synchronization
            transposesinglegraph.syncallstreams(events+1*stream_size, streamptr, stream_size);
            for(int bi=0; bi<batchsize; bi++){
                // phase 3 transpose
                for(int i=0; i<config_vec[bi].Mtg; i++){
                    Hgemv_Phase3_Transpose_driver(cudap3transptrs_vec[bi].Abp[i], cudap3transptrs_vec[bi].xbp[i],
                                                  cudap3transptrs_vec[bi].ybp[i],
                                                  cudap3transptrs_vec[bi].Ms[i], cudap3transptrs_vec[bi].Ks[i],
                                                  streamptr[i % stream_size]);
                }
            }
            // final merge
            transposesinglegraph.syncstream0(events+2*stream_size, streamptr, stream_size);
            cudaStreamEndCapture(streamptr[0], &graph);
            cudaGraphInstantiate(&instance, graph,
                                 nullptr, nullptr, 0);
            graphCreated = true;
        }
        cudaGraphLaunch(instance, streamptr[0]);
        cudaStreamSynchronize(streamptr[0]);
    }
    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::MVM_SingleGraphNoTranspose()
    {
        auto & graphCreated = singlegraph.graphCreated;
        auto & event_start = singlegraph.event_start;
        auto & events = singlegraph.events;
        auto & graph = singlegraph.graph;
        auto & instance = singlegraph.instance;
        if(!graphCreated){
            cudaStreamBeginCapture(streamptr[0],cudaStreamCaptureModeGlobal);
            singlegraph.syncotherstreams(event_start, streamptr, stream_size);
            // phase 1
            for(int bi=0; bi<batchsize; bi++){
                for(int i=0; i<config_vec[bi].Ntg; i++){
                    if(cudap1ptrs_vec[bi].Ms[i] != 0){
                        Hgemv_Phase1_driver(cudap1ptrs_vec[bi].Abp[i], cudap1ptrs_vec[bi].xbp[i], cudap1ptrs_vec[bi].ybp[i],
                                            cudap1ptrs_vec[bi].Ms[i], cudap1ptrs_vec[bi].Ks[i], streamptr[i % stream_size]);
                    }
                }
            }
            // phase 1 synchronization
            singlegraph.syncallstreams(events, streamptr, stream_size);
            // phase 2
            for(int bi=0; bi<batchsize; bi++){
                phase2_nosplit<DeviceType>(cudap1ptrs_vec[bi].y,
                                           d_phase2mapping_vec[bi], cudap3ptrs_vec[bi].x,
                                           config_vec[bi].granksum, streamptr[bi%stream_size]);
            }
            // phase 2 synchronization
            singlegraph.syncallstreams(events+1*stream_size, streamptr, stream_size);
            // phase 3
            for(int bi=0; bi<batchsize; bi++){
                for(int i=0; i<config_vec[bi].Mtg; i++){
                    Hgemv_Phase3_driver(cudap3ptrs_vec[bi].Abp[i],cudap3ptrs_vec[bi].xbp[i], cudap3ptrs_vec[bi].ybp[i],
                                        cudap3ptrs_vec[bi].Ms[i], cudap3ptrs_vec[bi].Ks[i], streamptr[i % stream_size]);
                }
            }
            // final merge
            singlegraph.syncstream0(events+2*stream_size, streamptr, stream_size);
            cudaStreamEndCapture(streamptr[0], &graph);
            cudaGraphInstantiate(&instance, graph,
                                 nullptr, nullptr, 0);
            graphCreated = true;
        }
        cudaGraphLaunch(instance, streamptr[0]);
        cudaStreamSynchronize(streamptr[0]);
    }


    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::TryConjugateResults() {
        if(!conjugate) return;
        if(transpose){
            for(int bi=0; bi<config_vec.size(); bi++){
                ConjugateDriver<DeviceType>(cudap3transptrs_vec[bi].y,config_vec[bi].originN, streamptr[0]);
                CopyDataB2HD(cpuinstvec[bi]->p3transptrs.y, cudap3transptrs_vec[bi].y,cpuinstvec[bi]->config.originM);
            }
        }else{
            for(int bi=0; bi<config_vec.size(); bi++){
                ConjugateDriver<DeviceType>(cudap3ptrs_vec[bi].y,config_vec[bi].originM,streamptr[0]);
                CopyDataB2HD(cpuinstvec[bi]->p3ptrs.y, cudap3ptrs_vec[bi].y,cpuinstvec[bi]->config.originM);
            }
        }
    }


    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType, DeviceType>::CopyBackResults()
    {
        int offset = 0, origin = 0;
        for(int bi=0; bi<batchsize; bi++){
            // use cpu pointers to send output
            if(transpose){
                CopyDataB2HD(cpuinstvec[bi]->p1transptrs.y, cudap1transptrs_vec[bi].y, cpuinstvec[bi]->config.granksum);
                CopyDataB2HD(cpuinstvec[bi]->p3transptrs.x, cudap3transptrs_vec[bi].x, cpuinstvec[bi]->config.granksum);
                CopyDataB2HD(cpuinstvec[bi]->p3transptrs.y, cudap3transptrs_vec[bi].y, cpuinstvec[bi]->config.originM);
                origin = cpuinstvec[bi]->config.originM;
                memcpy(finalresults.data() + offset,cpuinstvec[bi]->p3transptrs.y, sizeof(HostType) * origin);
                offset += cpuinstvec[bi]->config.originM;
            }else{

                CopyDataB2HD(cpuinstvec[bi]->p1ptrs.y, cudap1ptrs_vec[bi].y, cpuinstvec[bi]->config.granksum);
                CopyDataB2HD(cpuinstvec[bi]->p3ptrs.x, cudap3ptrs_vec[bi].x, cpuinstvec[bi]->config.granksum);
                CopyDataB2HD(cpuinstvec[bi]->p3ptrs.y, cudap3ptrs_vec[bi].y, cpuinstvec[bi]->config.originM);
                origin = cpuinstvec[bi]->config.originM;
                memcpy(finalresults.data() + offset,cpuinstvec[bi]->p3ptrs.y, sizeof(HostType) * origin);
                offset += cpuinstvec[bi]->config.originM;
            }
            cpuinstvec[bi]->CopyToFinalresults();
        }
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaFP16<HostType,DeviceType>::MemoryFree(){
        for(int bi=0; bi<batchsize; bi++){
            cpuinstvec[bi]->MemoryFree();
            FreecuHostMemory(cudap1ptrs_vec[bi].Abp);
            FreecuHostMemory(cudap1ptrs_vec[bi].xbp);
            FreecuHostMemory(cudap1ptrs_vec[bi].ybp);
            FreeDeviceMemory(cudap1ptrs_vec[bi].A);
            FreeDeviceMemory(cudap1ptrs_vec[bi].x);
            FreeDeviceMemory(cudap1ptrs_vec[bi].y);

            FreecuHostMemory(cudap3ptrs_vec[bi].Abp);
            FreecuHostMemory(cudap3ptrs_vec[bi].xbp);
            FreecuHostMemory(cudap3ptrs_vec[bi].ybp);
            FreeDeviceMemory(cudap3ptrs_vec[bi].A);
            FreeDeviceMemory(cudap3ptrs_vec[bi].x);
            FreeDeviceMemory(cudap3ptrs_vec[bi].y);

            FreecuHostMemory(cudap1transptrs_vec[bi].Abp);
            FreecuHostMemory(cudap1transptrs_vec[bi].xbp);
            FreecuHostMemory(cudap1transptrs_vec[bi].ybp);
            FreeDeviceMemory(cudap1transptrs_vec[bi].y);

            FreecuHostMemory(cudap3transptrs_vec[bi].Abp);
            FreecuHostMemory(cudap3transptrs_vec[bi].xbp);
            FreecuHostMemory(cudap3transptrs_vec[bi].ybp);
            FreeDeviceMemory(cudap3transptrs_vec[bi].y);
        }
    }

    template class BatchTlrmvmcudaFP16<complex<float>, cuHalfComplex>;
    template class BatchTlrmvmcudaFP16<complex<float>, cubfComplex>;

}