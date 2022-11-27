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
#include "BatchTlrmvmcudaINT8.hpp"
#include "cudakernel.cuh"
#include <chrono>

namespace cudatlrmvm
{
    template<typename HostType, typename DeviceType>
    BatchTlrmvmcudaINT8<HostType, DeviceType>::BatchTlrmvmcudaINT8(vector<TlrmvmConfig> tlrmvmconfigvec)
    :config_vec(tlrmvmconfigvec),batchsize(tlrmvmconfigvec.size())
    {
        cout << "calling Batch TlrmvmcudaINT8" << endl;
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
        init = cuHalfComplex(0.0,0.0);
    }

    template<typename HostType, typename DeviceType>
    BatchTlrmvmcudaINT8<HostType, DeviceType>::BatchTlrmvmcudaINT8(){}
    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::StreamInit(int streamsize){
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
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::MemoryInit(){
        P3Ahalfbuffer.resize(stream_size);
        P3xhalfbuffer.resize(batchsize);
        P3maxxinfo.resize(batchsize);
        Ubases.resize(batchsize);
        Vbases.resize(batchsize);
        xinput.resize(batchsize);
        p3xint8.resize(batchsize);
        p3xreductionbuffer.resize(batchsize);
        auto start = std::chrono::steady_clock::now();
#pragma omp parallel for default(none)
        for(int bi=0; bi<batchsize; bi++){
            cpuinstvec[bi]->MemoryInit();
            I8PhasePointersCopyNonPointers(Vbases[bi], cpuinstvec[bi]->p1ptrs);
            I8PhasePointersCopyNonPointers(Ubases[bi], cpuinstvec[bi]->p3ptrs);
            //transpose need to do it manually
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
        Phase2PrepareTranspose();
        // finally allocate compute buffer
        for(int i=0; i<stream_size; i++) GetDeviceMemory(&P3Ahalfbuffer[i], cbmaxinfo.maxA);
        for(int i=0; i<batchsize; i++) GetDeviceMemory(&P3xhalfbuffer[i], Ubases[i].Xcnt);
        for(int i=0; i<batchsize; i++) {
            GetDeviceMemory(&P3maxxinfo[i], Ubases[i].Ms.size());
        }
        for(int i=0; i<batchsize; i++) GetDeviceMemory(&p3xint8[i].xbuffer, Ubases[i].Xcnt);
        for(int i=0; i<batchsize; i++) GetDeviceMemory(&p3xreductionbuffer[i], 128 * Ubases[i].Ms.size());
        for(int i=0; i<batchsize; i++) GetDeviceMemory(&p3xint8[i].p3xreductionbuffer_device, 128 * Ubases[i].Ms.size());
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType, DeviceType>::SetTransposeConjugate(bool transpose, bool conjugate){
        this->transpose = transpose;
        this->conjugate = conjugate;
        for(auto &x : cpuinstvec) x->SetTransposeConjugate(transpose, conjugate);
    }


    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::setX(HostType * xvector, size_t xlength){
        int offset = 0;
        assert(xlength == config_vec[0].originN * batchsize);
        for(int i=0; i<cpuinstvec.size(); i++){
            cpuinstvec[i]->setX(xvector + offset , config_vec[i].originN);
            offset += config_vec[i].originN;
        }
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::TryConjugateXvec() {
        for(int bi=0; bi<batchsize; bi++){
            // no transpose logic
            cpuinstvec[bi]->TryConjugateXvec();
            auto & curxptr = xinput[bi];
            size_t xoffset = 0;
            for(int oi=0; oi<config_vec[bi].Ntg; oi++){
                CopyDataB2HD(curxptr.xbuffer + xoffset, curxptr.maxx[oi],
                             cpuinstvec[bi]->p1ptrs.x + xoffset, curxptr.xelems[oi]);
                xoffset += curxptr.xelems[oi];
                curxptr.xelemsoffset[oi] = xoffset;
            }
            CopyDataB2HD(curxptr.maxx_device, curxptr.maxx.data(), curxptr.maxx.size());
        }
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::Phase1(){
        cudaDeviceSynchronize();
        for(int bi=0; bi<batchsize; bi++){
            auto & vbase = Vbases[bi];
            auto & ubase = Ubases[bi];
            auto & xvec = xinput[bi];
            // phase 1
            for(int i=0; i<config_vec[bi].Ntg; i++){
                if(vbase.Ms[i] != 0){
                    if(i == 0) {
                        Igemv_Phase1_driver(vbase.Abuffer,
                                            xvec.xbuffer,
                                            vbase.maxA[i], xvec.maxx[i],
                                            vbase.ybuffer,
                                            vbase.Ms[i], vbase.Ks[i], streamptr[i % stream_size]);
                    }else{
                        Igemv_Phase1_driver(vbase.Abuffer + vbase.Aelemsoffset[i-1],
                                            xvec.xbuffer + xvec.xelemsoffset[i-1],
                                            vbase.maxA[i], xvec.maxx[i],
                                            vbase.ybuffer + vbase.yelemsoffset[i-1],
                                            vbase.Ms[i], vbase.Ks[i], streamptr[i % stream_size]);
                    }
                }
            }
        }
        cudaDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::Phase1Transpose() {
        cudaDeviceSynchronize();
        for(int bi=0; bi<batchsize; bi++){
            auto & vbase = Vbases[bi];
            auto & ubase = Ubases[bi];
            auto & xvec = xinput[bi];
            // phase 1 transpose
            size_t offset = 0;
            for(int i=0; i<config_vec[bi].Ntg; i++){
                if(vbase.Ms[i] != 0){
                    if(i == 0) {
                        Igemv_Phase1_Transpose_driver(ubase.Abuffer,
                                                      xvec.xbuffer,
                                                      ubase.maxA[i], xvec.maxx[i],
                                                      vbase.ybuffer,
                                                      ubase.Ks[i], ubase.Ms[i], streamptr[i % stream_size]);
                        offset += ubase.Ks[i];
                    }else{
                        Igemv_Phase1_Transpose_driver(ubase.Abuffer + ubase.Aelemsoffset[i-1],
                                                      xvec.xbuffer + xvec.xelemsoffset[i-1],
                                                      ubase.maxA[i], xvec.maxx[i],
                                                      vbase.ybuffer + offset,
                                                      ubase.Ks[i], ubase.Ms[i], streamptr[i % stream_size]);
                        offset += ubase.Ks[i];
                    }
                }
            }
        }
        cudaDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::Phase1Prepare() {
        // load data from cpu to Vbases
        for(int bi=0; bi<batchsize; bi++){
            auto & curbase = Vbases[bi];
            auto & curxptr = xinput[bi];
            size_t curbatch = curbase.Ms.size();
            // update cbmaxinfo
            cbmaxinfo.maxbatchsize = max(cbmaxinfo.maxbatchsize, curbatch);
            cbmaxinfo.maxA = max(cbmaxinfo.maxA, curbase.Acnt);
            cbmaxinfo.maxx = max(cbmaxinfo.maxx, curbase.Xcnt);
            cbmaxinfo.maxy = max(cbmaxinfo.maxy, curbase.Ycnt);
            // allocate Base pointers, A
            GetDeviceMemory(&curbase.Abuffer, curbase.Acnt);

            curbase.maxA.resize(curbatch);
            GetDeviceMemory(&curbase.maxA_device, curbatch);

            curbase.Aelems.resize(curbatch);
            curbase.Aelemsoffset.resize(curbatch);
            GetDeviceMemory(&curbase.Aelemsoffset_device, curbatch);

            // allocate Base pointers, y
            GetDeviceMemory(&curbase.ybuffer, curbase.Ycnt);
            curbase.yelems.resize(curbatch);
            curbase.yelemsoffset.resize(curbatch);

            // allocate x pointers, x
            curbase.xelems.resize(curbatch);
            curbase.xelemsoffset.resize(curbatch);

            // allocate x input pointers
            GetDeviceMemory(&curxptr.xbuffer, curbase.Xcnt);
            curxptr.maxx.resize(curbatch);
            curxptr.xelems.resize(curbatch);
            GetDeviceMemory(&curxptr.maxx_device, curbatch);
            curxptr.xelemsoffset.resize(curbatch);


            size_t Aoffset = 0; size_t yoffset = 0; size_t xoffset = 0;
            for(int oi = 0; oi < curbatch; oi++) {
                // A related
                curbase.Aelems[oi] = curbase.Ms[oi] * curbase.Ks[oi];
                CopyDataB2HD(curbase.Abuffer + Aoffset, curbase.maxA[oi],
                             cpuinstvec[bi]->p1ptrs.A + Aoffset, curbase.Aelems[oi]);
                Aoffset += curbase.Aelems[oi];
                curbase.Aelemsoffset[oi] = Aoffset;
                // yelems
                curbase.yelems[oi] = curbase.Ms[oi];
                yoffset += curbase.yelems[oi];
                curbase.yelemsoffset[oi] = yoffset;
                // x related
                curxptr.xelems[oi] = curbase.Ks[oi] * curbase.Ns[oi];
                curbase.xelems[oi] = curbase.Ks[oi] * curbase.Ns[oi];
                CopyDataB2HD(curxptr.xbuffer + xoffset, curxptr.maxx[oi],
                             cpuinstvec[bi]->p1ptrs.x + xoffset, curxptr.xelems[oi]);
                xoffset += curxptr.xelems[oi];
                curxptr.xelemsoffset[oi] = xoffset;
                curbase.xelemsoffset[oi] = xoffset;
            }
            // load max info to device
            CopyDataB2HD(curbase.Aelemsoffset_device, curbase.Aelemsoffset.data(),
                         curbase.Aelemsoffset.size());
            CopyDataB2HD(curbase.maxA_device, curbase.maxA.data(), curbase.maxA.size());
            CopyDataB2HD(curxptr.maxx_device, curxptr.maxx.data(), curxptr.maxx.size());
        }
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::Phase2() {
        cudaDeviceSynchronize();
        for(int bi=0; bi<batchsize; bi++){
            auto & vbase = Vbases[bi];
            auto & ubase = Ubases[bi];
            auto & curxptr = p3xint8[bi];
            size_t launchlen = 0;
            for(int k=0; k<curxptr.xelems.size();k++) {
                launchlen += ((curxptr.xelems[k]+127)/128)*128;
            }
            phase2_Int8_driver(vbase.ybuffer,
                               d_phase2mapping2_vec[bi],
                               P3xhalfbuffer[bi],
                               launchlen, curxptr.xelemsoffset.back(),
                               curxptr.xelems_device,
                               curxptr.xelemsoffset_device,
                               curxptr.p3xreductionbuffer_device,
                               curxptr.maxx_device,
                               curxptr.xbuffer,
                               ubase.Ms.size(),
                               streamptr[bi%stream_size]);
        }
        cudaDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::Phase2Transpose() {
        cudaDeviceSynchronize();
        for(int bi=0; bi<batchsize; bi++){
            // phase 2 transpose
            auto & vbase = Vbases[bi];
            auto & ubase = Ubases[bi];
            phase2_nosplit<cuHalfComplex>(vbase.ybuffer,
                                          d_phase2mapping_transpose_vec[bi], P3xhalfbuffer[bi],
                                          config_vec[bi].granksum, streamptr[bi%stream_size]);
        }
        cudaDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::Phase2Prepare(){
        d_phase2mapping_vec = new size_t*[batchsize];
        for(int bi=0; bi<batchsize; bi++){
            GetDeviceMemory(&d_phase2mapping_vec[bi], cpuinstvec[bi]->h_phase2mapping.size());
            CopyDataB2HD(d_phase2mapping_vec[bi], cpuinstvec[bi]->h_phase2mapping.data(),
                         cpuinstvec[bi]->h_phase2mapping.size());
        }
        d_phase2mapping2_vec = new size_t*[batchsize];
        for(int bi=0; bi<batchsize; bi++){
            GetDeviceMemory(&d_phase2mapping2_vec[bi], cpuinstvec[bi]->h_phase2mapping2.size());
            CopyDataB2HD(d_phase2mapping2_vec[bi], cpuinstvec[bi]->h_phase2mapping2.data(),
                         cpuinstvec[bi]->h_phase2mapping2.size());
        }
    }
    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::Phase2PrepareTranspose(){
        d_phase2mapping_transpose_vec = new size_t*[batchsize];
        for(int bi=0; bi<batchsize; bi++){
            GetDeviceMemory(&d_phase2mapping_transpose_vec[bi], cpuinstvec[bi]->h_phase2mappingTranspose.size());
            CopyDataB2HD(d_phase2mapping_transpose_vec[bi], cpuinstvec[bi]->h_phase2mappingTranspose.data(),
                         cpuinstvec[bi]->h_phase2mappingTranspose.size());
        }
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::Phase3() {
        cudaDeviceSynchronize();
        for(int bi=0; bi<batchsize; bi++){
            auto & vbase = Vbases[bi];
            auto & ubase = Ubases[bi];
            auto & xvec = p3xint8[bi];
            for(int i=0; i<config_vec[bi].Mtg; i++){
                if(ubase.Ms[i] != 0) {
                    if(i == 0){
                        New_Igemv_Phase3_driver(ubase.Abuffer,
                                            xvec.xbuffer, ubase.maxA_device, xvec.maxx_device,
                                            ubase.ybuffer,ubase.Ms[i],ubase.Ks[i],
                                            i,streamptr[i % stream_size]);
                    }else{
                        New_Igemv_Phase3_driver(ubase.Abuffer + ubase.Aelemsoffset[i-1],
                                            xvec.xbuffer + xvec.xelemsoffset[i-1],
                                            ubase.maxA_device, xvec.maxx_device,
                                            ubase.ybuffer + ubase.yelemsoffset[i-1],ubase.Ms[i],
                                            ubase.Ks[i],i,streamptr[i % stream_size]);
                    }
                    cudaDeviceSynchronize();
                }
            }
        }
        cudaDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::Phase3Transpose() {
        cudaDeviceSynchronize();
        for(int bi=0; bi<batchsize; bi++){
            // phase 3 transpose
            auto & vbase = Vbases[bi];
            auto & ubase = Ubases[bi];
            auto & xvec = xinput[bi];
            // phase 3
            size_t xoffset = 0; size_t yoffset = 0;
            for(int i=0; i<config_vec[bi].Mtg; i++){
                if(vbase.Ms[i] != 0) {
                    if(i == 0){
                        // convert I8 data buffer to float 16 buffer
                        Phase3UpcastingDriver(P3Ahalfbuffer[i%stream_size], vbase.Abuffer,
                                              vbase.maxA[i], 0,
                                              vbase.Aelems[i], i,streamptr[i%stream_size]);
                        Hgemv_Phase3_Transpose_driver(P3Ahalfbuffer[i%stream_size],
                                                      P3xhalfbuffer[bi],
                                                      ubase.ybuffer,
                                                      vbase.Ks[i], vbase.Ms[i], streamptr[i % stream_size]);
                    }else{
                        // convert I8 data buffer to float 16 buffer
                        Phase3UpcastingDriver(P3Ahalfbuffer[i%stream_size], vbase.Abuffer,
                                              vbase.maxA[i], vbase.Aelemsoffset[i-1],
                                              vbase.Aelems[i], i,streamptr[i%stream_size]);
                        Hgemv_Phase3_Transpose_driver(P3Ahalfbuffer[i%stream_size] + vbase.Aelemsoffset[i-1],
                                                      P3xhalfbuffer[bi] + xoffset,
                                                      ubase.ybuffer + yoffset,
                                                      vbase.Ks[i], vbase.Ms[i], streamptr[i % stream_size]);
                    }
                    yoffset += vbase.Ks[i];
                    xoffset += vbase.Ms[i];
                }
            }
        }
        cudaDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::Phase3Prepare() {
        // load data from cpu to Vbases
        for(int bi=0; bi<batchsize; bi++){
            auto & curbase = Ubases[bi];
            auto & xinput = p3xint8[bi];
            size_t curbatch = curbase.Ms.size();
            // update cbmaxinfo
            cbmaxinfo.maxbatchsize = max(cbmaxinfo.maxbatchsize, curbatch);
            cbmaxinfo.maxA = max(cbmaxinfo.maxA, curbase.Acnt);
            cbmaxinfo.maxx = max(cbmaxinfo.maxx, curbase.Xcnt);
            cbmaxinfo.maxy = max(cbmaxinfo.maxy, curbase.Ycnt);
            // allocate Base pointers, A
            GetDeviceMemory(&curbase.Abuffer, curbase.Acnt);
            curbase.maxA.resize(curbatch);
            GetDeviceMemory(&curbase.maxA_device, curbatch);
            curbase.Aelems.resize(curbatch);
            curbase.Aelemsoffset.resize(curbatch);
            GetDeviceMemory(&curbase.Aelemsoffset_device, curbatch);
            // allocate Base pointers, y
            GetDeviceMemory(&curbase.ybuffer, curbase.Ycnt);
            curbase.yelems.resize(curbatch);
            curbase.yelemsoffset.resize(curbatch);
            // allocate Base pointers, x
            curbase.xelems.resize(curbatch);
            curbase.xelemsoffset.resize(curbatch);

            GetDeviceMemory(&xinput.xbuffer, curbase.Xcnt);
            GetDeviceMemory(&xinput.maxx_device, curbatch);
            GetDeviceMemory(&xinput.xelems_device, curbatch);
            GetDeviceMemory(&xinput.xelemsoffset_device, curbatch);
            xinput.maxx.resize(curbatch);
            xinput.xelems.resize(curbatch);
            xinput.xelemsoffset.resize(curbatch);
            size_t Aoffset = 0; size_t yoffset = 0; size_t xoffset = 0;

            for(int oi = 0; oi < curbatch; oi++) {
                // A related
                curbase.Aelems[oi] = curbase.Ms[oi] * curbase.Ks[oi];
                CopyDataB2HD(curbase.Abuffer + Aoffset, curbase.maxA[oi],
                             cpuinstvec[bi]->p3ptrs.A + Aoffset, curbase.Aelems[oi]);
                Aoffset += curbase.Aelems[oi];
                curbase.Aelemsoffset[oi] = Aoffset;
                // y related
                curbase.yelems[oi] = curbase.Ms[oi];
                yoffset += curbase.yelems[oi];
                curbase.yelemsoffset[oi] = yoffset;
                // x related
                curbase.xelems[oi] = curbase.Ks[oi];
                xinput.xelems[oi] = curbase.Ks[oi];
                xoffset += curbase.xelems[oi];
                curbase.xelemsoffset[oi] = xoffset;
                xinput.xelemsoffset[oi] = xoffset;
            }
            // load max info to device
            CopyDataB2HD(curbase.Aelemsoffset_device, curbase.Aelemsoffset.data(),
                         curbase.Aelemsoffset.size());
            CopyDataB2HD(curbase.maxA_device, curbase.maxA.data(), curbase.maxA.size());
            CopyDataB2HD(xinput.xelems_device, xinput.xelems.data(), xinput.xelems.size());
            CopyDataB2HD(xinput.xelemsoffset_device, xinput.xelemsoffset.data(), xinput.xelemsoffset.size());
        }
    }


    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::MVM_MultiGraph(){
        if(transpose){
            MVM_MultiGraphTranspose();
        }else{
            MVM_MultiGraphNoTranspose();
        }
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::MVM_MultiGraphTranspose(){
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
                auto & vbase = Vbases[bi];
                auto & ubase = Ubases[bi];
                auto & xvec = xinput[bi];
                // phase 1
                size_t offset = 0;
                for(int i=0; i<config_vec[bi].Ntg; i++){
                    if(vbase.Ms[i] != 0){
                        if(i == 0) {
                            Igemv_Phase1_Transpose_driver(ubase.Abuffer,
                                                xvec.xbuffer,
                                                ubase.maxA[i], xvec.maxx[i],
                                                vbase.ybuffer,
                                                ubase.Ks[i], ubase.Ms[i], streamptr[i % stream_size]);
                            offset += ubase.Ks[i];
                        }else{
                            Igemv_Phase1_Transpose_driver(ubase.Abuffer + ubase.Aelemsoffset[i-1],
                                                xvec.xbuffer + xvec.xelemsoffset[i-1],
                                                ubase.maxA[i], xvec.maxx[i],
                                                vbase.ybuffer + offset,
                                                ubase.Ks[i], ubase.Ms[i], streamptr[i % stream_size]);
                            offset += ubase.Ks[i];
                        }
                    }
                }
                for(int streami=1; streami < stream_size; streami++){
                    cudaEventRecord(events[bi][streami], streamptr[streami]);
                }
                for(int streami=1; streami < stream_size; streami++){
                    cudaStreamWaitEvent(streamptr[0], events[bi][streami]);
                }
                // phase 2
                phase2_nosplit<cuHalfComplex>(vbase.ybuffer,
                                              d_phase2mapping_transpose_vec[bi], P3xhalfbuffer[bi],
                                              config_vec[bi].granksum, streamptr[0]);
                cudaEventRecord(events[bi][0], streamptr[0]);
                for(int streami=1; streami < stream_size; streami++){
                    cudaStreamWaitEvent(streamptr[streami], events[bi][0]);
                }
                // phase 3
                size_t xoffset = 0; size_t yoffset = 0;
                for(int i=0; i<config_vec[bi].Mtg; i++){
                    if(vbase.Ms[i] != 0) {
                        if(i == 0){
                            // convert I8 data buffer to float 16 buffer
                            Phase3UpcastingDriver(P3Ahalfbuffer[i%stream_size], vbase.Abuffer,
                                                  vbase.maxA[i], 0,
                                                  vbase.Aelems[i], i,streamptr[i%stream_size]);
                            Hgemv_Phase3_Transpose_driver(P3Ahalfbuffer[i%stream_size],
                                                P3xhalfbuffer[bi],
                                                ubase.ybuffer,
                                                vbase.Ks[i], vbase.Ms[i], streamptr[i % stream_size]);
                        }else{
                            // convert I8 data buffer to float 16 buffer
                            Phase3UpcastingDriver(P3Ahalfbuffer[i%stream_size], vbase.Abuffer,
                                                  vbase.maxA[i], vbase.Aelemsoffset[i-1],
                                                  vbase.Aelems[i], i,streamptr[i%stream_size]);
                            Hgemv_Phase3_Transpose_driver(P3Ahalfbuffer[i%stream_size] + vbase.Aelemsoffset[i-1],
                                                P3xhalfbuffer[bi] + xoffset,
                                                ubase.ybuffer + yoffset,
                                                vbase.Ks[i], vbase.Ms[i], streamptr[i % stream_size]);
                        }
                        yoffset += vbase.Ks[i];
                        xoffset += vbase.Ms[i];
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
                cudaGraphInstantiate(&instance[bi], graph[bi], nullptr, nullptr, 0);
                graphCreated[bi] = true;
            }
            cudaGraphLaunch(instance[bi], streamptr[0]);
            cudaStreamSynchronize(streamptr[0]);
        }
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::MVM_MultiGraphNoTranspose()
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
                auto & vbase = Vbases[bi];
                auto & ubase = Ubases[bi];
                auto & xvec = xinput[bi];
                // phase 1
                for(int i=0; i<config_vec[bi].Ntg; i++){
                    if(vbase.Ms[i] != 0){
                        if(i == 0) {
                            Igemv_Phase1_driver(vbase.Abuffer,
                                                xvec.xbuffer,
                                                vbase.maxA[i], xvec.maxx[i],
                                                vbase.ybuffer,
                                                vbase.Ms[i], vbase.Ks[i], streamptr[i % stream_size]);
                        }else{
                            Igemv_Phase1_driver(vbase.Abuffer + vbase.Aelemsoffset[i-1],
                                                xvec.xbuffer + xvec.xelemsoffset[i-1],
                                                vbase.maxA[i], xvec.maxx[i],
                                                vbase.ybuffer + vbase.yelemsoffset[i-1],
                                                vbase.Ms[i], vbase.Ks[i], streamptr[i % stream_size]);
                        }
                    }
                }
                for(int streami=1; streami < stream_size; streami++){
                    cudaEventRecord(events[bi][streami], streamptr[streami]);
                }
                for(int streami=1; streami < stream_size; streami++){
                    cudaStreamWaitEvent(streamptr[0], events[bi][streami]);
                }
                // phase 2
                phase2_nosplit<cuHalfComplex>(vbase.ybuffer,
                                           d_phase2mapping_vec[bi], P3xhalfbuffer[bi],
                                           config_vec[bi].granksum, streamptr[0]);
                cudaEventRecord(events[bi][0], streamptr[0]);
                for(int streami=1; streami < stream_size; streami++){
                    cudaStreamWaitEvent(streamptr[streami], events[bi][0]);
                }
                // phase 3
                for(int i=0; i<config_vec[bi].Mtg; i++){
                    if(ubase.Ms[i] != 0) {
                        if(i == 0){
                            Phase3UpcastingDriver(P3Ahalfbuffer[i%stream_size],ubase.Abuffer,
                                                  ubase.maxA[i],0,
                                                  ubase.Aelems[i],
                                                  i, streamptr[i%stream_size]);
                            Hgemv_Phase3_driver(P3Ahalfbuffer[i%stream_size],
                                                P3xhalfbuffer[bi],
                                                ubase.ybuffer,
                                                ubase.Ms[i], ubase.Ks[i], streamptr[i % stream_size]);
                        }else{
                            Phase3UpcastingDriver(P3Ahalfbuffer[i%stream_size],ubase.Abuffer,
                                                  ubase.maxA[i],ubase.Aelemsoffset[i-1],
                                                  ubase.Aelems[i],
                                                  i, streamptr[i%stream_size]);
                            Hgemv_Phase3_driver(P3Ahalfbuffer[i%stream_size] + ubase.Aelemsoffset[i-1],
                                                P3xhalfbuffer[bi] + ubase.xelemsoffset[i-1],
                                                ubase.ybuffer + ubase.yelemsoffset[i-1],
                                                ubase.Ms[i], ubase.Ks[i], streamptr[i % stream_size]);
                        }
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
                cudaGraphInstantiate(&instance[bi],
                                     graph[bi], nullptr, nullptr, 0);
                graphCreated[bi] = true;
            }
            cudaGraphLaunch(instance[bi], streamptr[0]);
            cudaStreamSynchronize(streamptr[0]);
        }
    }



    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::MVM_SingleGraph()
    {
        if(transpose){
            MVM_SingleGraphTranspose();
        }else{
            MVM_SingleGraphNoTranspose();
        }
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::MVM_SingleGraphTranspose()
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
                auto & vbase = Vbases[bi];
                auto & ubase = Ubases[bi];
                auto & xvec = xinput[bi];
                // phase 1 transpose
                size_t offset = 0;
                for(int i=0; i<config_vec[bi].Ntg; i++){
                    if(vbase.Ms[i] != 0){
                        if(i == 0) {
                            Igemv_Phase1_Transpose_driver(ubase.Abuffer,
                                                          xvec.xbuffer,
                                                          ubase.maxA[i], xvec.maxx[i],
                                                          vbase.ybuffer,
                                                          ubase.Ks[i], ubase.Ms[i], streamptr[i % stream_size]);
                            offset += ubase.Ks[i];
                        }else{
                            Igemv_Phase1_Transpose_driver(ubase.Abuffer + ubase.Aelemsoffset[i-1],
                                                          xvec.xbuffer + xvec.xelemsoffset[i-1],
                                                          ubase.maxA[i], xvec.maxx[i],
                                                          vbase.ybuffer + offset,
                                                          ubase.Ks[i], ubase.Ms[i], streamptr[i % stream_size]);
                            offset += ubase.Ks[i];
                        }
                    }
                }
            }
            // phase 1 synchronization
            transposesinglegraph.syncallstreams(events, streamptr, stream_size);
            for(int bi=0; bi<batchsize; bi++){
                // phase 2 transpose
                auto & vbase = Vbases[bi];
                auto & ubase = Ubases[bi];
                phase2_nosplit<cuHalfComplex>(vbase.ybuffer,
                                              d_phase2mapping_transpose_vec[bi], P3xhalfbuffer[bi],
                                              config_vec[bi].granksum, streamptr[0]);
            }
            // phase 2 synchronization
            transposesinglegraph.syncallstreams(events+1*stream_size, streamptr, stream_size);
            for(int bi=0; bi<batchsize; bi++){
                // phase 3 transpose
                auto & vbase = Vbases[bi];
                auto & ubase = Ubases[bi];
                auto & xvec = xinput[bi];
                // phase 3
                size_t xoffset = 0; size_t yoffset = 0;
                for(int i=0; i<config_vec[bi].Mtg; i++){
                    if(vbase.Ms[i] != 0) {
                        if(i == 0){
                            // convert I8 data buffer to float 16 buffer
                            Phase3UpcastingDriver(P3Ahalfbuffer[i%stream_size], vbase.Abuffer,
                                                  vbase.maxA[i], 0,
                                                  vbase.Aelems[i], i,streamptr[i%stream_size]);
                            Hgemv_Phase3_Transpose_driver(P3Ahalfbuffer[i%stream_size],
                                                          P3xhalfbuffer[bi],
                                                          ubase.ybuffer,
                                                          vbase.Ks[i], vbase.Ms[i], streamptr[i % stream_size]);
                        }else{
                            // convert I8 data buffer to float 16 buffer
                            Phase3UpcastingDriver(P3Ahalfbuffer[i%stream_size], vbase.Abuffer,
                                                  vbase.maxA[i], vbase.Aelemsoffset[i-1],
                                                  vbase.Aelems[i], i,streamptr[i%stream_size]);
                            Hgemv_Phase3_Transpose_driver(P3Ahalfbuffer[i%stream_size] + vbase.Aelemsoffset[i-1],
                                                          P3xhalfbuffer[bi] + xoffset,
                                                          ubase.ybuffer + yoffset,
                                                          vbase.Ks[i], vbase.Ms[i], streamptr[i % stream_size]);
                        }
                        yoffset += vbase.Ks[i];
                        xoffset += vbase.Ms[i];
                    }
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
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::MVM_SingleGraphNoTranspose()
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
                auto & vbase = Vbases[bi];
                auto & ubase = Ubases[bi];
                auto & xvec = xinput[bi];
                // phase 1
                for(int i=0; i<config_vec[bi].Ntg; i++){
                    if(vbase.Ms[i] != 0){
                        if(i == 0) {
                            Igemv_Phase1_driver(vbase.Abuffer,
                                                xvec.xbuffer,
                                                vbase.maxA[i], xvec.maxx[i],
                                                vbase.ybuffer,
                                                vbase.Ms[i], vbase.Ks[i], streamptr[i%stream_size]);
                        }else{
                            Igemv_Phase1_driver(vbase.Abuffer + vbase.Aelemsoffset[i-1],
                                                xvec.xbuffer + xvec.xelemsoffset[i-1],
                                                vbase.maxA[i], xvec.maxx[i],
                                                vbase.ybuffer + vbase.yelemsoffset[i-1],
                                                vbase.Ms[i], vbase.Ks[i], streamptr[i%stream_size]);
                        }
                    }
                }
            }
            // phase 1 synchronization
            singlegraph.syncallstreams(events, streamptr, stream_size);
            // phase 2
            for(int bi=0; bi<batchsize; bi++){
                auto & vbase = Vbases[bi];
                auto & ubase = Ubases[bi];
                auto & curxptr = p3xint8[bi];
                size_t launchlen = 0;
                for(int k=0; k<curxptr.xelems.size();k++) {
                    launchlen += ((curxptr.xelems[k]+127)/128)*128;
                }
                phase2_Int8_driver(vbase.ybuffer,
                                   d_phase2mapping2_vec[bi],
                                   P3xhalfbuffer[bi],
                                   launchlen, curxptr.xelemsoffset.back(),
                                   curxptr.xelems_device,
                                   curxptr.xelemsoffset_device,
                                   curxptr.p3xreductionbuffer_device,
                                   curxptr.maxx_device,
                                   curxptr.xbuffer,
                                   ubase.Ms.size(),
                                   streamptr[bi%stream_size]);
            }
            // phase 2 synchronization
            singlegraph.syncallstreams(events+1*stream_size, streamptr, stream_size);
            // phase 3
            for(int bi=0; bi<batchsize; bi++){
                auto & vbase = Vbases[bi];
                auto & ubase = Ubases[bi];
                auto & xvec = p3xint8[bi];
                for(int i=0; i<config_vec[bi].Mtg; i++){
                    if(ubase.Ms[i] != 0) {
                        if(i == 0){
                            New_Igemv_Phase3_driver(ubase.Abuffer,
                                                    xvec.xbuffer, ubase.maxA_device, xvec.maxx_device,
                                                    ubase.ybuffer,ubase.Ms[i],ubase.Ks[i],
                                                    i,streamptr[i%stream_size]);
                        }else{
                            New_Igemv_Phase3_driver(ubase.Abuffer + ubase.Aelemsoffset[i-1],
                                                    xvec.xbuffer + xvec.xelemsoffset[i-1],
                                                    ubase.maxA_device, xvec.maxx_device,
                                                    ubase.ybuffer + ubase.yelemsoffset[i-1],ubase.Ms[i],
                                                    ubase.Ks[i],i,streamptr[i%stream_size]);
                        }
                    }
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
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::TryConjugateResults() {
        if(!conjugate) return;
        if(transpose){
            for(int bi=0; bi<config_vec.size(); bi++){
                ConjugateDriver<cuHalfComplex>(Ubases[bi].ybuffer,config_vec[bi].originN, streamptr[0]);
                CopyDataB2HD(cpuinstvec[bi]->p3transptrs.y, Ubases[bi].ybuffer,cpuinstvec[bi]->config.originM);
            }
        }else{
            for(int bi=0; bi<config_vec.size(); bi++){
                ConjugateDriver<cuHalfComplex>(Ubases[bi].ybuffer,config_vec[bi].originN, streamptr[0]);
                CopyDataB2HD(cpuinstvec[bi]->p3ptrs.y, Ubases[bi].ybuffer,cpuinstvec[bi]->config.originM);
            }
        }
    }


    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType, DeviceType>::CopyBackResults()
    {
        int offset = 0, origin = 0;
        for(int bi=0; bi<batchsize; bi++){
            auto & ubase = Ubases[bi];
            auto & vbase = Vbases[bi];
            auto & cpuinst = cpuinstvec[bi];
            // use cpu pointers to send output
            if(transpose){
                CopyDataB2HD(cpuinst->p1transptrs.y, vbase.ybuffer, cpuinst->config.granksum);
                CopyDataB2HD(cpuinst->p3transptrs.x, P3xhalfbuffer[bi], cpuinst->config.granksum);
                CopyDataB2HD(cpuinst->p3transptrs.y, ubase.ybuffer, cpuinst->config.originM);
                origin = cpuinst->config.originM;
                memcpy(finalresults.data() + offset,cpuinst->p3transptrs.y, sizeof(HostType) * origin);
                offset += cpuinst->config.originM;
            }else{
                CopyDataB2HD(cpuinst->p1ptrs.y, vbase.ybuffer, cpuinst->config.granksum);
                CopyDataB2HD(cpuinst->p3ptrs.x, P3xhalfbuffer[bi], cpuinst->config.granksum);
                CopyDataB2HD(cpuinst->p3ptrs.y, ubase.ybuffer, cpuinst->config.originM);
                origin = cpuinst->config.originM;
                memcpy(finalresults.data() + offset,cpuinst->p3ptrs.y, sizeof(HostType) * origin);
                offset += cpuinst->config.originM;
            }
            cpuinst->CopyToFinalresults();
        }
    }

    template<typename HostType, typename DeviceType>
    void BatchTlrmvmcudaINT8<HostType,DeviceType>::MemoryFree(){

    }


    template class BatchTlrmvmcudaINT8<complex<float>, cuInt8Complex>;

}