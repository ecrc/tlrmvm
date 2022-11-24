//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas.h>

#include "../../common/Common.hpp"
#include "../../common/AppUtil.hpp"
#include "../cpu/TlrmvmCPU.hpp"
#include "Tlrmvmhip.hpp"
#include "TlrmvmhipConstRank.hpp"
#include "hipkernel.cuh"

namespace hiptlrmvm
{
    template<typename HostType, typename DeviceType>
    TlrmvmhipConstRank<HostType, DeviceType>::TlrmvmhipConstRank() {}

    template<typename HostType, typename DeviceType>
    TlrmvmhipConstRank<HostType, DeviceType>::TlrmvmhipConstRank(TlrmvmConfig tlrmvmconfig)
            :config(tlrmvmconfig)
    {
        transpose = false;
        conjugate = false;
        init_alpha_beta(alpha, beta);
        tlrmvmcpu = std::make_shared<TlrmvmCPU<HostType>>(tlrmvmconfig);
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType, DeviceType>::UpdateConfig(TlrmvmConfig &tlrmvmconfig)
    {
//        transpose = false;
//        conjugate = false;
//        init_alpha_beta(alpha, beta);
//        tlrmvmcpu->UpdateConfig(tlrmvmconfig);
        cout << "UpdateConfig not implemented." << endl;
        exit(0);
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::setX(HostType * xvector, size_t xlength){
        tlrmvmcpu->setX(xvector, xlength);
        tlrmvmcpu->TryConjugateXvec();
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::TryConjugateXvec() {
        // no transpose logic
        tlrmvmcpu->TryConjugateXvec();
        CopyDataB2HD((HostType*)this->cudap1ptrs.x, tlrmvmcpu->p1ptrs.x, tlrmvmcpu->xmat.Shape()[0]);
    }
    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::TryConjugateResults() {
        if(!conjugate) return;
        if(transpose){
            ConjugateDriver<DeviceType>(cudap3transptrs.y, config.originN, stream);
            CopyDataB2HD(tlrmvmcpu->p3transptrs.y, (HostType*)cudap3transptrs.y, tlrmvmcpu->config.originM);
        }else{
            ConjugateDriver<DeviceType>(cudap3ptrs.y, config.originM, stream);
            CopyDataB2HD(tlrmvmcpu->p3ptrs.y, (HostType*)cudap3ptrs.y, tlrmvmcpu->config.originM);
        }
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::StreamInit(int streamsize){
        hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
        hipblasCreate(&cublashandle);
        hipblasSetStream(cublashandle, stream);
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::StreamDestroy(){
        hipblasDestroy(cublashandle);
        hipStreamDestroy(stream);
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::MemoryInit(){
        tlrmvmcpu->MemoryInit();
        PhasePointersCopyNonPointers<HostType, DeviceType>(cudap1ptrs, tlrmvmcpu->p1ptrs);
        PhasePointersCopyNonPointers<HostType, DeviceType>(cudap3ptrs, tlrmvmcpu->p3ptrs);
        PhasePointersCopyNonPointers<HostType, DeviceType>(cudap1transptrs, tlrmvmcpu->p1transptrs);
        PhasePointersCopyNonPointers<HostType, DeviceType>(cudap3transptrs, tlrmvmcpu->p3transptrs);
        Phase1GetMembuffer();
        AllocatePhase1Buffer();
        Phase1CopyData();
        Phase2Prepare();
        Phase3GetMembuffer();
        AllocatePhase3Buffer();
        Phase3CopyData();
        // transpose
        Phase1GetMembufferTranspose();
        AllocatePhase1BufferTranspose();
        Phase1CopyDataTranspose();
        Phase2PrepareTranspose();
        Phase3GetMembufferTranspose();
        AllocatePhase3BufferTranspose();
        Phase3CopyDataTranspose();

        // init batch pointers
        GetDeviceMemory(&d_p1Aptrs, cudap1ptrs.Ms.size());
        GetDeviceMemory(&d_p1xptrs, cudap1ptrs.Ms.size());
        GetDeviceMemory(&d_p1yptrs, cudap1ptrs.Ms.size());
        GetDeviceMemory(&d_p3Aptrs, cudap3ptrs.Ms.size());
        GetDeviceMemory(&d_p3xptrs, cudap3ptrs.Ms.size());
        GetDeviceMemory(&d_p3yptrs, cudap3ptrs.Ms.size());

        CopyDataB2HD((HostType**)d_p1Aptrs, (HostType**)cudap1ptrs.Abp, cudap1ptrs.Ms.size());
        CopyDataB2HD((HostType**)d_p1xptrs, (HostType**)cudap1ptrs.xbp, cudap1ptrs.Ms.size());
        CopyDataB2HD((HostType**)d_p1yptrs, (HostType**)cudap1ptrs.ybp, cudap1ptrs.Ms.size());
        CopyDataB2HD((HostType**)d_p3Aptrs, (HostType**)cudap3ptrs.Abp, cudap3ptrs.Ms.size());
        CopyDataB2HD((HostType**)d_p3xptrs, (HostType**)cudap3ptrs.xbp, cudap3ptrs.Ms.size());
        CopyDataB2HD((HostType**)d_p3yptrs, (HostType**)cudap3ptrs.ybp, cudap3ptrs.Ms.size());

        GetDeviceMemory(&d_p1transAptrs, cudap1transptrs.Ms.size());
        GetDeviceMemory(&d_p1transxptrs, cudap1transptrs.Ms.size());
        GetDeviceMemory(&d_p1transyptrs, cudap1transptrs.Ms.size());
        GetDeviceMemory(&d_p3transAptrs, cudap3transptrs.Ms.size());
        GetDeviceMemory(&d_p3transxptrs, cudap3transptrs.Ms.size());
        GetDeviceMemory(&d_p3transyptrs, cudap3transptrs.Ms.size());

        CopyDataB2HD((HostType**)d_p1transAptrs, (HostType**)cudap1transptrs.Abp, cudap1transptrs.Ms.size());
        CopyDataB2HD((HostType**)d_p1transxptrs, (HostType**)cudap1transptrs.xbp, cudap1transptrs.Ms.size());
        CopyDataB2HD((HostType**)d_p1transyptrs, (HostType**)cudap1transptrs.ybp, cudap1transptrs.Ms.size());
        CopyDataB2HD((HostType**)d_p3transAptrs, (HostType**)cudap3transptrs.Abp, cudap3transptrs.Ms.size());
        CopyDataB2HD((HostType**)d_p3transxptrs, (HostType**)cudap3transptrs.xbp, cudap3transptrs.Ms.size());
        CopyDataB2HD((HostType**)d_p3transyptrs, (HostType**)cudap3transptrs.ybp, cudap3transptrs.Ms.size());
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::MemoryFree(){
        tlrmvmcpu->MemoryFree();
        FreehipHostMemory(cudap1ptrs.Abp);
        FreehipHostMemory(cudap1ptrs.xbp);
        FreehipHostMemory(cudap1ptrs.ybp);
        FreeDeviceMemory(cudap1ptrs.A);
        FreeDeviceMemory(cudap1ptrs.x);
        FreeDeviceMemory(cudap1ptrs.y);

        FreehipHostMemory(cudap3ptrs.Abp);
        FreehipHostMemory(cudap3ptrs.xbp);
        FreehipHostMemory(cudap3ptrs.ybp);
        FreeDeviceMemory(cudap3ptrs.A);
        FreeDeviceMemory(cudap3ptrs.x);
        FreeDeviceMemory(cudap3ptrs.y);

        FreehipHostMemory(cudap1transptrs.Abp);
        FreehipHostMemory(cudap1transptrs.xbp);
        FreehipHostMemory(cudap1transptrs.ybp);
        FreeDeviceMemory(cudap1transptrs.y);

        FreehipHostMemory(cudap3transptrs.Abp);
        FreehipHostMemory(cudap3transptrs.xbp);
        FreehipHostMemory(cudap3transptrs.ybp);
        FreeDeviceMemory(cudap3transptrs.y);

        FreeDeviceMemory(d_p1Aptrs);
        FreeDeviceMemory(d_p1xptrs);
        FreeDeviceMemory(d_p1yptrs);
        FreeDeviceMemory(d_p3Aptrs);
        FreeDeviceMemory(d_p3xptrs);
        FreeDeviceMemory(d_p3yptrs);

        FreeDeviceMemory(d_p1transAptrs);
        FreeDeviceMemory(d_p1transxptrs);
        FreeDeviceMemory(d_p1transyptrs);
        FreeDeviceMemory(d_p3transAptrs);
        FreeDeviceMemory(d_p3transxptrs);
        FreeDeviceMemory(d_p3transyptrs);
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::Phase1(){
        hipDeviceSynchronize();
        for(int i=0; i<config.Ntg; i++){
            if(cudap1ptrs.Ms[i] != 0){
                hipblasgemv(cublashandle, HIPBLAS_OP_N,
                            cudap1ptrs.Ms[i], cudap1ptrs.Ks[i],
                            &alpha, (const DeviceType*)cudap1ptrs.Abp[i], cudap1ptrs.Ms[i],
                            (const DeviceType*)cudap1ptrs.xbp[i], 1, &beta,
                            cudap1ptrs.ybp[i], 1);
            }
        }
        hipDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::Phase1Transpose(){
        hipDeviceSynchronize();
        for(int i=0; i<config.Ntg; i++){
            if(cudap1transptrs.Ms[i] != 0){
                hipblasgemv(cublashandle, HIPBLAS_OP_T,
                            cudap1transptrs.Ks[i], cudap1transptrs.Ms[i],
                            &alpha, (const DeviceType*)cudap1transptrs.Abp[i],
                            cudap1transptrs.Ks[i],
                            (const DeviceType*)cudap1transptrs.xbp[i], 1, &beta,
                            cudap1transptrs.ybp[i], 1);
            }
        }
        hipDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::Phase1GetMembuffer(){
        int batchsize = cudap1ptrs.Ms.size();
        GethipHostMemory(&cudap1ptrs.Abp, batchsize);
        GethipHostMemory(&cudap1ptrs.xbp, batchsize);
        GethipHostMemory(&cudap1ptrs.ybp, batchsize);
    }
    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::Phase1GetMembufferTranspose()
    {
        int batchsize = cudap1ptrs.Ms.size();
        GethipHostMemory(&cudap1transptrs.Abp, batchsize);
        GethipHostMemory(&cudap1transptrs.xbp, batchsize);
        GethipHostMemory(&cudap1transptrs.ybp, batchsize);
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::AllocatePhase1Buffer(){
        GetDeviceMemory(&cudap1ptrs.A, cudap1ptrs.Acnt);
        GetDeviceMemory(&cudap1ptrs.x, cudap1ptrs.Xcnt);
        GetDeviceMemory(&cudap1ptrs.y, cudap1ptrs.Ycnt);
        cudap1ptrs.Abp[0] = cudap1ptrs.A;
        cudap1ptrs.xbp[0] = cudap1ptrs.x;
        cudap1ptrs.ybp[0] = cudap1ptrs.y;
    }
    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::AllocatePhase1BufferTranspose(){
        cudap1transptrs.A = cudap3ptrs.A;
        cudap1transptrs.x = cudap1ptrs.x;
        GetDeviceMemory(&cudap1transptrs.y, cudap1transptrs.Ycnt);
        cudap1transptrs.Abp[0] = cudap3ptrs.A; // use phase 3, U bases
        cudap1transptrs.xbp[0] = cudap1ptrs.x; // use phase 1, x
        cudap1transptrs.ybp[0] = cudap1transptrs.y; // create a new buffer
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::Phase1CopyData(){
        auto AvMs = cudap1ptrs.Ms;
        auto AvNs = cudap1ptrs.Ns;
        auto AvKs = cudap1ptrs.Ks;
        for(int i=1; i<config.Ntg; i++){
            size_t AvMK = AvMs[i-1] * AvKs[i-1];
            size_t AvKN = AvKs[i-1] * AvNs[i-1];
            size_t AvMN = AvMs[i-1] * AvNs[i-1];
            cudap1ptrs.Abp[i] =cudap1ptrs.Abp[i-1] + AvMK;
            cudap1ptrs.xbp[i] = cudap1ptrs.xbp[i-1] + AvKN;
            cudap1ptrs.ybp[i] = cudap1ptrs.ybp[i-1] + AvMN;
        }
        // load phase1 A,x to GPU
        CopyDataB2HD((HostType*)cudap1ptrs.A, tlrmvmcpu->p1ptrs.A, tlrmvmcpu->p1ptrs.Acnt);
        CopyDataB2HD((HostType*)cudap1ptrs.x, tlrmvmcpu->p1ptrs.x, tlrmvmcpu->p1ptrs.Xcnt);
    }
    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::Phase1CopyDataTranspose(){
        for(int i=1; i<cudap1transptrs.Ms.size(); i++){
            size_t AvMK = cudap1transptrs.Ms[i-1] * cudap1transptrs.Ks[i-1];
            size_t AvKN = cudap1transptrs.Ks[i-1] * cudap1transptrs.Ns[i-1];
            size_t AvMN = cudap1transptrs.Ms[i-1] * cudap1transptrs.Ns[i-1];
            cudap1transptrs.Abp[i] = cudap1transptrs.Abp[i-1] + AvMK;
            cudap1transptrs.xbp[i] = cudap1transptrs.xbp[i-1] + AvKN;
            cudap1transptrs.ybp[i] = cudap1transptrs.ybp[i-1] + AvMN;
        }
        // no need to copy data. data is copied in normal node.
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::Phase2(){
        hipDeviceSynchronize();
        phase2_nosplit<DeviceType>(cudap1ptrs.y, d_phase2mapping, cudap3ptrs.x,
                                   config.granksum, stream);
        hipDeviceSynchronize();
    }
    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::Phase2Transpose(){
        hipDeviceSynchronize();
        phase2_nosplit<DeviceType>(cudap1transptrs.y, d_phase2mapping_transpose,
                                   cudap3transptrs.x, config.granksum, stream);
        hipDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::Phase2Prepare(){
        GetDeviceMemory(&d_phase2mapping, tlrmvmcpu->h_phase2mapping.size());
        CopyDataB2HD(d_phase2mapping, tlrmvmcpu->h_phase2mapping.data(),tlrmvmcpu->h_phase2mapping.size());
    }
    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::Phase2PrepareTranspose(){
        GetDeviceMemory(&d_phase2mapping_transpose, tlrmvmcpu->h_phase2mappingTranspose.size());
        CopyDataB2HD(d_phase2mapping_transpose, tlrmvmcpu->h_phase2mappingTranspose.data(),
                     tlrmvmcpu->h_phase2mappingTranspose.size());
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::Phase3(){
        hipDeviceSynchronize();
        for(int i=0; i<config.Mtg; i++){
            hipblasgemv(cublashandle, HIPBLAS_OP_N,
                        cudap3ptrs.Ms[i], cudap3ptrs.Ks[i],
                        &alpha, (const DeviceType*)cudap3ptrs.Abp[i], cudap3ptrs.Ms[i],
                        (const DeviceType*)cudap3ptrs.xbp[i], 1, &beta,cudap3ptrs.ybp[i], 1);
        }
        hipDeviceSynchronize();
    }
    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::Phase3Transpose(){
        hipDeviceSynchronize();
        for(int i=0; i<config.Mtg; i++){
            hipblasgemv(cublashandle, HIPBLAS_OP_T,
                        cudap3transptrs.Ks[i], cudap3transptrs.Ms[i],
                        &alpha, (const DeviceType*)cudap3transptrs.Abp[i],
                        cudap3transptrs.Ks[i],
                        (const DeviceType*)cudap3transptrs.xbp[i], 1, &beta,cudap3transptrs.ybp[i], 1);
        }
        hipDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::Phase3GetMembuffer(){
        int batchsize = cudap3ptrs.Ms.size();
        GethipHostMemory(&cudap3ptrs.Abp, batchsize);
        GethipHostMemory(&cudap3ptrs.xbp, batchsize);
        GethipHostMemory(&cudap3ptrs.ybp, batchsize);
    }
    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::Phase3GetMembufferTranspose(){
        int batchsize = cudap3transptrs.Ms.size();
        GethipHostMemory(&cudap3transptrs.Abp, batchsize);
        GethipHostMemory(&cudap3transptrs.xbp, batchsize);
        GethipHostMemory(&cudap3transptrs.ybp, batchsize);
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::AllocatePhase3Buffer(){
        GetDeviceMemory(&cudap3ptrs.A, cudap3ptrs.Acnt);
        GetDeviceMemory(&cudap3ptrs.x, cudap3ptrs.Xcnt);
        GetDeviceMemory(&cudap3ptrs.y, cudap3ptrs.Ycnt);
        cudap3ptrs.Abp[0] = cudap3ptrs.A;
        cudap3ptrs.xbp[0] = cudap3ptrs.x;
        cudap3ptrs.ybp[0] = cudap3ptrs.y;
    }
    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::AllocatePhase3BufferTranspose(){
        cudap3transptrs.A = cudap1ptrs.A;
        cudap3transptrs.x = cudap3ptrs.x;
        GetDeviceMemory(&cudap3transptrs.y, cudap3transptrs.Ycnt);
        cudap3transptrs.Abp[0] = cudap1ptrs.A; // use phase 1, V bases
        cudap3transptrs.xbp[0] = cudap3ptrs.x; // use phase 3, x
        cudap3transptrs.ybp[0] = cudap3transptrs.y; // create a new buffer
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::Phase3CopyData(){
        auto AuMs = cudap3ptrs.Ms;
        auto AuNs = cudap3ptrs.Ns;
        auto AuKs = cudap3ptrs.Ks;
        for(int i=1; i<tlrmvmcpu->config.Mtg; i++){
            size_t AuMK = AuMs[i-1] * AuKs[i-1];
            size_t AuKN = AuKs[i-1] * AuNs[i-1];
            size_t AuMN = AuMs[i-1] * AuNs[i-1];
            cudap3ptrs.Abp[i] = cudap3ptrs.Abp[i-1] + AuMK;
            cudap3ptrs.xbp[i] = cudap3ptrs.xbp[i-1] + AuKN;
            cudap3ptrs.ybp[i] = cudap3ptrs.ybp[i-1] + AuMN;
        }
        // load phase 3 A to GPU
        CopyDataB2HD((HostType*)cudap3ptrs.A, tlrmvmcpu->p3ptrs.A, cudap3ptrs.Acnt);
    }
    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::Phase3CopyDataTranspose(){
        for(int i=1; i<cudap3transptrs.Ms.size(); i++){
            size_t AvMK = cudap3transptrs.Ms[i-1] * cudap3transptrs.Ks[i-1];
            size_t AvKN = cudap3transptrs.Ks[i-1] * cudap3transptrs.Ns[i-1];
            size_t AvMN = cudap3transptrs.Ms[i-1] * cudap3transptrs.Ns[i-1];
            cudap3transptrs.Abp[i] = cudap3transptrs.Abp[i-1] + AvMK;
            cudap3transptrs.xbp[i] = cudap3transptrs.xbp[i-1] + AvKN;
            cudap3transptrs.ybp[i] = cudap3transptrs.ybp[i-1] + AvMN;
        }
        // no need to copy data.
    }


    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::MVMTranspose()
    {
        hipDeviceSynchronize();
        for(int i=0; i<config.Ntg; i++){
            if(cudap1transptrs.Ms[i] != 0){
                hipblasgemv(cublashandle, HIPBLAS_OP_T,
                            cudap1transptrs.Ks[i], cudap1transptrs.Ms[i],
                            &alpha, (const DeviceType*)cudap1transptrs.Abp[i],
                            cudap1transptrs.Ks[i],
                            (const DeviceType*)cudap1transptrs.xbp[i], 1, &beta,
                            cudap1transptrs.ybp[i], 1);
            }
        }
        hipDeviceSynchronize();
        phase2_nosplit<DeviceType>(cudap1transptrs.y, d_phase2mapping_transpose,
                                   cudap3transptrs.x, config.granksum, stream);
        hipDeviceSynchronize();
        for(int i=0; i<config.Mtg; i++){
            hipblasgemv(cublashandle, HIPBLAS_OP_T,
                        cudap3transptrs.Ks[i], cudap3transptrs.Ms[i],
                        &alpha, (const DeviceType*)cudap3transptrs.Abp[i],
                        cudap3transptrs.Ks[i],
                        (const DeviceType*)cudap3transptrs.xbp[i], 1, &beta,cudap3transptrs.ybp[i], 1);
        }
        hipDeviceSynchronize();
    }
    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType,DeviceType>::MVMNoTranspose()
    {
        hipDeviceSynchronize();
        hipblasgemmbatched(cublashandle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                          cudap1ptrs.Ms[0],cudap1ptrs.Ns[0],cudap1ptrs.Ks[0],
                          &alpha, (const DeviceType**)d_p1Aptrs, cudap1ptrs.Ms[0],
                          (const DeviceType**)d_p1xptrs, cudap1ptrs.Ks[0],
                          &beta,d_p1yptrs, cudap1ptrs.Ms[0], cudap1ptrs.Ms.size());
//        for(int i=0; i<config.Ntg; i++){
//            if(cudap1ptrs.Ms[i] != 0){
//                hipblasgemv(cublashandle, HIPBLAS_OP_N,
//                            cudap1ptrs.Ms[i], cudap1ptrs.Ks[i],
//                            &alpha, (const DeviceType*)cudap1ptrs.Abp[i], cudap1ptrs.Ms[i],
//                            (const DeviceType*)cudap1ptrs.xbp[i], 1, &beta,
//                            cudap1ptrs.ybp[i], 1);
//            }
//        }
        hipDeviceSynchronize();
        phase2_nosplit<DeviceType>(cudap1ptrs.y, d_phase2mapping, cudap3ptrs.x,
                                   config.granksum, stream);
        hipDeviceSynchronize();
        hipblasgemmbatched(cublashandle, HIPBLAS_OP_N, HIPBLAS_OP_N,
                          cudap3ptrs.Ms[0],cudap3ptrs.Ns[0],cudap3ptrs.Ks[0],
                          &alpha, (const DeviceType**)d_p3Aptrs, cudap3ptrs.Ms[0],
                          (const DeviceType**)d_p3xptrs, cudap3ptrs.Ks[0],
                          &beta,d_p3yptrs, cudap3ptrs.Ms[0], cudap3ptrs.Ms.size());
//        for(int i=0; i<config.Mtg; i++){
//            hipblasgemv(cublashandle, HIPBLAS_OP_N,
//                        cudap3ptrs.Ms[i], cudap3ptrs.Ks[i],
//                        &alpha, (const DeviceType*)cudap3ptrs.Abp[i], cudap3ptrs.Ms[i],
//                        (const DeviceType*)cudap3ptrs.xbp[i], 1, &beta,cudap3ptrs.ybp[i], 1);
//        }
        hipDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType, DeviceType>::CopyBackResults()
    {
        // use cpu pointers to send output
        if(transpose){
            CopyDataB2HD(tlrmvmcpu->p1transptrs.y, (HostType*)cudap1transptrs.y, tlrmvmcpu->config.granksum);
            CopyDataB2HD(tlrmvmcpu->p3transptrs.x, (HostType*)cudap3transptrs.x, tlrmvmcpu->config.granksum);
            CopyDataB2HD(tlrmvmcpu->p3transptrs.y, (HostType*)cudap3transptrs.y, tlrmvmcpu->config.originM);
        }else{
            CopyDataB2HD(tlrmvmcpu->p1ptrs.y, (HostType*)cudap1ptrs.y, tlrmvmcpu->config.granksum);
            CopyDataB2HD(tlrmvmcpu->p3ptrs.x, (HostType*)cudap3ptrs.x, tlrmvmcpu->config.granksum);
            CopyDataB2HD(tlrmvmcpu->p3ptrs.y, (HostType*)cudap3ptrs.y, tlrmvmcpu->config.originM);
        }
        tlrmvmcpu->CopyToFinalresults();
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType, DeviceType>::MVM() {
        if(transpose){
            MVMTranspose();
        }else{
            MVMNoTranspose();
        }
    }

    template<typename HostType, typename DeviceType>
    void TlrmvmhipConstRank<HostType, DeviceType>::SetTransposeConjugate(bool transpose, bool conjugate) {
        this->transpose = transpose;
        this->conjugate = conjugate;
        tlrmvmcpu->SetTransposeConjugate(transpose, conjugate);
    }



    template class TlrmvmhipConstRank<float, float>;
    template class TlrmvmhipConstRank<double, double>;
    template class TlrmvmhipConstRank<complex<float>, hipComplex>;
    template class TlrmvmhipConstRank<complex<double>, hipDoubleComplex>;

} // namespace cudatlrmvm

