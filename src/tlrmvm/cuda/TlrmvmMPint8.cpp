//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#include "Tlrmvmcuda.hpp"
#include "TlrmvmMPint8.hpp"
#include "cudakernel.cuh"
#include "../../common/AppUtil.hpp"
#include "../../common/Common.hpp"
#include "../cpu/TlrmvmCPU.hpp"
#include "../../common/cuda/cublasInterface.hpp"


namespace cudatlrmvm
{

    CUDAI8Phase1Pointers::CUDAI8Phase1Pointers() {}
    CUDAI8Phase3Pointers::CUDAI8Phase3Pointers() {}

    TlrmvmMPint8::TlrmvmMPint8(){}

    TlrmvmMPint8::TlrmvmMPint8(TlrmvmConfig tlrmvmconfig)
            :config(tlrmvmconfig)
    {
        transpose = false;
        conjugate = false;
        init_alpha_beta(alpha, beta);
        tlrmvmcpu = std::make_shared<TlrmvmCPU<complex<float>>>(tlrmvmconfig);
    }

    void TlrmvmMPint8::UpdateConfig(TlrmvmConfig &tlrmvmconfig)
    {
        cout << "UpdateConfig not implemented." << endl;
        exit(0);
    }

    void TlrmvmMPint8::setX(complex<float> * xvector, size_t xlength)
    {
        tlrmvmcpu->setX((complex<float>*)xvector, xlength);
        TryConjugateXvec();
    }

    void TlrmvmMPint8::TryConjugateXvec() {
        // no transpose logic
//        tlrmvmcpu->TryConjugateXvec();
        auto x_h = new cuComplex[tlrmvmcpu->p1ptrs.Xcnt];
        auto xi8_h = new cuInt8Complex [tlrmvmcpu->p1ptrs.Xcnt];
        memcpy(x_h, tlrmvmcpu->p1ptrs.x, sizeof(complex<float>) * tlrmvmcpu->p1ptrs.Xcnt);
        cuComplex minx;minx.x = x_h[0].x;minx.y = x_h[0].y;
        cudap1ptrs.maxx.resize(config.Ntg,minx);
        for(int i=0; i<config.Ntg; i++) cudap1ptrs.maxx[i] = minx;
        // get max
        for(int i=0; i<config.Ntg; i++){
            for(size_t j=0; j<config.nb; j++){
                cudap1ptrs.maxx[i].x = max(abs(x_h[i * config.nb + j].x), cudap1ptrs.maxx[i].x);
                cudap1ptrs.maxx[i].y = max(abs(x_h[i * config.nb + j].y), cudap1ptrs.maxx[i].y);
            }
        }
        // convert to int8
        for(int i=0; i<config.Ntg; i++){
            for(size_t j=0; j<config.nb; j++){
                xi8_h[i * config.nb + j].x = (int8_t)(x_h[i*config.nb+j].x / cudap1ptrs.maxx[i].x * 125.0);
                xi8_h[i * config.nb + j].y = (int8_t)(x_h[i*config.nb+j].y / cudap1ptrs.maxx[i].y * 125.0);
            }
            if(i == 0)
                cout << (int16_t)xi8_h[i].x << endl;
        }
        // load phase1 A,x to GPU
        CopyDataB2HD(cudap1ptrs.x, xi8_h, tlrmvmcpu->p1ptrs.Xcnt);
        delete[] x_h;
        delete[] xi8_h;
    }

    void TlrmvmMPint8::TryConjugateResults() {
//        if(!conjugate) return;
//        if(transpose){
//            ConjugateDriver<cuHalfComplex>(cudap3transptrs.y, config.originN, streamptr[0]);
//            CopyDataB2HD(tlrmvmcpu->p3transptrs.y, (complex<float>*)cudap3transptrs.y, tlrmvmcpu->config.originM);
//        }else{
//            ConjugateDriver<cuHalfComplex>(cudap3ptrs.y, config.originM, streamptr[0]);
//            CopyDataB2HD(tlrmvmcpu->p3ptrs.y, (complex<float>*)cudap3ptrs.y, tlrmvmcpu->config.originM);
//        }
    }

    void TlrmvmMPint8::StreamInit(int streamsize){
        this->stream_size = streamsize;
        streamptr = new cudaStream_t[streamsize];
        cublashandleptr = new cublasHandle_t[streamsize];
        for(int i=0; i<streamsize; i++)
            cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
        for(int i=0; i<streamsize; i++)
            cublasCreate(&cublashandleptr[i]);
        for(int i=0; i<streamsize; i++)
            cublasSetStream(cublashandleptr[i], streamptr[i]);
        CUDACHECK(cudaEventCreate(&event_start));
        CUDACHECK(cudaEventCreate(&event_phase2finish));
        events = new cudaEvent_t[2*streamsize];
        for(int i=0; i<2*streamsize; i++) CUDACHECK(cudaEventCreate(&events[i]));
        // graph
        graphCreated = false;
    }

    void TlrmvmMPint8::StreamDestroy(){
        for(int i=0; i<stream_size; i++) cublasDestroy(cublashandleptr[i]);
        for(int i=0; i<stream_size; i++) cudaStreamDestroy(streamptr[i]);
        delete[] cublashandleptr;
        delete[] streamptr;
        CUDACHECK(cudaEventDestroy(event_start));
        CUDACHECK(cudaEventDestroy(event_phase2finish));
        for(int i=0; i<2*stream_size; i++) CUDACHECK(cudaEventDestroy(events[i]));
        delete[] events;
        // graph
        if(graphCreated){
            cudaGraphExecDestroy(instance);
            cudaGraphDestroy(graph);
        }
    }

    void TlrmvmMPint8::MemoryInit(){
        tlrmvmcpu->MemoryInit();
        cudap1ptrs.Acnt = tlrmvmcpu->p1ptrs.Acnt;
        cudap3ptrs.Acnt = tlrmvmcpu->p3ptrs.Acnt;
        cudap1ptrs.Xcnt = tlrmvmcpu->p1ptrs.Xcnt;
        cudap3ptrs.Xcnt = tlrmvmcpu->p3ptrs.Xcnt;
        cudap1ptrs.Ycnt = tlrmvmcpu->p1ptrs.Ycnt;
        cudap3ptrs.Ycnt = tlrmvmcpu->p3ptrs.Ycnt;

        cudap1ptrs.Ms = tlrmvmcpu->p1ptrs.Ms;
        cudap3ptrs.Ms = tlrmvmcpu->p3ptrs.Ms;
        cudap1ptrs.Ks = tlrmvmcpu->p1ptrs.Ks;
        cudap3ptrs.Ks = tlrmvmcpu->p3ptrs.Ks;
        cudap1ptrs.Ns = tlrmvmcpu->p1ptrs.Ns;
        cudap3ptrs.Ns = tlrmvmcpu->p3ptrs.Ns;

        Phase1GetMembuffer();
        AllocatePhase1Buffer();
        Phase1CopyData();
        Phase2Prepare();
        Phase3GetMembuffer();
        AllocatePhase3Buffer();
        Phase3CopyData();
//        // transpose
//        Phase1GetMembufferTranspose();
//        AllocatePhase1BufferTranspose();
//        Phase1CopyDataTranspose();
//        Phase2PrepareTranspose();
//        Phase3GetMembufferTranspose();
//        AllocatePhase3BufferTranspose();
//        Phase3CopyDataTranspose();
    }

    void TlrmvmMPint8::MemoryFree(){
        tlrmvmcpu->MemoryFree();
        FreecuHostMemory(cudap1ptrs.Abp);
        FreecuHostMemory(cudap1ptrs.xbp);
        FreecuHostMemory(cudap1ptrs.ybp);
        FreeDeviceMemory(cudap1ptrs.A);
        FreeDeviceMemory(cudap1ptrs.x);
        FreeDeviceMemory(cudap1ptrs.y);

        FreecuHostMemory(cudap3ptrs.Abp);
        FreecuHostMemory(cudap3ptrs.xbp);
        FreecuHostMemory(cudap3ptrs.ybp);
        FreeDeviceMemory(cudap3ptrs.A);
        FreeDeviceMemory(cudap3ptrs.x);
        FreeDeviceMemory(cudap3ptrs.y);

//        FreecuHostMemory(cudap1transptrs.Abp);
//        FreecuHostMemory(cudap1transptrs.xbp);
//        FreecuHostMemory(cudap1transptrs.ybp);
//        FreeDeviceMemory(cudap1transptrs.y);
//
//        FreecuHostMemory(cudap3transptrs.Abp);
//        FreecuHostMemory(cudap3transptrs.xbp);
//        FreecuHostMemory(cudap3transptrs.ybp);
//        FreeDeviceMemory(cudap3transptrs.y);
    }

    void TlrmvmMPint8::Phase1(){
        cudaDeviceSynchronize();
        for(int i=0; i<config.Ntg; i++){
            if(cudap1ptrs.Ms[i] != 0){
                Igemv_Phase1_driver(cudap1ptrs.Abp[i], cudap1ptrs.xbp[i],
                                    cudap1ptrs.maxA[i], cudap1ptrs.maxx[i],
                                    cudap1ptrs.ybp[i],
                                    cudap1ptrs.Ms[i], cudap1ptrs.Ks[i], streamptr[i % stream_size]);
            }
        }
        cudaDeviceSynchronize();
    }

    void TlrmvmMPint8::Phase1Transpose(){
//        cudaDeviceSynchronize();
//        for(int i=0; i<config.Ntg; i++){
//            if(cudap1transptrs.Ms[i] != 0){
//                cublasgemv(cublashandleptr[i%stream_size], CUBLAS_OP_T,
//                           cudap1transptrs.Ks[i], cudap1transptrs.Ms[i],
//                           &alpha, (const cuComplex *)cudap1transptrs.Abp[i],
//                           cudap1transptrs.Ks[i],
//                           (const cuComplex *)cudap1transptrs.xbp[i], 1, &beta,
//                           cudap1transptrs.ybp[i], 1);
//            }
//        }
//        cudaDeviceSynchronize();
    }

    void TlrmvmMPint8::Phase1GetMembuffer(){
        int batchsize = cudap1ptrs.Ms.size();
        GetcuHostMemory(&cudap1ptrs.Abp, batchsize);
        GetcuHostMemory(&cudap1ptrs.xbp, batchsize);
        GetcuHostMemory(&cudap1ptrs.ybp, batchsize);
    }

    void TlrmvmMPint8::Phase1GetMembufferTranspose()
    {
//        int batchsize = cudap1ptrs.Ms.size();
//        GetcuHostMemory(&cudap1transptrs.Abp, batchsize);
//        GetcuHostMemory(&cudap1transptrs.xbp, batchsize);
//        GetcuHostMemory(&cudap1transptrs.ybp, batchsize);
    }

    void TlrmvmMPint8::AllocatePhase1Buffer(){
        GetDeviceMemory(&cudap1ptrs.A, cudap1ptrs.Acnt);
        GetDeviceMemory(&cudap1ptrs.x, cudap1ptrs.Xcnt);
        GetDeviceMemory(&cudap1ptrs.y, cudap1ptrs.Ycnt);
        cudap1ptrs.Abp[0] = cudap1ptrs.A;
        cudap1ptrs.xbp[0] = cudap1ptrs.x;
        cudap1ptrs.ybp[0] = cudap1ptrs.y;
    }

    void TlrmvmMPint8::AllocatePhase1BufferTranspose(){
//        cudap1transptrs.A = cudap3ptrs.A;
//        cudap1transptrs.x = cudap1ptrs.x;
//        GetDeviceMemory(&cudap1transptrs.y, cudap1transptrs.Ycnt);
//        cudap1transptrs.Abp[0] = cudap3ptrs.A; // use phase 3, U bases
//        cudap1transptrs.xbp[0] = cudap1ptrs.x; // use phase 1, x
//        cudap1transptrs.ybp[0] = cudap1transptrs.y; // create a new buffer
    }

    void TlrmvmMPint8::Phase1CopyData(){
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
        auto A_h = new cuComplex[tlrmvmcpu->p1ptrs.Acnt];
        auto x_h = new cuComplex[tlrmvmcpu->p1ptrs.Xcnt];
        auto Ai8_h = new cuInt8Complex [tlrmvmcpu->p1ptrs.Acnt];
        auto xi8_h = new cuInt8Complex [tlrmvmcpu->p1ptrs.Xcnt];
        memcpy(A_h, tlrmvmcpu->p1ptrs.A, sizeof(complex<float>) * tlrmvmcpu->p1ptrs.Acnt);
        memcpy(x_h, tlrmvmcpu->p1ptrs.x, sizeof(complex<float>) * tlrmvmcpu->p1ptrs.Xcnt);

        cuComplex minA;minA.x = A_h[0].x;minA.y = A_h[0].y;
        cuComplex minx;minx.x = x_h[0].x;minx.y = x_h[0].y;
        cudap1ptrs.maxA.resize(config.Ntg,minA);
        cudap1ptrs.maxx.resize(config.Ntg,minx);
        for(int i=0; i<config.Ntg; i++) cudap1ptrs.maxA[i] = minA;
        for(int i=0; i<config.Ntg; i++) cudap1ptrs.maxx[i] = minx;
        size_t Aoffset = 0;
        // get max
        for(int i=0; i<config.Ntg; i++){
            for(size_t j=0; j<config.nb; j++){
                cudap1ptrs.maxx[i].x = max(abs(x_h[i * config.nb + j].x), cudap1ptrs.maxx[i].x);
                cudap1ptrs.maxx[i].y = max(abs(x_h[i * config.nb + j].y), cudap1ptrs.maxx[i].y);
            }
            for(size_t j=0; j<cudap1ptrs.Ms[i]*cudap1ptrs.Ks[i]; j++){
                cudap1ptrs.maxA[i].x = max(abs(A_h[Aoffset + j].x), cudap1ptrs.maxA[i].x);
                cudap1ptrs.maxA[i].y = max(abs(A_h[Aoffset + j].y), cudap1ptrs.maxA[i].y);
            }
            Aoffset += cudap1ptrs.Ms[i]*cudap1ptrs.Ks[i];
        }
        // convert to int8
        Aoffset = 0;
        for(int i=0; i<config.Ntg; i++){
            for(size_t j=0; j<config.nb; j++){
                xi8_h[i * config.nb + j].x = (int8_t)(x_h[i*config.nb+j].x / cudap1ptrs.maxx[i].x * 125.0);
                xi8_h[i * config.nb + j].y = (int8_t)(x_h[i*config.nb+j].y / cudap1ptrs.maxx[i].y * 125.0);
            }
            for(size_t j=0; j<cudap1ptrs.Ms[i]*cudap1ptrs.Ks[i]; j++){
                Ai8_h[Aoffset + j].x = (int8_t)(A_h[Aoffset + j].x / cudap1ptrs.maxA[i].x * 125.0);
                Ai8_h[Aoffset + j].y = (int8_t)(A_h[Aoffset + j].y / cudap1ptrs.maxA[i].y * 125.0);
            }
            Aoffset += cudap1ptrs.Ms[i]*cudap1ptrs.Ks[i];
        }

        // load phase1 A,x to GPU
        CopyDataB2HD(cudap1ptrs.A, Ai8_h, tlrmvmcpu->p1ptrs.Acnt);
        CopyDataB2HD(cudap1ptrs.x, xi8_h, tlrmvmcpu->p1ptrs.Xcnt);
        delete[] A_h;
        delete[] x_h;
        delete[] Ai8_h;
        delete[] xi8_h;
    }

    void TlrmvmMPint8::Phase1CopyDataTranspose(){
//        for(int i=1; i<cudap1transptrs.Ms.size(); i++){
//            size_t AvMK = cudap1transptrs.Ms[i-1] * cudap1transptrs.Ks[i-1];
//            size_t AvKN = cudap1transptrs.Ks[i-1] * cudap1transptrs.Ns[i-1];
//            size_t AvMN = cudap1transptrs.Ms[i-1] * cudap1transptrs.Ns[i-1];
//            cudap1transptrs.Abp[i] = cudap1transptrs.Abp[i-1] + AvMK;
//            cudap1transptrs.xbp[i] = cudap1transptrs.xbp[i-1] + AvKN;
//            cudap1transptrs.ybp[i] = cudap1transptrs.ybp[i-1] + AvMN;
//        }
    }

    void TlrmvmMPint8::Phase2(){
        cudaDeviceSynchronize();
        phase2_nosplit<cuHalfComplex>(cudap1ptrs.y, d_phase2mapping, cudap3ptrs.x,
                                      config.workmatgranksum, streamptr[0]);
        cudaDeviceSynchronize();
    }
    void TlrmvmMPint8::Phase2Transpose(){
//        cudaDeviceSynchronize();
//        phase2_nosplit<cuHalfComplex>(cudap1transptrs.y, d_phase2mapping_transpose,
//                                      cudap3transptrs.x, config.granksum, streamptr[0]);
//        cudaDeviceSynchronize();
    }

    void TlrmvmMPint8::Phase2Prepare(){
        GetDeviceMemory(&d_phase2mapping, tlrmvmcpu->h_phase2mapping.size());
        CopyDataB2HD(d_phase2mapping, tlrmvmcpu->h_phase2mapping.data(),
                     tlrmvmcpu->h_phase2mapping.size());
    }
    void TlrmvmMPint8::Phase2PrepareTranspose(){
        GetDeviceMemory(&d_phase2mapping_transpose, tlrmvmcpu->h_phase2mappingTranspose.size());
        CopyDataB2HD(d_phase2mapping_transpose, tlrmvmcpu->h_phase2mappingTranspose.data(),
                     tlrmvmcpu->h_phase2mappingTranspose.size());
    }

    void TlrmvmMPint8::Phase3(){
        cudaDeviceSynchronize();
        for(int i=0; i<config.Mtg; i++){
//            Igemv_Phase3_driver(cudap3ptrs.Abp[i], cudap3ptrs.xbp[i],cudap3ptrs.maxA[i],
//                                cudap3ptrs.ybp[i],
//                                cudap3ptrs.Ms[i], cudap3ptrs.Ks[i], streamptr[i % stream_size]);
        }
        cudaDeviceSynchronize();
        CUDACHECK(cudaGetLastError());
    }
    void TlrmvmMPint8::Phase3Transpose(){
        cudaDeviceSynchronize();
        for(int i=0; i<config.Mtg; i++){
//            cublasgemv(cublashandleptr[i%stream_size], CUBLAS_OP_T,
//                       cudap3transptrs.Ks[i], cudap3transptrs.Ms[i],
//                       &alpha, (const cuComplex*)cudap3transptrs.Abp[i],
//                       cudap3transptrs.Ks[i],
//                       (const cuComplex*)cudap3transptrs.xbp[i], 1, &beta,cudap3transptrs.ybp[i], 1);
        }
        cudaDeviceSynchronize();
    }

    void TlrmvmMPint8::Phase3GetMembuffer(){
        int batchsize = cudap3ptrs.Ms.size();
        GetcuHostMemory(&cudap3ptrs.Abp, batchsize);
        GetcuHostMemory(&cudap3ptrs.xbp, batchsize);
        GetcuHostMemory(&cudap3ptrs.ybp, batchsize);
    }
    void TlrmvmMPint8::Phase3GetMembufferTranspose(){
//        int batchsize = cudap3transptrs.Ms.size();
//        GetcuHostMemory(&cudap3transptrs.Abp, batchsize);
//        GetcuHostMemory(&cudap3transptrs.xbp, batchsize);
//        GetcuHostMemory(&cudap3transptrs.ybp, batchsize);
    }

    void TlrmvmMPint8::AllocatePhase3Buffer(){
        GetDeviceMemory(&cudap3ptrs.A, cudap3ptrs.Acnt);
        GetDeviceMemory(&cudap3ptrs.x, cudap3ptrs.Xcnt);
        GetDeviceMemory(&cudap3ptrs.y, cudap3ptrs.Ycnt);
        cudap3ptrs.Abp[0] = cudap3ptrs.A;
        cudap3ptrs.xbp[0] = cudap3ptrs.x;
        cudap3ptrs.ybp[0] = cudap3ptrs.y;
    }
    void TlrmvmMPint8::AllocatePhase3BufferTranspose(){
//        cudap3transptrs.A = cudap1ptrs.A;
//        cudap3transptrs.x = cudap3ptrs.x;
//        GetDeviceMemory(&cudap3transptrs.y, cudap3transptrs.Ycnt);
//        cudap3transptrs.Abp[0] = cudap1ptrs.A; // use phase 1, V bases
//        cudap3transptrs.xbp[0] = cudap3ptrs.x; // use phase 3, x
//        cudap3transptrs.ybp[0] = cudap3transptrs.y; // create a new buffer
    }

    void TlrmvmMPint8::Phase3CopyData(){
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
//        auto A_h = new cuComplex [cudap3ptrs.Acnt];
//        auto Ai8_h = new cuInt8Complex [cudap3ptrs.Acnt];
//        memcpy(A_h, tlrmvmcpu->p3ptrs.A, sizeof(complex<float>) * cudap3ptrs.Acnt);
//        cuComplex minA;
//        minA.x = A_h[0].x;
//        minA.y = A_h[0].y;
//        cudap3ptrs.maxA.resize(config.Ntg,minA);
//        for(int i=0; i<config.Ntg; i++) cudap3ptrs.maxA[i] = minA;
//        size_t Aoffset = 0;
//        // get max
//        for(int i=0; i<config.Ntg; i++){
//            for(size_t j=0; j<cudap3ptrs.Ms[i]*cudap3ptrs.Ks[i]; j++){
//                cudap3ptrs.maxA[i].x = max(abs(A_h[Aoffset + j].x), cudap3ptrs.maxA[i].x);
//                cudap3ptrs.maxA[i].y = max(abs(A_h[Aoffset + j].y), cudap3ptrs.maxA[i].y);
//            }
//            Aoffset += cudap3ptrs.Ms[i]*cudap3ptrs.Ks[i];
//        }
//        // convert to int8
//        Aoffset = 0;
//        for(int i=0; i<config.Ntg; i++){
//            for(size_t j=0; j<cudap3ptrs.Ms[i]*cudap3ptrs.Ks[i]; j++){
//                Ai8_h[Aoffset + j].x = (int8_t)(A_h[Aoffset + j].x / cudap3ptrs.maxA[i].x * 125.0);
//                Ai8_h[Aoffset + j].y = (int8_t)(A_h[Aoffset + j].y / cudap3ptrs.maxA[i].y * 125.0);
//            }
//            Aoffset += cudap3ptrs.Ms[i]*cudap3ptrs.Ks[i];
//        }
//        // load phase1 A,x to GPU
//        CopyDataB2HD(cudap3ptrs.A,Ai8_h,cudap3ptrs.Acnt);
//        delete[] A_h;
//        delete[] Ai8_h;
        // copy compute A
        auto cA_h = new cuHalfComplex[cudap3ptrs.Acnt];
        size_t Aoffset = 0;
        for(int i=0; i<config.Ntg; i++){
            for(size_t j=0; j<cudap3ptrs.Ms[i]*cudap3ptrs.Ks[i]; j++){
                cA_h[Aoffset + j].x = (half)(tlrmvmcpu->p3ptrs.A[Aoffset + j].real());
                cA_h[Aoffset + j].y = (half)(tlrmvmcpu->p3ptrs.A[Aoffset + j].imag());
            }
            Aoffset += cudap3ptrs.Ms[i]*cudap3ptrs.Ks[i];
        }
        CopyDataB2HD(cudap3ptrs.A, cA_h, cudap3ptrs.Acnt);
        delete[] cA_h;
    }

    void TlrmvmMPint8::Phase3CopyDataTranspose(){
//        for(int i=1; i<cudap3transptrs.Ms.size(); i++){
//            size_t AvMK = cudap3transptrs.Ms[i-1] * cudap3transptrs.Ks[i-1];
//            size_t AvKN = cudap3transptrs.Ks[i-1] * cudap3transptrs.Ns[i-1];
//            size_t AvMN = cudap3transptrs.Ms[i-1] * cudap3transptrs.Ns[i-1];
//            cudap3transptrs.Abp[i] = cudap3transptrs.Abp[i-1] + AvMK;
//            cudap3transptrs.xbp[i] = cudap3transptrs.xbp[i-1] + AvKN;
//            cudap3transptrs.ybp[i] = cudap3transptrs.ybp[i-1] + AvMN;
//        }
    }

    void TlrmvmMPint8::MVMGraphTranspose() {
//        if(!graphCreated){
//            cudaStreamBeginCapture(streamptr[0],cudaStreamCaptureModeGlobal);
//            cudaEventRecord(event_start, streamptr[0]);
//            for(int streami=1; streami<stream_size; streami++){
//                cudaStreamWaitEvent(streamptr[streami], event_start);
//            }
//            // phase 1 transpose
//            for(int i=0; i<config.Ntg; i++){
//                if(cudap1transptrs.Ms[i] != 0){
//                    cublasgemv(cublashandleptr[i%stream_size], CUBLAS_OP_T,
//                               cudap1transptrs.Ks[i], cudap1transptrs.Ms[i],
//                               &alpha, (const cuComplex*)cudap1transptrs.Abp[i],
//                               cudap1transptrs.Ks[i],
//                               (const cuComplex*)cudap1transptrs.xbp[i], 1, &beta,
//                               cudap1transptrs.ybp[i], 1);
//                }
//            }
//            for(int streami=1; streami < stream_size; streami++){
//                cudaEventRecord(events[streami], streamptr[streami]);
//            }
//            for(int streami=1; streami < stream_size; streami++){
//                cudaStreamWaitEvent(streamptr[0], events[streami]);
//            }
//            // phase 2 transpose
//            phase2_nosplit<cuComplex>(cudap1transptrs.y, d_phase2mapping_transpose,
//                                       cudap3transptrs.x, config.granksum, streamptr[0]);
//            cudaEventRecord(events[0], streamptr[0]);
//            for(int streami=1; streami < stream_size; streami++){
//                cudaStreamWaitEvent(streamptr[streami], events[0]);
//            }
//            // phase 3 transpose
//            for(int i=0; i<config.Mtg; i++){
//                cublasgemv(cublashandleptr[i%stream_size], CUBLAS_OP_T,
//                           cudap3transptrs.Ks[i], cudap3transptrs.Ms[i],
//                           &alpha, (const cuComplex*)cudap3transptrs.Abp[i],
//                           cudap3transptrs.Ks[i],
//                           (const cuComplex*)cudap3transptrs.xbp[i], 1, &beta,cudap3transptrs.ybp[i], 1);
//            }
//            // final merge
//            for(int streami=1; streami < stream_size; streami++){
//                cudaEventRecord(events[stream_size + streami], streamptr[streami]);
//            }
//            for(int streami=1; streami < stream_size; streami++){
//                cudaStreamWaitEvent(streamptr[0], events[stream_size + streami]);
//            }
//            cudaStreamEndCapture(streamptr[0], &graph);
//            cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
//            graphCreated = true;
//        }
//        cudaGraphLaunch(instance, streamptr[0]);
//        cudaStreamSynchronize(streamptr[0]);
    }

    void TlrmvmMPint8::MVMGraphNoTranspose() {
        if(!graphCreated){
            cudaStreamBeginCapture(streamptr[0],cudaStreamCaptureModeGlobal);
            cudaEventRecord(event_start, streamptr[0]);
            for(int streami=1; streami<stream_size; streami++){
                cudaStreamWaitEvent(streamptr[streami], event_start);
            }

            for(int i=0; i<config.Ntg; i++){
                if(cudap1ptrs.Ms[i] != 0){
                    Igemv_Phase1_driver(cudap1ptrs.Abp[i], cudap1ptrs.xbp[i],
                                        cudap1ptrs.maxA[i], cudap1ptrs.maxx[i],
                                        cudap1ptrs.ybp[i],
                                        cudap1ptrs.Ms[i], cudap1ptrs.Ks[i], streamptr[i % stream_size]);
                }
            }

            for(int streami=1; streami < stream_size; streami++){
                cudaEventRecord(events[streami], streamptr[streami]);
            }
            for(int streami=1; streami < stream_size; streami++){
                cudaStreamWaitEvent(streamptr[0], events[streami]);
            }
            // phase 2
            phase2_nosplit<cuHalfComplex>(cudap1ptrs.y, d_phase2mapping, cudap3ptrs.x,
                                          config.workmatgranksum, streamptr[0]);
            cudaEventRecord(events[0], streamptr[0]);
            for(int streami=1; streami < stream_size; streami++){
                cudaStreamWaitEvent(streamptr[streami], events[0]);
            }
            // phase 3
//            for(int i=0; i<config.Mtg; i++){
//                Igemv_Phase3_driver(cudap3ptrs.Abp[i], cudap3ptrs.xbp[i],cudap3ptrs.maxA[i],
//                                    cudap3ptrs.ybp[i],
//                                    cudap3ptrs.Ms[i], cudap3ptrs.Ks[i], streamptr[i % stream_size]);
//            }
            for(int i=0; i<config.Mtg; i++){
                Hgemv_Phase3_driver(cudap3ptrs.Abp[i], cudap3ptrs.xbp[i], cudap3ptrs.ybp[i],
                                    cudap3ptrs.Ms[i], cudap3ptrs.Ks[i], streamptr[i % stream_size]);
            }
            // final merge
            for(int streami=1; streami < stream_size; streami++){
                cudaEventRecord(events[stream_size + streami], streamptr[streami]);
            }
            for(int streami=1; streami < stream_size; streami++){
                cudaStreamWaitEvent(streamptr[0], events[stream_size + streami]);
            }
            cudaStreamEndCapture(streamptr[0], &graph);
            cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
            graphCreated = true;
        }
        cudaGraphLaunch(instance, streamptr[0]);
        cudaStreamSynchronize(streamptr[0]);
    }

    void TlrmvmMPint8::MVMGraph(){
        if(transpose){
            MVMGraphTranspose();
        }else{
            MVMGraphNoTranspose();
        }
    }

    void TlrmvmMPint8::MVMTranspose()
    {
//        cudaDeviceSynchronize();
//        for(int i=0; i<config.Ntg; i++){
//            if(cudap1transptrs.Ms[i] != 0){
//                cublasgemv(cublashandleptr[i%stream_size], CUBLAS_OP_T,
//                           cudap1transptrs.Ks[i], cudap1transptrs.Ms[i],
//                           &alpha, (const cuComplex*)cudap1transptrs.Abp[i],
//                           cudap1transptrs.Ks[i],
//                           (const cuComplex*)cudap1transptrs.xbp[i], 1, &beta,
//                           cudap1transptrs.ybp[i], 1);
//            }
//        }
//        cudaDeviceSynchronize();
//        phase2_nosplit<cuComplex>(cudap1transptrs.y, d_phase2mapping_transpose,
//                                   cudap3transptrs.x, config.granksum, streamptr[0]);
//        cudaDeviceSynchronize();
//        for(int i=0; i<config.Mtg; i++){
//            cublasgemv(cublashandleptr[i%stream_size], CUBLAS_OP_T,
//                       cudap3transptrs.Ks[i], cudap3transptrs.Ms[i],
//                       &alpha, (const cuComplex*)cudap3transptrs.Abp[i],
//                       cudap3transptrs.Ks[i],
//                       (const cuComplex*)cudap3transptrs.xbp[i], 1, &beta,cudap3transptrs.ybp[i], 1);
//        }
//        cudaDeviceSynchronize();
    }
    void TlrmvmMPint8::MVMNoTranspose()
    {
        cudaDeviceSynchronize();
        for(int i=0; i<config.Ntg; i++){
            if(cudap1ptrs.Ms[i] != 0){
                Igemv_Phase1_driver(cudap1ptrs.Abp[i], cudap1ptrs.xbp[i],
                                    cudap1ptrs.maxA[i], cudap1ptrs.maxx[i],
                                    cudap1ptrs.ybp[i],
                                    cudap1ptrs.Ms[i], cudap1ptrs.Ks[i], streamptr[i % stream_size]);
            }
        }
        cudaDeviceSynchronize();
        phase2_nosplit<cuHalfComplex>(cudap1ptrs.y, d_phase2mapping, cudap3ptrs.x,
                                      config.workmatgranksum, streamptr[0]);
        cudaDeviceSynchronize();
//        for(int i=0; i<config.Mtg; i++){
//            Igemv_Phase3_driver(cudap3ptrs.Abp[i], cudap3ptrs.xbp[i],cudap3ptrs.maxA[i],
//                                cudap3ptrs.ybp[i],
//                                cudap3ptrs.Ms[i], cudap3ptrs.Ks[i], streamptr[i % stream_size]);
//        }
        cudaDeviceSynchronize();
    }

    void TlrmvmMPint8::CopyBackResults()
    {
        // use cpu pointers to send output
        if(transpose){
//            CopyDataB2HD(tlrmvmcpu->p1transptrs.y, (complex<float>*)cudap1transptrs.y, tlrmvmcpu->config.granksum);
//            CopyDataB2HD(tlrmvmcpu->p3transptrs.x, (complex<float>*)cudap3transptrs.x, tlrmvmcpu->config.granksum);
//            CopyDataB2HD(tlrmvmcpu->p3transptrs.y, (complex<float>*)cudap3transptrs.y, tlrmvmcpu->config.originM);
        }else{
            auto halfp1y = new cuHalfComplex[tlrmvmcpu->config.workmatgranksum];
            auto halfp3x = new cuHalfComplex[tlrmvmcpu->config.workmatgranksum];
            auto halfp3y = new cuHalfComplex[tlrmvmcpu->config.originM];
            CopyDataB2HD(halfp1y, cudap1ptrs.y, tlrmvmcpu->config.workmatgranksum);
            CopyDataB2HD(halfp3x, cudap3ptrs.x, tlrmvmcpu->config.workmatgranksum);
            CopyDataB2HD(halfp3y, cudap3ptrs.y, tlrmvmcpu->config.originM);
            for(size_t i=0; i<tlrmvmcpu->config.workmatgranksum; i++){
                tlrmvmcpu->p1ptrs.y[i] = complex<float>((float) halfp1y[i].x, (float)halfp1y[i].y);
                tlrmvmcpu->p3ptrs.x[i] = complex<float>((float) halfp3x[i].x, (float)halfp3x[i].y);
            }
            for(size_t i=0; i<tlrmvmcpu->config.originM; i++)
                tlrmvmcpu->p3ptrs.y[i] = complex<float>((float) halfp3y[i].x, (float)halfp3y[i].y);
            delete[] halfp1y;
            delete[] halfp3x;
            delete[] halfp3y;
        }
        tlrmvmcpu->CopyToFinalresults();
    }

    void TlrmvmMPint8::MVM() {
        if(transpose){
            MVMTranspose();
        }else{
            MVMNoTranspose();
        }
    }

    void TlrmvmMPint8::SetTransposeConjugate(bool transpose, bool conjugate) {
        this->transpose = transpose;
        this->conjugate = conjugate;
        tlrmvmcpu->SetTransposeConjugate(transpose, conjugate);
    }

} // namespace cudatlrmvm

