//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#include "Tlrmvmcuda.hpp"
#include "TlrmvmMPfp16.hpp"
#include "cudakernel.cuh"
#include "../../common/AppUtil.hpp"
#include "../../common/Common.hpp"
#include "../cpu/TlrmvmCPU.hpp"
#include "../../common/cuda/cublasInterface.hpp"


namespace cudatlrmvm
{

    TlrmvmMPfp16::TlrmvmMPfp16(){}

    TlrmvmMPfp16::TlrmvmMPfp16(TlrmvmConfig tlrmvmconfig)
    :config(tlrmvmconfig)
    {
        transpose = false;
        conjugate = false;
        init_alpha_beta(alpha, beta);
        tlrmvmcpu = std::make_shared<TlrmvmCPU<complex<float>>>(tlrmvmconfig);
    }

    void TlrmvmMPfp16::UpdateConfig(TlrmvmConfig &tlrmvmconfig)
    {
        cout << "UpdateConfig not implemented." << endl;
        exit(0);
    }

    void TlrmvmMPfp16::setX(complex<float> * xvector, size_t xlength)
    {
        tlrmvmcpu->setX((complex<float>*)xvector, xlength);
//        for(int i=0; i<10; i++) cout << xvector[i].real() << ", " << xvector[i].imag() << endl;
        TryConjugateXvec();
    }

    void TlrmvmMPfp16::TryConjugateXvec() {
        // no transpose logic
        tlrmvmcpu->TryConjugateXvec();
        auto fp16ptr = new cuHalfComplex[tlrmvmcpu->xmat.Shape()[0]];
        cout << "------" << endl;
        for(size_t i=0; i<tlrmvmcpu->xmat.Shape()[0]; i++){
            if(i < 10)
            cout << tlrmvmcpu->p1ptrs.x[i] << endl;
            fp16ptr[i] = cuHalfComplex(tlrmvmcpu->p1ptrs.x[i].real(), tlrmvmcpu->p1ptrs.x[i].imag());
            if(i < 10)
            cout << (float)fp16ptr[i].x << endl;
        }
        cout << "--------" << endl;
        CopyDataB2HD(cudap1ptrs.x, fp16ptr, tlrmvmcpu->xmat.Shape()[0]);
        delete[] fp16ptr;
    }

    void TlrmvmMPfp16::TryConjugateResults() {
        if(!conjugate) return;
        if(transpose){
            ConjugateDriver<cuHalfComplex>(cudap3transptrs.y, config.originN, streamptr[0]);
            CopyDataB2HD(tlrmvmcpu->p3transptrs.y, (complex<float>*)cudap3transptrs.y, tlrmvmcpu->config.originM);
        }else{
            ConjugateDriver<cuHalfComplex>(cudap3ptrs.y, config.originM, streamptr[0]);
            CopyDataB2HD(tlrmvmcpu->p3ptrs.y, (complex<float>*)cudap3ptrs.y, tlrmvmcpu->config.originM);
        }
    }

    void TlrmvmMPfp16::StreamInit(int streamsize){
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

    void TlrmvmMPfp16::StreamDestroy(){
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

    void TlrmvmMPfp16::MemoryInit(){
        tlrmvmcpu->MemoryInit();
        PhasePointersCopyNonPointers<complex<float>, cuHalfComplex>(cudap1ptrs, tlrmvmcpu->p1ptrs);
        PhasePointersCopyNonPointers<complex<float>, cuHalfComplex>(cudap3ptrs, tlrmvmcpu->p3ptrs);
        PhasePointersCopyNonPointers<complex<float>, cuHalfComplex>(cudap1transptrs, tlrmvmcpu->p1transptrs);
        PhasePointersCopyNonPointers<complex<float>, cuHalfComplex>(cudap3transptrs, tlrmvmcpu->p3transptrs);
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
    }

    void TlrmvmMPfp16::MemoryFree(){
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

        FreecuHostMemory(cudap1transptrs.Abp);
        FreecuHostMemory(cudap1transptrs.xbp);
        FreecuHostMemory(cudap1transptrs.ybp);
        FreeDeviceMemory(cudap1transptrs.y);

        FreecuHostMemory(cudap3transptrs.Abp);
        FreecuHostMemory(cudap3transptrs.xbp);
        FreecuHostMemory(cudap3transptrs.ybp);
        FreeDeviceMemory(cudap3transptrs.y);
    }

    void TlrmvmMPfp16::Phase1(){
        cudaDeviceSynchronize();
        for(int i=0; i<config.Ntg; i++){
            if(cudap1ptrs.Ms[i] != 0){
                Hgemv_Phase1_driver(cudap1ptrs.Abp[i], cudap1ptrs.xbp[i], cudap1ptrs.ybp[i],
                                    cudap1ptrs.Ms[i], cudap1ptrs.Ks[i], streamptr[i % stream_size]);
            }
        }
        cudaDeviceSynchronize();
    }

    void TlrmvmMPfp16::Phase1Transpose(){
        cudaDeviceSynchronize();
        for(int i=0; i<config.Ntg; i++){
            if(cudap1transptrs.Ms[i] != 0){
//                cublasgemv(cublashandleptr[i%stream_size], CUBLAS_OP_T,
//                           cudap1transptrs.Ks[i], cudap1transptrs.Ms[i],
//                           &alpha, (const cuComplex *)cudap1transptrs.Abp[i],
//                           cudap1transptrs.Ks[i],
//                           (const cuComplex *)cudap1transptrs.xbp[i], 1, &beta,
//                           cudap1transptrs.ybp[i], 1);
            }
        }
        cudaDeviceSynchronize();
    }

    void TlrmvmMPfp16::Phase1GetMembuffer(){
        int batchsize = cudap1ptrs.Ms.size();
        GetcuHostMemory(&cudap1ptrs.Abp, batchsize);
        GetcuHostMemory(&cudap1ptrs.xbp, batchsize);
        GetcuHostMemory(&cudap1ptrs.ybp, batchsize);
    }

    void TlrmvmMPfp16::Phase1GetMembufferTranspose()
    {
        int batchsize = cudap1ptrs.Ms.size();
        GetcuHostMemory(&cudap1transptrs.Abp, batchsize);
        GetcuHostMemory(&cudap1transptrs.xbp, batchsize);
        GetcuHostMemory(&cudap1transptrs.ybp, batchsize);
    }

    void TlrmvmMPfp16::AllocatePhase1Buffer(){
        GetDeviceMemory(&cudap1ptrs.A, cudap1ptrs.Acnt);
        GetDeviceMemory(&cudap1ptrs.x, cudap1ptrs.Xcnt);
        GetDeviceMemory(&cudap1ptrs.y, cudap1ptrs.Ycnt);
        cudap1ptrs.Abp[0] = cudap1ptrs.A;
        cudap1ptrs.xbp[0] = cudap1ptrs.x;
        cudap1ptrs.ybp[0] = cudap1ptrs.y;
    }

    void TlrmvmMPfp16::AllocatePhase1BufferTranspose(){
        cudap1transptrs.A = cudap3ptrs.A;
        cudap1transptrs.x = cudap1ptrs.x;
        GetDeviceMemory(&cudap1transptrs.y, cudap1transptrs.Ycnt);
        cudap1transptrs.Abp[0] = cudap3ptrs.A; // use phase 3, U bases
        cudap1transptrs.xbp[0] = cudap1ptrs.x; // use phase 1, x
        cudap1transptrs.ybp[0] = cudap1transptrs.y; // create a new buffer
    }

    void TlrmvmMPfp16::Phase1CopyData(){
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
        auto Ahalf_h = new cuHalfComplex[tlrmvmcpu->p1ptrs.Acnt];
        auto xhalf_h = new cuHalfComplex[tlrmvmcpu->p1ptrs.Xcnt];
        for(size_t i=0; i<tlrmvmcpu->p1ptrs.Acnt; i++)
            Ahalf_h[i] = cuHalfComplex(tlrmvmcpu->p1ptrs.A[i].real(), tlrmvmcpu->p1ptrs.A[i].imag());
        for(size_t i=0; i<tlrmvmcpu->p1ptrs.Xcnt; i++)
            xhalf_h[i] = cuHalfComplex(tlrmvmcpu->p1ptrs.x[i].real(), tlrmvmcpu->p1ptrs.x[i].imag());
        // load phase1 A,x to GPU
        CopyDataB2HD((cuHalfComplex*)cudap1ptrs.A, Ahalf_h, tlrmvmcpu->p1ptrs.Acnt);
        CopyDataB2HD((cuHalfComplex*)cudap1ptrs.x, xhalf_h, tlrmvmcpu->p1ptrs.Xcnt);
        delete[] Ahalf_h;
        delete[] xhalf_h;
    }

    void TlrmvmMPfp16::Phase1CopyDataTranspose(){
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

    void TlrmvmMPfp16::Phase2(){
        cudaDeviceSynchronize();
        phase2_nosplit<cuHalfComplex>(cudap1ptrs.y, d_phase2mapping, cudap3ptrs.x,
                                   config.workmatgranksum, streamptr[0]);
        cudaDeviceSynchronize();
    }
    void TlrmvmMPfp16::Phase2Transpose(){
        cudaDeviceSynchronize();
        phase2_nosplit<cuHalfComplex>(cudap1transptrs.y, d_phase2mapping_transpose,
                                   cudap3transptrs.x, config.granksum, streamptr[0]);
        cudaDeviceSynchronize();
    }

    void TlrmvmMPfp16::Phase2Prepare(){
        GetDeviceMemory(&d_phase2mapping, tlrmvmcpu->h_phase2mapping.size());
        CopyDataB2HD(d_phase2mapping, tlrmvmcpu->h_phase2mapping.data(),
                     tlrmvmcpu->h_phase2mapping.size());
    }
    void TlrmvmMPfp16::Phase2PrepareTranspose(){
        GetDeviceMemory(&d_phase2mapping_transpose, tlrmvmcpu->h_phase2mappingTranspose.size());
        CopyDataB2HD(d_phase2mapping_transpose, tlrmvmcpu->h_phase2mappingTranspose.data(),
                     tlrmvmcpu->h_phase2mappingTranspose.size());
    }

    void TlrmvmMPfp16::Phase3(){
        cudaDeviceSynchronize();
        for(int i=0; i<config.Mtg; i++){
            Hgemv_Phase3_driver(cudap3ptrs.Abp[i], cudap3ptrs.xbp[i], cudap3ptrs.ybp[i],
                                cudap3ptrs.Ms[i], cudap3ptrs.Ks[i], streamptr[i % stream_size]);
        }
        cudaDeviceSynchronize();
        CUDACHECK(cudaGetLastError());
    }
    void TlrmvmMPfp16::Phase3Transpose(){
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

    void TlrmvmMPfp16::Phase3GetMembuffer(){
        int batchsize = cudap3ptrs.Ms.size();
        GetcuHostMemory(&cudap3ptrs.Abp, batchsize);
        GetcuHostMemory(&cudap3ptrs.xbp, batchsize);
        GetcuHostMemory(&cudap3ptrs.ybp, batchsize);
    }
    void TlrmvmMPfp16::Phase3GetMembufferTranspose(){
        int batchsize = cudap3transptrs.Ms.size();
        GetcuHostMemory(&cudap3transptrs.Abp, batchsize);
        GetcuHostMemory(&cudap3transptrs.xbp, batchsize);
        GetcuHostMemory(&cudap3transptrs.ybp, batchsize);
    }

    void TlrmvmMPfp16::AllocatePhase3Buffer(){
        GetDeviceMemory(&cudap3ptrs.A, cudap3ptrs.Acnt);
        GetDeviceMemory(&cudap3ptrs.x, cudap3ptrs.Xcnt);
        GetDeviceMemory(&cudap3ptrs.y, cudap3ptrs.Ycnt);
        cudap3ptrs.Abp[0] = cudap3ptrs.A;
        cudap3ptrs.xbp[0] = cudap3ptrs.x;
        cudap3ptrs.ybp[0] = cudap3ptrs.y;

    }
    void TlrmvmMPfp16::AllocatePhase3BufferTranspose(){
        cudap3transptrs.A = cudap1ptrs.A;
        cudap3transptrs.x = cudap3ptrs.x;
        GetDeviceMemory(&cudap3transptrs.y, cudap3transptrs.Ycnt);
        cudap3transptrs.Abp[0] = cudap1ptrs.A; // use phase 1, V bases
        cudap3transptrs.xbp[0] = cudap3ptrs.x; // use phase 3, x
        cudap3transptrs.ybp[0] = cudap3transptrs.y; // create a new buffer
    }

    void TlrmvmMPfp16::Phase3CopyData(){
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
        auto Ahalf_h = new cuHalfComplex[cudap3ptrs.Acnt];
        for(size_t i=0; i<cudap3ptrs.Acnt; i++)
            Ahalf_h[i] = cuHalfComplex(tlrmvmcpu->p3ptrs.A[i].real(), tlrmvmcpu->p3ptrs.A[i].imag());
        // load phase 3 A to GPU
        CopyDataB2HD((cuHalfComplex*)cudap3ptrs.A, Ahalf_h, cudap3ptrs.Acnt);
        delete[] Ahalf_h;
    }
    void TlrmvmMPfp16::Phase3CopyDataTranspose(){
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

    void TlrmvmMPfp16::MVMGraphTranspose() {
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

    void TlrmvmMPfp16::MVMGraphNoTranspose() {
        if(!graphCreated){
            cudaStreamBeginCapture(streamptr[0],cudaStreamCaptureModeGlobal);
            cudaEventRecord(event_start, streamptr[0]);
            for(int streami=1; streami<stream_size; streami++){
                cudaStreamWaitEvent(streamptr[streami], event_start);
            }
            // phase 1
            for(int i=0; i<config.Ntg; i++){
                if(cudap1ptrs.Ms[i] != 0){
                    Hgemv_Phase1_driver(cudap1ptrs.Abp[i], cudap1ptrs.xbp[i], cudap1ptrs.ybp[i],
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
                                          config.granksum, streamptr[0]);
            cudaEventRecord(events[0], streamptr[0]);
            for(int streami=1; streami < stream_size; streami++){
                cudaStreamWaitEvent(streamptr[streami], events[0]);
            }
            // phase 3
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

    void TlrmvmMPfp16::MVMGraph(){
        if(transpose){
            MVMGraphTranspose();
        }else{
            MVMGraphNoTranspose();
        }
    }

    void TlrmvmMPfp16::MVMTranspose()
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
    void TlrmvmMPfp16::MVMNoTranspose()
    {
        cudaDeviceSynchronize();
        for(int i=0; i<config.Ntg; i++){
            if(cudap1ptrs.Ms[i] != 0){
                Hgemv_Phase1_driver(cudap1ptrs.Abp[i], cudap1ptrs.xbp[i], cudap1ptrs.ybp[i],
                                    cudap1ptrs.Ms[i], cudap1ptrs.Ks[i], streamptr[i % stream_size]);
            }
        }
        cudaDeviceSynchronize();
        phase2_nosplit<cuHalfComplex>(cudap1ptrs.y, d_phase2mapping, cudap3ptrs.x,
                                  config.granksum, streamptr[0]);
        cudaDeviceSynchronize();
        for(int i=0; i<config.Mtg; i++){
            Hgemv_Phase3_driver(cudap3ptrs.Abp[i], cudap3ptrs.xbp[i], cudap3ptrs.ybp[i],
                                cudap3ptrs.Ms[i], cudap3ptrs.Ks[i], streamptr[i % stream_size]);
        }
        cudaDeviceSynchronize();
    }

    void TlrmvmMPfp16::CopyBackResults()
    {
        // use cpu pointers to send output
        if(transpose){
            CopyDataB2HD(tlrmvmcpu->p1transptrs.y, (complex<float>*)cudap1transptrs.y, tlrmvmcpu->config.granksum);
            CopyDataB2HD(tlrmvmcpu->p3transptrs.x, (complex<float>*)cudap3transptrs.x, tlrmvmcpu->config.granksum);
            CopyDataB2HD(tlrmvmcpu->p3transptrs.y, (complex<float>*)cudap3transptrs.y, tlrmvmcpu->config.originM);
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

    void TlrmvmMPfp16::MVM() {
        if(transpose){
            MVMTranspose();
        }else{
            MVMNoTranspose();
        }
    }

    void TlrmvmMPfp16::SetTransposeConjugate(bool transpose, bool conjugate) {
        this->transpose = transpose;
        this->conjugate = conjugate;
        tlrmvmcpu->SetTransposeConjugate(transpose, conjugate);
    }

} // namespace cudatlrmvm

