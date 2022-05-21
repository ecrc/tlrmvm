#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h> // if you need CUBLAS v2, include before magma.h
#include <cstdint>
#include "../../common/Common.hpp"
#include "../../common/AppUtil.hpp"
#include "../cpu/TlrmvmCPU.hpp"

#include "../../common/cuda/Util.hpp"
#include "Tlrmvmcuda.hpp"
#include "cudakernel.cuh"

namespace cudatlrmvm
{

    template<typename T>
    CUDAPhasePointers<T>::CUDAPhasePointers(){}

    template struct CUDAPhasePointers<float>;
    template struct CUDAPhasePointers<double>;
    template struct CUDAPhasePointers<cuComplex>;
    template struct CUDAPhasePointers<cuDoubleComplex>;
    template struct CUDAPhasePointers<cuHalfComplex>;
    template struct CUDAPhasePointers<cubfComplex>;
    template struct CUDAPhasePointers<int8_t>;

    template<typename SrcType, typename DestType>
    void PhasePointersCopyNonPointers(CUDAPhasePointers<DestType> &dest, const PhasePointers<SrcType> &src){
        dest.Acnt = src.Acnt;
        dest.Xcnt = src.Xcnt;
        dest.Ycnt = src.Ycnt;
        dest.Ms = src.Ms;
        dest.Ks = src.Ks;
        dest.Ns = src.Ns;
    }

    template void PhasePointersCopyNonPointers<float,float>(CUDAPhasePointers<float> &,
            const PhasePointers<float>&);
    template void PhasePointersCopyNonPointers<double,double>(CUDAPhasePointers<double> &,
            const PhasePointers<double>&);
    template void PhasePointersCopyNonPointers<complex<float>,cuComplex>
            (CUDAPhasePointers<cuComplex> &, const PhasePointers<complex<float>>&);
    template void PhasePointersCopyNonPointers<complex<double>,cuDoubleComplex>
            (CUDAPhasePointers<cuDoubleComplex> &, const PhasePointers<complex<double>>&);
    template void PhasePointersCopyNonPointers<complex<float>,cuHalfComplex>
            (CUDAPhasePointers<cuHalfComplex> &, const PhasePointers<complex<float>>&);
    template void PhasePointersCopyNonPointers<complex<float>,cubfComplex>
            (CUDAPhasePointers<cubfComplex> &, const PhasePointers<complex<float>>&);

    void I8PhasePointersCopyNonPointers(CUDAI8basesPointers& dest, const PhasePointers<complex<float>>& src){
        dest.Acnt = src.Acnt;
        dest.Xcnt = src.Xcnt;
        dest.Ycnt = src.Ycnt;
        dest.Ms = src.Ms;
        dest.Ks = src.Ks;
        dest.Ns = src.Ns;
    }

    template<typename HostType, typename DeviceType>
    Tlrmvmcuda<HostType, DeviceType>::Tlrmvmcuda() {}

    template<typename HostType, typename DeviceType>
    Tlrmvmcuda<HostType, DeviceType>::Tlrmvmcuda(TlrmvmConfig tlrmvmconfig)
    :config(tlrmvmconfig)
    {
        transpose = false;
        conjugate = false;
        init_alpha_beta(alpha, beta);
        tlrmvmcpu = std::make_shared<TlrmvmCPU<HostType>>(tlrmvmconfig);
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType, DeviceType>::UpdateConfig(TlrmvmConfig &tlrmvmconfig)
    {
//        transpose = false;
//        conjugate = false;
//        init_alpha_beta(alpha, beta);
//        tlrmvmcpu->UpdateConfig(tlrmvmconfig);
        cout << "UpdateConfig not implemented." << endl;
        exit(0);
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::setX(HostType * xvector, size_t xlength){
        tlrmvmcpu->setX(xvector, xlength);
        TryConjugateXvec();
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::TryConjugateXvec() {
        // no transpose logic
        tlrmvmcpu->TryConjugateXvec();
        CopyDataB2HD((HostType*)this->cudap1ptrs.x, tlrmvmcpu->p1ptrs.x, tlrmvmcpu->xmat.Shape()[0]);
    }
    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::TryConjugateResults() {
        if(!conjugate) return;
        if(transpose){
            ConjugateDriver<DeviceType>(cudap3transptrs.y, config.originN, streamptr[0]);
            CopyDataB2HD(tlrmvmcpu->p3transptrs.y, (HostType*)cudap3transptrs.y, tlrmvmcpu->config.originM);
        }else{
            ConjugateDriver<DeviceType>(cudap3ptrs.y, config.originM, streamptr[0]);
            CopyDataB2HD(tlrmvmcpu->p3ptrs.y, (HostType*)cudap3ptrs.y, tlrmvmcpu->config.originM);
        }
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::StreamInit(int streamsize){
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

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::StreamDestroy(){
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

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::MemoryInit(){
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
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::MemoryFree(){
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

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::Phase1(){
        cudaDeviceSynchronize();
        for(int i=0; i<config.Ntg; i++){
            if(cudap1ptrs.Ms[i] != 0){
                cublasgemv(cublashandleptr[i%stream_size], CUBLAS_OP_N,
                cudap1ptrs.Ms[i], cudap1ptrs.Ks[i],
                &alpha, (const DeviceType*)cudap1ptrs.Abp[i], cudap1ptrs.Ms[i],
                (const DeviceType*)cudap1ptrs.xbp[i], 1, &beta,
                cudap1ptrs.ybp[i], 1);
            }
        }
        cudaDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::Phase1Transpose(){
        cudaDeviceSynchronize();
        for(int i=0; i<config.Ntg; i++){
            if(cudap1transptrs.Ms[i] != 0){
                cublasgemv(cublashandleptr[i%stream_size], CUBLAS_OP_T,
                           cudap1transptrs.Ks[i], cudap1transptrs.Ms[i],
                           &alpha, (const DeviceType*)cudap1transptrs.Abp[i],
                           cudap1transptrs.Ks[i],
                           (const DeviceType*)cudap1transptrs.xbp[i], 1, &beta,
                           cudap1transptrs.ybp[i], 1);
            }
        }
        cudaDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::Phase1GetMembuffer(){
        int batchsize = cudap1ptrs.Ms.size();
        GetcuHostMemory(&cudap1ptrs.Abp, batchsize);
        GetcuHostMemory(&cudap1ptrs.xbp, batchsize);
        GetcuHostMemory(&cudap1ptrs.ybp, batchsize);
    }
    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::Phase1GetMembufferTranspose()
    {
        int batchsize = cudap1ptrs.Ms.size();
        GetcuHostMemory(&cudap1transptrs.Abp, batchsize);
        GetcuHostMemory(&cudap1transptrs.xbp, batchsize);
        GetcuHostMemory(&cudap1transptrs.ybp, batchsize);
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::AllocatePhase1Buffer(){
        GetDeviceMemory(&cudap1ptrs.A, cudap1ptrs.Acnt);
        GetDeviceMemory(&cudap1ptrs.x, cudap1ptrs.Xcnt);
        GetDeviceMemory(&cudap1ptrs.y, cudap1ptrs.Ycnt);
        cudap1ptrs.Abp[0] = cudap1ptrs.A;
        cudap1ptrs.xbp[0] = cudap1ptrs.x;
        cudap1ptrs.ybp[0] = cudap1ptrs.y;
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::AllocatePhase1BufferTranspose(){
        cudap1transptrs.A = cudap3ptrs.A;
        cudap1transptrs.x = cudap1ptrs.x;
        GetDeviceMemory(&cudap1transptrs.y, cudap1transptrs.Ycnt);
        cudap1transptrs.Abp[0] = cudap3ptrs.A; // use phase 3, U bases
        cudap1transptrs.xbp[0] = cudap1ptrs.x; // use phase 1, x
        cudap1transptrs.ybp[0] = cudap1transptrs.y; // create a new buffer
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::Phase1CopyData(){
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
    void Tlrmvmcuda<HostType,DeviceType>::Phase1CopyDataTranspose(){
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
    void Tlrmvmcuda<HostType,DeviceType>::Phase2(){
        cudaDeviceSynchronize();
        phase2_nosplit<DeviceType>(cudap1ptrs.y, d_phase2mapping, cudap3ptrs.x,
                                   config.granksum, streamptr[0]);
        cudaDeviceSynchronize();
    }
    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::Phase2Transpose(){
        cudaDeviceSynchronize();
        phase2_nosplit<DeviceType>(cudap1transptrs.y, d_phase2mapping_transpose,
                                   cudap3transptrs.x, config.granksum, streamptr[0]);
        cudaDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::Phase2Prepare(){
        GetDeviceMemory(&d_phase2mapping, tlrmvmcpu->h_phase2mapping.size());
        CopyDataB2HD(d_phase2mapping, tlrmvmcpu->h_phase2mapping.data(),tlrmvmcpu->h_phase2mapping.size());
    }
    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::Phase2PrepareTranspose(){
        GetDeviceMemory(&d_phase2mapping_transpose, tlrmvmcpu->h_phase2mappingTranspose.size());
        CopyDataB2HD(d_phase2mapping_transpose, tlrmvmcpu->h_phase2mappingTranspose.data(),
                     tlrmvmcpu->h_phase2mappingTranspose.size());
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::Phase3(){
        cudaDeviceSynchronize();
        for(int i=0; i<config.Mtg; i++){
            cublasgemv(cublashandleptr[i%stream_size], CUBLAS_OP_N,
            cudap3ptrs.Ms[i], cudap3ptrs.Ks[i],
            &alpha, (const DeviceType*)cudap3ptrs.Abp[i], cudap3ptrs.Ms[i],
            (const DeviceType*)cudap3ptrs.xbp[i], 1, &beta,cudap3ptrs.ybp[i], 1);
        }
        cudaDeviceSynchronize();
        CUDACHECK(cudaGetLastError());
    }
    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::Phase3Transpose(){
        cudaDeviceSynchronize();
        for(int i=0; i<config.Mtg; i++){
            cublasgemv(cublashandleptr[i%stream_size], CUBLAS_OP_T,
                       cudap3transptrs.Ks[i], cudap3transptrs.Ms[i],
                       &alpha, (const DeviceType*)cudap3transptrs.Abp[i],
                       cudap3transptrs.Ks[i],
                       (const DeviceType*)cudap3transptrs.xbp[i], 1, &beta,cudap3transptrs.ybp[i], 1);
        }
        cudaDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::Phase3GetMembuffer(){
        int batchsize = cudap3ptrs.Ms.size();
        GetcuHostMemory(&cudap3ptrs.Abp, batchsize);
        GetcuHostMemory(&cudap3ptrs.xbp, batchsize);
        GetcuHostMemory(&cudap3ptrs.ybp, batchsize);
    }
    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::Phase3GetMembufferTranspose(){
        int batchsize = cudap3transptrs.Ms.size();
        GetcuHostMemory(&cudap3transptrs.Abp, batchsize);
        GetcuHostMemory(&cudap3transptrs.xbp, batchsize);
        GetcuHostMemory(&cudap3transptrs.ybp, batchsize);
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::AllocatePhase3Buffer(){
        GetDeviceMemory(&cudap3ptrs.A, cudap3ptrs.Acnt);
        GetDeviceMemory(&cudap3ptrs.x, cudap3ptrs.Xcnt);
        GetDeviceMemory(&cudap3ptrs.y, cudap3ptrs.Ycnt);
        cudap3ptrs.Abp[0] = cudap3ptrs.A;
        cudap3ptrs.xbp[0] = cudap3ptrs.x;
        cudap3ptrs.ybp[0] = cudap3ptrs.y;
    }
    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::AllocatePhase3BufferTranspose(){
        cudap3transptrs.A = cudap1ptrs.A;
        cudap3transptrs.x = cudap3ptrs.x;
        GetDeviceMemory(&cudap3transptrs.y, cudap3transptrs.Ycnt);
        cudap3transptrs.Abp[0] = cudap1ptrs.A; // use phase 1, V bases
        cudap3transptrs.xbp[0] = cudap3ptrs.x; // use phase 3, x
        cudap3transptrs.ybp[0] = cudap3transptrs.y; // create a new buffer
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::Phase3CopyData(){
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
    void Tlrmvmcuda<HostType,DeviceType>::Phase3CopyDataTranspose(){
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
    void Tlrmvmcuda<HostType, DeviceType>::MVMGraphTranspose() {
        if(!graphCreated){
            cudaStreamBeginCapture(streamptr[0],cudaStreamCaptureModeGlobal);
            cudaEventRecord(event_start, streamptr[0]);
            for(int streami=1; streami<stream_size; streami++){
                cudaStreamWaitEvent(streamptr[streami], event_start);
            }
            // phase 1 transpose
            for(int i=0; i<config.Ntg; i++){
                if(cudap1transptrs.Ms[i] != 0){
                    cublasgemv(cublashandleptr[i%stream_size], CUBLAS_OP_T,
                               cudap1transptrs.Ks[i], cudap1transptrs.Ms[i],
                               &alpha, (const DeviceType*)cudap1transptrs.Abp[i],
                               cudap1transptrs.Ks[i],
                               (const DeviceType*)cudap1transptrs.xbp[i], 1, &beta,
                               cudap1transptrs.ybp[i], 1);
                }
            }
            for(int streami=1; streami < stream_size; streami++){
                cudaEventRecord(events[streami], streamptr[streami]);
            }
            for(int streami=1; streami < stream_size; streami++){
                cudaStreamWaitEvent(streamptr[0], events[streami]);
            }
            // phase 2 transpose
            phase2_nosplit<DeviceType>(cudap1transptrs.y, d_phase2mapping_transpose,
                                       cudap3transptrs.x, config.granksum, streamptr[0]);
            cudaEventRecord(events[0], streamptr[0]);
            for(int streami=1; streami < stream_size; streami++){
                cudaStreamWaitEvent(streamptr[streami], events[0]);
            }
            // phase 3 transpose
            for(int i=0; i<config.Mtg; i++){
                cublasgemv(cublashandleptr[i%stream_size], CUBLAS_OP_T,
                           cudap3transptrs.Ks[i], cudap3transptrs.Ms[i],
                           &alpha, (const DeviceType*)cudap3transptrs.Abp[i],
                           cudap3transptrs.Ks[i],
                           (const DeviceType*)cudap3transptrs.xbp[i], 1, &beta,cudap3transptrs.ybp[i], 1);
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

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType, DeviceType>::MVMGraphNoTranspose() {
        if(!graphCreated){
            cudaStreamBeginCapture(streamptr[0],cudaStreamCaptureModeGlobal);
            cudaEventRecord(event_start, streamptr[0]);
            for(int streami=1; streami<stream_size; streami++){
                cudaStreamWaitEvent(streamptr[streami], event_start);
            }
            // phase 1
            for(int i=0; i<config.Ntg; i++){
                if(cudap1ptrs.Ms[i] != 0){
                    cublasgemv(cublashandleptr[i%stream_size], CUBLAS_OP_N,
                               cudap1ptrs.Ms[i], cudap1ptrs.Ks[i],
                               &alpha, (const DeviceType*)cudap1ptrs.Abp[i], cudap1ptrs.Ms[i],
                               (const DeviceType*)cudap1ptrs.xbp[i], 1, &beta,
                               cudap1ptrs.ybp[i], 1);
                }
            }
            for(int streami=1; streami < stream_size; streami++){
                cudaEventRecord(events[streami], streamptr[streami]);
            }
            for(int streami=1; streami < stream_size; streami++){
                cudaStreamWaitEvent(streamptr[0], events[streami]);
            }
            // phase 2
            phase2_nosplit<DeviceType>(cudap1ptrs.y, d_phase2mapping, cudap3ptrs.x,
                                       config.granksum, streamptr[0]);
            cudaEventRecord(events[0], streamptr[0]);
            for(int streami=1; streami < stream_size; streami++){
                cudaStreamWaitEvent(streamptr[streami], events[0]);
            }
            // phase 3
            for(int i=0; i<config.Mtg; i++){
                cublasgemv(cublashandleptr[i%stream_size], CUBLAS_OP_N,
                           cudap3ptrs.Ms[i], cudap3ptrs.Ks[i],
                           &alpha, (const DeviceType*)cudap3ptrs.Abp[i], cudap3ptrs.Ms[i],
                           (const DeviceType*)cudap3ptrs.xbp[i], 1, &beta,cudap3ptrs.ybp[i], 1);
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

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::MVMGraph(){
        if(transpose){
            MVMGraphTranspose();
        }else{
            MVMGraphNoTranspose();
        }
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::MVMTranspose()
    {
        cudaDeviceSynchronize();
        for(int i=0; i<config.Ntg; i++){
            if(cudap1transptrs.Ms[i] != 0){
                cublasgemv(cublashandleptr[i%stream_size], CUBLAS_OP_T,
                           cudap1transptrs.Ks[i], cudap1transptrs.Ms[i],
                           &alpha, (const DeviceType*)cudap1transptrs.Abp[i],
                           cudap1transptrs.Ks[i],
                           (const DeviceType*)cudap1transptrs.xbp[i], 1, &beta,
                           cudap1transptrs.ybp[i], 1);
            }
        }
        cudaDeviceSynchronize();
        phase2_nosplit<DeviceType>(cudap1transptrs.y, d_phase2mapping_transpose,
                                   cudap3transptrs.x, config.granksum, streamptr[0]);
        cudaDeviceSynchronize();
        for(int i=0; i<config.Mtg; i++){
            cublasgemv(cublashandleptr[i%stream_size], CUBLAS_OP_T,
                       cudap3transptrs.Ks[i], cudap3transptrs.Ms[i],
                       &alpha, (const DeviceType*)cudap3transptrs.Abp[i],
                       cudap3transptrs.Ks[i],
                       (const DeviceType*)cudap3transptrs.xbp[i], 1, &beta,cudap3transptrs.ybp[i], 1);
        }
        cudaDeviceSynchronize();
    }
    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType,DeviceType>::MVMNoTranspose()
    {
        cudaDeviceSynchronize();
        for(int i=0; i<config.Ntg; i++){
            if(cudap1ptrs.Ms[i] != 0){
                cublasgemv(cublashandleptr[i%stream_size], CUBLAS_OP_N,
                           cudap1ptrs.Ms[i], cudap1ptrs.Ks[i],
                           &alpha, (const DeviceType*)cudap1ptrs.Abp[i], cudap1ptrs.Ms[i],
                           (const DeviceType*)cudap1ptrs.xbp[i], 1, &beta,
                           cudap1ptrs.ybp[i], 1);
            }
        }
        cudaDeviceSynchronize();
        phase2_nosplit<DeviceType>(cudap1ptrs.y, d_phase2mapping, cudap3ptrs.x,
                                   config.granksum, streamptr[0]);
        cudaDeviceSynchronize();
        for(int i=0; i<config.Mtg; i++){
            cublasgemv(cublashandleptr[i%stream_size], CUBLAS_OP_N,
                       cudap3ptrs.Ms[i], cudap3ptrs.Ks[i],
                       &alpha, (const DeviceType*)cudap3ptrs.Abp[i], cudap3ptrs.Ms[i],
                       (const DeviceType*)cudap3ptrs.xbp[i], 1, &beta,cudap3ptrs.ybp[i], 1);
        }
        cudaDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType, DeviceType>::CopyBackResults()
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
    void Tlrmvmcuda<HostType, DeviceType>::MVM() {
        if(transpose){
            MVMTranspose();
        }else{
            MVMNoTranspose();
        }
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmcuda<HostType, DeviceType>::SetTransposeConjugate(bool transpose, bool conjugate) {
        this->transpose = transpose;
        this->conjugate = conjugate;
        tlrmvmcpu->SetTransposeConjugate(transpose, conjugate);
    }



    template class Tlrmvmcuda<float, float>;
    template class Tlrmvmcuda<double, double>;
    template class Tlrmvmcuda<complex<float>, cuComplex>;
    template class Tlrmvmcuda<complex<double>, cuDoubleComplex>;

} // namespace cudatlrmvm

