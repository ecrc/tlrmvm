#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas.h>

#include "../../common/Common.hpp"
#include "../../common/AppUtil.hpp"
#include "../cpu/TlrmvmCPU.hpp"
#include "Tlrmvmhip.hpp"
#include "hipkernel.cuh"

namespace hiptlrmvm
{

    template<typename T>
    HIPPhasePointers<T>::HIPPhasePointers(){}

    template struct HIPPhasePointers<float>;
    template struct HIPPhasePointers<double>;
    template struct HIPPhasePointers<hipComplex>;
    template struct HIPPhasePointers<hipDoubleComplex>;

    template<typename SrcType, typename DestType>
    void PhasePointersCopyNonPointers(HIPPhasePointers<DestType> &dest, const PhasePointers<SrcType> &src){
        dest.Acnt = src.Acnt;
        dest.Xcnt = src.Xcnt;
        dest.Ycnt = src.Ycnt;
        dest.Ms = src.Ms;
        dest.Ks = src.Ks;
        dest.Ns = src.Ns;
    }

    template void PhasePointersCopyNonPointers<float,float>(HIPPhasePointers<float> &,
                                                            const PhasePointers<float>&);
    template void PhasePointersCopyNonPointers<double,double>(HIPPhasePointers<double> &,
                                                              const PhasePointers<double>&);
    template void PhasePointersCopyNonPointers<complex<float>,hipComplex>
            (HIPPhasePointers<hipComplex> &, const PhasePointers<complex<float>>&);
    template void PhasePointersCopyNonPointers<complex<double>,hipDoubleComplex>
            (HIPPhasePointers<hipDoubleComplex> &, const PhasePointers<complex<double>>&);

    template<typename HostType, typename DeviceType>
    Tlrmvmhip<HostType, DeviceType>::Tlrmvmhip() {}

    template<typename HostType, typename DeviceType>
    Tlrmvmhip<HostType, DeviceType>::Tlrmvmhip(TlrmvmConfig tlrmvmconfig)
            :config(tlrmvmconfig)
    {
        transpose = false;
        conjugate = false;
        init_alpha_beta(alpha, beta);
        tlrmvmcpu = std::make_shared<TlrmvmCPU<HostType>>(tlrmvmconfig);
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType, DeviceType>::UpdateConfig(TlrmvmConfig &tlrmvmconfig)
    {
//        transpose = false;
//        conjugate = false;
//        init_alpha_beta(alpha, beta);
//        tlrmvmcpu->UpdateConfig(tlrmvmconfig);
        cout << "UpdateConfig not implemented." << endl;
        exit(0);
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::setX(HostType * xvector, size_t xlength){
        tlrmvmcpu->setX(xvector, xlength);
        tlrmvmcpu->TryConjugateXvec();
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::TryConjugateXvec() {
        // no transpose logic
        tlrmvmcpu->TryConjugateXvec();
        CopyDataB2HD((HostType*)this->cudap1ptrs.x, tlrmvmcpu->p1ptrs.x, tlrmvmcpu->xmat.Shape()[0]);
    }
    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::TryConjugateResults() {
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
    void Tlrmvmhip<HostType,DeviceType>::StreamInit(int streamsize){
        this->stream_size = streamsize;
        streamptr = new hipStream_t[streamsize];
        cublashandleptr = new hipblasHandle_t[streamsize];
        for(int i=0; i<streamsize; i++)
            hipStreamCreateWithFlags(&streamptr[i], hipStreamNonBlocking);
        for(int i=0; i<streamsize; i++)
            hipblasCreate(&cublashandleptr[i]);
        for(int i=0; i<streamsize; i++)
            hipblasSetStream(cublashandleptr[i], streamptr[i]);
        HIPCHECK(hipEventCreate(&event_start));
        HIPCHECK(hipEventCreate(&event_phase2finish));
        events = new hipEvent_t[2*streamsize];
        for(int i=0; i<2*streamsize; i++) HIPCHECK(hipEventCreate(&events[i]));
        // graph
        graphCreated = false;
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::StreamDestroy(){
        for(int i=0; i<stream_size; i++) hipblasDestroy(cublashandleptr[i]);
        for(int i=0; i<stream_size; i++) hipStreamDestroy(streamptr[i]);
        delete[] cublashandleptr;
        delete[] streamptr;
        HIPCHECK(hipEventDestroy(event_start));
        HIPCHECK(hipEventDestroy(event_phase2finish));
        for(int i=0; i<2*stream_size; i++) HIPCHECK(hipEventDestroy(events[i]));
        delete[] events;
        // graph
        if(graphCreated){
            hipGraphExecDestroy(instance);
            hipGraphDestroy(graph);
        }
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::MemoryInit(){
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
    void Tlrmvmhip<HostType,DeviceType>::MemoryFree(){
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
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::Phase1(){
        hipDeviceSynchronize();
        for(int i=0; i<config.Ntg; i++){
            if(cudap1ptrs.Ms[i] != 0){
                hipblasgemv(cublashandleptr[i%stream_size], HIPBLAS_OP_N,
                           cudap1ptrs.Ms[i], cudap1ptrs.Ks[i],
                           &alpha, (const DeviceType*)cudap1ptrs.Abp[i], cudap1ptrs.Ms[i],
                           (const DeviceType*)cudap1ptrs.xbp[i], 1, &beta,
                           cudap1ptrs.ybp[i], 1);
            }
        }
        hipDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::Phase1Transpose(){
        hipDeviceSynchronize();
        for(int i=0; i<config.Ntg; i++){
            if(cudap1transptrs.Ms[i] != 0){
                hipblasgemv(cublashandleptr[i%stream_size], HIPBLAS_OP_T,
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
    void Tlrmvmhip<HostType,DeviceType>::Phase1GetMembuffer(){
        int batchsize = cudap1ptrs.Ms.size();
        GethipHostMemory(&cudap1ptrs.Abp, batchsize);
        GethipHostMemory(&cudap1ptrs.xbp, batchsize);
        GethipHostMemory(&cudap1ptrs.ybp, batchsize);
    }
    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::Phase1GetMembufferTranspose()
    {
        int batchsize = cudap1ptrs.Ms.size();
        GethipHostMemory(&cudap1transptrs.Abp, batchsize);
        GethipHostMemory(&cudap1transptrs.xbp, batchsize);
        GethipHostMemory(&cudap1transptrs.ybp, batchsize);
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::AllocatePhase1Buffer(){
        GetDeviceMemory(&cudap1ptrs.A, cudap1ptrs.Acnt);
        GetDeviceMemory(&cudap1ptrs.x, cudap1ptrs.Xcnt);
        GetDeviceMemory(&cudap1ptrs.y, cudap1ptrs.Ycnt);
        cudap1ptrs.Abp[0] = cudap1ptrs.A;
        cudap1ptrs.xbp[0] = cudap1ptrs.x;
        cudap1ptrs.ybp[0] = cudap1ptrs.y;
    }
    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::AllocatePhase1BufferTranspose(){
        cudap1transptrs.A = cudap3ptrs.A;
        cudap1transptrs.x = cudap1ptrs.x;
        GetDeviceMemory(&cudap1transptrs.y, cudap1transptrs.Ycnt);
        cudap1transptrs.Abp[0] = cudap3ptrs.A; // use phase 3, U bases
        cudap1transptrs.xbp[0] = cudap1ptrs.x; // use phase 1, x
        cudap1transptrs.ybp[0] = cudap1transptrs.y; // create a new buffer
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::Phase1CopyData(){
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
    void Tlrmvmhip<HostType,DeviceType>::Phase1CopyDataTranspose(){
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
    void Tlrmvmhip<HostType,DeviceType>::Phase2(){
        hipDeviceSynchronize();
        phase2_nosplit<DeviceType>(cudap1ptrs.y, d_phase2mapping, cudap3ptrs.x,
                                   config.granksum, streamptr[0]);
        hipDeviceSynchronize();
    }
    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::Phase2Transpose(){
        hipDeviceSynchronize();
        phase2_nosplit<DeviceType>(cudap1transptrs.y, d_phase2mapping_transpose,
                                   cudap3transptrs.x, config.granksum, streamptr[0]);
        hipDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::Phase2Prepare(){
        GetDeviceMemory(&d_phase2mapping, tlrmvmcpu->h_phase2mapping.size());
        CopyDataB2HD(d_phase2mapping, tlrmvmcpu->h_phase2mapping.data(),tlrmvmcpu->h_phase2mapping.size());
    }
    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::Phase2PrepareTranspose(){
        GetDeviceMemory(&d_phase2mapping_transpose, tlrmvmcpu->h_phase2mappingTranspose.size());
        CopyDataB2HD(d_phase2mapping_transpose, tlrmvmcpu->h_phase2mappingTranspose.data(),
                     tlrmvmcpu->h_phase2mappingTranspose.size());
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::Phase3(){
        hipDeviceSynchronize();
        for(int i=0; i<config.Mtg; i++){
            hipblasgemv(cublashandleptr[i%stream_size], HIPBLAS_OP_N,
                       cudap3ptrs.Ms[i], cudap3ptrs.Ks[i],
                       &alpha, (const DeviceType*)cudap3ptrs.Abp[i], cudap3ptrs.Ms[i],
                       (const DeviceType*)cudap3ptrs.xbp[i], 1, &beta,cudap3ptrs.ybp[i], 1);
        }
        hipDeviceSynchronize();
    }
    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::Phase3Transpose(){
        hipDeviceSynchronize();
        for(int i=0; i<config.Mtg; i++){
            hipblasgemv(cublashandleptr[i%stream_size], HIPBLAS_OP_T,
                       cudap3transptrs.Ks[i], cudap3transptrs.Ms[i],
                       &alpha, (const DeviceType*)cudap3transptrs.Abp[i],
                       cudap3transptrs.Ks[i],
                       (const DeviceType*)cudap3transptrs.xbp[i], 1, &beta,cudap3transptrs.ybp[i], 1);
        }
        hipDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::Phase3GetMembuffer(){
        int batchsize = cudap3ptrs.Ms.size();
        GethipHostMemory(&cudap3ptrs.Abp, batchsize);
        GethipHostMemory(&cudap3ptrs.xbp, batchsize);
        GethipHostMemory(&cudap3ptrs.ybp, batchsize);
    }
    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::Phase3GetMembufferTranspose(){
        int batchsize = cudap3transptrs.Ms.size();
        GethipHostMemory(&cudap3transptrs.Abp, batchsize);
        GethipHostMemory(&cudap3transptrs.xbp, batchsize);
        GethipHostMemory(&cudap3transptrs.ybp, batchsize);
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::AllocatePhase3Buffer(){
        GetDeviceMemory(&cudap3ptrs.A, cudap3ptrs.Acnt);
        GetDeviceMemory(&cudap3ptrs.x, cudap3ptrs.Xcnt);
        GetDeviceMemory(&cudap3ptrs.y, cudap3ptrs.Ycnt);
        cudap3ptrs.Abp[0] = cudap3ptrs.A;
        cudap3ptrs.xbp[0] = cudap3ptrs.x;
        cudap3ptrs.ybp[0] = cudap3ptrs.y;
    }
    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::AllocatePhase3BufferTranspose(){
        cudap3transptrs.A = cudap1ptrs.A;
        cudap3transptrs.x = cudap3ptrs.x;
        GetDeviceMemory(&cudap3transptrs.y, cudap3transptrs.Ycnt);
        cudap3transptrs.Abp[0] = cudap1ptrs.A; // use phase 1, V bases
        cudap3transptrs.xbp[0] = cudap3ptrs.x; // use phase 3, x
        cudap3transptrs.ybp[0] = cudap3transptrs.y; // create a new buffer
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::Phase3CopyData(){
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
    void Tlrmvmhip<HostType,DeviceType>::Phase3CopyDataTranspose(){
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
    void Tlrmvmhip<HostType, DeviceType>::MVMGraphTranspose() {
        if(!graphCreated){
            hipStreamBeginCapture(streamptr[0],hipStreamCaptureModeGlobal);
            hipEventRecord(event_start, streamptr[0]);
            for(int streami=1; streami<stream_size; streami++){
                hipStreamWaitEvent(streamptr[streami], event_start,0);
            }
            // phase 1 transpose
            for(int i=0; i<config.Ntg; i++){
                if(cudap1transptrs.Ms[i] != 0){
                    hipblasgemv(cublashandleptr[i%stream_size], HIPBLAS_OP_T,
                               cudap1transptrs.Ks[i], cudap1transptrs.Ms[i],
                               &alpha, (const DeviceType*)cudap1transptrs.Abp[i],
                               cudap1transptrs.Ks[i],
                               (const DeviceType*)cudap1transptrs.xbp[i], 1, &beta,
                               cudap1transptrs.ybp[i], 1);
                }
            }
            for(int streami=1; streami < stream_size; streami++){
                hipEventRecord(events[streami], streamptr[streami]);
            }
            for(int streami=1; streami < stream_size; streami++){
                hipStreamWaitEvent(streamptr[0], events[streami],0);
            }
            // phase 2 transpose
            phase2_nosplit<DeviceType>(cudap1transptrs.y, d_phase2mapping_transpose,
                                       cudap3transptrs.x, config.granksum, streamptr[0]);
            hipEventRecord(events[0], streamptr[0]);
            for(int streami=1; streami < stream_size; streami++){
                hipStreamWaitEvent(streamptr[streami], events[0],0);
            }
            // phase 3 transpose
            for(int i=0; i<config.Mtg; i++){
                hipblasgemv(cublashandleptr[i%stream_size], HIPBLAS_OP_T,
                           cudap3transptrs.Ks[i], cudap3transptrs.Ms[i],
                           &alpha, (const DeviceType*)cudap3transptrs.Abp[i],
                           cudap3transptrs.Ks[i],
                           (const DeviceType*)cudap3transptrs.xbp[i], 1, &beta,cudap3transptrs.ybp[i], 1);
            }
            // final merge
            for(int streami=1; streami < stream_size; streami++){
                hipEventRecord(events[stream_size + streami], streamptr[streami]);
            }
            for(int streami=1; streami < stream_size; streami++){
                hipStreamWaitEvent(streamptr[0], events[stream_size + streami],0);
            }
            hipStreamEndCapture(streamptr[0], &graph);
            hipGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
            graphCreated = true;
        }
        hipGraphLaunch(instance, streamptr[0]);
        hipStreamSynchronize(streamptr[0]);
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType, DeviceType>::MVMGraphNoTranspose() {
        if(!graphCreated){
            hipStreamBeginCapture(streamptr[0],hipStreamCaptureModeGlobal);
            hipEventRecord(event_start, streamptr[0]);
            for(int streami=1; streami<stream_size; streami++){
                hipStreamWaitEvent(streamptr[streami], event_start,0);
            }
            // phase 1
            for(int i=0; i<config.Ntg; i++){
                if(cudap1ptrs.Ms[i] != 0){
                    hipblasgemm(cublashandleptr[i%stream_size],HIPBLAS_OP_N, HIPBLAS_OP_N,
                                cudap1ptrs.Ms[i], cudap1ptrs.Ns[i], cudap1ptrs.Ks[i],
                                &alpha, (const DeviceType*)cudap1ptrs.Abp[i], cudap1ptrs.Ms[i],
                                (const DeviceType*)cudap1ptrs.xbp[i], cudap1ptrs.Ks[i],
                                &beta, cudap1ptrs.ybp[i], cudap1ptrs.Ms[i]);
//                    hipblasgemv(cublashandleptr[i%stream_size], HIPBLAS_OP_N,
//                               cudap1ptrs.Ms[i], cudap1ptrs.Ks[i],
//                               &alpha, (const DeviceType*)cudap1ptrs.Abp[i], cudap1ptrs.Ms[i],
//                               (const DeviceType*)cudap1ptrs.xbp[i], 1, &beta,
//                               cudap1ptrs.ybp[i], 1);
                }
            }
            for(int streami=1; streami < stream_size; streami++){
                hipEventRecord(events[streami], streamptr[streami]);
            }
            for(int streami=1; streami < stream_size; streami++){
                hipStreamWaitEvent(streamptr[0], events[streami],0);
            }
            // phase 2
            phase2_nosplit<DeviceType>(cudap1ptrs.y, d_phase2mapping, cudap3ptrs.x,
                                       config.granksum, streamptr[0]);
            hipEventRecord(events[0], streamptr[0]);
            for(int streami=1; streami < stream_size; streami++){
                hipStreamWaitEvent(streamptr[streami], events[0],0);
            }
            // phase 3
            for(int i=0; i<config.Mtg; i++){
                hipblasgemm(cublashandleptr[i%stream_size],HIPBLAS_OP_N, HIPBLAS_OP_N,
                            cudap3ptrs.Ms[i], cudap3ptrs.Ns[i], cudap3ptrs.Ks[i],
                            &alpha, (const DeviceType*)cudap3ptrs.Abp[i], cudap3ptrs.Ms[i],
                            (const DeviceType*)cudap3ptrs.xbp[i], cudap3ptrs.Ks[i],
                            &beta,cudap3ptrs.ybp[i], cudap3ptrs.Ms[i]);
//                hipblasgemv(cublashandleptr[i%stream_size], HIPBLAS_OP_N,
//                           cudap3ptrs.Ms[i], cudap3ptrs.Ks[i],
//                           &alpha, (const DeviceType*)cudap3ptrs.Abp[i], cudap3ptrs.Ms[i],
//                           (const DeviceType*)cudap3ptrs.xbp[i], 1, &beta,cudap3ptrs.ybp[i], 1);
            }
            // final merge
            for(int streami=1; streami < stream_size; streami++){
                hipEventRecord(events[stream_size + streami], streamptr[streami]);
            }
            for(int streami=1; streami < stream_size; streami++){
                hipStreamWaitEvent(streamptr[0], events[stream_size + streami],0);
            }
            hipStreamEndCapture(streamptr[0], &graph);
            hipGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
            graphCreated = true;
        }
        hipGraphLaunch(instance, streamptr[0]);
        hipStreamSynchronize(streamptr[0]);
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::MVMGraph(){
        if(transpose){
            MVMGraphTranspose();
        }else{
            MVMGraphNoTranspose();
        }
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::MVMTranspose()
    {
        hipDeviceSynchronize();
        for(int i=0; i<config.Ntg; i++){
            if(cudap1transptrs.Ms[i] != 0){
                hipblasgemv(cublashandleptr[i%stream_size], HIPBLAS_OP_T,
                           cudap1transptrs.Ks[i], cudap1transptrs.Ms[i],
                           &alpha, (const DeviceType*)cudap1transptrs.Abp[i],
                           cudap1transptrs.Ks[i],
                           (const DeviceType*)cudap1transptrs.xbp[i], 1, &beta,
                           cudap1transptrs.ybp[i], 1);
            }
        }
        hipDeviceSynchronize();
        phase2_nosplit<DeviceType>(cudap1transptrs.y, d_phase2mapping_transpose,
                                   cudap3transptrs.x, config.granksum, streamptr[0]);
        hipDeviceSynchronize();
        for(int i=0; i<config.Mtg; i++){
            hipblasgemv(cublashandleptr[i%stream_size], HIPBLAS_OP_T,
                       cudap3transptrs.Ks[i], cudap3transptrs.Ms[i],
                       &alpha, (const DeviceType*)cudap3transptrs.Abp[i],
                       cudap3transptrs.Ks[i],
                       (const DeviceType*)cudap3transptrs.xbp[i], 1, &beta,cudap3transptrs.ybp[i], 1);
        }
        hipDeviceSynchronize();
    }
    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType,DeviceType>::MVMNoTranspose()
    {
        hipDeviceSynchronize();
        for(int i=0; i<config.Ntg; i++){
            if(cudap1ptrs.Ms[i] != 0){
                hipblasgemv(cublashandleptr[i%stream_size], HIPBLAS_OP_N,
                           cudap1ptrs.Ms[i], cudap1ptrs.Ks[i],
                           &alpha, (const DeviceType*)cudap1ptrs.Abp[i], cudap1ptrs.Ms[i],
                           (const DeviceType*)cudap1ptrs.xbp[i], 1, &beta,
                           cudap1ptrs.ybp[i], 1);
            }
        }
        hipDeviceSynchronize();
        phase2_nosplit<DeviceType>(cudap1ptrs.y, d_phase2mapping, cudap3ptrs.x,
                                   config.granksum, streamptr[0]);
        hipDeviceSynchronize();
        for(int i=0; i<config.Mtg; i++){
            hipblasgemv(cublashandleptr[i%stream_size], HIPBLAS_OP_N,
                       cudap3ptrs.Ms[i], cudap3ptrs.Ks[i],
                       &alpha, (const DeviceType*)cudap3ptrs.Abp[i], cudap3ptrs.Ms[i],
                       (const DeviceType*)cudap3ptrs.xbp[i], 1, &beta,cudap3ptrs.ybp[i], 1);
        }
        hipDeviceSynchronize();
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType, DeviceType>::CopyBackResults()
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
    void Tlrmvmhip<HostType, DeviceType>::MVM() {
        if(transpose){
            MVMTranspose();
        }else{
            MVMNoTranspose();
        }
    }

    template<typename HostType, typename DeviceType>
    void Tlrmvmhip<HostType, DeviceType>::SetTransposeConjugate(bool transpose, bool conjugate) {
        this->transpose = transpose;
        this->conjugate = conjugate;
        tlrmvmcpu->SetTransposeConjugate(transpose, conjugate);
    }



    template class Tlrmvmhip<float, float>;
    template class Tlrmvmhip<double, double>;
    template class Tlrmvmhip<complex<float>, hipComplex>;
    template class Tlrmvmhip<complex<double>, hipDoubleComplex>;

} // namespace cudatlrmvm

