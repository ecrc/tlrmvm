#pragma once 
#include "TlrmvmCPU.hpp"
#include <CL/sycl.hpp>

#include "../../common/Common.hpp"
#ifdef USE_MPI
#include <mpi.h>
#endif
#include <complex>


using namespace tlrmat;

namespace tlrmvm{

template<typename T>
class TlrmvmDPCPP : public TlrmvmBase<T>
{
    public:
    TlrmvmDPCPP(TlrmvmConfig tlrmvmconfig);
    virtual void InitData() = 0;
    void FreeData();
    void MemoryInit();
    void Phase1();
    void Phase2();
    void Phase3();
    void AllocatePhase1Buffer();
    void AllocatePhase3Buffer();
    sycl::device device;
    sycl::queue device_queue;
};


// class TlrmvmDPCPP_Astronomy: public TlrmvmDPCPP<float>
// {
//     public:
//     TlrmvmDPCPP_Astronomy(TlrmvmConfig tlrmvmconfig);
//     void InitData();
//     using TlrmvmDPCPP<float>::device_queue;
//     using TlrmvmDPCPP<float>::MemoryInit;
//     void Phase1();
//     void Phase2();
//     void Phase3();
//     using TlrmvmDPCPP<float>::FreeData;
// };



// class TlrmvmDPCPP_SeismicRedatuming: public TlrmvmDPCPP<complex<float>>
// {
//     public:
//     TlrmvmDPCPP_SeismicRedatuming(TlrmvmConfig tlrmvmconfig);
//     void InitData();
//     using TlrmvmDPCPP<complex<float>>::device_queue;
//     using TlrmvmDPCPP<complex<float>>::MemoryInit;
//     void Phase1();
//     void Phase2();
//     void Phase3();
//     using TlrmvmDPCPP<complex<float>>::FreeData;
// };



}

