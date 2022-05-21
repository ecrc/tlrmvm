
#ifdef USE_MPI
#include <mpi.h>
#endif
#include <oneapi/mkl.hpp>

#include "TlrmvmCPU.hpp"
#include "TlrmvmDPCPP.hpp"
#include "common/Common.hpp"
#include "../../common/AppUtil.hpp"

namespace tlrmvm {


template<typename T>
TlrmvmDPCPP<T>::TlrmvmDPCPP(TlrmvmConfig tlrmvmconfig)
:TlrmvmBase<T>(tlrmvmconfig)
{
    // Create a queue on the default device.
    device = sycl::device(sycl::cpu_selector());
    device_queue = sycl::queue(device);
    std::cout << "Device: "
    << device_queue.get_device().get_info<sycl::info::device::name>()
    << std::endl;
}


template<typename T>
void TlrmvmDPCPP<T>::MemoryInit(){
    this->InitData();
    this->Phase1GetMembuffer();
    this->AllocatePhase1Buffer();
    this->Phase1CopyData();
    this->Phase2Prepare();
    this->Phase3GetMembuffer();
    this->AllocatePhase3Buffer();
    this->Phase3CopyData();
    this->FreeData();
}

template<typename T>
void TlrmvmDPCPP<T>::AllocatePhase1Buffer(){
    // host memory  phase 1
    this->h_Av = sycl::malloc_shared<T>(this->phase1Acnt, device_queue);
    this->h_x = sycl::malloc_shared<T>(this->phase1Xcnt, device_queue);
    this->h_yv = sycl::malloc_shared<T>(this->phase1Ycnt, device_queue);
    this->h_xbp[0] = this->h_x;
    this->h_Avbp[0] = this->h_Av;
    this->h_yvbp[0] = this->h_yv;
}

template<typename T>
void TlrmvmDPCPP<T>::AllocatePhase3Buffer(){
    // host memory  phase 3
    this->h_Au = sycl::malloc_shared<T>(this->phase3Acnt, device_queue);
    this->h_yu = sycl::malloc_shared<T>(this->phase3Xcnt, device_queue);
    this->h_y = sycl::malloc_shared<T>(this->phase3Ycnt, device_queue);
    this->h_yout = sycl::malloc_shared<T>(this->phase3Ycnt, device_queue);
    this->h_Aubp[0] = this->h_Au;
    this->h_yubp[0] = this->h_yu;
    this->h_ybp[0] = this->h_y;
    this->h_youtbp[0] = this->h_yout;
}

template<typename T>
void TlrmvmDPCPP<T>::FreeData(){
    if(this->DataAv) delete[] this->DataAv;
    if(this->DataAu) delete[] this->DataAu;
    if(this->Datax) delete[] this->Datax;
}


template class TlrmvmDPCPP<float>;
template class TlrmvmDPCPP<complex<float>>;
template class TlrmvmDPCPP<double>;
template class TlrmvmDPCPP<complex<double>>;




// TlrmvmDPCPP_Astronomy::TlrmvmDPCPP_Astronomy(TlrmvmConfig tlrmvmconfig)
// :TlrmvmDPCPP<float>(tlrmvmconfig)
// {}

// void TlrmvmDPCPP_Astronomy::InitData(){
//     string datafolder = this->tlrmvmconfig.datafolder;
//     size_t granksum = this->OrgRmat.Sum();
//     int nb = this->tlrmvmconfig.nb;
//     string acc = this->tlrmvmconfig.acc;
//     string id = this->tlrmvmconfig.mavisid;
//     ReadAstronomyBinary(datafolder+"/V", &this->DataAv, granksum * nb, acc ,nb, id);
//     ReadAstronomyBinary(datafolder+"/U", &this->DataAu, granksum * nb, acc ,nb, id);  
//     this->Datax = new float[this->tlrmvmconfig.Ntg * nb];
//     for(int i=0; i<this->tlrmvmconfig.Ntg*nb; i++){
//         this->Datax[i] = (float)0.1;
//     }
//     xmat = Matrix<float>(this->Datax, this->tlrmvmconfig.Ntg * nb, 1);
// }

// void TlrmvmDPCPP_Astronomy::Phase1(){
//     float alpha = 1.0;
//     float beta = 0.0;
//     auto transA = oneapi::mkl::transpose::nontrans;
//     for(int i=0; i<this->Ntg; i++){
//         if(this->colsum[i] != 0){
//             oneapi::mkl::blas::gemv(device_queue, 
//             transA, this->AvMs[i],
//             this->AvKs[i], alpha, this->h_Avbp[i], 
//             this->AvMs[i], this->h_xbp[i],
//             1, beta, this->h_yvbp[i], 1);
//         }
//     }
// }

// void TlrmvmDPCPP_Astronomy::Phase2(){
//     sycl::range<1>num_items{workmatgranksum};
//     this->device_queue.parallel_for(num_items, [this](int i){this->h_yu[this->h_phase2mapping[i]] = this->h_yv[i];});
// }

// void TlrmvmDPCPP_Astronomy::Phase3(){
//     float alpha = 1.0;
//     float beta = 0.0;
//     auto transA = oneapi::mkl::transpose::nontrans;
//     for(int i=0; i<this->Mtg; i++){
        
//         if(this->rowsum[i] != 0){
//             oneapi::mkl::blas::gemv(device_queue, 
//             transA, this->AuMs[i],
//             this->AuKs[i], alpha, this->h_Aubp[i], 
//             this->AuMs[i], this->h_yubp[i],
//             1, beta, this->h_ybp[i], 1);
//         }
//     }
// }


// //////////////////////  Seismic Redtauming

// TlrmvmDPCPP_SeismicRedatuming::TlrmvmDPCPP_SeismicRedatuming(TlrmvmConfig tlrmvmconfig)
// :TlrmvmDPCPP<complex<float>>(tlrmvmconfig)
// {}

// void TlrmvmDPCPP_SeismicRedatuming::InitData(){
//     string datafolder = this->tlrmvmconfig.datafolder;
//     size_t granksum = this->OrgRmat.Sum();
//     int nb = this->tlrmvmconfig.nb;
//     string acc = this->tlrmvmconfig.acc;
//     int freqid = this->tlrmvmconfig.seismicfreq;
//     ReadSeismicBinary(datafolder+"/V", &this->DataAv, granksum * nb, acc ,nb, freqid);
//     ReadSeismicBinary(datafolder+"/U", &this->DataAu, granksum * nb, acc ,nb, freqid);  
//     complex<float> *originx;

//     ReadSeismicBinaryX(datafolder, &originx, originN, acc, nb, freqid);
//     this->Datax = new complex<float>[this->tlrmvmconfig.Ntg * nb];
//     for(int i=0; i<this->tlrmvmconfig.Ntg * nb; i++){
//         if(i < originN) this->Datax[i] = originx[i];
//         else this->Datax[i] = complex<float>(0.0,0.0);
//     }
//     delete[] originx;
//     xmat = Matrix<complex<float>>(this->Datax, this->tlrmvmconfig.Ntg * nb, 1);
// }

// void TlrmvmDPCPP_SeismicRedatuming::Phase1(){
//     complex<float> alpha = complex<float>(1.0,0.0);
//     complex<float> beta = complex<float>(0.0,0.0);
//     auto transA = oneapi::mkl::transpose::nontrans;
//     for(int i=0; i<this->Ntg; i++){
//         if(this->colsum[i] != 0){
//             oneapi::mkl::blas::gemv(device_queue, 
//             transA, this->AvMs[i],
//             this->AvKs[i], alpha, this->h_Avbp[i], 
//             this->AvMs[i], this->h_xbp[i],
//             1, beta, this->h_yvbp[i], 1);
//         }
//     }
// }

// void TlrmvmDPCPP_SeismicRedatuming::Phase2(){
//     sycl::range<1>num_items{workmatgranksum};
//     this->device_queue.parallel_for(num_items, [this](int i){this->h_yu[this->h_phase2mapping[i]] = this->h_yv[i];});
// }

// void TlrmvmDPCPP_SeismicRedatuming::Phase3(){
//     float alpha = 1.0;
//     float beta = 0.0;
//     auto transA = oneapi::mkl::transpose::nontrans;
//     for(int i=0; i<this->Mtg; i++){
        
//         if(this->rowsum[i] != 0){
//             oneapi::mkl::blas::gemv(device_queue, 
//             transA, this->AuMs[i],
//             this->AuKs[i], alpha, this->h_Aubp[i], 
//             this->AuMs[i], this->h_yubp[i],
//             1, beta, this->h_ybp[i], 1);
//         }
//     }
// }


}

