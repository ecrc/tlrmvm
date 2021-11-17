#ifndef TLRMVM_CPU_H
#define TLRMVM_CPU_H

#include "common/Common.h"
#ifdef USE_MPI
#include <mpi.h>
#endif 
#include <complex>

using namespace tlrmat;

namespace tlrmvm{

/**
 * @brief Tlrmvm Config struct for input of tlrmvm.
 * 
 */
struct TlrmvmConfig{
    int originM;
    int originN;
    int Mtg;
    int Ntg;
    int nb;
    string datafolder;
    string acc;
    string problemname;
    Matrix<int> Maskmat;
    TlrmvmConfig(int originM, int originN, int nb, 
    string datafolder, string acc, string problemname);
};

template<typename T>
class TlrmvmBase{
    public:
    TlrmvmBase(TlrmvmConfig tlrmvmconfig);

    void InitData();
    virtual void MemoryInit() = 0;    
    virtual void MemoryFree() = 0;    
    
    void Phase1();
    void Phase1GetMembuffer();
    virtual void AllocatePhase1Buffer() = 0;
    void Phase1CopyData();

    void Phase2();
    void Phase2Prepare();

    void Phase3();
    void Phase3GetMembuffer();
    virtual void AllocatePhase3Buffer() = 0;
    void Phase3CopyData();
    
    virtual void FreeData() = 0;

    void MVM();
    void MPIMVM();
    void setX(T * xvector, size_t xlength);
    TlrmvmConfig tlrmvmconfig;
    // total elements counts for phase1 and phase3
    size_t phase1Acnt;
    size_t phase1Xcnt;
    size_t phase1Ycnt;
    size_t phase3Acnt;
    size_t phase3Xcnt;
    size_t phase3Ycnt;

    // global 
    int originM;
    int originN;
    int paddingM;
    int paddingN;
    int Mtg;
    int Ntg;
    int nb;
    Matrix<int> OrgRmat;
    Matrix<int> WorkRmat;
    Matrix<int> Maskmat;
    vector<int> colsum;
    vector<int> rowsum;
    vector<int> gcolsum;
    Matrix<T> xmat;
    size_t workmatgranksum;
    size_t granksum;

    // phase 1
    vector<size_t> AvMs;
    vector<size_t> AvKs;
    vector<size_t> AvNs;

    T *DataAv;
    T *Datax;
    T *DataAu;

    // phase1 cpu
    T *h_Av;
    T *h_x;
    T *h_yv;
    T *h_yvout;
    T **h_Avbp;
    T **h_xbp;
    T **h_yvbp;

    // phase 2  
    vector<size_t> h_phase2mapping;

    // phase 3
    vector<size_t> AuMs;
    vector<size_t> AuKs;
    vector<size_t> AuNs;

    T *h_Au;
    T *h_yu;
    T *h_y;
    T *h_yout;
    T **h_Aubp;
    T **h_yubp;
    T **h_ybp;
    T **h_youtbp;
    int *d_growrank;
    
    T alpha;
    T beta;
};

template<typename T>
class TlrmvmCPU : public TlrmvmBase<T>
{
    public:
    TlrmvmCPU(TlrmvmConfig tlrmvmconfig);

    using TlrmvmBase<T>::InitData;
    void FreeData();
    void MemoryInit();
    void MemoryFree();
    using TlrmvmBase<T>::Phase1;
    using TlrmvmBase<T>::Phase1GetMembuffer;
    void AllocatePhase1Buffer();
    using TlrmvmBase<T>::Phase1CopyData;

    using TlrmvmBase<T>::Phase2;
    using TlrmvmBase<T>::Phase2Prepare;

    using TlrmvmBase<T>::Phase3;
    using TlrmvmBase<T>::Phase3GetMembuffer;
    void AllocatePhase3Buffer();
    using TlrmvmBase<T>::Phase3CopyData;
    using TlrmvmBase<T>::MVM;
    using TlrmvmBase<T>::setX;
    
    using TlrmvmBase<T>::tlrmvmconfig;
    // total elements counts for phase1 and phase3
    using TlrmvmBase<T>::phase1Acnt;
    using TlrmvmBase<T>::phase1Xcnt;
    using TlrmvmBase<T>::phase1Ycnt;
    using TlrmvmBase<T>::phase3Acnt;
    using TlrmvmBase<T>::phase3Xcnt;
    using TlrmvmBase<T>::phase3Ycnt;

    // global 
    using TlrmvmBase<T>::originM;
    using TlrmvmBase<T>::originN;
    using TlrmvmBase<T>::paddingM;
    using TlrmvmBase<T>::paddingN;
    using TlrmvmBase<T>::Mtg;
    using TlrmvmBase<T>::Ntg;
    using TlrmvmBase<T>::nb;
    using TlrmvmBase<T>::OrgRmat;
    using TlrmvmBase<T>::WorkRmat;
    using TlrmvmBase<T>::Maskmat;
    using TlrmvmBase<T>::colsum;
    using TlrmvmBase<T>::rowsum;
    using TlrmvmBase<T>::gcolsum;
    using TlrmvmBase<T>::xmat;
    using TlrmvmBase<T>::workmatgranksum;
    using TlrmvmBase<T>::granksum;

    // phase 1
    using TlrmvmBase<T>::AvMs;
    using TlrmvmBase<T>::AvKs;
    using TlrmvmBase<T>::AvNs;

    using TlrmvmBase<T>::DataAv;
    using TlrmvmBase<T>::Datax;
    using TlrmvmBase<T>::DataAu;

    // phase1 cpu
    using TlrmvmBase<T>::h_Av;
    using TlrmvmBase<T>::h_x;
    using TlrmvmBase<T>::h_yv;
    using TlrmvmBase<T>::h_yvout;
    using TlrmvmBase<T>::h_Avbp;
    using TlrmvmBase<T>::h_xbp;
    using TlrmvmBase<T>::h_yvbp;

    // phase 2  
    using TlrmvmBase<T>::h_phase2mapping;

    // phase 3
    using TlrmvmBase<T>::AuMs;
    using TlrmvmBase<T>::AuKs;
    using TlrmvmBase<T>::AuNs;

    using TlrmvmBase<T>::h_Au;
    using TlrmvmBase<T>::h_yu;
    using TlrmvmBase<T>::h_y;
    using TlrmvmBase<T>::h_yout;
    using TlrmvmBase<T>::h_Aubp;
    using TlrmvmBase<T>::h_yubp;
    using TlrmvmBase<T>::h_ybp;
    using TlrmvmBase<T>::h_youtbp;
    using TlrmvmBase<T>::d_growrank;
    
    using TlrmvmBase<T>::alpha;
    using TlrmvmBase<T>::beta;
};

int CalculatePadding(int originDim, int nb);

template<typename T>
size_t TLRMVMBytesProcessed(size_t granksum, size_t nb, size_t M, size_t N);


}

#endif // TLRMVM_CPU_H


