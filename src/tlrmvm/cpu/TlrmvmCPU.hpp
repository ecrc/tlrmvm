// @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                     All rights reserved.

#ifndef TLRMVM_CPU_H
#define TLRMVM_CPU_H

#include "../../common/Common.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif 
#include <complex>


/**
 * @brief Tlrmvm Config struct for input of tlrmvm.
 *
 */


struct TlrmvmConfig{
    int originM;
    int originN;
    int paddingM;
    int paddingN;
    int granksum;
    int workmatgranksum;
    int Mtg;
    int Ntg;
    int nb;
    string datafolder;
    string acc;
    string problemname;
    Matrix<int> Maskmat;
    Matrix<int> OrgRmat;
    Matrix<int> WorkRmat;
    vector<int> colsum;
    vector<int> rowsum;
    vector<int> gcolsum;
    bool isconstrank;
    int constrank;
    TlrmvmConfig(int originM, int originN, int nb, string datafolder, string acc, string problemname);
    TlrmvmConfig(int originM, int originN, int nb, int constranksize);
    TlrmvmConfig();
    void UpdateMaskmat(Matrix<int> maskmat);
    void PrintMaskmat();
};



template<typename T>
struct PhasePointers{
    PhasePointers();
    size_t Acnt;
    size_t Xcnt;
    size_t Ycnt;
    vector<size_t> Ms;
    vector<size_t> Ks;
    vector<size_t> Ns;
    T *A;
    T *x;
    T *y;
    T **Abp;
    T **xbp;
    T **ybp;
};

template<typename T>
class TlrmvmBase{
    // TLR-MVM base class, the base class exists because dpcpp use different memory allocation
    // scheme. So compared to TLRMVMCPU, base class didn't do memory allocation.
    public:
    explicit TlrmvmBase(TlrmvmConfig tlrmvmconfig);
    TlrmvmBase();
    void UpdateConfig(TlrmvmConfig &tlrmvmConfig);
    void SetTransposeConjugate(bool transpose, bool conjugate);
    // loading raw U,V bases
    void InitData();
    void FreeData();
    // Memory init, the combination of 3 phases preparation.
    virtual void MemoryInit() = 0;
    virtual void MemoryFree() = 0;

    // phase 1 functions
    virtual void Phase1();
    virtual void Phase1Transpose();
    virtual void Phase1GetMembuffer();
    virtual void Phase1GetMembufferTranspose();
    virtual void AllocatePhase1Buffer() = 0;
    virtual void AllocatePhase1BufferTranspose() = 0;
    virtual void Phase1CopyData();
    virtual void Phase1CopyDataTranspose();

    // phase 2 functions
    virtual void Phase2();
    virtual void Phase2Transpose();
    virtual void Phase2Prepare();
    virtual void Phase2PrepareTranspose();

    // phase 3 functions
    virtual void Phase3();
    virtual void Phase3Transpose();
    virtual void Phase3GetMembuffer();
    virtual void Phase3GetMembufferTranspose();
    virtual void AllocatePhase3Buffer() = 0;
    virtual void AllocatePhase3BufferTranspose() = 0;
    virtual void Phase3CopyData();
    virtual void Phase3CopyDataTranspose();

    // TLRMVM function, main algorithm function!
    virtual void MVM();
    virtual void MVMNoTranspose();
    virtual void MVMTranspose();

    void MPIMVM();
    // seperate 2 function to convert input and output.
    void TryConjugateXvec();
    void TryConjugateResults();

    // just do copy, only responsible for choosing trans/no trans pointers
    void CopyToFinalresults();

    // load x vector
    virtual void setX(T * xvector, size_t xlength);

    Matrix<T> xmat;
    // data member
    TlrmvmConfig config;
    bool transpose;
    bool conjugate;
    T alpha;
    T beta;
    // raw UV bases pointers
    T *DataAv;
    T *Datax;
    T *DataAu;
    // phase 1 pointers
    PhasePointers<T> p1ptrs;
    PhasePointers<T> p1transptrs;
    // phase 2, mapping vector
    vector<size_t> h_phase2mapping;
    vector<size_t> h_phase2mapping2;
    vector<size_t> h_phase2mappingTranspose;
    // phase 3 pointers
    PhasePointers<T> p3ptrs;
    PhasePointers<T> p3transptrs;
    T * finalresults;
};

template<typename T>
class TlrmvmCPU : public TlrmvmBase<T>
{
    public:
    explicit TlrmvmCPU(TlrmvmConfig tlrmvmconfig);
    TlrmvmCPU();

    using TlrmvmBase<T>::InitData;
    using TlrmvmBase<T>::FreeData;
    using TlrmvmBase<T>::SetTransposeConjugate;
    using TlrmvmBase<T>::UpdateConfig;

    virtual void MemoryInit();
    virtual void MemoryFree();

    using TlrmvmBase<T>::Phase1;
    using TlrmvmBase<T>::Phase1Transpose;
    using TlrmvmBase<T>::Phase1GetMembuffer;
    using TlrmvmBase<T>::Phase1GetMembufferTranspose;
    virtual void AllocatePhase1Buffer();
    virtual void AllocatePhase1BufferTranspose();
    using TlrmvmBase<T>::Phase1CopyData;
    using TlrmvmBase<T>::Phase1CopyDataTranspose;

    using TlrmvmBase<T>::Phase2;
    using TlrmvmBase<T>::Phase2Transpose;
    using TlrmvmBase<T>::Phase2Prepare;
    using TlrmvmBase<T>::Phase2PrepareTranspose;

    using TlrmvmBase<T>::Phase3;
    using TlrmvmBase<T>::Phase3Transpose;
    using TlrmvmBase<T>::Phase3GetMembuffer;
    using TlrmvmBase<T>::Phase3GetMembufferTranspose;
    virtual void AllocatePhase3Buffer();
    virtual void AllocatePhase3BufferTranspose();
    using TlrmvmBase<T>::Phase3CopyData;
    using TlrmvmBase<T>::Phase3CopyDataTranspose;
    using TlrmvmBase<T>::MVM;
    using TlrmvmBase<T>::setX;
    using TlrmvmBase<T>::CopyToFinalresults;
    // configure
    using TlrmvmBase<T>::config;
    using TlrmvmBase<T>::transpose;
    using TlrmvmBase<T>::conjugate;

    // raw data pointers
    using TlrmvmBase<T>::DataAv;
    using TlrmvmBase<T>::Datax;
    using TlrmvmBase<T>::DataAu;
    using TlrmvmBase<T>::alpha;
    using TlrmvmBase<T>::beta;
    // phase1 cpu pointers
    using TlrmvmBase<T>::p1ptrs;
    using TlrmvmBase<T>::p1transptrs;
    // phase 2
    using TlrmvmBase<T>::h_phase2mapping;
    using TlrmvmBase<T>::h_phase2mappingTranspose;
    // phase3 cpu pointers
    using TlrmvmBase<T>::p3ptrs;
    using TlrmvmBase<T>::p3transptrs;
    using TlrmvmBase<T>::finalresults;
};

template<typename T>
class BatchTlrmvmCPU{
public:
    explicit BatchTlrmvmCPU(vector<TlrmvmConfig> tlrmvmvec);
    vector<TlrmvmCPU<T>> cpuinst;
    void MemoryInit();
    void MemoryFree();
};


int CalculatePadding(int originDim, int nb);

template<typename T>
size_t TLRMVMBytesProcessed(size_t granksum, size_t nb, size_t M, size_t N);

#endif // TLRMVM_CPU_H


