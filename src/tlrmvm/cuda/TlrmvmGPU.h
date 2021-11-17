#pragma once 

#include "common/Common.h"
#include "tlrmvm/cpu/TlrmvmCPU.h"
#include "common/cuda/Util.h"
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h> // if you need CUBLAS v2, include before magma.h

using namespace cudatlrmat;
using namespace tlrmvm;

namespace cudatlrmvm
{

template<typename HostType, typename DeviceType>
class TlrmvmGPU: public TlrmvmCPU<HostType>
{
    public:
    TlrmvmGPU(TlrmvmConfig tlrmvmconfig);
    using TlrmvmCPU<HostType>::InitData;
    void StreamInit(int streamsize);
    void StreamDestroy();
    void FreeData();
    void MemoryInit();
    void MemoryFree();
    void Phase1();
    void Phase1GetMembuffer();
    void AllocatePhase1Buffer();
    void Phase1CopyData();
    void Phase2();
    void Phase2Prepare();
    void Phase3();
    void Phase3GetMembuffer();
    void AllocatePhase3Buffer();
    void Phase3CopyData();
    void MVM();
    void setX(HostType * xvector, size_t xlength);
    void CopyBackResults();

    // GPU pointers
    cudaStream_t * streamptr;
    cublasHandle_t * cublashandleptr;
    int streamsize;
    // cuda graph
    cudaGraph_t graph;
    bool graphCreated;
    cudaGraphExec_t instance;
    cudaEvent_t *events;
    cudaEvent_t event_start;
    cudaEvent_t event_phase2finish;
    
    DeviceType alpha;
    DeviceType beta;
    // phase1 gpu
    DeviceType *d_Av;
    DeviceType *d_x;
    DeviceType *d_yv;
    DeviceType *d_yvout;
    DeviceType **d_Avbp;
    DeviceType **d_xbp;
    DeviceType **d_yvbp;
    DeviceType **d_yvoutbp;
    DeviceType *d_finaly;
    int *d_complexcolrank;

    size_t *d_phase2mapping;

    DeviceType *d_Au;
    DeviceType *d_yu;
    DeviceType *d_y;
    DeviceType *d_yout;
    DeviceType **d_Aubp;
    DeviceType **d_yubp;
    DeviceType **d_ybp;
    DeviceType **d_youtbp;
    int *d_complexrowrank;

};




class ComplexPtr{
    public:
    ComplexPtr(int M, int N, int nb, Matrix<int> OrgRmat, Matrix<int> FP32Rmat);
    void InitData(string datafolder, string acc, int freqx, size_t originN);
    void LoadRealDataset(complex<float>* DataAv, complex<float> *DataAu, complex<float>*Datax);
    void CopyData2GPU();
    void CopyResult2CPU();
    void FreeData();

    // global 
    int M;
    int N;
    int Mtg;
    int Ntg;
    int nb;
    size_t originN;
    Matrix<int> OrgRmat;
    Matrix<int> ComplexRmat;
    Matrix<int> Maskmat;
    vector<int> colsum;
    vector<int> rowsum;
    vector<int> gcolsum;
    size_t complexgranksum;


    // phase 1
    vector<size_t> AvMs;
    vector<size_t> AvKs;
    vector<size_t> AvNs;

    // phase1 cpu
    complex<float> *h_Av;
    complex<float> *h_x;
    complex<float> *h_yv;
    complex<float> *h_yvout;
    complex<float> *h_Avbp[39];
    complex<float> *h_xbp[39];
    complex<float> *h_yvbp[39];
    complex<float> *h_yvoutbp[39];
    complex<float>* h_finaly;

    // phase1 gpu
    cuComplex *d_Av;
    cuComplex *d_x;
    cuComplex *d_yv;
    cuComplex *d_yvout;
    cuComplex *d_Avbp[39];
    cuComplex *d_xbp[39];
    cuComplex *d_yvbp[39];
    cuComplex *d_yvoutbp[39];
    cuComplex *d_finaly;
    int *d_complexcolrank;

    // phase 2  
    vector<size_t> h_phase2mapping;
    size_t* d_phase2mapping;

    // phase 3
    vector<size_t> AuMs;
    vector<size_t> AuKs;
    vector<size_t> AuNs;
    complex<float> *h_Au;
    complex<float> *h_yu;
    complex<float> *h_y;
    complex<float> *h_yout;
    complex<float> *h_Aubp[39];
    complex<float> *h_yubp[39];
    complex<float> *h_ybp[39];
    complex<float> *h_youtbp[39];
    
    

    cuComplex *d_Au;
    cuComplex *d_yu;
    cuComplex *d_y;
    cuComplex *d_yout;
    cuComplex *d_Aubp[39];
    cuComplex *d_yubp[39];
    cuComplex *d_ybp[39];
    cuComplex *d_youtbp[39];
    int *d_complexrowrank;

};


} // 



