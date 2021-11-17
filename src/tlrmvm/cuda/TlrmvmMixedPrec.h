#pragma once 

#include "common/Common.h"
#include "tlrmvm/cpu/TlrmvmCPU.h"
#include "common/cuda/Util.h"
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h> // if you need CUBLAS v2, include before magma.h

using namespace cudatlrmat;

namespace cudatlrmvm
{

struct MAXMINInfo{
    float maxreal;
    float minreal;
    float maximag;
    float minimag;
    MAXMINInfo()
    :maxreal(0),minreal(0),maximag(0),minimag(0)
    {}
};

class Float32Ptr{
    public:
    Float32Ptr(int M, int N, int nb, Matrix<int> OrgRmat, Matrix<int> FP32Rmat);
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
    Matrix<int> FP32Rmat;
    Matrix<int> Maskmat;
    vector<int> colsum;
    vector<int> rowsum;
    vector<int> gcolsum;
    size_t fp32granksum;


    // phase 1
    vector<size_t> AvMs;
    vector<size_t> AvKs;
    vector<size_t> AvNs;

    // phase1 cpu
    float *h_Av[2];
    float *h_x;
    float *h_yv[2];
    float *h_yvout[2];
    float *h_Avbp[39][2];
    float *h_xbp[39];
    float *h_yvbp[39][2];
    float *h_yvoutbp[39][2];
    complex<float>* h_finaly;

    // phase1 gpu
    float *d_Av[2];
    float *d_x;
    float *d_yv[2];
    float *d_yvout[2];
    float *d_Avbp[39][2];
    float *d_xbp[39];
    float *d_yvbp[39][2];
    float *d_yvoutbp[39][2];
    cuComplex *d_finaly;
    int *d_fp32colrank;

    // phase 2  
    vector<size_t> h_phase2mapping;
    size_t* d_phase2mapping;

    // phase 3
    vector<size_t> AuMs;
    vector<size_t> AuKs;
    vector<size_t> AuNs;
    float *h_Au[2];
    float *h_yu;
    float *h_y[2];
    float *h_yout[2];
    float *h_Aubp[39][2];
    float *h_yubp[39];
    float *h_ybp[39][2];
    float *h_youtbp[39][2];
    int *d_fp32rowrank;
    

    float *d_Au[2];
    float *d_yu;
    float *d_y[2];
    float *d_yout[2];
    float *d_Aubp[39][2];
    float *d_yubp[39];
    float *d_ybp[39][2];
    float *d_youtbp[39][2];

};

class Float16Ptr{
    public:
    Float16Ptr(int M, int N, int nb, Matrix<int> OrgRmat, Matrix<int> FP16Rmat);
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
    Matrix<int> FP16Rmat;
    Matrix<int> Maskmat;
    vector<int> colsum;
    vector<int> rowsum;
    vector<int> gcolsum;
    size_t fp16granksum;


    // phase 1
    vector<size_t> AvMs;
    vector<size_t> AvKs;
    vector<size_t> AvNs;

    // phase1 cpu
    half *h_Av[2];
    half *h_x;
    half *h_yv[2];
    half *h_yvout[2];
    half *h_Avbp[39][2];
    half *h_xbp[39];
    half *h_yvbp[39][2];
    half *h_yvoutbp[39][2];
    complex<float>* h_finaly;

    // phase1 gpu
    half *d_Av[2];
    half *d_x;
    half *d_yv[2];
    half *d_yvout[2];
    half *d_Avbp[39][2];
    half *d_xbp[39];
    half *d_yvbp[39][2];
    half *d_yvoutbp[39][2];
    cuComplex *d_finaly;
    int *d_fp16colrank;

    // phase 2  
    vector<size_t> h_phase2mapping;
    size_t* d_phase2mapping;

    // phase 3
    vector<size_t> AuMs;
    vector<size_t> AuKs;
    vector<size_t> AuNs;
    half *h_Au[2];
    half *h_yu;
    half *h_y[2];
    half *h_yout[2];
    half *h_Aubp[39][2];
    half *h_yubp[39];
    half *h_ybp[39][2];
    half *h_youtbp[39][2];
    int *d_fp16rowrank;
    

    half *d_Au[2];
    half *d_yu;
    half *d_y[2];
    half *d_yout[2];
    half *d_Aubp[39][2];
    half *d_yubp[39];
    half *d_ybp[39][2];
    half *d_youtbp[39][2];

};


// class Int8Ptr{
//     public:
//     Int8Ptr(int M, int N, int nb, Matrix<int> OrgRmat, Matrix<int> Int8Rmat);
//     void InitData(string datafolder, string acc, int freqx, size_t originN);
//     void LoadRealDataset(complex<float>* DataAv, complex<float> *DataAu, complex<float>*Datax);
//     void CopyData2GPU();
//     void CopyResult2CPU();
//     void FreeData();

//     // global 
//     int M;
//     int N;
//     int Mtg;
//     int Ntg;
//     int nb;
//     size_t originN;
//     Matrix<int> OrgRmat;
//     Matrix<int> Int8Rmat;
//     Matrix<int> Maskmat;
//     vector<int> colsum;
//     vector<int> rowsum;
//     vector<int> gcolsum;
//     size_t int8granksum;

//     // phase 1
//     vector<size_t> AvMs;
//     vector<size_t> AvKs;
//     vector<size_t> AvNs;

//     int8_t MAXINT8 = 126;

//     // Int8 special max member
//     float Avcolmax[39];
//     float xtilemax[39];
//     float yutilemax[39];
//     float Aurowmax[39];

//     // phase1 cpu real memory 
//     complex<float> *h_Av_complex; // only on CPU, for calculation of Avcolmax
//     int8_t *h_Av;
//     int8_t *h_x;
    
//     complex<float> *h_yv;

//     // phase 1 cpu, batch pointers
//     int8_t *h_Avbp[39];
//     int8_t *h_xbp[39];
//     complex<float> *h_yvbp[39];
    

//     // phase1 gpu real memory
//     int8_t *d_Av;
//     int8_t *d_x;
//     cuComplex  *d_yv;

//     // phase 1 gpu batch pointers
//     int8_t *d_Avbp[39];
//     int8_t *d_xbp[39];
//     cuComplex *d_yvbp[39];

//     // cuComplex *d_finaly;
//     int *d_int8colrank;

// };


class Int8Ptr{
    public:
    Int8Ptr(int M, int N, int nb, Matrix<int> OrgRmat, Matrix<int> Int8Rmat);
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
    Matrix<int> Int8Rmat;
    Matrix<int> Maskmat;
    vector<int> colsum;
    vector<int> colsum_withpadding;
    vector<int> rowsum;
    vector<int> gcolsum;
    size_t int8granksum;
    size_t int8granksum_withpadding;
    float MAXINT8 = 126;

    float Avtilemax[39][39][2];
    float xtilemax[39][2];
    float yutilemax[39][39][2];
    float Autilemax[39][39][2];

    float h_Avtilemax[39*39*2];
    float d_Avtilemax[39*39*2];
    float h_xtilemax[39*2];
    float d_xtilemax[39*2];
    float h_yutilemax[39*39*2];
    float d_yutilemax[39*39*2];
    float h_Autilemax[39*39*2];
    float d_Autilemax[39*39*2];

    // phase 1
    vector<size_t> AvMs;
    vector<size_t> AvKs;
    vector<size_t> AvNs;

    // phase1 cpu
    int8_t *h_Av[2];
    int8_t *h_x;
    int32_t *h_yv[2];
    float *h_yvout[2];
    int8_t *h_Avbp[39][2];
    int8_t *h_xbp[39];
    int32_t *h_yvbp[39][2];
    float *h_yvoutbp[39][2];
    complex<float>* h_finaly;

    // phase1 gpu
    int8_t *d_Av[2];
    int8_t *d_x;
    int32_t *d_yv[2];
    float *d_yvout[2];
    int8_t *d_Avbp[39][2];
    int8_t *d_xbp[39];
    int32_t *d_yvbp[39][2];
    float *d_yvoutbp[39][2];
    cuComplex *d_finaly;
    int *d_int8colrank;
    int *d_int8colrank_withpadding;

    // phase 2  
    vector<size_t> h_phase2mapping;
    size_t* d_phase2mapping;

    // // phase 3
    // vector<size_t> AuMs;
    // vector<size_t> AuKs;
    // vector<size_t> AuNs;
    // half *h_Au[2];
    // half *h_yu;
    // half *h_y[2];
    // half *h_yout[2];
    // half *h_Aubp[39][2];
    // half *h_yubp[39];
    // half *h_ybp[39][2];
    // half *h_youtbp[39][2];
    // int *d_fp16rowrank;
    

    // half *d_Au[2];
    // half *d_yu;
    // half *d_y[2];
    // half *d_yout[2];
    // half *d_Aubp[39][2];
    // half *d_yubp[39];
    // half *d_ybp[39][2];
    // half *d_youtbp[39][2];

};


}


