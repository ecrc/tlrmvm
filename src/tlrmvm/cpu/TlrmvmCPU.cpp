
// @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                     All rights reserved.

#include <memory.h>
#include <cassert>

#include "TlrmvmCPU.hpp"
#include "../../common/Common.hpp"
#include "../../common/AppUtil.hpp"
#include "../../common/cpu/Util.hpp"
#include <memory>
#include <string.h>
#include <stdio.h>
#ifdef USE_MPI
#include <mpi.h>
#endif

int CalculatePadding(int originDim, int nb){
    return ( originDim / nb + (originDim % nb != 0) ) * nb;
}


TlrmvmConfig::TlrmvmConfig(){}

TlrmvmConfig::TlrmvmConfig(int originM, int originN, int nb,
string datafolder, string acc, string problemname)
:datafolder(datafolder),acc(acc),problemname(problemname),
originM(originM), originN(originN),nb(nb)
{
    isconstrank = false;
    paddingM = CalculatePadding(originM, nb);
    paddingN = CalculatePadding(originN, nb);
    Mtg = CalculatePadding(originM, nb) / nb;
    Ntg = CalculatePadding(originN, nb) / nb;
    char filename[200];
    sprintf(filename, "%s/%s_Rmat_nb%d_acc%s.bin",datafolder.c_str(),
            problemname.c_str(),nb, acc.c_str());
    OrgRmat = Matrix<int>::Fromfile(filename, Mtg, Ntg);
    granksum = OrgRmat.Sum();
    Maskmat = Matrix<int>(Mtg,Ntg);
    Maskmat.Fill(1);
    WorkRmat = OrgRmat.ApplyMask(Maskmat);
    colsum = WorkRmat.ColSum();
    rowsum = WorkRmat.RowSum();
    gcolsum = OrgRmat.ColSum();
    workmatgranksum = WorkRmat.Sum();
}

TlrmvmConfig::TlrmvmConfig(int originM, int originN, int nb, int constrank)
:datafolder(""),acc(""),problemname(""),
originM(originM), originN(originN),nb(nb)
{
    isconstrank = true;
    paddingM = CalculatePadding(originM, nb);
    paddingN = CalculatePadding(originN, nb);
    Mtg = CalculatePadding(originM, nb) / nb;
    Ntg = CalculatePadding(originN, nb) / nb;
    OrgRmat = Matrix<int>(Mtg, Ntg);
    OrgRmat.Fill(constrank);
    granksum = OrgRmat.Sum();
    Maskmat = Matrix<int>(Mtg,Ntg);
    Maskmat.Fill(1);
    WorkRmat = OrgRmat.ApplyMask(Maskmat);
    colsum = WorkRmat.ColSum();
    rowsum = WorkRmat.RowSum();
    gcolsum = OrgRmat.ColSum();
    workmatgranksum = WorkRmat.Sum();
}

void TlrmvmConfig::UpdateMaskmat(Matrix<int> maskmat) {
    this->Maskmat = maskmat;
    WorkRmat = OrgRmat.ApplyMask(Maskmat);
    colsum = WorkRmat.ColSum();
    rowsum = WorkRmat.RowSum();
    gcolsum = OrgRmat.ColSum();
    workmatgranksum = WorkRmat.Sum();
}

void TlrmvmConfig::PrintMaskmat() {
    cout << Maskmat << endl;
}

template<typename T>
PhasePointers<T>::PhasePointers() {}

template struct PhasePointers<float>;
template struct PhasePointers<double>;
template struct PhasePointers<complex<float>>;
template struct PhasePointers<complex<double>>;


template<typename T>
TlrmvmBase<T>::TlrmvmBase(TlrmvmConfig tlrmvmconfig)
:config(tlrmvmconfig)
{
    transpose = false;
    conjugate = false;
    init_alpha_beta(alpha, beta);
    finalresults = new T[config.originM];
    memset(finalresults, 0, sizeof(T) * config.originM);
}

template<typename T>
TlrmvmBase<T>::TlrmvmBase(){}

template<typename T>
void TlrmvmBase<T>::UpdateConfig(TlrmvmConfig &tlrmvmConfig)
{
    this->config = tlrmvmConfig;
}

template<typename T>
void TlrmvmBase<T>::SetTransposeConjugate(bool transpose, bool conjugate) {
    this->transpose = transpose;
    this->conjugate = conjugate;
}

template<typename T>
void TlrmvmBase<T>::InitData(){
    if(config.isconstrank){
        size_t elems = config.granksum * config.nb;
        DataAu = new T[elems];
        for(int i=0; i<elems; i++) DataAu[i] = 0.001;
        DataAv = new T[elems];
        for(int i=0; i<elems; i++) DataAv[i] = 0.001;
        Datax = new T[config.Ntg * config.nb];
        memset(Datax, 0, sizeof(T) * config.Ntg * config.nb);
        RandomX(Datax, config.originN);
        this->xmat = Matrix<T>(Datax, config.paddingN, 1);
    }else{
        char filename[300];
        sprintf(filename, "%s/%s_Ubases_nb%d_acc%s.bin",
                config.datafolder.c_str(), config.problemname.c_str(),config.nb, config.acc.c_str());
        size_t elems = config.granksum * config.nb;
        LoadBinary(filename, &DataAu, elems);
        sprintf(filename, "%s/%s_Vbases_nb%d_acc%s.bin",
                config.datafolder.c_str(), config.problemname.c_str(),config.nb, config.acc.c_str());
        LoadBinary(filename, &DataAv, elems);
        Datax = new T[config.Ntg * config.nb];
        memset(Datax, 0, sizeof(T) * config.Ntg * config.nb);
        RandomX(Datax, config.originN);
        this->xmat = Matrix<T>(Datax, config.paddingN, 1);
    }
}

template<typename T>
void TlrmvmBase<T>::FreeData(){
    delete[] DataAu;
    delete[] DataAv;
    delete[] Datax;
}

template<typename T>
void TlrmvmBase<T>::setX(T * xvector, size_t xlength){
    assert(xlength == this->config.originN);
    auto tmpx = new T[config.paddingN];
    memset(tmpx, 0, sizeof(T) * config.paddingN);
    CopyData(tmpx, xvector, xlength);
    this->xmat = Matrix<T>(tmpx, config.paddingN, 1);
    delete[] tmpx;
    // move data from Datax to hx
    T * xwalkptr = this->xmat.RawPtr();
    size_t offset = 0;
    for(int i=0; i<p1ptrs.Ms.size(); i++){
        for(int j=0; j<config.nb; j++){
            *(p1ptrs.x + offset + j) = *(xwalkptr + i*config.nb + j);
        }
        offset += config.nb;
    }
}

template<typename T>
void TlrmvmBase<T>::Phase1GetMembuffer(){
    GetHostMemory(&p1ptrs.Abp, config.Ntg);
    GetHostMemory(&p1ptrs.xbp, config.Ntg);
    GetHostMemory(&p1ptrs.ybp, config.Ntg);
    for(int i=0; i<config.Ntg; i++){
        p1ptrs.Ms.push_back(config.colsum[i]);
        p1ptrs.Ks.push_back(config.nb);
        p1ptrs.Ns.push_back(1);
    }
    p1ptrs.Acnt = 0;
    p1ptrs.Xcnt = 0;
    p1ptrs.Ycnt = 0;
    for(int i=0; i<config.Ntg; i++){
        p1ptrs.Acnt += p1ptrs.Ms[i] * p1ptrs.Ks[i];
        p1ptrs.Xcnt += p1ptrs.Ks[i] * p1ptrs.Ns[i];
        p1ptrs.Ycnt += p1ptrs.Ms[i] * p1ptrs.Ns[i];
    }
}

template<typename T>
void TlrmvmBase<T>::Phase1GetMembufferTranspose() {
    GetHostMemory(&p1transptrs.Abp, config.Mtg);
    GetHostMemory(&p1transptrs.xbp, config.Mtg);
    GetHostMemory(&p1transptrs.ybp, config.Mtg);
    for(int i=0; i<config.Mtg; i++){
        p1transptrs.Ms.push_back(config.rowsum[i]);
        p1transptrs.Ks.push_back(config.nb);
        p1transptrs.Ns.push_back(1);
    }
    p1transptrs.Acnt = 0;
    p1transptrs.Xcnt = 0;
    p1transptrs.Ycnt = 0;
    for(int i=0; i<config.Mtg; i++){
        p1transptrs.Acnt += p1transptrs.Ms[i] * p1transptrs.Ks[i];
        p1transptrs.Xcnt += p1transptrs.Ks[i] * p1transptrs.Ns[i];
        p1transptrs.Ycnt += p1transptrs.Ms[i] * p1transptrs.Ns[i];
    }
}

template<typename T>
void TlrmvmBase<T>::Phase1CopyData(){
    for(int i=1; i<p1ptrs.Ms.size(); i++){
        size_t AvMK = p1ptrs.Ms[i-1] * p1ptrs.Ks[i-1];
        size_t AvKN = p1ptrs.Ks[i-1] * p1ptrs.Ns[i-1];
        size_t AvMN = p1ptrs.Ms[i-1] * p1ptrs.Ns[i-1];
        p1ptrs.Abp[i] = p1ptrs.Abp[i-1] + AvMK;
        p1ptrs.xbp[i] = p1ptrs.xbp[i-1] + AvKN;
        p1ptrs.ybp[i] = p1ptrs.ybp[i-1] + AvMN;
    }

    // move data from DataAv to Av
    T *Avwalkptr = DataAv;
    for(int i=0; i<p1ptrs.Ms.size(); i++){
        // column start pointers
        T *colptr = p1ptrs.Abp[i];
        size_t lda = config.gcolsum[i];
        for(int nbi = 0; nbi < config.nb; nbi++){
            for(int j=0; j < config.Mtg; j++){
                int currank = config.OrgRmat.GetElem(j,i);
               if(config.WorkRmat.GetElem(j,i) == config.OrgRmat.GetElem(j,i)){
                   for(int k=0; k<currank; k++){
                       *(colptr+k) = *(Avwalkptr + k);
                   }
                   colptr += currank;
                }
                Avwalkptr += currank;
            }
        }
    }
    // move data from Datax to hx
    T * xwalkptr = this->xmat.RawPtr();
    size_t offset = 0;
    for(int i=0; i<p1ptrs.Ms.size(); i++){
        for(int j=0; j<config.nb; j++){
            *(p1ptrs.x + offset + j) = *(xwalkptr + i*config.nb + j);
        }
        offset += config.nb;
    }
}

template<typename T>
void TlrmvmBase<T>::Phase1CopyDataTranspose() {
    for(int i=1; i<p1transptrs.Ms.size(); i++){
        size_t AvMK = p1transptrs.Ms[i-1] * p1transptrs.Ks[i-1];
        size_t AvKN = p1transptrs.Ks[i-1] * p1transptrs.Ns[i-1];
        size_t AvMN = p1transptrs.Ms[i-1] * p1transptrs.Ns[i-1];
        p1transptrs.Abp[i] = p1transptrs.Abp[i-1] + AvMK;
        p1transptrs.xbp[i] = p1transptrs.xbp[i-1] + AvKN;
        p1transptrs.ybp[i] = p1transptrs.ybp[i-1] + AvMN;
    }
}

template<typename T>
void TlrmvmBase<T>::Phase2Prepare(){
    // phase 2
    vector<vector<vector<int>>> phase2record;
    phase2record.resize(config.Mtg, vector<vector<int>>()); // Mtg row
    for(auto &x : phase2record) x.resize(config.Ntg, vector<int>()); // Ntg col
    size_t p2walker = 0;
    for(int i=0; i<config.Mtg; i++){
        for(int j=0; j<config.Ntg; j++){
            if(config.WorkRmat.GetElem(i,j) != 0){
                int currank = config.WorkRmat.GetElem(i,j);
                for(int k=0; k<currank; k++){
                    phase2record[i][j].push_back(p2walker++);
                }
            }
        }
    }
    // unfold
    for(int i=0; i<config.Ntg; i++){
        for(int j=0; j<config.Mtg; j++){
            if(config.WorkRmat.GetElem(j,i) != 0){
                int currank = config.WorkRmat.GetElem(j,i);
                for(int k=0; k<currank; k++){
                    h_phase2mapping.push_back(phase2record[j][i][k]);
                }
            }
        }
    }
    h_phase2mapping2.clear();
    phase2record.clear();
    phase2record.resize(config.Mtg, vector<vector<int>>()); // Mtg row
    for(auto &x : phase2record) x.resize(config.Ntg, vector<int>()); // Ntg col
    p2walker = 0;
    for(int i=0; i<config.Ntg; i++){
        for(int j=0; j<config.Mtg; j++){
            if(config.WorkRmat.GetElem(j,i) != 0){
                int currank = config.WorkRmat.GetElem(j,i);
                for(int k=0; k<currank; k++){
                    phase2record[j][i].push_back(p2walker++);
                }
            }
        }
    }
    // unfold
    for(int i=0; i<config.Mtg; i++){
        for(int j=0; j<config.Ntg; j++){
            if(config.WorkRmat.GetElem(i,j) != 0){
                int currank = config.WorkRmat.GetElem(i,j);
                for(int k=0; k<currank; k++){
                    h_phase2mapping2.push_back(phase2record[i][j][k]);
                }
            }
        }
    }

}

template<typename T>
void TlrmvmBase<T>::Phase2PrepareTranspose() {
    // phase 2
    vector<vector<vector<int>>> phase2record;
    phase2record.resize(config.Mtg, vector<vector<int>>()); // Mtg row
    for(auto &x : phase2record) x.resize(config.Ntg, vector<int>()); // Ntg col
    size_t p2walker = 0;
    for(int i=0; i<config.Mtg; i++){
        for(int j=0; j<config.Ntg; j++){
            if(config.WorkRmat.GetElem(j,i) != 0){
                int currank = config.WorkRmat.GetElem(j,i);
                for(int k=0; k<currank; k++){
                    phase2record[i][j].push_back(p2walker++);
                }
            }
        }
    }
    // unfold
    for(int i=0; i<config.Ntg; i++){
        for(int j=0; j<config.Mtg; j++){
            if(config.WorkRmat.GetElem(i,j) != 0){
                int currank = config.WorkRmat.GetElem(i,j);
                for(int k=0; k<currank; k++){
                    h_phase2mappingTranspose.push_back(phase2record[j][i][k]);
                }
            }
        }
    }
}

template<typename T>
void TlrmvmBase<T>::Phase3GetMembuffer(){
    GetHostMemory(&p3ptrs.Abp, config.Mtg);
    GetHostMemory(&p3ptrs.xbp, config.Mtg);
    GetHostMemory(&p3ptrs.ybp, config.Mtg);
    for(int i=0; i<config.Mtg; i++){
        p3ptrs.Ms.push_back(config.nb);
        p3ptrs.Ks.push_back(config.rowsum[i]);
        p3ptrs.Ns.push_back(1);
    }
    p3ptrs.Acnt = 0;
    p3ptrs.Xcnt = 0;
    p3ptrs.Ycnt = 0;
    for(int i=0; i<config.Mtg; i++){
        p3ptrs.Acnt += p3ptrs.Ms[i] * p3ptrs.Ks[i];
        p3ptrs.Xcnt += p3ptrs.Ks[i] * p3ptrs.Ns[i];
        p3ptrs.Ycnt += p3ptrs.Ms[i] * p3ptrs.Ns[i];
    }
}
template<typename T>
void TlrmvmBase<T>::Phase3GetMembufferTranspose() {
    GetHostMemory(&p3transptrs.Abp, config.Ntg);
    GetHostMemory(&p3transptrs.xbp, config.Ntg);
    GetHostMemory(&p3transptrs.ybp, config.Ntg);
    for(int i=0; i<config.Ntg; i++){
        p3transptrs.Ms.push_back(config.nb);
        p3transptrs.Ks.push_back(config.colsum[i]);
        p3transptrs.Ns.push_back(1);
    }
    p3transptrs.Acnt = 0;
    p3transptrs.Xcnt = 0;
    p3transptrs.Ycnt = 0;
    for(int i=0; i<config.Ntg; i++){
        p3transptrs.Acnt += p3transptrs.Ms[i] * p3transptrs.Ks[i];
        p3transptrs.Xcnt += p3transptrs.Ks[i] * p3transptrs.Ns[i];
        p3transptrs.Ycnt += p3transptrs.Ms[i] * p3transptrs.Ns[i];
    }
}
template<typename T>
void TlrmvmBase<T>::Phase3CopyData(){
    for(int i=1; i<config.Mtg; i++){
        size_t AuMK = p3ptrs.Ms[i-1] * p3ptrs.Ks[i-1];
        size_t AuKN = p3ptrs.Ks[i-1] * p3ptrs.Ns[i-1];
        size_t AuMN = p3ptrs.Ms[i-1] * p3ptrs.Ns[i-1];
        p3ptrs.Abp[i] = p3ptrs.Abp[i-1] + AuMK;
        p3ptrs.xbp[i] = p3ptrs.xbp[i-1] + AuKN;
        p3ptrs.ybp[i] = p3ptrs.ybp[i-1] + AuMN;
    }

    // move data Au to memory buffer
    T *colptr = p3ptrs.A;
    T *dataauwalker = DataAu;
    for(int i=0; i<config.Mtg; i++)
    {
        for(int j=0; j<config.Ntg; j++){
            int currank = config.OrgRmat.GetElem(i,j);
            if(config.WorkRmat.GetElem(i,j) == config.OrgRmat.GetElem(i,j)){
                for(size_t k=0; k<currank*config.nb; k++){
                    *(colptr) = *(dataauwalker+k);
                    colptr++;
                }
            }
            dataauwalker += currank * config.nb;
        }
    }
}
template<typename T>
void TlrmvmBase<T>::Phase3CopyDataTranspose() {
    for(int i=1; i<p3transptrs.Ms.size(); i++){
        size_t AvMK = p3transptrs.Ms[i-1] * p3transptrs.Ks[i-1];
        size_t AvKN = p3transptrs.Ks[i-1] * p3transptrs.Ns[i-1];
        size_t AvMN = p3transptrs.Ms[i-1] * p3transptrs.Ns[i-1];
        p3transptrs.Abp[i] = p3transptrs.Abp[i-1] + AvMK;
        p3transptrs.xbp[i] = p3transptrs.xbp[i-1] + AvKN;
        p3transptrs.ybp[i] = p3transptrs.ybp[i-1] + AvMN;
    }
}
template<typename T>
void TlrmvmBase<T>::Phase1(){
    #pragma omp parallel for default(none)
    for(int i=0; i<p1ptrs.Ms.size(); i++){
        if(p1ptrs.Ms[i] != 0){
            cblasgemv(CblasColMajor, CblasNoTrans, p1ptrs.Ms[i],
                      p1ptrs.Ks[i], alpha, p1ptrs.Abp[i],
                      p1ptrs.Ms[i], p1ptrs.xbp[i],
                      1, beta, p1ptrs.ybp[i], 1);
        }
    }
}

template<typename T>
void TlrmvmBase<T>::Phase1Transpose(){
#pragma omp parallel for default(none)
    for(int i=0; i<p1transptrs.Ms.size(); i++){
        if(p1transptrs.Ms[i] != 0){
            cblasgemv(CblasColMajor, CblasTrans, p1transptrs.Ks[i],
                      p1transptrs.Ms[i], alpha, p1transptrs.Abp[i],
                      p1transptrs.Ks[i], // leading dimension of A in phase 3
                      p1transptrs.xbp[i],
                      1, beta, p1transptrs.ybp[i], 1);
        }
    }
}

template<typename T>
void TlrmvmBase<T>::Phase2(){
    #pragma omp parallel for default(none)
    for(int i=0; i<config.workmatgranksum; i++){
        p3ptrs.x[h_phase2mapping[i]] = p1ptrs.y[i];
    }
}
template<typename T>
void TlrmvmBase<T>::Phase2Transpose() {
#pragma omp parallel for default(none)
    for(int i=0; i<config.workmatgranksum; i++){
        p3transptrs.x[h_phase2mappingTranspose[i]] = p1transptrs.y[i];
    }
}

template<typename T>
void TlrmvmBase<T>::Phase3(){
#pragma omp parallel for default(none)
    for(int i=0; i<p3ptrs.Ms.size(); i++){
        if(p3ptrs.Ks[i] != 0){
            cblasgemv(CblasColMajor, CblasNoTrans, p3ptrs.Ms[i],
                      p3ptrs.Ks[i], alpha, p3ptrs.Abp[i],
                      p3ptrs.Ms[i], p3ptrs.xbp[i],
                      1, beta, p3ptrs.ybp[i], 1);
        }
    }
}

template<typename T>
void TlrmvmBase<T>::Phase3Transpose(){
#pragma omp parallel for default(none)
    for(int i=0; i<p3transptrs.Ms.size(); i++){
        if(p3transptrs.Ks[i] != 0){
            cblasgemv(CblasColMajor, CblasTrans, p3transptrs.Ks[i],
                      p3transptrs.Ms[i], alpha, p3transptrs.Abp[i],
                      p3transptrs.Ks[i], // leading dimension of A in phase 1
                      p3transptrs.xbp[i],
                      1, beta, p3transptrs.ybp[i], 1);
        }
    }
}

template<typename T>
void TlrmvmBase<T>::MVM(){
    if(transpose){
        MVMTranspose();
    }else{
        MVMNoTranspose();
    }
}

template<typename T>
void TlrmvmBase<T>::MVMNoTranspose() {
#pragma omp parallel for default(none)
    for(int i=0; i<p1ptrs.Ms.size(); i++){
        if(p1ptrs.Ms[i] != 0){
            cblasgemv(CblasColMajor, CblasNoTrans, p1ptrs.Ms[i],
                      p1ptrs.Ks[i], alpha, p1ptrs.Abp[i],
                      p1ptrs.Ms[i], p1ptrs.xbp[i],
                      1, beta, p1ptrs.ybp[i], 1);
        }
    }

#pragma omp parallel for default(none)
    for(int i=0; i<config.workmatgranksum; i++){
        p3ptrs.x[h_phase2mapping[i]] = p1ptrs.y[i];
    }

#pragma omp parallel for default(none)
    for(int i=0; i<p3ptrs.Ms.size(); i++){
        if(p3ptrs.Ms[i] != 0){
            cblasgemv(CblasColMajor, CblasNoTrans, p3ptrs.Ms[i],
                      p3ptrs.Ks[i], alpha, p3ptrs.Abp[i],
                      p3ptrs.Ms[i], p3ptrs.xbp[i],
                      1, beta, p3ptrs.ybp[i], 1);
        }
    }
}

template<typename T>
void TlrmvmBase<T>::MVMTranspose() {
#pragma omp parallel for default(none)
    for(int i=0; i<p1transptrs.Ms.size(); i++){
        if(p1transptrs.Ms[i] != 0){
            cblasgemv(CblasColMajor, CblasTrans, p1transptrs.Ks[i],
                      p1transptrs.Ms[i], alpha, p1transptrs.Abp[i],
                      p1transptrs.Ks[i], // leading dimension of A in phase 3
                      p1transptrs.xbp[i],
                      1, beta, p1transptrs.ybp[i], 1);
        }
    }

#pragma omp parallel for default(none)
    for(int i=0; i<config.workmatgranksum; i++){
        p3transptrs.x[h_phase2mappingTranspose[i]] = p1transptrs.y[i];
    }
#pragma omp parallel for default(none)
    for(int i=0; i<p3transptrs.Ms.size(); i++){
        if(p3transptrs.Ms[i] != 0){
            cblasgemv(CblasColMajor, CblasTrans, p3transptrs.Ks[i],
                      p3transptrs.Ms[i], alpha, p3transptrs.Abp[i],
                      p3transptrs.Ks[i], // leading dimension of A in phase 1
                      p3transptrs.xbp[i],
                      1, beta, p3transptrs.ybp[i], 1);
        }
    }
}

template<typename T>
void TlrmvmBase<T>::MPIMVM(){
    Phase1();
    Phase2();
    Phase3();
}

template<typename T>
void TlrmvmBase<T>::TryConjugateXvec() {
    if(!conjugate) return;
    // no transpose logic, input is same
    T * xwalkptr = this->xmat.RawPtr();
    size_t offset = 0;
    for(int i=0; i<p1ptrs.Ms.size(); i++){
        for(int j=0; j<config.nb; j++){
            *(p1ptrs.x + offset + j) = ElementwiseConjugate(*(xwalkptr + i*config.nb + j));
        }
        offset += config.nb;
    }
}

template<typename T>
void TlrmvmBase<T>::TryConjugateResults() {
    if(!conjugate) return;
    if(transpose){
#pragma omp parallel for default(none)
        for(int i=0; i<config.originM; i++){
            T cur = p3transptrs.y[i];
            finalresults[i] = ElementwiseConjugate(cur);
        }
    }else{
#pragma omp parallel for default(none)
        for(int i=0; i<config.originM; i++){
            T cur = p3ptrs.y[i];
            finalresults[i] = ElementwiseConjugate(cur);
        }
    }
}

template<typename T>
void TlrmvmBase<T>::CopyToFinalresults() {
    if(transpose){
        for(int i=0; i<config.originN; i++) finalresults[i] = p3transptrs.y[i];
    }else{
        for(int i=0; i<config.originM; i++) finalresults[i] = p3ptrs.y[i];
    }
}

template class TlrmvmBase<float>;
template class TlrmvmBase<complex<float>>;
template class TlrmvmBase<double>;
template class TlrmvmBase<complex<double>>;


template<typename T>
TlrmvmCPU<T>::TlrmvmCPU(TlrmvmConfig tlrmvmconfig):TlrmvmBase<T>::TlrmvmBase(tlrmvmconfig)
{}

template<typename T>
TlrmvmCPU<T>::TlrmvmCPU(){}

template<typename T>
void TlrmvmCPU<T>::AllocatePhase1Buffer(){
    // host memory  phase 1
    GetHostMemory(&p1ptrs.A, p1ptrs.Acnt);
    GetHostMemory(&p1ptrs.x, p1ptrs.Xcnt);
    GetHostMemory(&p1ptrs.y, p1ptrs.Ycnt);
    p1ptrs.Abp[0] = p1ptrs.A;
    p1ptrs.xbp[0] = p1ptrs.x;
    p1ptrs.ybp[0] = p1ptrs.y;
}

template<typename T>
void TlrmvmCPU<T>::AllocatePhase1BufferTranspose(){
    // host memory  phase 1
    p1transptrs.A = p3ptrs.A;
    p1transptrs.x = p1ptrs.x;
    GetHostMemory(&p1transptrs.y, p1transptrs.Ycnt);
    p1transptrs.Abp[0] = p3ptrs.A; // use phase 3, U bases
    p1transptrs.xbp[0] = p1ptrs.x; // use phase 1, x
    p1transptrs.ybp[0] = p1transptrs.y; // create a new buffer
}

template<typename T>
void TlrmvmCPU<T>::AllocatePhase3Buffer(){
    // host memory  phase 3
    GetHostMemory(&p3ptrs.A, p3ptrs.Acnt);
    GetHostMemory(&p3ptrs.x, p3ptrs.Xcnt);
    GetHostMemory(&p3ptrs.y, p3ptrs.Ycnt);
    p3ptrs.Abp[0] = p3ptrs.A;
    p3ptrs.xbp[0] = p3ptrs.x;
    p3ptrs.ybp[0] = p3ptrs.y;
}

template<typename T>
void TlrmvmCPU<T>::AllocatePhase3BufferTranspose(){
    // host memory  phase 3
    p3transptrs.A = p1ptrs.A;
    p3transptrs.x = p3ptrs.x;
    GetHostMemory(&p3transptrs.y, p3transptrs.Ycnt);
    p3transptrs.Abp[0] = p1ptrs.A; // use phase 1, V bases
    p3transptrs.xbp[0] = p3ptrs.x; // use phase 3, x
    p3transptrs.ybp[0] = p3transptrs.y; // create a new buffer
}

template<typename T>
void TlrmvmCPU<T>::MemoryInit(){
    InitData();
    Phase1GetMembuffer();
    AllocatePhase1Buffer();
    Phase1CopyData();
    Phase2Prepare();
    Phase3GetMembuffer();
    AllocatePhase3Buffer();
    Phase3CopyData();
    FreeData(); // release raw data buffer
    Phase1GetMembufferTranspose();
    AllocatePhase1BufferTranspose();
    Phase1CopyDataTranspose();
    Phase2PrepareTranspose();
    Phase3GetMembufferTranspose();
    AllocatePhase3BufferTranspose();
    Phase3CopyDataTranspose();
}

template<typename T>
void TlrmvmCPU<T>::MemoryFree(){
    FreeHostMemory(finalresults);
    // phase 1 pointer
    FreeHostMemory(p1ptrs.A);
    FreeHostMemory(p1ptrs.x);
    FreeHostMemory(p1ptrs.y);
    FreeHostMemory(p1ptrs.Abp);
    FreeHostMemory(p1ptrs.xbp);
    FreeHostMemory(p1ptrs.ybp);
    // phase 3 pointer
    FreeHostMemory(p3ptrs.A);
    FreeHostMemory(p3ptrs.x);
    FreeHostMemory(p3ptrs.y);
    FreeHostMemory(p3ptrs.Abp);
    FreeHostMemory(p3ptrs.xbp);
    FreeHostMemory(p3ptrs.ybp);
    // phase 1 transpose pointer
    FreeHostMemory(p1transptrs.y);
    // phase 1 transpose pointer
    FreeHostMemory(p3transptrs.y);
}

template class TlrmvmCPU<float>;
template class TlrmvmCPU<complex<float>>;
template class TlrmvmCPU<double>;
template class TlrmvmCPU<complex<double>>;

template<typename T>
BatchTlrmvmCPU<T>::BatchTlrmvmCPU(vector<TlrmvmConfig> tlrmvmvec) {
    cpuinst.resize(tlrmvmvec.size());
    for(int i=0; i<tlrmvmvec.size(); i++){
        cpuinst[i].UpdateConfig(tlrmvmvec[i]);
    }
}

template<typename T>
void BatchTlrmvmCPU<T>::MemoryInit() {
    for(auto &x : cpuinst) x.MemoryInit();
}

template<typename T>
void BatchTlrmvmCPU<T>::MemoryFree() {
    for(auto &x : cpuinst) x.MemoryFree();
}
template class BatchTlrmvmCPU<float>;
template class BatchTlrmvmCPU<complex<float>>;
template class BatchTlrmvmCPU<double>;
template class BatchTlrmvmCPU<complex<double>>;

template<typename T>
size_t TLRMVMBytesProcessed(size_t granksum, size_t nb, size_t M, size_t N){
    // phase 1
    unsigned long int phase1 = granksum*nb + N + granksum;
    // phase 2
    unsigned long int shuffle = 2 * granksum;
    // phase 3
    unsigned long int phase2 = granksum*nb + granksum + M;
    return sizeof(T) * (phase1 + shuffle + phase2);
}
template size_t TLRMVMBytesProcessed<float>(size_t, size_t, size_t, size_t);
template size_t TLRMVMBytesProcessed<complex<float>>(size_t, size_t, size_t, size_t);



