#include "TlrmvmCPU.h"
#include "common/Common.h"
#include "common/AppUtil.h"
#include <memory.h>
#include <cassert>
#ifdef USE_MPI
#include <mpi.h>
#endif 

using namespace tlrmat;

namespace tlrmvm{

int CalculatePadding(int originDim, int nb){
    return ( originDim / nb + (originDim % nb != 0) ) * nb;
}

TlrmvmConfig::TlrmvmConfig(int originM, int originN, int nb, 
string datafolder, string acc, string problemname)
:datafolder(datafolder), acc(acc), problemname(problemname),
originM(originM), originN(originN), nb(nb)
{
    Mtg = CalculatePadding(originM, nb) / nb;
    Ntg = CalculatePadding(originN, nb) / nb;
}

void initalphabeta(float &alpha, float &beta){
    alpha = (float)1.0;
    beta = (float)0.0;
}
void initalphabeta(complex<float> &alpha, complex<float> &beta){
    alpha = complex<float>(1.0,0.0);
    beta = complex<float>(0.0,0.0);
}
void initalphabeta(double &alpha, double &beta){
    alpha = (double)1.0;
    beta = (double)0.0;
}
void initalphabeta(complex<double> &alpha, complex<double> &beta){
    alpha = complex<double>(1.0,0.0);
    beta = complex<double>(0.0,0.0);
}

template<typename T>
TlrmvmBase<T>::TlrmvmBase(TlrmvmConfig tlrmvmconfig)
:tlrmvmconfig(tlrmvmconfig)
{
    originM = tlrmvmconfig.originM;
    originN = tlrmvmconfig.originN;
    nb = tlrmvmconfig.nb;
    paddingM = CalculatePadding(originM, nb);
    paddingN = CalculatePadding(originN, nb);
    Ntg = paddingN / nb;
    Mtg = paddingM / nb;
    // load Rmat
    char filename[200];
    sprintf(filename, "%s/%s_Rmat_nb%d_acc%s.bin", 
    tlrmvmconfig.datafolder.c_str(), tlrmvmconfig.problemname.c_str(),
    tlrmvmconfig.nb, tlrmvmconfig.acc.c_str());
    OrgRmat = Matrix<int>::Fromfile(filename, Mtg, Ntg);
    granksum = OrgRmat.Sum();
    Maskmat = tlrmvmconfig.Maskmat;
    WorkRmat = OrgRmat.ApplyMask(Maskmat);
    colsum = WorkRmat.ColSum();
    initalphabeta(alpha, beta);
}

template<typename T>
void TlrmvmBase<T>::InitData(){
    char filename[200];
    sprintf(filename, "%s/%s_Ubases_nb%d_acc%s.bin", 
    tlrmvmconfig.datafolder.c_str(), tlrmvmconfig.problemname.c_str(), 
    tlrmvmconfig.nb, tlrmvmconfig.acc.c_str());
    size_t elems = granksum * tlrmvmconfig.nb;
    LoadBinary(filename, &DataAu, elems);
    sprintf(filename, "%s/%s_Vbases_nb%d_acc%s.bin", 
    tlrmvmconfig.datafolder.c_str(), tlrmvmconfig.problemname.c_str(), 
    tlrmvmconfig.nb, tlrmvmconfig.acc.c_str());
    LoadBinary(filename, &DataAv, elems);
    T * tmpdatax;
    sprintf(filename, "%s/%s_x.bin", tlrmvmconfig.datafolder.c_str(), 
    tlrmvmconfig.problemname.c_str());
    LoadBinary(filename, &tmpdatax, tlrmvmconfig.originN);
    Datax = new T[tlrmvmconfig.Ntg * nb];
    memset(Datax, 0, sizeof(T) * tlrmvmconfig.Ntg * nb);
    for(int i=0; i<tlrmvmconfig.originN; i++){
        Datax[i] = tmpdatax[i];
    }
    delete[] tmpdatax;
    this->xmat = Matrix<T>(Datax, paddingN, 1);
}

template<typename T>
void TlrmvmBase<T>::setX(T * xvector, size_t xlength){
    assert(xlength == this->originN);
    memset(Datax, 0, sizeof(T) * this->paddingN);
    CopyData(Datax, xvector, xlength);
    this->xmat = Matrix<T>(Datax, paddingN, 1);
}

template<typename T>
void TlrmvmBase<T>::Phase1GetMembuffer(){
    GetHostMemory(&h_Avbp, Ntg);
    GetHostMemory(&h_xbp, Ntg);
    GetHostMemory(&h_yvbp, Ntg);

    for(int i=0; i<Ntg; i++){
        AvMs.push_back(colsum[i]);
        AvKs.push_back(nb);
        AvNs.push_back(1);
    }
    workmatgranksum = WorkRmat.Sum();

    phase1Acnt = 0;
    phase1Xcnt = 0;
    phase1Ycnt = 0;

    for(int i=0; i<Ntg; i++){
        phase1Acnt += AvMs[i] * AvKs[i];
        phase1Xcnt += AvKs[i] * AvNs[i];
        phase1Ycnt += AvMs[i] * AvNs[i];
    }
}

template<typename T>
void TlrmvmBase<T>::Phase1CopyData(){
    for(int i=1; i<Ntg; i++){
        size_t AvMK = AvMs[i-1] * AvKs[i-1];
        size_t AvKN = AvKs[i-1] * AvNs[i-1];
        size_t AvMN = AvMs[i-1] * AvNs[i-1];
        h_Avbp[i] = h_Avbp[i-1] + AvMK;
        h_xbp[i] = h_xbp[i-1] + AvKN;
        h_yvbp[i] = h_yvbp[i-1] + AvMN;
    }

    
    // move data from DataAv to Av
    gcolsum = OrgRmat.ColSum();
    T *Avwalkptr = DataAv;
    for(int i=0; i<Ntg; i++){
        // column start pointers
        T *colptr = h_Avbp[i];
        size_t lda = gcolsum[i];
        for(int nbi = 0; nbi < nb; nbi++){
            for(int j=0; j < Mtg; j++){
                int currank = OrgRmat.GetElem(j,i);
               if(WorkRmat.GetElem(j,i) == OrgRmat.GetElem(j,i)){
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
    T * xwalkptr = Datax;
    size_t offset = 0;
    for(int i=0; i<Ntg; i++){
        for(int j=0; j<nb; j++){
            *(h_x + offset + j) = *(xwalkptr + i*nb + j);
        }
        offset += nb;
    }
}


template<typename T>
void TlrmvmBase<T>::Phase2Prepare(){
    // phase 2
    vector<vector<vector<vector<int>>>> phase2record;
    phase2record.resize(Mtg, vector<vector<vector<int>>>()); // Mtg row
    for(int i=0; i<Mtg; i++) phase2record[i].resize(Ntg, vector<vector<int>>()); // Ntg col
    for(int i=0; i<Mtg; i++){
        for(int j=0; j<Ntg; j++){
            phase2record[i][j].resize(2, vector<int>());
        }
    }
    size_t p2walker = 0;
    for(int i=0; i<Mtg; i++){
        for(int j=0; j<Ntg; j++){
            if(WorkRmat.GetElem(i,j) != 0){
                int currank = WorkRmat.GetElem(i,j);
                for(int k=0; k<currank; k++){
                    phase2record[i][j][0].push_back(p2walker++);
                }
            }
        }
    }
    // unfold
    for(int i=0; i<Ntg; i++){
        for(int j=0; j<Mtg; j++){
            if(WorkRmat.GetElem(j,i) != 0){
                int currank = WorkRmat.GetElem(j,i);
                for(int k=0; k<currank; k++){
                    h_phase2mapping.push_back(phase2record[j][i][0][k]);
                }
            }
        }
    }
}

template<typename T>
void TlrmvmBase<T>::Phase3GetMembuffer(){
    GetHostMemory(&h_Aubp, Mtg);
    GetHostMemory(&h_yubp, Mtg);
    GetHostMemory(&h_ybp, Mtg);
    GetHostMemory(&h_youtbp, Mtg);
    // phase 3
    rowsum = WorkRmat.RowSum();
    for(int i=0; i<Mtg; i++){
        AuMs.push_back(nb);
        AuKs.push_back(rowsum[i]);
        AuNs.push_back(1);
    }
    
    phase3Acnt = 0;
    phase3Xcnt = 0;
    phase3Ycnt = 0;
    for(int i=0; i<AuMs.size(); i++){
        phase3Acnt += AuMs[i] * AuKs[i];
        phase3Xcnt += AuKs[i] * AuNs[i];
        phase3Ycnt += AuMs[i] * AuNs[i];
    }
}

template<typename T>
void TlrmvmBase<T>::Phase3CopyData(){
    for(int i=1; i<Mtg; i++){
        size_t AuMK = AuMs[i-1] * AuKs[i-1];
        size_t AuKN = AuKs[i-1] * AuNs[i-1];
        size_t AuMN = AuMs[i-1] * AuNs[i-1];

        h_Aubp[i] = h_Aubp[i-1] + AuMK;
        h_yubp[i] = h_yubp[i-1] + AuKN;
        h_ybp[i] = h_ybp[i-1] + AuMN;
        h_youtbp[i] = h_youtbp[i-1] + AuMN;
    }

    // move data Au to memory buffer
    T *colptr = h_Au;
    T *dataauwalker = DataAu;
    for(int i=0; i<Mtg; i++)
    {
        for(int j=0; j<Ntg; j++){
            int currank = OrgRmat.GetElem(i,j);
            if(Maskmat.GetElem(i, j) == 1){
                for(size_t k=0; k<currank*nb; k++){
                    *(colptr) = *(dataauwalker+k);
                    colptr++;
                }
            }
            dataauwalker += currank * nb;
        }
    }
}

template<typename T>
void TlrmvmBase<T>::Phase1(){
    #pragma omp parallel for 
    for(int i=0; i<Ntg; i++){
        if(AvMs[i] != 0){
            cblasgemv(CblasColMajor, CblasNoTrans, AvMs[i],
            AvKs[i], alpha, h_Avbp[i], 
            AvMs[i], h_xbp[i],
            1, beta, h_yvbp[i], 1);
        }
    }
}

template<typename T>
void TlrmvmBase<T>::Phase2(){
    #pragma omp parallel for 
    for(int i=0; i<workmatgranksum; i++){
        h_yu[h_phase2mapping[i]] = h_yv[i];
    }
}

template<typename T>
void TlrmvmBase<T>::Phase3(){
    #pragma omp parallel for 
    for(int i=0; i<Mtg; i++){
        if(AuMs[i] != 0){
            cblasgemv(CblasColMajor, CblasNoTrans, AuMs[i],
            AuKs[i], alpha, h_Aubp[i], 
            AuMs[i], h_yubp[i],
            1, beta, h_ybp[i], 1);
        }
    }

}

template<typename T>
void TlrmvmBase<T>::MVM(){
    Phase1();
    Phase2();
    Phase3();
}

template<typename T>
void TlrmvmBase<T>::MPIMVM(){
    Phase1();
    Phase2();
    Phase3();
}

template class TlrmvmBase<float>;
template class TlrmvmBase<complex<float>>;
template class TlrmvmBase<double>;
template class TlrmvmBase<complex<double>>;


template<typename T>
TlrmvmCPU<T>::TlrmvmCPU(TlrmvmConfig tlrmvmconfig)
:TlrmvmBase<T>::TlrmvmBase(tlrmvmconfig)
{}

template<typename T>
void TlrmvmCPU<T>::FreeData(){
    if(DataAv) delete[] DataAv;
    if(DataAu) delete[] DataAu;
    if(Datax) delete[] Datax;
}

template<typename T>
void TlrmvmCPU<T>::AllocatePhase1Buffer(){
    // host memory  phase 1
    GetHostMemory(&h_Av, phase1Acnt);
    GetHostMemory(&h_x, phase1Xcnt);
    GetHostMemory(&h_yv, phase1Ycnt);
    h_xbp[0] = h_x;
    h_Avbp[0] = h_Av;
    h_yvbp[0] = h_yv;
}

template<typename T>
void TlrmvmCPU<T>::AllocatePhase3Buffer(){
    // host memory  phase 3
    GetHostMemory(&h_Au, phase3Acnt);
    GetHostMemory(&h_yu, phase3Xcnt);
    GetHostMemory(&h_y, phase3Ycnt);
    GetHostMemory(&h_yout, phase3Ycnt);
    h_Aubp[0] = h_Au;
    h_yubp[0] = h_yu;
    h_ybp[0] = h_y;
    h_youtbp[0] = h_yout;
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
    FreeData();
}


template<typename T>
void TlrmvmCPU<T>::MemoryFree(){
    // here we free tlrmvm buffer
    FreeHostMemory(h_Av);
    FreeHostMemory(h_x);
    FreeHostMemory(h_yv);
    FreeHostMemory(h_Avbp);
    FreeHostMemory(h_xbp);
    FreeHostMemory(h_yvbp);
    FreeHostMemory(h_Au);
    FreeHostMemory(h_yu);
    FreeHostMemory(h_y);
    FreeHostMemory(h_yout);
    FreeHostMemory(h_Aubp);
    FreeHostMemory(h_yubp);
    FreeHostMemory(h_ybp);
    FreeHostMemory(h_youtbp);

}

template class TlrmvmCPU<float>;
template class TlrmvmCPU<complex<float>>;
template class TlrmvmCPU<double>;
template class TlrmvmCPU<complex<double>>;


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

} // namespace tlrmvm


