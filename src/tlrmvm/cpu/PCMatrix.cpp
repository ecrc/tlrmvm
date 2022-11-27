//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#include <cassert>
#include <string>
#include <vector>
#include <cstring>

#include "../../common/Common.hpp"
#include "../../common/AppUtil.hpp"
#include "TlrmvmCPU.hpp"
#include "PCMatrix.hpp"

using namespace std;

template<typename T>
PCMatrix<T>::PCMatrix(string datafolder, string acc, int nb, int originM, int originN)
:isconstrank(false), datafolder(datafolder), acc(acc),nb(nb), originM(originM), originN(originN){
    paddingM = CalculatePadding(originM, nb);
    paddingN = CalculatePadding(originN, nb);
    Mtglobal = paddingM / nb;
    Ntglobal = paddingN / nb;
    maskmat = Matrix<int>(Mtglobal,Ntglobal);
    for(int i=0; i<Mtglobal * Ntglobal; i++) maskmat.RawPtr()[i] = 1;
}

template<typename T>
PCMatrix<T>::PCMatrix(string acc, int nb, int constranksize, int originM, int originN)
        :acc(acc),nb(nb), isconstrank(true),constranksize(constranksize),originM(originM), originN(originN){
    paddingM = CalculatePadding(originM, nb);
    paddingN = CalculatePadding(originN, nb);
    Mtglobal = paddingM / nb;
    Ntglobal = paddingN / nb;
    maskmat = Matrix<int>(Mtglobal,Ntglobal);
    for(int i=0; i<Mtglobal * Ntglobal; i++) maskmat.RawPtr()[i] = 1;
}

template <typename T>
void PCMatrix<T>::BuildTLRMatrices(Matrix<T>& Umat, Matrix<T>& Vmat){
    Utilemat.resize(Mtglobal, vector<Matrix<T>>(Ntglobal, Matrix<T>()));
    Vtilemat.resize(Mtglobal, vector<Matrix<T>>(Ntglobal, Matrix<T>()));
    // copy Av
    size_t prev = 0;
    size_t post = 0;
    vector<int> rowsum = Rmat.RowSum();
    vector<int> colsum = Rmat.ColSum();
    for(int i=0; i<Ntglobal; i++){
        post += colsum[i];
        size_t innerprev = 0;
        size_t innerpost = 0;
        Matrix<T> curVblock(colsum[i], nb);
        memcpy(curVblock.RawPtr(), Vmat.RawPtr()+prev*nb, sizeof(T) * colsum[i] * nb);
        for(int j=0; j<Mtglobal; j++){
            innerpost += Rmat.GetElem(j,i);
            Vtilemat[j][i] = curVblock.Block({innerprev,innerpost},{0,(size_t)nb});
            innerprev += Rmat.GetElem(j,i);
        }
        prev += colsum[i];
    }
    // copy Au
    prev = 0;
    post = 0;
    for(int i=0; i<Mtglobal; i++){
        post += colsum[i];
        size_t innerprev = 0;
        size_t innerpost = 0;
        Matrix<T> curUblock(nb, rowsum[i]);
        memcpy(curUblock.RawPtr(), Umat.RawPtr()+prev*nb, sizeof(T) * rowsum[i] * nb);
        for(int j=0; j<Ntglobal; j++){
            innerpost += Rmat.GetElem(i,j);
            Utilemat[i][j] = curUblock.Block({0,(size_t)nb},{innerprev, innerpost});
            innerprev += Rmat.GetElem(i,j);
        }
        prev += rowsum[i];
    }
}

template<typename T>
void PCMatrix<T>::RecoverDense(){
    Densemat.clear();
    Densemat.resize(Mtglobal, vector<Matrix<T>>(Ntglobal, Matrix<T>()));
    for(int i=0; i<Mtglobal; i++){
        for(int j=0; j<Ntglobal; j++){
            Densemat[i][j] = Utilemat[i][j] * Vtilemat[i][j];
            if(maskmat.GetElem(i, j) == 0){
                Densemat[i][j].Fill(0.0);
            }
        }
    }
}

template<typename T>
Matrix<T> PCMatrix<T>::GetDense(){
    if(Densemat.size() != Mtglobal || Densemat[0].size() != Ntglobal){
        RecoverDense();
    }
    size_t totalelems = (size_t)Mtglobal * (size_t) Ntglobal * nb * nb;
    vector<T> bigvec(totalelems, 0);
    size_t Biglda = Mtglobal * nb;
    for(int i=0; i<Mtglobal; i++){
        for(int j=0; j<Ntglobal; j++){
            Matrix<T> curblock = Densemat[i][j];
            for(int p=0; p<nb; p++){
                for(int q=0; q<nb; q++){
                    bigvec[(j * nb + p) * Biglda + i * nb + q] = curblock.RawPtr()[p*nb+q];
                }
            }
        }
    }
    return Matrix<T>(bigvec, Mtglobal *nb , Ntglobal * nb);
}

template<typename T>
Matrix<T> PCMatrix<T>::MVM(const Matrix<T> & rhs){
    return GetDense() * rhs;
}

template <typename T>
void PCMatrix<T>::setX(Matrix<T>& xvec){
    assert(xvec.Col() == 1);
    Xvec.clear();
    Xvec.resize(Ntglobal, Matrix<T>());
    assert(xvec.Row() * xvec.Col() == paddingN);
    for(int i=0; i<Ntglobal; i++){
        Xvec[i] = xvec.Block({i*nb,(i+1)*nb},{0,1});
    }
}

template <typename T>
Matrix<T> PCMatrix<T>::Phase1(){
    middiley.clear();
    middiley.resize(Mtglobal, vector<Matrix<T>>(Ntglobal, Matrix<T>()));
    for(int i=0; i<Mtglobal; i++){
        for(int j=0; j<Ntglobal; j++){
            middiley[i][j] = Vtilemat[i][j] * Xvec[j];
            if(maskmat.GetElem(i,j) == 0) middiley[i][j].Fill(0.0);
        }
    }
    Matrix<T> rety(Rmat.Sum(), 1);
    size_t offset = 0;
    for(int i=0; i<Ntglobal; i++){
        for(int j=0; j<Mtglobal; j++){
            memcpy(rety.RawPtr() + offset, middiley[j][i].RawPtr(), Rmat.GetElem(j,i) * sizeof(T));
            offset += Rmat.GetElem(j,i);
        }
    }
    return rety;
}

template <typename T>
Matrix<T> PCMatrix<T>::Phase1Transpose(){
    middiley.clear();
    middiley.resize(Mtglobal, vector<Matrix<T>>(Ntglobal, Matrix<T>()));
    for(int i=0; i<Mtglobal; i++){
        for(int j=0; j<Ntglobal; j++){
            middiley[i][j] = Utilemat[j][i].Transpose() * Xvec[j];
            if(maskmat.GetElem(i,j) == 0) middiley[i][j].Fill(0.0);
        }
    }
    Matrix<T> rety(Rmat.Sum(), 1);
    size_t offset = 0;
    for(int i=0; i<Ntglobal; i++){
        for(int j=0; j<Mtglobal; j++){
            memcpy(rety.RawPtr() + offset, middiley[j][i].RawPtr(),
                   Rmat.GetElem(i,j) * sizeof(T));
            offset += Rmat.GetElem(i,j);
        }
    }
    return rety;
}

template <typename T>
Matrix<T> PCMatrix<T>::Phase2(){
    assert(middiley.size() == Mtglobal && middiley[0].size() == Ntglobal);
    Matrix<T> rety(Rmat.Sum(), 1);
    size_t offset = 0;
    for(int i=0; i<Mtglobal; i++){
        for(int j=0; j<Ntglobal; j++){
            memcpy(rety.RawPtr() + offset, middiley[i][j].RawPtr(), Rmat.GetElem(i,j) * sizeof(T));
            offset += Rmat.GetElem(i,j);
        }
    }
    return rety;
}
template<typename T>
Matrix<T> PCMatrix<T>::Phase2Transpose() {
    assert(middiley.size() == Mtglobal && middiley[0].size() == Ntglobal);
    Matrix<T> rety(Rmat.Sum(), 1);
    size_t offset = 0;
    for(int i=0; i<Mtglobal; i++){
        for(int j=0; j<Ntglobal; j++){
            memcpy(rety.RawPtr() + offset, middiley[i][j].RawPtr(), Rmat.GetElem(j,i) * sizeof(T));
            offset += Rmat.GetElem(j,i);
        }
    }
    return rety;
}

template <typename T>
Matrix<T> PCMatrix<T>::Phase3(){
    Matrix<T> yout(paddingM, 1);
    memset(yout.RawPtr(), 0, sizeof(T) * paddingM);
    size_t rowoffset = 0;
    for(int i=0; i<Mtglobal; i++){
        Matrix<T> blocky(nb, 1);
        memset(blocky.RawPtr(), 0, sizeof(T) * nb);
        for(int j=0; j<Ntglobal; j++){
            blocky += Utilemat[i][j] * middiley[i][j];
        }
        memcpy(yout.RawPtr() + rowoffset, blocky.RawPtr(), nb * sizeof(T));
        rowoffset += nb;
    }
    return yout;
}

template <typename T>
Matrix<T> PCMatrix<T>::Phase3Transpose(){
    Matrix<T> yout(paddingN, 1);
    memset(yout.RawPtr(), 0, sizeof(T) * paddingN);
    size_t rowoffset = 0;
    for(int i=0; i<Mtglobal; i++){
        Matrix<T> blocky(nb, 1);
        memset(blocky.RawPtr(), 0, sizeof(T) * nb);
        for(int j=0; j<Ntglobal; j++){
            blocky += Vtilemat[j][i].Transpose() * middiley[i][j];
        }
        memcpy(yout.RawPtr() + rowoffset, blocky.RawPtr(), nb * sizeof(T));
        rowoffset += nb;
    }
    return yout;
}

template class PCMatrix<float>;
template class PCMatrix<complex<float>>;

/////////////////////////////////////////////////////////

FPPCMatrix::FPPCMatrix(string datafolder, 
string acc, int nb, string problemname, int originM, int originN)
:PCMatrix(datafolder, acc, nb, originM, originN), problemname(problemname){
    LoadData();
}



void FPPCMatrix::LoadData(){
    float *DataAv, *DataAu;
    size_t granksum;
    if(isconstrank){
        Rmat = Matrix<int>(Mtglobal, Ntglobal);
        Rmat.Fill(constranksize);
        granksum = Rmat.Sum();
        size_t elems = granksum * nb;
        DataAu = new float[elems];
        DataAv = new float[elems];
        for(int i=0; i<elems; i++){
            DataAu[i] = 0.001;
            DataAv[i] = 0.001;
        }
    }else{
        int *DataR;
        char filename[200];
        sprintf(filename, "%s/%s_Rmat_nb%d_acc%s.bin",
                datafolder.c_str(), problemname.c_str(), (int)nb, acc.c_str());
        LoadBinary(filename, &DataR, Mtglobal * Ntglobal);
        Rmat = Matrix<int>(DataR, Mtglobal, Ntglobal);
        delete[] DataR;
        granksum = Rmat.Sum();
        sprintf(filename, "%s/%s_Ubases_nb%d_acc%s.bin",
                datafolder.c_str(), problemname.c_str(), (int)nb, acc.c_str());
        size_t elems = granksum * nb;
        LoadBinary(filename, &DataAu, elems);
        sprintf(filename, "%s/%s_Vbases_nb%d_acc%s.bin",
                datafolder.c_str(), problemname.c_str(), (int)nb, acc.c_str());
        LoadBinary(filename, &DataAv, elems);

    }
    Matrix<float>Vmat(DataAv, granksum, nb);
    Matrix<float>Umat(DataAu, granksum, nb);
    BuildTLRMatrices(Umat, Vmat);
    delete[] DataAv;
    delete[] DataAu;
}

double FPPCMatrix::GetDifference(Matrix<float>& m1, Matrix<float>& m2, size_t idx){
    return fabs(m1.RawPtr()[idx]-m2.RawPtr()[idx]);
}

Matrix<float> FPPCMatrix::GetX(){
    Matrix<float> ret(paddingN, 1);
    #pragma omp parallel for
    for(int i=0; i<paddingN; i++) ret.RawPtr()[i] = float(rand()/RAND_MAX);
    setX(ret);
    return ret;
}

/////////////////////////////////////////////////////////


Matrix<complex<float>> CFPPCMatrix::GetX(){
    Matrix<complex<float>> ret(paddingN, 1);
    memset(ret.RawPtr(), 0, sizeof(complex<float>) * paddingN );
    complex<float> *tmpx;
    ReadSeismicBinaryX(datafolder, &tmpx, originN, acc, nb, id);
    CopyData(ret.RawPtr(), tmpx, originN);
    setX(ret);
    return ret;
}


CFPPCMatrix::CFPPCMatrix(string datafolder, string acc, int nb, string problem, int originM, int originN)
:PCMatrix(datafolder, acc, nb, originM, originN){
    this->problemname = problem;
    LoadData();
}

CFPPCMatrix::CFPPCMatrix(string acc, int nb, int constranksize, int originM, int originN)
        :PCMatrix(acc, nb, constranksize, originM, originN){
    LoadData();
}

void CFPPCMatrix::LoadData(){
    complex<float> *DataAv, *DataAu;
    size_t granksum;
    if(isconstrank){
        Rmat = Matrix<int>(Mtglobal, Ntglobal);
        Rmat.Fill(constranksize);
        granksum = Rmat.Sum();
        size_t elems = granksum * nb;
        DataAu = new complex<float>[elems];
        DataAv = new complex<float>[elems];
        for(int i=0; i<elems; i++){
            DataAu[i] = 0.001;
            DataAv[i] = 0.001;
        }
    }else{
        int *DataR;
        char filename[200];
        sprintf(filename, "%s/%s_Rmat_nb%d_acc%s.bin",
                datafolder.c_str(), this->problemname.c_str(), (int)nb, acc.c_str());
        LoadBinary(filename, &DataR, Mtglobal * Ntglobal);
        Rmat = Matrix<int>(DataR, Mtglobal, Ntglobal);
        granksum = Rmat.Sum();
        sprintf(filename, "%s/%s_Ubases_nb%d_acc%s.bin",
                datafolder.c_str(), problemname.c_str(), (int)nb, acc.c_str());
        size_t elems = granksum * nb;
        LoadBinary(filename, &DataAu, elems);
        sprintf(filename, "%s/%s_Vbases_nb%d_acc%s.bin",
                datafolder.c_str(), problemname.c_str(), (int)nb, acc.c_str());
        LoadBinary(filename, &DataAv, elems);
        delete[] DataR;
    }
    Matrix<complex<float>>Vmat(DataAv, granksum, nb);
    Matrix<complex<float>>Umat(DataAu, granksum, nb);
    BuildTLRMatrices(Umat, Vmat);
    delete[] DataAv;
    delete[] DataAu;
}

double CFPPCMatrix::GetDifference(Matrix<complex<float>>& m1, Matrix<complex<float>>& m2, size_t idx){
    auto tmp = m1.RawPtr()[idx]-m2.RawPtr()[idx];
    return sqrt(tmp.real()*tmp.real() + tmp.imag()*tmp.imag());
}




