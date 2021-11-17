#ifndef PCMATRIX_H
#define PCMATRIX_H

#include <vector>
#include <string>

#include "common/Common.h"

using tlrmat::Matrix;
using namespace std;
namespace tlrmvm
{

/**
 * @brief A Proof of Concent Matrix class for TLRMVM.
 * The goal of matrix is to do Error Check for TLRMVM.
 * This is a virtual class, 
 * one should create a children class for each application.
 * 
 * @tparam T 
 */
template<typename T>
class PCMatrix{
public:
    PCMatrix(string datafolder, string acc, int nb, int originM, int originN);
    virtual void LoadData() = 0;
    void RecoverDense();
    Matrix<T> GetUTile(int i, int j) { return Utilemat[i][j]; }
    Matrix<T> GetVTile(int i, int j) { return Vtilemat[i][j]; }
    Matrix<T> GetMiddley(int i, int j) { return middiley[i][j]; }
    Matrix<T> GetDense();
    Matrix<int> GetR() { return Rmat; }
    Matrix<T> MVM(const Matrix<T>& rhs);
    virtual Matrix<T> GetX() = 0;
    Matrix<T> GetXtile(int i) {return Xvec[i];}
    void setX(Matrix<T>& xvec);
    Matrix<T> Phase1();
    Matrix<T> Phase2();
    Matrix<T> Phase3();
    void Tofile();
    virtual double GetDifference(Matrix<T>& m1, Matrix<T>& m2, size_t idx) = 0;
    
    string datafolder;
    string acc;
    size_t nb;
    size_t originM;
    size_t originN;
    size_t paddingM;
    size_t paddingN;
    size_t Mtglobal;
    size_t Ntglobal;

protected:

    void BuildTLRMatrices(Matrix<T>& Umat, Matrix<T>& Vmat);
    Matrix<int> Vprefixmat;
    Matrix<int> Uprefixmat;
    vector<Matrix<T>> Xvec;
    vector<vector<Matrix<T>>> middiley;
    // vector<int> Vprefixsum;
    // vector<int> Uprefixsum;
    Matrix<int> Rmat;
    vector<vector<Matrix<T>>> Utilemat;
    vector<vector<Matrix<T>>> Vtilemat;
    vector<vector<Matrix<T>>> Densemat;
};



typedef PCMatrix<float> FPCMAT;
typedef PCMatrix<complex<float>> CFPCMAT;

/**
 * @brief FPPCMatrix
 * 
 */
class FPPCMatrix : public FPCMAT
{
public:
    FPPCMatrix(string datafolder, string acc, int nb, string id, int originM, int originN);
    void LoadData();
    double GetDifference(Matrix<float>& m1, Matrix<float>& m2, size_t idx);
    Matrix<float> GetX();
    
    string id;
    string problemname;
    using FPCMAT::datafolder;
    using FPCMAT::acc;
    using FPCMAT::nb;
    using FPCMAT::originM;
    using FPCMAT::originN;
    using FPCMAT::paddingM;
    using FPCMAT::paddingN;
    using FPCMAT::Mtglobal;
    using FPCMAT::Ntglobal;
private:
    using FPCMAT::BuildTLRMatrices;    
    using FPCMAT::Rmat;
    using FPCMAT::Utilemat;
    using FPCMAT::Vtilemat;
    using FPCMAT::Densemat;  
};

/**
 * @brief CFPPCMatrix
 * 
 */
class CFPPCMatrix : public CFPCMAT
{
public:
    CFPPCMatrix(string datafolder, string acc, int nb, string problemname, int originM, int originN);
    void LoadData();
    double GetDifference(Matrix<complex<float>>& m1, Matrix<complex<float>>& m2, size_t idx);
    Matrix<complex<float>> GetX();
    Matrix<float> GetReal(Matrix<complex<float>> &tile);
    Matrix<float> GetImag(Matrix<complex<float>> &tile);
    int id;
    string problemname;
    using CFPCMAT::datafolder;
    using CFPCMAT::acc;
    using CFPCMAT::nb;
    using CFPCMAT::originM;
    using CFPCMAT::originN;
    using CFPCMAT::paddingM;
    using CFPCMAT::paddingN;
    using CFPCMAT::Mtglobal;
    using CFPCMAT::Ntglobal;
private:
    using CFPCMAT::BuildTLRMatrices;    
    using CFPCMAT::Rmat;
    using CFPCMAT::Utilemat;
    using CFPCMAT::Vtilemat;
    using CFPCMAT::Densemat;  
};

} // namespace tlrmvm



#endif 


