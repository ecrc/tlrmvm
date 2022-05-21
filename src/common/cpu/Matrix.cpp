
#include <complex>
#include <cstring>
#include <cassert>
#include <iostream>
#include <iomanip>

#include "Matrix.hpp"
#include "BlasInterface.hpp"
#include "Util.hpp"
#include <fstream>

using std::cout;
using std::endl;
using std::complex;


/*******************
* Matrix Class
********************/
template <typename T>
Matrix<T>::Matrix()
{
    M = N = lda = 0;
}


template <typename T>
Matrix<T>::Matrix(size_t M, size_t N)
:M(M),N(N),lda(M)
{
    assert(lda == M);
    dataptr.resize(M*N,0);
}

template <typename T>
Matrix<T>::Matrix(const Matrix& rhs)
{
    M = rhs.Row();
    N = rhs.Col();
    lda = rhs.Lda();
    dataptr.resize(M*N,0);
    memcpy(dataptr.data(), rhs.RawPtr(), sizeof(T) * M * N);
}

template <typename T>
Matrix<T>::Matrix(T *inptr, size_t M, size_t N){
    this->M = M;
    this->N = N;
    this->lda = M;
    dataptr.resize(M*N,0);
    memcpy(dataptr.data(), inptr, sizeof(T) * M * N);
}
template<typename T>
Matrix<T>::Matrix(vector<T> invec, size_t M, size_t N)
{
    this->M = M;
    this->N = N;
    this->lda = M;
    assert(invec.size() == M * N);
    dataptr = invec;
}

template <typename T>
Matrix<T>::~Matrix(){}

// template <typename T>
// void Matrix<T>::CopyFromPointer(T * srcptr)
// {
//     memcpy(dataptr.data(), srcptr, sizeof(T) * M * N);
// }


template <typename T>
T Matrix<T>::Sum()
{
    T ret; memset(&ret, 0, sizeof(T));
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            ret += dataptr[i + j * lda];
        }
    }
    return ret;
}


size_t getabs(size_t val1, size_t val2){
    return fabs(val1 - val2);
}

double getabs(int val1, int val2){
    return fabs(val1 - val2);
}

double getabs(float val1, float val2){
    return fabs(val1-val2);
}

double getabs(double val1, double val2){
    return fabs(val1-val2);
}


double getabs(complex<float> val1, complex<float> val2){
    return (double)abs(val1 - val2);
}

double getabs(complex<double> val1, complex<double> val2){
    return (double)abs(val1 - val2);
}

size_t getabs(size_t val1){
    return (size_t)fabs(val1);
}

double getabs(int val1){
    return (double)fabs(val1);
}

double getabs(float val1){
    return (double)fabs(val1);
}

double getabs(double val1){
    return (double)fabs(val1);
}

double getabs(complex<float> val1){
    return (double)abs(val1);
}

double getabs(complex<double> val1){
    return (double)abs(val1);
}

double getmax(double v, double u){return fmax(v,u);}

double getmax(complex<double> v, complex<double> u) {return 0;}

double getmax(float v, float u){return fmax(v,u);}

double getmax(complex<float> v, complex<float> u) {return 0;}



template<typename T>
Matrix<T> Matrix<T>::Row(size_t idx){
    assert(idx < M);
    vector<T> rvec;
    for(size_t i=0; i<N; i++)
    rvec.push_back(dataptr[idx + i * M]);
    return Matrix<T>(rvec, 1, N);
}

template<typename T>
Matrix<T> Matrix<T>::Col(size_t idx){
    assert(idx < N);
    vector<T> rvec;
    for(size_t i=0; i<M; i++)
    rvec.push_back(dataptr[i + idx * M]);
    return Matrix<T>(rvec, M, 1);
}

template<typename T>
Matrix<T> Matrix<T>::Block(vector<size_t> rowidx, vector<size_t> colidx){
    vector<T> tmpdata;
    size_t newM = rowidx[1] - rowidx[0];
    size_t newN = colidx[1] - colidx[0];
    size_t ul_m = rowidx[0]; // upper left starting point - row
    size_t ul_n = colidx[0]; // upper left starting point - col
    assert(newM > 0 && newN > 0);
    assert(rowidx[0] >= 0 && rowidx[1] <= M);
    assert(colidx[0] >= 0 && colidx[1] <= N);
    tmpdata.resize(newM*newN, 0);
    for(size_t i=0; i<newN; i++){
        for(size_t j=0; j<newM; j++){
            tmpdata[j + i * newM] = dataptr[(ul_n + i) * M + (ul_m + j)];
        }
    }
    return Matrix<T>(tmpdata, newM, newN);
}

template<typename T>
Matrix<T> Matrix<T>::Block(vector<size_t> rowidx){
    if(N != 1){
        cout << "Single selection Must be a vector case." << endl;
    }
    return Block(rowidx, {0,1});
}

template <typename T>
vector<T> Matrix<T>::RowSum()
{
    vector<T> ret;
    for(int i=0; i<M; i++){
        T curs; memset(&curs, 0, sizeof(T));
        for(int j=0; j<N; j++){
            curs += dataptr[i + j * lda];
        }
        ret.push_back(curs);
    }
    return ret;
}

template <typename T>
vector<T> Matrix<T>::ColSum()
{
    vector<T> ret;
    for(int i=0; i<N; i++){
        T curs; memset(&curs, 0, sizeof(T));
        for(int j=0; j<M; j++){
            curs += dataptr[i*lda + j];
        }
        ret.push_back(curs);
    }
    return ret;
}

template <typename T>
Matrix<T> Matrix<T>::Transpose()
{
    Matrix<T> tmpmat(N, M);
    vector<T>& buffer = tmpmat.Datavec();
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            buffer[i * N + j] = dataptr[i + j * M];
        }
    }
    return tmpmat;
}

template<typename T>
void Matrix<T>::SetDiagBand(int bandlen){
    if(M != N) {
        cout << "not support" << endl;
        exit(0);
    }
    for(int i=0; i<M; i++){
        for(int j=0; j<M; j++){
            if(abs(i-j) <= bandlen-1){
                SetElem(i,j,1);
            }
        }
    }
}

template<typename T>
void Matrix<T>::SetOffDiagBand(int bandlen){
    if(M != N) {
        cout << "not support" << endl;
        exit(0);
    }
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            if(abs(i-j) >= M - bandlen){
                SetElem(i,j,1);
            }
        }
    }
}

template<typename T>
Matrix<T> Matrix<T>::Conjugate()
{
    Matrix<T> tmpmat(M, N);
    vector<T>& buffer = tmpmat.Datavec();
    for(size_t i=0; i<buffer.size(); i++){
        buffer[i] = ElementwiseConjugate(dataptr[i]);
    }
    return tmpmat;
}


template <typename T>
void Matrix<T>::Fill(T val)
{
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            dataptr[i + j * lda] = val;
        }
    }
}

template<typename T>
void Matrix<T>::Fillrandom() {
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            dataptr[i + j * lda] = (T)(rand() % 1024) / (T)1024;
        }
    }
}

template <typename T>
Matrix<T>& Matrix<T>::operator=( const Matrix<T>& rhs)
{
    M = rhs.M;
    N = rhs.N;
    lda = rhs.lda;
    dataptr = rhs.Datavec();
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator+=( const Matrix<T>& rhs)
{
    assert(rhs.Row() == M && rhs.Col() == N && rhs.Lda() == lda);
    auto rhsvec = rhs.Datavec();
    for(size_t i=0; i<M * N; i++)
    dataptr[i] += rhsvec[i];
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-=( const Matrix<T>& rhs)
{
    assert(rhs.Row() == M && rhs.Col() == N && rhs.Lda() == lda);
    auto rhsvec = rhs.Datavec();
    for(size_t i=0; i<M * N; i++)
    dataptr[i] -= rhsvec[i];
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator*=( const Matrix<T>& rhs)
{
    assert(rhs.Row() == N);
    Matrix<T> ret(M,rhs.Col());
    T *C = ret.RawPtr();
    T *B = rhs.RawPtr();
    gemm(dataptr.data(), B, C, M, rhs.Col(), N);
    *this = ret;
    return *this;
}

template <typename T>
void Matrix<T>::Tofile(string abspath){
    FILE * file;
    if((file = fopen(abspath.c_str(), "wb")))
    {
        size_t ret = fwrite(dataptr.data(), sizeof(T), dataptr.size(), file);
        fclose(file);
    }else{
        cout << "Write Fail, No such file path: " << abspath << endl;
    }
}

template <typename T>
Matrix<T> Matrix<T>::Fromfile(string abspath, size_t M, size_t N){
    FILE *file;
    Matrix<T> retmatrix(M, N);

    // this->M = M;
    // this->N = N;
    // this->lda = M;
    // size_t length = M * N;

    // dataptr.resize(length, 0);
    if((file = fopen(abspath.c_str(), "rb")))
    {
        size_t ret = fread(retmatrix.Datavec().data(), sizeof(T),
        M * N, file);
    }else{
        cout << "Read Fail, No such file path: " << abspath << endl;
    }
    return retmatrix;
}

template<typename T>
Matrix<T> Matrix<T>::ApplyMask(Matrix<int> maskmat){
    if(maskmat.Row() != this->Row() || maskmat.Col() != this->Col()){
        cout << "maskmat dim not equal." << endl;
        exit(0);
    }
    Matrix<T> retmatrix = *this;
    for(int i=0; i<maskmat.Row(); i++){
        for(int j=0; j<maskmat.Col(); j++){
            if(!maskmat.GetElem(i,j))
            retmatrix.SetElem(i,j,(T)(0.0));
        }
    }
    return retmatrix;
}


template <typename T>
Matrix<T> Matrix<T>::operator+( const Matrix<T>& rhs)
{
    assert(rhs.Row() == M && rhs.Col() == N && rhs.Lda() == lda);
    return Matrix<T>(*this) += rhs;
}

template <typename T>
Matrix<T> Matrix<T>::operator-( const Matrix<T>& rhs)
{
    assert(rhs.Row() == M && rhs.Col() == N && rhs.Lda() == lda);
    return Matrix<T>(*this) -= rhs;
}

template <typename T>
Matrix<T> Matrix<T>::operator*( const Matrix<T>& rhs)
{
    assert(rhs.Row() == N);
    return Matrix<T>(*this) *= rhs;
}


template <typename T>
double Matrix<T>::allclose(Matrix<T> & reference){
    double maxnumerator = 0.0;
    double maxdenominator = getabs(reference.GetElem(0,0));
//    assert(reference.Row() == M && reference.Col() == N && reference.Lda() == lda);
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            maxnumerator = fmax( maxnumerator, getabs(reference.GetElem(i,j) , GetElem(i,j)) );
            maxdenominator = fmax(maxdenominator, getabs(reference.GetElem(i,j)));
        }
    }
    if(maxdenominator == 0){
      if(maxnumerator == 0) {
          return 0;
      }else{
          cout << "all close denominator is 0, numerator is " << maxnumerator << endl;
      }
    }
    return maxnumerator / maxdenominator;
}

template class Matrix<int>;
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<complex<float>>;
template class Matrix<complex<double>>;
template class Matrix<size_t>;

template<typename T>
ostream& operator<<(ostream& os, const Matrix<T>& mat){
    for(int i=0; i<mat.Row(); i++){
        for(int j=0; j<mat.Col(); j++){
            os << setprecision(4) << setw(2) << mat.GetElem(i,j) << "  ";
        }
        os << endl;
    }
    return os;
}

template<typename T>
void Matrix<T>::ToText(string abspath) {
    std::ofstream out(abspath);
    out << *this << endl;

}



template ostream& operator<<(ostream& os, const Matrix<int>& mat);
template ostream& operator<<(ostream& os, const Matrix<float>& mat);
template ostream& operator<<(ostream& os, const Matrix<double>& mat);
template ostream& operator<<(ostream& os, const Matrix<complex<float>>& mat);
template ostream& operator<<(ostream& os, const Matrix<complex<double>>& mat);


