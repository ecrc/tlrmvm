#include <iostream>
#include <unistd.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <iomanip>
#include <complex>


// intel
#include <oneapi/mkl.hpp>
#include <mkl.h>


using namespace std;
using std::endl;








string PathJoin(vector<string> paths){
    string res = "";
    for(int i=0; i<paths.size()-1; i++){
        res += paths[i];
        res += "/";
    }
    res += paths.back();
    return res;
}


void cblasgemm(const CBLAS_ORDER order, const CBLAS_TRANSPOSE trans_a, 
const CBLAS_TRANSPOSE trans_b,int m, int n, int k, 
float alpha, const float *A, int lda, 
const float *B, int ldb, float beta, float *C, int ldc){
    cblas_sgemm(order, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void cblasgemm(const CBLAS_ORDER order, const CBLAS_TRANSPOSE trans_a, 
const CBLAS_TRANSPOSE trans_b,int m, int n, int k, 
double alpha, const double *A, int lda, 
const double *B, int ldb, double beta, double *C, int ldc){
    cblas_dgemm(order, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void cblasgemm(const CBLAS_ORDER order, const CBLAS_TRANSPOSE trans_a, 
const CBLAS_TRANSPOSE trans_b,int m, int n, int k, complex<float> alpha, 
const complex<float> *A, int lda, 
const complex<float> *B, int ldb, complex<float> beta, complex<float> *C, int ldc){
    cblas_cgemm(order, trans_a, trans_b, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

void cblasgemm(const CBLAS_ORDER order, const CBLAS_TRANSPOSE trans_a, 
const CBLAS_TRANSPOSE trans_b,int m, int n, int k, complex<double> alpha, 
const complex<double> *A, int lda, 
const complex<double> *B, int ldb, complex<double> beta, complex<double> *C, int ldc){
    cblas_zgemm(order, trans_a, trans_b, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

void gemm(const int *A, const int *B, int *C, int m, int n, int k){
    float alpha(1.0), beta(1.0);
    float *fA, *fB, *fC;
    fA = new float[m*k];
    fB = new float[k*n];
    fC = new float[m*n];
    #pragma omp parallel for 
    for(size_t i=0; i<m*k; i++) fA[i] = (float)A[i];
    #pragma omp parallel for 
    for(size_t i=0; i<k*n; i++) fB[i] = (float)B[i];
    #pragma omp parallel for 
    for(size_t i=0; i<m*n; i++) fC[i] = (float)C[i];
    cblasgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 
    alpha, fA, m, fB, k, beta, fC, m);
    #pragma omp parallel for 
    for(size_t i=0; i<m*n; i++) C[i] = (int)fC[i];    
}

void gemm(const float *A, const float *B, float *C, int m, int n, int k){
    float alpha(1.0), beta(1.0);
    cblasgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 
    alpha, A, m, B, k, beta, C, m);

}

void gemm(const double *A, const double *B, double *C, int m, int n, int k){
    double alpha(1.0), beta(1.0);
    cblasgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 
    alpha, A, m, B, k, beta, C, m);

}

void gemm(const complex<float> *A, const complex<float> *B, complex<float> *C, int m, int n, int k){
    complex<float> alpha(1.0,0.0), beta(1.0,0.0);
    cblasgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, m, B, k, beta, C, m);
}

void gemm(const complex<double> *A, const complex<double> *B, complex<double> *C, int m, int n, int k){
    complex<double> alpha(1.0,0.0), beta(1.0,0.0);
    cblasgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, m, B, k, beta, C, m);
}



template<typename T>
class Matrix{
public:

    Matrix(); //!< Empty constructor, donothing.

    Matrix(T *inptr, size_t M, size_t N); //!< Accept an initialized pointer as constructor.
    
    Matrix(vector<T> invec, size_t M, size_t N); //!< Accept an initialized pointer as constructor.

    Matrix(size_t M, size_t N); //!< Random number constructor.
    
    Matrix(const Matrix&); //!< Copy constructor.

    ~Matrix(); //!< Deconstructor.

    T GetElem(size_t row, size_t col) const {return dataptr[row+col*M];}  //!< Get element at Index(i,j)
    
    void SetElem(size_t row, size_t col, T val) {dataptr[row+col*M] = val;} // set element at Index(i,j)
    
    void Fill(T val); //!< Fill the data pointer with value val.

    T Sum(); //!< Sum of matrix.

    vector<T> RowSum(); //!< Sum along the row.
    
    vector<T> ColSum(); //!< Sum along the column.

    Matrix<T> Transpose();

    size_t Row() const {return M;} //!< return Row.

    Matrix<T> Row(size_t idx);

    size_t Col() const {return N;} //!< return Column.

    Matrix<T> Col(size_t idx);

    Matrix<T> Block(vector<size_t> rowidx, vector<size_t> colidx);

    Matrix<T> Block(vector<size_t> rowidx);

    size_t Lda() const {return lda;} //!< return leading dimension.

    vector<size_t> Shape() const { return {M, N}; }
    T * RawPtr() const {return (T*)dataptr.data();} //!< return raw pointer.
    
    vector<T>& Datavec() const { return (vector<T>&)dataptr; }
    void CopyFromPointer(T * srcptr);

    double allclose(Matrix<T> & rhs);
    void Tofile(string abspath);
    static Matrix<T> Fromfile(string abspath, size_t M, size_t N);

    Matrix<T> ApplyMask(Matrix<int> maskmat);

    // operator overloading
    Matrix<T>& operator=(const Matrix<T>& rhs);
    Matrix<T> operator+(const Matrix<T>& rhs);
    Matrix<T>& operator+=(const Matrix<T>& rhs);
    Matrix<T> operator-(const Matrix<T>& rhs);
    Matrix<T>& operator-=(const Matrix<T>& rhs);
    Matrix<T> operator*(const Matrix<T>& rhs);
    Matrix<T>& operator*=(const Matrix<T>& rhs);

    template<typename U> // https://stackoverflow.com/questions/4660123/overloading-friend-operator-for-template-class
    friend ostream& operator<<(ostream& os, const Matrix<U>& mat);


protected:
    vector<T> dataptr;
    size_t M;
    size_t N;
    size_t lda;
};

template<typename T>
ostream& operator<<(ostream& os, const Matrix<T>& mat);

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

template <typename T>
void Matrix<T>::CopyFromPointer(T * srcptr)
{
    memcpy(dataptr.data(), srcptr, sizeof(T) * M * N);
}


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



template <typename T>
void Matrix<T>::Fill(T val)
{
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            dataptr[i + j * lda] = val;
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
    if(maskmat.Row() != this->Row() || maskmat.Col() != this->Col()) {
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
    assert(reference.Row() == M && reference.Col() == N && reference.Lda() == lda);
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            maxnumerator = fmax( maxnumerator, getabs(reference.GetElem(i,j) , GetElem(i,j)) );
            maxdenominator = fmax(maxdenominator, getabs(reference.GetElem(i,j)));
        }
    }
    return maxnumerator / maxdenominator;
}

template class Matrix<int>;
template class Matrix<float>;
template class Matrix<double>;
template class Matrix<complex<float>>;
template class Matrix<complex<double>>;

template<typename T>
ostream& operator<<(ostream& os, const Matrix<T>& mat){
    for(int i=0; i<mat.Row(); i++){
        for(int j=0; j<mat.Col(); j++){
            os << std::setprecision(4) << std::setw(7) << mat.GetElem(i,j) << "  ";
        }
        os << endl;
    }
    return os;
}

template ostream& operator<<(ostream& os, const Matrix<int>& mat);
template ostream& operator<<(ostream& os, const Matrix<float>& mat);
template ostream& operator<<(ostream& os, const Matrix<double>& mat);
template ostream& operator<<(ostream& os, const Matrix<complex<float>>& mat);
template ostream& operator<<(ostream& os, const Matrix<complex<double>>& mat);

int CalculatePadding(int originDim, int nb){
    return ( originDim / nb + (originDim % nb != 0) ) * nb;
}

template<typename T>
void ReadBinary(string absfilepath, T * outbuffer, size_t length){
    FILE *f = fopen(absfilepath.c_str(), "rb");
    int ret = fread(outbuffer, sizeof(T), length, f); 
    assert(ret == length);
    fclose(f);
}

template void ReadBinary<int>(string, int*, size_t);
template void ReadBinary<float>(string, float*, size_t);
template void ReadBinary<double>(string, double*, size_t);


void ReadAstronomyBinary(string prefix, int ** outbuffer, size_t length, string acc, int nb, string id){
    char postfix[200];
    sprintf(postfix, "/R_id%s_nb%d_acc%s.bin", id.c_str(), nb, acc.c_str());
    string pstr(postfix);
    int * retptr;
    retptr = new int[length];
    ReadBinary(prefix + postfix, retptr, length);
    outbuffer[0] = retptr;
}
void ReadAstronomyBinary(string prefix, float ** outbuffer, size_t length, string acc, int nb, string id){
    char postfix[200];
    float *dataptr;
    sprintf(postfix, "_id%s_nb%d_acc%s.bin", id.c_str(), nb, acc.c_str());
    dataptr = new float[length];
    ReadBinary(prefix+postfix, dataptr, length);
    outbuffer[0] = dataptr;
}

void ReadSeismicBinary(string prefix, int ** outbuffer, size_t length, string acc, int nb, int id){
    char postfix[200];
    sprintf(postfix, "R-Mck_freqslice%d_nb%d_acc%s.bin", id, nb, acc.c_str());
    string pstr(postfix);
    int * retptr;
    retptr = new int[length];
    string abspath = PathJoin({prefix, postfix});
    ReadBinary(abspath, retptr, length);
    outbuffer[0] = retptr;
}
void ReadSeismicBinary(string prefix, complex<float> ** outptr, size_t length, 
string acc, int nb, int id){
    char postfix[200];
    vector<string> UVstr = {"_real_Mck_freqslice", "_imag_Mck_freqslice"};
    float *dataptr[4];
    for(int i=0; i<2; i++){
        sprintf(postfix, "%s%d_nb%d_acc%s.bin", UVstr[i].c_str(), id, nb, acc.c_str());
        dataptr[i] = new float[length];
        ReadBinary(prefix+postfix, dataptr[i], length);
    }
    complex<float> * tmpptr = new complex<float>[length];
    for(size_t i=0; i<length; i++){
        tmpptr[i] = complex<float>(dataptr[0][i], dataptr[1][i]);
    }
    outptr[0] = tmpptr;
    for(int i=0; i<2; i++) delete[] dataptr[i];
}

void ReadSeismicBinaryX(string prefix, complex<float> ** outX, size_t length, string acc, int nb, int id){
    char postfix[200];
    float *dataptr[2];
    string xstr = "xvector/fdplus_";
    sprintf(postfix, "%s%d_real.bin", xstr.c_str(), id);
    dataptr[0] = new float[length];
    ReadBinary(PathJoin({prefix, postfix}), dataptr[0], length);
    sprintf(postfix, "%s%d_imag.bin", xstr.c_str(), id);
    dataptr[1] = new float[length];
    ReadBinary(PathJoin({prefix, postfix}), dataptr[1], length);
    complex<float> * Xptr = new complex<float>[length];

    for(size_t i=0; i<length; i++){
        Xptr[i] = complex<float>(dataptr[0][i], dataptr[1][i]);
    }
    outX[0] = Xptr;
    for(int i=0; i<2; i++) delete[] dataptr[i];
}

class ArgsParser{
    public:
    ArgsParser(){};
    ArgsParser(int argc, char**argv){
        for(int i=1; i<argc; i++){
            string tmp = string(argv[i]);
            if(tmp.substr(0,2) != "--") continue;
            else{
                int s = 0;
                while(s < tmp.size() && tmp[s] != '=') s++;
                if(s == tmp.size()) continue;
                argmap[tmp.substr(2,s-2)] = tmp.substr(s+1,tmp.size()-2-1);
            }
        }
    }
    int getint(string key){
        if(argmap.find(key) == argmap.end())
        {cout << "key error in getint" << endl; exit(0);}
        return atoi(argmap[key].c_str());
    }
    string getstring(string key){
        if(argmap.find(key) == argmap.end())
        {cout << "key error in getstring" << endl; exit(0);}
        return argmap[key];
    }
    unordered_map<string, string> argmap;
};




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

class AstronomyPCMatrix : public FPCMAT
{
public:
    AstronomyPCMatrix(string datafolder, string acc, int nb, string id, int originM, int originN);
    void LoadData();
    double GetDifference(Matrix<float>& m1, Matrix<float>& m2, size_t idx);
    Matrix<float> GetX();
    
    string id;
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



template<typename T>
PCMatrix<T>::PCMatrix(string datafolder, string acc, int nb, int originM, int originN)
:datafolder(datafolder), acc(acc),nb(nb), originM(originM), originN(originN){
    paddingM = CalculatePadding(originM, nb);
    paddingN = CalculatePadding(originN, nb);
    Mtglobal = paddingM / nb;
    Ntglobal = paddingN / nb;
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
    // for(int i=0; i < Mtglobal; i++){
    //     for(int j=0; j<Ntglobal; j++){
    //         char c1[50], c2[50];
    //         sprintf(c1, "U%d_%d.bin",i,j);
    //         sprintf(c2, "V%d_%d.bin",i,j);
    //         Utilemat[i][j].Tofile(string(c1));
    //         Vtilemat[i][j].Tofile(string(c2));
    //     }
    // }
}


template<typename T>
void PCMatrix<T>::RecoverDense(){
    Densemat.clear();
    Densemat.resize(Mtglobal, vector<Matrix<T>>(Ntglobal, Matrix<T>()));
    for(int i=0; i<Mtglobal; i++){
        for(int j=0; j<Ntglobal; j++){
            Densemat[i][j] = Utilemat[i][j] * Vtilemat[i][j];
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
    assert(xvec.Row() * xvec.Col() == originN || xvec.Row() * xvec.Col() == paddingN);
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

template class PCMatrix<float>;
template class PCMatrix<complex<float>>;

/////////////////////////////////////////////////////////

AstronomyPCMatrix::AstronomyPCMatrix(string datafolder, string acc, int nb, string id, int originM, int originN)
:PCMatrix(datafolder, acc, nb, originM, originN), id(id){
    LoadData();
}

void AstronomyPCMatrix::LoadData(){
    int *DataR;
    float *DataAv, *DataAu;
    ReadAstronomyBinary(datafolder, &DataR, Mtglobal * Ntglobal, acc, nb, id);
    Rmat = Matrix<int>(DataR, Mtglobal, Ntglobal);
    size_t granksum = Rmat.Sum();
    ReadAstronomyBinary(datafolder+"/V", &DataAv, granksum * nb, acc ,nb, id);
    ReadAstronomyBinary(datafolder+"/U", &DataAu, granksum * nb, acc ,nb, id);
    Matrix<float>Vmat(DataAv, granksum, nb);
    Matrix<float>Umat(DataAu, granksum, nb);
    // Rmat.Tofile("Rmat.bin");
    // Vmat.Tofile("Vmat.bin");
    // Umat.Tofile("Umat.bin");
    BuildTLRMatrices(Umat, Vmat);
    delete[] DataR;
    delete[] DataAv;
    delete[] DataAu;
}

double AstronomyPCMatrix::GetDifference(Matrix<float>& m1, Matrix<float>& m2, size_t idx){
    return fabs(m1.RawPtr()[idx]-m2.RawPtr()[idx]);
}

Matrix<float> AstronomyPCMatrix::GetX(){
    Matrix<float> ret(paddingN, 1);
    #pragma omp parallel for
    for(int i=0; i<paddingN; i++) ret.RawPtr()[i] = float(rand()/RAND_MAX);
    setX(ret);
    return ret;
}




struct TlrmvmConfig{
    int originM;
    int originN;
    int Mtg;
    int Ntg;
    int nb;
    string datafolder;
    string Rfilepath;
    string acc;
    string mavisid;
    Matrix<int> Maskmat;
    int ldaA;
    int ldaB;
    int ldaC;
    TlrmvmConfig(int originM, int originN, int nb, 
    string rfilepath, string datafolder, string acc, string mavisid);
};

TlrmvmConfig::TlrmvmConfig(int originM, int originN, int nb, 
string rfilepath, string datafolder, string acc, string mavisid)
:datafolder(datafolder),Rfilepath(rfilepath), acc(acc), mavisid(mavisid),
originM(originM), originN(originN), nb(nb),ldaA(-1), ldaB(-1), ldaC(-1)
{
    Mtg = CalculatePadding(originM, nb) / nb;
    Ntg = CalculatePadding(originN, nb) / nb;
}


template<typename T>
void CopyData(T *dst, T *src, size_t n){
    memcpy(dst, src, sizeof(T) * n);
}

template void CopyData<int>(int*, int*, size_t);
template void CopyData<int*>(int**, int**, size_t);
template void CopyData<float>(float*, float*, size_t);
template void CopyData<float*>(float* *, float* *, size_t);
template void CopyData<double>(double*, double*, size_t);
template void CopyData<double*>(double**, double**, size_t);
template void CopyData<complex<float>>(complex<float>*, complex<float>*, size_t);
template void CopyData(complex<float>**, complex<float>**, size_t);
template void CopyData(complex<double>*, complex<double>*, size_t);
template void CopyData(complex<double>**, complex<double>**, size_t);


template<typename T>
void GetHostMemory(T **A, size_t n){
    T * tmp = new T[n];
    A[0] = tmp;
}

template void GetHostMemory(int8_t **, size_t);
template void GetHostMemory(int **, size_t);
template void GetHostMemory(unsigned long int **, size_t);
template void GetHostMemory(float ***, size_t);
template void GetHostMemory(float **, size_t);
template void GetHostMemory(double **, size_t);
template void GetHostMemory(complex<float> ***, size_t);
template void GetHostMemory(complex<float> **, size_t);
template void GetHostMemory(complex<double> **, size_t);



int main(int argc, char**argv){
    int originM;
    int originN;
    int nb;
    string acc;
    string datafolder;
    string mavisid;
    string rankfile;
    string Ufile;
    string Vfile;
    vector<double> timestat;
    vector<double> bandstat;
    double bytesprocessed;
    size_t granksum;
    auto argparser = ArgsParser(argc, argv);
    originM = argparser.getint("M");
    originN = argparser.getint("N");
    nb = argparser.getint("nb");
    acc = argparser.getstring("acc");
    mavisid = argparser.getstring("mavisid");
    datafolder = argparser.getstring("datafolder");
    char rpath[100]; 
    sprintf(rpath, "%s/R_id%s_nb%d_acc%s.bin", datafolder.c_str(), mavisid.c_str(), nb, acc.c_str());
    rankfile = string(rpath);

    TlrmvmConfig tlrmvmconfig(originM, originN, nb, rankfile, datafolder, acc, mavisid);
    Matrix<int> maskmat(tlrmvmconfig.Mtg, tlrmvmconfig.Ntg);
    maskmat.Fill(1);
    tlrmvmconfig.Maskmat = maskmat;
    size_t Mtg = tlrmvmconfig.Mtg;
    size_t Ntg = tlrmvmconfig.Ntg;
    auto OrgRmat = Matrix<int>::Fromfile(tlrmvmconfig.Rfilepath, Mtg, Ntg);
    granksum = OrgRmat.Sum();
    auto Maskmat = tlrmvmconfig.Maskmat;
    auto WorkRmat = OrgRmat.ApplyMask(Maskmat);
    auto colsum = WorkRmat.ColSum();
    datafolder = tlrmvmconfig.datafolder;
    granksum = OrgRmat.Sum();
    nb = tlrmvmconfig.nb;
    acc = tlrmvmconfig.acc;
    mavisid = tlrmvmconfig.mavisid;
    float * DataAv, * DataAu, *Datax;
    ReadAstronomyBinary(datafolder+"/V", &DataAv, granksum * nb, acc ,nb, mavisid);
    ReadAstronomyBinary(datafolder+"/U", &DataAu, granksum * nb, acc ,nb, mavisid);  
    Datax = new float[tlrmvmconfig.Ntg * nb];
    for(int i=0; i<tlrmvmconfig.Ntg*nb; i++){
        Datax[i] = (float)0.1;
    }
    auto xmat = Matrix<float>(Datax, tlrmvmconfig.Ntg * nb, 1);

    auto device = sycl::device(sycl::cpu_selector());
    auto device_queue = sycl::queue(device);
    // data pointer
    size_t phase1Acnt;
    size_t phase1Xcnt;
    size_t phase1Ycnt;
    size_t phase3Acnt;
    size_t phase3Xcnt;
    size_t phase3Ycnt;
    // phase1 
    vector<size_t> AvMs;
    vector<size_t> AvKs;
    vector<size_t> AvNs;


    // phase1 cpu
    float *h_Av;
    float *h_x;
    float *h_yv;
    float *h_yvout;
    float **h_Avbp;
    float **h_xbp;
    float **h_yvbp;

    // phase 2  
    vector<size_t> h_phase2mapping;

    // phase 3
    vector<size_t> AuMs;
    vector<size_t> AuKs;
    vector<size_t> AuNs;

    float *h_Au;
    float *h_yu;
    float *h_y;
    float *h_yout;
    float **h_Aubp;
    float **h_yubp;
    float **h_ybp;
    float **h_youtbp;
    int *d_fp32rowrank;

    GetHostMemory(&h_Avbp, Ntg);
    GetHostMemory(&h_xbp, Ntg);
    GetHostMemory(&h_yvbp, Ntg);


    for(int i=0; i<Ntg; i++){
        AvMs.push_back(colsum[i]);
        AvKs.push_back(nb);
        AvNs.push_back(1);
    }
    size_t workmatgranksum = WorkRmat.Sum();

    phase1Acnt = 0;
    phase1Xcnt = 0;
    phase1Ycnt = 0;

    for(int i=0; i<Ntg; i++){
        phase1Acnt += AvMs[i] * AvKs[i];
        phase1Xcnt += AvKs[i] * AvNs[i];
        phase1Ycnt += AvMs[i] * AvNs[i];
    }


    // host memory  phase 1
    h_Av = sycl::malloc_shared<float>(phase1Acnt, device_queue);
    h_x = sycl::malloc_shared<float>(phase1Xcnt, device_queue);
    h_yv = sycl::malloc_shared<float>(phase1Ycnt, device_queue);
    h_xbp[0] = h_x;
    h_Avbp[0] = h_Av;
    h_yvbp[0] = h_yv;

    for(int i=1; i<Ntg; i++){
        size_t AvMK = AvMs[i-1] * AvKs[i-1];
        size_t AvKN = AvKs[i-1] * AvNs[i-1];
        size_t AvMN = AvMs[i-1] * AvNs[i-1];
        h_Avbp[i] = h_Avbp[i-1] + AvMK;
        h_xbp[i] = h_xbp[i-1] + AvKN;
        h_yvbp[i] = h_yvbp[i-1] + AvMN;
    }

    
    // move data from DataAv to Av
    vector<int> gcolsum = OrgRmat.ColSum();
    float *Avwalkptr = DataAv;
    for(int i=0; i<Ntg; i++){
        // column start pointers
        float *colptr = h_Avbp[i];
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
    float * xwalkptr = Datax;
    size_t offset = 0;
    for(int i=0; i<Ntg; i++){
        for(int j=0; j<nb; j++){
            *(h_x + offset + j) = *(xwalkptr + i*nb + j);
        }
        offset += nb;
    }
    
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
        for(int j=0; j<Ntg; j++){
            if(WorkRmat.GetElem(i,j) != 0){
                int currank = WorkRmat.GetElem(i,j);
                for(int k=0; k<currank; k++){
                    // phase2record[i][j][1].push_back(p2walker++);
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
    for(int i=0; i<Ntg; i++){
        for(int j=0; j<Mtg; j++){
            if(WorkRmat.GetElem(j,i) != 0){
                int currank = WorkRmat.GetElem(j,i);
                for(int k=0; k<currank; k++){
                    // h_phase2mapping.push_back(phase2record[j][i][1][k]);
                }
            }
        }
    }

    GetHostMemory(&h_Aubp, Ntg);
    GetHostMemory(&h_yubp, Ntg);
    GetHostMemory(&h_ybp, Ntg);
    GetHostMemory(&h_youtbp, Ntg);
    // phase 3
    auto rowsum = WorkRmat.RowSum();
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

    GetHostMemory(&h_Au, phase3Acnt); // Au 
    GetHostMemory(&h_yu, phase3Xcnt); // yu 
    GetHostMemory(&h_y, phase3Ycnt); // y 
    GetHostMemory(&h_yout, phase3Ycnt); // yout
    h_Aubp[0] = h_Au;
    h_yubp[0] = h_yu;
    h_ybp[0] = h_y;
    h_youtbp[0] = h_yout;

    for(int i=1; i<Ntg; i++){
        size_t AuMK = AuMs[i-1] * AuKs[i-1];
        size_t AuKN = AuKs[i-1] * AuNs[i-1];
        size_t AuMN = AuMs[i-1] * AuNs[i-1];

        h_Aubp[i] = h_Aubp[i-1] + AuMK;
        h_yubp[i] = h_yubp[i-1] + AuKN;
        h_ybp[i] = h_ybp[i-1] + AuMN;
        h_youtbp[i] = h_youtbp[i-1] + rowsum[i-1];
    }

    // move data Au to memory buffer
    float *colptr = h_Au;
    float *dataauwalker = DataAu;
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

    delete[] DataAv;
    delete[] DataAu;
    delete[] Datax;
    sycl::range<1>num_items{workmatgranksum};
    float alpha = 1.0;
    float beta = 0.0;
    auto transA = oneapi::mkl::transpose::nontrans;
    vector<double> rawtime;
    
    auto start = std::chrono::steady_clock::now();
    // phase 1
    unsigned long * h_phase2mappingptr = new unsigned long [h_phase2mapping.size()];
    memcpy(h_phase2mappingptr, h_phase2mapping.data(), sizeof(unsigned long)*h_phase2mapping.size());
    for(int i=0; i<Ntg; i++){
        if(colsum[i] != 0){
            oneapi::mkl::blas::gemv(device_queue, 
            transA, AvMs[i],
            AvKs[i], alpha, h_Avbp[i], 
            AvMs[i], h_xbp[i],
            1, beta, h_yvbp[i], 1);
        }
    }

    device_queue.wait_and_throw();

    // phase 2
    device_queue.parallel_for(num_items, [h_yu,h_phase2mappingptr,h_yv](int i){h_yu[h_phase2mappingptr[i]] = h_yv[i];});
    device_queue.wait_and_throw();

    // phase 3
    for(int i=0; i<Mtg; i++){        
        if(rowsum[i] != 0){
            oneapi::mkl::blas::gemv(device_queue, 
            transA, AuMs[i],
            AuKs[i], alpha, h_Aubp[i], 
            AuMs[i], h_yubp[i],
            1, beta, h_ybp[i], 1);
        }
    }
    device_queue.wait_and_throw();

    auto end = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    delete[] h_phase2mappingptr;


    AstronomyPCMatrix astropcmat(datafolder, acc, nb, mavisid, originM, originN);
    astropcmat.LoadData();
    astropcmat.setX(xmat);
    astropcmat.GetDense();
    Matrix<float> yv_pc = astropcmat.Phase1();
    auto hyv = Matrix<float>(h_yv, workmatgranksum, 1);
    cout << " Phase 1 Correctness : " << hyv.allclose(yv_pc) << endl;
    Matrix<float> yu_pc = astropcmat.Phase2();
    auto hyu = Matrix<float>(h_yu, workmatgranksum, 1);
    cout << " Phase 2 Correctness : " << hyu.allclose(yu_pc) << endl;
    Matrix<float> y_pc = astropcmat.Phase3();
    auto paddingM = astropcmat.paddingM;
    auto hy = Matrix<float>(h_y, paddingM, 1);
    cout << " Phase 3 Correctness : "<< hy.allclose(y_pc) << endl;

    
}