#include "common/cpu/MemoryManagement.h"
#include "common/cpu/Util.h"
#include <complex>
#include <assert.h>
#include <string.h>
namespace tlrmat {

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
    memset(tmp, 0, sizeof(T) * n);
    A[0] = tmp;
}

template void GetHostMemory(int8_t **, size_t);
template void GetHostMemory(int **, size_t);
template void GetHostMemory(unsigned long int **, size_t);
template void GetHostMemory(float **, size_t);
template void GetHostMemory(float ***, size_t);
template void GetHostMemory(double **, size_t);
template void GetHostMemory(double ***, size_t);
template void GetHostMemory(complex<float> **, size_t);
template void GetHostMemory(complex<float> ***, size_t);
template void GetHostMemory(complex<double> **, size_t);
template void GetHostMemory(complex<double> ***, size_t);

template<typename T>
void GetHostMemory(T **A, T **B, T **C, size_t m, size_t k, size_t n, T val){
    T * tmpA = new T[m * k];
    T * tmpB = new T[k * n];
    T * tmpC = new T[m * n];
    for(size_t i=0; i<m * k; i++){
        tmpA[i] = val;
    }
    for(size_t i=0; i<k * n; i++){
        tmpB[i] = val;
    }
    for(size_t i=0; i<m * n; i++){
        tmpC[i] = val;
    }
    A[0] = tmpA;
    B[0] = tmpB;
    C[0] = tmpC;
}
template void GetHostMemory<float>(float **, float **, float **, size_t, 
size_t, size_t, float);
template void GetHostMemory<complex<float>>(complex<float> **, complex<float> **, 
complex<float> **, size_t, size_t, size_t, complex<float>);

template<typename T>
void CalculateBatchedMemoryLayout(T *A, T *B, T *C,
T **Abatchpointer, T **Bbatchpointer, T **Cbatchpointer,
vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns){
    Abatchpointer[0] = A;
    Bbatchpointer[0] = B;
    Cbatchpointer[0] = C;
    for(int bi=1; bi<Ms.size(); bi++){
        Abatchpointer[bi] = Abatchpointer[bi-1] + Ms[bi-1] * Ks[bi-1];
        Bbatchpointer[bi] = Bbatchpointer[bi-1] + Ks[bi-1] * Ns[bi-1];
        Cbatchpointer[bi] = Cbatchpointer[bi-1] + Ms[bi-1] * Ns[bi-1];
    }
}

template void CalculateBatchedMemoryLayout(float *, float *, float *,
float **, float **, float **, vector<size_t>,vector<size_t>,vector<size_t>);

template void CalculateBatchedMemoryLayout(complex<float> *, complex<float> *, complex<float> *,
complex<float> **, complex<float> **, complex<float> **, vector<size_t>,vector<size_t>,vector<size_t>);


template<typename T>
void GetHostMemoryBatched(T **A, T **B, T **C, 
T ***Abatchpointer, T ***Bbatchpointer, T ***Cbatchpointer,
vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns, T val){
    T *tmpA, *tmpB, *tmpC;
    T **tmpAbp, **tmpBbp, **tmpCbp;
    size_t Atotalelems, Btotalelems, Ctotalelems;
    CaluclateTotalElements(Ms, Ks, Ns, Atotalelems, Btotalelems, Ctotalelems);
    // allocate memory space for A
    tmpA = new T[Atotalelems];
    for(size_t i=0; i<Atotalelems; i++) tmpA[i] = val;
    // allocate memory space for B
    tmpB = new T[Btotalelems];
    for(size_t i=0; i<Btotalelems; i++) tmpB[i] = val;
    // allocate memory space for C
    tmpC = new T[Ctotalelems];
    for(size_t i=0; i<Ctotalelems; i++) tmpC[i] = val;
    assert(Ms.size() == Ks.size() && Ks.size() == Ns.size());
    size_t batchsize = Ms.size();
    tmpAbp = new T*[batchsize];
    tmpBbp = new T*[batchsize];
    tmpCbp = new T*[batchsize];
    CalculateBatchedMemoryLayout(tmpA, tmpB, tmpC, tmpAbp, tmpBbp, tmpCbp, Ms, Ks, Ns);
    Abatchpointer[0] = tmpAbp;
    Bbatchpointer[0] = tmpBbp;
    Cbatchpointer[0] = tmpCbp;
    A[0] = tmpA;
    B[0] = tmpB;
    C[0] = tmpC;
}

template void GetHostMemoryBatched(float **, float **, float **,
float ***, float ***, float ***, vector<size_t>, vector<size_t>, vector<size_t>, float val);

template void GetHostMemoryBatched(complex<float> **, complex<float> **, complex<float> **,
complex<float> ***, complex<float> ***, complex<float> ***, vector<size_t>, 
vector<size_t>, vector<size_t>, complex<float> val);


template<typename T>
void FreeHostMemory(T *A){
    delete[] A;
}
template void FreeHostMemory<int8_t>(int8_t*);
template void FreeHostMemory<float>(float*);
template void FreeHostMemory<complex<float>>(complex<float>*);
template void FreeHostMemory<double>(double*);
template void FreeHostMemory<complex<double>>(complex<double>*);
template void FreeHostMemory<float*>(float**);
template void FreeHostMemory<complex<float>*>(complex<float>**);
template void FreeHostMemory<double*>(double**);
template void FreeHostMemory<complex<double>*>(complex<double>**);

template<typename T>
void FreeHostMemory(T *A, T *B, T *C){
    delete[] A;
    delete[] B;
    delete[] C;
}

template void FreeHostMemory<float>(float*, float*, float*);
template void FreeHostMemory<complex<float>>(complex<float>*, 
complex<float>*, complex<float>*);

template<typename T>
void FreeHostMemoryBatched(T *A, T *B, T *C, 
T **Abatchpointer, T **Bbatchpointer, T **Cbatchpointer){
    delete[] A;
    delete[] B;
    delete[] C;
    delete[] Abatchpointer;
    delete[] Bbatchpointer;
    delete[] Cbatchpointer;
}

template void FreeHostMemoryBatched(float *A, float *B, float *C, 
float **Abatchpointer, float **Bbatchpointer, float **Cbatchpointer);

template void FreeHostMemoryBatched(complex<float> *A, complex<float> *B, 
complex<float> *C, complex<float> **Abatchpointer, 
complex<float> **Bbatchpointer, complex<float> **Cbatchpointer);



void CaluclateTotalElements(vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns,
size_t &Atotal, size_t &Btotal, size_t &Ctotal){
    Atotal = Btotal = Ctotal = 0; 
    for(int i=0; i<Ms.size(); i++){
        Atotal += Ms[i] * Ks[i];
    }
    for(int i=0; i<Ms.size(); i++){
        Btotal += Ks[i] * Ns[i];
    }
    for(int i=0; i<Ms.size(); i++){
        Ctotal += Ms[i] * Ns[i];
    }
}






} // namespace tlrmat

