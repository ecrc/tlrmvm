#ifndef MEMORYMANAGEMENT_H
#define MEMORYMANAGEMENT_H

#include <unistd.h>
#include <vector>

using namespace std;

namespace tlrmat
{

template<typename T>
void CopyData(T *dst, T *src, size_t n);

template<typename T>
void GetHostMemory(T **A, size_t n);

template<typename T>
void GetHostMemory(T **A, T **B, T **C, size_t m, size_t k, size_t n, T val);

template<typename T>
void CalculateBatchedMemoryLayout(T *A, T *B, T *C, 
T **Abatchpointer, T **Bbatchpointer, T ** Cbatchpointer,
vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns);

template<typename T>
void GetHostMemoryBatched(T **A, T **B, T **C, 
T ***Abatchpointer, T ***Bbatchpointer, T ***Cbatchpointer,
vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns, T val);

template <typename T>
void FreeHostMemory(T *A);

template<typename T>
void FreeHostMemory(T *A, T *B, T *C);

template<typename T>
void FreeHostMemoryBatched(T *A, T *B, T *C, 
T **Abatchpointer, T **Bbatchpointer, T **Cbatchpointer);


void CaluclateTotalElements(vector<size_t> Ms, vector<size_t> Ks, vector<size_t> Ns,
size_t &Atotal, size_t &Btotal, size_t &Ctotal);


} // namespace tlrmat


#endif // MEMORYMANAGEMENT_H


