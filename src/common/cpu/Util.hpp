//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#pragma once

#include <unistd.h>
#include <complex>
#include <vector>
#include <string>
#include <iostream>
#include <unordered_map>

using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::complex;
using std::unordered_map;


double gettime(void);


/**
 * @brief Init length element in ptr as val.
 * 
 * @tparam T 
 * @param ptr 
 * @param length 
 * @param val 
 */
template <typename T>
void Init(T *ptr, size_t length, T val);

/**
 * @brief Calculate prefix sum of Array.
 * 
 * @param Array Input Array.
 * @param length length of Array.
 * @return vector<int> 
 */
vector<int> PrefixSum(int * Array, size_t length);


/**
 * @brief Join string in paths as a path.
 * 
 * @param paths 
 * @return string 
 */
string PathJoin(vector<string> paths);


double Median(std::vector<double> &rawptr);


double NetlibError(int * y_res, int * y_check, size_t length);
double NetlibError(float *, float *, size_t length);
double NetlibError(double *, double *, size_t length);
double NetlibError(complex<float> *, complex<float> *, size_t length);
double NetlibError(complex<double> *, complex<double> *, size_t length);



void GtestLog(string logstr);



class ArgsParser{
    public:
    ArgsParser()=default;
    ArgsParser(int argc, char**argv);
    int getint(string key);
    string getstring(string key);
    bool getbool(string key);
    unordered_map<string, string> argmap;
};

template<typename T>
void LoadBinary(char *filename, T **databuffer, unsigned long elems);

void init_alpha_beta(float &alpha, float &beta);

void init_alpha_beta(complex<float> &alpha, complex<float> &beta);

void init_alpha_beta(double &alpha, double &beta);

void init_alpha_beta(complex<double> &alpha, complex<double> &beta);

int ElementwiseConjugate(int x);
float ElementwiseConjugate(float x);
double ElementwiseConjugate(double x);
complex<float> ElementwiseConjugate(complex<float> x);
complex<double> ElementwiseConjugate(complex<double> x);
size_t ElementwiseConjugate(size_t x);

template<typename T>
void CopyData(T *dst, T *src, size_t n);

template<typename T>
void GetHostMemory(T **A, size_t n);

template <typename T>
void FreeHostMemory(T *A);

template<typename T>
void FreeHostMemory(T *A, T *B, T *C);

void CaluclateTotalElements(std::vector<size_t> Ms, std::vector<size_t> Ks, std::vector<size_t> Ns,
                            size_t &Atotal, size_t &Btotal, size_t &Ctotal);






