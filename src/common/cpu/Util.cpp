
//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#include <complex>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <string>
#include <memory.h>
#include <ctime>
#include <sys/time.h>
#include "Util.hpp"
#include <fcntl.h>

using namespace std;

double gettime(void)
{
	struct timeval tp;
	gettimeofday( &tp, NULL );
	return tp.tv_sec + 1e-6 * tp.tv_usec;
}

template <typename T>
void Init(T *ptr, size_t length, T val)
{
    for (size_t i = 0; i < length; i++)
    {
        ptr[i] = val;
    }
}

template void Init<float>(float*, size_t, float);
template void Init<double>(double*, size_t, double);
template void Init<std::complex<float>>(std::complex<float>*, size_t, std::complex<float>);
template void Init<std::complex<double>>(std::complex<double>*, size_t, std::complex<double>);


vector<int> PrefixSum(int * Array, size_t length){
    vector<int> ret(length, 0);
    for(int i=1; i<length; i++){
        ret[i] = ret[i-1] + Array[i-1];
    }
    return ret;
}


string PathJoin(vector<string> paths){
    string res = "";
    for(int i=0; i<paths.size()-1; i++){
        res += paths[i];
        res += "/";
    }
    res += paths.back();
    return res;
}

double Median(std::vector<double> &rawptr){
    std::sort(rawptr.begin(), rawptr.end());
    size_t mid = rawptr.size() / 2;
    if(rawptr.size() % 2 == 0) return 0.5 * (rawptr[mid] + rawptr[mid+1]);
    return rawptr[mid];
}

// new calculation of error: see https://www.netlib.org/lapack/lug/node75.html

double NetlibError(int * y_res, int * y_check, size_t length){
    int * y_diff = new int[length];
    for(int i=0; i<length; i++) y_diff[i] = y_res[i] - y_check[i];
    double max_diff = 0.0, max_y_check = 0.0;
    for(int i=0; i<length; i++){
        if(fabs(y_diff[i]) > max_diff) max_diff = fabs(y_diff[i]);
        if(fabs(y_check[i]) > max_y_check) max_y_check = fabs(y_check[i]);
    }
    return max_diff / max_y_check;
}
double NetlibError(float * y_res, float * y_check, size_t length){
    float * y_diff = new float[length];
    for(int i=0; i<length; i++) y_diff[i] = y_res[i] - y_check[i];
    double max_diff = 0.0, max_y_check = 0.0;
    for(int i=0; i<length; i++){
        if(fabs(y_diff[i]) > max_diff) max_diff = fabs(y_diff[i]);
        if(fabs(y_check[i]) > max_y_check) max_y_check = fabs(y_check[i]);
    }
    if(max_y_check == 0 && max_diff == 0){
        return 0;
    }else if(max_y_check == 0 && max_diff != 0){
        return max_y_check;
    }
    return max_diff / max_y_check;
}
double NetlibError(double * y_res, double * y_check, size_t length){
    double * y_diff = new double[length];
    for(int i=0; i<length; i++) y_diff[i] = y_res[i] - y_check[i];
    double max_diff = 0.0, max_y_check = 0.0;
    for(int i=0; i<length; i++){
        if(fabs(y_diff[i]) > max_diff) max_diff = fabs(y_diff[i]);
        if(fabs(y_check[i]) > max_y_check) max_y_check = fabs(y_check[i]);
    }
    return max_diff / max_y_check;
}
double NetlibError(complex<float> * y_res, complex<float> * y_check, size_t length){
    complex<float> * y_diff = new complex<float>[length];
    for(int i=0; i<length; i++) y_diff[i] = y_res[i] - y_check[i];
    double max_diff = 0.0, max_y_check = 0.0;
    for(int i=0; i<length; i++){
        if(abs(y_diff[i]) > max_diff) max_diff = abs(y_diff[i]);
        if(abs(y_check[i]) > max_y_check) max_y_check = abs(y_check[i]);
    }
    return max_diff / max_y_check;
}
double NetlibError(complex<double> *y_res, complex<double> *y_check, size_t length){
    complex<double> * y_diff = new complex<double>[length];
    for(int i=0; i<length; i++) y_diff[i] = y_res[i] - y_check[i];
    double max_diff = 0.0, max_y_check = 0.0;
    for(int i=0; i<length; i++){
        if(abs(y_diff[i]) > max_diff) max_diff = abs(y_diff[i]);
        if(abs(y_check[i]) > max_y_check) max_y_check = abs(y_check[i]);
    }
    return max_diff / max_y_check;
}

void GtestLog(string logstr){
    std::cerr << "\033[32m[ APP INFO ] " << logstr.c_str() << "\033[0m"<< endl;
}

ArgsParser::ArgsParser(int argc, char**argv){
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

int ArgsParser::getint(string key){
    if(argmap.find(key) == argmap.end())
    {cout << "key error in getint:" << key << endl; exit(0);}
    return atoi(argmap[key].c_str());
}

string ArgsParser::getstring(string key){
    if(argmap.find(key) == argmap.end())
    {cout << "key error in getstring: "<< key << endl; exit(0);}
    return argmap[key];
}

bool ArgsParser::getbool(string key){
    if(argmap.find(key) == argmap.end())
    {cout << "key error in getbool: "<< key << endl; exit(0);}
    return atoi(argmap[key].c_str());
}

template<typename T>
void LoadBinary(char *filename, T **databuffer, unsigned long elems)
{
    int fd = open(filename, O_RDONLY);
    T *tmpbuffer = new T[elems];
    int ret = read(fd, (void*)tmpbuffer, sizeof(T) * elems);
    close(fd);
    databuffer[0] = tmpbuffer;
}
template void LoadBinary<int>(char*, int **, unsigned long);
template void LoadBinary<float>(char*, float **, unsigned long);
template void LoadBinary<double>(char*, double **, unsigned long);
template void LoadBinary<complex<float>>(char*, complex<float> **, unsigned long);
template void LoadBinary<complex<double>>(char*, complex<double> **, unsigned long);


void init_alpha_beta(float &alpha, float &beta){
    alpha = (float)1.0;
    beta = (float)0.0;
}

void init_alpha_beta(complex<float> &alpha, complex<float> &beta){
    alpha = complex<float>(1.0,0.0);
    beta = complex<float>(0.0,0.0);
}
void init_alpha_beta(double &alpha, double &beta){
    alpha = (double)1.0;
    beta = (double)0.0;
}
void init_alpha_beta(complex<double> &alpha, complex<double> &beta){
    alpha = complex<double>(1.0,0.0);
    beta = complex<double>(0.0,0.0);
}

size_t ElementwiseConjugate(size_t x){return x;}

int ElementwiseConjugate(int x){return x;}

float ElementwiseConjugate(float x){return x;}

double ElementwiseConjugate(double x){return x;}

complex<float> ElementwiseConjugate(complex<float> x)
{return complex<float>(x.real(), -x.imag());}

complex<double> ElementwiseConjugate(complex<double> x)
{return complex<double>(x.real(), -x.imag());}



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
template void CopyData<std::complex<float>>(std::complex<float>*, std::complex<float>*, size_t);
template void CopyData(std::complex<float>**, std::complex<float>**, size_t);
template void CopyData(std::complex<double>*, std::complex<double>*, size_t);
template void CopyData(std::complex<double>**, std::complex<double>**, size_t);


template<typename T>
void GetHostMemory(T **A, size_t n){
    T * tmp = new T[n];
    memset(tmp, 0, sizeof(T) * n);
    A[0] = tmp;
}
template void GetHostMemory(char **, size_t);
template void GetHostMemory(char ***, size_t);
template void GetHostMemory(unsigned short **, size_t);
template void GetHostMemory(unsigned short ***, size_t);
template void GetHostMemory(short **, size_t);
template void GetHostMemory(short ***, size_t);
template void GetHostMemory(int8_t **, size_t);
template void GetHostMemory(int8_t ***, size_t);
template void GetHostMemory(int **, size_t);
template void GetHostMemory(int ***, size_t);
template void GetHostMemory(unsigned long int **, size_t);
template void GetHostMemory(unsigned long int ***, size_t);
template void GetHostMemory(float **, size_t);
template void GetHostMemory(float ***, size_t);
template void GetHostMemory(double **, size_t);
template void GetHostMemory(double ***, size_t);
template void GetHostMemory(std::complex<float> **, size_t);
template void GetHostMemory(std::complex<float> ***, size_t);
template void GetHostMemory(std::complex<double> **, size_t);
template void GetHostMemory(std::complex<double> ***, size_t);
#ifdef USE_FUJITSU
template void GetHostMemory(__fp16 **, size_t);
template void GetHostMemory(__fp16 ***, size_t);
#endif

template<typename T>
void FreeHostMemory(T *A){
    delete[] A;
}
template void FreeHostMemory<int8_t>(int8_t*);
template void FreeHostMemory<float>(float*);
template void FreeHostMemory<std::complex<float>>(std::complex<float>*);
template void FreeHostMemory<double>(double*);
template void FreeHostMemory<std::complex<double>>(std::complex<double>*);
template void FreeHostMemory<float*>(float**);
template void FreeHostMemory<std::complex<float>*>(std::complex<float>**);
template void FreeHostMemory<double*>(double**);
template void FreeHostMemory<std::complex<double>*>(std::complex<double>**);


void CaluclateTotalElements(std::vector<size_t> Ms, std::vector<size_t> Ks, std::vector<size_t> Ns,
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

