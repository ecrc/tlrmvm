#include "Util.h"
#include <complex>
#include <algorithm>
#include <iostream>
#include <cassert>
#include <string>
#include <time.h>
#include <sys/time.h>


using namespace std;
namespace tlrmat{

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
    {cout << "key error in getint" << endl; exit(0);}
    return atoi(argmap[key].c_str());
}

string ArgsParser::getstring(string key){
    if(argmap.find(key) == argmap.end())
    {cout << "key error in getstring" << endl; exit(0);}
    return argmap[key];
}


template<typename T>
void LoadBinary(char *filename, T **databuffer, unsigned long elems)
{
    FILE *f = fopen(filename, "rb");
    T *tmpbuffer = new T[elems];
    int ret = fread(tmpbuffer, sizeof(T), elems, f); 
    assert(ret == elems);
    fclose(f);
    databuffer[0] = tmpbuffer;
}
template void LoadBinary<int>(char*, int **, unsigned long);
template void LoadBinary<float>(char*, float **, unsigned long);
template void LoadBinary<double>(char*, double **, unsigned long);
template void LoadBinary<complex<float>>(char*, complex<float> **, unsigned long);
template void LoadBinary<complex<double>>(char*, complex<double> **, unsigned long);


} //namespace tlrmat




 
