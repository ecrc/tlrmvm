#ifndef UTIL_H
#define UTIL_H

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

namespace tlrmat{

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
    ArgsParser(){};
    ArgsParser(int argc, char**argv);
    int getint(string key);
    string getstring(string key);
    unordered_map<string, string> argmap;
};

template<typename T>
void LoadBinary(char *filename, T **databuffer, unsigned long elems);


} // namespace tlrmat

#endif // UTIL_H





