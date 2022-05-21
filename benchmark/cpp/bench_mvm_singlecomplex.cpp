#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <string>
#include <unordered_map>
#include <chrono>
#include <vector>
#include <iostream>
#include <algorithm>
#include <complex>
using namespace std;


#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef USE_CBLAS
#include <cblas.h>
#endif

#ifdef USE_BLIS
#include <blis.h>
#endif

void naive_impl(complex<float> *A, complex<float>* x, complex<float> *y, int m, int n){
    for(int i=0; i<m; i++){
        y[i] = complex<float>(0.0, 0.0);
        for(int j=0; j<n; j++){
            auto v1 = A[(long)i + (long)j * (long)m];
            auto v2 = x[j];
            y[i] += complex<float>(v1.real()*v2.real()-v1.imag()*v2.imag(), v1.real()*v2.imag()+v1.imag()*v2.real());
        }
    }
}

void Initdata_cpu(complex<float> *A, complex<float>* x, complex<float> *y, int m, int n){
    memset(y, 0, sizeof(complex<float>) * m);
    for(int i=0; i<n; i++){
        x[i] = complex<float>((float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX);
    }
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            long tmp = (long)i * (long)n + (long)j;
            A[tmp] = complex<float>((float)rand() / (float)RAND_MAX, (float)rand() / (float)RAND_MAX);
        }
    }
}

double mean(double *y, int dim){
    double ret = 0.0;
#pragma omp parallel for
    for(int i=0; i<dim; i++){
        ret += y[i];
    }
    ret /= dim;
    return ret;
}

double checkcorrectness(complex<float> *y, complex<float> *ynaive, int dim, int sumdim){
    double *rerr = (double*)malloc(sizeof(double)*dim);
    for(int i=0; i<dim; i++){
        rerr[i] =fabs(y[i] - ynaive[i]) / fabs(ynaive[i]);
    }
    double meanval = mean(rerr, dim);
    free(rerr);
    return meanval;
}


class ArgsParser{
public:
    ArgsParser()=default;
    ArgsParser(int argc, const char**argv){
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
    bool getbool(string key){
        if(argmap.find(key) == argmap.end())
        {cout << "key error in getint" << endl; exit(0);}
        return atoi(argmap[key].c_str());
    }
    unordered_map<string, string> argmap;
};



int main(int argc, const char** argv)
{
    auto argparser = ArgsParser(argc, argv);
    auto m = argparser.getint("M");
    auto n = argparser.getint("N");
    auto check = argparser.getbool("check");
    auto loopsize = argparser.getint("loopsize");
    auto A = (complex<float> *)malloc( (long)m*(long)n*sizeof( complex<float> ));
    auto x = (complex<float> *)malloc( (long)n*sizeof( complex<float> ));
    auto y = (complex<float> *)malloc( (long)m*sizeof( complex<float> ));
    auto ynaive = (complex<float> *)malloc( (long)m*sizeof( complex<float> ));
    Initdata_cpu(A,x,y,m,n);
    vector<double> rawtime;
    complex<float> alpha,beta;
    alpha = complex<float>(1.0,0.0); beta = complex<float>(0.0,0.0);
    for(int nr=0; nr<loopsize; nr++){
        auto start = std::chrono::steady_clock::now();
        cblas_cgemv(CblasColMajor, CblasNoTrans,m, n,
                    &alpha, A, m, x, 1, &beta, y, 1);
        auto end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        rawtime.push_back(elapsed_time*1e-6);
    }
    naive_impl(A,x,ynaive,m,n);
    auto error = checkcorrectness(y,ynaive,m,n);
    cout << "error " << error << endl;
    std::sort(rawtime.begin(), rawtime.end());
    cout << "median " << rawtime[rawtime.size() / 2] * 1e6 << " us."<< endl;
    double bytes = m * n * sizeof(complex<float>) + (m+n) * sizeof(complex<float>);
    cout << "U and V bases size: " << bytes * 1e-6 << " MB." << endl;
    cout << "Bandwidth " << bytes / rawtime[rawtime.size() / 2] * 1e-9 << " GB/s" << endl;
    return 0;
}
