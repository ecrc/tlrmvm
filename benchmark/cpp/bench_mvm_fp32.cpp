#include <stdlib.h>
#include <math.h>
#include <memory.h>
#include <string>
#include <unordered_map>
#include <chrono>
#include <vector>
#include <iostream>
#include <algorithm>
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

void naive_impl(float *A, float* x, float *y, int m, int n){
    for(int i=0; i<m; i++){
        y[i] = (float)0.0;
        for(int j=0; j<n; j++){
            y[i] += A[(long)i + (long)j * (long)m] * x[j];
        }
    }
}

void Initdata_cpu(float *A, float* x, float *y, int m, int n){
    memset(y, 0, sizeof(float) * m);
    for(int i=0; i<n; i++){
        x[i] = (float)rand() / (float)RAND_MAX;
    }
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            long tmp = (long)i * (long)n + (long)j;
            A[tmp] = (float)rand() / (float)RAND_MAX;
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

double checkcorrectness(float *y, float *ynaive, int dim, int sumdim){
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
    auto A = (float *)malloc( (long)m*(long)n*sizeof( float ));
    auto x = (float *)malloc( (long)n*sizeof( float ));
    auto y = (float *)malloc( (long)m*sizeof( float ));
    auto ynaive = (float *)malloc( (long)m*sizeof( float ));
    Initdata_cpu(A,x,y,m,n);
    vector<double> rawtime;
    float alpha,beta;
    alpha = 1.0; beta = 0.0;
    for(int nr=0; nr<loopsize; nr++){
        auto start = std::chrono::steady_clock::now();
        cblas_sgemv(CblasColMajor, CblasNoTrans,m, n,
                    alpha, A, m, x, 1, beta, y, 1);
        auto end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        rawtime.push_back(elapsed_time*1e-6);
    }
    naive_impl(A,x,ynaive,m,n);
    auto error = checkcorrectness(y,ynaive,m,n);
    cout << "error " << error << endl;
    std::sort(rawtime.begin(), rawtime.end());
    cout << "median " << rawtime[rawtime.size() / 2] * 1e6 << " us."<< endl;
    double bytes = m * n * sizeof(float) + (m+n) * sizeof(float);
    cout << "U and V bases size: " << bytes * 1e-6 << " MB." << endl;
    cout << "Bandwidth " << bytes / rawtime[rawtime.size() / 2] * 1e-9 << " GB/s" << endl;
    return 0;
}
