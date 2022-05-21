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

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

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

void checkcudaerror(cudaError_t st){
    if(st != cudaSuccess){
        printf("cuda status failed. \n");
        exit(0);
    }
}
void checkcudastatus(cublasStatus_t st){
    if(st != CUBLAS_STATUS_SUCCESS){
        printf("cuda status failed. \n");
        exit(0);
    }
}

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

    cudaSetDevice(0);
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    float *d_A, *d_y, *d_x;
    // malloc device
    cudaStat = cudaMalloc ((void**)&d_A, (long)m*(long)n*sizeof(float));
    checkcudaerror(cudaStat);
    cudaStat = cudaMalloc ((void**)&d_x, (long)n*sizeof(float));
    checkcudaerror(cudaStat);
    cudaStat = cudaMalloc ((void**)&d_y, (long)m*sizeof(float));
    checkcudaerror(cudaStat);
    stat = cublasCreate(&handle);
    checkcudastatus(stat);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    vector<double> rawtime;
    float alpha,beta;
    alpha = 1.0; beta = 0.0;
    cudaMemcpy(d_A, A, (long)m*(long)n * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(d_x, x, (long)n * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(d_y, y, (long)m * sizeof(float), cudaMemcpyDefault);

    for(int nr=0; nr<loopsize; nr++){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        cublasSgemv(handle, CUBLAS_OP_N, m, n,
                    &alpha, d_A, m, d_x, 1, &beta, d_y, 1);
        cudaEventRecord(stop);
        cudaDeviceSynchronize();
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        rawtime.push_back(milliseconds*1e-3);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(A, d_A, (long)m*(long)n  * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(x, d_x, (long)n * sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(y, d_y, (long)m * sizeof(float), cudaMemcpyDefault);
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
