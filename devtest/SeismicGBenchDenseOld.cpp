#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <memory.h>
#include <cuda.h>
#include <mpi.h>
#include <nccl.h>
#include <complex>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h> // if you need CUBLAS v2, include before magma.h
#include <cuda_fp16.h>
#include "common/Common.h"
#include "common/AppUtil.h"
#include "tlrmvm/Tlrmvm.h"
#include "tlrmvm/Tlrmvmcuda.h"

#include "benchmark/benchmark.h"
#include "benchmark/benchmark.h"

using namespace std;
using namespace tlrmat;
using namespace cudatlrmat;
using namespace benchmark;
using namespace cudatlrmvm;

unordered_map<string, string> inputmap;








class SeismicFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) {
        if(inputmap.size() < 6){
            cout << "not enough args " << endl;
            exit(1);
        }
        datafolder = inputmap["datafolder"];
        acc = inputmap["acc"];
        string freqstr = inputmap["freqlist"];
        nb = atoi(inputmap["nb"].c_str());
        originM = atoi(inputmap["M"].c_str());
        originN = atoi(inputmap["N"].c_str());
        streamsize = atoi(inputmap["streamsize"].c_str());
        // complex 
        GetHostMemory(&hA, originM * originN);
        GetHostMemory(&hx, originM);
        GetHostMemory(&hy, originN);
        for(size_t i=0; i<originM*originN; i++){
            hA[i] = complex<float>(1.0,-1.0);
        }
        for(size_t i=0; i < originN; i++){
            hx[i] = complex<float>(1.0,1.0);
        }
        GetDeviceMemory(reinterpret_cast<cuComplex**>(&A), originM * originN);
        GetDeviceMemory(reinterpret_cast<cuComplex**>(&x), originM * originN);
        GetDeviceMemory(reinterpret_cast<cuComplex**>(&y), originM * originN);
        CopyDataB2HD((complex<float>*)A, hA, originM * originN);
        CopyDataB2HD((complex<float>*)x, hx, originN);

        // Float32
        GetHostMemory(&hostfloatA_r, originM * originN);
        GetHostMemory(&hostfloatA_i, originM * originN);
        GetHostMemory(&hostfloatx_ri, 2*originN);
        GetHostMemory(&hostfloaty_r, 2*originM);
        GetHostMemory(&hostfloaty_i, 2*originM);

        for(size_t i=0; i<originM*originN; i++){
            hostfloatA_r[i] = 1.0;
            hostfloatA_i[i] = -1.0;
        }
        for(size_t i=0; i < 2*originN; i++){
            hostfloatx_ri[i] = 1.0;
        }
        
        GetDeviceMemory(&floatA_r, originM * originN);
        GetDeviceMemory(&floatA_i, originM * originN);
        GetDeviceMemory(&floatx_ri, 2*originN);
        GetDeviceMemory(&floaty_r, 2*originM);
        GetDeviceMemory(&floaty_i, 2*originM);
        CopyDataB2HD(floatA_r, hostfloatA_r, originM * originN);
        CopyDataB2HD(floatA_i, hostfloatA_i, originM * originN);
        CopyDataB2HD(floatx_ri, hostfloatx_ri, 2*originN);

        // Float 16
        GetHostHalfMemory(&hosthalfA_r, originM * originN);
        GetHostHalfMemory(&hosthalfA_i, originM * originN);
        GetHostHalfMemory(&hosthalfx_ri, 2*originN);
        GetHostHalfMemory(&hosthalfy_r, 2*originM);
        GetHostHalfMemory(&hosthalfy_i, 2*originM);

        for(size_t i=0; i<originM*originN; i++){
            hosthalfA_r[i] = 1.0;
            hosthalfA_i[i] = -1.0;
        }
        for(size_t i=0; i < 2*originN; i++){
            hosthalfx_ri[i] = 1.0;
        }
        
        GetDeviceMemory(&halfA_r, originM * originN);
        GetDeviceMemory(&halfA_i, originM * originN);
        GetDeviceMemory(&halfx_ri, 2*originN);
        GetDeviceMemory(&halfy_r, 2*originM);
        GetDeviceMemory(&halfy_i, 2*originM);
        CopyDataB2HD(halfA_r, hosthalfA_r, originM * originN);
        CopyDataB2HD(halfA_i, hosthalfA_i, originM * originN);
        CopyDataB2HD(halfx_ri, hosthalfx_ri, 2*originN);

        // Int 8
        GetHostMemory(&hostint8A_r, originM * originN);
        GetHostMemory(&hostint8A_i, originM * originN);
        GetHostMemory(&hostint8x_ri, 2*originN);
        GetHostMemory(&hostint8y_r, 2*originM);
        GetHostMemory(&hostint8y_i, 2*originM);
        GetHostMemory(&hostinty_r, 2*originN);
        GetHostMemory(&hostinty_i, 2*originN);

        for(size_t i=0; i<originM*originN; i++){
            hostint8A_r[i] = (int8_t)1;
            hostint8A_i[i] = (int8_t)(-1);
        }
        for(size_t i=0; i < 2*originN; i++){
            hostint8x_ri[i] = (int8_t)1;
        }
        
        GetDeviceMemory(&int8A_r, originM * originN);
        GetDeviceMemory(&int8A_i, originM * originN);
        GetDeviceMemory(&int8x_ri, 2*originN);
        GetDeviceMemory(&int8y_r, 2*originM);
        GetDeviceMemory(&int8y_i, 2*originM);
        GetDeviceMemory(&inty_r, 2*originM);
        GetDeviceMemory(&inty_i, 2*originM);
        CopyDataB2HD(int8A_r, hostint8A_r, originM * originN);
        CopyDataB2HD(int8A_i, hostint8A_i, originM * originN);
        CopyDataB2HD(int8x_ri, hostint8x_ri, 2*originN);


    }

    void TearDown(const ::benchmark::State& state) {
        FreeHostMemory(hA);
        FreeHostMemory(hx);
        FreeHostMemory(hy);
        FreeDeviceMemory(reinterpret_cast<cuComplex*>(A));
        FreeDeviceMemory(reinterpret_cast<cuComplex*>(x));
        FreeDeviceMemory(reinterpret_cast<cuComplex*>(y));
    }

    string datafolder;
    string acc;
    int nb;
    int originM;
    int originN;
    int streamsize;
    int Mtglobal;
    int Ntglobal;
    // complex
    complex<float> *hA;
    complex<float> *hx;
    complex<float> *hy;
    complex<float> *A;
    complex<float> *x;
    complex<float> *y;
    // Float 32
    float * hostfloatA_r;
    float * hostfloatA_i;
    float * hostfloatx_ri;
    float * hostfloaty_r;
    float * hostfloaty_i;
    float * floatA_r;
    float * floatA_i;
    float * floatx_ri;
    float * floaty_r;
    float * floaty_i;

    // Float 16 host
    half * hosthalfA_r;
    half * hosthalfA_i;
    half * hosthalfx_ri;
    half * hosthalfy_r;
    half * hosthalfy_i;
    half * halfA_r;
    half * halfA_i;
    half * halfx_ri;
    half * halfy_r;
    half * halfy_i;

    // Int 8
    int8_t * hostint8A_r;
    int8_t * hostint8A_i;
    int8_t * hostint8x_ri;
    int8_t * hostint8y_r;
    int8_t * hostint8y_i;
    int * hostinty_r;
    int * hostinty_i;
    int8_t * int8A_r;
    int8_t * int8A_i;
    int8_t * int8x_ri;
    int8_t * int8y_r;
    int8_t * int8y_i;
    int * inty_r;
    int * inty_i;

};


// BENCHMARK_DEFINE_F(SeismicFixture, FloatinFloatoutTest)(benchmark::State& state) {
//     cublasHandle_t handle;
//     cublasCreate(&handle);
//     float alpha;
//     float beta;
//     alpha = 1.0;
//     beta = 0.0;
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     vector<double> rawtime;
//     for (auto _ : state) {
//         cudaEventRecord(start);
//         cudaDeviceSynchronize();

//         CUBLASCHECK(cublasgemmex(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
//         originM, 2, originN, 
//         &alpha, (const void*)floatA_r, CUDA_R_32F, originM, 
//         (const void*)floatx_ri, CUDA_R_32F, originN, 
//         &beta, floaty_r, CUDA_R_32F, originM, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

//         CUBLASCHECK(cublasgemmex(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
//         originM, 2, originN, 
//         &alpha, (const void*)halfA_i, CUDA_R_32F, originM, 
//         (const void*)floatx_ri, CUDA_R_32F, originN, 
//         &beta, floaty_i, CUDA_R_32F, originM, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        
//         cudaDeviceSynchronize();
//         cudaEventRecord(stop);
//         float milliseconds = 0;
//         cudaEventSynchronize(stop);
//         cudaEventElapsedTime(&milliseconds, start, stop);
//         state.SetIterationTime(milliseconds*1e-3);
//         rawtime.push_back(milliseconds*1e-3);
//     }
//     cublasDestroy(handle);
//     double bytes = 2*sizeof(half) * 
//     (originM * originN + originM + originN) * 
//     (double)state.iterations();
//     state.counters["BandWidth"] =
//     Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
//     CopyDataB2HD(hostfloaty_r, floaty_r, 2*originM);
//     CopyDataB2HD(hostfloaty_i, floaty_i, 2*originM);
//     for(size_t i=0; i<originM; i++){
//         if(hostfloaty_r[i] != 19602){
//             cout << hostfloaty_r[i] << endl;
//             break;
//         }
//         if(hostfloaty_i[i] != 0){
//             cout << hostfloaty_i[i] << endl;
//             break;
//         }
//     }
// }

// BENCHMARK_REGISTER_F(SeismicFixture, FloatinFloatoutTest)
// ->Unit(benchmark::kMicrosecond)
// ->UseManualTime()
// ->Iterations(1)
// ->Repetitions(1);


BENCHMARK_DEFINE_F(SeismicFixture, HalfinFloatoutTest)(benchmark::State& state) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaStream_t stream;
    cublasSetStream_v2(handle, stream);
    float alpha;
    float beta;
    alpha = 1.0;
    beta = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    vector<double> rawtime;
    for (auto _ : state) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);

        CUBLASCHECK(cublasgemmex(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        originM, 2, originN, 
        &alpha, (const void*)halfA_r, CUDA_R_16F, originM, 
        (const void*)halfx_ri, CUDA_R_16F, originN, 
        &beta, floaty_r, CUDA_R_32F, originM, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        CUBLASCHECK(cublasgemmex(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        originM, 2, originN, 
        &alpha, (const void*)halfA_i, CUDA_R_16F, originM, 
        (const void*)halfx_ri, CUDA_R_16F, originN, 
        &beta, floaty_i, CUDA_R_32F, originM, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
        
        // merge_float_realimag(floaty_r, floaty_i, originM, stream);
        cudaStreamSynchronize(stream);
        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
    }
    cublasDestroy(handle);
    double bytes = 2*sizeof(half) * 
    (originM * originN + originM + originN) * 
    (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
    CopyDataB2HD(hostfloaty_r, floaty_r, 2*originM);
    CopyDataB2HD(hostfloaty_i, floaty_i, 2*originM);
    for(size_t i=0; i<originM; i++){
        if(hostfloaty_r[i] != 19602){
            cout << hostfloaty_r[i] << endl;
            break;
        }
        if(hostfloaty_i[i] != 0){
            cout << hostfloaty_i[i] << endl;
            break;
        }
    }
    // it's broken
}

BENCHMARK_REGISTER_F(SeismicFixture, HalfinFloatoutTest)
->Unit(benchmark::kMicrosecond)
->UseManualTime()
// ->Iterations(1)
->Repetitions(1);


BENCHMARK_DEFINE_F(SeismicFixture, HalfinHalfoutTest)(benchmark::State& state) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaStream_t stream;
    cublasSetStream_v2(handle, stream);
    half alpha;
    half beta;
    alpha = 1.0;
    beta = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    vector<double> rawtime;
    for (auto _ : state) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        CUBLASCHECK(cublasgemmex(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        originM, 2, originN, 
        &alpha, (const void*)halfA_r, CUDA_R_16F, originM, 
        (const void*)halfx_ri, CUDA_R_16F, originN, 
        &beta, halfy_r, CUDA_R_16F, originM, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
        CUBLASCHECK(cublasgemmex(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        originM, 2, originN, 
        &alpha, (const void*)halfA_i, CUDA_R_16F, originM, 
        (const void*)halfx_ri, CUDA_R_16F, originN, 
        &beta, halfy_i, CUDA_R_16F, originM, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
        merge_half_realimag(halfy_r, halfy_i, originM, stream);
        cudaStreamSynchronize(stream);
        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
    }
    cublasDestroy(handle);
    double bytes = 2*sizeof(half) * (originM * originN + originM + originN) * (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
}

BENCHMARK_REGISTER_F(SeismicFixture, HalfinHalfoutTest)
->Unit(benchmark::kMicrosecond)
->UseManualTime()
// ->Iterations(1)
->Repetitions(1);


BENCHMARK_DEFINE_F(SeismicFixture, IntinFloatoutTest)(benchmark::State& state) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaStream_t stream;
    cublasSetStream_v2(handle, stream);

    float alpha;
    float beta;
    alpha = 1.0;
    beta = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    vector<double> rawtime;
    for (auto _ : state) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        CUBLASCHECK(cublasgemmex(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        originM, 2, originN, 
        &alpha, (const void*)int8A_r, CUDA_R_8I, originM, 
        (const void*)int8x_ri, CUDA_R_8I, originN, 
        &beta, floaty_r, CUDA_R_32F, originM, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        CUBLASCHECK(cublasgemmex(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
        originM, 2, originN, 
        &alpha, (const void*)int8A_i, CUDA_R_8I, originM, 
        (const void*)int8x_ri, CUDA_R_8I, originN, 
        &beta, floaty_i, CUDA_R_32F, originM, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        // merge_float_realimag(floaty_r, floaty_i, originM, stream);
        cudaStreamSynchronize(stream);
        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
    }
    cublasDestroy(handle);
    double bytes = 2*sizeof(half) * 
    (originM * originN + originM + originN) * 
    (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
    CopyDataB2HD(hostfloaty_r, floaty_r, 2*originM);
    CopyDataB2HD(hostfloaty_i, floaty_i, 2*originM);
    for(size_t i=0; i<originM; i++){
        if(hostfloaty_r[i] != 19602){
            cout << hostfloaty_r[i] << endl;
            break;
        }
        if(hostfloaty_i[i] != 0){
            cout << hostfloaty_i[i] << endl;
            break;
        }
    }
}

BENCHMARK_REGISTER_F(SeismicFixture, IntinFloatoutTest)
->Unit(benchmark::kMicrosecond)
->UseManualTime()
// ->Iterations(1)
->Repetitions(1);

BENCHMARK_DEFINE_F(SeismicFixture, DenseTest)(benchmark::State& state) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    cuComplex alpha;
    cuComplex beta;
    alpha.x = 1.0;
    alpha.y = 0.0;
    beta.x = 0.0;
    beta.y = 0.0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    vector<double> rawtime;
    for (auto _ : state) {
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        CUBLASCHECK(cublasgemv(handle, CUBLAS_OP_N, originM, originN, 
        &alpha, (const cuComplex*)A, originM, 
        (const cuComplex*)x, 1, &beta, (cuComplex*)y, 1));    
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
    }
    cublasDestroy(handle);
    double bytes = sizeof(complex<float>) * 
    (originM * originN + originM + originN) * 
    (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
    // CopyDataB2HD(hy, (complex<float>*)y, originM);
    // for(size_t i=0; i<originM; i++){
    //     if(hy[i].real() != -9801){
    //         cout << "wrong results " << endl;
    //     }
    //     if(hy[i].imag() != 29403){
    //         cout << "wrong results " << endl;
    //     }
    // }
}


BENCHMARK_REGISTER_F(SeismicFixture, DenseTest)
->Unit(benchmark::kMicrosecond)
->UseManualTime()
// ->Iterations(200)
->Repetitions(1);

// for input of benchmark
int main(int argc, char **argv){
    ::benchmark::Initialize(&argc, argv);
    for(int i=1; i<argc; i++){
        string tmp = string(argv[i]);
        if(tmp.substr(0,2) != "--") continue;
        else{
            int s = 0;
            while(s < tmp.size() && tmp[s] != '=') s++;
            if(s == tmp.size()) continue;
            inputmap[tmp.substr(2,s-2)] = tmp.substr(s+1,tmp.size()-2-1);
        }
    }
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}


