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
using namespace tlrmvm;

#define F first
#define S second
#define PB push_back

#define REP(i,a,b) for (int i = a; i < b; i++)


unordered_map<string, string> inputmap;


class SeismicFixture : public benchmark::Fixture {

public:


    void loadint8(){
        for(int i=0; i<freqlist.size(); i++){
            complex<float> *DataAv;
            complex<float> *DataAu;
            complex<float> *Datax;
            int x = freqlist[i];
            auto granksum = Rmats[freqlist[i]].Sum();
            globalranksum.push_back(granksum);
            ReadSeismicBinary(datafolder+"/V", &DataAv, granksum * nb, acc ,nb, freqlist[i]);
            ReadSeismicBinary(datafolder+"/U", &DataAu, granksum * nb, acc ,nb, freqlist[i]);
            ReadSeismicBinaryX(datafolder, &Datax, originN, acc, nb, freqlist[i]);
            FP32pointers.push_back(GPUTLRMVMPointer<float>(streamsize, true)); // use complex split for half and int8 precision
            FP16pointers.push_back(GPUTLRMVMPointer<half>(streamsize, true)); // use complex split for half and int8 precision
            INT8pointers.push_back(GPUTLRMVMPointer<int8_t>(streamsize, true)); // use complex split for half and int8 precision
            for(int si=0; si < streamsize; si++){
                INT8pointers.back().InitCPUPointer(DataAv, DataAu, Datax, Rmats[freqlist[i]], INT8Rmats[x][si], si, nb);
                INT8pointers.back().InitGPUPointer(si);
            }
            delete[] DataAv;delete[] DataAu;delete[] Datax;
        }   
    }

    void loadFP16(){
        for(int i=0; i<freqlist.size(); i++){
            complex<float> *DataAv;
            complex<float> *DataAu;
            complex<float> *Datax;
            int x = freqlist[i];
            auto granksum = Rmats[freqlist[i]].Sum();
            globalranksum.push_back(granksum);
            ReadSeismicBinary(datafolder+"/V", &DataAv, granksum * nb, acc ,nb, freqlist[i]);
            ReadSeismicBinary(datafolder+"/U", &DataAu, granksum * nb, acc ,nb, freqlist[i]);
            ReadSeismicBinaryX(datafolder, &Datax, originN, acc, nb, freqlist[i]);
            FP32pointers.push_back(GPUTLRMVMPointer<float>(streamsize, true)); // use complex split for half and int8 precision
            FP16pointers.push_back(GPUTLRMVMPointer<half>(streamsize, true)); // use complex split for half and int8 precision
            INT8pointers.push_back(GPUTLRMVMPointer<int8_t>(streamsize, true)); // use complex split for half and int8 precision
            for(int si=0; si < streamsize; si++){
                FP16pointers.back().InitCPUPointer(DataAv, DataAu, Datax, Rmats[freqlist[i]], FP16Rmats[x][si], si, nb);
                FP16pointers.back().InitGPUPointer(si);
            }
            delete[] DataAv;delete[] DataAu;delete[] Datax;
        }   
    }

    void loadFP32(){
        for(int i=0; i<freqlist.size(); i++){
            complex<float> *DataAv;
            complex<float> *DataAu;
            complex<float> *Datax;
            int x = freqlist[i];
            auto granksum = Rmats[freqlist[i]].Sum();
            globalranksum.push_back(granksum);
            ReadSeismicBinary(datafolder+"/V", &DataAv, granksum * nb, acc ,nb, freqlist[i]);
            ReadSeismicBinary(datafolder+"/U", &DataAu, granksum * nb, acc ,nb, freqlist[i]);
            ReadSeismicBinaryX(datafolder, &Datax, originN, acc, nb, freqlist[i]);
            FP32pointers.push_back(GPUTLRMVMPointer<float>(streamsize, true)); // use complex split for half and int8 precision
            FP16pointers.push_back(GPUTLRMVMPointer<half>(streamsize, true)); // use complex split for half and int8 precision
            INT8pointers.push_back(GPUTLRMVMPointer<int8_t>(streamsize, true)); // use complex split for half and int8 precision
            for(int si=0; si < streamsize; si++){
                FP32pointers.back().InitCPUPointer(DataAv, DataAu, Datax, Rmats[freqlist[i]], FP32Rmats[x][si], si, nb);
                FP32pointers.back().InitGPUPointer(si);
            }
            delete[] DataAv;delete[] DataAu;delete[] Datax;
        }   
    }


    void SetUp(const ::benchmark::State& state) {
        
        // datafolder = inputmap["datafolder"];
        // acc = inputmap["acc"];
        // string freqstr = inputmap["freqlist"];
        // if(freqstr == "full"){
        //     for(int i=0; i<3; i++){
        //         freqlist.push_back(i);
        //     }
        // }else{
        //     freqlist.push_back(atoi(freqstr.c_str()));
        // }

        // nb = atoi(inputmap["nb"].c_str());
        // originM = atoi(inputmap["M"].c_str());
        // originN = atoi(inputmap["N"].c_str());
        // usestream = atoi(inputmap["usestream"].c_str());
        // streamsize = atoi(inputmap["streamsize"].c_str());
        // masktype = inputmap["masktype"];
        // if(masktype == "Banded"){
        //     bandlength = atoi(inputmap["bandlength"].c_str());
        // }else if(masktype == "userinput"){
        //     maskfilename = inputmap["maskfile"];
        // }
        // paddingM = CalculatePadding(originM, nb);
        // paddingN = CalculatePadding(originN, nb);
        // Mtglobal = paddingM / nb;
        // Ntglobal = paddingN / nb;
        // // stream init
        // for(int i=0; i<streamsize; i++) 
        // streamexecsize.push_back(Ntglobal / streamsize);
        // for(int i=0; i<Ntglobal % streamsize; i++) streamexecsize[i]++;
        // streamexecoffset.clear();
        // streamexecoffset.push_back(0);
        // for(int i=1; i<streamsize;i++) 
        // streamexecoffset.push_back(streamexecsize[i-1] + streamexecoffset[i-1]);
        // auto maskmat = Matrix<int>(Mtglobal, Ntglobal);
        // maskmat.Fromfile(PathJoin({datafolder, "maskmatband20.bin"}), Mtglobal, Ntglobal);
        // maskmat.Fill(4);
        //     // split precision
        // for(auto x : freqlist){
        //     int *DataR;
        //     ReadSeismicBinary(datafolder, &DataR, Mtglobal * Ntglobal, acc, nb, x);
        //     Matrix<int> Rmat = Matrix<int>(DataR, Mtglobal, Ntglobal);
        //     Rmats[x] = Rmat;        
        //     // split to each precision
        //     for(auto y : streamexecsize){
        //         FP32Rmats[x].push_back(Matrix<int>(Mtglobal, y));
        //         FP16Rmats[x].push_back(Matrix<int>(Mtglobal, y));
        //         INT8Rmats[x].push_back(Matrix<int>(Mtglobal, y));
        //     }
        //     for(int i=0; i<maskmat.Row(); i++){
        //         int streamidcnt = 0;
        //         for(int j=0; j<maskmat.Col(); j++){
        //             int curstream = streamidcnt % streamsize;
        //             int currank = Rmat.GetElem(i,j);
        //             switch (maskmat.GetElem(i,j))
        //             {
        //             case 4:/* FP32 */
        //                 FP32Rmats[x][curstream].SetElem(i, j / streamsize, currank);
        //                 break;
        //             case 2:/* FP16 */
        //                 FP16Rmats[x][curstream].SetElem(i, j / streamsize, currank);
        //                 break;
        //             case 1:/* INT8 */
        //                 INT8Rmats[x][curstream].SetElem(i, j / streamsize, currank);
        //                 break;
        //             default:
        //                 break;
        //             }
        //             streamidcnt++;
        //         }
        //     }
        // }
    }

    void TearDown(const ::benchmark::State& state) {

    }

    string datafolder;
    string acc;
    vector<int> freqlist;
    string masktype;
    string maskfilename;
    string densetest;
    int bandlength;
    int nb;
    int originM;
    int originN;
    int streamsize;
    bool usestream;
    int Mtglobal;
    int Ntglobal;
    int paddingM;
    int paddingN;
    vector<size_t> Ntlocals;
    vector<size_t> streamexecsize;
    vector<size_t> streamexecoffset;
    vector<size_t> globalranksum;
    unordered_map< int, Matrix<int> > Rmats;  // key is freqid
    unordered_map< int, vector<Matrix<int>> > Maskmats;  // key is stream
    unordered_map< int, vector<Matrix<int>> > FP32Rmats; // key is freqid
    unordered_map< int, vector<Matrix<int>> > FP16Rmats; // key is freqid
    unordered_map< int, vector<Matrix<int>> > INT8Rmats; // key is freqid
    GPUTLRMVMPointer<float> DenseFP32exp;
    GPUTLRMVMPointer<float> DenseFP16exp;
    // For each frequency matrix, you have one combination
    vector<GPUTLRMVMPointer<float>> FP32pointers;
    vector<GPUTLRMVMPointer<half>> FP16pointers;
    vector<GPUTLRMVMPointer<int8_t>> INT8pointers;

};

BENCHMARK_DEFINE_F(SeismicFixture, DenseFP32Test)(benchmark::State& state) {
    DenseFP32exp = GPUTLRMVMPointer<float>((int)state.range(0),(int)state.range(0));
    DenseFP32exp.InitDenseFP32Pointer();
    cublasHandle_t handle;
    cublasCreate(&handle);
    // timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cuComplex alpha; 
    cuComplex beta;
    alpha.x = 1.0;
    alpha.y = 0.0;
    beta.x = 0.0;
    beta.y = 0.0;
    vector<double> rawtime;
    for(auto st : state){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        CUBLASCHECK(cublasgemv(handle, CUBLAS_OP_N, 
        DenseFP32exp.originM, DenseFP32exp.originN, 
        &alpha, DenseFP32exp.A, DenseFP32exp.originM, 
        DenseFP32exp.x, 1, &beta, DenseFP32exp.y, 1));
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
    }
    CopyDataB2HD(DenseFP32exp.hy, reinterpret_cast<complex<float>*>(DenseFP32exp.y), DenseFP32exp.originM);

    for(int i=0; i<originM; i++){
        if(DenseFP32exp.hy[i].real() != -DenseFP32exp.originN && DenseFP32exp.hy[i].imag() != 3 * DenseFP32exp.originN){
            cout << i << " wrong res " << DenseFP32exp.hy[i].real() << ", " << DenseFP32exp.hy[i].imag() << endl;
            break;
        }
    }
    double bytes = sizeof(complex<float>) * 
    (DenseFP32exp.originM * DenseFP32exp.originN + DenseFP32exp.originM + DenseFP32exp.originN) * 
    (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
    cublasDestroy(handle);
    DenseFP32exp.FreeDenseFP32Pointer();
}

BENCHMARK_REGISTER_F(SeismicFixture, DenseFP32Test)
->Unit(benchmark::kMicrosecond)
->RangeMultiplier(2)->Range(256, 256<<4)
// ->Iterations(1)
->UseManualTime()
->Repetitions(1);



BENCHMARK_DEFINE_F(SeismicFixture, DenseFP16Test)(benchmark::State& state) {
    DenseFP16exp = GPUTLRMVMPointer<float>((int)state.range(0),(int)state.range(0));
    DenseFP16exp.InitDenseFP16Pointer();
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);
    // timer 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    half alpha; 
    half beta;
    alpha = (half)1.0;
    beta = (half)0.0;
    vector<double> rawtime;
    size_t N ;
    size_t M ;
    N = M = DenseFP16exp.originN;
    for(auto st : state){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        CUBLASCHECK(cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        DenseFP16exp.originM, 2, DenseFP16exp.originN, 
        &alpha, DenseFP16exp.Areal_half, DenseFP16exp.originM, 
        DenseFP16exp.x_half, N, &beta, DenseFP16exp.yreal_half, M));
        CUBLASCHECK(cublasgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        DenseFP16exp.originM, 2, DenseFP16exp.originN, 
        &alpha, DenseFP16exp.Aimag_half, DenseFP16exp.originM, 
        DenseFP16exp.x_half, N, &beta, DenseFP16exp.yimag_half, M));
        merge_half_realimag(DenseFP16exp.yreal_half, DenseFP16exp.yimag_half, M, stream);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
    }
    CopyDataB2HD(DenseFP16exp.hyreal_half, DenseFP16exp.yreal_half, M);
    CopyDataB2HD(DenseFP16exp.hyimag_half, DenseFP16exp.yimag_half, M);
    REP(i,0,M){
        DenseFP16exp.hy[i] = complex<float>((float)DenseFP16exp.hyreal_half[i],(float)DenseFP16exp.hyimag_half[i]);
    }
    double bytes = sizeof(half) * 2 * 
    (DenseFP16exp.originM * DenseFP16exp.originN + DenseFP16exp.originM + DenseFP16exp.originN) * 
    (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
    cublasDestroy(handle);
    DenseFP16exp.FreeDenseFP16Pointer();
}

BENCHMARK_REGISTER_F(SeismicFixture, DenseFP16Test)
->Unit(benchmark::kMicrosecond)
->RangeMultiplier(2)->Range(256, 256<<4)
->UseManualTime()
->Repetitions(1);



BENCHMARK_DEFINE_F(SeismicFixture, DenseINT8Test)(benchmark::State& state) {
    auto Denseint8exp = GPUTLRMVMPointer<float>((int)state.range(0),(int)state.range(0));
    Denseint8exp.InitDenseINT8Pointer();
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);
    // timer 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float alpha; 
    float beta;
    alpha = (float)1.0;
    beta = (float)0.0;
    vector<double> rawtime;
    size_t N ;
    size_t M ;
    N = M = Denseint8exp.originN;
    for(auto st : state){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        M, 2, N, &alpha, (const int8_t*)Denseint8exp.i8Areal,  CUDA_R_8I, M, 
        (const int8_t*)Denseint8exp.i8xreal, CUDA_R_8I, N, &beta, 
        Denseint8exp.i8fyreal,  CUDA_R_32F, M));
        CUBLASCHECK(cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        M, 2, N, &alpha, (const int8_t*)Denseint8exp.i8Aimag, CUDA_R_8I, M, 
        (const int8_t*)Denseint8exp.i8ximag, CUDA_R_8I, N, &beta,
        Denseint8exp.i8fyimag, CUDA_R_32F, M));
        merge_float_realimag(Denseint8exp.i8fyreal, Denseint8exp.i8fyimag, 
        Denseint8exp.i8cyout, M, stream);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
    }
    // CopyDataB2HD(Denseint8exp.i8hfyreal, Denseint8exp.i8fyreal, 2*M);
    // CopyDataB2HD(Denseint8exp.i8hfyimag, Denseint8exp.i8fyimag, 2*M);
    CopyDataB2HD(Denseint8exp.i8hcyout, reinterpret_cast<complex<float>*>(Denseint8exp.i8cyout), M);
    // REP(i,0,M){
    //     Denseint8exp.hy[i] = complex<float>((float)Denseint8exp.i8hfyreal[i],(float)Denseint8exp.i8hfyimag[i]);
    // }
    for(int i=0; i<originM; i++){
        // if (Denseint8exp.i8hfyreal[i] != originM){
        //     cout << "real wrong " << i << " , " <<  Denseint8exp.i8hfyreal[i] << endl;
        //     break;
        // }
        // if (Denseint8exp.i8hfyimag[i] != 2*originM){
        //     cout << "imag wrong " << endl;
        //     break;
        // }
        if(Denseint8exp.i8hcyout[i].real() != -originN && Denseint8exp.i8hcyout[i].imag() != 3 * originM){
            cout << "real value -9801 and 29403. Wrong res for Half, but it's okay, " << Denseint8exp.hy[i].real() << ", " << Denseint8exp.hy[i].imag() << endl;
            break;
        }
    }
    double bytes = sizeof(int8_t) * 2 * 
    (Denseint8exp.originM * Denseint8exp.originN + Denseint8exp.originM + Denseint8exp.originN) * 
    (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
    cublasDestroy(handle);
    Denseint8exp.FreeDenseINT8Pointer();
}

BENCHMARK_REGISTER_F(SeismicFixture, DenseINT8Test)
->Unit(benchmark::kMicrosecond)
->Arg(9984)
->UseManualTime()
->Repetitions(1);



BENCHMARK_DEFINE_F(SeismicFixture, SPFP32_Pointer_tlrmvmTest)(benchmark::State& state){



}

// BENCHMARK_REGISTER_F(SeismicFixture, SPFP32_Pointer_tlrmvmTest)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(1)
// ->UseManualTime()
// ->Repetitions(1);


BENCHMARK_DEFINE_F(SeismicFixture, SPFP32_NoDM_tlrmvmTest)(benchmark::State& state) {

    // Matrix<complex<float>> yv;
    // yv.Fromfile("yvout.bin", 9884, 1);
    // freq = 100
    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);
    for(int fqi=0; fqi < freqlist.size(); fqi++){
        for(int si=0; si < streamsize; si++){
            FP32pointers[fqi].CopyCPU2GPU(si, streamptr[si]);
        }
    }
    cudaDeviceSynchronize();
    for(int fqi=0; fqi < freqlist.size(); fqi++){
        for(int si=0; si < streamsize; si++){
            FP32pointers[fqi].Phase1(si, streamptr[si], cublashandleptr[si]);
        }
    }
    for (auto _ : state) {


    }
    cudaDeviceSynchronize();
    for(int fqi=0; fqi < freqlist.size(); fqi++){
        for(int si=0; si < streamsize; si++){
            FP32pointers[fqi].CopyGPU2CPU(si, streamptr[si]);
        }
    }
    for(int i=0; i<streamsize; i++) cublasDestroy(cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cudaStreamDestroy(streamptr[i]);

}

// BENCHMARK_REGISTER_F(SeismicFixture, SPFP32_NoDM_tlrmvmTest)
// ->Unit(benchmark::kMicrosecond)
// ->UseManualTime()
// ->Iterations(1)
// ->Repetitions(1);


// BENCHMARK_DEFINE_F(SeismicFixture, SPFP32_DM_tlrmvmTest)(benchmark::State& state) {

//     Matrix<complex<float>> yv;
//     yv.Fromfile("yvout.bin", 9884, 1);
//     // freq = 100
//     cudaStream_t * streamptr = new cudaStream_t[streamsize];
//     cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
//     for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
//     for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
//     for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

//     for (auto _ : state) {
//         int workcnt = 0;
//         for(auto x : FP32pointers){
//             int workid = workcnt % streamsize;
//             x.CopyCPU2GPU(streamptr[workid]);
            
//             x.CopyGPU2CPU(streamptr[workid]);
//             workcnt += 1;
//         }
//     }
    
//     for(int i=0; i<streamsize; i++) cublasDestroy(cublashandleptr[i]);
//     for(int i=0; i<streamsize; i++) cudaStreamDestroy(streamptr[i]);

// }

// BENCHMARK_REGISTER_F(SeismicFixture, SPFP32_DM_tlrmvmTest)
// ->Unit(benchmark::kMicrosecond)
// ->UseManualTime()
// ->Iterations(1)
// ->Repetitions(1);


// for input of benchmark
int main(int argc, char **argv) {
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