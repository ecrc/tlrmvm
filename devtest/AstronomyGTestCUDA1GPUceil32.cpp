#include <stdio.h>
#include <algorithm>
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
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h> // if you need CUBLAS v2, include before magma.h
#include "nvToolsExt.h"

#ifdef USE_NVTX
#include "nvToolsExt.h"

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
    int color_id = cid; \
    color_id = color_id%num_colors;\
    nvtxEventAttributes_t eventAttrib = {0}; \
    eventAttrib.version = NVTX_VERSION; \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType = NVTX_COLOR_ARGB; \
    eventAttrib.color = colors[color_id]; \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
    eventAttrib.message.ascii = name; \
    nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif

#include "common/Common.h"
#include "common/AppUtil.h"
#include "tlrmvm/Tlrmvm.h"
#include "tlrmvm/Tlrmvmcuda.h"

#include "benchmark/benchmark.h"
#include "AstronomyUtil.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace std;
using namespace tlrmat;
using namespace tlrmvm;
using namespace cudatlrmat;
using namespace cudatlrmvm;
using ::testing::Pointwise;
using ::testing::NanSensitiveFloatNear;

vector<string> g_command_line_arg_vec;

class AstronomySingleGPUTest: public testing::Test {

protected:

    void SetUp() {

        if(g_command_line_arg_vec.size() < 6){
            cout << "not enough args " << endl;
            exit(1);
        }
        datafolder = g_command_line_arg_vec[0];
        acc = g_command_line_arg_vec[1];
        id = g_command_line_arg_vec[2];
        nb = atoi(g_command_line_arg_vec[3].c_str());
        originM = atoi(g_command_line_arg_vec[4].c_str());
        originN = atoi(g_command_line_arg_vec[5].c_str());
        STREAMSIZE = atoi(g_command_line_arg_vec[6].c_str());
        paddingM = CalculatePadding(originM, nb);
        paddingN = CalculatePadding(originN, nb);
        Mtglobal = paddingM / nb;
        Ntglobal = paddingN / nb;

        // load Data
        float *DataAv, *DataAu;
        int *DataR;        
        ReadAstronomyBinary(datafolder, &DataR, Mtglobal * Ntglobal, acc, nb, id);
        Rmat = Matrix<int>(DataR, Mtglobal, Ntglobal);
        granksum = Rmat.Sum();
        ReadAstronomyBinary(datafolder+"/V", &DataAv, granksum * nb, acc ,nb, id);
        ReadAstronomyBinary(datafolder+"/U", &DataAu, granksum * nb, acc ,nb, id);

        /**
         * Phase 1 preparation
         */
        // calculate layout for Av: Phase 1
        vector<int> colrsum = Rmat.ColSum();
        float val = 0.0;
        for(int i=0; i<Ntglobal; i++){
            AvMs.push_back( ceil32(colrsum[i]) );
            ceil32colsum.push_back(ceil32(colrsum[i]));
            AvKs.push_back(nb);
            AvNs.push_back(1);
        }
        // Matrix<int> ceil32Rmat(ceil32colsum, Mtglobal, Ntglobal);

        // get host memory
        GetHostMemoryBatched(&hAv, &hx, &hyv, &hAvbp, &hxbp, &hyvbp, AvMs, AvKs, AvNs, val);
        CaluclateTotalElements(AvMs, AvKs, AvNs, Avtotalelems, xtotalelems, yvtotalelems);
        
        
        size_t DataAvacc = 0;
        size_t hAvacc = 0;

        for(int i=0; i<Ntglobal; i++){
            for(int j=0; j<nb; j++){
                memcpy(hAv + hAvacc, DataAv + DataAvacc, sizeof(float) * Rmat.ColSum()[i]);
                DataAvacc += Rmat.ColSum()[i];
                hAvacc += AvMs[i];
            }    
        }
        delete[] DataAv;
        // get random X
        #pragma omp parallel for 
        for(int i=0; i<paddingN; i++){
            hx[i] = 1.0;
        }
        float avgx256 = 0.0;
        for(int i=0; i<256; i++) avgx256 += hx[i];
        cout << avgx256 << endl;
        avgx256 = 0.0;
        for(int i=0; i<256; i++) avgx256 += hAv[i*AvMs[0]];
        cout << avgx256 << endl;
        
        // get device memory
        GetDeviceMemoryBatched(&Av, &x, &yv, 
        &Avbatchpointer, &xbatchpointer, &yvbatchpointer, AvMs, AvKs, AvNs);
        cudatlrmat::CopyDataB2HD(Av, x, yv, hAv, hx, hyv, AvMs, AvKs, AvNs);
        
        GetDeviceMemory(&Avcolslda, Ntglobal);
        CopyDataB2HD(Avcolslda, ceil32colsum.data(), ceil32colsum.size());
        vector<int> tmpval = PrefixSum(ceil32colsum.data(), ceil32colsum.size());
        GetDeviceMemory(&Avcolsldaprefix, Ntglobal);
        CopyDataB2HD(Avcolsldaprefix, tmpval.data(), tmpval.size());
        granksum = 0;
        for(auto x : ceil32colsum) granksum += x;
        cout << "granksum "<< granksum << endl;
        cout <<" origin ranksum " << Rmat.Sum() << endl;
        // GetDeviceMemory(&Avrowsprefix, Rmat.Col()+1);
        // auto Avrowsprefix_vec = PrefixSum(Rmat.ColSum().data(), Rmat.Col());
        // Avrowsprefix_vec.push_back(Rmat.Sum());
        // CopyDataB2HD(Avrowsprefix, Avrowsprefix_vec.data(), Rmat.Col()+1);

        // Avbatchpointer_h = new float*[Ntglobal];
        // xbatchpointer_h = new float*[Ntglobal];
        // yvbatchpointer_h = new float*[Ntglobal];
        // cudatlrmat::CopyDataB2HD(Avbatchpointer_h, Avbatchpointer, Ntglobal);
        // cudatlrmat::CopyDataB2HD(xbatchpointer_h, xbatchpointer, Ntglobal);
        // cudatlrmat::CopyDataB2HD(yvbatchpointer_h, yvbatchpointer, Ntglobal);

        // /**
        //  * Phase 3 preparation
        //  */
        // vector<int> rowrsum = Rmat.RowSum();
        // for(int i=0; i<Mtglobal; i++){
        //     AuMs.push_back(nb);
        //     AuKs.push_back( (size_t)rowrsum[i] );
        //     AuNs.push_back(1);
        // }
        // GetHostMemoryBatched(&hAu, &hyu, &hy, 
        // &hAubp, &hyubp, &hybp, AuMs, AuKs, AuNs, val);
        // CaluclateTotalElements(AuMs, AuKs, AuNs, Autotalelems, yutotalelems, ytotalelems);
        // memcpy(hAu, DataAu, sizeof(float) * Autotalelems);
        // delete[] DataAu;

        // GetDeviceMemoryBatched(&Au, &yu, &y, 
        // &Aubatchpointer, &yubatchpointer, &ybatchpointer, AuMs, AuKs, AuNs);
        // Aubatchpointer_h = new float*[Mtglobal];
        // yubatchpointer_h = new float*[Mtglobal];
        // ybatchpointer_h = new float*[Mtglobal];
        // cudatlrmat::CopyDataB2HD(Aubatchpointer_h, Aubatchpointer, Mtglobal);
        // cudatlrmat::CopyDataB2HD(yubatchpointer_h, yubatchpointer, Mtglobal);
        // cudatlrmat::CopyDataB2HD(ybatchpointer_h, ybatchpointer, Mtglobal);

        // cudatlrmat::CopyDataB2HD(Av, x, yv, hAv, hx, hyv, AvMs, AvKs, AvNs);
        // cudatlrmat::CopyDataB2HD(Au, yu, y, hAu, hyu, hy, AuMs, AuKs, AuNs);

        // // get phase2 offset
        // GetDeviceMemory(&offsetphase2, (size_t)granksum);
        // tlrmvm::CalculatePhase2Offset(&offsetphase2_h, Mtglobal, Ntglobal, Rmat);
        // cudatlrmat::CopyDataB2HD(offsetphase2, offsetphase2_h, granksum);

        alpha = 1.0;
        beta = 0.0;

        // bytesprocessed = 0.0;
        // unsigned long int phase1 = granksum*nb + paddingN + granksum;
        // unsigned long int shuffle = 2 * granksum;
        // unsigned long int phase2 = granksum*nb + granksum + paddingM;
        // bytesprocessed = sizeof(float) * (phase1 + shuffle + phase2);
        // timestat.clear();
        // LOOPSIZE = 5000;
        // SKIPROUND = 1000;
    }

    void TearDown() {

        // FreeHostMemoryBatched(hAv, hx, hyv, hAvbp, hxbp, hyvbp);
        // FreeHostMemoryBatched(hAu, hyu, hy, hAubp, hyubp, hybp);
        // FreeDeviceMemoryBatched(Av,x,yv,Avbatchpointer, xbatchpointer,yvbatchpointer);
        // FreeDeviceMemoryBatched(Au,yu,y,Aubatchpointer, yubatchpointer,ybatchpointer);
        // delete[] offsetphase2_h;
        // FreeDeviceMemory(offsetphase2);
        // delete[] Avbatchpointer_h; delete[] xbatchpointer_h; delete[] yvbatchpointer_h;
        // delete[] Aubatchpointer_h; delete[] yubatchpointer_h; delete[] ybatchpointer_h;

    }

    void getbandstat(){
        bandstat.clear();
        std::sort(timestat.begin(), timestat.end());
        for(auto x : timestat){
            bandstat.push_back( bytesprocessed / x * 1e-9);
        }
    }
    vector<double> getaveragestat(){
        getbandstat();
        double medianbd(0.0), mediantime(0.0);
        mediantime = (double)timestat[timestat.size()/2];
        medianbd = (double)bandstat[timestat.size()/2];
        // for(auto x : timestat) avgtime += x;
        // for(auto x : bandstat) avgbd += x;
        double sz = timestat.size();
        double maxval, minval;
        minval = maxval = timestat[0];
        for(auto x : timestat){
            maxval = fmax(maxval, x);
            minval = fmin(minval, x);
        }
        return {mediantime*1e6, medianbd, maxval*1e6, minval*1e6};
    }

    void displayavgstat(){
        vector<double> avgstat = getaveragestat();
        cout<< "Max Time: " << avgstat[2] << ", Min Time: " << avgstat[3] \
        << " Averge Time: " << avgstat[0] << " us, " << "Average Bandwidth: " << avgstat[1] \
        << " GB/s." << endl;
    }
    int * Avcolsldaprefix;
    int * Avcolslda;
    float *hAv;
    float *hx;
    float *hyv;
    float **hAvbp;
    float **hxbp;
    float **hyvbp;

    float *Av;
    float *x;
    float *yv;
    float **Avbatchpointer;
    float **xbatchpointer;
    float **yvbatchpointer;
    float **Avbatchpointer_h;
    float **xbatchpointer_h;
    float **yvbatchpointer_h;

    float *hAu;
    float *hyu;
    float *hy; 
    float **hAubp;
    float **hyubp;
    float **hybp;

    float *Au;
    float *yu;
    float *y;
    float **Aubatchpointer;
    float **yubatchpointer;
    float **ybatchpointer;
    float **Aubatchpointer_h;
    float **yubatchpointer_h;
    float **ybatchpointer_h;

    vector<size_t> AvMs;
    vector<size_t> AvKs;
    vector<size_t> AvNs;
    vector<size_t> AuMs;
    vector<size_t> AuKs;
    vector<size_t> AuNs;

    size_t Avtotalelems;
    size_t xtotalelems;
    size_t yvtotalelems;
    size_t Autotalelems;
    size_t yutotalelems;
    size_t ytotalelems;

    unsigned long int* offsetphase2;
    unsigned long int* offsetphase2_h;
    float alpha;
    float beta;

    size_t Mtglobal;
    size_t Ntglobal;
    
    int paddingM;
    int paddingN;

    double bytesprocessed;
    vector<double> timestat;
    vector<double> bandstat;
    vector<int> ceil32colsum;
    int LOOPSIZE;
    int SKIPROUND;
    int STREAMSIZE;
    // input
    string datafolder;
    string acc;
    string id;
    int nb;
    int originM;
    int originN;
    Matrix<int> Rmat;
    Matrix<int> Rtransmat;
    vector<int> Rmatprefix;
    vector<int> Rtransprefix;
    unsigned long int granksum;

};

TEST_F(AstronomySingleGPUTest, Phase1_Correctness){
    int LOOPSIZE = 10;
    // cublasHandle_t cublashandle;
    // cublasCreate_v2(&cublashandle);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for(int i=0; i<LOOPSIZE; i++){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        phase1driver(Av, x, yv, Avcolsldaprefix, Avcolslda, nb, Ntglobal, granksum);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cout << milliseconds << endl;
    }
    

    AstronomyPCMatrix astropcmat(datafolder, acc, nb, id, originM, originN);
    Matrix<float> xvec(hx, paddingN, 1);
    astropcmat.setX(xvec);
    Matrix<float> yv_pc = astropcmat.Phase1();
    vector<float> hyvresult(Rmat.Sum(),0);    
    cudatlrmat::CopyDataB2HD(hyvresult.data(), yv, Rmat.Sum());
    int offsethyv = 0;
    int offsetyv = 0;
    for(int i=0; i<Ntglobal; i++){
        cout << "x val " << ceil32colsum[i] << endl;
        double err = NetlibError(hyvresult.data()+offsethyv, 
        yv_pc.RawPtr()+offsetyv, Rmat.ColSum()[i]);
        // cout << hyvresult.data()[2] << "," << yv_pc.RawPtr()[2] << endl;
        offsethyv += ceil32colsum[i];
        offsetyv += Rmat.ColSum()[i];
        std::cerr << "\033[32m[ APP INFO ] Phase 1 error = " << err << "\033[0m"<< endl;
    }
    

}

// TEST_F(AstronomySingleGPUTest, Phase2_Correctness){
//     int STREAMSIZE = 4;
//     int LOOPSIZE = 10;
//     cublasHandle_t cublashandle;
//     cublasCreate_v2(&cublashandle);

//     cudaDeviceSynchronize();

//     for(int i=0; i<Ntglobal; i++){
//         CUBLASCHECK(
//         cublasgemv(cublashandle, CUBLAS_OP_N,
//         AvMs[i], AvKs[i],
//         &alpha, (const float*)Avbatchpointer_h[i], AvMs[i], 
//         (const float*)xbatchpointer_h[i], 1, &beta, 
//         yvbatchpointer_h[i], 1);
//         );
//     }
//     cudaDeviceSynchronize();
//     phase2dirver(yu, yv, granksum, offsetphase2);

//     cudaDeviceSynchronize();

//     AstronomyPCMatrix astropcmat(datafolder, acc, nb, id, originM, originN);
//     Matrix<float> xvec(hx, paddingN, 1);
//     astropcmat.setX(xvec);
//     Matrix<float> yv_pc = astropcmat.Phase1();
//     Matrix<float> yu_pc = astropcmat.Phase2();
//     vector<float> hyuresult(Rmat.Sum(),0);
//     cudatlrmat::CopyDataB2HD(hyuresult.data(), yu, Rmat.Sum());

//     double err = NetlibError(hyuresult.data(), yu_pc.RawPtr(), granksum);
//     std::cerr << "\033[32m[ APP INFO ] Phase 2 error = " << err << "\033[0m"<< endl;
// }

// TEST_F(AstronomySingleGPUTest, Phase3_Correctness){
//     cublasHandle_t cublashandle;
//     cublasCreate_v2(&cublashandle);
//     cudaDeviceSynchronize();
//     for(int i=0; i<Ntglobal; i++){
//         CUBLASCHECK(
//         cublasgemv(cublashandle, CUBLAS_OP_N,
//         AvMs[i], AvKs[i],
//         &alpha, (const float*)Avbatchpointer_h[i], AvMs[i], 
//         (const float*)xbatchpointer_h[i], 1, &beta, 
//         yvbatchpointer_h[i], 1);
//         );
//     }
//     cudaDeviceSynchronize();
//     phase2dirver(yu, yv, granksum, offsetphase2);
//     cudaDeviceSynchronize();
//     for(int i=0; i<Mtglobal; i++){
//         CUBLASCHECK(
//         cublasgemv(cublashandle, CUBLAS_OP_N,
//         AuMs[i], AuKs[i],
//         &alpha, (const float*)Aubatchpointer_h[i], AuMs[i], 
//         (const float*)yubatchpointer_h[i], 1, &beta, 
//         ybatchpointer_h[i], 1);
//         );
//     }

//     cudaDeviceSynchronize();

//     AstronomyPCMatrix astropcmat(datafolder, acc, nb, id, originM, originN);
//     Matrix<float> xvec(hx, paddingN, 1);
//     astropcmat.setX(xvec);
//     Matrix<float> yv_pc = astropcmat.Phase1();
//     Matrix<float> yu_pc = astropcmat.Phase2();
//     Matrix<float> y_pc = astropcmat.Phase3();
//     vector<float> hyresult(paddingM,0);
//     cudatlrmat::CopyDataB2HD(hyresult.data(), y, paddingM);

//     double err = NetlibError(hyresult.data(), y_pc.RawPtr(),originM);
//     std::cerr << "\033[32m[ APP INFO ] Phase 3 error = " << err << "\033[0m"<< endl;
// }




class MyTestEnvironment : public testing::Environment {
 public:
  explicit MyTestEnvironment(const vector<string> &command_line_arg) {
    g_command_line_arg_vec = command_line_arg;
  }
};

int main(int argc, char **argv) {
  vector<string> command_line_arg_vec;
  testing::InitGoogleTest(&argc, argv);
  for(int i=0; i<argc-1; i++){
      char tmp[200];
      sprintf(tmp, "%s", argv[i+1]);
      command_line_arg_vec.push_back(string(tmp));
  }
  testing::AddGlobalTestEnvironment(new MyTestEnvironment(command_line_arg_vec));
  return RUN_ALL_TESTS();
}