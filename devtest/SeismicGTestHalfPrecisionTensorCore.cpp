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
#include <cuda_fp16.h>
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

    size_t ceil32(size_t val){
        return (val / 32 + 1) * 32;
    }

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
            AvKs.push_back(nb);
            AvNs.push_back(1);
        }
        // get host memory
        GetHostMemoryBatched(&hAv, &hx, &hyv, &hAvbp, &hxbp, &hyvbp, AvMs, AvKs, AvNs, val);
        CaluclateTotalElements(AvMs, AvKs, AvNs, Avtotalelems, xtotalelems, yvtotalelems);
        size_t DataAvacc = 0;
        size_t hAvacc = 0;
        for(int i=0; i<Ntglobal; i++){
            memcpy(hAv + hAvacc, DataAv + DataAvacc, sizeof(float) * Avtotalelems);
            DataAvacc += Rmat.ColSum()[i];
            hAvacc += AvMs[i] * nb;
        }
        
        delete[] DataAv;
        // get device memory
        // GetDeviceMemoryBatched(&Av, &x, &yv, 
        // &Avbatchpointer, &xbatchpointer, &yvbatchpointer, AvMs, AvKs, AvNs);
        // Avbatchpointer_h = new float*[Ntglobal];
        // xbatchpointer_h = new float*[Ntglobal];
        // yvbatchpointer_h = new float*[Ntglobal];
        // cudatlrmat::CopyDataB2HD(Avbatchpointer_h, Avbatchpointer, Ntglobal);
        // cudatlrmat::CopyDataB2HD(xbatchpointer_h, xbatchpointer, Ntglobal);
        // cudatlrmat::CopyDataB2HD(yvbatchpointer_h, yvbatchpointer, Ntglobal);

        // GetDeviceMemory()
        
    }

    void TearDown() {

    }
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

}



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