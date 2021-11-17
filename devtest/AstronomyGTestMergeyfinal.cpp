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


void Getybatchpointer_inner(vector<int> splitstream, float **y, float ***ybp, 
int vectorlength, int streamsize){

    float *tmpy;
    GetDeviceMemory(&tmpy, vectorlength);

    float **tmpybp;
    CUDACHECK(cudaMalloc(tmpybp, sizeof(float*)*streamsize));

    float **tmpyh;
    tmpyh = new float*[streamsize];
    tmpyh[0] = tmpy;
    for(int i=1; i<streamsize; i++) {
        tmpyh[i] = tmpyh[0] + splitstream[i-1];
    }

    CopyDataB2HD(tmpybp, tmpyh, streamsize);
    y[0] = tmpy;
    ybp[0] = tmpybp;
    delete[] tmpyh;
}


void Getybatchpointer(int **ysizeinstream, float **y, float ****ybp, int vectorlength,
int streamsize){
    
    // size of stream
    vector<int> splitstream;
    for(int i=0; i<streamsize; i++){
        splitstream.push_back(vectorlength/streamsize);
    }
    for(int i=0; i<vectorlength%streamsize; i++){
        splitstream[i]++;
    }
    int *tmpysizeinstream;
    GetDeviceMemory(&tmpysizeinstream, streamsize);
    CopyDataB2HD(tmpysizeinstream, splitstream.data(), splitstream.size());
    ysizeinstream[0] = tmpysizeinstream;
    // y is given, size -> streamsize, get square pointers
    vector< vector<float*> > ptvec;
    vector<float*> beginptr;
    for(int i=0; i<streamsize; i++){
        beginptr.push_back(y[i]);
    }
    ptvec.push_back(beginptr);
    for(int i=1; i<streamsize; i++){
        vector<float*> tmpptrset;
        for(int j=0; j<streamsize; j++){
            tmpptrset.push_back(ptvec[i-1][j] + splitstream[i-1]);
        }
        ptvec.push_back(tmpptrset);
    }
    // paste as output
    float ***tmpybp;
    CUDACHECK(cudaMalloc(&tmpybp, sizeof(float**)*streamsize));
    for(int i=0; i<streamsize; i++){
        CUDACHECK(cudaMalloc(&tmpybp[i], sizeof(float*)* streamsize));
        CUDACHECK(cudaMemcpy(tmpybp[i], ptvec[i].data(), sizeof(float*) * streamsize, cudaMemcpyDefault));
    }
    ybp[0] = tmpybp;
}


class MergeyfinalGPUTest : public testing::Test {
protected:
    void SetUp() {
        vectorlength = atoi(g_command_line_arg_vec[0].c_str());
        streamsize = atoi(g_command_line_arg_vec[1].c_str());
        hy = new float*[streamsize];
        y = new float*[streamsize];
        for(int i=0; i<streamsize; i++){
            GetHostMemory(&hy[i], vectorlength);
            GetDeviceMemory(&y[i], vectorlength);
            for(int j=0; j<vectorlength; j++) hy[i][j] = (float) rand() / RAND_MAX;
            CopyDataB2HD(y[i], hy[i], vectorlength);
        }
        // naive cpu merge
        for(int i=1; i<streamsize; i++){
            for(int j=0; j<vectorlength; j++){
                hy[0][j] += hy[i][j];
            }
        }
    }

    void TearDown() {
        for(int i=0; i<streamsize; i++){
            delete[] hy[i];
            FreeDeviceMemory(y[i]);
        }
        delete[] hy;
        delete[] y;
    }

    int vectorlength;
    int streamsize;
    float **hy;
    float **y;
    float *** ybp;

};


TEST_F(MergeyfinalGPUTest, Basic){
    float ** dy;
    CUDACHECK(cudaMalloc(&dy, sizeof(float*)*streamsize));
    CopyDataB2HD(dy, y, streamsize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for(int i=0; i<100; i++){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        MergeyfinalBasicDriver(dy, vectorlength, streamsize);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cout << milliseconds << endl;
    }
    
    // float *ycheck;
    // GetHostMemory(&ycheck, vectorlength);
    // CopyDataB2HD(ycheck, y[0], vectorlength);
    // for(int i=0; i<vectorlength; i++){
    //     assert(fabs(ycheck[i] - hy[0][i]) < 1e-4);
    // }
}

TEST_F(MergeyfinalGPUTest, Advanced){

}










/**************************************
*   Basic Setting
***************************************/

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