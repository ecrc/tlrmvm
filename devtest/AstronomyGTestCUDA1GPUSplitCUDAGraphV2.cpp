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
// #include <cub/cub.cuh>
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
        float val = 1.0;
        for(int i=0; i<Ntglobal; i++){
            AvMs.push_back( (size_t)colrsum[i] );
            AvKs.push_back(nb);
            AvNs.push_back(1);
        }
        // get host memory
        GetHostMemoryBatched(&hAv, &hx, &hyv, &hAvbp, &hxbp, &hyvbp, AvMs, AvKs, AvNs, val);
        CaluclateTotalElements(AvMs, AvKs, AvNs, Avtotalelems, xtotalelems, yvtotalelems);
        memcpy(hAv, DataAv, sizeof(float) * Avtotalelems);
        
        // get device memory
        GetDeviceMemoryBatched(&Av, &x, &yv, 
        &Avbatchpointer, &xbatchpointer, &yvbatchpointer, AvMs, AvKs, AvNs);
        Avbatchpointer_h = new float*[Ntglobal];
        xbatchpointer_h = new float*[Ntglobal];
        yvbatchpointer_h = new float*[Ntglobal];
        cudatlrmat::CopyDataB2HD(Avbatchpointer_h, Avbatchpointer, Ntglobal);
        cudatlrmat::CopyDataB2HD(xbatchpointer_h, xbatchpointer, Ntglobal);
        cudatlrmat::CopyDataB2HD(yvbatchpointer_h, yvbatchpointer, Ntglobal);

        /**
         * Phase 3 preparation
         */
        vector<int> rowrsum = Rmat.RowSum();
        for(int i=0; i<Mtglobal; i++){
            AuMs.push_back(nb);
            AuKs.push_back( (size_t)rowrsum[i] );
            AuNs.push_back(1);
        }
        GetHostMemoryBatched(&hAu, &hyu, &hy, 
        &hAubp, &hyubp, &hybp, AuMs, AuKs, AuNs, val);
        CaluclateTotalElements(AuMs, AuKs, AuNs, Autotalelems, yutotalelems, ytotalelems);
        memcpy(hAu, DataAu, sizeof(float) * Autotalelems);
        
        // get random X
        #pragma omp parallel for 
        for(int i=0; i<paddingN; i++){
            // hx[i] = (float)rand() / RAND_MAX;
            hx[i] = 1.0;
        }

        GetDeviceMemoryBatched(&Au, &yu, &y, 
        &Aubatchpointer, &yubatchpointer, &ybatchpointer, AuMs, AuKs, AuNs);
        Aubatchpointer_h = new float*[Mtglobal];
        yubatchpointer_h = new float*[Mtglobal];
        ybatchpointer_h = new float*[Mtglobal];
        cudatlrmat::CopyDataB2HD(Aubatchpointer_h, Aubatchpointer, Mtglobal);
        cudatlrmat::CopyDataB2HD(yubatchpointer_h, yubatchpointer, Mtglobal);
        cudatlrmat::CopyDataB2HD(ybatchpointer_h, ybatchpointer, Mtglobal);

        cudatlrmat::CopyDataB2HD(Av, x, yv, hAv, hx, hyv, AvMs, AvKs, AvNs);
        cudatlrmat::CopyDataB2HD(Au, yu, y, hAu, hyu, hy, AuMs, AuKs, AuNs);

        // get phase2 offset
        GetDeviceMemory(&offsetphase2, (size_t)granksum);
        tlrmvm::CalculatePhase2Offset(&offsetphase2_h, Mtglobal, Ntglobal, Rmat);
        cudatlrmat::CopyDataB2HD(offsetphase2, offsetphase2_h, granksum);

        alpha = 1.0;
        beta = 0.0;

        bytesprocessed = 0.0;
        unsigned long int phase1 = granksum*nb + paddingN + granksum;
        unsigned long int shuffle = 2 * granksum;
        unsigned long int phase2 = granksum*nb + granksum + paddingM;
        bytesprocessed = sizeof(float) * (phase1 + shuffle + phase2);
        timestat.clear();
        LOOPSIZE = 5000;
        SKIPROUND = 1000;


        // graph preparation
        hAv_graph = new float*[STREAMSIZE];
        hx_graph = new float*[STREAMSIZE];
        hyv_graph = new float*[STREAMSIZE];
        hAvbp_graph = new float**[STREAMSIZE];
        hxbp_graph = new float**[STREAMSIZE];
        hyvbp_graph = new float**[STREAMSIZE];
        hAu_graph = new float*[STREAMSIZE];
        hyu_graph = new float*[STREAMSIZE];
        hy_graph = new float*[STREAMSIZE];
        hAubp_graph = new float**[STREAMSIZE];
        hyubp_graph = new float**[STREAMSIZE];
        hybp_graph = new float**[STREAMSIZE];

        Av_graph = new float*[STREAMSIZE];
        x_graph = new float*[STREAMSIZE];
        yv_graph = new float*[STREAMSIZE];
        Avbatchpointer_graph = new float**[STREAMSIZE];
        xbatchpointer_graph = new float**[STREAMSIZE];
        yvbatchpointer_graph = new float**[STREAMSIZE];
        Avbatchpointer_h_graph = new float**[STREAMSIZE];
        xbatchpointer_h_graph = new float**[STREAMSIZE];
        yvbatchpointer_h_graph = new float**[STREAMSIZE];
        Au_graph = new float*[STREAMSIZE];
        yu_graph = new float*[STREAMSIZE];
        y_graph = new float*[STREAMSIZE];
        CUDACHECK(cudaMalloc(&y_graph_ondevice, sizeof(float*) * STREAMSIZE));
        Aubatchpointer_graph = new float**[STREAMSIZE];
        yubatchpointer_graph = new float**[STREAMSIZE];
        ybatchpointer_graph = new float**[STREAMSIZE];
        Aubatchpointer_h_graph = new float**[STREAMSIZE];
        yubatchpointer_h_graph = new float**[STREAMSIZE];
        ybatchpointer_h_graph = new float**[STREAMSIZE];
        Avtotalelems_graph = new size_t[STREAMSIZE];
        xtotalelems_graph = new size_t[STREAMSIZE];
        yvtotalelems_graph = new size_t[STREAMSIZE];
        Autotalelems_graph = new size_t[STREAMSIZE];
        yutotalelems_graph = new size_t[STREAMSIZE];
        ytotalelems_graph = new size_t[STREAMSIZE];
        offsetphase2_graph = new unsigned long int*[STREAMSIZE];
        offsetphase2_h_graph = new unsigned long int*[STREAMSIZE];

        // split across Nt
        for(int i=0; i<STREAMSIZE; i++) 
        streamexecsize.push_back(Ntglobal / STREAMSIZE);
        for(int i=0; i<Ntglobal % STREAMSIZE; i++) streamexecsize[i]++;
        // get Rmatset
        vector<vector<int>> Rmatsetvec(STREAMSIZE, vector<int>());
        vector<Matrix<int>> Rmatset(STREAMSIZE, Matrix<int>());
        AvMs_graph.resize(STREAMSIZE, vector<size_t>());
        AvKs_graph.resize(STREAMSIZE, vector<size_t>());
        AvNs_graph.resize(STREAMSIZE, vector<size_t>());
        AuMs_graph.resize(STREAMSIZE, vector<size_t>());
        AuKs_graph.resize(STREAMSIZE, vector<size_t>());
        AuNs_graph.resize(STREAMSIZE, vector<size_t>());
        for(int i=0; i<Ntglobal; i++){
            int streamid = i % STREAMSIZE;
            int cntid = i / STREAMSIZE;
            Matrix<int> Rmat_subset = Rmat.Block({0,Mtglobal}, {(size_t)i,(size_t)i+1});
            for(int k=0; k<Mtglobal; k++) Rmatsetvec[streamid].push_back(Rmat_subset.RawPtr()[k]);
            AvMs_graph[streamid].push_back(Rmat_subset.ColSum()[0]);
            AvKs_graph[streamid].push_back(nb);
            AvNs_graph[streamid].push_back(1);
        }
        for(int i=0; i<STREAMSIZE; i++){
            Rmatset[i] = Matrix<int>(Rmatsetvec[i], Mtglobal, Rmatsetvec[i].size() / Mtglobal);
        }
        // check Rmatset
        int checkranksum = 0;
        for(int i=0; i<STREAMSIZE; i++) checkranksum += Rmatset[i].Sum();
        int granksum = Rmat.Sum();
        assert(checkranksum == granksum);
        for(int i=0; i<STREAMSIZE; i++)
        granksum_graph.push_back(Rmatset[i].Sum());


        for(int i=0; i<STREAMSIZE; i++){
            auto colsum = Rmatset[i].ColSum();
            for(int j=0; j<colsum.size(); j++){
                AvMs_graph[i].push_back(colsum[j]);
                AvKs_graph[i].push_back(nb);
                AvNs_graph[i].push_back(1); 
            }
            auto rowsum = Rmatset[i].RowSum();
            for(int j=0; j<rowsum.size(); j++){
                AuMs_graph[i].push_back(nb);
                AuKs_graph[i].push_back(rowsum[j]);
                AuNs_graph[i].push_back(1);
            }
        }

        for(int i=0; i<STREAMSIZE; i++){
            GetHostMemoryBatched(&hAv_graph[i], &hx_graph[i], &hyv_graph[i], 
            &hAvbp_graph[i], &hxbp_graph[i], &hyvbp_graph[i], AvMs_graph[i], 
            AvKs_graph[i], AvNs_graph[i], val);
            GetHostMemoryBatched(&hAu_graph[i], &hyu_graph[i], &hy_graph[i], 
            &hAubp_graph[i], &hyubp_graph[i], &hybp_graph[i], 
            AuMs_graph[i], AuKs_graph[i], AuNs_graph[i], val);
        }

        // copy Phase 1 pointer
        size_t colprefix = 0;
        vector<float*> mtAvpointer(STREAMSIZE, 0);
        vector<float*> mtxpointer(STREAMSIZE, 0);
        for(int i=0; i<STREAMSIZE; i++){
            mtAvpointer[i] = hAv_graph[i];
            mtxpointer[i] = hx_graph[i];
        }
    
        size_t prefix = 0;
        auto globalcolsum = Rmat.ColSum();
        for(int i=0; i<Ntglobal; i++){
            int streamid = i % STREAMSIZE;
            int cntid = i / STREAMSIZE;
            memcpy(mtAvpointer[streamid], DataAv + prefix * nb, globalcolsum[i] * nb * sizeof(float));
            memcpy(mtxpointer[streamid], hx + i * nb, nb * sizeof(float));
            mtAvpointer[streamid] += globalcolsum[i]*nb;
            mtxpointer[streamid] += nb;
            prefix += globalcolsum[i];
        }
        vector<float*> ntAupointer(STREAMSIZE, 0);
        for(int i=0; i<STREAMSIZE; i++){
            ntAupointer[i] = hAu_graph[i];
        }
        float *Aurunning = DataAu;
        prefix = 0;
        for(int i=0; i<Mtglobal; i++){
            for(int j=0; j<Ntglobal; j++){
                int streamid = j % STREAMSIZE;
                int currank = Rmat.GetElem(i,j);
                memcpy(ntAupointer[streamid], Aurunning, currank * nb * sizeof(float));
                ntAupointer[streamid] += currank*nb;
                Aurunning += currank*nb;
            }
        }

        for(int i=0; i<STREAMSIZE; i++){
            // phase 1 device
            int Ntlocal = Rmatset[i].Col();
            GetDeviceMemoryBatched(&Av_graph[i], &x_graph[i], &yv_graph[i], 
            &Avbatchpointer_graph[i], &xbatchpointer_graph[i], &yvbatchpointer_graph[i], 
            AvMs_graph[i], AvKs_graph[i], AvNs_graph[i]);
            
            cudatlrmat::CopyDataB2HD(Av_graph[i], x_graph[i], yv_graph[i], 
            hAv_graph[i], hx_graph[i], hyv_graph[i], AvMs_graph[i], AvKs_graph[i], AvNs_graph[i]);

            Avbatchpointer_h_graph[i] = new float*[Ntlocal];
            xbatchpointer_h_graph[i] = new float*[Ntlocal];
            yvbatchpointer_h_graph[i] = new float*[Ntlocal];
            cudatlrmat::CopyDataB2HD(Avbatchpointer_h_graph[i], Avbatchpointer_graph[i], Ntlocal);
            cudatlrmat::CopyDataB2HD(xbatchpointer_h_graph[i], xbatchpointer_graph[i], Ntlocal);
            cudatlrmat::CopyDataB2HD(yvbatchpointer_h_graph[i], yvbatchpointer_graph[i], Ntlocal);

            // phase 2 device
            size_t localranksum = Rmatset[i].Sum();
            GetDeviceMemory(&offsetphase2_graph[i], (size_t)localranksum);
            tlrmvm::CalculatePhase2Offset(&offsetphase2_h_graph[i], Mtglobal, Ntlocal, Rmatset[i]);
            cudatlrmat::CopyDataB2HD(offsetphase2_graph[i], offsetphase2_h_graph[i], localranksum);

            // phase 3 device
            GetDeviceMemoryBatched(&Au_graph[i], &yu_graph[i], &y_graph[i], 
            &Aubatchpointer_graph[i], &yubatchpointer_graph[i], &ybatchpointer_graph[i], 
            AuMs_graph[i], AuKs_graph[i], AuNs_graph[i]);
            cudatlrmat::CopyDataB2HD(Au_graph[i], yu_graph[i], y_graph[i], 
            hAu_graph[i], hyu_graph[i], hy_graph[i], AuMs_graph[i], AuKs_graph[i], AuNs_graph[i]);
            Aubatchpointer_h_graph[i] = new float*[Mtglobal];
            yubatchpointer_h_graph[i] = new float*[Mtglobal];
            ybatchpointer_h_graph[i] = new float*[Mtglobal];
            cudatlrmat::CopyDataB2HD(Aubatchpointer_h_graph[i], Aubatchpointer_graph[i], Mtglobal);
            cudatlrmat::CopyDataB2HD(yubatchpointer_h_graph[i], yubatchpointer_graph[i], Mtglobal);
            cudatlrmat::CopyDataB2HD(ybatchpointer_h_graph[i], ybatchpointer_graph[i], Mtglobal);

        }
        CopyDataB2HD(y_graph_ondevice, y_graph, STREAMSIZE);
        delete[] DataAv; 
        delete[] DataAu;
        delete[] DataR;


    }

    void TearDown() {

        // graph destroy
        // delete[] Av_graph;
        // delete[] x_graph;


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


    float *hAv;
    float *hx;
    float *hyv;
    float **hAvbp;
    float **hxbp;
    float **hyvbp;

    float *hAu;
    float *hyu;
    float *hy; 
    float **hAubp;
    float **hyubp;
    float **hybp;

    // new pointer for cudagraph
    vector<size_t> streamexecsize;
    vector<size_t> streamexecoffset;

    float **hAv_graph;
    float **hx_graph;
    float **hyv_graph;
    float ***hAvbp_graph;
    float ***hxbp_graph;
    float ***hyvbp_graph;
    float **hAu_graph;
    float **hyu_graph;
    float **hy_graph; 
    float ***hAubp_graph;
    float ***hyubp_graph;
    float ***hybp_graph;

    float **Av_graph;
    float **x_graph;
    float **yv_graph;
    float ***Avbatchpointer_graph;
    float ***xbatchpointer_graph;
    float ***yvbatchpointer_graph;
    float ***Avbatchpointer_h_graph;
    float ***xbatchpointer_h_graph;
    float ***yvbatchpointer_h_graph;
    float **Au_graph;
    float **yu_graph;
    float **y_graph;
    float **y_graph_ondevice;
    float ***Aubatchpointer_graph;
    float ***yubatchpointer_graph;
    float ***ybatchpointer_graph;
    float ***Aubatchpointer_h_graph;
    float ***yubatchpointer_h_graph;
    float ***ybatchpointer_h_graph;
    float *yfinal_graph;
    vector<vector<size_t>> AvMs_graph;
    vector<vector<size_t>> AvKs_graph;
    vector<vector<size_t>> AvNs_graph;
    vector<vector<size_t>> AuMs_graph;
    vector<vector<size_t>> AuKs_graph;
    vector<vector<size_t>> AuNs_graph;
    unsigned long int **offsetphase2_graph;
    unsigned long int **offsetphase2_h_graph;
    size_t *Avtotalelems_graph;
    size_t *xtotalelems_graph;
    size_t *yvtotalelems_graph;
    size_t *Autotalelems_graph;
    size_t *yutotalelems_graph;
    size_t *ytotalelems_graph;
    vector<size_t> granksum_graph;


    float *Av;
    float *x;
    float *yv;
    float **Avbatchpointer;
    float **xbatchpointer;
    float **yvbatchpointer;
    float **Avbatchpointer_h;
    float **xbatchpointer_h;
    float **yvbatchpointer_h;

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
    size_t Ntlocal;
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



TEST_F(AstronomySingleGPUTest, NoCUDAGraphPhase1){

    cout << "Stream size " << STREAMSIZE << endl;
    int LOOPSIZE = 1;
    cudaStream_t * streamptr = new cudaStream_t[STREAMSIZE];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[STREAMSIZE];
    for(int i=0; i<STREAMSIZE; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<STREAMSIZE; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<STREAMSIZE; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);
    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // cuda graph
    bool graphCreated=false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaEvent_t *events;
    cudaEvent_t event_start;
    cudaEvent_t event_phase2finish;
    CUDACHECK(cudaEventCreate(&event_start));
    CUDACHECK(cudaEventCreate(&event_phase2finish));
    events = new cudaEvent_t[2*STREAMSIZE];
    for(int i=0; i<2*STREAMSIZE; i++) CUDACHECK(cudaEventCreate(&events[i]));
    
#ifdef USE_NVTX
    PUSH_RANGE("CUDAGraph", 1)
#endif 
    for(int loopi=0; loopi < LOOPSIZE; loopi++){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
#ifdef USE_NVTX
        PUSH_RANGE("ITERATION", loopi)
#endif 
        for(int si=0; si < STREAMSIZE; si++){
            for(int i=0; i<streamexecsize[si]; i++){
                CUBLASCHECK(cublasgemv(cublashandleptr[si], CUBLAS_OP_N,
                AvMs_graph[si][i], AvKs_graph[si][i],
                &alpha, (const float*)Avbatchpointer_h_graph[si][i], 
                AvMs_graph[si][i], 
                (const float*)xbatchpointer_h_graph[si][i], 1, &beta, 
                yvbatchpointer_h_graph[si][i], 1));
            }
            // break;
            // phase2dirver(yu_graph[si], yv_graph[si], granksum_graph[si], offsetphase2_graph[si], streamptr[si]);
            // for(int i=0; i<Mtglobal; i++){
            //     CUBLASCHECK(cublasgemv(cublashandleptr[si], CUBLAS_OP_N,
            //     AuMs_graph[si][i], AuKs_graph[si][i],
            //     &alpha, (const float*)Aubatchpointer_h_graph[si][i], AuMs_graph[si][i], 
            //     (const float*)yubatchpointer_h_graph[si][i], 1, &beta, 
            //     ybatchpointer_h_graph[si][i], 1));
            // }
        }
        for(int si=0; si < STREAMSIZE; si++) cudaStreamSynchronize(streamptr[si]);
        // MergeyfinalBasicDriver(y_graph_ondevice, paddingM, STREAMSIZE);   
#ifdef USE_NVTX
        POP_RANGE
#endif 
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cout << "time per loop " << milliseconds << endl;
    }
#ifdef USE_NVTX
    POP_RANGE
#endif 
//     delete[] events;

    // AstronomyPCMatrix astropcmat(datafolder, acc, nb, id, originM, originN);
    // Matrix<float> xvec(hx, paddingN, 1);
    // astropcmat.setX(xvec);
    // Matrix<float> yv_pc = astropcmat.Phase1();
    // size_t prev = 0, post = 0;
    // size_t colacc = 0;
    // size_t yvacc = 0;
    // for(int si=0; si<STREAMSIZE; si++){
    //     post += streamexecsize[si];
    //     vector<int> curcolsum = Rmat.Block({0,Mtglobal}, {prev, post}).ColSum();
    //     for(int i = 0; i<streamexecsize[si]; i++){
    //         vector<float> tmpres(curcolsum[i], 0);
    //         CopyDataB2HD(tmpres.data(), yvbatchpointer_h_graph[si][i], curcolsum[i]);
    //         double err = NetlibError(tmpres.data(), yv_pc.RawPtr() + yvacc, curcolsum[i]);
    //         yvacc += curcolsum[i];
    //         // std::cerr << "\033[32m[ APP INFO ] MultiStream error = " << err << "\033[0m"<< endl;
    //     }
    //     vector<float> hyresult(streamexecsize[si],0);
    //     prev += streamexecsize[si];
    // }
}




// TEST_F(AstronomySingleGPUTest, NoCUDAGraphPhase2){

//     cout << "Stream size " << STREAMSIZE << endl;
//     int LOOPSIZE = 1;
//     cudaStream_t * streamptr = new cudaStream_t[STREAMSIZE];
//     cublasHandle_t * cublashandleptr = new cublasHandle_t[STREAMSIZE];
//     for(int i=0; i<STREAMSIZE; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
//     for(int i=0; i<STREAMSIZE; i++) cublasCreate_v2(&cublashandleptr[i]);
//     for(int i=0; i<STREAMSIZE; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);
//     // timing
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     // cuda graph
//     bool graphCreated=false;
//     cudaGraph_t graph;
//     cudaGraphExec_t instance;
//     cudaEvent_t *events;
//     cudaEvent_t event_start;
//     cudaEvent_t event_phase2finish;
//     CUDACHECK(cudaEventCreate(&event_start));
//     CUDACHECK(cudaEventCreate(&event_phase2finish));
//     events = new cudaEvent_t[2*STREAMSIZE];
//     for(int i=0; i<2*STREAMSIZE; i++) CUDACHECK(cudaEventCreate(&events[i]));
    
// #ifdef USE_NVTX
//     PUSH_RANGE("CUDAGraph", 1)
// #endif 
//     for(int loopi=0; loopi < LOOPSIZE; loopi++){
//         cudaDeviceSynchronize();
//         cudaEventRecord(start);
// #ifdef USE_NVTX
//         PUSH_RANGE("ITERATION", loopi)
// #endif 
//         for(int si=0; si < STREAMSIZE; si++){
//             for(int i=0; i<streamexecsize[si]; i++){
//                 CUBLASCHECK(cublasgemv(cublashandleptr[si], CUBLAS_OP_N,
//                 AvMs_graph[si][i], AvKs_graph[si][i],
//                 &alpha, (const float*)Avbatchpointer_h_graph[si][i], 
//                 AvMs_graph[si][i], 
//                 (const float*)xbatchpointer_h_graph[si][i], 1, &beta, 
//                 yvbatchpointer_h_graph[si][i], 1));
//             }
//             // break;
//             phase2dirver(yu_graph[si], yv_graph[si], granksum_graph[si], offsetphase2_graph[si], streamptr[si]);
//             CUDACHECK(cudaGetLastError());
//             // for(int i=0; i<Mtglobal; i++){
//             //     CUBLASCHECK(cublasgemv(cublashandleptr[si], CUBLAS_OP_N,
//             //     AuMs_graph[si][i], AuKs_graph[si][i],
//             //     &alpha, (const float*)Aubatchpointer_h_graph[si][i], AuMs_graph[si][i], 
//             //     (const float*)yubatchpointer_h_graph[si][i], 1, &beta, 
//             //     ybatchpointer_h_graph[si][i], 1));
//             // }
//             CUDACHECK(cudaGetLastError());
//         }
//         CUDACHECK(cudaGetLastError());
//         for(int si=0; si < STREAMSIZE; si++) cudaStreamSynchronize(streamptr[si]);
//         MergeyfinalBasicDriver(y_graph_ondevice, paddingM, STREAMSIZE, streamptr[0]);   
// #ifdef USE_NVTX
//         POP_RANGE
// #endif 
//         cudaDeviceSynchronize();
//         cudaEventRecord(stop);
        
//         cudaEventSynchronize(stop);
//         float milliseconds = 0;
//         cudaEventElapsedTime(&milliseconds, start, stop);
//         cout << "time per loop " << milliseconds << endl;
//     }
// #ifdef USE_NVTX
//     POP_RANGE
// #endif 
// //     delete[] events;

//     AstronomyPCMatrix astropcmat(datafolder, acc, nb, id, originM, originN);
//     Matrix<float> xvec(hx, paddingN, 1);
//     astropcmat.setX(xvec);
//     Matrix<float> yv_pc = astropcmat.Phase1();
//     Matrix<float> yu_pc = astropcmat.Phase2();
//     Matrix<float> y_pc = astropcmat.Phase3();
//     size_t prev = 0, post = 0;
//     size_t colacc = 0;
//     size_t yvacc = 0;
//     cout << streamexecsize[0] << endl;
//     for(int si=0; si < STREAMSIZE; si++){
//         size_t prefix = 0;
//         for(int i=0; i<Mtglobal; i++){

//             for(int j=0; j<streamexecsize[si]; j++){
//                 auto midy = astropcmat.GetMiddley(i, prev + j);
//                 int len = midy.Row();
//                 vector<float> midyvec(midy.Row(), 0);        
//                 CopyDataB2HD(midyvec.data(), yu_graph[si]+prefix, midy.Row());
//                 prefix += midy.Row();
//                 double err = NetlibError(midyvec.data(), midy.RawPtr(), midy.Row());
//                 std::cerr << "\033[32m[ APP INFO ] MultiStream error = " << err << "\033[0m"<< endl;
//             }
//         }

//         prev += streamexecsize[si];
//     }
// }


TEST_F(AstronomySingleGPUTest, NoCUDAGraphPhase3){

    cout << "Stream size " << STREAMSIZE << endl;
    int LOOPSIZE = 1;
    cudaStream_t * streamptr = new cudaStream_t[STREAMSIZE];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[STREAMSIZE];
    for(int i=0; i<STREAMSIZE; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<STREAMSIZE; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<STREAMSIZE; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);
    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // cuda graph
    bool graphCreated=false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaEvent_t *events;
    cudaEvent_t event_start;
    cudaEvent_t event_phase2finish;
    CUDACHECK(cudaEventCreate(&event_start));
    CUDACHECK(cudaEventCreate(&event_phase2finish));
    events = new cudaEvent_t[2*STREAMSIZE];
    for(int i=0; i<2*STREAMSIZE; i++) CUDACHECK(cudaEventCreate(&events[i]));
    
#ifdef USE_NVTX
    PUSH_RANGE("CUDAGraph", 1)
#endif 
    for(int loopi=0; loopi < LOOPSIZE; loopi++){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
#ifdef USE_NVTX
        PUSH_RANGE("ITERATION", loopi)
#endif 
        for(int si=0; si < STREAMSIZE; si++){
            for(int i=0; i<streamexecsize[si]; i++){
                CUBLASCHECK(cublasgemv(cublashandleptr[si], CUBLAS_OP_N,
                AvMs_graph[si][i], AvKs_graph[si][i],
                &alpha, (const float*)Avbatchpointer_h_graph[si][i], 
                AvMs_graph[si][i], 
                (const float*)xbatchpointer_h_graph[si][i], 1, &beta, 
                yvbatchpointer_h_graph[si][i], 1));
            }
            // break;
            phase2dirver(yu_graph[si], yv_graph[si], granksum_graph[si], offsetphase2_graph[si], streamptr[si]);
            for(int i=0; i<Mtglobal; i++){
                CUBLASCHECK(cublasgemv(cublashandleptr[si], CUBLAS_OP_N,
                AuMs_graph[si][i], AuKs_graph[si][i],
                &alpha, (const float*)Aubatchpointer_h_graph[si][i], AuMs_graph[si][i], 
                (const float*)yubatchpointer_h_graph[si][i], 1, &beta, 
                ybatchpointer_h_graph[si][i], 1));
            }
            CUDACHECK(cudaGetLastError());
        }
        CUDACHECK(cudaGetLastError());
        cudaDeviceSynchronize();
        // MergeyfinalBasicDriver(y_graph_ondevice, paddingM, STREAMSIZE);   
#ifdef USE_NVTX
        POP_RANGE
#endif 
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cout << "time per loop " << milliseconds << endl;
    }
#ifdef USE_NVTX
    POP_RANGE
#endif 
//     delete[] events;

    AstronomyPCMatrix astropcmat(datafolder, acc, nb, id, originM, originN);
    Matrix<float> xvec(hx, paddingN, 1);
    astropcmat.setX(xvec);
    Matrix<float> yv_pc = astropcmat.Phase1();
    Matrix<float> yu_pc = astropcmat.Phase2();
    Matrix<float> y_pc = astropcmat.Phase3();
    // size_t prev = 0, post = 0;
    // size_t colacc = 0;
    // size_t yvacc = 0;


    vector<float> hyresult(paddingM, 0);
    cudatlrmat::CopyDataB2HD(hyresult.data(), y_graph[0], paddingM);
    double err = NetlibError(hyresult.data(), y_pc.RawPtr(),originM);
    std::cerr << "\033[32m[ APP INFO ] MultiStream error = " << err << "\033[0m"<< endl;


    // for(int si=0; si < STREAMSIZE; si++){
    //     size_t prefix = 0;
    //     for(int i=0; i<Mtglobal; i++){

    //         for(int j=0; j<streamexecsize[si]; j++){
    //             auto utile = astropcmat.GetUTile(i, prev + j);
    //             vector<float> utilevec(utile.Row()*utile.Col(), 0);        
    //             CopyDataB2HD(utilevec.data(), Au_graph[si]+prefix, utile.Row()*utile.Col());
    //             prefix += utile.Row()*utile.Col();
    //             double err = NetlibError(utilevec.data(), utile.RawPtr(), utile.Row()*utile.Col());
    //             std::cerr << "\033[32m[ APP INFO ] MultiStream error = " << err << "\033[0m"<<  "utilerow" << utile.Row() << endl;
    //         }
    //     }

    //     prev += streamexecsize[si];
    // }
    // prev = 0;
    // for(int si=0; si < STREAMSIZE; si++){
    //     size_t prefix = 0;
    //     for(int i=0; i<Mtglobal; i++){

    //         for(int j=0; j<streamexecsize[si]; j++){
    //             auto utile = astropcmat.GetUTile(i, prev + j);
    //             vector<float> utilevec(utile.Row()*utile.Col(), 0);        
    //             CopyDataB2HD(utilevec.data(), Au_graph[si]+prefix, utile.Row()*utile.Col());
    //             prefix += utile.Row()*utile.Col();
    //             double err = NetlibError(utilevec.data(), utile.RawPtr(), utile.Row()*utile.Col());
    //             std::cerr << "\033[32m[ APP INFO ] MultiStream error = " << err << "\033[0m"<<  "utilerow" << utile.Row() << endl;
    //         }
    //     }

    //     prev += streamexecsize[si];
    // }
    
    
    // prev = 0;
    // for(int si=0; si < STREAMSIZE; si++){
    //     for(int mti=0; mti < Mtglobal; mti++){
    //         Matrix<float> yfirstblock(nb, 1);
    //         yfirstblock.Fill(0);
    //         for(int j=0; j<streamexecsize[si]; j++){
    //             auto curau = astropcmat.GetUTile(mti, j+prev);
    //             auto curyu = astropcmat.GetMiddley(mti, j+prev);
    //             yfirstblock += curau * curyu;
    //         }
    //         vector<float> devyfirstblock(nb, 0);
    //         CopyDataB2HD(devyfirstblock.data(), y_graph[si]+mti*nb, nb);
    //         // cout << "yfirstblock " << yfirstblock.RawPtr()[0] << endl;
    //         double err = NetlibError(devyfirstblock.data(), yfirstblock.RawPtr(),nb);
    //         std::cerr << "\033[32m[ APP INFO ] MultiStream error = " << err << "\033[0m"<< endl;
    //     }
    //     prev += streamexecsize[si];
    // }
}



// TEST_F(AstronomySingleGPUTest, NoCUDAGraphPhase3MOVETOGRAPH){

//     cout << "Stream size " << STREAMSIZE << endl;
//     int LOOPSIZE = 20;
//     cudaStream_t * streamptr = new cudaStream_t[STREAMSIZE];
//     cublasHandle_t * cublashandleptr = new cublasHandle_t[STREAMSIZE];
//     for(int i=0; i<STREAMSIZE; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
//     for(int i=0; i<STREAMSIZE; i++) cublasCreate_v2(&cublashandleptr[i]);
//     for(int i=0; i<STREAMSIZE; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);
//     // timing
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     // cuda graph
//     bool graphCreated=false;
//     cudaGraph_t graph;
//     cudaGraphExec_t instance;
//     cudaEvent_t *events;
//     cudaEvent_t event_start;
//     cudaEvent_t event_finish;
//     CUDACHECK(cudaEventCreate(&event_start));
//     CUDACHECK(cudaEventCreate(&event_finish));
//     events = new cudaEvent_t[2*STREAMSIZE];
//     for(int i=0; i<2*STREAMSIZE; i++) CUDACHECK(cudaEventCreate(&events[i]));
    
//     int si = 0;
//     int i = 0;
//     CUBLASCHECK(cublasgemv(cublashandleptr[si], CUBLAS_OP_N,
//     AvMs_graph[si][i], AvKs_graph[si][i],
//     &alpha, (const float*)Avbatchpointer_h_graph[si][i], 
//     AvMs_graph[si][i], 
//     (const float*)xbatchpointer_h_graph[si][i], 1, &beta, 
//     yvbatchpointer_h_graph[si][i], 1));

//     CUDACHECK(cudaGetLastError());
//     cout << "outside graph execution pass "<< endl;
// #ifdef USE_NVTX
//     PUSH_RANGE("CUDAGraph", 1)
// #endif 
//     for(int loopi=0; loopi < LOOPSIZE; loopi++){
//         cudaDeviceSynchronize();
//         cudaEventRecord(start);
// #ifdef USE_NVTX
//         PUSH_RANGE("ITERATION", loopi)
// #endif 
//         if(!graphCreated){
//             cudaStreamBeginCapture(streamptr[0],cudaStreamCaptureModeGlobal);
//             cudaEventRecord(event_start, streamptr[0]);
//             for(int streami=1; streami<STREAMSIZE; streami++){
//                 cudaStreamWaitEvent(streamptr[streami], event_start);
//             }

//             for(int si=0; si < STREAMSIZE; si++){
//                 for(int i=0; i<streamexecsize[si]; i++){
//                     CUBLASCHECK(cublasgemv(cublashandleptr[si], CUBLAS_OP_N,
//                     AvMs_graph[si][i], AvKs_graph[si][i],
//                     &alpha, (const float*)Avbatchpointer_h_graph[si][i], 
//                     AvMs_graph[si][i], 
//                     (const float*)xbatchpointer_h_graph[si][i], 1, &beta, 
//                     yvbatchpointer_h_graph[si][i], 1));
//                 }
//             }
//             for(int streami=1; streami < STREAMSIZE; streami++){
//                 cudaEventRecord(events[streami], streamptr[streami]);
//             }
//             for(int streami=1; streami < STREAMSIZE; streami++){
//                 cudaStreamWaitEvent(streamptr[0], events[streami]);
//             }
//             phase2dirver(yu_graph[si], yv_graph[si], granksum_graph[si], offsetphase2_graph[si], streamptr[si]);
            
//             for(int si=0; si<STREAMSIZE; si++){
//                 for(int i=0; i<Mtglobal; i++){
//                     CUBLASCHECK(cublasgemv(cublashandleptr[si], CUBLAS_OP_N,
//                     AuMs_graph[si][i], AuKs_graph[si][i],
//                     &alpha, (const float*)Aubatchpointer_h_graph[si][i], AuMs_graph[si][i], 
//                     (const float*)yubatchpointer_h_graph[si][i], 1, &beta, 
//                     ybatchpointer_h_graph[si][i], 1));
//                 }
//             }
                
//             // for(int si=0; si < STREAMSIZE; si++){
//             //     for(int i=0; i<streamexecsize[si]; i++){
//             //         CUBLASCHECK(cublasgemv(cublashandleptr[si], CUBLAS_OP_N,
//             //         AvMs_graph[si][i], AvKs_graph[si][i],
//             //         &alpha, (const float*)Avbatchpointer_h_graph[si][i], 
//             //         AvMs_graph[si][i], 
//             //         (const float*)xbatchpointer_h_graph[si][i], 1, &beta, 
//             //         yvbatchpointer_h_graph[si][i], 1));
//             //     }
//             //     phase2dirver(yu_graph[si], yv_graph[si], granksum_graph[si], offsetphase2_graph[si], streamptr[si]);
//             //     for(int i=0; i<Mtglobal; i++){
//             //         CUBLASCHECK(cublasgemv(cublashandleptr[si], CUBLAS_OP_N,
//             //         AuMs_graph[si][i], AuKs_graph[si][i],
//             //         &alpha, (const float*)Aubatchpointer_h_graph[si][i], AuMs_graph[si][i], 
//             //         (const float*)yubatchpointer_h_graph[si][i], 1, &beta, 
//             //         ybatchpointer_h_graph[si][i], 1));
//             //     }
//             // }
            

//             MergeyfinalBasicDriver(y_graph_ondevice, paddingM, STREAMSIZE, streamptr[0]);
//             // cudaEventRecord(event_finish,streamptr[0]);   
//             // for(int streami=1; streami < STREAMSIZE; streami++){
//             //     cudaStreamWaitEvent(streamptr[streami], event_finish);
//             // }
//             cudaStreamEndCapture(streamptr[0], &graph);
//             cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
//             graphCreated = true;
            

//         }

//         cudaGraphLaunch(instance, streamptr[0]);
//         cudaStreamSynchronize(streamptr[0]);
// #ifdef USE_NVTX
//         POP_RANGE
// #endif 
//         cudaDeviceSynchronize();
//         cudaEventRecord(stop);
        
//         cudaEventSynchronize(stop);
//         float milliseconds = 0;
//         cudaEventElapsedTime(&milliseconds, start, stop);
//         cout << "time per loop " << milliseconds << endl;
//     }
// #ifdef USE_NVTX
//     POP_RANGE
// #endif 
// //     delete[] events;

//     AstronomyPCMatrix astropcmat(datafolder, acc, nb, id, originM, originN);
//     Matrix<float> xvec(hx, paddingN, 1);
//     astropcmat.setX(xvec);
//     Matrix<float> yv_pc = astropcmat.Phase1();
//     Matrix<float> yu_pc = astropcmat.Phase2();
//     Matrix<float> y_pc = astropcmat.Phase3();
//     size_t prev = 0, post = 0;
//     size_t colacc = 0;
//     size_t yvacc = 0;

//     vector<float> hyresult(paddingM, 0);
//     vector<float *> devptrback(STREAMSIZE,0);
//     CopyDataB2HD(devptrback.data(), y_graph_ondevice, STREAMSIZE);
//     cudatlrmat::CopyDataB2HD(hyresult.data(), devptrback[0], paddingM);
//     double err = NetlibError(hyresult.data(), y_pc.RawPtr(),originM);
//     std::cerr << "\033[32m[ APP INFO ] MultiStream error = " << err << "\033[0m"<< endl;


//     // cout << streamexecsize[0] << endl;
//     // vector<float> checkfinal(paddingM, 0);
//     // vector<float> copyfinal(paddingM, 0);
//     // vector<float *> devptrback(STREAMSIZE,0);
//     // CopyDataB2HD(devptrback.data(), y_graph_ondevice, STREAMSIZE);
//     // for(int si=0; si<STREAMSIZE; si++){
//     //     CopyDataB2HD(copyfinal.data(), devptrback[si], paddingM);
//     //     for(int k=0; k<paddingM; k++) checkfinal[k] += copyfinal[k];
//     // }
//     // double err = NetlibError(checkfinal.data(), y_pc.RawPtr(), paddingM);
//     // std::cerr << "\033[32m[ APP INFO ] MultiStream error = " << err << "\033[0m" << endl;


// }

TEST_F(AstronomySingleGPUTest, CUDAGraphAsync){

    cout << "Stream size " << STREAMSIZE << endl;
    int LOOPSIZE = 20;
    cudaStream_t * streamptr = new cudaStream_t[STREAMSIZE];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[STREAMSIZE];
    for(int i=0; i<STREAMSIZE; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<STREAMSIZE; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<STREAMSIZE; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);
    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // cuda graph
    bool graphCreated=false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaEvent_t *events;
    cudaEvent_t event_start;
    cudaEvent_t event_phase2finish;
    CUDACHECK(cudaEventCreate(&event_start));
    CUDACHECK(cudaEventCreate(&event_phase2finish));
    events = new cudaEvent_t[2*STREAMSIZE];
    for(int i=0; i<2*STREAMSIZE; i++) CUDACHECK(cudaEventCreate(&events[i]));
    
    for(int loopi=0; loopi < LOOPSIZE; loopi++){
        cudaDeviceSynchronize();
#ifdef USE_NVTX
        PUSH_RANGE("ITERATION", loopi)
#endif 
        cudaEventRecord(start);
        if(!graphCreated){
            cudaStreamBeginCapture(streamptr[0],cudaStreamCaptureModeGlobal);
            cudaEventRecord(event_start, streamptr[0]);
            for(int streami=1; streami<STREAMSIZE; streami++){
                cudaStreamWaitEvent(streamptr[streami], event_start);
            }
            for(int si=0; si < STREAMSIZE; si++){
                for(int i=0; i<streamexecsize[si]; i++){
                    CUBLASCHECK(cublasgemv(cublashandleptr[si], CUBLAS_OP_N,
                    AvMs_graph[si][i], AvKs_graph[si][i],
                    &alpha, (const float*)Avbatchpointer_h_graph[si][i], 
                    AvMs_graph[si][i], 
                    (const float*)xbatchpointer_h_graph[si][i], 1, &beta, 
                    yvbatchpointer_h_graph[si][i], 1));
                }
                phase2dirver(yu_graph[si], yv_graph[si], granksum_graph[si], 
                offsetphase2_graph[si], streamptr[si]);
                for(int i=0; i<Mtglobal; i++){
                    CUBLASCHECK(cublasgemv(cublashandleptr[si], CUBLAS_OP_N,
                    AuMs_graph[si][i], AuKs_graph[si][i],
                    &alpha, (const float*)Aubatchpointer_h_graph[si][i], AuMs_graph[si][i], 
                    (const float*)yubatchpointer_h_graph[si][i], 1, &beta, 
                    ybatchpointer_h_graph[si][i], 1));
                }
            }
            for(int streami=1; streami < STREAMSIZE; streami++){
                cudaEventRecord(events[streami], streamptr[streami]);
            }
            for(int streami=1; streami < STREAMSIZE; streami++){
                cudaStreamWaitEvent(streamptr[0], events[streami]);
            }
            MergeyfinalBasicDriver(y_graph_ondevice, paddingM, STREAMSIZE, streamptr[0]);
            cudaStreamEndCapture(streamptr[0], &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated=true;
        }
        cudaGraphLaunch(instance, streamptr[0]);
        cudaStreamSynchronize(streamptr[0]);
        cudaEventRecord(stop);
#ifdef USE_NVTX
        POP_RANGE
#endif 
        cudaDeviceSynchronize();
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        cout << "time per loop " << milliseconds << endl;
    }

    AstronomyPCMatrix astropcmat(datafolder, acc, nb, id, originM, originN);
    Matrix<float> xvec(hx, paddingN, 1);
    astropcmat.setX(xvec);
    Matrix<float> yv_pc = astropcmat.Phase1();
    Matrix<float> yu_pc = astropcmat.Phase2();
    Matrix<float> y_pc = astropcmat.Phase3();
    vector<float> hyresult(paddingM,0);
    cudatlrmat::CopyDataB2HD(hyresult.data(), y_graph[0], paddingM);
    double err = NetlibError(hyresult.data(), y_pc.RawPtr(),originM);
    std::cerr << "\033[32m[ APP INFO ] MultiStream error = " << err << "\033[0m"<< endl;
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