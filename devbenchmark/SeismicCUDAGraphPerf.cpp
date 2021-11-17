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

#include "benchmark/benchmark.h"
#include "benchmark/benchmark.h"

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


Matrix<complex<float>> mergerealimag(float * real, float *imag, size_t row, size_t col){
    Matrix<complex<float>> ret(row, col);
    for(int i=0; i<col; i++){
        for(int j=0; j<row; j++){
            ret.SetElem(j,i,complex<float>(real[i*row+j], imag[i*row+j]));
        }
    }
    return ret;
}


class SeismicFixture : public benchmark::Fixture {

public:
    void StreamInit(){
        // stream init
        for(int i=0; i<streamsize; i++) 
        streamexecsize.push_back(Ntglobal / streamsize);
        for(int i=0; i<Ntglobal % streamsize; i++) streamexecsize[i]++;
        streamexecoffset.clear();
        streamexecoffset.push_back(0);
        for(int i=1; i<streamsize;i++) 
        streamexecoffset.push_back(streamexecsize[i-1] + streamexecoffset[i-1]); 
    }



    void getmaskmat(int freqx){
        masktype = inputmap["masktype"];
        if(masktype == "Banded"){
            bandlength = atoi(inputmap["bandlength"].c_str());

        }else if(masktype == "userinput"){
            maskfilename = inputmap["maskfile"];
        }
        // once you got your mask mat you can generate FP32, FP16, Int8
        auto maskmat = Matrix<int>(Mtglobal, Ntglobal);
        // user define from file
        // maskmat.Fromfile(PathJoin({datafolder, "maskmatband20.bin"}), Mtglobal, Ntglobal);
        // full FP32
        maskmat.Fill(4); // full FP32
        
        // maskmat.Fill(2); // full FP16


        // After init maskmat, fill the selected rankmat info,
        Maskmats[freqx] = maskmat;
        FP32Rmats[freqx] = Matrix<int>(maskmat.Row(), maskmat.Col());
        FP32Rmats[freqx].Fill(0);
        for(int i=0; i<maskmat.Row(); i++){
            int streamidcnt = 0;
            for(int j=0; j<maskmat.Col(); j++){
                int currank = Rmats[freqx].GetElem(i,j);
                switch (maskmat.GetElem(i,j))
                {
                case 4:/* FP32 */
                    FP32Rmats[freqx].SetElem(i,j,currank);
                    break;
                case 2:/* FP16 */
                    FP16Rmats[freqx].SetElem(i,j,currank);
                    break;
                case 1:/* INT8 */
                    INT8Rmats[freqx].SetElem(i,j,currank);
                    break;
                default:
                    break;
                }
                streamidcnt++;
            }
        }
    }

    void SetUp(const ::benchmark::State& state) {
        
        datafolder = inputmap["datafolder"];
        acc = inputmap["acc"];
        string freqstr = inputmap["freqlist"];
        if(freqstr == "full"){
            for(int i=0; i<3; i++){
                freqlist.push_back(i);
            }
        }else{
            freqlist.push_back(atoi(freqstr.c_str()));
        }
        nb = atoi(inputmap["nb"].c_str());
        originM = atoi(inputmap["M"].c_str());
        originN = atoi(inputmap["N"].c_str());
        streamsize = atoi(inputmap["streamsize"].c_str());

        paddingM = CalculatePadding(originM, nb);
        paddingN = CalculatePadding(originN, nb);
        Mtglobal = paddingM / nb;
        Ntglobal = paddingN / nb;
            // split precision
        cout << freqlist.size() << endl;
        for(auto x : freqlist){
            int *DataR;
            ReadSeismicBinary(datafolder, &DataR, Mtglobal * Ntglobal, acc, nb, x);
            Matrix<int> Rmat = Matrix<int>(DataR, Mtglobal, Ntglobal);
            Rmats[x] = Rmat; // just generate Rmat in setup
            getmaskmat(x);
            delete[] DataR;
        }
    }

    void TearDown(const ::benchmark::State& state) {

    }

    double TLRMVMBytes(Matrix<int> Rmat, size_t dtypesize){
        auto granksum = Rmat.Sum();
        unsigned long int phase1 = granksum*nb + paddingN + granksum;
        unsigned long int shuffle = 2 * granksum;
        unsigned long int phase2 = granksum*nb + granksum + paddingM;
        return dtypesize * (phase1 + shuffle + phase2);
    }
    // problem related
    string datafolder;
    string acc;
    vector<int> freqlist;
    
    string densetest;
    
    // Size of problem
    int nb;
    int originM;
    int originN;
    int paddingM;
    int paddingN;
    int Mtglobal;
    int Ntglobal;

    // maskmat
    string masktype;
    string maskfilename;
    int bandlength;

    int streamsize;
    

    vector<size_t> Ntlocals;
    vector<size_t> streamexecsize;
    vector<size_t> streamexecoffset;
    vector<size_t> globalranksum;
    unordered_map< int, Matrix<int> > Rmats;  // key is freqid
    unordered_map< int, Matrix<int> > Maskmats;  // key is stream
    unordered_map< int, Matrix<int> > FP32Rmats; // key is freqid
    unordered_map< int, Matrix<int> > FP16Rmats; // key is freqid
    unordered_map< int, Matrix<int> > INT8Rmats; // key is freqid


};

double getcomplexnorm(complex<float> v1, complex<float> v2){
    return abs(v1-v2) / abs(v2);
}

BENCHMARK_DEFINE_F(SeismicFixture, SinglePtrBenchmark)(benchmark::State& state)
{
    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Float32Ptr fp32ptr(paddingM, paddingN, nb, Rmats[freqlist[0]], FP32Rmats[freqlist[0]]);
    fp32ptr.InitData(datafolder, acc, freqlist[0], originN);

    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    fp32ptr.CopyData2GPU();

    float alpha = 1.0;
    float beta = 0.0;
    vector<double> rawtime;

#ifdef USE_NVTX
    PUSH_RANGE("CUDAGraph", 1)
#endif 
size_t loopi = 0;
    for(auto st : state){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
#ifdef USE_NVTX
        PUSH_RANGE("ITERATION", loopi)
#endif 
        // phase 1
        for(int i=0; i<39; i++){
            if(fp32ptr.AvMs[i] == 0) continue;
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp32ptr.AvMs[i], fp32ptr.AvNs[i], fp32ptr.AvKs[i], 
                &alpha, fp32ptr.d_Avbp[i][0], fp32ptr.AvMs[i], 
                fp32ptr.d_xbp[i], fp32ptr.AvKs[i], 
                &beta, fp32ptr.d_yvbp[i][0], fp32ptr.AvMs[i]);
            );
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp32ptr.AvMs[i], fp32ptr.AvNs[i], fp32ptr.AvKs[i], 
                &alpha, fp32ptr.d_Avbp[i][1], fp32ptr.AvMs[i], 
                fp32ptr.d_xbp[i], fp32ptr.AvKs[i], 
                &beta, fp32ptr.d_yvbp[i][1], fp32ptr.AvMs[i]);
            );
        }
        // phase 2
        phase2(fp32ptr.d_yv[0], fp32ptr.d_yv[1], 
        fp32ptr.d_fp32colrank, fp32ptr.Ntg,fp32ptr.d_phase2mapping, 
        fp32ptr.d_yu, 
        fp32ptr.fp32granksum, streamptr[0]);
        cudaDeviceSynchronize();
        // phase 3
        for(int i=0; i<39; i++){
            if(fp32ptr.AuMs[i] != 0){
                CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                    fp32ptr.AuMs[i], fp32ptr.AuNs[i], fp32ptr.AuKs[i], 
                    &alpha, fp32ptr.d_Aubp[i][0], fp32ptr.AuMs[i], 
                    fp32ptr.d_yubp[i], fp32ptr.AuKs[i], 
                    &beta, fp32ptr.d_ybp[i][0], fp32ptr.AuMs[i]);
                );
                CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                    fp32ptr.AuMs[i], fp32ptr.AuNs[i], fp32ptr.AuKs[i], 
                    &alpha, fp32ptr.d_Aubp[i][1], fp32ptr.AuMs[i], 
                    fp32ptr.d_yubp[i], fp32ptr.AuKs[i], 
                    &beta, fp32ptr.d_ybp[i][1], fp32ptr.AuMs[i]);
                ); 
            }
        }
        // final merge
        phase3_merge(fp32ptr.d_y[0], fp32ptr.d_y[1], fp32ptr.nb, fp32ptr.d_finaly, fp32ptr.M, streamptr[0]);
        cudaDeviceSynchronize();
#ifdef USE_NVTX
        POP_RANGE
#endif 
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
        loopi++;
    }

#ifdef USE_NVTX
    POP_RANGE
#endif 
    for(int i=0; i<streamsize; i++) cudaStreamDestroy(streamptr[i]);
    for(int i=0; i<streamsize; i++) cublasDestroy_v2(cublashandleptr[i]);
    delete[] streamptr;
    delete[] cublashandleptr;
    double totalbytes = TLRMVMBytes(fp32ptr.FP32Rmat, 2*sizeof(float)) * (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>(totalbytes), Counter::kIsRate, Counter::kIs1000);

    SeismicPCMatrix seismicpcmat(datafolder, 
    acc, nb, freqlist[0], originM, originN);
    auto xmat = seismicpcmat.GetX();
    auto yv = seismicpcmat.Phase1();
    auto yu = seismicpcmat.Phase2();
    auto y = seismicpcmat.Phase3();
    // check finaly 
    CopyDataB2HD(fp32ptr.h_finaly, reinterpret_cast<complex<float>*>(fp32ptr.d_finaly), fp32ptr.M);
    auto finalyc = Matrix<complex<float>>(fp32ptr.h_finaly, fp32ptr.M, 1);
    cout << "final y " << finalyc.allclose(y) << endl;

    fp32ptr.FreeData();

}



BENCHMARK_DEFINE_F(SeismicFixture, SinglePtrStreamBenchmark)(benchmark::State& state)
{
    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Float32Ptr fp32ptr(paddingM, paddingN, nb, Rmats[freqlist[0]], FP32Rmats[freqlist[0]]);
    fp32ptr.InitData(datafolder, acc, freqlist[0], originN);

    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    fp32ptr.CopyData2GPU();

    float alpha = 1.0;
    float beta = 0.0;
    vector<double> rawtime;
    cout << "here " << endl;    
#ifdef USE_NVTX
    PUSH_RANGE("CUDAGraph", 1)
#endif 
size_t loopi = 0;
    for(auto st : state){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
#ifdef USE_NVTX
        PUSH_RANGE("ITERATION", loopi)
#endif 
        fp32ptr.CopyData2GPU();
        // phase 1
        for(int i=0; i<39; i++){
            int streamid = (i) % (streamsize);
            if(fp32ptr.AvMs[i] == 0) continue;
            // CUBLASCHECK(
                cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                fp32ptr.AvMs[i], fp32ptr.AvNs[i], fp32ptr.AvKs[i], 
                &alpha, fp32ptr.d_Avbp[i][0], fp32ptr.AvMs[i], 
                fp32ptr.d_xbp[i], fp32ptr.AvKs[i], 
                &beta, fp32ptr.d_yvbp[i][0], fp32ptr.AvMs[i]);
            // );
            // CUBLASCHECK(
                cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                fp32ptr.AvMs[i], fp32ptr.AvNs[i], fp32ptr.AvKs[i], 
                &alpha, fp32ptr.d_Avbp[i][1], fp32ptr.AvMs[i], 
                fp32ptr.d_xbp[i], fp32ptr.AvKs[i], 
                &beta, fp32ptr.d_yvbp[i][1], fp32ptr.AvMs[i]);
            // );
        }
        cudaDeviceSynchronize();
        // phase 2
        phase2(fp32ptr.d_yv[0], fp32ptr.d_yv[1], 
        fp32ptr.d_fp32colrank, fp32ptr.Ntg,fp32ptr.d_phase2mapping, 
        fp32ptr.d_yu, 
        fp32ptr.fp32granksum, streamptr[0]);
        cudaDeviceSynchronize();
        // phase 3
        for(int i=0; i<39; i++){
            int streamid = (i) % (streamsize);
            if(fp32ptr.AuMs[i] != 0){
                // CUBLASCHECK(
                    cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                    fp32ptr.AuMs[i], fp32ptr.AuNs[i], fp32ptr.AuKs[i], 
                    &alpha, fp32ptr.d_Aubp[i][0], fp32ptr.AuMs[i], 
                    fp32ptr.d_yubp[i], fp32ptr.AuKs[i], 
                    &beta, fp32ptr.d_ybp[i][0], fp32ptr.AuMs[i]);
                // );
                // CUBLASCHECK(
                    cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                    fp32ptr.AuMs[i], fp32ptr.AuNs[i], fp32ptr.AuKs[i], 
                    &alpha, fp32ptr.d_Aubp[i][1], fp32ptr.AuMs[i], 
                    fp32ptr.d_yubp[i], fp32ptr.AuKs[i], 
                    &beta, fp32ptr.d_ybp[i][1], fp32ptr.AuMs[i]);
                // ); 
            }
        }
        // final merge
        cudaDeviceSynchronize();
        phase3_merge(fp32ptr.d_y[0], fp32ptr.d_y[1], fp32ptr.nb, fp32ptr.d_finaly, fp32ptr.M, streamptr[0]);
        cudaDeviceSynchronize();
#ifdef USE_NVTX
        POP_RANGE
#endif 
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
        loopi++;
    }

#ifdef USE_NVTX
    POP_RANGE
#endif 
    for(int i=0; i<streamsize; i++) cudaStreamDestroy(streamptr[i]);
    for(int i=0; i<streamsize; i++) cublasDestroy_v2(cublashandleptr[i]);
    delete[] streamptr;
    delete[] cublashandleptr;
    double totalbytes = TLRMVMBytes(fp32ptr.FP32Rmat, 2*sizeof(float)) * (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>(totalbytes), Counter::kIsRate, Counter::kIs1000);

    SeismicPCMatrix seismicpcmat(datafolder, 
    acc, nb, freqlist[0], originM, originN);
    auto xmat = seismicpcmat.GetX();
    auto yv = seismicpcmat.Phase1();
    auto yu = seismicpcmat.Phase2();
    auto y = seismicpcmat.Phase3();
    // check finaly 
    CopyDataB2HD(fp32ptr.h_finaly, reinterpret_cast<complex<float>*>(fp32ptr.d_finaly), fp32ptr.M);
    auto finalyc = Matrix<complex<float>>(fp32ptr.h_finaly, fp32ptr.M, 1);
    cout << "final y " << finalyc.allclose(y) << endl;

    fp32ptr.FreeData();

}



BENCHMARK_DEFINE_F(SeismicFixture, TwoPtrStreamBenchmark)(benchmark::State& state)
{
    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto curgRmat = Rmats[freqlist[0]];
    auto Rmat1 = Rmats[freqlist[0]];
    auto Rmat2 = Rmats[freqlist[0]];
    // left part zero / right part one
    // for(int i=0; i<Rmat1.Row(); i++){
    //     for(int j=0; j<Rmat1.Col(); j++){
    //         if(j < Rmat1.Col()/2) Rmat1.SetElem(i,j, 0);
    //         else Rmat2.SetElem(i,j,0);
    //     }
    // }

    for(int i=0; i<Rmat1.Row(); i++){
        for(int j=0; j<Rmat1.Col(); j++){
            if( abs(i-j) % 2 == 0) Rmat1.SetElem(i,j,0);
            else Rmat2.SetElem(i,j,0);
        }
    }
    auto Rmat3 = Rmat1 + Rmat2;
    // cout << Rmat3.allclose(Rmats[freqlist[0]]) << endl;
    // cout << Rmat1.Block({0,10},{0,10}) << endl;
    // cout << Rmat2.Block({0,10},{0,10}) << endl;
    // we want to split this rmats
    
    Float32Ptr fp32ptr1(paddingM, paddingN, nb, Rmats[freqlist[0]], Rmat1);
    Float32Ptr fp32ptr2(paddingM, paddingN, nb, Rmats[freqlist[0]], Rmat2);
    fp32ptr1.InitData(datafolder, acc, freqlist[0], originN);
    fp32ptr2.InitData(datafolder, acc, freqlist[0], originN);


    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);


    fp32ptr1.CopyData2GPU();
    fp32ptr2.CopyData2GPU();
    vector<double> rawtime;
    float alpha = 1.0;
    float beta = 0.0;
    size_t loopi = 0;
    for(auto st : state){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        // phase 1
        for(int i=0; i<39; i++){
            int streamid = i % streamsize;
            if(fp32ptr1.AvMs[i] != 0){
                // CUBLASCHECK(
                    cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                    fp32ptr1.AvMs[i], fp32ptr1.AvNs[i], fp32ptr1.AvKs[i], 
                    &alpha, fp32ptr1.d_Avbp[i][0], fp32ptr1.AvMs[i], 
                    fp32ptr1.d_xbp[i], fp32ptr1.AvKs[i], 
                    &beta, fp32ptr1.d_yvbp[i][0], fp32ptr1.AvMs[i]);
                // );
                // CUBLASCHECK(
                    cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                    fp32ptr1.AvMs[i], fp32ptr1.AvNs[i], fp32ptr1.AvKs[i], 
                    &alpha, fp32ptr1.d_Avbp[i][1], fp32ptr1.AvMs[i], 
                    fp32ptr1.d_xbp[i], fp32ptr1.AvKs[i], 
                    &beta, fp32ptr1.d_yvbp[i][1], fp32ptr1.AvMs[i]);
                // );
            }
            if(fp32ptr2.AvMs[i] != 0){
                // CUBLASCHECK(
                    cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                    fp32ptr2.AvMs[i], fp32ptr2.AvNs[i], fp32ptr2.AvKs[i], 
                    &alpha, fp32ptr2.d_Avbp[i][0], fp32ptr2.AvMs[i], 
                    fp32ptr2.d_xbp[i], fp32ptr2.AvKs[i], 
                    &beta, fp32ptr2.d_yvbp[i][0], fp32ptr2.AvMs[i]);
                // );
                // CUBLASCHECK(
                    cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                    fp32ptr2.AvMs[i], fp32ptr2.AvNs[i], fp32ptr2.AvKs[i], 
                    &alpha, fp32ptr2.d_Avbp[i][1], fp32ptr2.AvMs[i], 
                    fp32ptr2.d_xbp[i], fp32ptr2.AvKs[i], 
                    &beta, fp32ptr2.d_yvbp[i][1], fp32ptr2.AvMs[i]);
                // );
            }
        }
        cudaDeviceSynchronize();
        // phase 2
        phase2(fp32ptr1.d_yv[0], fp32ptr1.d_yv[1], 
        fp32ptr1.d_fp32colrank, fp32ptr1.Ntg,fp32ptr1.d_phase2mapping, 
        fp32ptr1.d_yu, 
        fp32ptr1.fp32granksum, streamptr[0]);
        phase2(fp32ptr2.d_yv[0], fp32ptr2.d_yv[1], 
        fp32ptr2.d_fp32colrank, fp32ptr2.Ntg,fp32ptr2.d_phase2mapping, 
        fp32ptr2.d_yu, 
        fp32ptr2.fp32granksum, streamptr[1]);
        cudaDeviceSynchronize();
        // phase 3
        for(int i=0; i<39; i++){
            int streamid = i % streamsize;
            if(fp32ptr1.AuMs[i] != 0){
                // CUBLASCHECK(
                    cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                    fp32ptr1.AuMs[i], fp32ptr1.AuNs[i], fp32ptr1.AuKs[i], 
                    &alpha, fp32ptr1.d_Aubp[i][0], fp32ptr1.AuMs[i], 
                    fp32ptr1.d_yubp[i], fp32ptr1.AuKs[i], 
                    &beta, fp32ptr1.d_ybp[i][0], fp32ptr1.AuMs[i]);
                // );
                // CUBLASCHECK(
                    cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                    fp32ptr1.AuMs[i], fp32ptr1.AuNs[i], fp32ptr1.AuKs[i], 
                    &alpha, fp32ptr1.d_Aubp[i][1], fp32ptr1.AuMs[i], 
                    fp32ptr1.d_yubp[i], fp32ptr1.AuKs[i], 
                    &beta, fp32ptr1.d_ybp[i][1], fp32ptr1.AuMs[i]);
                // ); 
            }
            if(fp32ptr2.AuMs[i] != 0){
                // CUBLASCHECK(
                    cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                    fp32ptr2.AuMs[i], fp32ptr2.AuNs[i], fp32ptr2.AuKs[i], 
                    &alpha, fp32ptr2.d_Aubp[i][0], fp32ptr2.AuMs[i], 
                    fp32ptr2.d_yubp[i], fp32ptr2.AuKs[i], 
                    &beta, fp32ptr2.d_ybp[i][0], fp32ptr2.AuMs[i]);
                // );
                // CUBLASCHECK(
                    cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                    fp32ptr2.AuMs[i], fp32ptr2.AuNs[i], fp32ptr2.AuKs[i], 
                    &alpha, fp32ptr2.d_Aubp[i][1], fp32ptr2.AuMs[i], 
                    fp32ptr2.d_yubp[i], fp32ptr2.AuKs[i], 
                    &beta, fp32ptr2.d_ybp[i][1], fp32ptr2.AuMs[i]);
                // ); 
            }
        }
        cudaDeviceSynchronize();
        // final merge
        phase3_merge(fp32ptr1.d_y[0], fp32ptr1.d_y[1],
        fp32ptr1.nb, fp32ptr1.d_finaly, fp32ptr1.M, streamptr[0]);
        phase3_merge(fp32ptr2.d_y[0], fp32ptr2.d_y[1],
        fp32ptr2.nb, fp32ptr2.d_finaly, fp32ptr2.M, streamptr[1]);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
        loopi++;

    }



    double fp32ptr1bytes = TLRMVMBytes(fp32ptr1.FP32Rmat, 2*sizeof(float)) * (double)state.iterations();
    double fp32ptr2bytes = TLRMVMBytes(fp32ptr2.FP32Rmat, 2*sizeof(float)) * (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>((fp32ptr1bytes+fp32ptr2bytes)), Counter::kIsRate, Counter::kIs1000);






    SeismicPCMatrix seismicpcmat(datafolder, 
    acc, nb, freqlist[0], originM, originN);
    auto xmat = seismicpcmat.GetX();
    auto yv = seismicpcmat.Phase1();
    auto yu = seismicpcmat.Phase2();
    auto y = seismicpcmat.Phase3();
    
    // we directly do final test
    // check finaly 
    CopyDataB2HD(fp32ptr1.h_finaly, reinterpret_cast<complex<float>*>(fp32ptr1.d_finaly), fp32ptr1.M);
    CopyDataB2HD(fp32ptr2.h_finaly, reinterpret_cast<complex<float>*>(fp32ptr2.d_finaly), fp32ptr2.M);

    auto finalyc1 = Matrix<complex<float>>(fp32ptr1.h_finaly, fp32ptr1.M, 1);
    auto finalyc2 = Matrix<complex<float>>(fp32ptr2.h_finaly, fp32ptr2.M, 1);
    auto finalyc = finalyc1 + finalyc2;
    cout << " Two pointers final y " << finalyc.allclose(y) << endl;

    fp32ptr1.FreeData();
    fp32ptr2.FreeData();


}

BENCHMARK_DEFINE_F(SeismicFixture, SinglePtrStreamCUDAGraphBenchmark)
(benchmark::State& state)
{
    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Float32Ptr fp32ptr(paddingM, paddingN, nb, Rmats[freqlist[0]], FP32Rmats[freqlist[0]]);
    fp32ptr.InitData(datafolder, acc, freqlist[0], originN);

    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);


    // cuda graph
    bool graphCreated=false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaEvent_t *events;
    cudaEvent_t event_start;
    cudaEvent_t event_phase2finish;
    CUDACHECK(cudaEventCreate(&event_start));
    CUDACHECK(cudaEventCreate(&event_phase2finish));
    events = new cudaEvent_t[2*streamsize];
    for(int i=0; i<2*streamsize; i++) CUDACHECK(cudaEventCreate(&events[i]));



    fp32ptr.CopyData2GPU();
    float alpha = 1.0;
    float beta = 0.0;
    vector<double> rawtime;
#ifdef USE_NVTX
    PUSH_RANGE("CUDAGraph", 1)
#endif 
size_t loopi = 0;
    for(auto st : state){
        cudaDeviceSynchronize();
        cudaEventRecord(start);

#ifdef USE_NVTX
        PUSH_RANGE("ITERATION", loopi)
#endif 
        if(!graphCreated){
            cudaStreamBeginCapture(streamptr[0],cudaStreamCaptureModeGlobal);
            cudaEventRecord(event_start, streamptr[0]);
            for(int streami=1; streami<streamsize; streami++){
                cudaStreamWaitEvent(streamptr[streami], event_start);
            }
            // phase 1
            for(int i=0; i<39; i++){
                int streamid = (i) % (streamsize);
                int streamidnext = (i+20) % (streamsize);
                if(fp32ptr.AvMs[i] == 0) continue;
                CUBLASCHECK(
                    cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                    fp32ptr.AvMs[i], fp32ptr.AvNs[i], fp32ptr.AvKs[i], 
                    &alpha, fp32ptr.d_Avbp[i][0], fp32ptr.AvMs[i], 
                    fp32ptr.d_xbp[i], fp32ptr.AvKs[i], 
                    &beta, fp32ptr.d_yvbp[i][0], fp32ptr.AvMs[i]););
                CUBLASCHECK(
                    cublasgemm(cublashandleptr[streamidnext],CUBLAS_OP_N, CUBLAS_OP_N,
                    fp32ptr.AvMs[i], fp32ptr.AvNs[i], fp32ptr.AvKs[i], 
                    &alpha, fp32ptr.d_Avbp[i][1], fp32ptr.AvMs[i], 
                    fp32ptr.d_xbp[i], fp32ptr.AvKs[i], 
                    &beta, fp32ptr.d_yvbp[i][1], fp32ptr.AvMs[i]););
            }
            for(int streami=1; streami < streamsize; streami++){
                cudaEventRecord(events[streami], streamptr[streami]);
            }
            for(int streami=1; streami < streamsize; streami++){
                cudaStreamWaitEvent(streamptr[0], events[streami]);
            }
            // phase 2
            phase2(fp32ptr.d_yv[0], fp32ptr.d_yv[1], 
            fp32ptr.d_fp32colrank, fp32ptr.Ntg,fp32ptr.d_phase2mapping, 
            fp32ptr.d_yu, 
            fp32ptr.fp32granksum, streamptr[0]);
            cudaEventRecord(events[0], streamptr[0]);
            for(int streami=1; streami < streamsize; streami++){
                cudaStreamWaitEvent(streamptr[streami], events[0]);
            }
            // phase 3
            for(int i=0; i<39; i++){
                int streamid = (i) % (streamsize);
                int streamidnext = (i+20) % (streamsize);
                if(fp32ptr.AuMs[i] != 0){
                    CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                        fp32ptr.AuMs[i], fp32ptr.AuNs[i], fp32ptr.AuKs[i], 
                        &alpha, fp32ptr.d_Aubp[i][0], fp32ptr.AuMs[i], 
                        fp32ptr.d_yubp[i], fp32ptr.AuKs[i], 
                        &beta, fp32ptr.d_ybp[i][0], fp32ptr.AuMs[i]);
                    );
                    CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamidnext],CUBLAS_OP_N, CUBLAS_OP_N,
                        fp32ptr.AuMs[i], fp32ptr.AuNs[i], fp32ptr.AuKs[i], 
                        &alpha, fp32ptr.d_Aubp[i][1], fp32ptr.AuMs[i], 
                        fp32ptr.d_yubp[i], fp32ptr.AuKs[i], 
                        &beta, fp32ptr.d_ybp[i][1], fp32ptr.AuMs[i]);
                    ); 
                }
            }
            // final merge
            for(int streami=1; streami < streamsize; streami++){
                cudaEventRecord(events[streamsize + streami], streamptr[streami]);
            }
            for(int streami=1; streami < streamsize; streami++){
                cudaStreamWaitEvent(streamptr[0], events[streamsize + streami]);
            }
            phase3_merge(fp32ptr.d_y[0], fp32ptr.d_y[1], fp32ptr.nb, fp32ptr.d_finaly, fp32ptr.M, streamptr[0]);
            cudaStreamEndCapture(streamptr[0], &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated = true;
        }

        cudaGraphLaunch(instance, streamptr[0]);
        cudaStreamSynchronize(streamptr[0]);


#ifdef USE_NVTX
        POP_RANGE
#endif 
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
        loopi++;
    }

#ifdef USE_NVTX
    POP_RANGE
#endif 
    for(int i=0; i<streamsize; i++) cudaStreamDestroy(streamptr[i]);
    for(int i=0; i<streamsize; i++) cublasDestroy_v2(cublashandleptr[i]);
    delete[] streamptr;
    delete[] cublashandleptr;
    double totalbytes = TLRMVMBytes(fp32ptr.FP32Rmat, 2*sizeof(float)) * (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>(totalbytes), Counter::kIsRate, Counter::kIs1000);

    // SeismicPCMatrix seismicpcmat(datafolder, 
    // acc, nb, freqlist[0], originM, originN);
    // auto xmat = seismicpcmat.GetX();
    // auto yv = seismicpcmat.Phase1();
    // auto yu = seismicpcmat.Phase2();
    // auto y = seismicpcmat.Phase3();
    // // check finaly 
    // CopyDataB2HD(fp32ptr.h_finaly, reinterpret_cast<complex<float>*>(fp32ptr.d_finaly), fp32ptr.M);
    // auto finalyc = Matrix<complex<float>>(fp32ptr.h_finaly, fp32ptr.M, 1);
    // cout << "final y " << finalyc.allclose(y) << endl;

    fp32ptr.FreeData();

}

BENCHMARK_DEFINE_F(SeismicFixture, SingleComplexPtrStreamCUDAGraphBenchmark)
(benchmark::State& state)
{
    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    ComplexPtr complexptr(paddingM, paddingN, nb, Rmats[freqlist[0]], FP32Rmats[freqlist[0]]);
    complexptr.InitData(datafolder, acc, freqlist[0], originN);
    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);



    // cuda graph
    bool graphCreated=false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaEvent_t *events;
    cudaEvent_t event_start;
    cudaEvent_t event_phase2finish;
    CUDACHECK(cudaEventCreate(&event_start));
    CUDACHECK(cudaEventCreate(&event_phase2finish));
    events = new cudaEvent_t[2*streamsize];
    for(int i=0; i<2*streamsize; i++) CUDACHECK(cudaEventCreate(&events[i]));


    complexptr.CopyData2GPU();
    cuComplex alpha;
    cuComplex beta;
    alpha.x = 1.0; alpha.y = 0.0;
    beta.x = 0.0; beta.y = 0.0;
    vector<double> rawtime;

#ifdef USE_NVTX
    PUSH_RANGE("CUDAGraph", 1)
#endif 
size_t loopi = 0;
    for(auto st : state){
        cudaDeviceSynchronize();
        cudaEventRecord(start);

#ifdef USE_NVTX
        PUSH_RANGE("ITERATION", loopi)
#endif 
        if(!graphCreated){
            cudaStreamBeginCapture(streamptr[0],cudaStreamCaptureModeGlobal);
            cudaEventRecord(event_start, streamptr[0]);
            for(int streami=1; streami<streamsize; streami++){
                cudaStreamWaitEvent(streamptr[streami], event_start);
            }
            // phase 1
            for(int i=0; i<39; i++){
                int streamid = (i) % (streamsize);
                if(complexptr.AvMs[i] != 0){
                    CUBLASCHECK(cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                    complexptr.AvMs[i], complexptr.AvNs[i], complexptr.AvKs[i], 
                    &alpha, complexptr.d_Avbp[i], complexptr.AvMs[i], 
                    complexptr.d_xbp[i], complexptr.AvKs[i], 
                    &beta, complexptr.d_yvbp[i], complexptr.AvMs[i]));
                }
            }
            for(int streami=1; streami < streamsize; streami++){
                cudaEventRecord(events[streami], streamptr[streami]);
            }
            for(int streami=1; streami < streamsize; streami++){
                cudaStreamWaitEvent(streamptr[0], events[streami]);
            }
            // phase 2
            phase2_complex(complexptr.d_yv, complexptr.d_phase2mapping, complexptr.d_yu, 
            complexptr.complexgranksum, streamptr[0]);
            cudaEventRecord(events[0], streamptr[0]);
            for(int streami=1; streami < streamsize; streami++){
                cudaStreamWaitEvent(streamptr[streami], events[0]);
            }
            // phase 3
            for(int i=0; i<39; i++){
                int streamid = (i) % (streamsize);
                if(complexptr.AuMs[i] != 0){
                    CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                        complexptr.AuMs[i], complexptr.AuNs[i], complexptr.AuKs[i], 
                        &alpha, complexptr.d_Aubp[i], complexptr.AuMs[i], 
                        complexptr.d_yubp[i], complexptr.AuKs[i], 
                        &beta, complexptr.d_ybp[i], complexptr.AuMs[i])
                    );
                }
            }
            // final merge
            for(int streami=1; streami < streamsize; streami++){
                cudaEventRecord(events[streamsize + streami], streamptr[streami]);
            }
            for(int streami=1; streami < streamsize; streami++){
                cudaStreamWaitEvent(streamptr[0], events[streamsize + streami]);
            }
            cudaStreamEndCapture(streamptr[0], &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated = true;
        }

        cudaGraphLaunch(instance, streamptr[0]);
        cudaStreamSynchronize(streamptr[0]);


#ifdef USE_NVTX
        POP_RANGE
#endif 
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
        loopi++;
    }

#ifdef USE_NVTX
    POP_RANGE
#endif 
    for(int i=0; i<streamsize; i++) cudaStreamDestroy(streamptr[i]);
    for(int i=0; i<streamsize; i++) cublasDestroy_v2(cublashandleptr[i]);
    delete[] streamptr;
    delete[] cublashandleptr;
    double totalbytes = TLRMVMBytes(complexptr.ComplexRmat, 2*sizeof(float)) * (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>(totalbytes), Counter::kIsRate, Counter::kIs1000);

    SeismicPCMatrix seismicpcmat(datafolder, 
    acc, nb, freqlist[0], originM, originN);
    Matrix<complex<float>> xmat = seismicpcmat.GetX();
    size_t coloffset = 0;

    // check for final yv output
    CopyDataB2HD(complexptr.h_yv, complexptr.d_yv, complexptr.complexgranksum);
    CopyDataB2HD(complexptr.h_yu, complexptr.d_yu, complexptr.complexgranksum);
    CopyDataB2HD(complexptr.h_y, complexptr.d_y, complexptr.M);
    Matrix<complex<float>> densemat = seismicpcmat.GetDense();
    Matrix<complex<float>> yv = seismicpcmat.Phase1();
    auto cyv = Matrix<complex<float>>(complexptr.h_yv, complexptr.complexgranksum, 1);
    Matrix<complex<float>> yu = seismicpcmat.Phase2();
    auto y = seismicpcmat.Phase3();
    auto cyu = Matrix<complex<float>>(complexptr.h_yu, complexptr.complexgranksum, 1);
    auto finalyc = Matrix<complex<float>>(complexptr.h_y, complexptr.M, 1);
    cout << "yu output " << cyu.allclose(yu) << endl;
    cout << "final y " << finalyc.allclose(y) << endl;
    complexptr.FreeData();

}


BENCHMARK_DEFINE_F(SeismicFixture, SinglePtrFP16StreamCUDAGraphBenchmark)
(benchmark::State& state)
{
    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    Float16Ptr fp16ptr(paddingM, paddingN, nb, Rmats[freqlist[0]], FP32Rmats[freqlist[0]]);
    fp16ptr.InitData(datafolder, acc, freqlist[0], originN);

    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);


    // cuda graph
    bool graphCreated=false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaEvent_t *events;
    cudaEvent_t event_start;
    cudaEvent_t event_phase2finish;
    CUDACHECK(cudaEventCreate(&event_start));
    CUDACHECK(cudaEventCreate(&event_phase2finish));
    events = new cudaEvent_t[2*streamsize];
    for(int i=0; i<2*streamsize; i++) CUDACHECK(cudaEventCreate(&events[i]));


    fp16ptr.CopyData2GPU();
    half alpha = (half)1.0;
    half beta = (half)0.0;
    vector<double> rawtime;

#ifdef USE_NVTX
    PUSH_RANGE("CUDAGraph", 1)
#endif 
size_t loopi = 0;
    for(auto st : state){
        cudaDeviceSynchronize();
        cudaEventRecord(start);

#ifdef USE_NVTX
        PUSH_RANGE("ITERATION", loopi)
#endif 
        if(!graphCreated){
            cudaStreamBeginCapture(streamptr[0],cudaStreamCaptureModeGlobal);
            cudaEventRecord(event_start, streamptr[0]);
            for(int streami=1; streami<streamsize; streami++){
                cudaStreamWaitEvent(streamptr[streami], event_start);
            }
            // phase 1
            for(int i=0; i<39; i++){
                int streamid = (i) % (streamsize);
                int streamidnext = (i+20) % (streamsize);
                if(fp16ptr.AvMs[i] == 0) continue;
                // CUBLASCHECK(
                    cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                    fp16ptr.AvMs[i], fp16ptr.AvNs[i], fp16ptr.AvKs[i], 
                    &alpha, fp16ptr.d_Avbp[i][0], fp16ptr.AvMs[i], 
                    fp16ptr.d_xbp[i], fp16ptr.AvKs[i], 
                    &beta, fp16ptr.d_yvbp[i][0], fp16ptr.AvMs[i]);
                // );
                // CUBLASCHECK(
                    cublasgemm(cublashandleptr[streamidnext],CUBLAS_OP_N, CUBLAS_OP_N,
                    fp16ptr.AvMs[i], fp16ptr.AvNs[i], fp16ptr.AvKs[i], 
                    &alpha, fp16ptr.d_Avbp[i][1], fp16ptr.AvMs[i], 
                    fp16ptr.d_xbp[i], fp16ptr.AvKs[i], 
                    &beta, fp16ptr.d_yvbp[i][1], fp16ptr.AvMs[i]);
                // );
            }
            for(int streami=1; streami < streamsize; streami++){
                cudaEventRecord(events[streami], streamptr[streami]);
            }
            for(int streami=1; streami < streamsize; streami++){
                cudaStreamWaitEvent(streamptr[0], events[streami]);
            }
            // phase 2
            phase2_half(fp16ptr.d_yv[0], fp16ptr.d_yv[1], 
            fp16ptr.d_fp16colrank, fp16ptr.Ntg,fp16ptr.d_phase2mapping, 
            fp16ptr.d_yu, 
            fp16ptr.fp16granksum, streamptr[0]);
            cudaEventRecord(events[0], streamptr[0]);
            for(int streami=1; streami < streamsize; streami++){
                cudaStreamWaitEvent(streamptr[streami], events[0]);
            }
            // phase 3
            for(int i=0; i<39; i++){
                int streamid = (i) % (streamsize);
                int streamidnext = (i+20) % (streamsize);
                if(fp16ptr.AuMs[i] != 0){
                    // CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                        fp16ptr.AuMs[i], fp16ptr.AuNs[i], fp16ptr.AuKs[i], 
                        &alpha, fp16ptr.d_Aubp[i][0], fp16ptr.AuMs[i], 
                        fp16ptr.d_yubp[i], fp16ptr.AuKs[i], 
                        &beta, fp16ptr.d_ybp[i][0], fp16ptr.AuMs[i]);
                    // );
                    // CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamidnext],CUBLAS_OP_N, CUBLAS_OP_N,
                        fp16ptr.AuMs[i], fp16ptr.AuNs[i], fp16ptr.AuKs[i], 
                        &alpha, fp16ptr.d_Aubp[i][1], fp16ptr.AuMs[i], 
                        fp16ptr.d_yubp[i], fp16ptr.AuKs[i], 
                        &beta, fp16ptr.d_ybp[i][1], fp16ptr.AuMs[i]);
                    // ); 
                }
            }
            // final merge
            for(int streami=1; streami < streamsize; streami++){
                cudaEventRecord(events[streamsize + streami], streamptr[streami]);
            }
            for(int streami=1; streami < streamsize; streami++){
                cudaStreamWaitEvent(streamptr[0], events[streamsize + streami]);
            }
            phase3_merge_half(fp16ptr.d_y[0], fp16ptr.d_y[1], fp16ptr.nb, fp16ptr.d_finaly, fp16ptr.M, streamptr[0]);
            cudaStreamEndCapture(streamptr[0], &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated = true;
        }

        cudaGraphLaunch(instance, streamptr[0]);
        cudaStreamSynchronize(streamptr[0]);


#ifdef USE_NVTX
        POP_RANGE
#endif 
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
        loopi++;
    }

#ifdef USE_NVTX
    POP_RANGE
#endif 
    for(int i=0; i<streamsize; i++) cudaStreamDestroy(streamptr[i]);
    for(int i=0; i<streamsize; i++) cublasDestroy_v2(cublashandleptr[i]);
    delete[] streamptr;
    delete[] cublashandleptr;
    double totalbytes = TLRMVMBytes(fp16ptr.FP16Rmat, 2*sizeof(float)) * (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>(totalbytes), Counter::kIsRate, Counter::kIs1000);

    SeismicPCMatrix seismicpcmat(datafolder, 
    acc, nb, freqlist[0], originM, originN);
    auto xmat = seismicpcmat.GetX();
    auto yv = seismicpcmat.Phase1();
    auto yu = seismicpcmat.Phase2();
    auto y = seismicpcmat.Phase3();
    // check finaly 
    CopyDataB2HD(fp16ptr.h_finaly, reinterpret_cast<complex<float>*>(fp16ptr.d_finaly), fp16ptr.M);
    auto finalyc = Matrix<complex<float>>(fp16ptr.h_finaly, fp16ptr.M, 1);
    cout << "final y " << finalyc.allclose(y) << endl;

    fp16ptr.FreeData();

}


BENCHMARK_DEFINE_F(SeismicFixture, TwoPtrStreamCUDAGraphBenchmark)(benchmark::State& state)
{
    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto curgRmat = Rmats[freqlist[0]];
    auto Rmat1 = Rmats[freqlist[0]];
    auto Rmat2 = Rmats[freqlist[0]];
    // left part zero / right part one
    // for(int i=0; i<Rmat1.Row(); i++){
    //     for(int j=0; j<Rmat1.Col(); j++){
    //         if(j < Rmat1.Col()/2) Rmat1.SetElem(i,j, 0);
    //         else Rmat2.SetElem(i,j,0);
    //     }
    // }

    for(int i=0; i<Rmat1.Row(); i++){
        for(int j=0; j<Rmat1.Col(); j++){
            if( abs(i-j) % 2 == 0) Rmat1.SetElem(i,j,0);
            else Rmat2.SetElem(i,j,0);
        }
    }
    auto Rmat3 = Rmat1 + Rmat2;
    // cout << Rmat3.allclose(Rmats[freqlist[0]]) << endl;
    // cout << Rmat1.Block({0,10},{0,10}) << endl;
    // cout << Rmat2.Block({0,10},{0,10}) << endl;
    // we want to split this rmats
    



    Float32Ptr fp32ptr1(paddingM, paddingN, nb, Rmats[freqlist[0]], Rmat1);
    Float32Ptr fp32ptr2(paddingM, paddingN, nb, Rmats[freqlist[0]], Rmat2);
    fp32ptr1.InitData(datafolder, acc, freqlist[0], originN);
    fp32ptr2.InitData(datafolder, acc, freqlist[0], originN);


    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);


    // cuda graph
    bool graphCreated=false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaEvent_t *events;
    cudaEvent_t event_start;
    cudaEvent_t event_finish;
    CUDACHECK(cudaEventCreate(&event_start));
    CUDACHECK(cudaEventCreate(&event_finish));
    events = new cudaEvent_t[2*streamsize];
    for(int i=0; i<2*streamsize; i++) CUDACHECK(cudaEventCreate(&events[i]));




    fp32ptr1.CopyData2GPU();
    fp32ptr2.CopyData2GPU();
    vector<double> rawtime;
    float alpha = 1.0;
    float beta = 0.0;
    size_t loopi = 0;
    for(auto st : state){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        if(!graphCreated){
            cudaStreamBeginCapture(streamptr[0],cudaStreamCaptureModeGlobal);
            cudaEventRecord(event_start, streamptr[0]);
            for(int streami=1; streami<streamsize; streami++){
                cudaStreamWaitEvent(streamptr[streami], event_start);
            }
            // phase 1
            for(int i=0; i<39; i++){
                int streamid = i % streamsize;
                if(fp32ptr1.AvMs[i] != 0){
                    // CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                        fp32ptr1.AvMs[i], fp32ptr1.AvNs[i], fp32ptr1.AvKs[i], 
                        &alpha, fp32ptr1.d_Avbp[i][0], fp32ptr1.AvMs[i], 
                        fp32ptr1.d_xbp[i], fp32ptr1.AvKs[i], 
                        &beta, fp32ptr1.d_yvbp[i][0], fp32ptr1.AvMs[i]);
                    // );
                    // CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                        fp32ptr1.AvMs[i], fp32ptr1.AvNs[i], fp32ptr1.AvKs[i], 
                        &alpha, fp32ptr1.d_Avbp[i][1], fp32ptr1.AvMs[i], 
                        fp32ptr1.d_xbp[i], fp32ptr1.AvKs[i], 
                        &beta, fp32ptr1.d_yvbp[i][1], fp32ptr1.AvMs[i]);
                    // );
                }
                if(fp32ptr2.AvMs[i] != 0){
                    // CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                        fp32ptr2.AvMs[i], fp32ptr2.AvNs[i], fp32ptr2.AvKs[i], 
                        &alpha, fp32ptr2.d_Avbp[i][0], fp32ptr2.AvMs[i], 
                        fp32ptr2.d_xbp[i], fp32ptr2.AvKs[i], 
                        &beta, fp32ptr2.d_yvbp[i][0], fp32ptr2.AvMs[i]);
                    // );
                    // CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                        fp32ptr2.AvMs[i], fp32ptr2.AvNs[i], fp32ptr2.AvKs[i], 
                        &alpha, fp32ptr2.d_Avbp[i][1], fp32ptr2.AvMs[i], 
                        fp32ptr2.d_xbp[i], fp32ptr2.AvKs[i], 
                        &beta, fp32ptr2.d_yvbp[i][1], fp32ptr2.AvMs[i]);
                    // );
                }
            }
            for(int streami=0; streami < streamsize; streami++){
                cudaEventRecord(events[streami], streamptr[streami]);
            }
            for(int streami=1; streami < streamsize; streami++){
                cudaStreamWaitEvent(streamptr[0], events[streami]);
            }
            // phase 2
            phase2(fp32ptr1.d_yv[0], fp32ptr1.d_yv[1], 
            fp32ptr1.d_fp32colrank, fp32ptr1.Ntg,fp32ptr1.d_phase2mapping, 
            fp32ptr1.d_yu, 
            fp32ptr1.fp32granksum, streamptr[0]);
            phase2(fp32ptr2.d_yv[0], fp32ptr2.d_yv[1], 
            fp32ptr2.d_fp32colrank, fp32ptr2.Ntg,fp32ptr2.d_phase2mapping, 
            fp32ptr2.d_yu, 
            fp32ptr2.fp32granksum, streamptr[0]);
            cudaEventRecord(events[0], streamptr[0]);
            for(int streami=1; streami < streamsize; streami++){
                cudaStreamWaitEvent(streamptr[streami], events[0]);
            }
            // phase 3
            for(int i=0; i<39; i++){
                int streamid = i % streamsize;
                if(fp32ptr1.AuMs[i] != 0){
                    // CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                        fp32ptr1.AuMs[i], fp32ptr1.AuNs[i], fp32ptr1.AuKs[i], 
                        &alpha, fp32ptr1.d_Aubp[i][0], fp32ptr1.AuMs[i], 
                        fp32ptr1.d_yubp[i], fp32ptr1.AuKs[i], 
                        &beta, fp32ptr1.d_ybp[i][0], fp32ptr1.AuMs[i]);
                    // );
                    // CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                        fp32ptr1.AuMs[i], fp32ptr1.AuNs[i], fp32ptr1.AuKs[i], 
                        &alpha, fp32ptr1.d_Aubp[i][1], fp32ptr1.AuMs[i], 
                        fp32ptr1.d_yubp[i], fp32ptr1.AuKs[i], 
                        &beta, fp32ptr1.d_ybp[i][1], fp32ptr1.AuMs[i]);
                    // ); 
                }
                if(fp32ptr2.AuMs[i] != 0){
                    // CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                        fp32ptr2.AuMs[i], fp32ptr2.AuNs[i], fp32ptr2.AuKs[i], 
                        &alpha, fp32ptr2.d_Aubp[i][0], fp32ptr2.AuMs[i], 
                        fp32ptr2.d_yubp[i], fp32ptr2.AuKs[i], 
                        &beta, fp32ptr2.d_ybp[i][0], fp32ptr2.AuMs[i]);
                    // );
                    // CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                        fp32ptr2.AuMs[i], fp32ptr2.AuNs[i], fp32ptr2.AuKs[i], 
                        &alpha, fp32ptr2.d_Aubp[i][1], fp32ptr2.AuMs[i], 
                        fp32ptr2.d_yubp[i], fp32ptr2.AuKs[i], 
                        &beta, fp32ptr2.d_ybp[i][1], fp32ptr2.AuMs[i]);
                    // ); 
                }
            }
            // final merge
            for(int streami=0; streami < streamsize; streami++){
                cudaEventRecord(events[streamsize + streami], streamptr[streami]);
            }
            for(int streami=1; streami < streamsize; streami++){
                cudaStreamWaitEvent(streamptr[0], events[streamsize + streami]);
            }
            // final merge
            phase3_merge(fp32ptr1.d_y[0], fp32ptr1.d_y[1],
            fp32ptr1.nb, fp32ptr1.d_finaly, fp32ptr1.M, streamptr[0]);
            phase3_merge(fp32ptr2.d_y[0], fp32ptr2.d_y[1],
            fp32ptr2.nb, fp32ptr2.d_finaly, fp32ptr2.M, streamptr[0]);
            cudaStreamEndCapture(streamptr[0], &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated = true;
        }
        cudaGraphLaunch(instance, streamptr[0]);
        cudaStreamSynchronize(streamptr[0]);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
        loopi++;

    }

    for(int i=0; i<streamsize; i++) cudaStreamDestroy(streamptr[i]);
    for(int i=0; i<streamsize; i++) cublasDestroy_v2(cublashandleptr[i]);
    delete[] streamptr;
    delete[] cublashandleptr;

    double fp32ptr1bytes = TLRMVMBytes(fp32ptr1.FP32Rmat, 2*sizeof(float)) * (double)state.iterations();
    double fp32ptr2bytes = TLRMVMBytes(fp32ptr2.FP32Rmat, 2*sizeof(float)) * (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>((fp32ptr1bytes+fp32ptr2bytes)), Counter::kIsRate, Counter::kIs1000);






    SeismicPCMatrix seismicpcmat(datafolder, 
    acc, nb, freqlist[0], originM, originN);
    auto xmat = seismicpcmat.GetX();
    auto yv = seismicpcmat.Phase1();
    auto yu = seismicpcmat.Phase2();
    auto y = seismicpcmat.Phase3();
    
    // we directly do final test
    // check finaly 
    CopyDataB2HD(fp32ptr1.h_finaly, reinterpret_cast<complex<float>*>(fp32ptr1.d_finaly), fp32ptr1.M);
    CopyDataB2HD(fp32ptr2.h_finaly, reinterpret_cast<complex<float>*>(fp32ptr2.d_finaly), fp32ptr2.M);

    auto finalyc1 = Matrix<complex<float>>(fp32ptr1.h_finaly, fp32ptr1.M, 1);
    auto finalyc2 = Matrix<complex<float>>(fp32ptr2.h_finaly, fp32ptr2.M, 1);
    auto finalyc = finalyc1 + finalyc2;
    cout << " Two pointers final y " << finalyc.allclose(y) << endl;

    fp32ptr1.FreeData();
    fp32ptr2.FreeData();

}



BENCHMARK_DEFINE_F(SeismicFixture, FP32FP16CUDAGraphBenchmark)(benchmark::State& state)
{

    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto curgRmat = Rmats[freqlist[0]];
    auto Rmat1 = Rmats[freqlist[0]];
    auto Rmat2 = Rmats[freqlist[0]];
    // left part zero / right part one
    // for(int i=0; i<Rmat1.Row(); i++){
    //     for(int j=0; j<Rmat1.Col(); j++){
    //         if(j < Rmat1.Col()/2) Rmat1.SetElem(i,j, 0);
    //         else Rmat2.SetElem(i,j,0);
    //     }
    // }

    for(int i=0; i<Rmat1.Row(); i++){
        for(int j=0; j<Rmat1.Col(); j++){
            if( abs(i-j) % 2 == 0) Rmat1.SetElem(i,j,0);
            else Rmat2.SetElem(i,j,0);
        }
    }
    auto Rmat3 = Rmat1 + Rmat2;
    // cout << Rmat3.allclose(Rmats[freqlist[0]]) << endl;
    // cout << Rmat1.Block({0,10},{0,10}) << endl;
    // cout << Rmat2.Block({0,10},{0,10}) << endl;
    // we want to split this rmats
    
    Float32Ptr fp32ptr1(paddingM, paddingN, nb, Rmats[freqlist[0]], Rmat1);
    Float16Ptr fp16ptr2(paddingM, paddingN, nb, Rmats[freqlist[0]], Rmat2);
    fp32ptr1.InitData(datafolder, acc, freqlist[0], originN);
    fp16ptr2.InitData(datafolder, acc, freqlist[0], originN);


    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);


    // cuda graph
    bool graphCreated=false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    cudaEvent_t *events;
    cudaEvent_t event_start;
    cudaEvent_t event_finish;
    CUDACHECK(cudaEventCreate(&event_start));
    CUDACHECK(cudaEventCreate(&event_finish));
    events = new cudaEvent_t[2*streamsize];
    for(int i=0; i<2*streamsize; i++) CUDACHECK(cudaEventCreate(&events[i]));



    fp32ptr1.CopyData2GPU();
    fp16ptr2.CopyData2GPU();
    half alpha_half = (half)1.0;
    half beta_half = (half)0.0;
    float alpha_float = 1.0;
    float beta_float = 0.0;
    size_t loopi = 0;
    vector<double> rawtime;
    for(auto st : state){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        if(!graphCreated){
            cudaStreamBeginCapture(streamptr[0],cudaStreamCaptureModeGlobal);
            cudaEventRecord(event_start, streamptr[0]);
            for(int streami=1; streami<streamsize; streami++){
                cudaStreamWaitEvent(streamptr[streami], event_start);
            }
            // phase 1
            for(int i=0; i<39; i++){
                int streamid0 = i % streamsize;
                int streamid1 = (i+10) % (streamsize);
                int streamid2 = (i+20) % (streamsize);
                int streamid3 = (i+30) % (streamsize);
                if(fp32ptr1.AvMs[i] != 0){
                    // CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamid0],CUBLAS_OP_N, CUBLAS_OP_N,
                        fp32ptr1.AvMs[i], fp32ptr1.AvNs[i], fp32ptr1.AvKs[i], 
                        &alpha_float, fp32ptr1.d_Avbp[i][0], fp32ptr1.AvMs[i], 
                        fp32ptr1.d_xbp[i], fp32ptr1.AvKs[i], 
                        &beta_float, fp32ptr1.d_yvbp[i][0], fp32ptr1.AvMs[i]);
                    // );
                    // CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamid1],CUBLAS_OP_N, CUBLAS_OP_N,
                        fp32ptr1.AvMs[i], fp32ptr1.AvNs[i], fp32ptr1.AvKs[i], 
                        &alpha_float, fp32ptr1.d_Avbp[i][1], fp32ptr1.AvMs[i], 
                        fp32ptr1.d_xbp[i], fp32ptr1.AvKs[i], 
                        &beta_float, fp32ptr1.d_yvbp[i][1], fp32ptr1.AvMs[i]);
                    // );
                }
                if(fp16ptr2.AvMs[i] != 0){
                    // CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamid2],CUBLAS_OP_N, CUBLAS_OP_N,
                        fp16ptr2.AvMs[i], fp16ptr2.AvNs[i], fp16ptr2.AvKs[i], 
                        &alpha_half, fp16ptr2.d_Avbp[i][0], fp16ptr2.AvMs[i], 
                        fp16ptr2.d_xbp[i], fp16ptr2.AvKs[i], 
                        &beta_half, fp16ptr2.d_yvbp[i][0], fp16ptr2.AvMs[i]);
                    // );
                    // CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamid3],CUBLAS_OP_N, CUBLAS_OP_N,
                        fp16ptr2.AvMs[i], fp16ptr2.AvNs[i], fp16ptr2.AvKs[i], 
                        &alpha_half, fp16ptr2.d_Avbp[i][1], fp16ptr2.AvMs[i], 
                        fp16ptr2.d_xbp[i], fp16ptr2.AvKs[i], 
                        &beta_half, fp16ptr2.d_yvbp[i][1], fp16ptr2.AvMs[i]);
                    // );
                }
            }
            for(int streami=1; streami < streamsize; streami++){
                cudaEventRecord(events[streami], streamptr[streami]);
            }
            for(int streami=1; streami < streamsize; streami++){
                cudaStreamWaitEvent(streamptr[0], events[streami]);
            }
            // phase 2
            phase2(fp32ptr1.d_yv[0], fp32ptr1.d_yv[1], 
            fp32ptr1.d_fp32colrank, fp32ptr1.Ntg,fp32ptr1.d_phase2mapping, 
            fp32ptr1.d_yu, fp32ptr1.fp32granksum, streamptr[0]);
            phase2_half(fp16ptr2.d_yv[0], fp16ptr2.d_yv[1], 
            fp16ptr2.d_fp16colrank, fp16ptr2.Ntg,fp16ptr2.d_phase2mapping, 
            fp16ptr2.d_yu, fp16ptr2.fp16granksum, streamptr[0]);
            cudaEventRecord(events[0], streamptr[0]);
            for(int streami=1; streami < streamsize; streami++){
                cudaStreamWaitEvent(streamptr[streami], events[0]);
            }
            // phase 3
            for(int i=0; i<39; i++){
                int streamid0 = i % streamsize;
                int streamid1 = (i+10) % (streamsize);
                int streamid2 = (i+20) % (streamsize);
                int streamid3 = (i+30) % (streamsize);
                if(fp32ptr1.AuMs[i] != 0){
                    // CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamid0],CUBLAS_OP_N, CUBLAS_OP_N,
                        fp32ptr1.AuMs[i], fp32ptr1.AuNs[i], fp32ptr1.AuKs[i], 
                        &alpha_float, fp32ptr1.d_Aubp[i][0], fp32ptr1.AuMs[i], 
                        fp32ptr1.d_yubp[i], fp32ptr1.AuKs[i], 
                        &beta_float, fp32ptr1.d_ybp[i][0], fp32ptr1.AuMs[i]);
                    // );
                    // CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamid1],CUBLAS_OP_N, CUBLAS_OP_N,
                        fp32ptr1.AuMs[i], fp32ptr1.AuNs[i], fp32ptr1.AuKs[i], 
                        &alpha_float, fp32ptr1.d_Aubp[i][1], fp32ptr1.AuMs[i], 
                        fp32ptr1.d_yubp[i], fp32ptr1.AuKs[i], 
                        &beta_float, fp32ptr1.d_ybp[i][1], fp32ptr1.AuMs[i]);
                    // ); 
                }
                if(fp16ptr2.AuMs[i] != 0){
                    // CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamid2],CUBLAS_OP_N, CUBLAS_OP_N,
                        fp16ptr2.AuMs[i], fp16ptr2.AuNs[i], fp16ptr2.AuKs[i], 
                        &alpha_half, fp16ptr2.d_Aubp[i][0], fp16ptr2.AuMs[i], 
                        fp16ptr2.d_yubp[i], fp16ptr2.AuKs[i], 
                        &beta_half, fp16ptr2.d_ybp[i][0], fp16ptr2.AuMs[i]);
                    // );
                    // CUBLASCHECK(
                        cublasgemm(cublashandleptr[streamid3],CUBLAS_OP_N, CUBLAS_OP_N,
                        fp16ptr2.AuMs[i], fp16ptr2.AuNs[i], fp16ptr2.AuKs[i], 
                        &alpha_half, fp16ptr2.d_Aubp[i][1], fp16ptr2.AuMs[i], 
                        fp16ptr2.d_yubp[i], fp16ptr2.AuKs[i], 
                        &beta_half, fp16ptr2.d_ybp[i][1], fp16ptr2.AuMs[i]);
                    // ); 
                }
            }
            // final merge
            for(int streami=0; streami < streamsize; streami++){
                cudaEventRecord(events[streamsize + streami], streamptr[streami]);
            }
            for(int streami=1; streami < streamsize; streami++){
                cudaStreamWaitEvent(streamptr[0], events[streamsize + streami]);
            }
            phase3_merge(fp32ptr1.d_y[0], fp32ptr1.d_y[1],
            fp32ptr1.nb, fp32ptr1.d_finaly, fp32ptr1.M, streamptr[0]);
            phase3_merge_half(fp16ptr2.d_y[0], fp16ptr2.d_y[1],
            fp16ptr2.nb, fp16ptr2.d_finaly, fp16ptr2.M, streamptr[0]);
            cudaStreamEndCapture(streamptr[0], &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated = true;
        }
        cudaGraphLaunch(instance, streamptr[0]);
        cudaStreamSynchronize(streamptr[0]);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
        loopi++;
    }

    for(int i=0; i<streamsize; i++) cudaStreamDestroy(streamptr[i]);
    for(int i=0; i<streamsize; i++) cublasDestroy_v2(cublashandleptr[i]);
    delete[] streamptr;
    delete[] cublashandleptr;

    double fp32ptr1bytes = TLRMVMBytes(fp32ptr1.FP32Rmat, 2*sizeof(float)) * (double)state.iterations();
    double fp16ptr2bytes = TLRMVMBytes(fp16ptr2.FP16Rmat, 2*sizeof(half)) * (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>((fp32ptr1bytes+fp16ptr2bytes)), Counter::kIsRate, Counter::kIs1000);


    SeismicPCMatrix seismicpcmat(datafolder, 
    acc, nb, freqlist[0], originM, originN);
    auto xmat = seismicpcmat.GetX();
    auto yv = seismicpcmat.Phase1();
    auto yu = seismicpcmat.Phase2();
    auto y = seismicpcmat.Phase3();
    
    // we directly do final test
    // check finaly 
    CopyDataB2HD(fp32ptr1.h_finaly, reinterpret_cast<complex<float>*>(fp32ptr1.d_finaly), fp32ptr1.M);
    CopyDataB2HD(fp16ptr2.h_finaly, reinterpret_cast<complex<float>*>(fp16ptr2.d_finaly), fp16ptr2.M);

    auto finalyc1 = Matrix<complex<float>>(fp32ptr1.h_finaly, fp32ptr1.M, 1);
    auto finalyc2 = Matrix<complex<float>>(fp16ptr2.h_finaly, fp16ptr2.M, 1);
    auto finalyc = finalyc1 + finalyc2;
    cout << " Two pointers final y " << finalyc.allclose(y) << endl;

    fp32ptr1.FreeData();
    fp16ptr2.FreeData();

}



BENCHMARK_DEFINE_F(SeismicFixture, CUDA_C_8IBenchmark)(benchmark::State& state)
{
    size_t M ,N, K;
    M = 2046;
    N = 1;
    K = 256;

    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int32_t *i32Aptr, *i32Bptr;
    complex<float> *refCptr; 
    int32_t *refi32Cptr;
    
    int8_t *Aptr, *Bptr; 
    complex<float> *Cptr; // host ptr
    int8_t *d_Aptr, *d_Bptr; 
    cuComplex *d_Cptr; // device ptr

    // ref ptr
    GetHostMemory(&i32Aptr, M * K * 2);
    GetHostMemory(&i32Bptr, K * N * 2);
    GetHostMemory(&refCptr, M * N);
    GetHostMemory(&refi32Cptr, M * N);

    // data ptr
    GetHostMemory(&Aptr, M * K * 2);
    GetHostMemory(&Bptr, K * N * 2);
    GetHostMemory(&Cptr, M * N);
    // data ptr on gpu
    GetDeviceMemory(&d_Aptr, M * K * 2);
    GetDeviceMemory(&d_Bptr, K * N * 2);
    GetDeviceMemory(&d_Cptr, M * N);
    cuComplex alpha, beta;
    alpha.x = 1.0; alpha.y = 0.0; beta.x = 0.0; beta.y = 0.0;

    // init ref ptr
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            i32Aptr[i*2*N+j*2] = rand() % 126;
            i32Aptr[i*2*N+j*2+1] = rand() % 126;
            if(i == 0){
                i32Bptr[2*j] = rand() % 126;
                i32Bptr[2*j+1] = rand() % 126;
            }
        }
    }
    // gemv
    for(int i=0; i<M; i++){
        int32_t rpart = 0;
        int32_t ipart = 0;
        for(int j=0; j<N; j++){
            int32_t r1 = i32Aptr[2*i + j * 2 * M];
            int32_t i1 = i32Aptr[2*i + 1 + j * 2 * M];
            int32_t r2 = i32Bptr[2*j];
            int32_t i2 = i32Bptr[2*j + 1];
            rpart += r1 * r2 - i1 * i2;
            ipart += r1 * i2 + i1 * r2;
        }
        refCptr[i] = complex<float>( (float)rpart , (float)ipart );
    }
    auto ccheck = Matrix<complex<float>>(refCptr, M, 1);
    // cout << ccheck.Block({0,10}) << endl;

    // convert to int8
    for(int i=0; i< M * K * 2; i++){
        Aptr[i] = (int8_t) i32Aptr[i];
    }
    for(int i=0; i< K * N * 2; i++){
        Bptr[i] = (int8_t) i32Bptr[i];
    }
    CopyDataB2HD(d_Aptr, Aptr, M * K * 2);
    CopyDataB2HD(d_Bptr, Bptr, K * N * 2);

    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    for(auto st : state){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        CUBLASCHECK(cublascgemmex(cublashandleptr[0], CUBLAS_OP_N, CUBLAS_OP_N, 
        M, N, K, &alpha, d_Aptr, CUDA_C_8I, M, d_Bptr, CUDA_C_8I, K, &beta, 
        d_Cptr, CUDA_C_32F, M));
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
    }

    for(int i=0; i<streamsize; i++) cudaStreamDestroy(streamptr[i]);
    for(int i=0; i<streamsize; i++) cublasDestroy_v2(cublashandleptr[i]);
    delete[] streamptr;
    delete[] cublashandleptr;
    double bytes = 2 * sizeof(int8_t) * (M * K + M + K) * (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
    cudaDeviceSynchronize();

    CopyDataB2HD(Cptr, reinterpret_cast<complex<float>*>(d_Cptr), M);
    auto finalcheck = Matrix<complex<float>>(Cptr, M, 1);
    // cout << finalcheck.Block({0,10}) << endl;
    cout << finalcheck.allclose(ccheck) << endl;
    
}





BENCHMARK_DEFINE_F(SeismicFixture, CUDA_R_8IBenchmark)(benchmark::State& state)
{
    size_t M ,N, K;
    M = 9800;
    N = 1;
    K = 9800;

    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int32_t *i32Aptr, *i32Bptr;
    int32_t *refCptr;
    
    int8_t *Aptr, *Bptr; 
    int32_t *Cptr;
    int8_t *d_Aptr, *d_Bptr; 
    int32_t *d_Cptr;

    // ref ptr
    GetHostMemory(&i32Aptr, M * K);
    GetHostMemory(&i32Bptr, K * N);
    GetHostMemory(&refCptr, M * N);
    
    // data ptr
    GetHostMemory(&Aptr, M * K);
    GetHostMemory(&Bptr, K * N);
    GetHostMemory(&Cptr, M * N);

    // data ptr on gpu
    GetDeviceMemory(&d_Aptr, M * K);
    GetDeviceMemory(&d_Bptr, K * N);
    GetDeviceMemory(&d_Cptr, M * N);

    // init ref ptr
    for(int i=0; i<M; i++){
        for(int j=0; j<K; j++){
            i32Aptr[i*K+j] = (int)1.0;//rand() % 126;
            if(j == 0){
                i32Bptr[i] = (int)1.0;//rand() % 126;
            }
        }
    }
    auto Acheck = Matrix<int>(i32Aptr, M , K);
    // gemv
    for(int i=0; i<M; i++){
        int res = 0;
        for(int j=0; j<K; j++){
            res += i32Aptr[i + j * M] * i32Bptr[j];
        }
        refCptr[i] = res;
    }

    auto ccheck = Matrix<int>(refCptr, M, 1);
    cout << ccheck.Block({0,10}) << endl;

    // convert to int8
    for(int i=0; i< M * K; i++){
        Aptr[i] = (int8_t) i32Aptr[i];
    }
    for(int i=0; i< K * N; i++){
        Bptr[i] = (int8_t) i32Bptr[i];
    }
    CopyDataB2HD(d_Aptr, Aptr, M * K);
    CopyDataB2HD(d_Bptr, Bptr, K * N);

    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    int alpha = 1, beta = 0;
    for(auto st : state){
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        CUBLASCHECK(cublasGemmEx(cublashandleptr[0], CUBLAS_OP_N, CUBLAS_OP_N, 
        M, N, K, &alpha, d_Aptr, CUDA_R_8I, M, d_Bptr, CUDA_R_8I, K, &beta, 
        d_Cptr, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
    }

    for(int i=0; i<streamsize; i++) cudaStreamDestroy(streamptr[i]);
    for(int i=0; i<streamsize; i++) cublasDestroy_v2(cublashandleptr[i]);
    delete[] streamptr;
    delete[] cublashandleptr;
    double bytes = sizeof(int8_t) * (M * K + M + K) * (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
    cudaDeviceSynchronize();

    CopyDataB2HD(Cptr, d_Cptr, M);
    auto finalcheck = Matrix<int>(Cptr, M, 1);
    // cout << finalcheck.Block({0,10}) << endl;
    cout << finalcheck.allclose(ccheck) << endl;
    
}





BENCHMARK_DEFINE_F(SeismicFixture, CUDACGEMVBenchmark)(benchmark::State& state)
{
    // configuration
    size_t m = state.range(0);
    size_t k = state.range(1);
    size_t n = 1;
    size_t f_sz = sizeof(float);
    size_t c_sz = sizeof(cuComplex);
    complex<float> *hA, *hx, *hy;
    hA = new complex<float>[m * k];
    hx = new complex<float>[k];
    hy = new complex<float>[m];
    complex<float> val = complex<float>(1.0, 0.0);
    Init(hA, m * k, val);
    Init(hx, k, val);
    Init(hy, m, val);
    cuComplex *A, *x, *y;
    CUDACHECK(cudaMalloc(&A, m * k * c_sz));
    CUDACHECK(cudaMalloc(&x, k * c_sz));
    CUDACHECK(cudaMalloc(&y, m * c_sz));
    CUDACHECK(cudaMemcpy(A, hA, m * k * c_sz, cudaMemcpyDefault));
    CUDACHECK(cudaMemcpy(x, hx, k * c_sz, cudaMemcpyDefault));
    CUDACHECK(cudaMemcpy(y, hy, m * c_sz, cudaMemcpyDefault));
    // timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // create cublas handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));
    cuComplex alpha, beta;
    alpha.x = 1.0;
    alpha.y = 0.0;
    beta.x = 0.0;
    beta.y = 0.0;
    vector<double> rawtime;
    for(auto st : state){
        cudaEventRecord(start);
        cudaDeviceSynchronize();
        CUBLASCHECK(cublasgemv(handle, CUBLAS_OP_N, m, k, &alpha, A, m, x, 1, &beta, y, 1));
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        float milliseconds = 0;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        state.SetIterationTime(milliseconds*1e-3);
        rawtime.push_back(milliseconds*1e-3);
    }
    double bytes = sizeof(complex<float>) * (m * k + m + k) * (double)state.iterations();
    state.counters["BandWidth"] =
    Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1000);
    cublasDestroy(handle);
    CUDACHECK(cudaFree(A));
    CUDACHECK(cudaFree(x));
    CUDACHECK(cudaFree(y));
    delete[] hA;
    delete[] hx;
    delete[] hy;
}



// BENCHMARK_REGISTER_F(SeismicFixture, SinglePtrBenchmark) // 130GB,1.7ms on v100
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(50)
// ->UseManualTime()
// ->Repetitions(1);

// BENCHMARK_REGISTER_F(SeismicFixture, SinglePtrStreamBenchmark)
// ->Unit(benchmark::kMicrosecond)
// // ->Iterations(500)
// ->UseManualTime()
// ->Repetitions(1);

// BENCHMARK_REGISTER_F(SeismicFixture, TwoPtrStreamBenchmark)
// ->Unit(benchmark::kMicrosecond)
// // ->Iterations(50) 
// ->UseManualTime()
// ->Repetitions(1);


// BENCHMARK_REGISTER_F(SeismicFixture, SinglePtrStreamCUDAGraphBenchmark)
// ->Unit(benchmark::kMicrosecond)
// // ->Iterations(1)
// ->UseManualTime()
// ->Repetitions(1);


// BENCHMARK_REGISTER_F(SeismicFixture, SingleComplexPtrStreamCUDAGraphBenchmark)
// ->Unit(benchmark::kMicrosecond)
// // ->Iterations(1)
// ->UseManualTime()
// ->Repetitions(1);

BENCHMARK_REGISTER_F(SeismicFixture, SinglePtrFP16StreamCUDAGraphBenchmark)
->Unit(benchmark::kMicrosecond)
// ->Iterations(1)
->UseManualTime()
->Repetitions(1);

// BENCHMARK_REGISTER_F(SeismicFixture, TwoPtrStreamCUDAGraphBenchmark)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(1) 
// ->UseManualTime()
// ->Repetitions(1);


// BENCHMARK_REGISTER_F(SeismicFixture, CUDA_C_8IBenchmark)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(50)
// ->UseManualTime()
// ->Repetitions(1);



// BENCHMARK_REGISTER_F(SeismicFixture, FP32FP16CUDAGraphBenchmark)
// ->Unit(benchmark::kMicrosecond)
// // ->Iterations(1)
// ->UseManualTime()
// ->Repetitions(1);


// BENCHMARK_REGISTER_F(SeismicFixture, CUDA_R_8IBenchmark)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(5000)
// ->UseManualTime()
// ->Repetitions(1);

// BENCHMARK_REGISTER_F(SeismicFixture, CUDACGEMVBenchmark)
// ->Unit(benchmark::kMicrosecond)
// ->Args({1024, 256})
// // ->Iterations(5000)
// ->UseManualTime()
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