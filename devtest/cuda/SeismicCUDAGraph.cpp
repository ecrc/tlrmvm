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

Matrix<complex<float>> mergerealimag(half * real, half *imag, size_t row, size_t col){

    Matrix<complex<float>> ret(row, col);
    for(int i=0; i<col; i++){
        for(int j=0; j<row; j++){
            ret.SetElem(j,i,complex<float>((half)real[i*row+j], (half)imag[i*row+j]));
        }
    }
    return ret;
}

void ConvertFP16ToFP32(vector<float> &fp32vec, half* fp16vec, size_t length){
    fp32vec.resize(length, 0);
    for(int i=0; i<length; i++){
        fp32vec[i] = (float)fp16vec[i];
    }
}
void ConvertINT8ToINT32(vector<int> &intvec, int8_t* int8vec, size_t length){
    intvec.resize(length, 0);
    for(int i=0; i<length; i++){
        intvec[i] = (int)int8vec[i];
    }
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
        for(auto x : freqlist){
            int *DataR;
            ReadSeismicBinary(datafolder, &DataR, Mtglobal * Ntglobal, acc, nb, x);
            Matrix<int> Rmat = Matrix<int>(DataR, Mtglobal, Ntglobal);
            Rmats[x] = Rmat; // just generate Rmat in setup
            getmaskmat(x);
            delete[] DataR;
        }
    }

    void TearDown(const ::benchmark::State& state) {}

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


BENCHMARK_DEFINE_F(SeismicFixture, Phase1_Complex32Test)(benchmark::State& state){
    for(auto st : state){

    }
    ComplexPtr complexptr(paddingM, paddingN, nb, Rmats[freqlist[0]], FP32Rmats[freqlist[0]]);
    complexptr.InitData(datafolder, acc, freqlist[0], originN);
    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    complexptr.CopyData2GPU();
    // phase 1
    cuComplex alpha;
    cuComplex beta;
    alpha.x = 1.0; alpha.y = 0.0;
    beta.x = 0.0; beta.y = 0.0;
    for(int i=0; i<39; i++){
        if(complexptr.AvMs[i] != 0){
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
            complexptr.AvMs[i], complexptr.AvNs[i], complexptr.AvKs[i], 
            &alpha, complexptr.d_Avbp[i], complexptr.AvMs[i], 
            complexptr.d_xbp[i], complexptr.AvKs[i], 
            &beta, complexptr.d_yvbp[i], complexptr.AvMs[i]));
        }
    }
    cudaDeviceSynchronize();
    // missing the phase 2
    SeismicPCMatrix seismicpcmat(datafolder, 
    acc, nb, freqlist[0], originM, originN);
    Matrix<complex<float>> xmat = seismicpcmat.GetX();
    size_t coloffset = 0;

    // check for final yv output
    CopyDataB2HD(complexptr.h_yv, complexptr.d_yv, complexptr.complexgranksum);
    Matrix<complex<float>> densemat = seismicpcmat.GetDense();
    Matrix<complex<float>> yv = seismicpcmat.Phase1();
    auto cyv = Matrix<complex<float>>(complexptr.h_yv, complexptr.complexgranksum, 1);
    cout << "yv output " << cyv.allclose(yv) << endl;
    complexptr.FreeData();
}


BENCHMARK_DEFINE_F(SeismicFixture, Phase1_OneFP32Test)(benchmark::State& state){
    for(auto st : state){

    }
    Float32Ptr fp32ptr(paddingM, paddingN, nb, Rmats[freqlist[0]], FP32Rmats[freqlist[0]]);
    fp32ptr.InitData(datafolder, acc, freqlist[0], originN);
    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    fp32ptr.CopyData2GPU();
    // phase 1
    float alpha = 1.0;
    float beta = 0.0;
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
    merge_float_2floatout_realimag(fp32ptr.d_yv[0], fp32ptr.d_yv[1], 
    fp32ptr.d_fp32colrank, fp32ptr.Ntg,
    fp32ptr.d_yvout[0], fp32ptr.d_yvout[1], 
    fp32ptr.fp32granksum, streamptr[0]);
    cudaDeviceSynchronize();
    // // missing the phase 2
    SeismicPCMatrix seismicpcmat(datafolder, 
    acc, nb, freqlist[0], originM, originN);
    Matrix<complex<float>> xmat = seismicpcmat.GetX();
    // size_t coloffset = 0;

    // // check for middle yv output
    // for(int ntgi=0; ntgi<fp32ptr.Ntg; ntgi++){

    //     size_t curele = fp32ptr.colsum[ntgi];
    //     CopyDataB2HD(fp32ptr.h_yvout[0], fp32ptr.d_yvbp[ntgi][0], curele * 2);
    //     CopyDataB2HD(fp32ptr.h_yvout[1], fp32ptr.d_yvbp[ntgi][1], curele * 2);

    //     auto cyrr_ri = Matrix<float>(fp32ptr.h_yvout[0], curele, 2);
    //     auto cyir_ii = Matrix<float>(fp32ptr.h_yvout[1], curele, 2);

    //     // coloffset += curele * 2;
    //     size_t offset = 0;
    //     for(int i=0; i<fp32ptr.Mtg; i++){
    //         auto vtile = seismicpcmat.GetVTile(i,ntgi);
    //         auto xtile = seismicpcmat.GetXtile(ntgi);
    //         auto yrr = seismicpcmat.GetReal(vtile) * seismicpcmat.GetReal(xtile);
    //         auto yri = seismicpcmat.GetReal(vtile) * seismicpcmat.GetImag(xtile);
    //         auto yir = seismicpcmat.GetImag(vtile) * seismicpcmat.GetReal(xtile);
    //         auto yii = seismicpcmat.GetImag(vtile) * seismicpcmat.GetImag(xtile);
    //         auto v1 = cyrr_ri.Block({offset,offset + yrr.Row()}, {0,1}).allclose(yrr);
    //         auto v2 = cyrr_ri.Block({offset,offset + yrr.Row()}, {1,2}).allclose(yri);
    //         auto v3 = cyir_ii.Block({offset,offset + yir.Row()}, {0,1}).allclose(yir);
    //         auto v4 = cyir_ii.Block({offset,offset + yir.Row()}, {1,2}).allclose(yii);
    //         printf("%d %d %f %f %f %f \n", i, ntgi, v1, v2, v3,v4);
    //         offset += yrr.Row();
    //     }
    // }

    // check for final yv output
    CopyDataB2HD(fp32ptr.h_yvout[0], fp32ptr.d_yvout[0], fp32ptr.fp32granksum);
    CopyDataB2HD(fp32ptr.h_yvout[1], fp32ptr.d_yvout[1], fp32ptr.fp32granksum);
    float *rp = fp32ptr.h_yvout[0];
    float *ip = fp32ptr.h_yvout[1];
    Matrix<complex<float>> densemat = seismicpcmat.GetDense();
    Matrix<complex<float>> yv = seismicpcmat.Phase1();
    auto cyv = mergerealimag(fp32ptr.h_yvout[0], fp32ptr.h_yvout[1], fp32ptr.fp32granksum, 1);
    cout << "yv output " << cyv.allclose(yv) << endl;
    fp32ptr.FreeData();
}



BENCHMARK_DEFINE_F(SeismicFixture, Phase1_OneFP16Test)(benchmark::State& state){
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
    // we want to split this rmats
    
    Float16Ptr fp16ptr(paddingM, paddingN, nb, Rmats[freqlist[0]], Rmat3);
    fp16ptr.InitData(datafolder, acc, freqlist[0], originN);

    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    fp16ptr.CopyData2GPU();
    // phase 1
    half alpha = 1.0;
    half beta = 0.0;
    for(int i=0; i<39; i++){
        if(fp16ptr.AvMs[i] == 0) continue;
        CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
            fp16ptr.AvMs[i], fp16ptr.AvNs[i], fp16ptr.AvKs[i], 
            &alpha, fp16ptr.d_Avbp[i][0], fp16ptr.AvMs[i], 
            fp16ptr.d_xbp[i], fp16ptr.AvKs[i], 
            &beta, fp16ptr.d_yvbp[i][0], fp16ptr.AvMs[i]);
        );
        CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
            fp16ptr.AvMs[i], fp16ptr.AvNs[i], fp16ptr.AvKs[i], 
            &alpha, fp16ptr.d_Avbp[i][1], fp16ptr.AvMs[i], 
            fp16ptr.d_xbp[i], fp16ptr.AvKs[i], 
            &beta, fp16ptr.d_yvbp[i][1], fp16ptr.AvMs[i]);
        );
    }
    merge_half_2halfout_realimag(fp16ptr.d_yv[0], fp16ptr.d_yv[1], 
    fp16ptr.d_fp16colrank, fp16ptr.Ntg,
    fp16ptr.d_yvout[0], fp16ptr.d_yvout[1], 
    fp16ptr.fp16granksum, streamptr[0]);
    cudaDeviceSynchronize();
    // // missing the phase 2
    SeismicPCMatrix seismicpcmat(datafolder, 
    acc, nb, freqlist[0], originM, originN);
    Matrix<complex<float>> xmat = seismicpcmat.GetX();

    // check for middle yv output
    for(int ntgi=0; ntgi<fp16ptr.Ntg; ntgi++){

        size_t curele = fp16ptr.colsum[ntgi];
        CopyDataB2HD(fp16ptr.h_yvout[0], fp16ptr.d_yvbp[ntgi][0], curele * 2);
        CopyDataB2HD(fp16ptr.h_yvout[1], fp16ptr.d_yvbp[ntgi][1], curele * 2);
        vector<float> hyvreal;
        vector<float> hyvimag;
        ConvertFP16ToFP32(hyvreal, fp16ptr.h_yvout[0], curele * 2);
        ConvertFP16ToFP32(hyvimag, fp16ptr.h_yvout[1], curele * 2);
        auto cyrr_ri = Matrix<float>(hyvreal, curele, 2);
        auto cyir_ii = Matrix<float>(hyvimag, curele, 2);

        size_t offset = 0;
        for(int i=0; i<fp16ptr.Mtg; i++){
            auto vtile = seismicpcmat.GetVTile(i,ntgi);
            auto xtile = seismicpcmat.GetXtile(ntgi);
            auto yrr = seismicpcmat.GetReal(vtile) * seismicpcmat.GetReal(xtile);
            auto yri = seismicpcmat.GetReal(vtile) * seismicpcmat.GetImag(xtile);
            auto yir = seismicpcmat.GetImag(vtile) * seismicpcmat.GetReal(xtile);
            auto yii = seismicpcmat.GetImag(vtile) * seismicpcmat.GetImag(xtile);
            auto v1 = cyrr_ri.Block({offset,offset + yrr.Row()}, {0,1}).allclose(yrr);
            auto v2 = cyrr_ri.Block({offset,offset + yrr.Row()}, {1,2}).allclose(yri);
            auto v3 = cyir_ii.Block({offset,offset + yir.Row()}, {0,1}).allclose(yir);
            auto v4 = cyir_ii.Block({offset,offset + yir.Row()}, {1,2}).allclose(yii);
            printf("%d %d %f %f %f %f \n", i, ntgi, v1, v2, v3,v4);
            offset += yrr.Row();
        }
    }

    // check for final yv output
    CopyDataB2HD(fp16ptr.h_yvout[0], fp16ptr.d_yvout[0], fp16ptr.fp16granksum);
    CopyDataB2HD(fp16ptr.h_yvout[1], fp16ptr.d_yvout[1], fp16ptr.fp16granksum);
    Matrix<complex<float>> densemat = seismicpcmat.GetDense();
    Matrix<complex<float>> yv = seismicpcmat.Phase1();
    auto cyv = mergerealimag(fp16ptr.h_yvout[0], fp16ptr.h_yvout[1], fp16ptr.fp16granksum, 1);
    cout << "yv output " << cyv.allclose(yv) << endl;
    fp16ptr.FreeData();
    for(auto st : state){}
}




BENCHMARK_DEFINE_F(SeismicFixture, Phase1_TwoFP32Test)(benchmark::State& state){
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
    cout << Rmat3.allclose(Rmats[freqlist[0]]) << endl;
    cout << Rmat1.Block({0,10},{0,10}) << endl;
    cout << Rmat2.Block({0,10},{0,10}) << endl;
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
    // phase 1
    float alpha = 1.0;
    float beta = 0.0;
    for(int i=0; i<39; i++){
        if(fp32ptr1.AvMs[i] != 0){
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp32ptr1.AvMs[i], fp32ptr1.AvNs[i], fp32ptr1.AvKs[i], 
                &alpha, fp32ptr1.d_Avbp[i][0], fp32ptr1.AvMs[i], 
                fp32ptr1.d_xbp[i], fp32ptr1.AvKs[i], 
                &beta, fp32ptr1.d_yvbp[i][0], fp32ptr1.AvMs[i]);
            );
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp32ptr1.AvMs[i], fp32ptr1.AvNs[i], fp32ptr1.AvKs[i], 
                &alpha, fp32ptr1.d_Avbp[i][1], fp32ptr1.AvMs[i], 
                fp32ptr1.d_xbp[i], fp32ptr1.AvKs[i], 
                &beta, fp32ptr1.d_yvbp[i][1], fp32ptr1.AvMs[i]);
            );
        }
        if(fp32ptr2.AvMs[i] != 0){
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp32ptr2.AvMs[i], fp32ptr2.AvNs[i], fp32ptr2.AvKs[i], 
                &alpha, fp32ptr2.d_Avbp[i][0], fp32ptr2.AvMs[i], 
                fp32ptr2.d_xbp[i], fp32ptr2.AvKs[i], 
                &beta, fp32ptr2.d_yvbp[i][0], fp32ptr2.AvMs[i]);
            );
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp32ptr2.AvMs[i], fp32ptr2.AvNs[i], fp32ptr2.AvKs[i], 
                &alpha, fp32ptr2.d_Avbp[i][1], fp32ptr2.AvMs[i], 
                fp32ptr2.d_xbp[i], fp32ptr2.AvKs[i], 
                &beta, fp32ptr2.d_yvbp[i][1], fp32ptr2.AvMs[i]);
            );
        }
    }
    merge_float_2floatout_realimag(fp32ptr1.d_yv[0], fp32ptr1.d_yv[1], 
    fp32ptr1.d_fp32colrank, fp32ptr1.Ntg,
    fp32ptr1.d_yvout[0], fp32ptr1.d_yvout[1], 
    fp32ptr1.fp32granksum, streamptr[0]);
    merge_float_2floatout_realimag(fp32ptr2.d_yv[0], fp32ptr2.d_yv[1], 
    fp32ptr2.d_fp32colrank, fp32ptr2.Ntg,
    fp32ptr2.d_yvout[0], fp32ptr2.d_yvout[1], 
    fp32ptr2.fp32granksum, streamptr[0]);
    cudaDeviceSynchronize();
    // init pc matrix
    SeismicPCMatrix seismicpcmat(datafolder, 
    acc, nb, freqlist[0], originM, originN);
    Matrix<complex<float>> xmat = seismicpcmat.GetX();

    // check for middle yv output
    // for(int ntgi=0; ntgi<fp32ptr2.Ntg; ntgi++){
        
    //     size_t curele1 = fp32ptr1.colsum[ntgi];
    //     CopyDataB2HD(fp32ptr1.h_yvout[0], fp32ptr1.d_yvbp[ntgi][0], curele1 * 2);
    //     CopyDataB2HD(fp32ptr1.h_yvout[1], fp32ptr1.d_yvbp[ntgi][1], curele1 * 2);
    //     auto cyrr_ri1 = Matrix<float>(fp32ptr1.h_yvout[0], curele1, 2);
    //     auto cyir_ii1 = Matrix<float>(fp32ptr1.h_yvout[1], curele1, 2);

    //     size_t curele2 = fp32ptr2.colsum[ntgi];
    //     CopyDataB2HD(fp32ptr2.h_yvout[0], fp32ptr2.d_yvbp[ntgi][0], curele2 * 2);
    //     CopyDataB2HD(fp32ptr2.h_yvout[1], fp32ptr2.d_yvbp[ntgi][1], curele2 * 2);
    //     auto cyrr_ri2 = Matrix<float>(fp32ptr2.h_yvout[0], curele2, 2);
    //     auto cyir_ii2 = Matrix<float>(fp32ptr2.h_yvout[1], curele2, 2);

    //     size_t offsetptr1 = 0;
    //     size_t offsetptr2 = 0;
    //     for(int i=0; i<fp32ptr2.Mtg; i++){

    //         auto vtile = seismicpcmat.GetVTile(i,ntgi);
    //         auto xtile = seismicpcmat.GetXtile(ntgi);
    //         auto yrr = seismicpcmat.GetReal(vtile) * seismicpcmat.GetReal(xtile);
    //         auto yri = seismicpcmat.GetReal(vtile) * seismicpcmat.GetImag(xtile);
    //         auto yir = seismicpcmat.GetImag(vtile) * seismicpcmat.GetReal(xtile);
    //         auto yii = seismicpcmat.GetImag(vtile) * seismicpcmat.GetImag(xtile);

    //         double v1,v2,v3,v4; 
    //         if(fp32ptr2.Maskmat.GetElem(i,ntgi) != 0){
                
    //             v1 = cyrr_ri2.Block({offsetptr2,offsetptr2 + yrr.Row()}, {0,1}).allclose(yrr);
    //             v2 = cyrr_ri2.Block({offsetptr2,offsetptr2 + yrr.Row()}, {1,2}).allclose(yri);
    //             v3 = cyir_ii2.Block({offsetptr2,offsetptr2 + yir.Row()}, {0,1}).allclose(yir);
    //             v4 = cyir_ii2.Block({offsetptr2,offsetptr2 + yir.Row()}, {1,2}).allclose(yii);
    //             offsetptr2 += yrr.Row();
    //         }else{
    //             v1 = cyrr_ri1.Block({offsetptr1,offsetptr1 + yrr.Row()}, {0,1}).allclose(yrr);
    //             v2 = cyrr_ri1.Block({offsetptr1,offsetptr1 + yrr.Row()}, {1,2}).allclose(yri);
    //             v3 = cyir_ii1.Block({offsetptr1,offsetptr1 + yir.Row()}, {0,1}).allclose(yir);
    //             v4 = cyir_ii1.Block({offsetptr1,offsetptr1 + yir.Row()}, {1,2}).allclose(yii);
    //             offsetptr1 += yrr.Row();
    //         }
    //         printf("%d %d %f %f %f %f \n", i, ntgi, v1, v2, v3,v4);
    //     }
    // }

    // check for final yv output
    CopyDataB2HD(fp32ptr2.h_yvout[0], fp32ptr2.d_yvout[0], fp32ptr2.fp32granksum);
    CopyDataB2HD(fp32ptr2.h_yvout[1], fp32ptr2.d_yvout[1], fp32ptr2.fp32granksum);
    auto cyv2 = mergerealimag(fp32ptr2.h_yvout[0], fp32ptr2.h_yvout[1], fp32ptr2.fp32granksum, 1);
    CopyDataB2HD(fp32ptr1.h_yvout[0], fp32ptr1.d_yvout[0], fp32ptr1.fp32granksum);
    CopyDataB2HD(fp32ptr1.h_yvout[1], fp32ptr1.d_yvout[1], fp32ptr1.fp32granksum);
    auto cyv1 = mergerealimag(fp32ptr1.h_yvout[0], fp32ptr1.h_yvout[1], fp32ptr1.fp32granksum, 1);
    Matrix<complex<float>> densemat = seismicpcmat.GetDense();
    Matrix<complex<float>> yv = seismicpcmat.Phase1();
    size_t offsetptr2 = 0;
    size_t offsetptr1 = 0;
    size_t offset = 0;
    vector<complex<float>> recovervector;
    for(int i=0; i<fp32ptr2.Ntg; i++){
        for(int j=0; j<fp32ptr2.Mtg; j++){
            auto currank = fp32ptr2.OrgRmat.GetElem(j,i);
            if(fp32ptr2.Maskmat.GetElem(j,i) != 0){
                auto cp2 = cyv2.Block({offsetptr2,offsetptr2+currank});
                // auto cp2 = yv.Block({offset, offset+currank});
                // cout << "yv output " << cp1.allclose(cp2) << endl;
                auto dvec = cp2.Datavec();
                for(auto x : dvec) recovervector.push_back(x);
                offsetptr2 += currank;
            }else{
                auto cp1 = cyv1.Block({offsetptr1,offsetptr1+currank});
                // auto cp2 = yv.Block({offset, offset+currank});
                // cout << "yv output " << cp1.allclose(cp2) << endl;
                auto dvec = cp1.Datavec();
                for(auto x : dvec) recovervector.push_back(x);
                offsetptr1 += currank;
            }
            offset += currank;
        }
    }
    auto res = Matrix<complex<float>>(recovervector, recovervector.size(), 1);
    cout << res.allclose(yv) << endl;
}



BENCHMARK_DEFINE_F(SeismicFixture, Phase1_TwoFP16Test)(benchmark::State& state){
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
    cout << Rmat3.allclose(Rmats[freqlist[0]]) << endl;
    cout << Rmat1.Block({0,10},{0,10}) << endl;
    cout << Rmat2.Block({0,10},{0,10}) << endl;
    // we want to split this rmats
    
    Float16Ptr fp16ptr1(paddingM, paddingN, nb, Rmats[freqlist[0]], Rmat1);
    Float16Ptr fp16ptr2(paddingM, paddingN, nb, Rmats[freqlist[0]], Rmat2);
    fp16ptr1.InitData(datafolder, acc, freqlist[0], originN);
    fp16ptr2.InitData(datafolder, acc, freqlist[0], originN);

    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    fp16ptr1.CopyData2GPU();
    fp16ptr2.CopyData2GPU();
    // phase 1
    half alpha = (half)1.0;
    half beta = (half)0.0;
    for(int i=0; i<39; i++){
        if(fp16ptr1.AvMs[i] != 0){
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp16ptr1.AvMs[i], fp16ptr1.AvNs[i], fp16ptr1.AvKs[i], 
                &alpha, fp16ptr1.d_Avbp[i][0], fp16ptr1.AvMs[i], 
                fp16ptr1.d_xbp[i], fp16ptr1.AvKs[i], 
                &beta, fp16ptr1.d_yvbp[i][0], fp16ptr1.AvMs[i]);
            );
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp16ptr1.AvMs[i], fp16ptr1.AvNs[i], fp16ptr1.AvKs[i], 
                &alpha, fp16ptr1.d_Avbp[i][1], fp16ptr1.AvMs[i], 
                fp16ptr1.d_xbp[i], fp16ptr1.AvKs[i], 
                &beta, fp16ptr1.d_yvbp[i][1], fp16ptr1.AvMs[i]);
            );
        }
        if(fp16ptr2.AvMs[i] != 0){
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp16ptr2.AvMs[i], fp16ptr2.AvNs[i], fp16ptr2.AvKs[i], 
                &alpha, fp16ptr2.d_Avbp[i][0], fp16ptr2.AvMs[i], 
                fp16ptr2.d_xbp[i], fp16ptr2.AvKs[i], 
                &beta, fp16ptr2.d_yvbp[i][0], fp16ptr2.AvMs[i]);
            );
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp16ptr2.AvMs[i], fp16ptr2.AvNs[i], fp16ptr2.AvKs[i], 
                &alpha, fp16ptr2.d_Avbp[i][1], fp16ptr2.AvMs[i], 
                fp16ptr2.d_xbp[i], fp16ptr2.AvKs[i], 
                &beta, fp16ptr2.d_yvbp[i][1], fp16ptr2.AvMs[i]);
            );
        }
    }
    merge_half_2halfout_realimag(fp16ptr1.d_yv[0], fp16ptr1.d_yv[1], 
    fp16ptr1.d_fp16colrank, fp16ptr1.Ntg,
    fp16ptr1.d_yvout[0], fp16ptr1.d_yvout[1], 
    fp16ptr1.fp16granksum, streamptr[0]);
    merge_half_2halfout_realimag(fp16ptr2.d_yv[0], fp16ptr2.d_yv[1], 
    fp16ptr2.d_fp16colrank, fp16ptr2.Ntg,
    fp16ptr2.d_yvout[0], fp16ptr2.d_yvout[1], 
    fp16ptr2.fp16granksum, streamptr[0]);
    cudaDeviceSynchronize();
    // init pc matrix
    SeismicPCMatrix seismicpcmat(datafolder, 
    acc, nb, freqlist[0], originM, originN);
    Matrix<complex<float>> xmat = seismicpcmat.GetX();

    // check for middle yv output
    // for(int ntgi=0; ntgi<fp16ptr2.Ntg; ntgi++){
        
    //     size_t curele1 = fp16ptr1.colsum[ntgi];
    //     CopyDataB2HD(fp16ptr1.h_yvout[0], fp16ptr1.d_yvbp[ntgi][0], curele1 * 2);
    //     CopyDataB2HD(fp16ptr1.h_yvout[1], fp16ptr1.d_yvbp[ntgi][1], curele1 * 2);
    //     vector<float> rvec, ivec;
    //     ConvertFP16ToFP32(rvec, fp16ptr1.h_yvout[0], curele1 * 2);
    //     ConvertFP16ToFP32(ivec, fp16ptr1.h_yvout[1], curele1 * 2);
    //     auto cyrr_ri1 = Matrix<float>(rvec, curele1, 2);
    //     auto cyir_ii1 = Matrix<float>(ivec, curele1, 2);

    //     size_t curele2 = fp16ptr2.colsum[ntgi];
    //     CopyDataB2HD(fp16ptr2.h_yvout[0], fp16ptr2.d_yvbp[ntgi][0], curele2 * 2);
    //     CopyDataB2HD(fp16ptr2.h_yvout[1], fp16ptr2.d_yvbp[ntgi][1], curele2 * 2);
    //     ConvertFP16ToFP32(rvec, fp16ptr2.h_yvout[0], curele1 * 2);
    //     ConvertFP16ToFP32(ivec, fp16ptr2.h_yvout[1], curele1 * 2);
    //     auto cyrr_ri2 = Matrix<float>(rvec, curele2, 2);
    //     auto cyir_ii2 = Matrix<float>(ivec, curele2, 2);

    //     size_t offsetptr1 = 0;
    //     size_t offsetptr2 = 0;
    //     for(int i=0; i<fp16ptr2.Mtg; i++){

    //         auto vtile = seismicpcmat.GetVTile(i,ntgi);
    //         auto xtile = seismicpcmat.GetXtile(ntgi);
    //         auto yrr = seismicpcmat.GetReal(vtile) * seismicpcmat.GetReal(xtile);
    //         auto yri = seismicpcmat.GetReal(vtile) * seismicpcmat.GetImag(xtile);
    //         auto yir = seismicpcmat.GetImag(vtile) * seismicpcmat.GetReal(xtile);
    //         auto yii = seismicpcmat.GetImag(vtile) * seismicpcmat.GetImag(xtile);

    //         double v1,v2,v3,v4; 
    //         if(fp16ptr2.Maskmat.GetElem(i,ntgi) != 0){
                
    //             v1 = cyrr_ri2.Block({offsetptr2,offsetptr2 + yrr.Row()}, {0,1}).allclose(yrr);
    //             v2 = cyrr_ri2.Block({offsetptr2,offsetptr2 + yrr.Row()}, {1,2}).allclose(yri);
    //             v3 = cyir_ii2.Block({offsetptr2,offsetptr2 + yir.Row()}, {0,1}).allclose(yir);
    //             v4 = cyir_ii2.Block({offsetptr2,offsetptr2 + yir.Row()}, {1,2}).allclose(yii);
    //             offsetptr2 += yrr.Row();
    //         }else{
    //             v1 = cyrr_ri1.Block({offsetptr1,offsetptr1 + yrr.Row()}, {0,1}).allclose(yrr);
    //             v2 = cyrr_ri1.Block({offsetptr1,offsetptr1 + yrr.Row()}, {1,2}).allclose(yri);
    //             v3 = cyir_ii1.Block({offsetptr1,offsetptr1 + yir.Row()}, {0,1}).allclose(yir);
    //             v4 = cyir_ii1.Block({offsetptr1,offsetptr1 + yir.Row()}, {1,2}).allclose(yii);
    //             offsetptr1 += yrr.Row();
    //         }
    //         printf("%d %d %f %f %f %f \n", i, ntgi, v1, v2, v3,v4);
    //     }
    // }

    // check for final yv output
    CopyDataB2HD(fp16ptr2.h_yvout[0], fp16ptr2.d_yvout[0], fp16ptr2.fp16granksum);
    CopyDataB2HD(fp16ptr2.h_yvout[1], fp16ptr2.d_yvout[1], fp16ptr2.fp16granksum);
    auto cyv2 = mergerealimag(fp16ptr2.h_yvout[0], fp16ptr2.h_yvout[1], fp16ptr2.fp16granksum, 1);
    CopyDataB2HD(fp16ptr1.h_yvout[0], fp16ptr1.d_yvout[0], fp16ptr1.fp16granksum);
    CopyDataB2HD(fp16ptr1.h_yvout[1], fp16ptr1.d_yvout[1], fp16ptr1.fp16granksum);
    auto cyv1 = mergerealimag(fp16ptr1.h_yvout[0], fp16ptr1.h_yvout[1], fp16ptr1.fp16granksum, 1);
    Matrix<complex<float>> densemat = seismicpcmat.GetDense();
    Matrix<complex<float>> yv = seismicpcmat.Phase1();
    size_t offsetptr2 = 0;
    size_t offsetptr1 = 0;
    size_t offset = 0;
    vector<complex<float>> recovervector;
    for(int i=0; i<fp16ptr2.Ntg; i++){
        for(int j=0; j<fp16ptr2.Mtg; j++){
            auto currank = fp16ptr2.OrgRmat.GetElem(j,i);
            if(fp16ptr2.Maskmat.GetElem(j,i) != 0){
                auto cp2 = cyv2.Block({offsetptr2,offsetptr2+currank});
                // auto cp2 = yv.Block({offset, offset+currank});
                // cout << "yv output " << cp1.allclose(cp2) << endl;
                auto dvec = cp2.Datavec();
                for(auto x : dvec) recovervector.push_back(x);
                offsetptr2 += currank;
            }else{
                auto cp1 = cyv1.Block({offsetptr1,offsetptr1+currank});
                // auto cp2 = yv.Block({offset, offset+currank});
                // cout << "yv output " << cp1.allclose(cp2) << endl;
                auto dvec = cp1.Datavec();
                for(auto x : dvec) recovervector.push_back(x);
                offsetptr1 += currank;
            }
            offset += currank;
        }
    }
    auto res = Matrix<complex<float>>(recovervector, recovervector.size(), 1);
    cout << res.allclose(yv) << endl;
}



BENCHMARK_DEFINE_F(SeismicFixture, Phase2_OneFP32Test)(benchmark::State& state){
    Float32Ptr fp32ptr(paddingM, paddingN, nb, Rmats[freqlist[0]], FP32Rmats[freqlist[0]]);
    fp32ptr.InitData(datafolder, acc, freqlist[0], originN);

    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    fp32ptr.CopyData2GPU();
    // phase 1
    float alpha = 1.0;
    float beta = 0.0;
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
    // merge_float_2floatout_realimag(fp32ptr.d_yv[0], fp32ptr.d_yv[1], 
    // fp32ptr.d_fp32colrank, fp32ptr.Ntg,
    // fp32ptr.d_yvout[0], fp32ptr.d_yvout[1], 
    // fp32ptr.fp32granksum, streamptr[0]);
    phase2(fp32ptr.d_yv[0], fp32ptr.d_yv[1], 
    fp32ptr.d_fp32colrank, fp32ptr.Ntg,fp32ptr.d_phase2mapping, 
    fp32ptr.d_yu, 
    fp32ptr.fp32granksum, streamptr[0]);
    cudaDeviceSynchronize();
    // missing the phase 2
    SeismicPCMatrix seismicpcmat(datafolder, 
    acc, nb, freqlist[0], originM, originN);
    Matrix<complex<float>> xmat = seismicpcmat.GetX();
    // size_t coloffset = 0;

    // check for final yv output
    CopyDataB2HD(fp32ptr.h_yu, fp32ptr.d_yu, 2*fp32ptr.fp32granksum);
    vector<complex<float>> yuvec;
    size_t offset = 0;
    // recover yu
    for(int i=0; i<fp32ptr.rowsum.size(); i++){
        int curlda = fp32ptr.rowsum[i];
        for(int k=0; k<curlda; k++){
            auto tmp = complex<float>( fp32ptr.h_yu[offset + k], fp32ptr.h_yu[offset + curlda + k] );
            yuvec.push_back(tmp);
        }
        offset += 2 * curlda;
    }
    
    Matrix<complex<float>> densemat = seismicpcmat.GetDense();
    Matrix<complex<float>> yv = seismicpcmat.Phase1();
    Matrix<complex<float>> yu = seismicpcmat.Phase2();
    auto cyu = Matrix<complex<float>>(yuvec, yuvec.size(), 1);
    cyu.Tofile("cyu.bin");
    cout << "yu output " << cyu.allclose(yu) << endl;

}

BENCHMARK_DEFINE_F(SeismicFixture, Phase2_Complex32Test)(benchmark::State& state){
    for(auto st : state){

    }
    ComplexPtr complexptr(paddingM, paddingN, nb, Rmats[freqlist[0]], FP32Rmats[freqlist[0]]);
    complexptr.InitData(datafolder, acc, freqlist[0], originN);
    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    complexptr.CopyData2GPU();
    // phase 1
    cuComplex alpha;
    cuComplex beta;
    alpha.x = 1.0; alpha.y = 0.0;
    beta.x = 0.0; beta.y = 0.0;
    for(int i=0; i<39; i++){
        if(complexptr.AvMs[i] != 0){
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
            complexptr.AvMs[i], complexptr.AvNs[i], complexptr.AvKs[i], 
            &alpha, complexptr.d_Avbp[i], complexptr.AvMs[i], 
            complexptr.d_xbp[i], complexptr.AvKs[i], 
            &beta, complexptr.d_yvbp[i], complexptr.AvMs[i]));
        }
    }
    cudaDeviceSynchronize();
    phase2_complex(complexptr.d_yv, complexptr.d_phase2mapping, complexptr.d_yu, 
    complexptr.complexgranksum, streamptr[0]);
    cudaDeviceSynchronize();
    // missing the phase 2
    SeismicPCMatrix seismicpcmat(datafolder, 
    acc, nb, freqlist[0], originM, originN);
    Matrix<complex<float>> xmat = seismicpcmat.GetX();
    size_t coloffset = 0;

    // check for final yv output
    CopyDataB2HD(complexptr.h_yv, complexptr.d_yv, complexptr.complexgranksum);
    CopyDataB2HD(complexptr.h_yu, complexptr.d_yu, complexptr.complexgranksum);
    Matrix<complex<float>> densemat = seismicpcmat.GetDense();
    Matrix<complex<float>> yv = seismicpcmat.Phase1();
    auto cyv = Matrix<complex<float>>(complexptr.h_yv, complexptr.complexgranksum, 1);
    Matrix<complex<float>> yu = seismicpcmat.Phase2();
    auto cyu = Matrix<complex<float>>(complexptr.h_yu, complexptr.complexgranksum, 1);
    cout << "yu output " << cyu.allclose(yu) << endl;
    complexptr.FreeData();
}

BENCHMARK_DEFINE_F(SeismicFixture, Phase1SinglePtrManualCUDAGraphTest)
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


BENCHMARK_DEFINE_F(SeismicFixture, Phase2_OneFP16Test)(benchmark::State& state){

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
    // we want to split this rmats
    
    Float16Ptr fp16ptr(paddingM, paddingN, nb, Rmats[freqlist[0]], Rmat3);
    fp16ptr.InitData(datafolder, acc, freqlist[0], originN);

    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    fp16ptr.CopyData2GPU();
    // phase 1
    half alpha = 1.0;
    half beta = 0.0;
    for(int i=0; i<39; i++){
        if(fp16ptr.AvMs[i] == 0) continue;
        CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
            fp16ptr.AvMs[i], fp16ptr.AvNs[i], fp16ptr.AvKs[i], 
            &alpha, fp16ptr.d_Avbp[i][0], fp16ptr.AvMs[i], 
            fp16ptr.d_xbp[i], fp16ptr.AvKs[i], 
            &beta, fp16ptr.d_yvbp[i][0], fp16ptr.AvMs[i]);
        );
        CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
            fp16ptr.AvMs[i], fp16ptr.AvNs[i], fp16ptr.AvKs[i], 
            &alpha, fp16ptr.d_Avbp[i][1], fp16ptr.AvMs[i], 
            fp16ptr.d_xbp[i], fp16ptr.AvKs[i], 
            &beta, fp16ptr.d_yvbp[i][1], fp16ptr.AvMs[i]);
        );
    }
    // merge_half_2halfout_realimag(fp16ptr.d_yv[0], fp16ptr.d_yv[1], 
    // fp16ptr.d_fp16colrank, fp16ptr.Ntg,
    // fp16ptr.d_yvout[0], fp16ptr.d_yvout[1], 
    // fp16ptr.fp16granksum, streamptr[0]);
    // cudaDeviceSynchronize();
    phase2_half(fp16ptr.d_yv[0], fp16ptr.d_yv[1], 
    fp16ptr.d_fp16colrank, fp16ptr.Ntg,fp16ptr.d_phase2mapping, 
    fp16ptr.d_yu, 
    fp16ptr.fp16granksum, streamptr[0]);
    cudaDeviceSynchronize();
    // // missing the phase 2
    SeismicPCMatrix seismicpcmat(datafolder, 
    acc, nb, freqlist[0], originM, originN);
    Matrix<complex<float>> xmat = seismicpcmat.GetX();

    // size_t coloffset = 0;

    // check for final yv output
    CopyDataB2HD(fp16ptr.h_yu, fp16ptr.d_yu, 2*fp16ptr.fp16granksum);
    vector<complex<float>> yuvec;
    size_t offset = 0;
    // recover yu
    for(int i=0; i<fp16ptr.rowsum.size(); i++){
        int curlda = fp16ptr.rowsum[i];
        for(int k=0; k<curlda; k++){
            auto tmp = complex<float>( fp16ptr.h_yu[offset + k], fp16ptr.h_yu[offset + curlda + k] );
            yuvec.push_back(tmp);
        }
        offset += 2 * curlda;
    }
    
    Matrix<complex<float>> densemat = seismicpcmat.GetDense();
    Matrix<complex<float>> yv = seismicpcmat.Phase1();
    Matrix<complex<float>> yu = seismicpcmat.Phase2();
    auto cyu = Matrix<complex<float>>(yuvec, yuvec.size(), 1);

    cout << "yu output " << cyu.allclose(yu) << endl;

    fp16ptr.FreeData();

}



BENCHMARK_DEFINE_F(SeismicFixture, Phase3_OneFP32Test)(benchmark::State& state)
{

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
    cudaDeviceSynchronize();
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
    phase3_merge(fp32ptr.d_y[0], fp32ptr.d_y[1],
    fp32ptr.nb, fp32ptr.d_finaly, fp32ptr.M, streamptr[0]);
    cudaDeviceSynchronize();    
    SeismicPCMatrix seismicpcmat(datafolder, 
    acc, nb, freqlist[0], originM, originN);
    auto xmat = seismicpcmat.GetX();
    auto yv = seismicpcmat.Phase1();
    auto yu = seismicpcmat.Phase2();
    auto y = seismicpcmat.Phase3();
    

    // yu check
    
    vector<complex<float>> vec;
    size_t offset = 0;
    CopyDataB2HD(fp32ptr.h_yu, fp32ptr.d_yu, fp32ptr.fp32granksum * 2);
    for(int i=0; i<fp32ptr.Mtg; i++){
        int cur = fp32ptr.rowsum[i];
        auto cyu = Matrix<float>(fp32ptr.h_yu + 2 * offset , cur, 2);
        for(int j=0; j<fp32ptr.rowsum[i]; j++) {    
            vec.push_back( complex<float>( cyu.GetElem(j,0),cyu.GetElem(j,1)) );
        }
        offset += cur;
    }
    cout << "===============" << endl;
    cout << vec.size() << ", " << fp32ptr.fp32granksum << endl;
    
    auto complexyu = Matrix<complex<float>>(vec, fp32ptr.fp32granksum, 1);
    offset = 0;
    for(int i=0; i<fp32ptr.Ntg; i++){
        auto vtile = seismicpcmat.GetVTile(0,i);
        auto xtile = seismicpcmat.GetXtile(i);
        auto midy = vtile * xtile;
        int cur = fp32ptr.FP32Rmat.GetElem(0,i);
        cout << complexyu.Block({offset,offset + cur}).allclose(midy) << endl;
        offset += cur;
    }
    // check Au

    auto matau1 = Matrix<float>(fp32ptr.h_Au[0], nb, fp32ptr.fp32granksum);
    auto matau2 = Matrix<float>(fp32ptr.h_Au[1], nb, fp32ptr.fp32granksum);
    vec.clear();
    for(int i=0; i<fp32ptr.fp32granksum; i++) 
    {
        for(int j=0; j<nb; j++){
            float v1 = matau1.GetElem(j,i);
            float v2 = matau2.GetElem(j,i);
            vec.push_back(complex<float>(v1,v2));
        }
    }
    auto matau = Matrix<complex<float>>(vec, nb, fp32ptr.fp32granksum);
    offset = 0;
    for(int i=0; i<fp32ptr.Ntg; i++){
        auto utile = seismicpcmat.GetUTile(0,i);

        cout << matau.Block({0,(size_t)nb},{offset, offset+fp32ptr.FP32Rmat.GetElem(0,i)}).allclose(utile) << endl;
        offset += fp32ptr.FP32Rmat.GetElem(0,i);
    }
    auto firstrow = matau.Block({0,(size_t)nb},{0,(size_t)fp32ptr.rowsum[0]}) * complexyu.Block({0,(size_t)fp32ptr.rowsum[0]});
    cout << firstrow.Row() <<".," << firstrow.Col() << endl;
    
    size_t sznb = (size_t)nb;
    // check y
    offset = 0;
    for(size_t i=0; i<fp32ptr.Mtg; i++){
        size_t cur = fp32ptr.rowsum[i];
        auto newA = matau.Block({0,sznb},{offset, offset+cur});
        auto newX = complexyu.Block({offset,offset + cur});
        auto finalres = newA * newX;
        auto yb = y.Block({i*sznb,(i+1)*sznb});
        cout << finalres.allclose(yb) << endl;
        auto newrr = seismicpcmat.GetReal(newA) * seismicpcmat.GetReal(newX);
        auto newri = seismicpcmat.GetReal(newA) * seismicpcmat.GetImag(newX);
        auto newir = seismicpcmat.GetImag(newA) * seismicpcmat.GetReal(newX);
        auto newii = seismicpcmat.GetImag(newA) * seismicpcmat.GetImag(newX);
        
        CopyDataB2HD(fp32ptr.h_y[0], fp32ptr.d_y[0] + 2*nb*i, nb * 2);
        CopyDataB2HD(fp32ptr.h_y[1], fp32ptr.d_y[1] + 2*nb*i, nb * 2);
        auto cyrr_ri = Matrix<float>(fp32ptr.h_y[0], nb, 2);
        auto cyir_ii = Matrix<float>(fp32ptr.h_y[1], nb, 2);
        cout << "rr" << cyrr_ri.Block({0,sznb},{0,1}).allclose(newrr) << endl;
        cout << "ri" << cyrr_ri.Block({0,sznb},{1,2}).allclose(newri) << endl;
        cout << "ir" << cyir_ii.Block({0,sznb},{0,1}).allclose(newir) << endl;
        cout << "ii" << cyir_ii.Block({0,sznb},{1,2}).allclose(newii) << endl;
        offset += cur;
    }
    
    // check finaly 
    CopyDataB2HD(fp32ptr.h_finaly, reinterpret_cast<complex<float>*>(fp32ptr.d_finaly), fp32ptr.M);
    auto finalyc = Matrix<complex<float>>(fp32ptr.h_finaly, fp32ptr.M, 1);
    cout << "final y" << finalyc.allclose(y) << endl;

    fp32ptr.FreeData();


}


BENCHMARK_DEFINE_F(SeismicFixture, Phase3_Complex32Test)(benchmark::State& state){
    for(auto st : state){

    }
    ComplexPtr complexptr(paddingM, paddingN, nb, Rmats[freqlist[0]], FP32Rmats[freqlist[0]]);
    complexptr.InitData(datafolder, acc, freqlist[0], originN);
    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    complexptr.CopyData2GPU();
    // phase 1
    cuComplex alpha;
    cuComplex beta;
    alpha.x = 1.0; alpha.y = 0.0;
    beta.x = 0.0; beta.y = 0.0;
    for(int i=0; i<39; i++){
        if(complexptr.AvMs[i] != 0){
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
            complexptr.AvMs[i], complexptr.AvNs[i], complexptr.AvKs[i], 
            &alpha, complexptr.d_Avbp[i], complexptr.AvMs[i], 
            complexptr.d_xbp[i], complexptr.AvKs[i], 
            &beta, complexptr.d_yvbp[i], complexptr.AvMs[i]));
        }
    }
    cudaDeviceSynchronize();
    phase2_complex(complexptr.d_yv, complexptr.d_phase2mapping, complexptr.d_yu, 
    complexptr.complexgranksum, streamptr[0]);
    cudaDeviceSynchronize();
    for(int i=0; i<39; i++){
        if(complexptr.AuMs[i] != 0){
            CUBLASCHECK(
                cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                complexptr.AuMs[i], complexptr.AuNs[i], complexptr.AuKs[i], 
                &alpha, complexptr.d_Aubp[i], complexptr.AuMs[i], 
                complexptr.d_yubp[i], complexptr.AuKs[i], 
                &beta, complexptr.d_ybp[i], complexptr.AuMs[i])
            );
        }
    }
    cudaDeviceSynchronize();
    // missing the phase 2
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

BENCHMARK_DEFINE_F(SeismicFixture, Phase3_OneFP16Test)(benchmark::State& state){

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
    // we want to split this rmats
    
    Float16Ptr fp16ptr(paddingM, paddingN, nb, Rmats[freqlist[0]], Rmat3);
    fp16ptr.InitData(datafolder, acc, freqlist[0], originN);

    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    fp16ptr.CopyData2GPU();
    // phase 1
    half alpha = 1.0;
    half beta = 0.0;
    for(int i=0; i<39; i++){
        if(fp16ptr.AvMs[i] == 0) continue;
        CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
            fp16ptr.AvMs[i], fp16ptr.AvNs[i], fp16ptr.AvKs[i], 
            &alpha, fp16ptr.d_Avbp[i][0], fp16ptr.AvMs[i], 
            fp16ptr.d_xbp[i], fp16ptr.AvKs[i], 
            &beta, fp16ptr.d_yvbp[i][0], fp16ptr.AvMs[i]);
        );
        CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
            fp16ptr.AvMs[i], fp16ptr.AvNs[i], fp16ptr.AvKs[i], 
            &alpha, fp16ptr.d_Avbp[i][1], fp16ptr.AvMs[i], 
            fp16ptr.d_xbp[i], fp16ptr.AvKs[i], 
            &beta, fp16ptr.d_yvbp[i][1], fp16ptr.AvMs[i]);
        );
    }
    // merge_half_2halfout_realimag(fp16ptr.d_yv[0], fp16ptr.d_yv[1], 
    // fp16ptr.d_fp16colrank, fp16ptr.Ntg,
    // fp16ptr.d_yvout[0], fp16ptr.d_yvout[1], 
    // fp16ptr.fp16granksum, streamptr[0]);
    cudaDeviceSynchronize();
    phase2_half(fp16ptr.d_yv[0], fp16ptr.d_yv[1], 
    fp16ptr.d_fp16colrank, fp16ptr.Ntg,fp16ptr.d_phase2mapping, 
    fp16ptr.d_yu, fp16ptr.fp16granksum, streamptr[0]);
    cudaDeviceSynchronize();
    // phase 3
    for(int i=0; i<39; i++){
        cout << fp16ptr.AuMs[i] << ", " << fp16ptr.AuNs[i] << ", " << fp16ptr.AuKs[i] << endl;
        if(fp16ptr.AuMs[i] != 0){
            CUBLASCHECK(cublasGemmEx(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp16ptr.AuMs[i], fp16ptr.AuNs[i], fp16ptr.AuKs[i], 
                &alpha, fp16ptr.d_Aubp[i][0], CUDA_R_16F, fp16ptr.AuMs[i], 
                fp16ptr.d_yubp[i], CUDA_R_16F, fp16ptr.AuKs[i], 
                &beta, fp16ptr.d_ybp[i][0], CUDA_R_16F, fp16ptr.AuMs[i], CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
            );
            CUBLASCHECK(cublasGemmEx(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp16ptr.AuMs[i], fp16ptr.AuNs[i], fp16ptr.AuKs[i], 
                &alpha, fp16ptr.d_Aubp[i][1], CUDA_R_16F,fp16ptr.AuMs[i], 
                fp16ptr.d_yubp[i], CUDA_R_16F, fp16ptr.AuKs[i], 
                &beta, fp16ptr.d_ybp[i][1], CUDA_R_16F, fp16ptr.AuMs[i], CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
            );
        }
    }
    cudaDeviceSynchronize();    
    
    // CopyDataB2HD(fp16ptr.h_Au[0], fp16ptr.d_Au[0], fp16ptr.AuMs[0] * fp16ptr.AuKs[0]);
    // vector<float> sancheck;
    // ConvertFP16ToFP32(sancheck, fp16ptr.h_Au[0], fp16ptr.AuMs[0] * fp16ptr.AuKs[0]);
    // auto back_Au = Matrix<float>(sancheck, fp16ptr.AuMs[0] , fp16ptr.AuKs[0]);


    // final merge
    phase3_merge_half(fp16ptr.d_y[0], fp16ptr.d_y[1],
    fp16ptr.nb, fp16ptr.d_finaly, fp16ptr.M, streamptr[0]);
    cudaDeviceSynchronize();

    // SeismicPCMatrix seismicpcmat(datafolder, 
    // acc, nb, freqlist[0], originM, originN);
    // Matrix<complex<float>> xmat = seismicpcmat.GetX();
    // CopyDataB2HD(fp16ptr.h_yu, fp16ptr.d_yu, 2*fp16ptr.fp16granksum);
    // vector<complex<float>> yuvec;
    // size_t offset = 0;
    // // recover yu
    // for(int i=0; i<fp16ptr.rowsum.size(); i++){
    //     int curlda = fp16ptr.rowsum[i];
    //     for(int k=0; k<curlda; k++){
    //         auto tmp = complex<float>( fp16ptr.h_yu[offset + k], fp16ptr.h_yu[offset + curlda + k] );
    //         yuvec.push_back(tmp);
    //     }
    //     offset += 2 * curlda;
    // }
    
    // Matrix<complex<float>> densemat = seismicpcmat.GetDense();
    // Matrix<complex<float>> yv = seismicpcmat.Phase1();
    // Matrix<complex<float>> yu = seismicpcmat.Phase2();
    // auto cyu = Matrix<complex<float>>(yuvec, yuvec.size(), 1);

    // cout << "yu output " << cyu.allclose(yu) << endl;


    SeismicPCMatrix seismicpcmat(datafolder, 
    acc, nb, freqlist[0], originM, originN);
    auto xmat = seismicpcmat.GetX();
    auto yv = seismicpcmat.Phase1();
    auto yu = seismicpcmat.Phase2();
    auto y = seismicpcmat.Phase3();
    
    vector<complex<float>> vec;
    size_t offset = 0;
    CopyDataB2HD(fp16ptr.h_yu, fp16ptr.d_yu, fp16ptr.fp16granksum * 2);
    for(int i=0; i<fp16ptr.Mtg; i++){
        int cur = fp16ptr.rowsum[i];
        vector<float> yuvec;
        ConvertFP16ToFP32(yuvec, fp16ptr.h_yu+2*offset, cur * 2);
        auto cyu = Matrix<float>(yuvec , cur, 2);
        for(int j=0; j<fp16ptr.rowsum[i]; j++) {    
            vec.push_back( complex<float>( cyu.GetElem(j,0),cyu.GetElem(j,1)) );
        }
        offset += cur;
    }
    cout << "===============" << endl;
    cout << vec.size() << ", " << fp16ptr.fp16granksum << endl;
    
    auto complexyu = Matrix<complex<float>>(vec, fp16ptr.fp16granksum, 1);
    offset = 0;
    for(int i=0; i<fp16ptr.Ntg; i++){
        auto vtile = seismicpcmat.GetVTile(0,i);
        auto xtile = seismicpcmat.GetXtile(i);
        auto midy = vtile * xtile;
        int cur = fp16ptr.FP16Rmat.GetElem(0,i);
        cout << complexyu.Block({offset,offset + cur}).allclose(midy) << endl;
        offset += cur;
    }
    // check Au
    vector<float> au0, au1;
    ConvertFP16ToFP32(au0, fp16ptr.h_Au[0], nb * fp16ptr.fp16granksum);
    ConvertFP16ToFP32(au1, fp16ptr.h_Au[1], nb * fp16ptr.fp16granksum);

    auto matau1 = Matrix<float>(au0, nb, fp16ptr.fp16granksum);
    auto matau2 = Matrix<float>(au1, nb, fp16ptr.fp16granksum);
    vec.clear();
    for(int i=0; i<fp16ptr.fp16granksum; i++) 
    {
        for(int j=0; j<nb; j++){
            float v1 = matau1.GetElem(j,i);
            float v2 = matau2.GetElem(j,i);
            vec.push_back(complex<float>(v1,v2));
        }
    }
    auto matau = Matrix<complex<float>>(vec, nb, fp16ptr.fp16granksum);
    offset = 0;
    for(int i=0; i<fp16ptr.Ntg; i++){
        auto utile = seismicpcmat.GetUTile(0,i);

        cout << matau.Block({0,(size_t)nb},{offset, offset+fp16ptr.FP16Rmat.GetElem(0,i)}).allclose(utile) << endl;
        offset += fp16ptr.FP16Rmat.GetElem(0,i);
    }
    auto firstrow = matau.Block({0,(size_t)nb},{0,(size_t)fp16ptr.rowsum[0]}) * complexyu.Block({0,(size_t)fp16ptr.rowsum[0]});
    cout << firstrow.Row() <<".," << firstrow.Col() << endl;
    
    size_t sznb = (size_t)nb;
    // check y
    offset = 0;
    for(size_t i=0; i<fp16ptr.Mtg; i++){
        size_t cur = fp16ptr.rowsum[i];
        auto newA = matau.Block({0,sznb},{offset, offset+cur});
        auto newX = complexyu.Block({offset,offset + cur});
        auto finalres = newA * newX;
        auto yb = y.Block({i*sznb,(i+1)*sznb});
        cout << finalres.allclose(yb) << endl;
        auto newrr = seismicpcmat.GetReal(newA) * seismicpcmat.GetReal(newX);
        auto newri = seismicpcmat.GetReal(newA) * seismicpcmat.GetImag(newX);
        auto newir = seismicpcmat.GetImag(newA) * seismicpcmat.GetReal(newX);
        auto newii = seismicpcmat.GetImag(newA) * seismicpcmat.GetImag(newX);
        
        CopyDataB2HD(fp16ptr.h_y[0], fp16ptr.d_y[0] + 2*nb*i, nb * 2);
        CopyDataB2HD(fp16ptr.h_y[1], fp16ptr.d_y[1] + 2*nb*i, nb * 2);
        vector<float> y0,y1;
        ConvertFP16ToFP32(y0, fp16ptr.h_y[0], nb*2);
        ConvertFP16ToFP32(y1, fp16ptr.h_y[1], nb*2);
        auto cyrr_ri = Matrix<float>(y0, nb, 2);
        auto cyir_ii = Matrix<float>(y1, nb, 2);
        cout << "rr" << cyrr_ri.Block({0,sznb},{0,1}).allclose(newrr) << endl;
        cout << "ri" << cyrr_ri.Block({0,sznb},{1,2}).allclose(newri) << endl;
        cout << "ir" << cyir_ii.Block({0,sznb},{0,1}).allclose(newir) << endl;
        cout << "ii" << cyir_ii.Block({0,sznb},{1,2}).allclose(newii) << endl;
        offset += cur;
    }

    CopyDataB2HD(fp16ptr.h_finaly, reinterpret_cast<complex<float>*>(fp16ptr.d_finaly), fp16ptr.M);
    auto finalyc = Matrix<complex<float>>(fp16ptr.h_finaly, fp16ptr.M, 1);
    cout << finalyc.Block({0,10}) << endl;
    cout << "final y" << finalyc.allclose(y) << endl;

    fp16ptr.FreeData();

}



BENCHMARK_DEFINE_F(SeismicFixture, Phase3_TwoFP32Test)(benchmark::State& state)
{
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
    cout << Rmat3.allclose(Rmats[freqlist[0]]) << endl;
    cout << Rmat1.Block({0,10},{0,10}) << endl;
    cout << Rmat2.Block({0,10},{0,10}) << endl;
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
    float alpha = 1.0;
    float beta = 0.0;
    // phase 1
    for(int i=0; i<39; i++){
        if(fp32ptr1.AvMs[i] != 0){
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp32ptr1.AvMs[i], fp32ptr1.AvNs[i], fp32ptr1.AvKs[i], 
                &alpha, fp32ptr1.d_Avbp[i][0], fp32ptr1.AvMs[i], 
                fp32ptr1.d_xbp[i], fp32ptr1.AvKs[i], 
                &beta, fp32ptr1.d_yvbp[i][0], fp32ptr1.AvMs[i]);
            );
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp32ptr1.AvMs[i], fp32ptr1.AvNs[i], fp32ptr1.AvKs[i], 
                &alpha, fp32ptr1.d_Avbp[i][1], fp32ptr1.AvMs[i], 
                fp32ptr1.d_xbp[i], fp32ptr1.AvKs[i], 
                &beta, fp32ptr1.d_yvbp[i][1], fp32ptr1.AvMs[i]);
            );
        }
        if(fp32ptr2.AvMs[i] != 0){
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp32ptr2.AvMs[i], fp32ptr2.AvNs[i], fp32ptr2.AvKs[i], 
                &alpha, fp32ptr2.d_Avbp[i][0], fp32ptr2.AvMs[i], 
                fp32ptr2.d_xbp[i], fp32ptr2.AvKs[i], 
                &beta, fp32ptr2.d_yvbp[i][0], fp32ptr2.AvMs[i]);
            );
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp32ptr2.AvMs[i], fp32ptr2.AvNs[i], fp32ptr2.AvKs[i], 
                &alpha, fp32ptr2.d_Avbp[i][1], fp32ptr2.AvMs[i], 
                fp32ptr2.d_xbp[i], fp32ptr2.AvKs[i], 
                &beta, fp32ptr2.d_yvbp[i][1], fp32ptr2.AvMs[i]);
            );
        }
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
    cudaDeviceSynchronize();
    // phase 3
    for(int i=0; i<39; i++){
        if(fp32ptr1.AuMs[i] != 0){
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp32ptr1.AuMs[i], fp32ptr1.AuNs[i], fp32ptr1.AuKs[i], 
                &alpha, fp32ptr1.d_Aubp[i][0], fp32ptr1.AuMs[i], 
                fp32ptr1.d_yubp[i], fp32ptr1.AuKs[i], 
                &beta, fp32ptr1.d_ybp[i][0], fp32ptr1.AuMs[i]);
            );
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp32ptr1.AuMs[i], fp32ptr1.AuNs[i], fp32ptr1.AuKs[i], 
                &alpha, fp32ptr1.d_Aubp[i][1], fp32ptr1.AuMs[i], 
                fp32ptr1.d_yubp[i], fp32ptr1.AuKs[i], 
                &beta, fp32ptr1.d_ybp[i][1], fp32ptr1.AuMs[i]);
            ); 
        }
        if(fp32ptr2.AuMs[i] != 0){
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp32ptr2.AuMs[i], fp32ptr2.AuNs[i], fp32ptr2.AuKs[i], 
                &alpha, fp32ptr2.d_Aubp[i][0], fp32ptr2.AuMs[i], 
                fp32ptr2.d_yubp[i], fp32ptr2.AuKs[i], 
                &beta, fp32ptr2.d_ybp[i][0], fp32ptr2.AuMs[i]);
            );
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp32ptr2.AuMs[i], fp32ptr2.AuNs[i], fp32ptr2.AuKs[i], 
                &alpha, fp32ptr2.d_Aubp[i][1], fp32ptr2.AuMs[i], 
                fp32ptr2.d_yubp[i], fp32ptr2.AuKs[i], 
                &beta, fp32ptr2.d_ybp[i][1], fp32ptr2.AuMs[i]);
            ); 
        }
    }
    // final merge
    phase3_merge(fp32ptr1.d_y[0], fp32ptr1.d_y[1],
    fp32ptr1.nb, fp32ptr1.d_finaly, fp32ptr1.M, streamptr[0]);
    phase3_merge(fp32ptr2.d_y[0], fp32ptr2.d_y[1],
    fp32ptr2.nb, fp32ptr2.d_finaly, fp32ptr2.M, streamptr[0]);
    cudaDeviceSynchronize();
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



BENCHMARK_DEFINE_F(SeismicFixture, Phase3_TwoFP16Test)(benchmark::State& state)
{
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
    cout << Rmat3.allclose(Rmats[freqlist[0]]) << endl;
    cout << Rmat1.Block({0,10},{0,10}) << endl;
    cout << Rmat2.Block({0,10},{0,10}) << endl;
    // we want to split this rmats
    
    Float16Ptr fp16ptr1(paddingM, paddingN, nb, Rmats[freqlist[0]], Rmat1);
    Float16Ptr fp16ptr2(paddingM, paddingN, nb, Rmats[freqlist[0]], Rmat2);
    fp16ptr1.InitData(datafolder, acc, freqlist[0], originN);
    fp16ptr2.InitData(datafolder, acc, freqlist[0], originN);


    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    fp16ptr1.CopyData2GPU();
    fp16ptr2.CopyData2GPU();
    half alpha = 1.0;
    half beta = 0.0;
    // phase 1
    for(int i=0; i<39; i++){
        if(fp16ptr1.AvMs[i] != 0){
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp16ptr1.AvMs[i], fp16ptr1.AvNs[i], fp16ptr1.AvKs[i], 
                &alpha, fp16ptr1.d_Avbp[i][0], fp16ptr1.AvMs[i], 
                fp16ptr1.d_xbp[i], fp16ptr1.AvKs[i], 
                &beta, fp16ptr1.d_yvbp[i][0], fp16ptr1.AvMs[i]);
            );
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp16ptr1.AvMs[i], fp16ptr1.AvNs[i], fp16ptr1.AvKs[i], 
                &alpha, fp16ptr1.d_Avbp[i][1], fp16ptr1.AvMs[i], 
                fp16ptr1.d_xbp[i], fp16ptr1.AvKs[i], 
                &beta, fp16ptr1.d_yvbp[i][1], fp16ptr1.AvMs[i]);
            );
        }
        if(fp16ptr2.AvMs[i] != 0){
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp16ptr2.AvMs[i], fp16ptr2.AvNs[i], fp16ptr2.AvKs[i], 
                &alpha, fp16ptr2.d_Avbp[i][0], fp16ptr2.AvMs[i], 
                fp16ptr2.d_xbp[i], fp16ptr2.AvKs[i], 
                &beta, fp16ptr2.d_yvbp[i][0], fp16ptr2.AvMs[i]);
            );
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp16ptr2.AvMs[i], fp16ptr2.AvNs[i], fp16ptr2.AvKs[i], 
                &alpha, fp16ptr2.d_Avbp[i][1], fp16ptr2.AvMs[i], 
                fp16ptr2.d_xbp[i], fp16ptr2.AvKs[i], 
                &beta, fp16ptr2.d_yvbp[i][1], fp16ptr2.AvMs[i]);
            );
        }
    }
    // phase 2
    phase2_half(fp16ptr1.d_yv[0], fp16ptr1.d_yv[1], 
    fp16ptr1.d_fp16colrank, fp16ptr1.Ntg,fp16ptr1.d_phase2mapping, 
    fp16ptr1.d_yu, 
    fp16ptr1.fp16granksum, streamptr[0]);
    phase2_half(fp16ptr2.d_yv[0], fp16ptr2.d_yv[1], 
    fp16ptr2.d_fp16colrank, fp16ptr2.Ntg,fp16ptr2.d_phase2mapping, 
    fp16ptr2.d_yu, 
    fp16ptr2.fp16granksum, streamptr[0]);
    cudaDeviceSynchronize();
    // phase 3
    for(int i=0; i<39; i++){
        if(fp16ptr1.AuMs[i] != 0){
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp16ptr1.AuMs[i], fp16ptr1.AuNs[i], fp16ptr1.AuKs[i], 
                &alpha, fp16ptr1.d_Aubp[i][0], fp16ptr1.AuMs[i], 
                fp16ptr1.d_yubp[i], fp16ptr1.AuKs[i], 
                &beta, fp16ptr1.d_ybp[i][0], fp16ptr1.AuMs[i]);
            );
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp16ptr1.AuMs[i], fp16ptr1.AuNs[i], fp16ptr1.AuKs[i], 
                &alpha, fp16ptr1.d_Aubp[i][1], fp16ptr1.AuMs[i], 
                fp16ptr1.d_yubp[i], fp16ptr1.AuKs[i], 
                &beta, fp16ptr1.d_ybp[i][1], fp16ptr1.AuMs[i]);
            ); 
        }
        if(fp16ptr2.AuMs[i] != 0){
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp16ptr2.AuMs[i], fp16ptr2.AuNs[i], fp16ptr2.AuKs[i], 
                &alpha, fp16ptr2.d_Aubp[i][0], fp16ptr2.AuMs[i], 
                fp16ptr2.d_yubp[i], fp16ptr2.AuKs[i], 
                &beta, fp16ptr2.d_ybp[i][0], fp16ptr2.AuMs[i]);
            );
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp16ptr2.AuMs[i], fp16ptr2.AuNs[i], fp16ptr2.AuKs[i], 
                &alpha, fp16ptr2.d_Aubp[i][1], fp16ptr2.AuMs[i], 
                fp16ptr2.d_yubp[i], fp16ptr2.AuKs[i], 
                &beta, fp16ptr2.d_ybp[i][1], fp16ptr2.AuMs[i]);
            ); 
        }
    }
    // final merge
    phase3_merge_half(fp16ptr1.d_y[0], fp16ptr1.d_y[1],
    fp16ptr1.nb, fp16ptr1.d_finaly, fp16ptr1.M, streamptr[0]);
    phase3_merge_half(fp16ptr2.d_y[0], fp16ptr2.d_y[1],
    fp16ptr2.nb, fp16ptr2.d_finaly, fp16ptr2.M, streamptr[0]);
    cudaDeviceSynchronize();
    SeismicPCMatrix seismicpcmat(datafolder, 
    acc, nb, freqlist[0], originM, originN);
    auto xmat = seismicpcmat.GetX();
    auto yv = seismicpcmat.Phase1();
    auto yu = seismicpcmat.Phase2();
    auto y = seismicpcmat.Phase3();
    
    // we directly do final test
    // check finaly 
    CopyDataB2HD(fp16ptr1.h_finaly, reinterpret_cast<complex<float>*>(fp16ptr1.d_finaly), fp16ptr1.M);
    CopyDataB2HD(fp16ptr2.h_finaly, reinterpret_cast<complex<float>*>(fp16ptr2.d_finaly), fp16ptr2.M);

    auto finalyc1 = Matrix<complex<float>>(fp16ptr1.h_finaly, fp16ptr1.M, 1);
    auto finalyc2 = Matrix<complex<float>>(fp16ptr2.h_finaly, fp16ptr2.M, 1);
    auto finalyc = finalyc1 + finalyc2;
    cout << " Two pointers final y " << finalyc.allclose(y) << endl;

    fp16ptr1.FreeData();
    fp16ptr2.FreeData();

}

// BENCHMARK_REGISTER_F(SeismicFixture, Phase3_TwoFP16Test)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(1)
// ->UseManualTime()
// ->Repetitions(1);


BENCHMARK_DEFINE_F(SeismicFixture, Phase3_OneFP32OneFP16Test)(benchmark::State& state)
{
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
    cout << Rmat3.allclose(Rmats[freqlist[0]]) << endl;
    cout << Rmat1.Block({0,10},{0,10}) << endl;
    cout << Rmat2.Block({0,10},{0,10}) << endl;
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

    fp32ptr1.CopyData2GPU();
    fp16ptr2.CopyData2GPU();
    half alpha_half = (half)1.0;
    half beta_half = (half)0.0;
    float alpha_float = 1.0;
    float beta_float = 0.0;
    // phase 1
    for(int i=0; i<39; i++){
        if(fp32ptr1.AvMs[i] != 0){
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp32ptr1.AvMs[i], fp32ptr1.AvNs[i], fp32ptr1.AvKs[i], 
                &alpha_float, fp32ptr1.d_Avbp[i][0], fp32ptr1.AvMs[i], 
                fp32ptr1.d_xbp[i], fp32ptr1.AvKs[i], 
                &beta_float, fp32ptr1.d_yvbp[i][0], fp32ptr1.AvMs[i]);
            );
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp32ptr1.AvMs[i], fp32ptr1.AvNs[i], fp32ptr1.AvKs[i], 
                &alpha_float, fp32ptr1.d_Avbp[i][1], fp32ptr1.AvMs[i], 
                fp32ptr1.d_xbp[i], fp32ptr1.AvKs[i], 
                &beta_float, fp32ptr1.d_yvbp[i][1], fp32ptr1.AvMs[i]);
            );
        }
        if(fp16ptr2.AvMs[i] != 0){
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp16ptr2.AvMs[i], fp16ptr2.AvNs[i], fp16ptr2.AvKs[i], 
                &alpha_half, fp16ptr2.d_Avbp[i][0], fp16ptr2.AvMs[i], 
                fp16ptr2.d_xbp[i], fp16ptr2.AvKs[i], 
                &beta_half, fp16ptr2.d_yvbp[i][0], fp16ptr2.AvMs[i]);
            );
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp16ptr2.AvMs[i], fp16ptr2.AvNs[i], fp16ptr2.AvKs[i], 
                &alpha_half, fp16ptr2.d_Avbp[i][1], fp16ptr2.AvMs[i], 
                fp16ptr2.d_xbp[i], fp16ptr2.AvKs[i], 
                &beta_half, fp16ptr2.d_yvbp[i][1], fp16ptr2.AvMs[i]);
            );
        }
    }
    // phase 2
    phase2(fp32ptr1.d_yv[0], fp32ptr1.d_yv[1], 
    fp32ptr1.d_fp32colrank, fp32ptr1.Ntg,fp32ptr1.d_phase2mapping, 
    fp32ptr1.d_yu, fp32ptr1.fp32granksum, streamptr[0]);
    phase2_half(fp16ptr2.d_yv[0], fp16ptr2.d_yv[1], 
    fp16ptr2.d_fp16colrank, fp16ptr2.Ntg,fp16ptr2.d_phase2mapping, 
    fp16ptr2.d_yu, fp16ptr2.fp16granksum, streamptr[0]);
    cudaDeviceSynchronize();
    // phase 3
    for(int i=0; i<39; i++){
        if(fp32ptr1.AuMs[i] != 0){
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp32ptr1.AuMs[i], fp32ptr1.AuNs[i], fp32ptr1.AuKs[i], 
                &alpha_float, fp32ptr1.d_Aubp[i][0], fp32ptr1.AuMs[i], 
                fp32ptr1.d_yubp[i], fp32ptr1.AuKs[i], 
                &beta_float, fp32ptr1.d_ybp[i][0], fp32ptr1.AuMs[i]);
            );
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp32ptr1.AuMs[i], fp32ptr1.AuNs[i], fp32ptr1.AuKs[i], 
                &alpha_float, fp32ptr1.d_Aubp[i][1], fp32ptr1.AuMs[i], 
                fp32ptr1.d_yubp[i], fp32ptr1.AuKs[i], 
                &beta_float, fp32ptr1.d_ybp[i][1], fp32ptr1.AuMs[i]);
            ); 
        }
        if(fp16ptr2.AuMs[i] != 0){
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp16ptr2.AuMs[i], fp16ptr2.AuNs[i], fp16ptr2.AuKs[i], 
                &alpha_half, fp16ptr2.d_Aubp[i][0], fp16ptr2.AuMs[i], 
                fp16ptr2.d_yubp[i], fp16ptr2.AuKs[i], 
                &beta_half, fp16ptr2.d_ybp[i][0], fp16ptr2.AuMs[i]);
            );
            CUBLASCHECK(cublasgemm(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
                fp16ptr2.AuMs[i], fp16ptr2.AuNs[i], fp16ptr2.AuKs[i], 
                &alpha_half, fp16ptr2.d_Aubp[i][1], fp16ptr2.AuMs[i], 
                fp16ptr2.d_yubp[i], fp16ptr2.AuKs[i], 
                &beta_half, fp16ptr2.d_ybp[i][1], fp16ptr2.AuMs[i]);
            ); 
        }
    }
    // final merge
    phase3_merge(fp32ptr1.d_y[0], fp32ptr1.d_y[1],
    fp32ptr1.nb, fp32ptr1.d_finaly, fp32ptr1.M, streamptr[0]);
    phase3_merge_half(fp16ptr2.d_y[0], fp16ptr2.d_y[1],
    fp16ptr2.nb, fp16ptr2.d_finaly, fp16ptr2.M, streamptr[0]);
    cudaDeviceSynchronize();
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




BENCHMARK_DEFINE_F(SeismicFixture, CUDAC8ITest)(benchmark::State& state)
{
    size_t M ,N, K;
    M = 1024;
    N = 1;
    K = 256;

    // auto Amat = Matrix<complex<float>>(M,K);
    // auto Bmat = Matrix<complex<float>>(K,N);
    // auto Cmat = Matrix<complex<float>>(M,N);


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
    alpha.x = 1.0; alpha.y = 0.0; beta.x = 1.0; beta.y = 0.0;

    // init ref ptr
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            i32Aptr[i*2*N+j*2] = rand() % 127;
            i32Aptr[i*2*N+j*2+1] = rand() % 127;
            if(i == 0){
                i32Bptr[2*j] = rand() % 127;
                i32Bptr[2*j+1] = rand() % 127;
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
    cout << ccheck.Block({0,10}) << endl;

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

    CUBLASCHECK(cublascgemmex(cublashandleptr[0], CUBLAS_OP_N, CUBLAS_OP_N, 
    M, N, K, &alpha, d_Aptr, CUDA_C_8I, M, d_Bptr, CUDA_C_8I, K, &beta, 
    d_Cptr, CUDA_C_32F, M));

    cudaDeviceSynchronize();
    CopyDataB2HD(Cptr, reinterpret_cast<complex<float>*>(d_Cptr), M);
    auto finalcheck = Matrix<complex<float>>(Cptr, M, 1);
    cout << finalcheck.Block({0,10}) << endl;
    cout << finalcheck.allclose(ccheck) << endl;

}



BENCHMARK_DEFINE_F(SeismicFixture, Phase1_OneInt8Test)(benchmark::State& state){
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
    // we want to split this rmats
    
    Int8Ptr int8ptr(paddingM, paddingN, nb, Rmats[freqlist[0]], Rmat3);
    int8ptr.InitData(datafolder, acc, freqlist[0], originN);

    cudaStream_t * streamptr = new cudaStream_t[streamsize];
    cublasHandle_t * cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    // vector<int> i32vec;
    // ConvertINT8ToINT32(i32vec, int8ptr.h_Av, 1000);
    // auto Int8Av = Matrix<int>(i32vec, 10, 10);
    // cout << Int8Av << endl;
    // auto floatAv = Matrix<complex<float>>(int8ptr.h_Av_complex, 10,10);
    // cout << floatAv << endl;


    int8ptr.CopyData2GPU();

    // phase 1
    int alpha = 1;
    int beta = 0;
    for(int i=0; i<39; i++){
        if(int8ptr.AvMs[i] != 0){
            CUBLASCHECK(cublasGemmEx(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
            int8ptr.AvMs[i], int8ptr.AvNs[i], int8ptr.AvKs[i], 
            &alpha, int8ptr.d_Avbp[i][0], CUDA_R_8I, int8ptr.AvMs[i], 
            int8ptr.d_xbp[i], CUDA_R_8I, int8ptr.AvKs[i], 
            &beta, int8ptr.d_yvbp[i][0], CUDA_R_32I, int8ptr.AvMs[i], CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
            cudaDeviceSynchronize();
            // cout << int8ptr.AvMs[i] << ". " <<  int8ptr.AvNs[i] << ". " << int8ptr.AvKs[i] << endl;
            CUBLASCHECK(cublasGemmEx(cublashandleptr[0],CUBLAS_OP_N, CUBLAS_OP_N,
            int8ptr.AvMs[i], int8ptr.AvNs[i], int8ptr.AvKs[i], 
            &alpha, int8ptr.d_Avbp[i][1], CUDA_R_8I, int8ptr.AvMs[i], 
            int8ptr.d_xbp[i], CUDA_R_8I, int8ptr.AvKs[i], 
            &beta, int8ptr.d_yvbp[i][1], CUDA_R_32I, int8ptr.AvMs[i], CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));
        }

    }

    merge_int_2intout_realimag(int8ptr.d_yv[0], int8ptr.d_yv[1], 
    int8ptr.d_int8colrank, int8ptr.d_int8colrank_withpadding, 
    int8ptr.Ntg,
    int8ptr.d_yvout[0], int8ptr.d_yvout[1], 
    int8ptr.int8granksum_withpadding, int8ptr.int8granksum, streamptr[0]);
    cudaDeviceSynchronize();
    CopyDataB2HD(int8ptr.h_yvout[0], int8ptr.d_yvout[0], 10);
    // auto yv0 = Matrix<float>(int8ptr.h_yvout[0], int8ptr.int8granksum_withpadding, 1);
    // cout << yv0.Block({0,10}) << endl;
    // // missing the phase 2
    // SeismicPCMatrix seismicpcmat(datafolder, 
    // acc, nb, freqlist[0], originM, originN);
    // Matrix<complex<float>> xmat = seismicpcmat.GetX();

    // // check for middle yv output
    // for(int ntgi=0; ntgi<fp16ptr.Ntg; ntgi++){

    //     size_t curele = fp16ptr.colsum[ntgi];
    //     CopyDataB2HD(fp16ptr.h_yvout[0], fp16ptr.d_yvbp[ntgi][0], curele * 2);
    //     CopyDataB2HD(fp16ptr.h_yvout[1], fp16ptr.d_yvbp[ntgi][1], curele * 2);
    //     vector<float> hyvreal;
    //     vector<float> hyvimag;
    //     ConvertFP16ToFP32(hyvreal, fp16ptr.h_yvout[0], curele * 2);
    //     ConvertFP16ToFP32(hyvimag, fp16ptr.h_yvout[1], curele * 2);
    //     auto cyrr_ri = Matrix<float>(hyvreal, curele, 2);
    //     auto cyir_ii = Matrix<float>(hyvimag, curele, 2);

    //     size_t offset = 0;
    //     for(int i=0; i<fp16ptr.Mtg; i++){
    //         auto vtile = seismicpcmat.GetVTile(i,ntgi);
    //         auto xtile = seismicpcmat.GetXtile(ntgi);
    //         auto yrr = seismicpcmat.GetReal(vtile) * seismicpcmat.GetReal(xtile);
    //         auto yri = seismicpcmat.GetReal(vtile) * seismicpcmat.GetImag(xtile);
    //         auto yir = seismicpcmat.GetImag(vtile) * seismicpcmat.GetReal(xtile);
    //         auto yii = seismicpcmat.GetImag(vtile) * seismicpcmat.GetImag(xtile);
    //         auto v1 = cyrr_ri.Block({offset,offset + yrr.Row()}, {0,1}).allclose(yrr);
    //         auto v2 = cyrr_ri.Block({offset,offset + yrr.Row()}, {1,2}).allclose(yri);
    //         auto v3 = cyir_ii.Block({offset,offset + yir.Row()}, {0,1}).allclose(yir);
    //         auto v4 = cyir_ii.Block({offset,offset + yir.Row()}, {1,2}).allclose(yii);
    //         printf("%d %d %f %f %f %f \n", i, ntgi, v1, v2, v3,v4);
    //         offset += yrr.Row();
    //     }
    // }

    for(int i=0; i<streamsize; i++) cudaStreamDestroy(streamptr[i]);
    for(int i=0; i<streamsize; i++) cublasDestroy_v2(cublashandleptr[i]);
    delete[] streamptr;
    delete[] cublashandleptr;

    // // check for final yv output
    // CopyDataB2HD(fp16ptr.h_yvout[0], fp16ptr.d_yvout[0], fp16ptr.fp16granksum);
    // CopyDataB2HD(fp16ptr.h_yvout[1], fp16ptr.d_yvout[1], fp16ptr.fp16granksum);
    // Matrix<complex<float>> densemat = seismicpcmat.GetDense();
    // Matrix<complex<float>> yv = seismicpcmat.Phase1();
    // auto cyv = mergerealimag(fp16ptr.h_yvout[0], fp16ptr.h_yvout[1], fp16ptr.fp16granksum, 1);
    // cout << "yv output " << cyv.allclose(yv) << endl;
    // fp16ptr.FreeData();
}




// BENCHMARK_REGISTER_F(SeismicFixture, Phase1_Complex32Test)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(1)
// ->UseManualTime()
// ->Repetitions(1);


// BENCHMARK_REGISTER_F(SeismicFixture, Phase1_OneFP32Test)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(1)
// ->UseManualTime()
// ->Repetitions(1);

// BENCHMARK_REGISTER_F(SeismicFixture, Phase1_OneFP16Test)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(1)
// ->UseManualTime()
// ->Repetitions(1);

// BENCHMARK_REGISTER_F(SeismicFixture, Phase1_TwoFP32Test)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(1)
// ->UseManualTime()
// ->Repetitions(1);

// BENCHMARK_REGISTER_F(SeismicFixture, Phase1_TwoFP16Test)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(1)
// ->UseManualTime()
// ->Repetitions(1);

// BENCHMARK_REGISTER_F(SeismicFixture, Phase2_OneFP32Test)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(1)
// ->UseManualTime()
// ->Repetitions(1);

// BENCHMARK_REGISTER_F(SeismicFixture, Phase2_Complex32Test)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(1)
// ->UseManualTime()
// ->Repetitions(1);

// BENCHMARK_REGISTER_F(SeismicFixture, Phase2_OneFP16Test)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(1)
// ->UseManualTime()
// ->Repetitions(1);

// BENCHMARK_REGISTER_F(SeismicFixture, Phase3_OneFP32Test)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(1)
// ->UseManualTime()
// ->Repetitions(1);

BENCHMARK_REGISTER_F(SeismicFixture, Phase3_Complex32Test)
->Unit(benchmark::kMicrosecond)
->Iterations(1)
->UseManualTime()
->Repetitions(1);


// BENCHMARK_REGISTER_F(SeismicFixture, Phase3_OneFP16Test)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(1)
// ->UseManualTime()
// ->Repetitions(1);

// BENCHMARK_REGISTER_F(SeismicFixture, Phase3_TwoFP32Test)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(1)
// ->UseManualTime()
// ->Repetitions(1);

// BENCHMARK_REGISTER_F(SeismicFixture, Phase3_OneFP32OneFP16Test)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(1)
// ->UseManualTime()
// ->Repetitions(1);

// BENCHMARK_REGISTER_F(SeismicFixture, CUDAC8ITest)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(1)
// ->UseManualTime()
// ->Repetitions(1);


// BENCHMARK_REGISTER_F(SeismicFixture, Phase1_OneInt8Test)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(1)
// ->UseManualTime()
// ->Repetitions(1);


// BENCHMARK_REGISTER_F(SeismicFixture, Phase1SinglePtrManualCUDAGraphTest)
// ->Unit(benchmark::kMicrosecond)
// ->Iterations(1)
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