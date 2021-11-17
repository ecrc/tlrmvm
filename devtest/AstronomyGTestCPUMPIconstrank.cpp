
// CPU Library
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
#include <mpi.h>
#include <complex>
#include <string>
#include <vector>

#include "common/Common.h"
#include "common/AppUtil.h"
#include "tlrmvm/Tlrmvm.h"

#include "benchmark/benchmark.h"
#include "AstronomyUtil.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "omp.h"

using namespace std;
using namespace tlrmat;
using namespace tlrmvm;
using ::testing::Pointwise;
using ::testing::NanSensitiveFloatNear;

vector<string> g_command_line_arg_vec;

class AstronomyCPUTest: public testing::Test {

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
        constrank = atoi(g_command_line_arg_vec[6].c_str());
        if(nb < constrank){
            printf("wrong configuration nb %d constrank %d \n", nb, constrank);
            exit(1);
        }
        paddingM = CalculatePadding(originM, nb);
        paddingN = CalculatePadding(originN, nb);
        Mtglobal = paddingM / nb;
        Ntglobal = paddingN / nb;
        // load Data
        float *DataAv, *DataAu;
        int *DataR;        
        DataR = new int[Mtglobal * Ntglobal];
        // ReadAstronomyBinary(datafolder, &DataR, Mtglobal * Ntglobal, acc, nb, id);
        // replace DataR with constrank
        for(int i=0; i<Mtglobal; i++){
            for(int j=0; j<Ntglobal; j++){
                DataR[j*Mtglobal+i] = constrank;
            }
        }
        Rmat = Matrix<int>(DataR, Mtglobal, Ntglobal);
        granksum = Rmat.Sum();
        DataAv = new float[granksum * nb];
        DataAu = new float[granksum * nb];
        for(size_t i=0; i<granksum * nb; i++){
            DataAv[i] = 1.0;
            DataAu[i] = 1.0;
        }
        
        // ReadAstronomyBinary(datafolder+"/V", &DataAv, granksum * nb, acc ,nb, id);
        // ReadAstronomyBinary(datafolder+"/U", &DataAu, granksum * nb, acc ,nb, id);
        
        
        // MPI init 
        MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
        if(mpirank == 0)
        cout << "Mtg " << Mtglobal << "Ntg " << Ntglobal << endl;
        if(mpirank == 0)
        GtestLog( "MPI world size = " + to_string(mpisize) );


        // MPI Split Rmat Configuration
        splitcolids = SplitColCount(Rmat, mpisize);
        Rmatsubset = SplitInputRmatrix(Rmat, mpisize);
        size_t granksum = Rmat.Sum();
        size_t gr = 0;
        for(int i=0; i<Rmatsubset.size(); i++){
            gr += Rmatsubset[i].Sum();
        }
        assert(granksum == gr);
        subcolsum = Rmatsubset[mpirank].ColSum();
        Ntlocal = Rmatsubset[mpirank].Col();

        /**
         * Phase 1 preparation
         */
        // calculate layout for Av: Phase 1

        
        float val = 1.0;
        for(int i=0; i<Ntlocal; i++){
            AvMs.push_back( (size_t)subcolsum[i] );
            AvKs.push_back(nb);
            AvNs.push_back(1);
        }
        GetHostMemoryBatched(&Av, &x, &yv, &Avbatchpointer, &xbatchpointer, &yvbatchpointer, 
        AvMs, AvKs, AvNs, val);
        CaluclateTotalElements(AvMs, AvKs, AvNs, Avtotalelems, xtotalelems, yvtotalelems);


        // get random X
        Datax = new float[paddingN];
        if(mpirank == 0){
            #pragma omp parallel for 
            for(int i=0; i<paddingN; i++){
                Datax[i] = (float)rand() / (float)RAND_MAX;
                // Datax[i] = 1.0;
            }
            MPI_Bcast(Datax, paddingN, mpi_get_type<float>(), 0, MPI_COMM_WORLD);
        }else{
            MPI_Bcast(Datax, paddingN, mpi_get_type<float>(), 0, MPI_COMM_WORLD);   
        }
        for(int i=0; i < splitcolids[mpirank].size(); i++){
            int idx = splitcolids[mpirank][i];
            // memcpy(x + i*nb, Datax + idx * nb, nb * sizeof(float));
            CopyData(x + i * nb, Datax + idx * nb, nb);
        }
        MPI_Barrier(MPI_COMM_WORLD);


        auto globalrmatcolsum = Rmat.ColSum();
        prefixcolsum = PrefixSum(globalrmatcolsum.data(), Rmat.Col());
        float *Avptr = Av;
        for(auto idx : splitcolids[mpirank]){
            CopyData(Avptr, DataAv + prefixcolsum[idx] * nb, globalrmatcolsum[idx] * nb);
            Avptr += globalrmatcolsum[idx] * nb;
        }

        // phase 2 preparation
        tlrmvm::CalculatePhase2Offset(&offsetphase2_h, Mtglobal, Ntlocal, Rmatsubset[mpirank]);
        lranksum = Rmatsubset[mpirank].Sum();
        /**
         * Phase 3 preparation
         */
        subrowsum = Rmatsubset[mpirank].RowSum();
        for(int i=0; i<Mtglobal; i++){
            AuMs.push_back(nb);
            AuKs.push_back( (size_t)subrowsum[i] );
            AuNs.push_back(1);
        }
        GetHostMemoryBatched(&Au, &yu, &y, 
        &Aubatchpointer, &yubatchpointer, &ybatchpointer, AuMs, AuKs, AuNs, val);
        CaluclateTotalElements(AuMs, AuKs, AuNs, Autotalelems, yutotalelems, ytotalelems);
        yfinal = new float[paddingM];
        memset(yfinal, 0, sizeof(float)*paddingM );
        size_t accalongrow = 0;
        float * Auptr = Au;
        for(size_t i=0; i<Mtglobal; i++){
            auto rowsumprefix = PrefixSum(Rmat.Block({i,i+1}, {0, Ntglobal}).RawPtr(), Ntglobal);
            for(auto idx : splitcolids[mpirank]){
                CopyData(Auptr, DataAu + accalongrow * nb + rowsumprefix[idx] * nb, 
                Rmat.GetElem(i, idx) * nb);
                Auptr += Rmat.GetElem(i,idx) * nb;
            }
            accalongrow += Rmat.RowSum()[i];
        }
        
        alpha = 1.0;
        beta = 0.0;

        bytesprocessed = TLRMVMBytesProcessed<float>(granksum, (size_t)nb, paddingM, paddingN);
        timestat.clear();
        LOOPSIZE = 5000;
        SKIPROUND = 1000;

    }

    void TearDown() {

        // FreeHostMemoryBatched(hAv, hx, hyv, hAvbp, hxbp, hyvbp);
        // FreeHostMemoryBatched(hAu, hyu, hy, hAubp, hyubp, hybp);
        // delete[] offsetphase2_h;

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
    void displayrankstat(){
        cout << "NB," << nb << " Total rank, " << granksum << " Bytesprocessed, " << bytesprocessed * 1e-6 << endl;
    }
    void displayavgstat(){
        vector<double> avgstat = getaveragestat();
        cout<< "Max Time: " << avgstat[2] << ", Min Time: " << avgstat[3] \
        << " Median Time: " << avgstat[0] << " us, " << "Median Bandwidth: " << avgstat[1] \
        << " GB/s." << endl;
    }

    // data pointers
    float *Datax;

    float *Av;
    float *x;
    float *yv;
    float **Avbatchpointer;
    float **xbatchpointer;
    float **yvbatchpointer;

    float *Au;
    float *yu;
    float *y;
    float *yfinal;
    float **Aubatchpointer;
    float **yubatchpointer;
    float **ybatchpointer;

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

    // mpi config
    int mpirank;
    int mpisize;

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
    int constrank;
    int nb;
    int originM;
    int originN;
    vector<Matrix<int>> Rmatsubset;
    vector<int> subcolsum;
    vector<int> subrowsum;
    vector<vector<int>> splitcolids;
    Matrix<int> Rmat;
    Matrix<int> Rtransmat;
    vector<int> Rmatprefix;
    vector<int> Rtransprefix;
    vector<int> prefixcolsum;

    size_t granksum;
    size_t lranksum;


};

TEST_F(AstronomyCPUTest, Phase1_Correctness){
    for(int i=0; i<Ntlocal; i++){
        cblasgemv(CblasColMajor, CblasNoTrans, subcolsum[i], nb, 
        alpha, Avbatchpointer[i], subcolsum[i], xbatchpointer[i], 1, beta, yvbatchpointer[i], 1);
    }
    AstronomyPCMatrix astropcmat(datafolder, acc, nb, id, originM, originN);
    Matrix<float> xvec(Datax, paddingN, 1);
    astropcmat.setX(xvec);
    Matrix<float> yv_pc = astropcmat.Phase1();
    vector<float> hyvresult(Rmatsubset[mpirank].Sum(),0);
    size_t acc = 0;
    // for(auto idx : splitcolids[mpirank]){
    //     double err = NetlibError(yv + acc, yv_pc.RawPtr() + prefixcolsum[idx], Rmat.ColSum()[idx]);
    //     GtestLog("Phase 1 error = " + to_string(err));  
    //     acc+=Rmat.ColSum()[idx];
    // }
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST_F(AstronomyCPUTest, Phase2_Correctness){
    timestat.clear();
    // #pragma omp parallel
    // {
    //     #pragma omp single
    //     cout << "omp threads" << omp_get_num_threads() << endl;
    // }
    vector<double> phase1time;
    vector<double> phase3time;
    for(int loopi=0; loopi < LOOPSIZE; loopi++){
	MPI_Barrier(MPI_COMM_WORLD);
        double t1 = gettime();
        #pragma omp parallel for
        for(int i=0; i<Ntlocal; i++){
            cblasgemv(CblasColMajor, CblasNoTrans, subcolsum[i], nb, 
            alpha, Avbatchpointer[i], subcolsum[i], xbatchpointer[i], 1, beta, yvbatchpointer[i], 1);
        }
        double tphase1 = gettime();
        #pragma omp parallel for 
        for(int i=0; i<lranksum; i++){
            yu[i] = yv[offsetphase2_h[i]];
        }
        double tphase3start = gettime();
        #pragma omp parallel for 
        for(int i=0; i<Mtglobal; i++){
            cblasgemv(CblasColMajor, CblasNoTrans, nb, subrowsum[i], 
            alpha, Aubatchpointer[i], nb, yubatchpointer[i], 1, beta, ybatchpointer[i], 1);
        }
        MPI_Reduce(y, yfinal, paddingM, mpi_get_type<float>(), MPI_SUM, 0, MPI_COMM_WORLD);
        double t2 = gettime();
        timestat.push_back(t2-t1);
        phase1time.push_back( (tphase1-t1) / (t2-t1));
        phase3time.push_back( (t2-tphase3start) / (t2-t1) );
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // if(mpirank == 0){
    //     AstronomyPCMatrix astropcmat(datafolder, acc, nb, id, originM, originN);
    //     Matrix<float> xvec(Datax, paddingN, 1);
    //     astropcmat.setX(xvec);
    //     Matrix<float> yv_pc = astropcmat.Phase1();
    //     Matrix<float> yu_pc = astropcmat.Phase2();
    //     Matrix<float> y_pc = astropcmat.Phase3();

    //     vector<float> hyresult(paddingM, 0);
    //     double err = NetlibError(yfinal, y_pc.RawPtr(),originM);
    //     GtestLog("error: " + to_string(err));
    // }

    vector<double> maxtime(timestat.size(), 0);
    for(int i=0; i<timestat.size(); i++) {
        MPI_Allreduce(timestat.data(), maxtime.data(), timestat.size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    }
    timestat = maxtime;
    if(mpirank == 0){
        double avgp1 = 0.0;
        double avgp3 = 0.0;
        for(int i=0; i<phase1time.size(); i++){
            avgp1 += phase1time[i];
            avgp3 += phase3time[i];
        }
        cout << "phase1 time " << avgp1 / phase1time.size() << endl;
        cout << "phase3 time " << avgp3 / phase3time.size() << endl;
        displayrankstat();
        displayavgstat();
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

}



TEST_F(AstronomyCPUTest, VITE){
    displayrankstat();
    return;
    timestat.clear();
    for(int loopi=0; loopi < 5; loopi++){
	MPI_Barrier(MPI_COMM_WORLD);
        double t1 = gettime();
        #pragma omp parallel for
        for(int i=0; i<Ntlocal; i++){
            cblasgemv(CblasColMajor, CblasNoTrans, subcolsum[i], nb, 
            alpha, Avbatchpointer[i], subcolsum[i], xbatchpointer[i], 1, beta, yvbatchpointer[i], 1);
        }
        #pragma omp parallel for 
        for(int i=0; i<lranksum; i++){
            yu[i] = yv[offsetphase2_h[i]];
        }
        #pragma omp parallel for 
        for(int i=0; i<Mtglobal; i++){
            cblasgemv(CblasColMajor, CblasNoTrans, nb, subrowsum[i], 
            alpha, Aubatchpointer[i], nb, yubatchpointer[i], 1, beta, ybatchpointer[i], 1);
        }
        MPI_Reduce(y, yfinal, paddingM, mpi_get_type<float>(), MPI_SUM, 0, MPI_COMM_WORLD);
        double t2 = gettime();
        timestat.push_back(t2-t1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // if(mpirank == 0){
    //     AstronomyPCMatrix astropcmat(datafolder, acc, nb, id, originM, originN);
    //     Matrix<float> xvec(Datax, paddingN, 1);
    //     astropcmat.setX(xvec);
    //     Matrix<float> yv_pc = astropcmat.Phase1();
    //     Matrix<float> yu_pc = astropcmat.Phase2();
    //     Matrix<float> y_pc = astropcmat.Phase3();

    //     vector<float> hyresult(paddingM, 0);
    //     double err = NetlibError(yfinal, y_pc.RawPtr(),originM);
    //     GtestLog("error: " + to_string(err));
    // }

    // vector<double> maxtime(timestat.size(), 0);
    // for(int i=0; i<timestat.size(); i++) {
    //     MPI_Reduce(timestat.data(), maxtime.data(), timestat.size(), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    // }
    // timestat = maxtime;
    // if(mpirank == 0)
    // displayavgstat();
    MPI_Barrier(MPI_COMM_WORLD);

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
    MPI_Init(NULL,NULL);
    int mpirank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    for(int i=0; i<argc-1; i++){
        char tmp[200];
        sprintf(tmp, "%s", argv[i+1]);
        command_line_arg_vec.push_back(string(tmp));
    }
    testing::AddGlobalTestEnvironment(new MyTestEnvironment(command_line_arg_vec));
    ::testing::TestEventListeners& listeners =
    ::testing::UnitTest::GetInstance()->listeners();
    if (mpirank != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }
    auto ret = RUN_ALL_TESTS();
    MPI_Finalize();
    return 0;
}
