
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

#define dtype complex<float>

class SeismicCPUTest: public testing::Test {

protected:

    void SetUp() {

        if(g_command_line_arg_vec.size() < 6){
            cout << "not enough args " << endl;
            exit(1);
        }
        datafolder = g_command_line_arg_vec[0];
        acc = g_command_line_arg_vec[1];
        int freqstart = atoi(g_command_line_arg_vec[2].c_str());
        int freqend = atoi(g_command_line_arg_vec[3].c_str());
        nb = atoi(g_command_line_arg_vec[4].c_str());
        originM = atoi(g_command_line_arg_vec[5].c_str());
        originN = atoi(g_command_line_arg_vec[6].c_str());
        VendorName = g_command_line_arg_vec[7];
        LLCsizethreshold = atoi(g_command_line_arg_vec[8].c_str());

        paddingM = CalculatePadding(originM, nb);
        paddingN = CalculatePadding(originN, nb);
        Mtglobal = paddingM / nb;
        Ntglobal = paddingN / nb;
        // load R
        unordered_map< int, Matrix<int> > RmatVec;
        for(int i=freqstart; i<freqend; i++){
            int *DataR;
            ReadSeismicBinary(datafolder, &DataR, Mtglobal * Ntglobal, acc, nb, i);
            Rmat = Matrix<int>(DataR, Mtglobal, Ntglobal);        
            RmatVec[i] = Rmat;
        }
        // MPI INIT
        MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
        if(mpirank == 0)
        GtestLog( "MPI world size = " + to_string(mpisize) );
        
        // start to decide which mpi process should process how many subset of mat.
        mpigroupsize = 1;
        mpigroupid = mpirank;
        mpigrouprank = 0;
        MPI_Comm_split(MPI_COMM_WORLD, mpigrouprank, mpigrouprank, &mpigroupcomm);
        Ntlocal = Ntglobal / mpigroupsize;

        // 1. use even split strategy 
        workfreqid.clear();
        int nfreq = freqend - freqstart;
        for(int i=0; i<nfreq; i++){
            if (i % mpisize == mpirank)
            workfreqid.push_back(i+freqstart);
        }
        workfreqcnt = workfreqid.size();

        // load subset of data
        dtype **DataAv, **DataAu, **Datax;
        DataAv = new dtype*[workfreqid.size()];
        DataAu = new dtype*[workfreqid.size()];
        Datax = new dtype*[workfreqid.size()];
        for(int i=0; i<workfreqid.size(); i++){
            granksum = RmatVec[workfreqid[i]].Sum();
            ReadSeismicBinary(datafolder+"/V", &DataAv[i], granksum * nb, acc ,nb, workfreqid[i]);
            ReadSeismicBinary(datafolder+"/U", &DataAu[i], granksum * nb, acc ,nb, workfreqid[i]);
            ReadSeismicBinaryX(datafolder, &Datax[i], originN, acc, nb, workfreqid[i]);
        }

        // inside each mpi group split each frequency matrix
        for(auto wid : workfreqid){
            auto splitcolids_sub = SplitColCount(RmatVec[wid], mpigroupsize);
            splitcolids[wid] = splitcolids_sub;
            auto Rmatsubset_sub = SplitInputRmatrix(RmatVec[wid], mpigroupsize);
            Rmatsubset[wid] = Rmatsubset_sub;
        }
        
        /**
         * Phase 1 memory preparation
         */
        // calculate layout for Av: Phase 1
        dtype val = dtype(0.0,0.0);
        Av = new dtype*[workfreqcnt];
        x = new dtype*[workfreqcnt];
        yv = new dtype*[workfreqcnt];
        Avbatchpointer = new dtype**[workfreqcnt];
        xbatchpointer = new dtype**[workfreqcnt];
        yvbatchpointer = new dtype**[workfreqcnt];
        int cnt = 0;
        for(auto wid : workfreqid){
            auto tmpRmat = Rmatsubset[wid];
            auto subcolsum = tmpRmat[mpigrouprank].ColSum();
            cout << subcolsum.size() << endl;
            subColSumVec.push_back(subcolsum);
            for(int i=0; i<Ntlocal; i++){
                AvMs[wid].push_back( (size_t)subcolsum[i] );
                AvKs[wid].push_back(nb);
                AvNs[wid].push_back(1);
            }
            cout << "subcolsum 0," << subcolsum[0] << endl;
            size_t tmpAvtotal, tmpxtotal, tmpyvtotal;
            GetHostMemoryBatched(&Av[cnt], &x[cnt], &yv[cnt], 
            &Avbatchpointer[cnt], &xbatchpointer[cnt], &yvbatchpointer[cnt], 
            AvMs[wid], AvKs[wid], AvNs[wid], val);
            CaluclateTotalElements(AvMs[wid], AvKs[wid], AvNs[wid], 
            tmpAvtotal, tmpxtotal, tmpyvtotal);
            Avtotalelems[wid] = tmpAvtotal;
            xtotalelems[wid] = tmpxtotal;
            yvtotalelems[wid] = tmpyvtotal;
            cnt++;
        }
        
        // copy Av and x to memory buffer
        cnt = 0;
        for(auto wid : workfreqid){
            auto curcolsum = Rmatsubset[wid][mpigrouprank];
            auto prefixcurcolsum = 
            PrefixSum(curcolsum.ColSum().data(), Ntglobal);
            for(int i=0; i<Ntlocal; i++){
                CopyData(x[cnt] + i*nb, Datax[cnt] + i * nb, nb);
                CopyData(Av[cnt] + prefixcurcolsum[i], DataAv[cnt] + prefixcurcolsum[i], 
                curcolsum.ColSum()[i]);
            }
            cnt++;
        }
        

        // get random X
        // Datax = new float[paddingN];
        // if(mpirank == 0){
        //     #pragma omp parallel for 
        //     for(int i=0; i<paddingN; i++){
        //         Datax[i] = (float)rand() / RAND_MAX;
        //         // Datax[i] = 1.0;
        //     }
        //     MPI_Bcast(Datax, paddingN, mpi_get_type<float>(), 0, MPI_COMM_WORLD);
        // }else{
        //     MPI_Bcast(Datax, paddingN, mpi_get_type<float>(), 0, MPI_COMM_WORLD);   
        // }

        // for(int i=0; i < splitcolids[mpirank].size(); i++){
        //     int idx = splitcolids[mpirank][i];
        //     // memcpy(x + i*nb, Datax + idx * nb, nb * sizeof(float));
        //     CopyData(x + i * nb, Datax + idx * nb, nb);
        // }
        // MPI_Barrier(MPI_COMM_WORLD);


        // auto globalrmatcolsum = Rmat.ColSum();
        // prefixcolsum = PrefixSum(globalrmatcolsum.data(), Rmat.Col());
        // float *Avptr = Av;
        // for(auto idx : splitcolids[mpirank]){
        //     CopyData(Avptr, DataAv + prefixcolsum[idx] * nb, globalrmatcolsum[idx] * nb);
        //     Avptr += globalrmatcolsum[idx] * nb;
        // }

        // // phase 2 preparation
        // tlrmvm::CalculatePhase2Offset(&offsetphase2_h, Mtglobal, Ntlocal, Rmatsubset[mpirank]);
        // lranksum = Rmatsubset[mpirank].Sum();
        // /**
        //  * Phase 3 preparation
        //  */
        // subrowsum = Rmatsubset[mpirank].RowSum();
        // for(int i=0; i<Mtglobal; i++){
        //     AuMs.push_back(nb);
        //     AuKs.push_back( (size_t)subrowsum[i] );
        //     AuNs.push_back(1);
        // }
        // GetHostMemoryBatched(&Au, &yu, &y, 
        // &Aubatchpointer, &yubatchpointer, &ybatchpointer, AuMs, AuKs, AuNs, val);
        // CaluclateTotalElements(AuMs, AuKs, AuNs, Autotalelems, yutotalelems, ytotalelems);
        // yfinal = new float[paddingM];
        // memset(yfinal, 0, sizeof(float)*paddingM );
        // size_t accalongrow = 0;
        // float * Auptr = Au;
        // for(size_t i=0; i<Mtglobal; i++){
        //     auto rowsumprefix = PrefixSum(Rmat.Block({i,i+1}, {0, Ntglobal}).RawPtr(), Ntglobal);
        //     for(auto idx : splitcolids[mpirank]){
        //         CopyData(Auptr, DataAu + accalongrow * nb + rowsumprefix[idx] * nb, 
        //         Rmat.GetElem(i, idx) * nb);
        //         Auptr += Rmat.GetElem(i,idx) * nb;
        //     }
        //     accalongrow += Rmat.RowSum()[i];
        // }
        
        alpha = complex<float>(1.0,0.0);
        beta = complex<float>(0.0,0.0);

        // bytesprocessed = TLRMVMBytesProcessed<float>(granksum, (size_t)nb, paddingM, paddingN);
        // timestat.clear();
        // LOOPSIZE = 5000;
        // SKIPROUND = 1000;

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
    complex<float> *Datax;

    complex<float> **Av;
    complex<float> **x;
    complex<float> **yv;
    complex<float> ***Avbatchpointer;
    complex<float> ***xbatchpointer;
    complex<float> ***yvbatchpointer;

    complex<float> **Au;
    complex<float> **yu;
    complex<float> **y;
    complex<float> **yfinal;
    complex<float> ***Aubatchpointer;
    complex<float> ***yubatchpointer;
    complex<float> ***ybatchpointer;

    unordered_map< int, vector<size_t> > AvMs;
    unordered_map< int, vector<size_t> > AvKs;
    unordered_map< int, vector<size_t> > AvNs;
    unordered_map< int, vector<size_t> > AuMs;
    unordered_map< int, vector<size_t> > AuKs;
    unordered_map< int, vector<size_t> > AuNs;

    unordered_map< int, size_t > Avtotalelems;
    unordered_map< int, size_t > xtotalelems;
    unordered_map< int, size_t > yvtotalelems;

    unordered_map< int, size_t > Autotalelems;
    unordered_map< int, size_t > yutotalelems;
    unordered_map< int, size_t > ytotalelems;

    vector< unsigned long int* > offsetphase2;
    vector< unsigned long int* > offsetphase2_h;

    complex<float> alpha;
    complex<float> beta;

    // mpi config
    int mpirank;
    int mpisize;
    int mpigrouprank;
    int mpigroupsize;
    int mpigroupid;
    MPI_Comm mpigroupcomm;

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
    vector<int> idvec;
    int nb;
    int originM;
    int originN;
    unordered_map<int,  vector<Matrix<int>> > Rmatsubset;
    vector< vector<int> > subColSumVec;
    vector< vector<int> > subrowsum;

    unordered_map<int, vector<vector<int>> > splitcolids;  // 
    Matrix<int> Rmat;
    Matrix<int> Rtransmat;
    vector<int> Rmatprefix;
    vector<int> Rtransprefix;
    vector<int> prefixcolsum;
    vector<int> workfreqid;
    int workfreqcnt;
    size_t granksum;
    size_t lranksum;
    string VendorName;
    int LLCsizethreshold;
    
};

TEST_F(SeismicCPUTest, Phase1_Correctness){
    timestat.clear();
    for(int workcnt = 0; workcnt < workfreqcnt; workcnt++){
        for(int i=0; i<Ntlocal; i++){
            cblasgemv(CblasColMajor, CblasNoTrans, subColSumVec[workcnt][i],
            nb, alpha, Avbatchpointer[workcnt][i], subColSumVec[workcnt][i],
            xbatchpointer[workcnt][i],1, beta, yvbatchpointer[workcnt][i],1);
        }
        auto tmpAv = Matrix<dtype>(Avbatchpointer[workcnt][0], subColSumVec[workcnt][0], nb);
        tmpAv.Tofile("freqid_" + to_string(workcnt) + "_AV.bin");
    }
    vector<Matrix<dtype>> yvvec;
    vector<Matrix<dtype>> Avvec;
    vector<Matrix<dtype>> xvec;
    for(int workcnt = 0; workcnt < workfreqcnt; workcnt++){
        SeismicPCMatrix seismicpcmat("/datawaha/ecrc/hongy0a/seismic/compressdata/", acc, nb, workfreqid[workcnt], originM, originN);
        auto xmat = seismicpcmat.GetX();
        xvec.push_back(xmat);
        auto yv = seismicpcmat.Phase1();
        yvvec.push_back(yv);
        Avvec.push_back(seismicpcmat.GetVTile(0,0));
        
    }
    for(int workcnt = 0; workcnt < workfreqcnt; workcnt++){
        // double err = NetlibError(xvec[workcnt].RawPtr(), x[workcnt], 10);
        cout << "Avvec " << yvvec[workcnt].RawPtr()[0] << ", " << yv[workcnt][0] << endl;
        // GtestLog("Phase 1 error = " + to_string(err));  
    }
    
    // for(auto idx : splitcolids[mpirank]){
    //     double err = NetlibError(yv + acc, yv_pc.RawPtr() + prefixcolsum[idx], Rmat.ColSum()[idx]);
    //     GtestLog("Phase 1 error = " + to_string(err));  
    //     acc+=Rmat.ColSum()[idx];
    // }
}

// TEST_F(AstronomyCPUTest, Phase1_Correctness){
//     for(int i=0; i<Ntlocal; i++){
//         cblasgemv(CblasColMajor, CblasNoTrans, subcolsum[i], nb, 
//         alpha, Avbatchpointer[i], subcolsum[i], xbatchpointer[i], 1, beta, yvbatchpointer[i], 1);
//     }
//     AstronomyPCMatrix astropcmat(datafolder, acc, nb, id, originM, originN);
//     Matrix<float> xvec(Datax, paddingN, 1);
//     astropcmat.setX(xvec);
//     Matrix<float> yv_pc = astropcmat.Phase1();
//     vector<float> hyvresult(Rmatsubset[mpirank].Sum(),0);
//     size_t acc = 0;
//     // for(auto idx : splitcolids[mpirank]){
//     //     double err = NetlibError(yv + acc, yv_pc.RawPtr() + prefixcolsum[idx], Rmat.ColSum()[idx]);
//     //     GtestLog("Phase 1 error = " + to_string(err));  
//     //     acc+=Rmat.ColSum()[idx];
//     // }
//     MPI_Barrier(MPI_COMM_WORLD);
// }


// TEST_F(AstronomyCPUTest, Phase2_Correctness){
//     timestat.clear();
//     // #pragma omp parallel
//     // {
//     //     #pragma omp single
//     //     cout << "omp threads" << omp_get_num_threads() << endl;
//     // }

//     for(int loopi=0; loopi < LOOPSIZE; loopi++){
// 	MPI_Barrier(MPI_COMM_WORLD);
//         double t1 = gettime();
//         #pragma omp parallel for
//         for(int i=0; i<Ntlocal; i++){
//             cblasgemv(CblasColMajor, CblasNoTrans, subcolsum[i], nb, 
//             alpha, Avbatchpointer[i], subcolsum[i], xbatchpointer[i], 1, beta, yvbatchpointer[i], 1);
//         }
//         #pragma omp parallel for 
//         for(int i=0; i<lranksum; i++){
//             yu[i] = yv[offsetphase2_h[i]];
//         }
//         #pragma omp parallel for 
//         for(int i=0; i<Mtglobal; i++){
//             cblasgemv(CblasColMajor, CblasNoTrans, nb, subrowsum[i], 
//             alpha, Aubatchpointer[i], nb, yubatchpointer[i], 1, beta, ybatchpointer[i], 1);
//         }
//         MPI_Reduce(y, yfinal, paddingM, mpi_get_type<float>(), MPI_SUM, 0, MPI_COMM_WORLD);
//         double t2 = gettime();
//         timestat.push_back(t2-t1);
//     }

//     MPI_Barrier(MPI_COMM_WORLD);
//     if(mpirank == 0){
//         AstronomyPCMatrix astropcmat(datafolder, acc, nb, id, originM, originN);
//         Matrix<float> xvec(Datax, paddingN, 1);
//         astropcmat.setX(xvec);
//         Matrix<float> yv_pc = astropcmat.Phase1();
//         Matrix<float> yu_pc = astropcmat.Phase2();
//         Matrix<float> y_pc = astropcmat.Phase3();

//         vector<float> hyresult(paddingM, 0);
//         double err = NetlibError(yfinal, y_pc.RawPtr(),originM);
//         GtestLog("error: " + to_string(err));
//     }

//     vector<double> maxtime(timestat.size(), 0);
//     for(int i=0; i<timestat.size(); i++) {
//         MPI_Allreduce(timestat.data(), maxtime.data(), timestat.size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//     }
//     timestat = maxtime;
//     if(mpirank == 0)
//     displayavgstat();
//     MPI_Barrier(MPI_COMM_WORLD);

// }

// TEST_F(AstronomyCPUTest, VITE){
//     displayrankstat();
//     return;
//     timestat.clear();
//     for(int loopi=0; loopi < 5; loopi++){
// 	MPI_Barrier(MPI_COMM_WORLD);
//         double t1 = gettime();
//         #pragma omp parallel for
//         for(int i=0; i<Ntlocal; i++){
//             cblasgemv(CblasColMajor, CblasNoTrans, subcolsum[i], nb, 
//             alpha, Avbatchpointer[i], subcolsum[i], xbatchpointer[i], 1, beta, yvbatchpointer[i], 1);
//         }
//         #pragma omp parallel for 
//         for(int i=0; i<lranksum; i++){
//             yu[i] = yv[offsetphase2_h[i]];
//         }
//         #pragma omp parallel for 
//         for(int i=0; i<Mtglobal; i++){
//             cblasgemv(CblasColMajor, CblasNoTrans, nb, subrowsum[i], 
//             alpha, Aubatchpointer[i], nb, yubatchpointer[i], 1, beta, ybatchpointer[i], 1);
//         }
//         MPI_Reduce(y, yfinal, paddingM, mpi_get_type<float>(), MPI_SUM, 0, MPI_COMM_WORLD);
//         double t2 = gettime();
//         timestat.push_back(t2-t1);
//     }

//     MPI_Barrier(MPI_COMM_WORLD);
//     // if(mpirank == 0){
//     //     AstronomyPCMatrix astropcmat(datafolder, acc, nb, id, originM, originN);
//     //     Matrix<float> xvec(Datax, paddingN, 1);
//     //     astropcmat.setX(xvec);
//     //     Matrix<float> yv_pc = astropcmat.Phase1();
//     //     Matrix<float> yu_pc = astropcmat.Phase2();
//     //     Matrix<float> y_pc = astropcmat.Phase3();

//     //     vector<float> hyresult(paddingM, 0);
//     //     double err = NetlibError(yfinal, y_pc.RawPtr(),originM);
//     //     GtestLog("error: " + to_string(err));
//     // }

//     // vector<double> maxtime(timestat.size(), 0);
//     // for(int i=0; i<timestat.size(); i++) {
//     //     MPI_Reduce(timestat.data(), maxtime.data(), timestat.size(), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
//     // }
//     // timestat = maxtime;
//     // if(mpirank == 0)
//     // displayavgstat();
//     MPI_Barrier(MPI_COMM_WORLD);

// }

// TEST_F(AstronomyCPUTest, Phase2_recordfulltime){
//     timestat.clear();
//     for(int loopi=0; loopi < LOOPSIZE; loopi++){
// 	MPI_Barrier(MPI_COMM_WORLD);
//         double t1 = gettime();
//         #pragma omp parallel for
//         for(int i=0; i<Ntlocal; i++){
//             cblasgemv(CblasColMajor, CblasNoTrans, subcolsum[i], nb, 
//             alpha, Avbatchpointer[i], subcolsum[i], xbatchpointer[i], 1, beta, yvbatchpointer[i], 1);
//         }
//         #pragma omp parallel for 
//         for(int i=0; i<lranksum; i++){
//             yu[i] = yv[offsetphase2_h[i]];
//         }
//         #pragma omp parallel for 
//         for(int i=0; i<Mtglobal; i++){
//             cblasgemv(CblasColMajor, CblasNoTrans, nb, subrowsum[i], 
//             alpha, Aubatchpointer[i], nb, yubatchpointer[i], 1, beta, ybatchpointer[i], 1);
//         }
//         MPI_Reduce(y, yfinal, paddingM, mpi_get_type<float>(), MPI_SUM, 0, MPI_COMM_WORLD);
//         double t2 = gettime();
//         timestat.push_back(t2-t1);
//     }

//     MPI_Barrier(MPI_COMM_WORLD);
//     // if(mpirank == 0){
//     //     AstronomyPCMatrix astropcmat(datafolder, acc, nb, id, originM, originN);
//     //     Matrix<float> xvec(Datax, paddingN, 1);
//     //     astropcmat.setX(xvec);
//     //     Matrix<float> yv_pc = astropcmat.Phase1();
//     //     Matrix<float> yu_pc = astropcmat.Phase2();
//     //     Matrix<float> y_pc = astropcmat.Phase3();

//     //     vector<float> hyresult(paddingM, 0);
//     //     double err = NetlibError(yfinal, y_pc.RawPtr(),originM);
//     //     GtestLog("error: " + to_string(err));
//     // }

//     vector<double> maxtime(timestat.size(), 0);
//     for(int i=0; i<timestat.size(); i++) {
//         MPI_Allreduce(timestat.data(), maxtime.data(), timestat.size(), MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
//     }
//     timestat = maxtime;
//     if(mpirank == 0){
//         Matrix<double> timemat(timestat, timestat.size(), 1);
//         timemat.Tofile("timemat.bin");
//         displayrankstat();
//         displayavgstat();

//     }
    
//     MPI_Barrier(MPI_COMM_WORLD);

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
