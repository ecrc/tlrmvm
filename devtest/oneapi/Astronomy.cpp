
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
#include <chrono>

#include "common/Common.h"
#include "common/AppUtil.h"
#include "tlrmvm/Tlrmvm.h"
#include <oneapi/tbb.h>
#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "omp.h"

#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>

using namespace std;
using std::endl;
using namespace tlrmat;
using namespace tlrmvm;
using ::testing::Pointwise;
using ::testing::NanSensitiveFloatNear;

ArgsParser argparser;

class AstronomyCPUTest: public testing::Test {

protected:

    void SetUp() {
        originM = argparser.getint("M");
        originN = argparser.getint("N");
        nb = argparser.getint("nb");
        acc = argparser.getstring("acc");
        mavisid = argparser.getstring("mavisid");
        datafolder = argparser.getstring("datafolder");
        char rpath[100]; 
        sprintf(rpath, "%s/R_id%s_nb%d_acc%s.bin", datafolder.c_str(), mavisid.c_str(), nb, acc.c_str());
        rankfile = string(rpath);
        numthreads = argparser.getint("ompthreads");
    }

    void TearDown() {}
    int originM;
    int originN;
    int nb;
    int numthreads;
    string acc;
    string datafolder;
    string mavisid;
    string rankfile;
    string Ufile;
    string Vfile;
    vector<double> timestat;
    vector<double> bandstat;
    double bytesprocessed;
    size_t granksum;
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

};


TEST_F(AstronomyCPUTest, onempiprocess){

    TlrmvmConfig tlrmvmconfig(originM, originN, nb, rankfile, datafolder, acc, mavisid);
    Matrix<int> maskmat(tlrmvmconfig.Mtg, tlrmvmconfig.Ntg);
    maskmat.Fill(1);
    tlrmvmconfig.Maskmat = maskmat;
    TlrmvmDPCPP_Astronomy tlrmvmptr(tlrmvmconfig);
    tlrmvmptr.MemoryInit();
    double bytes = TLRMVMBytesProcessed<float>(tlrmvmptr.granksum, tlrmvmptr.nb, tlrmvmptr.paddingM, tlrmvmptr.paddingN);
    for(int i=0; i<1000; i++){
        auto start = std::chrono::steady_clock::now();
        tlrmvmptr.Phase1();
        tlrmvmptr.device_queue.wait_and_throw();
        tlrmvmptr.Phase2();
        tlrmvmptr.Phase3();
        tlrmvmptr.device_queue.wait_and_throw();
        auto end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        cout << "time end - start " << elapsed_time << ", BD " << bytes / elapsed_time * 1e-3 << endl;
    }
    AstronomyPCMatrix astropcmat(datafolder, acc, nb, mavisid, originM, originN);
    astropcmat.LoadData();
    astropcmat.setX(tlrmvmptr.xmat);
    astropcmat.GetDense();
    Matrix<float> yv_pc = astropcmat.Phase1();
    auto hyv = Matrix<float>(tlrmvmptr.h_yv, tlrmvmptr.workmatgranksum, 1);
    cout << " Phase 1 Correctness : " << hyv.allclose(yv_pc) << endl;
    Matrix<float> yu_pc = astropcmat.Phase2();
    auto hyu = Matrix<float>(tlrmvmptr.h_yu, tlrmvmptr.workmatgranksum, 1);
    cout << " Phase 2 Correctness : " << hyu.allclose(yu_pc) << endl;
    Matrix<float> y_pc = astropcmat.Phase3();
    auto hy = Matrix<float>(tlrmvmptr.h_y, tlrmvmptr.paddingM, 1);
    cout << " Phase 3 Correctness : "<< hy.allclose(y_pc) << endl;

}



TEST_F(AstronomyCPUTest, twompiprocess){
    int mpirank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    TlrmvmConfig tlrmvmconfig(originM, originN, nb, rankfile, datafolder, acc, mavisid);
    Matrix<int> maskmat(tlrmvmconfig.Mtg, tlrmvmconfig.Ntg);
    maskmat.Fill(0);
    for(int i=0; i<tlrmvmconfig.Mtg; i++){
        for(int j=0; j<tlrmvmconfig.Ntg; j++){
            if( j % 2 == mpirank ){
                maskmat.SetElem(i, j, 1);
            }
        }
    }
    tlrmvmconfig.Maskmat = maskmat;
    TlrmvmDPCPP_Astronomy tlrmvmptr(tlrmvmconfig);
    double bytes = TLRMVMBytesProcessed<float>(tlrmvmptr.granksum, tlrmvmptr.nb, tlrmvmptr.paddingM, tlrmvmptr.paddingN);
    tlrmvmptr.MemoryInit();
    for(int i=0; i<10000; i++){
        auto start = std::chrono::steady_clock::now();
        tlrmvmptr.Phase1();
        tlrmvmptr.device_queue.wait();
        tlrmvmptr.Phase2();
        tlrmvmptr.device_queue.wait();
        tlrmvmptr.Phase3();
        tlrmvmptr.device_queue.wait();
        MPI_Reduce(tlrmvmptr.h_y, tlrmvmptr.h_yout, tlrmvmptr.paddingM, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        auto end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        timestat.push_back(elapsed_time*1e-6);
    }
    
    if(mpirank == 0){
        AstronomyPCMatrix astropcmat(datafolder, acc, nb, mavisid, originM, originN);
        astropcmat.LoadData();
        astropcmat.setX(tlrmvmptr.xmat);
        astropcmat.GetDense();
        granksum = tlrmvmptr.granksum;
        bytesprocessed = TLRMVMBytesProcessed<float>(tlrmvmptr.granksum, tlrmvmptr.nb, tlrmvmptr.paddingM, tlrmvmptr.paddingN);
        getaveragestat();
        displayavgstat();
        Matrix<float> yv_pc = astropcmat.Phase1();
        // auto hyv = Matrix<float>(tlrmvmptr.h_yv, tlrmvmptr.workmatgranksum, 1);
        // cout << " Phase 1 Correctness : " << hyv.allclose(yv_pc) << endl;
        Matrix<float> yu_pc = astropcmat.Phase2();
        // auto hyu = Matrix<float>(tlrmvmptr.h_yu, tlrmvmptr.workmatgranksum, 1);
        // cout << " Phase 2 Correctness : " << hyu.allclose(yu_pc) << endl;
        Matrix<float> y_pc = astropcmat.Phase3();
        auto hy = Matrix<float>(tlrmvmptr.h_yout, tlrmvmptr.paddingM, 1);
        cout << " Phase 3 Correctness : "<< hy.allclose(y_pc) << endl;
    }
}


TEST_F(AstronomyCPUTest, twompiprocess_oneapi){
    int mpirank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    TlrmvmConfig tlrmvmconfig(originM, originN, nb, rankfile, datafolder, acc, mavisid);
    Matrix<int> maskmat(tlrmvmconfig.Mtg, tlrmvmconfig.Ntg);
    maskmat.Fill(0);
    
    for(int i=0; i<tlrmvmconfig.Mtg; i++){
        for(int j=0; j<tlrmvmconfig.Ntg; j++){
            if( j % 2 == mpirank ){
                maskmat.SetElem(i, j, 1);
            }
        }
    }
    tlrmvmconfig.Maskmat = maskmat;
    TlrmvmDPCPP_Astronomy *tlrmvmptr = new TlrmvmDPCPP_Astronomy(tlrmvmconfig);
    double bytes = TLRMVMBytesProcessed<float>(tlrmvmptr->granksum, tlrmvmptr->nb, tlrmvmptr->paddingM, tlrmvmptr->paddingN);
    tlrmvmptr->MemoryInit();
    float alpha = 1.0;
    float beta = 0.0;
    auto transA = oneapi::mkl::transpose::nontrans;
    sycl::range<1>num_items{tlrmvmptr->workmatgranksum};
    auto device = sycl::device(sycl::cpu_selector());
    tlrmvmptr->device_queue = sycl::queue(device);
    // tlrmvmptr->device_queue = sycl::queue(device);
    tlrmvmptr->device_queue.wait();
    for(int i=0; i<10000; i++){
        auto start = std::chrono::steady_clock::now();
        // phase 1
        for(int i=0; i<tlrmvmptr->Ntg; i++){
            if(tlrmvmptr->colsum[i] != 0){
                oneapi::mkl::blas::gemv(tlrmvmptr->device_queue, 
                transA, tlrmvmptr->AvMs[i],
                tlrmvmptr->AvKs[i], alpha, tlrmvmptr->h_Avbp[i], 
                tlrmvmptr->AvMs[i], tlrmvmptr->h_xbp[i],
                1, beta, tlrmvmptr->h_yvbp[i], 1);
            }
        }
        tlrmvmptr->device_queue.wait();
        
        // phase 2
        tlrmvmptr->device_queue.parallel_for(num_items, 
        [tlrmvmptr](int i){tlrmvmptr->h_yu[tlrmvmptr->h_phase2mapping[i]] = tlrmvmptr->h_yv[i];});
        tlrmvmptr->device_queue.wait();

        // phase 3
        for(int i=0; i<tlrmvmptr->Mtg; i++){
            if(tlrmvmptr->rowsum[i] != 0){
                oneapi::mkl::blas::gemv(tlrmvmptr->device_queue, 
                transA, tlrmvmptr->AuMs[i],
                tlrmvmptr->AuKs[i], alpha, tlrmvmptr->h_Aubp[i], 
                tlrmvmptr->AuMs[i], tlrmvmptr->h_yubp[i],
                1, beta, tlrmvmptr->h_ybp[i], 1);
            }
        }
        tlrmvmptr->device_queue.wait();
        MPI_Reduce(tlrmvmptr->h_y, tlrmvmptr->h_yout, tlrmvmptr->paddingM, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        auto end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        timestat.push_back(elapsed_time*1e-6);
    }
    
    if(mpirank == 0){
        AstronomyPCMatrix astropcmat(datafolder, acc, nb, mavisid, originM, originN);
        astropcmat.LoadData();
        astropcmat.setX(tlrmvmptr->xmat);
        astropcmat.GetDense();
        string filename;
        char filechar[1000];
        sprintf(filechar, "astronomy_oneapi%d_nb%d_mavisid%s_acc%s.bin", 2*numthreads, nb , mavisid.c_str(), acc.c_str());
        SaveTimeandBandwidth<float>(string(filechar), timestat, granksum, nb, originM, originN);
        granksum = tlrmvmptr->granksum;
        bytesprocessed = TLRMVMBytesProcessed<float>(tlrmvmptr->granksum, tlrmvmptr->nb, tlrmvmptr->paddingM, tlrmvmptr->paddingN);
        getaveragestat();
        displayavgstat();
        Matrix<float> yv_pc = astropcmat.Phase1();
        // auto hyv = Matrix<float>(tlrmvmptr->h_yv, tlrmvmptr->workmatgranksum, 1);
        // cout << " Phase 1 Correctness : " << hyv.allclose(yv_pc) << endl;
        Matrix<float> yu_pc = astropcmat.Phase2();
        // auto hyu = Matrix<float>(tlrmvmptr->h_yu, tlrmvmptr->workmatgranksum, 1);
        // cout << " Phase 2 Correctness : " << hyu.allclose(yu_pc) << endl;
        Matrix<float> y_pc = astropcmat.Phase3();
        auto hy = Matrix<float>(tlrmvmptr->h_yout, tlrmvmptr->paddingM, 1);
        cout << " Phase 3 Correctness : "<< hy.allclose(y_pc) << endl;
    }
}





TEST_F(AstronomyCPUTest, twompiprocess_openmp){
    int mpirank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    TlrmvmConfig tlrmvmconfig(originM, originN, nb, rankfile, datafolder, acc, mavisid);
    Matrix<int> maskmat(tlrmvmconfig.Mtg, tlrmvmconfig.Ntg);
    maskmat.Fill(0);
    
    for(int i=0; i<tlrmvmconfig.Mtg; i++){
        for(int j=0; j<tlrmvmconfig.Ntg; j++){
            if( j % 2 == mpirank ){
                maskmat.SetElem(i, j, 1);
            }
        }
    }
    // cout << "rank " << mpirank  << endl << " " << maskmat.Block({0,10},{0,10}) << endl;
    // return;
    tlrmvmconfig.Maskmat = maskmat;
    TlrmvmDPCPP_Astronomy *tlrmvmptr = new TlrmvmDPCPP_Astronomy(tlrmvmconfig);
    cout << tlrmvmptr->granksum << endl;
    double bytes = TLRMVMBytesProcessed<float>(tlrmvmptr->granksum, tlrmvmptr->nb, tlrmvmptr->paddingM, tlrmvmptr->paddingN);
    tlrmvmptr->MemoryInit();
    float alpha = 1.0;
    float beta = 0.0;
    auto transA = oneapi::mkl::transpose::nontrans;
    sycl::range<1>num_items{tlrmvmptr->workmatgranksum};
    
    for(int i=0; i<2000; i++){
        auto start = std::chrono::steady_clock::now();
        // phase 1
        #pragma omp parallel for 
        for(int i=0; i<tlrmvmptr->Ntg; i++){
            if(tlrmvmptr->colsum[i] != 0){
                cblas_sgemv(CblasColMajor, CblasNoTrans, tlrmvmptr->AvMs[i],
                tlrmvmptr->AvKs[i], alpha, tlrmvmptr->h_Avbp[i], 
                tlrmvmptr->AvMs[i], tlrmvmptr->h_xbp[i],
                1, beta, tlrmvmptr->h_yvbp[i], 1);
            }
        }
        // phase 2
        #pragma omp parallel for 
        for(int i=0; i<tlrmvmptr->workmatgranksum; i++){
            tlrmvmptr->h_yu[tlrmvmptr->h_phase2mapping[i]] = tlrmvmptr->h_yv[i];
        }
        // phase 3
        #pragma omp parallel for 
        for(int i=0; i<tlrmvmptr->Mtg; i++){
            if(tlrmvmptr->rowsum[i] != 0){
                cblas_sgemv(CblasColMajor, CblasNoTrans, tlrmvmptr->AuMs[i],
                tlrmvmptr->AuKs[i], alpha, tlrmvmptr->h_Aubp[i], 
                tlrmvmptr->AuMs[i], tlrmvmptr->h_yubp[i],
                1, beta, tlrmvmptr->h_ybp[i], 1);
            }
        }
        MPI_Reduce(tlrmvmptr->h_y, tlrmvmptr->h_yout, tlrmvmptr->paddingM, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        auto end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        timestat.push_back(elapsed_time*1e-6);
        // bandwidth.push_back(bytes / elapsed_time * 1e-3);
    }
    
    if(mpirank == 0){
        AstronomyPCMatrix astropcmat(datafolder, acc, nb, mavisid, originM, originN);
        astropcmat.LoadData();
        astropcmat.setX(tlrmvmptr->xmat);
        astropcmat.GetDense();
        int ompnumthreads;
        #pragma omp parallel
        {
            #pragma omp single
            ompnumthreads = omp_get_num_threads();
        }
        string filename;
        char filechar[1000];
        sprintf(filechar, "astronomy_openmp%d_nb%d_mavisid%s_acc%s.bin",2*ompnumthreads, nb , mavisid.c_str(), acc.c_str());
        SaveTimeandBandwidth<float>(string(filechar), timestat, granksum, nb, originM, originN);
        granksum = tlrmvmptr->granksum;
        bytesprocessed = TLRMVMBytesProcessed<float>(tlrmvmptr->granksum, tlrmvmptr->nb, tlrmvmptr->paddingM, tlrmvmptr->paddingN);
        getaveragestat();
        displayavgstat();
        Matrix<float> yv_pc = astropcmat.Phase1();
        // auto hyv = Matrix<float>(tlrmvmptr->h_yv, tlrmvmptr->workmatgranksum, 1);
        // cout << " Phase 1 Correctness : " << hyv.allclose(yv_pc) << endl;
        Matrix<float> yu_pc = astropcmat.Phase2();
        // auto hyu = Matrix<float>(tlrmvmptr->h_yu, tlrmvmptr->workmatgranksum, 1);
        // cout << " Phase 2 Correctness : " << hyu.allclose(yu_pc) << endl;
        Matrix<float> y_pc = astropcmat.Phase3();
        auto hy = Matrix<float>(tlrmvmptr->h_yout, tlrmvmptr->paddingM, 1);
        cout << " Phase 3 Correctness : "<< hy.allclose(y_pc) << endl;
    }

}


int main(int argc, char **argv) {
    MPI_Init(NULL,NULL);
    int mpirank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    testing::InitGoogleTest(&argc, argv);
    argparser = ArgsParser(argc, argv);
    ::testing::TestEventListeners& listeners =
    ::testing::UnitTest::GetInstance()->listeners();
    if (mpirank != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }
    auto ret = RUN_ALL_TESTS();
    MPI_Finalize();

    return 0;
}
