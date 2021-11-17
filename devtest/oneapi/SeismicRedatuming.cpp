
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

class SeismicRedatumingCPUTest: public testing::Test {

protected:

    void SetUp() {
        originM = argparser.getint("M");
        originN = argparser.getint("N");
        nb = argparser.getint("nb");
        acc = argparser.getstring("acc");
        seismicfreq = argparser.getint("seismicfreq");
        datafolder = argparser.getstring("datafolder");
        char rpath[100]; 
        sprintf(rpath, "%s/R-Mck_freqslice%d_nb%d_acc%s.bin", datafolder.c_str(), seismicfreq, nb, acc.c_str());
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
    int seismicfreq;
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


TEST_F(SeismicRedatumingCPUTest, onempiprocess){

    TlrmvmConfig tlrmvmconfig(originM, originN, nb, rankfile, datafolder, acc, seismicfreq);
    Matrix<int> maskmat(tlrmvmconfig.Mtg, tlrmvmconfig.Ntg);
    maskmat.Fill(1);
    tlrmvmconfig.Maskmat = maskmat;
    TlrmvmDPCPP_SeismicRedatuming tlrmvmptr(tlrmvmconfig);
    tlrmvmptr.MemoryInit();
    double bytes = TLRMVMBytesProcessed<float>(tlrmvmptr.granksum, tlrmvmptr.nb, tlrmvmptr.paddingM, tlrmvmptr.paddingN);
    for(int i=0; i<1; i++){
        auto start = std::chrono::steady_clock::now();
        tlrmvmptr.Phase1();
        tlrmvmptr.device_queue.wait_and_throw();
        tlrmvmptr.Phase2();
        tlrmvmptr.device_queue.wait_and_throw();
        tlrmvmptr.Phase3();
        tlrmvmptr.device_queue.wait_and_throw();
        auto end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        cout << "time end - start " << elapsed_time << ", BD " << bytes / elapsed_time * 1e-3 << endl;
    }
    SeismicPCMatrix seismicpcmat(datafolder, acc, nb, seismicfreq, originM, originN);
    seismicpcmat.LoadData();
    seismicpcmat.setX(tlrmvmptr.xmat);
    seismicpcmat.GetDense();
    Matrix<complex<float>> yv_pc = seismicpcmat.Phase1();
    auto hyv = Matrix<complex<float>>(tlrmvmptr.h_yv, tlrmvmptr.workmatgranksum, 1);
    cout << " Phase 1 Correctness : " << hyv.allclose(yv_pc) << endl;
    Matrix<complex<float>> yu_pc = seismicpcmat.Phase2();
    auto hyu = Matrix<complex<float>>(tlrmvmptr.h_yu, tlrmvmptr.workmatgranksum, 1);
    cout << " Phase 2 Correctness : " << hyu.allclose(yu_pc) << endl;
    Matrix<complex<float>> y_pc = seismicpcmat.Phase3();
    auto hy = Matrix<complex<float>>(tlrmvmptr.h_y, tlrmvmptr.paddingM, 1);
    cout << " Phase 3 Correctness : "<< hy.allclose(y_pc) << endl;

}



TEST_F(SeismicRedatumingCPUTest, twompiprocess){
    int mpirank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    TlrmvmConfig tlrmvmconfig(originM, originN, nb, rankfile, datafolder, acc, seismicfreq);
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
    TlrmvmDPCPP_SeismicRedatuming tlrmvmptr(tlrmvmconfig);
    double bytes = TLRMVMBytesProcessed<float>(tlrmvmptr.granksum, tlrmvmptr.nb, tlrmvmptr.paddingM, tlrmvmptr.paddingN);
    tlrmvmptr.MemoryInit();
    for(int i=0; i<1; i++){
        auto start = std::chrono::steady_clock::now();
        tlrmvmptr.Phase1();
        tlrmvmptr.device_queue.wait();
        tlrmvmptr.Phase2();
        tlrmvmptr.device_queue.wait();
        tlrmvmptr.Phase3();
        tlrmvmptr.device_queue.wait();
        MPI_Reduce(tlrmvmptr.h_y, tlrmvmptr.h_yout, tlrmvmptr.paddingM, MPI_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
        auto end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        timestat.push_back(elapsed_time*1e-6);
    }
    
    if(mpirank == 0){
        SeismicPCMatrix seismicpcmat(datafolder, acc, nb, seismicfreq, originM, originN);
        seismicpcmat.LoadData();
        seismicpcmat.setX(tlrmvmptr.xmat);
        seismicpcmat.GetDense();
        Matrix<complex<float>> yv_pc = seismicpcmat.Phase1();
        auto hyv = Matrix<complex<float>>(tlrmvmptr.h_yv, tlrmvmptr.workmatgranksum, 1);
        // cout << " Phase 1 Correctness : " << hyv.allclose(yv_pc) << endl;
        Matrix<complex<float>> yu_pc = seismicpcmat.Phase2();
        auto hyu = Matrix<complex<float>>(tlrmvmptr.h_yu, tlrmvmptr.workmatgranksum, 1);
        // cout << " Phase 2 Correctness : " << hyu.allclose(yu_pc) << endl;
        Matrix<complex<float>> y_pc = seismicpcmat.Phase3();
        auto hy = Matrix<complex<float>>(tlrmvmptr.h_yout, tlrmvmptr.paddingM, 1);
        cout << " Phase 3 Correctness : "<< hy.allclose(y_pc) << endl;
    }
}


vector<int64_t> Convert2Int64(vector<size_t> array){
    vector<int64_t> res;
    for(auto x : array) res.push_back((int64_t)x);
    return res;
}

vector<int> Convert2Int(vector<size_t> array){
    vector<int> res;
    for(auto x : array) res.push_back((int)x);
    return res;
}

TEST_F(SeismicRedatumingCPUTest, twompiprocess_oneapi){
    int mpirank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    TlrmvmConfig tlrmvmconfig(originM, originN, nb, rankfile, datafolder, acc, seismicfreq);
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
    TlrmvmDPCPP_SeismicRedatuming* tlrmvmptr = new TlrmvmDPCPP_SeismicRedatuming(tlrmvmconfig);
    double bytes = TLRMVMBytesProcessed<float>(tlrmvmptr->granksum, tlrmvmptr->nb, tlrmvmptr->paddingM, tlrmvmptr->paddingN);
    tlrmvmptr->MemoryInit();
    complex<float> alpha = complex<float>(1.0,0.0);
    complex<float> beta = complex<float>(0.0,0.0);
    auto transA = oneapi::mkl::transpose::nontrans;
    sycl::range<1>num_items{tlrmvmptr->workmatgranksum};
    cout << tlrmvmptr->granksum << endl;
    // build input gemvbatch
    
    // vector<int64_t> m_array = Convert2Int64(tlrmvmptr->AvMs);
    // vector<int64_t> n_array = Convert2Int64(tlrmvmptr->AvKs);
    // vector<oneapi::mkl::transpose> trans_array;
    // for(auto x : m_array) trans_array.push_back(oneapi::mkl::transpose::N);
    // vector<complex<float>> alpha_array(m_array.size(), 1.0);
    // const complex<float>** a_array = (const complex<float>**)tlrmvmptr->h_Avbp;
    // vector<int64_t> lda_array; for(auto x : m_array) lda_array.push_back(x);
    // const complex<float>** x_array = (const complex<float>**)tlrmvmptr->h_xbp;
    // vector<int64_t> incx_array(m_array.size(), 1);
    // vector<complex<float>> beta_array(m_array.size(), 0.0);
    // complex<float>** y_array = tlrmvmptr->h_yvbp;
    // vector<int64_t> incy_array(m_array.size(), 1);
    // int group_count = m_array.size();
    // vector<int64_t> group_size_array(m_array.size(), 1);

    // prepare cblas gemv batch input
    vector<int> m_array = Convert2Int(tlrmvmptr->AvMs);
    vector<int> n_array = Convert2Int(tlrmvmptr->AvKs);
    vector<CBLAS_TRANSPOSE> trans_array;
    for(auto x : m_array) trans_array.push_back(CBLAS_TRANSPOSE::CblasNoTrans);
    vector<complex<float>> alpha_array(m_array.size(), 1.0);
    const complex<float>** a_array = (const complex<float>**)tlrmvmptr->h_Avbp;
    vector<int> lda_array; for(auto x : m_array) lda_array.push_back(x);
    const complex<float>** x_array = (const complex<float>**)tlrmvmptr->h_xbp;
    vector<int> incx_array(m_array.size(), 1);
    vector<complex<float>> beta_array(m_array.size(), 0.0);
    complex<float>** y_array = tlrmvmptr->h_yvbp;
    vector<int> incy_array(m_array.size(), 1);
    int group_count = m_array.size();
    vector<int> group_size_array(m_array.size(), 1);
    for(int i=0; i<14000; i++){
        auto start = std::chrono::steady_clock::now();
        // phase 1
        // for(int i=0; i<tlrmvmptr->Ntg; i++){
        //     if(tlrmvmptr->colsum[i] != 0){
        //         // cblas_cgemv(CblasColMajor, CblasNoTrans, tlrmvmptr->AvMs[i],
        //         // tlrmvmptr->AvKs[i], alpha, tlrmvmptr->h_Avbp[i], 
        //         // tlrmvmptr->AvMs[i], tlrmvmptr->h_xbp[i],
        //         // 1, beta, tlrmvmptr->h_yvbp[i], 1);
        //         oneapi::mkl::blas::gemv(tlrmvmptr->device_queue, 
        //         transA, tlrmvmptr->AvMs[i],
        //         tlrmvmptr->AvKs[i], alpha, tlrmvmptr->h_Avbp[i], 
        //         tlrmvmptr->AvMs[i], tlrmvmptr->h_xbp[i],
        //         1, beta, tlrmvmptr->h_yvbp[i], 1);
        //     }
        // }
        for(int i=0; i<tlrmvmptr->Ntg; i++){
            if(tlrmvmptr->colsum[i] != 0){
                oneapi::mkl::blas::gemv(tlrmvmptr->device_queue, 
                transA, tlrmvmptr->AvMs[i],
                tlrmvmptr->AvKs[i], alpha, tlrmvmptr->h_Avbp[i], 
                tlrmvmptr->AvMs[i], tlrmvmptr->h_xbp[i],
                1, beta, tlrmvmptr->h_yvbp[i], 1);
            }
        }

        // cblas_cgemv_batch(CblasColMajor, trans_array.data(), m_array.data(), n_array.data(), alpha_array.data(), 
        // (const void**)a_array,lda_array.data(), (const void**)x_array, incx_array.data(),
        // beta_array.data(), (void**)y_array, incy_array.data(), group_count, group_size_array.data()); 

        tlrmvmptr->device_queue.wait();
        // phase 2
        tlrmvmptr->device_queue.parallel_for(num_items, [tlrmvmptr](int i){tlrmvmptr->h_yu[tlrmvmptr->h_phase2mapping[i]] = tlrmvmptr->h_yv[i];});
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
        MPI_Reduce(tlrmvmptr->h_y, tlrmvmptr->h_yout, tlrmvmptr->paddingM, MPI_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
        auto end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        timestat.push_back(elapsed_time*1e-6);
    }
    
    if(mpirank == 0){
        SeismicPCMatrix seismicpcmat(datafolder, acc, nb, seismicfreq, originM, originN);
        seismicpcmat.LoadData();
        seismicpcmat.setX(tlrmvmptr->xmat);
        seismicpcmat.GetDense();
        Matrix<complex<float>> yv_pc = seismicpcmat.Phase1();
        auto hyv = Matrix<complex<float>>(tlrmvmptr->h_yv, tlrmvmptr->workmatgranksum, 1);
        // cout << " Phase 1 Correctness : " << hyv.allclose(yv_pc) << endl;
        Matrix<complex<float>> yu_pc = seismicpcmat.Phase2();
        auto hyu = Matrix<complex<float>>(tlrmvmptr->h_yu, tlrmvmptr->workmatgranksum, 1);
        // cout << " Phase 2 Correctness : " << hyu.allclose(yu_pc) << endl;
        Matrix<complex<float>> y_pc = seismicpcmat.Phase3();
        auto hy = Matrix<complex<float>>(tlrmvmptr->h_yout, tlrmvmptr->paddingM, 1);
        cout << " Phase 3 Correctness : "<< hy.allclose(y_pc) << endl;
        string filename;
        char filechar[1000];
        sprintf(filechar, "seismic_oneapi%d_nb%d_freq%d_acc%s.bin", numthreads * 2, nb , seismicfreq, acc.c_str());
        SaveTimeandBandwidth<complex<float>>(string(filechar), timestat, granksum, nb, originM, originN);
        granksum = tlrmvmptr->granksum;
        bytesprocessed = TLRMVMBytesProcessed<complex<float>>(tlrmvmptr->granksum, tlrmvmptr->nb, tlrmvmptr->paddingM, tlrmvmptr->paddingN);
        getaveragestat();
        displayavgstat();
    }
}

TEST_F(SeismicRedatumingCPUTest, twompiprocess_openmp){
    int mpirank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    TlrmvmConfig tlrmvmconfig(originM, originN, nb, rankfile, datafolder, acc, seismicfreq);
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
    TlrmvmDPCPP_SeismicRedatuming* tlrmvmptr = new TlrmvmDPCPP_SeismicRedatuming(tlrmvmconfig);
    double bytes = TLRMVMBytesProcessed<float>(tlrmvmptr->granksum, tlrmvmptr->nb, tlrmvmptr->paddingM, tlrmvmptr->paddingN);
    tlrmvmptr->MemoryInit();
    complex<float> alpha = 1.0;
    complex<float> beta = 0.0;
    auto transA = oneapi::mkl::transpose::nontrans;
    sycl::range<1>num_items{tlrmvmptr->workmatgranksum};
    
    for(int i=0; i<2000; i++){
        auto start = std::chrono::steady_clock::now();
        // phase 1
        #pragma omp parallel for 
        for(int i=0; i<tlrmvmptr->Ntg; i++){
            if(tlrmvmptr->colsum[i] != 0){
                cblas_cgemv(CblasColMajor, CblasNoTrans, tlrmvmptr->AvMs[i],
                tlrmvmptr->AvKs[i], &alpha, tlrmvmptr->h_Avbp[i], 
                tlrmvmptr->AvMs[i], tlrmvmptr->h_xbp[i],
                1, &beta, tlrmvmptr->h_yvbp[i], 1);
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
                cblas_cgemv(CblasColMajor, CblasNoTrans, tlrmvmptr->AuMs[i],
                tlrmvmptr->AuKs[i], &alpha, tlrmvmptr->h_Aubp[i], 
                tlrmvmptr->AuMs[i], tlrmvmptr->h_yubp[i],
                1, &beta, tlrmvmptr->h_ybp[i], 1);
            }
        }
        MPI_Reduce(tlrmvmptr->h_y, tlrmvmptr->h_yout, tlrmvmptr->paddingM, MPI_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
        auto end = std::chrono::steady_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        timestat.push_back(elapsed_time*1e-6);
        // bandwidth.push_back(bytes / elapsed_time * 1e-3);
    }
    
    if(mpirank == 0){
        SeismicPCMatrix seismicpcmat(datafolder, acc, nb, seismicfreq, originM, originN);
        seismicpcmat.LoadData();
        seismicpcmat.setX(tlrmvmptr->xmat);
        seismicpcmat.GetDense();
        string filename;
        char filechar[1000];
        int ompnumthreads;
        #pragma omp parallel
        {
            #pragma omp single
            ompnumthreads = omp_get_num_threads();
        }
        sprintf(filechar, "seismic_openmp%d_nb%d_freq%d_acc%s.bin", ompnumthreads*2, nb , seismicfreq, acc.c_str());
        SaveTimeandBandwidth<complex<float>>(string(filechar), timestat, granksum, nb, originM, originN);
        granksum = tlrmvmptr->granksum;
        bytesprocessed = TLRMVMBytesProcessed<complex<float>>(tlrmvmptr->granksum, tlrmvmptr->nb, tlrmvmptr->paddingM, tlrmvmptr->paddingN);
        getaveragestat();
        displayavgstat();
        Matrix<complex<float>> yv_pc = seismicpcmat.Phase1();
        // auto hyv = Matrix<complex<float>>(tlrmvmptr->h_yv, tlrmvmptr->workmatgranksum, 1);
        // cout << " Phase 1 Correctness : " << hyv.allclose(yv_pc) << endl;
        Matrix<complex<float>> yu_pc = seismicpcmat.Phase2();
        // auto hyu = Matrix<complex<float>>(tlrmvmptr->h_yu, tlrmvmptr->workmatgranksum, 1);
        // cout << " Phase 2 Correctness : " << hyu.allclose(yu_pc) << endl;
        Matrix<complex<float>> y_pc = seismicpcmat.Phase3();
        auto hy = Matrix<complex<float>>(tlrmvmptr->h_yout, tlrmvmptr->paddingM, 1);
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
