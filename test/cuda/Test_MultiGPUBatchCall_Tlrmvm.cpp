//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#include <iostream>
#include <unistd.h>
#include <memory.h>
#include <mpi.h>
#include "common/Common.hpp"
#include "tlrmvm/Tlrmvm.hpp"
#include <algorithm>
using namespace cudatlrmvm;


template<typename T>
void CheckTLRMVM(shared_ptr<TlrmvmCPU<T>> &cpuinst, string datafolder,string threshold,
                 int nb, string problem, int originM, int originN, int ci, bool conjugate, bool transpose)
{
    CFPPCMatrix seismic(datafolder, threshold, nb, problem, originM, originN);
    if(conjugate){
        auto xmat = cpuinst->xmat;
        auto cjxmat = xmat.Conjugate();
        seismic.setX(cjxmat);
    }else{
        seismic.setX(cpuinst->xmat);
    }
    seismic.GetDense();
    cout << "freqid " << ci << endl;
    Matrix<complex<float>> yv_pc; Matrix<complex<float>> yu_pc; Matrix<complex<float>> y_pc;
    Matrix<complex<float>> hyv; Matrix<complex<float>> hyu; Matrix<complex<float>> hy;
    if(transpose) yv_pc = seismic.Phase1Transpose();
    else yv_pc = seismic.Phase1();
    if(transpose) hyv = Matrix<complex<float>>(cpuinst->p1transptrs.y,cpuinst->config.workmatgranksum, 1);
    else hyv = Matrix<complex<float>>(cpuinst->p1ptrs.y,cpuinst->config.workmatgranksum, 1);
    cout << " Phase 1 Correctness : " << hyv.allclose(yv_pc) << endl;

    if(transpose) yu_pc = seismic.Phase2Transpose();
    else yu_pc = seismic.Phase2();
    if(transpose) hyu = Matrix<complex<float>>(cpuinst->p3transptrs.x,cpuinst->config.workmatgranksum, 1);
    else hyu = Matrix<complex<float>>(cpuinst->p3ptrs.x,cpuinst->config.workmatgranksum, 1);
    cout << " Phase 2 Correctness : " << hyu.allclose(yu_pc) << endl;

    if(transpose) y_pc = seismic.Phase3Transpose();
    else y_pc = seismic.Phase3();
    if(transpose) hy = Matrix<complex<float>>(cpuinst->p3transptrs.y,cpuinst->config.originM, 1);
    else hy = Matrix<complex<float>>(cpuinst->p3ptrs.y,cpuinst->config.originM, 1);

    if(conjugate) y_pc = y_pc.Conjugate();
    cout << " Phase 3 Correctness : "<< hy.allclose(y_pc) << endl;
}


int main (int argc, char ** argv){
    auto argparser = ArgsParser(argc, argv);
    auto originM = argparser.getint("M");
    auto originN = argparser.getint("N");
    auto nb = argparser.getint("nb");
    auto threshold = argparser.getstring("threshold");
    auto datafolder = argparser.getstring("datafolder");
    auto streams = argparser.getint("streams");
    auto ordertype = argparser.getstring("ordertype");
    auto freqstart = argparser.getint("freqstart");
    auto freqend = argparser.getint("freqend");
    auto launchtype = argparser.getstring("launchtype");
    auto transpose = argparser.getbool("transpose");
    auto conjugate = argparser.getbool("conjugate");
    auto tlrmvmtype = argparser.getstring("TLRType");
    auto check = argparser.getbool("check");
    auto loopsize = argparser.getint("loopsize");

    MPI_Init(NULL,NULL);
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // get config for each frequency

    vector<int> totalfreqlist;
    int freqcnt = freqend - freqstart;
    for(int i=0; i<freqcnt; i++) totalfreqlist.push_back(i);
    vector<vector<int>> splitfreqlist;
    int cnt = 0;
    bool reverse = false;
    while (cnt < freqcnt){
        vector<int> tmp;
        int idx = 0;
        while (idx < world_size){
            tmp.push_back(cnt);
            cnt++;
            if(cnt>=freqcnt) break;
            idx++;
        }
        if(reverse){
            vector<int> reversetmp;
            for(int i=tmp.size()-1; i>=0 ; i--){
                reversetmp.push_back(tmp[i]);
            }
            splitfreqlist.push_back(reversetmp);
        }else{
            splitfreqlist.push_back(tmp);
        }
        if(reverse) reverse = false;
        else reverse = true;
    }
    vector<int> ownfreqlist;
    for(int i=0; i<splitfreqlist.size(); i++){
        if(splitfreqlist[i].size() > world_rank){
            ownfreqlist.push_back(splitfreqlist[i][world_rank] + freqstart);
        }
    }
//    sleep(world_rank);
//    cout << "rank " << world_rank;
//    for(int i=0; i<ownfreqlist.size();i++) cout << " " << ownfreqlist[i] ;
//    cout << endl;
//    sleep(0.1 * world_rank);
    if(world_rank == 0) cout << "OrderType " << ordertype << ", world size "  << world_size << endl;
    char rpath[300];
    vector<TlrmvmConfig> configs;
    for(int fi=0; fi<ownfreqlist.size(); fi++){
        sprintf(rpath, "Mode8_Order%s_Mck_freqslice_%d",ordertype.c_str(),ownfreqlist[fi]);
        auto problem = string(rpath);
        TlrmvmConfig config(originM, originN, nb, datafolder, threshold, problem);
        configs.push_back(config);
    }

    if(tlrmvmtype == "int8"){
        BatchTlrmvmcudaINT8<complex<float>, cuInt8Complex> batchtlrmvm(configs);
        batchtlrmvm.StreamInit(streams);
        batchtlrmvm.MemoryInit();
        batchtlrmvm.SetTransposeConjugate(transpose, conjugate);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        vector<double> rawtime;
        for(int i=0; i<loopsize; i++){
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);
            cudaEventRecord(start);
            if(launchtype == "singlegraph"){
                batchtlrmvm.MVM_SingleGraph();
            }else if(launchtype == "multigraph"){
                batchtlrmvm.MVM_MultiGraph();
            }else if(launchtype == "threephases"){
                if(transpose){
                    batchtlrmvm.Phase1Transpose();
                    batchtlrmvm.Phase2Transpose();
                    batchtlrmvm.Phase3Transpose();
                }else{
                    batchtlrmvm.Phase1();
                    batchtlrmvm.Phase2();
                    batchtlrmvm.Phase3();
                }
            }
            cudaEventRecord(stop);
            cudaDeviceSynchronize();
            cudaEventSynchronize(stop);
            MPI_Barrier(MPI_COMM_WORLD);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            rawtime.push_back(milliseconds*1e-3);
        }
        std::sort(rawtime.begin(),rawtime.end());
        cout << " doing int8 on rank " <<world_rank <<  ", median time " << rawtime[rawtime.size()/2] << endl;

//        batchtlrmvm.TryConjugateXvec();
//        batchtlrmvm.TryConjugateResults();
//        batchtlrmvm.CopyBackResults();
//        if(check){
//            for(int ci = freqstart; ci < freqend; ci++){
//                auto & cpuinst = batchtlrmvm.cpuinstvec[ci-freqstart];
//                sprintf(rpath, "Mode8_Order%s_Mck_freqslice_%d",ordertype.c_str(),ci);
//                auto problem = string(rpath);
//                CheckTLRMVM(cpuinst, datafolder, threshold, nb, problem, originM, originN, ci, conjugate, transpose);
//            }
//        }
    }else if(tlrmvmtype == "fp32"){
        BatchTlrmvmcuda<complex<float>, cuComplex> batchtlrmvm(configs);
        batchtlrmvm.StreamInit(streams);
        batchtlrmvm.MemoryInit();
        batchtlrmvm.SetTransposeConjugate(transpose, conjugate);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        vector<double> rawtime;
        for(int i=0; i<loopsize; i++){
            MPI_Barrier(MPI_COMM_WORLD);
            cudaDeviceSynchronize();
            cudaEventRecord(start);

            if(launchtype == "singlegraph"){
                batchtlrmvm.MVM_SingleGraph();
            }else if(launchtype == "multigraph"){
                batchtlrmvm.MVM_MultiGraph();
            }else if(launchtype == "threephases"){
                if(transpose){
                    batchtlrmvm.Phase1Transpose();
                    batchtlrmvm.Phase2Transpose();
                    batchtlrmvm.Phase3Transpose();
                }else{
                    batchtlrmvm.Phase1();
                    batchtlrmvm.Phase2();
                    batchtlrmvm.Phase3();
                }
            }

            cudaEventRecord(stop);
            cudaDeviceSynchronize();
            cudaEventSynchronize(stop);
            MPI_Barrier(MPI_COMM_WORLD);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            rawtime.push_back(milliseconds*1e-3);
        }
        std::sort(rawtime.begin(),rawtime.end());
        cout << " doing fp32 on rank " <<world_rank <<  ", median time " << rawtime[rawtime.size()/2] << endl;

//        batchtlrmvm.TryConjugateXvec();
//        batchtlrmvm.TryConjugateResults();
//        batchtlrmvm.CopyBackResults();
//        if(check){
//            for(int ci = freqstart; ci < freqend; ci++){
//                sprintf(rpath, "Mode8_Order%s_Mck_freqslice_%d",ordertype.c_str(),ci);
//                auto & cpuinst = batchtlrmvm.cpuinstvec[ci-freqstart];
//                auto problem = string(rpath);
//                CheckTLRMVM(cpuinst, datafolder, threshold, nb, problem, originM, originN, ci, conjugate, transpose);
//            }
//        }
    }else if(tlrmvmtype == "fp16"){
        BatchTlrmvmcudaFP16<complex<float>, cuHalfComplex> batchtlrmvm(configs);
        batchtlrmvm.StreamInit(streams);
        batchtlrmvm.MemoryInit();
        batchtlrmvm.SetTransposeConjugate(transpose, conjugate);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        vector<double> rawtime;
        for(int i=0; i<loopsize; i++){
            MPI_Barrier(MPI_COMM_WORLD);
            cudaDeviceSynchronize();
            cudaEventRecord(start);

            if(launchtype == "singlegraph"){
                batchtlrmvm.MVM_SingleGraph();
            }else if(launchtype == "multigraph"){
                batchtlrmvm.MVM_MultiGraph();
            }else if(launchtype == "threephases"){
                if(transpose){
                    batchtlrmvm.Phase1Transpose();
                    batchtlrmvm.Phase2Transpose();
                    batchtlrmvm.Phase3Transpose();
                }else{
                    batchtlrmvm.Phase1();
                    batchtlrmvm.Phase2();
                    batchtlrmvm.Phase3();
                }
            }

            cudaEventRecord(stop);
            cudaDeviceSynchronize();
            cudaEventSynchronize(stop);
            MPI_Barrier(MPI_COMM_WORLD);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            rawtime.push_back(milliseconds*1e-3);
        }
        std::sort(rawtime.begin(),rawtime.end());
        cout << " doing fp16 on rank " <<world_rank <<  ", median time " << rawtime[rawtime.size()/2] << endl;
//
//        batchtlrmvm.TryConjugateXvec();
//        batchtlrmvm.TryConjugateResults();
//        batchtlrmvm.CopyBackResults();
//        if(check){
//            for(int ci = freqstart; ci < freqend; ci++){
//                sprintf(rpath, "Mode8_Order%s_Mck_freqslice_%d",ordertype.c_str(),ci);
//                auto & cpuinst = batchtlrmvm.cpuinstvec[ci-freqstart];
//                auto problem = string(rpath);
//                CheckTLRMVM(cpuinst, datafolder, threshold, nb, problem, originM, originN, ci, conjugate, transpose);
//            }
//        }
    }else if(tlrmvmtype == "bf16"){
        BatchTlrmvmcudaFP16<complex<float>, cubfComplex> batchtlrmvm(configs);
        batchtlrmvm.StreamInit(streams);
        batchtlrmvm.MemoryInit();
        batchtlrmvm.SetTransposeConjugate(transpose, conjugate);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        vector<double> rawtime;
        for(int i=0; i<loopsize; i++){
            MPI_Barrier(MPI_COMM_WORLD);
            cudaDeviceSynchronize();
            cudaEventRecord(start);

            if(launchtype == "singlegraph"){
                batchtlrmvm.MVM_SingleGraph();
            }else if(launchtype == "multigraph"){
                batchtlrmvm.MVM_MultiGraph();
            }else if(launchtype == "threephases"){
                if(transpose){
                    batchtlrmvm.Phase1Transpose();
                    batchtlrmvm.Phase2Transpose();
                    batchtlrmvm.Phase3Transpose();
                }else{
                    batchtlrmvm.Phase1();
                    batchtlrmvm.Phase2();
                    batchtlrmvm.Phase3();
                }
            }

            cudaEventRecord(stop);
            cudaDeviceSynchronize();
            cudaEventSynchronize(stop);
            MPI_Barrier(MPI_COMM_WORLD);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            rawtime.push_back(milliseconds*1e-3);
        }
        std::sort(rawtime.begin(),rawtime.end());
        cout << " doing bf16 on rank " <<world_rank <<  ", median time " << rawtime[rawtime.size()/2] << endl;

//        batchtlrmvm.TryConjugateXvec();
//        batchtlrmvm.TryConjugateResults();
//        batchtlrmvm.CopyBackResults();
//        if(check){
//            for(int ci = freqstart; ci < freqend; ci++){
//                sprintf(rpath, "Mode8_Order%s_Mck_freqslice_%d",ordertype.c_str(),ci);
//                auto & cpuinst = batchtlrmvm.cpuinstvec[ci-freqstart];
//                auto problem = string(rpath);
//                CheckTLRMVM(cpuinst, datafolder, threshold, nb, problem, originM, originN, ci, conjugate, transpose);
//            }
//         }
    }

    MPI_Finalize();

    return 0;
}
