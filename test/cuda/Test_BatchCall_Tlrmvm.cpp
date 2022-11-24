//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#include <iostream>
#include <unistd.h>
#include <memory.h>
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

    // get config for each frequency
    char rpath[300];
    vector<TlrmvmConfig> configs;
    for(int fi=freqstart; fi<freqend; fi++){
        sprintf(rpath, "Mode8_Order%s_Mck_freqslice_%d",ordertype.c_str(),fi);
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
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            rawtime.push_back(milliseconds*1e-3);
        }
        std::sort(rawtime.begin(),rawtime.end());
        cout << "median time " << rawtime[rawtime.size()/2] << " s."<< endl;
        auto ggranksum = 0;
        for(auto &ptr : batchtlrmvm.cpuinstvec){
            ggranksum += ptr->config.granksum;
        }
        auto bytes = TLRMVMBytesProcessed<complex<float>>(ggranksum, nb, originM, originN);
        bytes /= 4.0; // int8 , div by 4
        cout << "Bandwidth: " << (double)bytes / rawtime[rawtime.size()/2] * 1e-9 << " GB/s." << endl;

        batchtlrmvm.TryConjugateXvec();
        batchtlrmvm.TryConjugateResults();
        batchtlrmvm.CopyBackResults();
        if(check){
            for(int ci = freqstart; ci < freqend; ci++){
                auto & cpuinst = batchtlrmvm.cpuinstvec[ci-freqstart];
                sprintf(rpath, "Mode8_Order%s_Mck_freqslice_%d",ordertype.c_str(),ci);
                auto problem = string(rpath);
                CheckTLRMVM(cpuinst, datafolder, threshold, nb, problem, originM, originN, ci, conjugate, transpose);
            }
        }
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
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            rawtime.push_back(milliseconds*1e-3);
        }
        std::sort(rawtime.begin(),rawtime.end());
        cout << "median time " << rawtime[rawtime.size()/2] << " s."<< endl;
        auto ggranksum = 0;
        for(auto &ptr : batchtlrmvm.cpuinstvec){
            ggranksum += ptr->config.granksum;
        }
        auto bytes = TLRMVMBytesProcessed<complex<float>>(ggranksum, nb, originM, originN);
        cout << "Bandwidth: " << (double)bytes / rawtime[rawtime.size()/2] * 1e-9 << " GB/s." << endl;

        batchtlrmvm.TryConjugateXvec();
        batchtlrmvm.TryConjugateResults();
        batchtlrmvm.CopyBackResults();
        if(check){
            for(int ci = freqstart; ci < freqend; ci++){
                sprintf(rpath, "Mode8_Order%s_Mck_freqslice_%d",ordertype.c_str(),ci);
                auto & cpuinst = batchtlrmvm.cpuinstvec[ci-freqstart];
                auto problem = string(rpath);
                CheckTLRMVM(cpuinst, datafolder, threshold, nb, problem, originM, originN, ci, conjugate, transpose);
            }
        }
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
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            rawtime.push_back(milliseconds*1e-3);
        }
        std::sort(rawtime.begin(),rawtime.end());
        cout << "median time " << rawtime[rawtime.size()/2] << " s."<< endl;
        auto ggranksum = 0;
        for(auto &ptr : batchtlrmvm.cpuinstvec){
            ggranksum += ptr->config.granksum;
        }
        auto bytes = TLRMVMBytesProcessed<complex<float>>(ggranksum, nb, originM, originN);
        bytes /= 2.0; // fp16 , div by 2
        cout << "Bandwidth: " << (double)bytes / rawtime[rawtime.size()/2] * 1e-9 << " GB/s." << endl;

        batchtlrmvm.TryConjugateXvec();
        batchtlrmvm.TryConjugateResults();
        batchtlrmvm.CopyBackResults();
        if(check){
            for(int ci = freqstart; ci < freqend; ci++){
                sprintf(rpath, "Mode8_Order%s_Mck_freqslice_%d",ordertype.c_str(),ci);
                auto & cpuinst = batchtlrmvm.cpuinstvec[ci-freqstart];
                auto problem = string(rpath);
                CheckTLRMVM(cpuinst, datafolder, threshold, nb, problem, originM, originN, ci, conjugate, transpose);
            }
        }
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
            cudaEventSynchronize(stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            rawtime.push_back(milliseconds*1e-3);
        }
        std::sort(rawtime.begin(),rawtime.end());
        cout << "median time " << rawtime[rawtime.size()/2] << " s."<< endl;
        auto ggranksum = 0;
        for(auto &ptr : batchtlrmvm.cpuinstvec){
            ggranksum += ptr->config.granksum;
        }
        auto bytes = TLRMVMBytesProcessed<complex<float>>(ggranksum, nb, originM, originN);
        bytes /= 2.0; // bf16 , div by 2
        cout << "Bandwidth: " << (double)bytes / rawtime[rawtime.size()/2] * 1e-9 << " GB/s." << endl;
        batchtlrmvm.TryConjugateXvec();
        batchtlrmvm.TryConjugateResults();
        batchtlrmvm.CopyBackResults();
        if(check){
            for(int ci = freqstart; ci < freqend; ci++){
                sprintf(rpath, "Mode8_Order%s_Mck_freqslice_%d",ordertype.c_str(),ci);
                auto & cpuinst = batchtlrmvm.cpuinstvec[ci-freqstart];
                auto problem = string(rpath);
                CheckTLRMVM(cpuinst, datafolder, threshold, nb, problem, originM, originN, ci, conjugate, transpose);
            }
        }
    }

    return 0;
}
