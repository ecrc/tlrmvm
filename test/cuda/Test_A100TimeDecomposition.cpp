//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#include <iostream>
#include <unistd.h>
#include <memory.h>
#include "common/Common.hpp"
#include "tlrmvm/Tlrmvm.hpp"
#include <algorithm>
using namespace cudatlrmvm;

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
        cout << "int8 median time " << rawtime[rawtime.size()/2] << " s."<< endl;

    }else if(tlrmvmtype == "int8_p1"){
        BatchTlrmvmcudaINT8_p1<complex<float>, cuInt8Complex> batchtlrmvm(configs);
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
        cout << "int8_p1 median time " << rawtime[rawtime.size()/2] << " s."<< endl;

    }else if(tlrmvmtype == "int8_p1p2"){
        BatchTlrmvmcudaINT8_p1p2<complex<float>, cuInt8Complex> batchtlrmvm(configs);
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
        cout << "int8_p1p2 median time " << rawtime[rawtime.size()/2] << " s."<< endl;

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
        cout << "fp32 median time " << rawtime[rawtime.size()/2] << " s."<< endl;

    }else if(tlrmvmtype == "fp32_p1"){
        BatchTlrmvmcuda_p1<complex<float>, cuComplex> batchtlrmvm(configs);
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
        cout << "fp32_p1 median time " << rawtime[rawtime.size()/2] << " s."<< endl;

    }else if(tlrmvmtype == "fp32_p1p2"){
        BatchTlrmvmcuda_p1p2<complex<float>, cuComplex> batchtlrmvm(configs);
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
        cout << "fp32_p1p2 median time " << rawtime[rawtime.size()/2] << " s."<< endl;

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
        cout << "fp16 median time " << rawtime[rawtime.size()/2] << " s."<< endl;

    }else if(tlrmvmtype == "fp16_p1"){
        BatchTlrmvmcudaFP16_p1<complex<float>, cuHalfComplex> batchtlrmvm(configs);
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
        cout << "fp16_p1 median time " << rawtime[rawtime.size()/2] << " s."<< endl;

    }else if(tlrmvmtype == "fp16_p1p2"){
        BatchTlrmvmcudaFP16_p1p2<complex<float>, cuHalfComplex> batchtlrmvm(configs);
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
        cout << "fp16_p1p2 median time " << rawtime[rawtime.size()/2] << " s."<< endl;

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
        cout << "bf16 median time " << rawtime[rawtime.size()/2] << " s."<< endl;
    }
    else if(tlrmvmtype == "bf16_p1"){
        BatchTlrmvmcudaFP16_p1<complex<float>, cubfComplex> batchtlrmvm(configs);
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
        cout << "bf16_p1 median time " << rawtime[rawtime.size()/2] << " s."<< endl;
    }
    else if(tlrmvmtype == "bf16_p1p2"){
        BatchTlrmvmcudaFP16_p1p2<complex<float>, cubfComplex> batchtlrmvm(configs);
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
        cout << "bf16_p1p2 median time " << rawtime[rawtime.size()/2] << " s."<< endl;

    }
    return 0;
}
