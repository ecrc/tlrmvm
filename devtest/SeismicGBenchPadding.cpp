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
#include "tlrmvm/Tlrmvmcuda.h"

#include "benchmark/benchmark.h"
#include "benchmark/benchmark.h"

using namespace std;
using namespace tlrmat;
using namespace cudatlrmat;
using namespace benchmark;


unordered_map<string, string> inputmap;


class SeismicFixture : public benchmark::Fixture {
public:
    void SetUp(const ::benchmark::State& state) {

    }

    void TearDown(const ::benchmark::State& state) {

    }
};

BENCHMARK_F(SeismicFixture, FooTest)(benchmark::State& st) {
    for (auto _ : st) {

    }
}



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


