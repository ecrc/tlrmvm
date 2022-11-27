//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#include <hipblas.h>
#include <iostream>
#include <hip/hip_runtime.h>
using namespace std;
int main(){
    hipEvent_t start;
    hipEvent_t stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);
    for(int i=0; i<10; i++){
        cout << "hello" << endl;
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    cout << "time "<< milliseconds << endl;

}
