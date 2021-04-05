/**
 * @copyright (c) 2020- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <memory.h>

#define real_t float

#ifdef USE_NVIDIA
 #define CUDACHECK(cmd) do {                         \
   cudaError_t e = cmd;                              \
   if( e != cudaSuccess ) {                          \
     printf("Failed: Cuda error %s:%d '%s'\n",             \
         __FILE__,__LINE__,cudaGetErrorString(e));   \
     exit(EXIT_FAILURE);                             \
   }                                                 \
 } while(0)
 
 
 #define NCCLCHECK(cmd) do {                         \
   ncclResult_t r = cmd;                             \
   if (r!= ncclSuccess) {                            \
     printf("Failed, NCCL error %s:%d '%s'\n",             \
         __FILE__,__LINE__,ncclGetErrorString(r));   \
     exit(EXIT_FAILURE);                             \
   }                                                 \
 } while(0)
 
 
 #define NCCLCHECK(cmd) do {                         \
   ncclResult_t r = cmd;                             \
   if (r!= ncclSuccess) {                            \
     printf("Failed, NCCL error %s:%d '%s'\n",             \
         __FILE__,__LINE__,ncclGetErrorString(r));   \
     exit(EXIT_FAILURE);                             \
   }                                                 \
 } while(0)
#endif 




double gettime(void);

void Init(real_t *A, int m, int n);