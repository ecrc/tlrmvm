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

#ifdef USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#else
#include <mpi.h>
#include <cblas.h>
#include <blis.h>
void bli_thread_set_num_threads( dim_t n_threads );
#endif

#ifdef USE_DOUBLE
#define real_t double 
#else
#define real_t float
#endif

int main(int argv, char * argv[]){


    /*
    * configuration
    * */
   int nb = 100;
   unsigned long int M = 5000, N = 20000;

    MPI_Status stat;
    int mpirank, mpisize;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);

    real_t *A_original, *Au, *Av, *yu, *yv, *x, *y, *yfinal, *y_check;

    int *tilerank;

    // create tile rank
    tilerank = (int*)malloc(sizof(int)*nb);
    for(int i=0; i<nb; i++){
        tilerank[i] = rand() % nb;
    }

    // allocate memory space 
    unsigned long int totalrank = 0;
    for(int i=0; i<nb; i++) {
        totalrank += tilerank[i]; // get total rank size
    }
    A_original = (real_t*)malloc(sizeof(real_t) * M * N);
    Au = (real_t*)malloc(sizeof(real_t) * totalrank * nb);
    Av = (real_t*)malloc(sizeof(real_t) * totalrank * nb);
    yu = (real_t*)malloc(sizeof(real_t) * totalrank);
    yv = (real_t*)malloc(sizeof(real_t) * totalrank);
    x = (real_t*)malloc(sizeof(real_t) * N);
    y = (real_t*)malloc(sizeof(real_t) * M);
    y_final = (real_t*)malloc(sizeof(real_t) * M);
    y_check = (real_t*)malloc(sizeof(real_t) * M);

    // fill random number
    // fill Au, Av, x with ranomd number
    // one can generate A_original using Au,Av
    // then using A_original gand x get y_check;



}
