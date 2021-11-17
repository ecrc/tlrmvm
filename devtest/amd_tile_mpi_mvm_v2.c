/**
 * @copyright (c) 2020- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <memory.h>
#include <blis.h>
#include <pthread.h>
#include <mpi.h>
#include <omp.h>
#include "util.h"

#define real_t float


/**
 * @brief AMD dense mvm but use tile format and MPI interface. 
 * This is an multithread version of tile mvm.
 */

int main(int argc, char* argv[])
{
    // MPI initialization
    MPI_Status stat;
    int mpirank, mpisize;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);

    unsigned long int m, n, padding_m, padding_n;
    int nb, ntile_x, ntile_y;
    m = atol(argv[1]);
    n = atol(argv[2]);
    nb = atoi(argv[3]);
    real_t alpha = 1.0, beta = 0.0;
    padding_m = (m / nb) * ( nb + (m % nb != 0) );
    padding_n = (n / nb) * ( nb + (n % nb != 0) );
    ntile_x = padding_n / nb;
    ntile_y = padding_m / nb;
    
    int threads_per_mpi;
    int num_threadgroup = mpisize;
    #pragma omp parallel
    {
        #pragma omp single
        threads_per_mpi = omp_get_num_threads();
    }
    int threadnums = mpisize * threads_per_mpi;

    int *rows_per_threadgroup;
    int *acc_rows_threadgroup;
    int *cols_per_thread;
    int *acc_cols_thread;
    int *gatherrv_rowsinfo;
    int *acc_gatherrv_rowsinfo;

    rows_per_threadgroup = malloc(sizeof(int) * mpisize);
    acc_rows_threadgroup = malloc(sizeof(int) * mpisize);
    gatherrv_rowsinfo = malloc(sizeof(int) * mpisize);
    acc_gatherrv_rowsinfo = malloc(sizeof(int) * mpisize);
    cols_per_thread = malloc(sizeof(int) * threads_per_mpi);
    acc_cols_thread = malloc(sizeof(int) * threads_per_mpi);

    calculate_layout(rows_per_threadgroup, acc_rows_threadgroup, 
    cols_per_thread, acc_cols_thread, ntile_x, ntile_y, 
    mpisize, threads_per_mpi);
    for(int i=0; i<mpisize; i++){
        gatherrv_rowsinfo[i] = rows_per_threadgroup[i] * nb;
    }
    acc_gatherrv_rowsinfo[0] = 0;
    for(int i=1; i<mpisize; i++){
        acc_gatherrv_rowsinfo[i] = acc_gatherrv_rowsinfo[i-1] + gatherrv_rowsinfo[i-1];
    }

    
    real_t *A_origin = malloc(sizeof(real_t) * padding_n * padding_m);
    real_t *x_origin = malloc(sizeof(real_t) * padding_n);
    real_t *y_origin = malloc(sizeof(real_t) * padding_m);

    Initdata(A_origin, x_origin, y_origin, padding_m, padding_n);
    cblas_sgemv(CblasColMajor, CblasNoTrans, 
    padding_m, padding_n, alpha, A_origin, 
    padding_m, x_origin, 1, beta, y_origin, 1);
    
    real_t **Atemp = malloc(sizeof(real_t*) * threads_per_mpi);
    real_t **ytemp = malloc(sizeof(real_t*) * threads_per_mpi);
    real_t **xtemp = malloc(sizeof(real_t*) * threads_per_mpi);
    real_t *y_local = malloc(sizeof(real_t) * nb * rows_per_threadgroup[mpirank]);
    real_t *y = malloc(sizeof(real_t) * padding_m);
    memset(y_local, 0, sizeof(real_t) * nb * rows_per_threadgroup[mpirank]);
    real_t *x = malloc(sizeof(real_t) * padding_n);
    for(int i=0; i<padding_n; i++) x[i] = 1.0;
    
    #pragma omp parallel
    {
        int tidx = omp_get_thread_num();
        int tidy = mpirank;
        Atemp[tidx] = aligned_alloc(128, sizeof(real_t) * nb * cols_per_thread[tidx] * 
        nb * rows_per_threadgroup[tidy]); 
        ytemp[tidx] = aligned_alloc(128, sizeof(real_t) * nb * rows_per_threadgroup[tidy] * cols_per_thread[tidx]);
        xtemp[tidx] = aligned_alloc(128, sizeof(real_t) * nb * cols_per_thread[tidx]);
        Initdata(Atemp[tidx], xtemp[tidx], ytemp[tidx], 
        nb * rows_per_threadgroup[tidy], nb * cols_per_thread[tidx]);
        // TODO: copy data from original layout to tile layout
    }
    int warmup = 200;
    for(int i=0; i<5000+warmup; i++){
        memset(y_local, 0, sizeof(real_t) * nb * rows_per_threadgroup[mpirank]);
        MPI_Barrier(MPI_COMM_WORLD);
        double st = MPI_Wtime();
        #pragma omp parallel
        {
            int tidx = omp_get_thread_num();
            int tidy = mpirank;
            for(int j=0; j<rows_per_threadgroup[mpirank]; j++){
                for(int k=0; k<cols_per_thread[tidx]; k++){
                    int offset = k + j * cols_per_thread[tidx];
                    cblas_sgemv(CblasColMajor, CblasNoTrans, nb, 
                    nb, alpha, Atemp[tidx] + nb * nb * offset,
                    nb, xtemp[tidx] + nb * k, 1, beta, ytemp[tidx] + nb * offset, 1);
                }
            }
            int stride = nb / 8;
            #pragma omp barrier
            for(int j=0; j<rows_per_threadgroup[mpirank]; j++){
                for(int p=tidx * stride; p < (tidx+1) * stride; p++){
                    for(int k=0; k < cols_per_thread[tidx]; k++){
                        for(int oth=0; oth < threads_per_mpi; oth++){
                            y_local[j*nb + p] += 
                            ytemp[oth][p + nb * (j * cols_per_thread[tidx] + k)];
                        }
                    }
                }
            }
        }
        MPI_Gatherv(y_local, nb*rows_per_threadgroup[mpirank], MPI_FLOAT, y, 
        gatherrv_rowsinfo, acc_gatherrv_rowsinfo, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double et = MPI_Wtime();
        double maxt, localt;
        localt = et - st;
        MPI_Reduce(&localt, &maxt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        char* info = malloc(sizeof(char)*100);
        double val = netlib_error(y, y_origin, padding_m);
        if(i < warmup) continue;
        sprintf(info, "dense");
        if(mpirank == 0){
            perfoutput(maxt, padding_m, padding_n, info, A_origin);
        }
        
    }

    MPI_Finalize();
    return 0;
}
