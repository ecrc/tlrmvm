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
#include <mpi.h>
#include <omp.h>
#include "util.h"

static __inline__ int64_t rdtsc_s(void)
{
  unsigned a, d; 
  asm volatile("cpuid" ::: "%rax", "%rbx", "%rcx", "%rdx");
  asm volatile("rdtsc" : "=a" (a), "=d" (d)); 
  return ((unsigned long)a) | (((unsigned long)d) << 32); 
}

static __inline__ int64_t rdtsc_e(void)
{
  unsigned a, d; 
  asm volatile("rdtscp" : "=a" (a), "=d" (d)); 
  asm volatile("cpuid" ::: "%rax", "%rbx", "%rcx", "%rdx");
  return ((unsigned long)a) | (((unsigned long)d) << 32); 
}

void * _mm_malloc(size_t s, size_t n) { return aligned_alloc(n, s); }
/**
 * @brief AMD dense mvm but use tile format and OpenMP. 
 * This is an multithread version of tile mvm.
 */

int main(int argc, const char** argv)
{
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
    int threads_per_group=8, threadnums;
    int num_threadgroup;
    #pragma omp parallel
    {
        #pragma omp single
        threadnums = omp_get_num_threads();
    }
    num_threadgroup = threadnums / threads_per_group;

    int *rows_per_threadgroup;
    int *acc_rows_threadgroup;
    int *cols_per_thread;
    int *acc_cols_thread;
    rows_per_threadgroup = malloc(sizeof(int) * num_threadgroup);
    acc_rows_threadgroup = malloc(sizeof(int) * num_threadgroup);
    cols_per_thread = malloc(sizeof(int) * threads_per_group);
    acc_cols_thread = malloc(sizeof(int) * threads_per_group);
    calculate_layout(rows_per_threadgroup, acc_rows_threadgroup, 
    cols_per_thread, acc_cols_thread, ntile_x, ntile_y, num_threadgroup, threads_per_group);

    real_t *A_origin = malloc(sizeof(real_t) * padding_n * padding_m);
    real_t *x_origin = malloc(sizeof(real_t) * padding_n);
    real_t *y_origin = malloc(sizeof(real_t) * padding_m);

    Initdata(A_origin, x_origin, y_origin, padding_m, padding_n);
    cblas_sgemv(CblasColMajor, CblasNoTrans, 
    padding_m, padding_n, alpha, A_origin, 
    padding_m, x_origin, 1, beta, y_origin, 1);
    
    real_t **Atemp = malloc(sizeof(real_t*) * threadnums);
    real_t **ytemp = malloc(sizeof(real_t*) * threadnums);
    real_t **xtemp = malloc(sizeof(real_t*) * threadnums);
    real_t *y = malloc(sizeof(real_t) * padding_m);
    memset(y, 0, sizeof(real_t) * padding_m);
    real_t *x = malloc(sizeof(real_t) * padding_n);
    for(int i=0; i<padding_n; i++) x[i] = 1.0;

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int tidx = tid % threads_per_group;
        int tidy = tid / threads_per_group;

        Atemp[tid] =malloc(sizeof(real_t) * nb * cols_per_thread[tidx] * 
        nb * rows_per_threadgroup[tidy]); 
        ytemp[tid] =malloc(sizeof(real_t) * nb * rows_per_threadgroup[tidy]);
        xtemp[tid] =malloc(sizeof(real_t) * nb * cols_per_thread[tidx]);
        Initdata(Atemp[tid], xtemp[tid], ytemp[tid], 
        nb * rows_per_threadgroup[tidy], nb * cols_per_thread[tidx]);
        // TODO: copy data from original layout to tile layout
    }

    for(int i=0; i<5000; i++){
        memset(y, 0, sizeof(real_t) * padding_m);
        sleep(0.001);
        double st = omp_get_wtime();
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int tidx = tid % threads_per_group;
            int tidy = tid / threads_per_group;    
            // v1: only 1 single mvm inside a thread
            cblas_sgemv(CblasColMajor, CblasNoTrans, nb * rows_per_threadgroup[tidy], 
            nb * cols_per_thread[tidx], alpha, Atemp[tid],
            nb * rows_per_threadgroup[tidy], xtemp[tid], 1, beta, ytemp[tid], 1);
            int stride = nb * rows_per_threadgroup[tidy] / threads_per_group;
            #pragma omp smid
            for(int j=tidx*stride; j<(tidx+1)*stride; j++){
                for(int k=0; k<threads_per_group; k++){
                    y[j + acc_rows_threadgroup[tidy] * nb] += 
                    ytemp[k + tidy*threads_per_group][j];
                }
            }
            #pragma omp barrier
        }
        double et = omp_get_wtime();
        real_t diffval = 0.0;
        for(int j=0; j < padding_m; j++){
            diffval += fabs(y[j] - y_origin[j]);
        }
        char* info = malloc(sizeof(char) * 100);
        sprintf(info, "dense_openmp_v1");
        perfoutput(et-st, padding_m, padding_n, info, A_origin);
    }
    return 0;
}
