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
#ifdef USE_BLIS
#include <blis.h>
#else
#include <cblas.h>
#endif
#include <omp.h>
#include "util.h"

/**
 * @brief Benchmark AMD Dense MVM. 
 * The simplest dense mvm.
 * AMD blis lib only support single thread mvm.
 * The performance (bandwidth) is very low.
 */

int main(int argc, const char** argv)
{
    unsigned long int m, n;
    m = atol(argv[1]);
    n = atol(argv[2]);
    int threadsnum;
    #pragma omp parallel 
    {
        #pragma omp single
        threadsnum = omp_get_num_threads();
    }
    printf("Threads numbers: %d\n", threadsnum);
    real_t *A = aligned_alloc(128, sizeof(real_t) * m * n);
    real_t *x = aligned_alloc(128, sizeof(real_t) * n);
    real_t *y = aligned_alloc(128, sizeof(real_t) * m);

    Initdata(A, x, y, (int)m, (int)n);
    char * info = malloc(sizeof(real_t) * 20);
#ifdef USE_BLIS
    sprintf(info, "AMD_BLISMVM");
#else
    sprintf(info, "AMD_OPENBLASMVM");
#endif
    real_t alpha = 1.0, beta = 0.0;
    for(int iter=0; iter<6000; iter++){
        double st = gettime();
#ifdef USE_DOUBLE
        cblas_dgemv(CblasColMajor, CblasNoTrans, 
        m, n, alpha, A, m, x, 1, beta, y, 1);
#else
        cblas_sgemv(CblasColMajor, CblasNoTrans, 
        m, n, alpha, A, m, x, 1, beta, y, 1);
#endif
        double et = gettime();
        if (iter < 3000) continue;
        perfoutput((double)et-st, m, n, info, A);
    }
    free(info);
    free(A);
    free(x);
    free(y);
}
