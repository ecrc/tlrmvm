/**
 * @copyright (c) 2020- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/
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
#include <blis.h>
#include <pthread.h>
#include <mpi.h>
#include <omp.h>
#define real_t float


int min(int a, int b){
  if(a < b) return a;
  return b;
}

double mean(double *y, int dim){
  double ret = 0.0;

  for(int i=0; i<dim; i++){
    ret += y[i];
  }
  ret /= dim;
  return ret;
}

double gettime(void)
{
  struct timeval tp;
  gettimeofday( &tp, NULL );
  return tp.tv_sec + 1e-6 * tp.tv_usec;
}

void naive_impl(real_t *A, real_t* x, real_t *y, int m, int n){
  for(int i=0; i<m; i++){
    y[i] = (real_t)0.0;
    for(int j=0; j<n; j++){
      y[i] += A[(long)i + (long)j * (long)m] * x[j];
    }
  }
}

void Initdata_cpu(real_t *A, real_t* x, real_t *y, int m, int n){
  memset(y, 0, sizeof(real_t) * m);

  for(int i=0; i<n; i++){
    // x[i] = (real_t)rand() / (real_t)RAND_MAX;
    x[i] = 1.0;
  }

  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      long tmp = (long)i * (long)n + (long)j;
    //   A[tmp] = (real_t)rand() / (real_t)RAND_MAX;
    A[tmp] = 1.0;
    }
  }
}


double checkcorrectness(real_t *y, real_t *ynaive, int dim, int sumdim){
  double *rerr = (double*)malloc(sizeof(double)*dim);

  for(int i=0; i<dim; i++){
    rerr[i] =fabs(y[i] - ynaive[i]) / fabs(ynaive[i]);
  }
  double meanval = mean(rerr, dim);
  free(rerr);
  return meanval;
}

int main(int argc, const char** argv)
{
    real_t *A, *y, *ycheck, *x;
    int m, n, nruns;
    nruns = 100;
    real_t alpha, beta;
    printf ("\n This benchmark computes real vector y=alpha*A*x+beta*y, "
    "where A is matrix, y and x are vectors alpha and beta are scalars\n\n");
    alpha = 1.0;
    beta = 0.0;
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    A = (real_t *)malloc((long)m*(long)n*sizeof( real_t ));
    x = (real_t *)malloc((long)n*sizeof( real_t ));
    y = (real_t *)malloc((long)m*sizeof( real_t ));
    ycheck = (real_t *)malloc((long)m*sizeof( real_t ));
    Initdata_cpu(A, x, y, m, n);
    double btes = m*n+m+n;
    double bw;
    for(int i=0; i<1000; i++){
        double st = MPI_Wtime();
        cblas_sgemv(CblasColMajor, CblasNoTrans, 
        m, n, alpha, A, m, x, 1, beta, ycheck, 1);
        double et = MPI_Wtime();
        double elps = et - st;
        bw = btes / elps;
        printf("time %f Bytes %f Bandwidth %f \n", et - st, btes*1e-6, bw*1e-9);
    }
    free(A);
    free(x);
    free(y);
    return 0;
}
