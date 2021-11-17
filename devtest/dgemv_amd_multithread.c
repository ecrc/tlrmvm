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

void convert_colmajor_2_tile(real_t *y_colmajor, real_t *y_tile, int m, int n, int nb){
    int numnbx = n / nb;
    int numnby = m / nb;
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            unsigned long int ty = j / nb;
            unsigned long int tx = i / nb;
            int ymod = j % nb;
            int xmod = i % nb;
            unsigned long int offset = nb * nb * (tx + ty * numnbx) + xmod * nb + ymod;
            y_tile[offset] = y_colmajor[i * m + j];
        }
    }
}

void convert_tile_2_colmajor(real_t *y_colmajor, real_t *y_tile, int m, int n, int nb){
    int numnbx = n / nb;
    int numnby = m / nb;

    #pragma omp parallel for collapse(2)
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            unsigned long int ty = j / nb;
            unsigned long int tx = i / nb;
            int ymod = j % nb;
            int xmod = i % nb;
            unsigned long int offset = nb * nb * (tx + ty * numnbx) + xmod * nb + ymod;
            y_colmajor[i * m + j] = y_tile[offset];
        }
    }
}


int main(int argc, const char** argv)
{
    int mpirank, mpisize;
    MPI_Init(&argc, (char***)&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    real_t *A_tile, *A_colmajor, *y, *ycheck, *x;
    int blockm, blockn, nruns, nb, globalm, globaln;
    nruns = 100;
    real_t alpha, beta;
    printf ("\n This benchmark computes real vector y=alpha*A*x+beta*y, "
    "where A is matrix, y and x are vectors alpha and beta are scalars\n\n");
    alpha = 1.0;
    beta = 0.0;
    blockm = atoi(argv[1]);
    blockn = atoi(argv[2]);
    nb = blockm;
    const int threadnums=8;
    globalm = blockm;
    globaln = blockn * threadnums;
    A_tile = (real_t *)malloc((long)globalm*(long)globaln*sizeof( real_t ));
    A_colmajor = (real_t *)malloc((long)globalm*(long)globaln*sizeof( real_t ));
    x = (real_t *)malloc((long)globaln*sizeof( real_t ));
    y = (real_t *)malloc((long)globalm*sizeof( real_t ));
    real_t **y_split;
    y_split = (real_t **)malloc(sizeof(real_t*)*threadnums);
    for(int i=0; i<threadnums; i++) y_split[i] = (real_t*)malloc(sizeof(real_t) * blockm);
    ycheck = (real_t *)malloc((long)globalm*sizeof( real_t ));
    memset(y, 0, sizeof(real_t) * globalm);
    memset(ycheck, 0, sizeof(real_t) * globalm);
    Initdata_cpu(A_colmajor, x, ycheck, globalm, globaln);
    // gemv on colmajor
    double st = gettime();
    cblas_sgemv(CblasColMajor, CblasNoTrans, 
    globalm, globaln, alpha, A_colmajor, globalm, x, 1, beta, ycheck, 1);
    double et = gettime();
    printf("globalm %d globaln %d cblas colmajor time is %f \n", globalm, globaln, et - st);
    st = gettime();
    cblas_sgemv(CblasColMajor, CblasNoTrans, 
    blockm, blockn, alpha, A_colmajor, blockm, x, 1, beta, ycheck, 1);
    et = gettime();
    printf("blockm %d blockn %d cblas colmajor time is %f \n", blockm, blockn, et - st);
    // gemv on 
    // st = gettime();
    // convert_tile_2_colmajor(A_colmajor, A_tile, globalm, globaln, blockm);
    // et = gettime();
    // printf("converting time %f \n", et -st );
    double bw;
    for(int nr=0; nr < 10000; nr++){
        st = MPI_Wtime();
        #pragma omp parallel
        {
            int id = omp_get_thread_num();
            cblas_sgemv(CblasColMajor, CblasNoTrans, 
            blockm, blockn, alpha, A_colmajor + id * blockm * blockn, blockm, x, 1, beta, y_split[id], 1);
        }
        et = MPI_Wtime();
        bw = sizeof(real_t) * (mpisize * globalm * globaln + mpisize * globalm + globaln) / (et-st);
        // printf("blockm %d blockn %d time is %f  bw is %f \n", blockm, blockn, et - st, bw*1e-9);    
    }
     
    // for(int nr=0; nr<2000; nr++){
    //     double stime = 0.0, etime = 0.0, executiontime = 0.0;
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     stime = MPI_Wtime();
    //     #pragma omp parallel
    //     {
    //         int id = omp_get_thread_num();
    //         cblas_sgemv(CblasColMajor, CblasNoTrans, 
    //         m, n, alpha, A+m*n*id, m, x, 1, beta, y+m*id, 1);
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     etime = MPI_Wtime();
    //     executiontime = etime - stime;
    //     double maxexetime;
    //     MPI_Reduce(&executiontime, &maxexetime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    //     double bd = sizeof(real_t)*(m*n+m+n)/maxexetime * 1e-9;
    //     double ds = sizeof(real_t)*(m*n+m+n);
    //     if(mpirank == 0)    
    //     printf("total id each mpi %d bd total data size %.3f MB iteration %d Time %.6f Bandwidth %.3f \n", totalid, ds*1e-6, nr, maxexetime, bd); 
    // }

    free(A_tile);
    free(A_colmajor);
    free(x);
    free(y);
    free(ycheck);
    MPI_Finalize();
  return 0;
}
