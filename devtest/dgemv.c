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

#ifdef USE_INTEL
#include <mkl.h>
#endif 

#ifdef USE_NVIDIA
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#endif

#if defined(USE_NEC)
#include <cblas.h>
#endif

#if defined(USE_AMD)
//#include <blis.h>
#include <cblas.h>
// #include <mkl.h>
//void bli_thread_set_num_threads( dim_t n_threads );
#endif

#ifdef USE_DOUBLE
#define real_t double
#else
#define real_t float
#endif 

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

void writestat(FILE *pf, double *y, int arrlen, char *desc){
  fprintf(pf, "%s\n",desc);
  for(int i=0; i<arrlen; i++){
    fprintf(pf,"%.3e",y[i]);
    if(i < arrlen-1) fprintf(pf," ");
    else fprintf(pf,"\n");
  }
}

void saveresults(int m, int n, double *timearr, double *presstat, double *bandwithstat, int arrlen, char *type){
  FILE *pf;
  char filename[100];
  memset(filename, 0, 100 * sizeof(char));
#ifdef USE_DOUBLE
  sprintf(filename, "log/M%dN%d_double_%s.txt", m, n, type);
#else
  sprintf(filename, "log/M%dN%d_single_%s.txt", m, n, type);
#endif 
  pf = fopen(filename, "w");
  fprintf(pf, "M\n");
  fprintf(pf, "%d\n", m);
  fprintf(pf, "N\n");
  fprintf(pf, "%d\n", n);
  fprintf(pf, "%s\n", "Precision");
#ifdef USE_DOUBLE
  fprintf(pf, "double\n");
#else
  fprintf(pf, "single\n");
#endif
  writestat(pf, timearr, arrlen, "Time");
  writestat(pf, presstat, arrlen, "Relative Error");
  writestat(pf, bandwithstat, arrlen, "Bandwith(GBytes/s)");
  fprintf(pf, "Exptype\n%s\n",type);
  fclose(pf);
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

#ifdef USE_NVIDIA

void checkcudaerror(cudaError_t st){
  if(st != cudaSuccess){
    printf("cuda status failed. \n");
    exit(0);
  }
}
void checkcudastatus(cublasStatus_t st){
  if(st != CUBLAS_STATUS_SUCCESS){
    printf("cuda status failed. \n");
    exit(0);
  }
}
#endif 

int main(int argc, const char* argv[])
{
  /**
   * @brief read input and init pointers
   * 
   */

  real_t *A, *y, *ynaive, *x;
  int m, n, nruns;
  int warmup = 10;
  real_t alpha, beta;
  char exptype[20];
  char filepath[100];
  printf ("\n This benchmark computes real vector y=alpha*A*x+beta*y, "
  "where A is matrix, y and x are vectors alpha and beta are scalars\n\n");
  alpha = 1.0;
  beta = 0.0;
  if(strcmp(argv[1], "fixed") == 0){
    assert(argc == 6);
    m = atoi(argv[2]);
    n = atoi(argv[3]);
    nruns = atoi(argv[4]);
    strcpy(exptype, argv[5]);
  }else if(strcmp(argv[1], "range") == 0){
    assert(argc == 6);
  }else{
    printf("not recognized input size mode, exit. \n\n");
    exit(0);
  }

  printf(" 1) m : %d n: %d alpha: %f beta: %f nruns: %d\n\n", m, n, alpha, beta, nruns);



/**
 * @brief 
 * allocate memory
 */

#ifdef USE_INTEL
  A = (real_t *)mkl_malloc( (long)m*(long)n*sizeof( real_t ), 64 );
  x = (real_t *)mkl_malloc( (long)n*sizeof( real_t ), 64 );
  y = (real_t *)mkl_malloc( (long)m*sizeof( real_t ), 64 );
  ynaive = (real_t *)mkl_malloc( (long)m*sizeof( real_t ), 64 );
  if (A == NULL || x == NULL || y == NULL) {
    printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
    mkl_free(A);
    mkl_free(x);
    mkl_free(y);
    return 1;
  }
  //int threads = mkl_get_max_threads();
  int threads = 0;
  char machinename[150];
  memset(machinename, 0, 150);
  gethostname(machinename,150);
  #ifdef USE_DOUBLE
  printf(" 2) use intel, mkl use %d threads and double precision on %s.\n\n", threads, machinename);
  #else
  printf(" 2) use intel, mkl use %d threads and single precision on %s.\n\n", threads, machinename);
  #endif
#endif 


#if defined(USE_NEC) || defined(USE_AMD)
  // malloc host 
  A = (real_t *)malloc( (long)m*(long)n*sizeof( real_t ));
  x = (real_t *)malloc( (long)n*sizeof( real_t ));
  y = (real_t *)malloc( (long)m*sizeof( real_t ));
  ynaive = (real_t *)malloc( (long)m*sizeof( real_t ));
  if (A == NULL || x == NULL || y == NULL) {
    printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
    free(A);
    free(x);
    free(y);
    return 1;
  }
  //int threads = omp_get_max_threads();
  int threads = 0;
  char machinename[150];
  memset(machinename, 0, 150);
  gethostname(machinename,150);
  #ifdef USE_DOUBLE
      #ifdef USE_NEC
          printf(" 2) use nec, nlc use %d threads and double precision on %s.\n\n", threads, machinename);
      #else
          printf(" 2) use amd, blis use %d threads and double precision on %s.\n\n", threads, machinename);
      #endif
  #else
      #ifdef USE_NEC
          printf(" 2) use nec, nlc use %d threads and single precision on %s.\n\n", threads, machinename);
      #else
          printf(" 2) use amd, blis use %d threads and single precision on %s.\n\n", threads, machinename);
      #endif
  #endif
#endif 

#ifdef USE_NVIDIA 
  cudaSetDevice(0); 
  // malloc host 
  A = (real_t *)malloc( (long)m*(long)n*sizeof( real_t ));
  x = (real_t *)malloc( (long)n*sizeof( real_t ));
  y = (real_t *)malloc( (long)m*sizeof( real_t ));
  ynaive = (real_t *)malloc( (long)m*sizeof( real_t ));
  cudaError_t cudaStat;
  cublasStatus_t stat;
  cublasHandle_t handle;
  real_t *d_A, *d_y, *d_x;
  // malloc device
  cudaStat = cudaMalloc ((void**)&d_A, (long)m*(long)n*sizeof(real_t));
  checkcudaerror(cudaStat);
  cudaStat = cudaMalloc ((void**)&d_x, (long)n*sizeof(real_t));
  checkcudaerror(cudaStat);
  cudaStat = cudaMalloc ((void**)&d_y, (long)m*sizeof(real_t));
  checkcudaerror(cudaStat);
  if (d_A == NULL || d_x == NULL || d_y == NULL) {
    printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
    return 1;
  }
  stat = cublasCreate(&handle);
  checkcudastatus(stat);
  char machinename[150];
  memset(machinename, 0, 150);
  gethostname(machinename,150);
  #ifdef USE_DOUBLE
  printf(" 2) use nvidia %s and double precision on %s.\n\n", exptype, machinename);
  #else
  printf(" 2) use nvidia %s and single precision on %s.\n\n", exptype, machinename);
  #endif 
#endif 



/**
 * @brief init naive implementation and stat info.
 * 
 */
  printf (" 3) Intializing matrix data with random number range from 0 to 1.\n\n");
  Initdata_cpu(A, x, y, m, n);
  naive_impl(A, x, ynaive, m, n);
  printf (" 4) Finish init, start to test, nruns is %d warmup is 10 rounds. \n\n", nruns);

  double *timestat = (double*)malloc(sizeof(double)*nruns);
  double *presstat = (double*)malloc(sizeof(double)*nruns);
  double *bandwithstat = (double*)malloc(sizeof(double)*nruns);


/**
 * @brief gemv various interface.
 * 
 */
#if defined(USE_INTEL) || defined(USE_NEC) || defined(USE_AMD)
for(int nr=0; nr<nruns+warmup; nr++){
  double stime = 0.0, etime = 0.0, executiontime = 0.0;
  Initdata_cpu(A, x, y, m, n);
  stime = gettime();
#ifdef USE_DOUBLE
  cblas_dgemv(CblasColMajor, CblasNoTrans, 
    m, n, alpha, A, m, x, 1, beta, y, 1);
#else 
  cblas_sgemv(CblasColMajor, CblasNoTrans, 
    m, n, alpha, A, m, x, 1, beta, y, 1);
#endif 
  etime = gettime();
  if(nr < warmup) continue;
  executiontime = etime - stime;
  double bd = sizeof(real_t)*((long)m*(long)n+m+n)/(executiontime * 1e9);
  printf("bd iteration %d Time %f Bandwidth %f \n", nr-warmup, executiontime, bd);
  
  timestat[nr-warmup] = executiontime;
  bandwithstat[nr-warmup] = bd;
  presstat[nr-warmup] = checkcorrectness(y,ynaive, m,n);
}
#endif 



#ifdef USE_NVIDIA
for(int nr=0; nr<nruns+warmup; nr++){
  double stime = 0.0, etime = 0.0, executiontime = 0.0;
  cudaMemcpy(d_A, A, (long)m*(long)n * sizeof(real_t), cudaMemcpyDefault);
  cudaMemcpy(d_x, x, (long)n * sizeof(real_t), cudaMemcpyDefault);
  cudaMemcpy(d_y, y, (long)m * sizeof(real_t), cudaMemcpyDefault);
  cudaDeviceSynchronize();
  stime = gettime();
#ifdef USE_DOUBLE
  cublasDgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_A, m, d_x, 1, &beta, d_y, 1);
#else 
  cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_A, m, d_x, 1, &beta, d_y, 1);
#endif 
  cudaDeviceSynchronize();
  etime = gettime();
  executiontime = etime-stime;
  cudaMemcpy(A, d_A, (long)m*(long)n  * sizeof(real_t), cudaMemcpyDefault);
  cudaMemcpy(x, d_x, (long)n * sizeof(real_t), cudaMemcpyDefault);
  cudaMemcpy(y, d_y, (long)m * sizeof(real_t), cudaMemcpyDefault);
  cudaDeviceSynchronize();
  if(nr < warmup) continue;
  
  double bd = sizeof(real_t)*((long)m*(long)n+m+n)/(executiontime * 1.0e9);
  timestat[nr-warmup] = executiontime;
  bandwithstat[nr-warmup] =  bd;
  printf("bd it %d %f \n", nr-warmup, bd);
  presstat[nr-warmup] = checkcorrectness(y,ynaive, m,n);
}
#endif



/**
 * @brief free space and write to file
 * 
 */
#ifdef USE_INTEL
  mkl_free(A);
  mkl_free(x);
  mkl_free(y);
//   double meanpres = mean(presstat,nruns);
//   saveresults(m,n,timestat, presstat, bandwithstat, nruns, exptype);
//   printf (" 5) mean precision: %.3e, deallocating memory and write results to files. \n\n", meanpres);
#endif

#if defined(USE_NEC) || defined(USE_AMD)
  free(A);
  free(x);
  free(y);
//   double meanpres = mean(presstat,nruns);
//   saveresults(m,n,timestat, presstat, bandwithstat, nruns, exptype);
//   printf (" 5) mean precision: %.3e, deallocating memory and write results to files. \n\n", meanpres);
#endif

#ifdef USE_NVIDIA
  free(A);
  free(x);
  free(y);
  cudaFree(d_A);
  cudaFree(d_x);
  cudaFree(d_y);
  double meanpres = mean(presstat,nruns);
  saveresults(m,n,timestat, presstat, bandwithstat, nruns, exptype);
  printf (" 5) mean precision: %.3e, deallocating memory and write results to files. \n\n", meanpres);
#endif 

  free(timestat);
  free(presstat);
  free(bandwithstat);
  return 0;
}
