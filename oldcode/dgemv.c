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
#include "mpi.h"
#include <cblas.h>
#endif

#if defined(USE_AMD)
#include <blis.h>
//#include <cblas.h>
void bli_thread_set_num_threads( dim_t n_threads );
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
  #pragma omp parallel for
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

void outstat(double *y, int arrlen, char *desc){
  printf("%s\n",desc);
  for(int i=0; i<arrlen; i++){
    printf("%.3e",y[i]);
    if(i < arrlen-1) printf(" ");
    else printf("\n");
  }
}
void printmatrix(real_t *y, int m, int n){
  for(int i=0; i<m; i++){
      for(int j=0; j<n; j++){
          printf("%.1e ",*(y + i + j*m));
      }
      printf("\n");
  }
  printf("\n");
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
  #pragma omp parallel for
  for(int i=0; i<m; i++){
    y[i] = (real_t)0.0;
    for(int j=0; j<n; j++){
      y[i] += A[(long)i + (long)j * (long)m] * x[j];
    }
  }
}

void Init(real_t *A, int m, int n){
  #pragma omp parallel for
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      long tmp = (long)i * (long)n + (long)j;
      A[tmp] = 1.0; //(real_t)rand() / (real_t)RAND_MAX;
    }
  }
}

void Initdata_cpu(real_t *A, real_t* x, real_t *y, int m, int n){
  memset(y, 0, sizeof(real_t) * m);
  #pragma omp parallel for
  for(int i=0; i<n; i++){
    x[i] = (real_t)rand() / (real_t)RAND_MAX;
  }
  #pragma omp parallel for
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      long tmp = (long)i * (long)n + (long)j;
      A[tmp] = (real_t)rand() / (real_t)RAND_MAX;
    }
  }
}


double checkcorrectness(real_t *y, real_t *ynaive, int dim, int sumdim){
  double *rerr = (double*)malloc(sizeof(double)*dim);
  #pragma omp parallel for
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



int main(int argc, char* argv[])
{
  /**
   * @brief read input and init pointers
   * 
   */

  real_t *A, *y, *y_final, *ynaive, *x;
  int m, n, nb, nruns;
  real_t *Au, *Av;
  real_t *yu, *yv;
  int warmup = 10;
  real_t alpha, beta;
  char exptype[20];
  char filepath[100];
  printf ("\n This benchmark computes real vector y=alpha*A*x+beta*y, "
  "where A is matrix, y and x are vectors alpha and beta are scalars\n\n");
  alpha = 1.0;
  beta = 0.0;
  if(strcmp(argv[1], "fixed") == 0){
    assert(argc == 7);
    m = atoi(argv[2]);
    n = atoi(argv[3]);
    nb = atoi(argv[4]);
    nruns = atoi(argv[5]);
    strcpy(exptype, argv[6]);
  }else if(strcmp(argv[1], "range") == 0){
    assert(argc == 7);
  }else{
    printf("not recognized input size mode, exit. \n\n");
    exit(0);
  }

  printf(" 1) m : %d n: %d nb: %d alpha: %f beta: %f nruns: %d\n\n", m, n, nb, alpha, beta, nruns);


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
  int threads = mkl_get_max_threads();
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
  MPI_Status stat;
  int rank, size;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int dim_ok = 1;
  if ((m%nb)!=0) {
    if (rank==0) printf("ERROR: m has to be multiple of nb (mt = m/nb)\n");
    dim_ok = 0;
  }
  if ((n%(nb*size))!=0) {
    if (rank==0) printf("ERROR: n has to be multiple of nb * #MPI (nt = n/nb/size and Avcols = n/size)\n");
    dim_ok = 0;
  }
  if ((m%4)!=0) {
    if (rank==0) printf("ERROR: m has to be multiple of 4 (Avrows = m/4)\n");
    dim_ok = 0;
  }
  if ((n%(4*size))!=0) {
    if (rank==0) printf("ERROR: n has to be multiple of 4 * #MPI (Aucols = n/4/size)\n");
    dim_ok = 0;
  }
  
  if (!dim_ok) {
    MPI_Finalize();
    return 1;
  }
  
  long int Aurows = m, Aucols = n/4/size;
  long int Avrows = m/4, Avcols = n/size;
  int nt = n/nb/size, mt = m/nb;

  Au = (real_t *)malloc( (long)Aurows*(long)Aucols*sizeof( real_t ));
  yu = (real_t *)malloc( (long)Aucols*(long)mt*sizeof( real_t ));
  memset(yu, 0, sizeof(real_t) * (long)Aucols*(long)mt);
  Av = (real_t *)malloc( (long)Avrows*(long)Avcols*sizeof( real_t ));
  yv = (real_t *)malloc( (long)Avrows*(long)nt*sizeof( real_t ));
  memset(yv, 0, sizeof(real_t) * (long)Avrows*(long)nt);
  A = (real_t *)malloc( (long)m*(long)n*sizeof( real_t ));
  x = (real_t *)malloc( (long)Avcols*sizeof( real_t ));
  y = (real_t *)malloc( (long)m*sizeof( real_t ));
  y_final = (real_t *)malloc( (long)m*sizeof( real_t ));
  ynaive = (real_t *)malloc( (long)m*sizeof( real_t ));
  if (A == NULL || x == NULL || y == NULL) {
    printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
    free(A);
    free(x);
    free(y);
    return 1;
  }
  int threads = omp_get_max_threads();
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
  printf (" 3) Initializing matrix data with random number range from 0 to 1.\n\n");
  //Initdata_cpu(A, x, y, m, n);
  //naive_impl(A, x, ynaive, m, n);
  printf (" 4) Finish init, start to test, nruns is %d warmup is %d rounds. \n\n", nruns, warmup);

  double *timestat = (double*)malloc(sizeof(double)*nruns);
  double *presstat = (double*)malloc(sizeof(double)*nruns);
  double *bandwithstat = (double*)malloc(sizeof(double)*nruns);


/**
 * @brief gemv various interface.
 * 
 */
#if defined(USE_INTEL) || defined(USE_NEC) || defined(USE_AMD)
for(int nr=0; nr<nruns+warmup; nr++){
  if (rank == 0)
     printf("nrun %d\n", nr+1);
  double stime = 0.0, etime = 0.0, executiontime = 0.0;
  //Initdata_cpu(A, x, y, m, n);
  Init(x, Avcols, 1);
  Init(Au, Aurows, Aucols);
  //Init(yu, Aucols, mt);
  Init(Av, Avrows, Avcols);
  //Init(yv, Avrows, nt);
  MPI_Barrier(MPI_COMM_WORLD);
  stime = gettime();
#ifdef USE_DOUBLE
  cblas_dgemv(CblasColMajor, CblasNoTrans, 
    m, n, alpha, A, m, x, 1, beta, y, 1);
#else 
   
  //printf (" 5) I am in single. \n\n");
  // PHASE 1
 //printmatrix(Av, Avrows, Avcols);
 //printmatrix(x, Avcols, 1);
 //printmatrix(yv, Avrows, nt);
  #pragma omp parallel for
  for(int j=0; j<nt; j++){
      //printf("Av matvec %d\n", j+1);
      cblas_sgemv(CblasColMajor, CblasNoTrans, 
        Avrows, nb, alpha, Av+j*nb*Avrows, Avrows, 
        x+j*nb, 1, beta, yv+j*Avrows, 1);
  }
 //printmatrix(Av, Avrows, Avcols);
 //printmatrix(x, Avcols, 1);
 //printmatrix(yv, Avrows, nt);

  // GATHERV
  //MPI_Gather( sendarray, 100, MPI_FLOAT, rbuf, 100, MPI_INT, root, MPI_COMM_WORLD); 
  //MPI_Gather( yv+j*Avrows, nb/4, MPI_FLOAT, rbuf, nb/4, MPI_IFLOAT, 0, MPI_COMM_WORLD); 

 //printmatrix(yu, Aucols, mt);

  // Local
#if 0
#endif
  #pragma omp parallel for
  for(int i=0; i<mt; i++){
     for(int j=0; j<nt; j++){
         for(int k=0; k<nb/4; k++){
             *(yu + j*nb/4 + i*Aucols + k) = *(yv + i*nb/4 + j*Avrows + k);
         }
     }
  }

 //printmatrix(yu, Aucols, mt);

  // PHASE 2
  #pragma omp parallel for
  for(int j=0; j<mt; j++){
      //printf("Au matvec %d\n", j+1);
      cblas_sgemv(CblasColMajor, CblasNoTrans, 
        nb, Aucols, alpha, Au+j*nb, Aurows, yu+j*Aucols, 
        1, beta, y+j*nb, 1);
  }

  MPI_Reduce(y, y_final, Aurows, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);


#endif 
  etime = gettime();
  if(nr < warmup) continue;
  executiontime = etime - stime;
  timestat[nr-warmup] = executiontime;
  bandwithstat[nr-warmup] = sizeof(real_t)*((long)m/4*(long)n+n+nt*Avrows + (long)m*(long)n/4+n+mt*Aucols+m)/(executiontime * 1e9);
  //presstat[nr-warmup] = checkcorrectness(y,ynaive, m,n);
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
  timestat[nr-warmup] = executiontime;
  bandwithstat[nr-warmup] =  sizeof(real_t)*((long)m*(long)n+m+n)/(executiontime * 1.0e9) ; 
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
  double meanpres = mean(presstat,nruns);
  saveresults(m,n,timestat, presstat, bandwithstat, nruns, exptype);
  printf (" 5) mean precision: %.3e, deallocating memory and write results to files. \n\n", meanpres);
#endif

#if defined(USE_NEC) || defined(USE_AMD)
  // write results
  char filename[200];
  sprintf(filename, "necresult_m%d_n%d.bin", m, n);
  FILE* file = fopen(filename, "wb");
  if(!fwrite(y_final, sizeof(real_t), m, file)){
    printf("fwrite fails\n");
  }
  for(int i=0; i<10; i++){
    printf("yfinal res is %f\n", y_final[i]);
  }
  fclose(file);
  free(A);
  free(x);
  free(y);
  free(y_final);
  free(Au);
  free(yu);
  free(Av);
  free(yv);
  //double meanpres = mean(presstat,nruns);
  //saveresults(m,n,timestat, presstat, bandwithstat, nruns, exptype);
  //printf (" 5) mean precision: %.3e, deallocating memory and write results to files. \n\n", meanpres);
  if (rank == 0) {
      outstat(timestat, nruns, "Time (s)");
      outstat(bandwithstat, nruns, "BW (GBytes/s)");
  }
  MPI_Finalize();
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
