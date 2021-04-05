#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include "cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cublas_v2.h>     // if you need CUBLAS v2, include before magma.h
#include <cuda.h>
#include <nccl.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <memory.h>
#define real_t float

#define TESTING_CHECK( err )                                                 \
    do {                                                                     \
        magma_int_t err_ = (err);                                            \
        if ( err_ != 0 ) {                                                   \
            fprintf( stderr, "Error: %s\nfailed at %s:%d: error %lld: %s\n", \
                     #err, __FILE__, __LINE__,                               \
                     (long long) err_, magma_strerror(err_) );               \
            exit(1);                                                         \
        }                                                                    \
    } while( 0 )

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
 
 void Init(real_t *A, int m, int n){
    #pragma omp parallel for
    for(int i=0; i<m; i++){
      for(int j=0; j<n; j++){
        long tmp = (long)i * (long)n + (long)j;
        // A[tmp] = 1.0; //(real_t)rand() / (real_t)RAND_MAX;
        // A[tmp] = (real_t)rand() / (real_t)RAND_MAX;
        A[tmp] = 1.0;
      }
    }
  }
  
  __global__ 
  void reshuffle(real_t * y_in, real_t * y_out, 
    int Aurows, int Aucols,
    int Avrows, int Avcols,
    int m, int n, int nt, int mt, int nb, int size){
  
    unsigned int threadidx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int threadidy = threadIdx.y + blockDim.y * blockIdx.y;
    if(threadidx >= nt || threadidy >= Avrows) return;
  
    unsigned int out_col = threadidy / (nb/4) ;
    unsigned int out_row = threadidx;
    unsigned int k = threadidy % (nb/4);
    y_out[out_col* Aucols + out_row * (nb/4) + k] = 
    y_in[threadidx * Avrows + threadidy];
  }
  

double gettime(void)
{
  struct timeval tp;
  gettimeofday( &tp, NULL );
  return tp.tv_sec + 1e-6 * tp.tv_usec;
}

int main(int argc, char* argv[])
{
  real_t *A, *y, *y_final, *ynaive, *x;
  int m, n, nb, nruns;
  real_t *Au, *Av;
  real_t *yu, *yv;
  int warmup = 10;
  real_t alpha, beta;
  char exptype[20];
  char filepath[100];

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

  printf(" 1) m : %d n: %d nb: %d alpha: %f beta: %f nruns: %d\n\n", m, 
  n, nb, alpha, beta, nruns);

  MPI_Status mpistat;
  int rank, size;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  printf("worker %d online \n", rank);
  cudaSetDevice(rank); 
  cublasHandle_t handle;
  cublasStatus_t state;
  state = cublasCreate(&handle);
  
  // nccl init
  ncclUniqueId id;
  if (rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));
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

  cudaStream_t stream;
  CUDACHECK(cudaStreamCreate(&stream));

  /*************************
  * Phase 1 allocation
  **************************/

  // batch pointers
  float **h_d_phase1A, **h_d_phase1B, **h_d_phase1C;
  int phase1batch = n / nb / size;
  int phase1M = Avrows;
  int phase1N = nb;

  h_d_phase1A = (float**)malloc(phase1batch * sizeof(float*));
  h_d_phase1B = (float**)malloc(phase1batch * sizeof(float*));
  h_d_phase1C = (float**)malloc(phase1batch * sizeof(float*));

  float **d_phase1A, **d_phase1B, **d_phase1C;
  cudaMalloc((void**)&d_phase1A, phase1batch*sizeof(float*));
  cudaMalloc((void**)&d_phase1B, phase1batch*sizeof(float*));
  cudaMalloc((void**)&d_phase1C, phase1batch*sizeof(float*));


  // cpu data buffer
  Av = (float *)malloc( (long)Avrows*(long)Avcols*sizeof( float ));
  yv = (float *)malloc( (long)Avrows*(long)nt*sizeof( float ));
  x = (float *)malloc( (long)Avcols*sizeof( float ));
  Init(Av, Avrows, Avcols);
  Init(yv, Avrows, nt);
  Init(x, Avcols,1);

  float *Au_d, *Av_d, *yv_d, *A_d, *x_d, *y_d, *y_final_d, *ynaive_d;
  float * yu_d;

  CUDACHECK(cudaMalloc((void**)&Av_d, (long)Avrows*(long)Avcols*sizeof( float )) );
  CUDACHECK(cudaMalloc((void**)&yv_d, (long)Avrows*(long)nt*sizeof( float )) );
  CUDACHECK(cudaMalloc((void**)&x_d, (long)Avcols*sizeof( float )) );

  CUDACHECK(cudaMemcpy(Av_d, Av, (long)Avrows*(long)Avcols*sizeof(float), cudaMemcpyDefault));
  CUDACHECK(cudaMemcpy(x_d, x, (long)nt*(long)nb*sizeof(float), cudaMemcpyDefault));
  CUDACHECK(cudaMemcpy(yv_d, yv, (long)nt*(long)Avrows*sizeof(float), cudaMemcpyDefault));
  

  // cpu phase1 impl
  for(int i=0; i<nt; i++){
    for(int r=0; r<phase1M; r++){
      float tmpval = 0.0;
      for(int c=0; c<phase1N; c++){
        tmpval += Av[i*phase1M*nb + c * phase1M + r] * x[i*nb+c];
      }
      yv[i*phase1M+r] = tmpval;
    }
  }


  // calculate phase1 pointer array
  h_d_phase1A[0] = Av_d;
  h_d_phase1B[0] = x_d;
  h_d_phase1C[0] = yv_d;
  for(int i=1; i<nt; i++){
    h_d_phase1A[i] = h_d_phase1A[i-1] + phase1M * phase1N;
    h_d_phase1B[i] = h_d_phase1B[i-1] + phase1N;
    h_d_phase1C[i] = h_d_phase1C[i-1] + phase1M;
  }
  // copy phase1 pointer array to gpu
  cudaMemcpy(d_phase1A, h_d_phase1A, sizeof(float*) * phase1batch, cudaMemcpyDefault);
  cudaMemcpy(d_phase1B, h_d_phase1B, sizeof(float*) * phase1batch, cudaMemcpyDefault);
  cudaMemcpy(d_phase1C, h_d_phase1C, sizeof(float*) * phase1batch, cudaMemcpyDefault);


  /*******************
  * phase 2
  * *****************/
  // batch pointer for phase2
  float ** h_d_phase2A,** h_d_phase2B,** h_d_phase2C;
  int phase2batch = m / nb;
  int phase2M = nb;
  int phase2N = n / 4 / size;

  h_d_phase2A = (float**)malloc(phase2batch * sizeof(float*));
  h_d_phase2B = (float**)malloc(phase2batch * sizeof(float*));
  h_d_phase2C = (float**)malloc(phase2batch * sizeof(float*));

  float **d_phase2A, **d_phase2B, **d_phase2C;
  cudaMalloc((void**)&d_phase2A, phase2batch*sizeof(float*));
  cudaMalloc((void**)&d_phase2B, phase2batch*sizeof(float*));
  cudaMalloc((void**)&d_phase2C, phase2batch*sizeof(float*));
  // cpu data buffer
  Au = (float *)malloc( (long)Aurows*(long)Aucols*sizeof( float ));
  yu = (float *)malloc( (long)Aucols*(long)mt*sizeof( float ));
  y = (float *)malloc( (long)m*sizeof( float ));
  
  Init(Au, Aurows, Aucols);
  
  CUDACHECK(cudaMalloc((void**)&Au_d, (long)Aurows*(long)Aucols*sizeof( float )) );
  CUDACHECK(cudaMalloc((void**)&yu_d, (long)Aucols*(long)mt*sizeof( float )) );
  CUDACHECK(cudaMalloc((void**)&y_d, (long)m*sizeof( float )) );

  CUDACHECK(cudaMemcpy(Au_d, Au, (long)Aurows*(long)Aucols*sizeof(float), cudaMemcpyDefault));

  // calculate phase2 pointer array
  h_d_phase2A[0] = Au_d;
  h_d_phase2B[0] = yu_d;
  h_d_phase2C[0] = y_d;
  for(int i=1; i<mt; i++){
    h_d_phase2A[i] = h_d_phase2A[i-1] + phase2M;
    h_d_phase2B[i] = h_d_phase2B[i-1] + phase2N;
    h_d_phase2C[i] = h_d_phase2C[i-1] + phase2M;
  }
  // copy phase2 pointer array to gpu
  cudaMemcpy(d_phase2A, h_d_phase2A, sizeof(float*) * phase2batch, cudaMemcpyDefault);
  cudaMemcpy(d_phase2B, h_d_phase2B, sizeof(float*) * phase2batch, cudaMemcpyDefault);
  cudaMemcpy(d_phase2C, h_d_phase2C, sizeof(float*) * phase2batch, cudaMemcpyDefault);


  CUDACHECK(cudaMalloc((void**)&y_final_d, (long)m*sizeof( real_t )) );
  y_final = (float*)malloc((long)m * sizeof(float));

  // shuffle configuration
  int thread_x = 128;
  int thread_y = 1;
  int nbx =  nt / thread_x + (nt % thread_x != 0);
  int nby =  Avrows;
  dim3 dimBlock(thread_x, thread_y);
  dim3 dimGrid(nbx, nby);



  double timearr[1000];
  for(int i=0; i<1200; i++){
    cudaDeviceSynchronize();
    cublasSetStream(handle, stream);

    double t1 = gettime();

    cublasSgemmBatched(handle,
    CUBLAS_OP_N,CUBLAS_OP_N,
    phase1M, 1, phase1N, &alpha, 
    (const float**)d_phase1A, phase1M, 
    (const float**)d_phase1B, phase1N,
    &beta, 
    d_phase1C, phase1M, phase1batch);

    reshuffle<<<dimGrid,dimBlock,0,stream>>>
    (yv_d, yu_d, Aurows, Aucols, Avrows, Avcols, m, n, nt, mt, nb, size);

    // cudaDeviceSynchronize();
    // cudaError_t err = cudaGetLastError();
    // if(err != cudaSuccess){
    //   printf("error %s \n", cudaGetErrorString(err));
    // }
    // cudaMemcpy(yv, yv_d, sizeof(float) * Avrows*nt, cudaMemcpyDefault);
    // cudaMemcpy(yu, yu_d, sizeof(float) * Aucols*mt, cudaMemcpyDefault);

    // printf("Aucols %d mt %d\n", Aucols, mt);
    // for(int i=0; i<Aucols*mt; i++){
    //   if(yu[i] != 100.0){
    //     printf("value is %f\n", yu[i]);
    //   }
    // }
    // CUDACHECK(cudaDeviceSynchronize());

    cublasSgemmBatched(handle,
    CUBLAS_OP_N,CUBLAS_OP_N,
    phase2M, 1, phase2N, &alpha,
    (const float**)d_phase2A, Aurows,
    (const float**)d_phase2B, phase2N,
    &beta, 
    d_phase2C, phase2M, phase2batch);

    //CUDACHECK(cudaDeviceSynchronize());

    // cudaMemcpy(y, y_d, sizeof(float) * m, cudaMemcpyDefault);
    // cudaMemcpy(Au, Au_d, sizeof(float) * 10, cudaMemcpyDefault);
    // cudaMemcpy(yu, yu_d, sizeof(float) * 10, cudaMemcpyDefault);
    // if(rank == 0){
    //   for(int i=0; i<m; i++){
    //     if(y[i] != 500000.000000 / size){
    //       printf("rank %d y %f yu %f Au %f\n", i, y[i],yu[i],Au[i]);
    //       break;
    //     }
    //   }
    // }
    
    NCCLCHECK(ncclReduce((const void*)y_d, (void*)y_final_d, Aurows, ncclFloat, 
    ncclSum, 0, comm, stream));
    CUDACHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    double t2 = gettime();
    double final_time;
    double t2t1 = t2-t1;
    MPI_Reduce(&t2t1, &final_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // cudaMemcpy(y_final, y_final_d, sizeof(float) * m, cudaMemcpyDefault);
    // if(rank ==0){
    //   for(int yi=0; yi<m; yi++){
    //     if(y_final[yi] != 500000.000000){
    //       printf("error %d %f\n", yi, y_final[yi]);
    //       break;
    //     }
    //   }
    // }
    

    
    if(i < 200)continue;
    double bd = sizeof(float)*
    ( (long)m/4*(long)n+n+nt*Avrows + (long)m*(long)n/4+n+mt*Aucols+m)
    / (final_time * 1e9);
    if(rank == 0)
    printf("run %d time %f Bd %f\n", i-200, final_time, bd);
    timearr[i-200] = final_time;
  }
  double totalt = 0.0;
  for(int i=0; i<1000; i++){
    totalt += timearr[i];
  }
  // printf("avg time %f\n", totalt/80);
  
  // float *farr = new float[Avrows];
  // cudaMemcpy(farr, h_d_phase1C[0], sizeof(float) * Avrows, cudaMemcpyDefault);
  double exetime = totalt/1000.0;
  // double phase1bd = sizeof(float)*((long)phase1M *(long)phase1N*(long)phase1batch + 
  // phase1M + phase1N*nb)/ (exetime * 1e9);
  // double phase2bd = sizeof(float)*((long)phase2M*(long)phase2N*(long)phase2batch + 
  // phase2M*nb+phase2N) / (exetime * 1e9);
  double totalbd = sizeof(float)*
  ( (long)m/4*(long)n+n+nt*Avrows + (long)m*(long)n/4+n+mt*Aucols+m)
  / (exetime * 1e9);
  if(rank == 0)
  printf("time is %f bd %f\n", exetime,totalbd);
  

  


  // int i,j,k,index;
  // if (argc < 4){
  //     printf("usage a.out batchsize dimM dimN\n");
  //     exit(0);
  // }   






  // // Number of A,B,C matrix sets
  // int batch_count = atoi(argv[1]);
  // int dimM = atoi(argv[2]);
  // int dimN = atoi(argv[3]);
  // printf("bs %d m %d n %d\n", batch_count, dimM, dimN);
  // // Allocate host storage for batch_count A,B,C square matrices
  // float **A, **B, **C, **resC;
  // A = (float**)malloc(batch_count*sizeof(float*));
  // B = (float**)malloc(batch_count*sizeof(float*));
  // C = (float**)malloc(batch_count*sizeof(float*));
  // resC = (float**)malloc(batch_count*sizeof(float*));
  // for(i=0; i<batch_count; i++) {
  //     A[i] = (float*)malloc(dimM*dimN*sizeof(float));
  //     B[i] = (float*)malloc(dimN*sizeof(float));
  //     C[i] = (float*)malloc(dimM*sizeof(float));
  //     resC[i] = (float*)malloc(dimM*sizeof(float));
  // }
  // // Fill A,B diagonals with k*sin(i) data, C diagonal with k*cos(i)^2
  // // Matrices are arranged column major
  // for(k=0; k<batch_count; k++) {
  //     for(j=0; j<dimN; j++) {
  //         (B[k])[j] = (float)rand() / RAND_MAX;
  //         for(i=0; i<dimM; i++) {
  //             (C[k])[i] = 0.0;
  //             index = j*dimM + i;
  //             (A[k])[index] = (float)rand() / RAND_MAX;
  //         } // i  
  //     } // j
  // } // k

  // for(k=0; k<batch_count; k++){
  //     for(i=0; i<dimM; i++){
  //         float tmpval = 0.0;
  //         for(j=0; j<dimN; j++){
  //             index = j * dimM + i;
  //             tmpval += A[k][index] * B[k][j];
  //         }
  //         resC[k][i] = tmpval;
  //     }
  // }


  // // Create host pointer array to device matrix storage
  // float **d_A, **d_B, **d_C, **h_d_A, **h_d_B, **h_d_C;
  // cudaMalloc((void**)&d_A, batch_count*sizeof(float*));
  // cudaMalloc((void**)&d_B, batch_count*sizeof(float*));
  // cudaMalloc((void**)&d_C, batch_count*sizeof(float*));
  // h_d_A = (float**)malloc(batch_count*sizeof(float*));
  // h_d_B = (float**)malloc(batch_count*sizeof(float*));
  // h_d_C = (float**)malloc(batch_count*sizeof(float*));
  
  // for(i=0; i<batch_count; i++) {
  //     cudaMalloc((void**)&h_d_A[i], dimM*dimN*sizeof(float));
  //     cudaMalloc((void**)&h_d_B[i], dimN*sizeof(float));
  //     cudaMalloc((void**)&h_d_C[i], dimM*sizeof(float));
  // }

  // // Copy the host array of device pointers to the device
  // cudaMalloc((void**)&d_A, batch_count*sizeof(float*));
  // cudaMalloc((void**)&d_B, batch_count*sizeof(float*));
  // cudaMalloc((void**)&d_C, batch_count*sizeof(float*));
  // cudaMemcpy(d_A, h_d_A, batch_count*sizeof(float*), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_B, h_d_B, batch_count*sizeof(float*), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_C, h_d_C, batch_count*sizeof(float*), cudaMemcpyHostToDevice);
  

  // // Create cublas instance
  // cublasHandle_t handle;
  // cublasCreate(&handle);
  
  // // Set input matrices on device
  // for(i=0; i<batch_count; i++) {
  //     cublasSetMatrix(dimM, dimN, sizeof(float), A[i], dimM, h_d_A[i], dimM);
  //     cublasSetMatrix(dimN, 1, sizeof(float), B[i], dimN, h_d_B[i], dimN);
  //     cublasSetMatrix(dimM, 1, sizeof(float), C[i], dimM, h_d_C[i], dimM);
  // }

  // // cublasGetMatrix(dimM, 1, sizeof(float), h_d_C[0], dimM, C[0], dimM);
  // // for(j=0; j<dimM; j++) {
  // //     printf("C 0 value %f \n", C[0][j]);
  // // }

  // // Set matrix coefficients
  // float alpha = 1.0;
  // float beta  = 0.0;
  // double totaltime = 0.0;
  // double totalbd = 0.0;
  // for(int nr=0; nr<100; nr++){
  //     cudaDeviceSynchronize();
  //     double t1 = gettime();
  //     cublasSgemmBatched(handle,
  //     CUBLAS_OP_N, CUBLAS_OP_N,
  //     dimM, 1, dimN,
  //     &alpha,
  //     (const float**)d_A, dimM,
  //     (const float**)d_B, dimN,
  //     &beta,
  //     d_C, dimM,
  //     batch_count);
  //     cudaDeviceSynchronize();
  //     double t2 = gettime();
  //     if (nr < 20) continue;
  //     double exetime = t2-t1;
  //     double bd = sizeof(float)*((long)dimM*(long)dimN*batch_count+dimM+dimN*batch_count)/(exetime * 1.0e9);
  //     // printf("time %f bd is %f \n", t2 - t1, bd);
  //     totaltime += exetime;
  //     totalbd += bd;
  // }
  // printf("avg time %f avg bd %f\n", totaltime / 80.0, totalbd / 80.0);
  // // Retrieve result matrix from device
  // for(i=0; i<batch_count; i++)
  //     cublasGetMatrix(dimM, 1, sizeof(float), h_d_C[i], dimM, C[i], dimM);
  // double diff = 0.0;
  // for(k=0; k<batch_count; k++) {
  //     for(j=0; j<dimM; j++) {
  //         // printf("resC %f C %f \n",resC[k][j], C[k][j]);
  //         diff += abs(resC[k][j] - C[k][j]);
  //     }
  // }
  // diff /= batch_count * dimM;
  // printf("avg diff %f \n", diff);

  // phase1 cleanup
  // free(Av);
  // free(yv);
  // free(x);
  // for(int i=0; i<phase1batch; i++) {
  //   cudaFree(h_d_phase1A[i]);
  //   cudaFree(h_d_phase1B[i]);
  //   cudaFree(h_d_phase1C[i]);
  //   cudaFree(d_phase1A);
  //   cudaFree(d_phase1B);
  //   cudaFree(d_phase1C);
  // }
  // free(h_d_phase1A);
  // free(h_d_phase1B);
  // free(h_d_phase1C);
  // // phase2 cleanup


  cublasDestroy(handle);
  ncclCommDestroy(comm);
  MPI_Finalize();
  return 0;
}
