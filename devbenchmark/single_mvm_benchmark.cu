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
 
 
 #include <cuda.h>
 #include <cuda_runtime.h>
 #include <cuda_runtime_api.h>
 #include <cublas_v2.h>
 
  
 #define real_t float
 
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
    sprintf(filename, "M%dN%d_single_%s.txt", m, n, type);
    pf = fopen(filename, "w");
    fprintf(pf, "M\n");
    fprintf(pf, "%d\n", m);
    fprintf(pf, "N\n");
    fprintf(pf, "%d\n", n);
    fprintf(pf, "%s\n", "Precision");
    fprintf(pf, "single\n");
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
    printf(" 2) use nvidia %s and single precision on %s.\n\n", exptype, machinename);



    /**
    * @brief init naive implementation and stat info.
    * 
    */
    Initdata_cpu(A, x, y, m, n);
    printf (" 3) Intializing matrix data with random number range from 0 to 1.\n\n");
    naive_impl(A, x, ynaive, m, n);
    printf (" 4) Finish init, start to test, nruns is %d warmup is 10 rounds. \n\n", nruns);

    double *timestat = (double*)malloc(sizeof(double)*nruns);
    double *presstat = (double*)malloc(sizeof(double)*nruns);
    double *bandwithstat = (double*)malloc(sizeof(double)*nruns);

    cudaMemcpy(d_A, A, (long)m*(long)n * sizeof(real_t), cudaMemcpyDefault);
    cudaMemcpy(d_x, x, (long)n * sizeof(real_t), cudaMemcpyDefault);
    cudaMemcpy(d_y, y, (long)m * sizeof(real_t), cudaMemcpyDefault);
    for(int nr=0; nr<nruns+warmup; nr++){
        double stime = 0.0, etime = 0.0, executiontime = 0.0;   
        cudaDeviceSynchronize();
        stime = gettime();
        cublasSgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_A, m, d_x, 1, &beta, d_y, 1);
        cudaDeviceSynchronize();
        etime = gettime();
        executiontime = etime-stime;
        if(nr < warmup) continue;
        timestat[nr-warmup] = executiontime;
        bandwithstat[nr-warmup] =  sizeof(real_t)*((long)m*(long)n+m+n)/(executiontime * 1.0e9) ; 
    }
    cudaMemcpy(A, d_A, (long)m*(long)n  * sizeof(real_t), cudaMemcpyDefault);
    cudaMemcpy(x, d_x, (long)n * sizeof(real_t), cudaMemcpyDefault);
    cudaMemcpy(y, d_y, (long)m * sizeof(real_t), cudaMemcpyDefault);
 
    free(A);
    free(x);
    free(y);
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    double meanpres = mean(presstat,nruns);
    saveresults(m,n,timestat, presstat, bandwithstat, nruns, exptype);
     
    free(timestat);
    free(presstat);
    free(bandwithstat);
    return 0;
 }