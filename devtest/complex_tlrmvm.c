/**
 * @copyright (c) 2020- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <memory.h>
#include <omp.h>
#include <mpi.h>
#include <complex.h>

#ifdef USE_INTEL or USE_AMDMKL
#include<mkl.h>
#endif 

#ifdef USE_AMD
#include <blis.h>
#endif

#ifdef USE_OPENBLAS or USE_NEC
#include <cblas.h>
#endif

#define real_t float _Complex


void Init(real_t * A, int m, int n){
  #pragma omp parallel for
  for(int i=0; i<m; i++){
    for(int j=0; j<n; j++){
      long tmp = (long)i * (long)n + (long)j;
      A[tmp] = 1.0;
    }
  }
}

float cargval(float _Complex v){
    float img = cimag(v);
    float real = creal(v);
    return sqrt(img*img+real*real);
}

double grand(){
    int num = 25;
    double x = 0;
    for(int i=0; i<num; i++){
        x += (double) rand() / RAND_MAX;
    }
    x -= num / 2.0;
    x /= sqrt(num);
    if (x < 0) x = -x;
    return x;
}

double gettime(void)
{
	struct timeval tp;
	gettimeofday( &tp, NULL );
	return tp.tv_sec + 1e-6 * tp.tv_usec;
}
void print_double_matrix(double *y, unsigned long int m, unsigned long int n,unsigned long int lda){
    if(lda == -1){
        lda = m;
    }
    for(unsigned long int i=0; i<m; i++){
        for(unsigned long int j=0; j<n; j++){
            printf("%.4f ",*(y + i + j*lda));
        }
        printf("\n");
    }
    printf("\n");
}
void print_float_matrix(real_t *y, unsigned long int m, unsigned long int n,unsigned long int lda){
    if(lda == -1){
        lda = m;
    }
    for(unsigned long int i=0; i<m; i++){
        for(unsigned long int j=0; j<n; j++){
            printf("%.4f ",*(y + i + j*lda));
        }
        printf("\n");
    }
    printf("\n");
}
void print_int_matrix(int *y, unsigned long int m ,unsigned long int n, unsigned long int lda){
    if(lda == -1){
        lda = m;
    }
    for(unsigned long int i=0; i<m; i++){
        for(unsigned long int j=0; j<n; j++){
            printf("%d ",*(y + i + j*lda));
        }
        printf("\n");
    }
    printf("\n");
}

void print_uint_matrix(unsigned long int *y, unsigned long int m ,unsigned long int n, unsigned long int lda){
    if(lda == -1){
        lda = m;
    }
    for(unsigned long int i=0; i<m; i++){
        for(unsigned long int j=0; j<n; j++){
            printf("%d ",*(y + i + j*lda));
        }
        printf("\n");
    }
    printf("\n");
}

void get_orignial_A(real_t *A, real_t *Au, real_t*Av, 
unsigned long int M, unsigned long int N, 
unsigned long int mt, unsigned long int nt, 
unsigned long int nb,
int* tilerank, int *Avrows, int *Aucols,
real_t *x, real_t *y_check,char *acc){
    real_t *tmpAu, *tmpAv, *tmpA, *tmpblockA, alpha, beta;
    alpha = 1.0; beta = 0.0;
    tmpblockA = (real_t*)malloc(sizeof(real_t)*nb*nb);
    unsigned long int av_rank_accumulator;
    unsigned long int au_rank_accumulator;
    unsigned long int avrows_accumulator = 0;
    unsigned long int aucols_accumulator = 0;
    au_rank_accumulator = 0;
    for(int i=0; i<nt; i++){
        av_rank_accumulator = 0;
        aucols_accumulator = 0;
        for(int j=0; j<mt; j++){
            // do calculation of au rank accumulator
            au_rank_accumulator = 0;
            for(int k=0; k<i; k++){
                au_rank_accumulator += tilerank[k*mt + j];
            }
            tmpAu = Au + nb * au_rank_accumulator + nb * aucols_accumulator;
            tmpAv = Av + av_rank_accumulator + nb * avrows_accumulator;
            cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            nb, nb, tilerank[i*mt+j], &alpha, tmpAu, nb, tmpAv, Avrows[i], &beta, tmpblockA, nb);
            // copy
            for(unsigned long int ii=0; ii<nb; ii++){
                for(unsigned long int ij=0; ij<nb; ij++){
                    unsigned long int gi = ii + i*nb;
                    unsigned long int gj = ij + j*nb;
                    A[gi*M+gj] = tmpblockA[ii*nb+ij];
                }
            }
            av_rank_accumulator += tilerank[i*mt+j];    
            aucols_accumulator += Aucols[j];
        }
        avrows_accumulator += Avrows[i];        
    }
    free(tmpblockA);
    // char orgA[100];
    // FILE *f;
    // sprintf(orgA, "./data/mck/cresults/originA_nb%d_acc%s_real.bin", nb, acc);
    // f = fopen(orgA, "wb");
    // float * abuffer = (float*)malloc(sizeof(float) * M * N);
    // for(int i=0; i<M; i++){
    //     for(int j=0; j<N; j++){
    //         abuffer[i*N+j] = creal(A[j*M+i]);
    //     }
    // }
    // fwrite(abuffer, sizeof(float), M*N, f);
    // fclose(f);

    // sprintf(orgA, "./data/mck/cresults/originA_nb%d_acc%s_imag.bin", nb, acc);
    // f = fopen(orgA, "wb");
    // for(int i=0; i<M; i++){
    //     for(int j=0; j<N; j++){
    //         abuffer[i*N+j] = cimag(A[j*M+i]);
    //     }
    // }
    // fwrite(abuffer, sizeof(float), M*N, f);
    // fclose(f);
    // free(abuffer);

    // // sgmv to get y_check 
    // for(int i=0; i<100; i++){
    //     double stime = gettime();
    //     cblas_cgemv(CblasColMajor, CblasNoTrans, 
    //     M, N, &alpha, A, M, 
    //     x, 1, &beta, y_check, 1);
    //     double etime = gettime();
    //     double exetime = etime - stime;
    //     double bts = sizeof(real_t)*((long)M*(long)N+M+N);
    //     double bd = bts /(exetime * 1e9);
    //     double flops = 2 * M * N;
    //     flops = flops / exetime * 1e-9;
    //     printf("GEMV: i %d, Total bytes %f MB, time %f seconds, bandwidth %f GB/s OP %f FLOPS/s\n", i, bts*1e-6, exetime, bd, flops);
    // }

    
}

int main(int argc, char * argv[]){

    // MPI initialization
    MPI_Status stat;
    int mpirank, mpisize;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    
    char inputtype[10];
    sprintf(inputtype, "%s", argv[1]);
    unsigned long int gM, gN, nb;
    unsigned long int constrank;// only useful in random mode
    char acc[10]; // only useful in mck mode
    char freq[10];
    char randtype[20];
    int isconstrank = 0;
    int usemck = 0;

    /************************************
    * MCK input / Random input
    * ***********************************/
    if(strcmp(inputtype,"mck") == 0){
        if(mpirank == 0) printf("\nWe are using mck data.\n");
        usemck = 1;
        sprintf(freq, "%s", argv[2]);
        sprintf(acc, "%s", argv[3]);
        gM = atol(argv[4]);gN = atol(argv[5]);nb = atol(argv[6]);
        // this gM, gN now is the real matrix size. We will calculate padding here.
        unsigned long int padding_m, padding_n;
        padding_m = gM;
        padding_n = gN;
        if(gM % nb != 0){
            padding_m = (gM / nb + 1) * nb;
        }
        if(gN % nb != 0){
            padding_n = (gN / nb + 1) * nb;
        }
        gM = padding_m;
        gN = padding_n;
    }else if(strcmp(inputtype, "random") == 0){
        // if(mpirank == 0) printf("\nWe are using random data.\n");
        gM = atol(argv[2]);gN = atol(argv[3]);nb = atol(argv[4]);
        unsigned long int padding_m, padding_n;
        padding_m = gM;
        padding_n = gN;
        if(gM % nb != 0){
            padding_m = (gM / nb + 1) * nb;
        }
        if(gN % nb != 0){
            padding_n = (gN / nb + 1) * nb;
        }
        gM = padding_m;
        gN = padding_n;
        sprintf(randtype, "%s", argv[5]);
        if(strcmp(randtype,"const") == 0){
            // printf("We are using const rank\n");
            isconstrank = 1;
            constrank = atoi(argv[6]);
        }
        }
    // dimension check
    if(gM % (nb) != 0 || gN % (nb) != 0){
        if(mpirank == 0){
            printf("\n\nM %d N %d nb %d mpisize %d is not supported.\n\n", gM, gN, nb, mpisize);
        }
        MPI_Finalize();
        exit(1);
    }
    // summary
    if(mpirank == 0){
        printf("\n\nInput Summary\n===========================\n"); 
        if(usemck){
            printf("M %d N %d nb %d mpisize %d mck rank.\n", gM, gN, nb, mpisize);
        }else{
            if(isconstrank){
                printf("M %d N %d nb %d mpisize %d const rank size %d.\n", gM, gN, nb, mpisize, constrank);
            }else if(strcmp(randtype, "small") == 0){
                printf("M %d N %d nb %d mpisize %d variable rank size close to small.\n", gM, gN, nb, mpisize);
            }else if(strcmp(randtype, "normal") == 0){
                printf("M %d N %d nb %d mpisize %d variable normal rank size.\n", gM, gN, nb, mpisize);
            }
        }
        printf("===========================\n");
    }
    // these pointers only exists at rank 0
    real_t *A_global, *Au_global, *Av_global, *x_global, *y_check;
    int *Avrows_global, *Aucols_global, *gtilerank;
    int mt_global = gM / nb;
    int nt_global = gN / nb;
    unsigned long int global_totalrank = 0;
    // deciding nt_local
    int nt_local = 0;
    for(int i=0; i<nt_global; i++){
        if(i % mpisize == mpirank) nt_local++;
    }
    unsigned long int lM = gM, lN = nb * nt_local;
    // local tilerank
    int *tilerank = (int*)malloc(sizeof(int) * mt_global * nt_local);

    if(mpirank == 0){
        /************************************ 
        * create a global example on rank 0, 
        * get true solution and distribute it
        * **********************************/
        // create global tilerank
        gtilerank = (int*)malloc(sizeof(int) * mt_global * nt_global);
        y_check = (real_t*)malloc(sizeof(real_t) * gM);
        if(usemck == 0){
            // if randtype is small pregenerate the max rank of each tile
            int *max_rank_small = (int*)malloc(sizeof(int) * mt_global*nt_global);
            for(int i=0; i<mt_global*nt_global;i++){
                max_rank_small[i] = (int)(grand() * (double)nb);
                if(max_rank_small[i] > nb) max_rank_small[i] = nb;
                if(max_rank_small[i] <= 0) max_rank_small[i] = 1;
            }
            // random tile
            for(int i=0; i< mt_global*nt_global; i++){
                // printf("i %d", i);
                if(isconstrank){
                    gtilerank[i] = (int)constrank;
                }else{
                    if(strcmp(randtype,"normal") == 0){
                        gtilerank[i] = (int)rand() % (int)nb;
                        if(gtilerank[i] == 0) gtilerank[i] = 1;
                    }else if(strcmp(randtype, "small") == 0){
                        gtilerank[i] = (int)rand() % max_rank_small[i];
                        if(gtilerank[i] == 0) gtilerank[i] = 1;
                    }else if(strcmp(randtype, "one") == 0){
                        gtilerank[i] = 1;
                    }
                }
            }
            free(max_rank_small);
        }else{
            // load mck rank
            char rankfilename[100];
#ifdef USE_MKL
        printf("loading ranking file \n");
        printf("rname /home/hongy0a/scratch/seismic_data/compressdata/R-Mck_freqslice%s_nb%d_acc%s.bin\n", freq, nb, acc);
        sprintf(rankfilename, "/home/hongy0a/scratch/seismic_data/compressdata/R-Mck_freqslice%s_nb%d_acc%s.bin",freq, nb, acc);
#else 
        sprintf(rankfilename, "./data/mck/compress/R-Mck_freqslice%s_nb%d_acc%s.bin",freq, nb, acc);
#endif
            printf("\n\nMck File: \n");
            printf("  Rankfile %s \n",rankfilename);
            FILE *f = fopen(rankfilename,"rb");
            fread(gtilerank, sizeof(int), mt_global*nt_global, f);
            fclose(f);
        }

        for(int i=0; i<mt_global * nt_global; i++){
            global_totalrank += gtilerank[i];
        }

#ifdef DEBUG
        printf("rank 0 gtilerank: \n");        
        print_int_matrix(gtilerank, mt_global, nt_global, mt_global);
        
#endif
        printf("Global global_totalrank %llu\n", global_totalrank);
        // genrating examples
        A_global = (real_t*)malloc(sizeof(real_t) * gM * gN);
        Au_global = (real_t*)malloc(sizeof(real_t) * global_totalrank * nb);
        Av_global = (real_t*)malloc(sizeof(real_t)* global_totalrank * nb);
        x_global = (real_t*)malloc(sizeof(real_t) * gN);
        if(usemck == 0){
            Init(Au_global, global_totalrank, nb);
            Init(Av_global, global_totalrank, nb);    
        }else{
            
            FILE *f;
            char Au_real_filename[100], Av_real_filename[100];
            char Au_imag_filename[100], Av_imag_filename[100];
            printf("use mck data\n");   
            float* Au_real_global = (float*)malloc(sizeof(float)*global_totalrank * nb);
            float* Au_imag_global = (float*)malloc(sizeof(float)*global_totalrank * nb);
            float* Av_real_global = (float*)malloc(sizeof(float)*global_totalrank * nb);
            float* Av_imag_global = (float*)malloc(sizeof(float)*global_totalrank * nb);
            printf(" here    \n");
#ifdef USE_MKL
        sprintf(Au_real_filename, "/home/hongy0a/scratch/seismic_data/compressdata/U_real_Mck_freqslice%s_nb%d_acc%s.bin", freq, nb, acc);
#else 
        sprintf(Au_real_filename, "./data/mck/compress/U_real_Mck_freqslice%s_nb%d_acc%s.bin", freq, nb, acc);
#endif 
            printf("  Au_real_tfile %s \n", Au_real_filename);
            fflush(stdout);
            f = fopen(Au_real_filename, "rb");
            fread(Au_real_global, sizeof(float), global_totalrank*nb, f);
            fclose(f);
            fflush(stdout);
#ifdef USE_MKL
        sprintf(Au_imag_filename, "/home/hongy0a/scratch/seismic_data/compressdata/U_imag_Mck_freqslice%s_nb%d_acc%s.bin",freq, nb, acc);
#else 
        sprintf(Au_imag_filename, "./data/mck/compress/U_imag_Mck_freqslice%s_nb%d_acc%s.bin",freq, nb, acc);
#endif
            printf("  Au_imag_tfile %s \n", Au_imag_filename);
            f = fopen(Au_imag_filename, "rb");
            fread(Au_imag_global , sizeof(float), global_totalrank*nb, f);
            fclose(f);
            for(int i=0; i<global_totalrank*nb; i++){
                Au_global[i] = Au_real_global[i] + Au_imag_global[i] * I;
            }
#ifdef USE_MKL
        sprintf(Av_real_filename, "/home/hongy0a/scratch/seismic_data/compressdata/V_real_Mck_freqslice%s_nb%d_acc%s.bin",freq, nb, acc);
#else 
        sprintf(Av_real_filename, "./data/mck/compress/V_real_Mck_freqslice%s_nb%d_acc%s.bin",freq, nb, acc);
#endif            
            printf("  Av_real_filename %s \n\n", Av_real_filename);
            f = fopen(Av_real_filename, "rb");
            fread(Av_real_global, sizeof(float), global_totalrank*nb, f);
            fclose(f);
#ifdef USE_MKL
        sprintf(Av_imag_filename, "/home/hongy0a/scratch/seismic_data/compressdata/V_imag_Mck_freqslice%s_nb%d_acc%s.bin",freq, nb, acc);
#else 
        sprintf(Av_imag_filename, "./data/mck/compress/V_imag_Mck_freqslice%d_nb%s_acc%s.bin",freq, nb, acc);
#endif
            printf("  Av_imag_filename %s \n\n", Av_imag_filename);
            f = fopen(Av_imag_filename,"rb");
            fread(Av_imag_global, sizeof(float), global_totalrank*nb, f);
            for(int i=0; i<global_totalrank*nb; i++){
                Av_global[i] = Av_real_global[i] + Av_imag_global[i] * I;
            }
            fclose(f);
            free(Au_real_global);
            free(Au_imag_global);
            free(Av_real_global);
            free(Av_imag_global);
        }
        Init(x_global, gN, 1);

#ifdef DEBUG
        // printf("Av_global \n");
        // print_float_matrix(x, gN, nb, totalrank);
        // printf("Au_global \n");
        // print_float_matrix(Au_global, totalrank, nb, totalrank);
        // printf("x_global \n");
        // print_float_matrix(x_global, gN, 1, gN);
#endif
        Avrows_global = (int*)malloc(sizeof(int)*nt_global);
        Aucols_global = (int*)malloc(sizeof(int)*mt_global);
        memset(Avrows_global, 0, sizeof(int) * nt_global);
        memset(Aucols_global, 0, sizeof(int) * mt_global);
        for(int i=0; i<nt_global; i++){
            for(int j=0; j<mt_global; j++){
                Avrows_global[i] += gtilerank[i*mt_global+j];
                Aucols_global[j] += gtilerank[i*mt_global+j];
            }
        }
        // get_orignial_A(
        //     A_global, Au_global, Av_global,
        //     gM, gN, mt_global, nt_global, nb,
        //     gtilerank, Avrows_global, Aucols_global,
        //     x_global, y_check, acc);
        MPI_Barrier(MPI_COMM_WORLD);
        for(int i=0; i<nt_global; i++){
            if(mpirank == 0){
                if(i % mpisize == 0){
                    int offset = i / mpisize;
                    memcpy(tilerank+offset * mt_global, 
                    gtilerank + i * mt_global, sizeof(int) * mt_global);
                }else{
                    int dst = i % mpisize;
                    int tag = i / mpisize;
                    MPI_Send(gtilerank + i*mt_global, 
                    mt_global, MPI_INT, dst, tag, MPI_COMM_WORLD);
                }
            }else{
                if(i % mpisize == mpirank){
                    int offset = i / mpisize;
                    MPI_Recv(tilerank+offset * mt_global, mt_global, MPI_INT,
                    0, offset, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
#ifdef DEBUG
        printf("rank 0 local rank: \n");
        print_int_matrix(tilerank, mt_global, nt_local, mt_global);
#endif
    }else{
        MPI_Barrier(MPI_COMM_WORLD);
        for(int i=0; i<nt_global; i++){
            if(i % mpisize == mpirank){
                int offset = i / mpisize;
                MPI_Recv(tilerank+offset * mt_global, mt_global, MPI_INT,
                0, offset, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
#ifdef DEBUG
        // sleep(mpirank);
        printf("rank %d local rank: \n", mpirank);
        print_int_matrix(tilerank, mt_global, nt_local, mt_global);
        fflush(stdout);
#endif
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    
    // after getting tilerank, allocate Au,Av,x,y
    // all the pointer below are locals
    real_t *Au, *Av, *yu, *yv, *x, *y, *y_final;
    unsigned long int totalrank = 0;
    for(int i=0; i<mt_global * nt_local; i++){
        totalrank += tilerank[i];
    }
    Au = (real_t*)malloc(sizeof(real_t) * totalrank * nb);
    Av = (real_t*)malloc(sizeof(real_t) * totalrank * nb);
    yu = (real_t*)malloc(sizeof(real_t) * totalrank);
    yv = (real_t*)malloc(sizeof(real_t) * totalrank);
    x = (real_t*)malloc(sizeof(real_t) * lN);
    y = (real_t*)malloc(sizeof(real_t) * gM);
    y_final = (real_t*)malloc(sizeof(real_t) * gM);
    int *Avrows, *Aucols;
    Avrows = (int*)malloc(sizeof(int)*nt_local);
    Aucols = (int*)malloc(sizeof(int)*mt_global);
    memset(Avrows, 0, sizeof(int) * nt_local);
    memset(Aucols, 0, sizeof(int) * mt_global);
    for(int i=0; i<nt_local; i++){
        for(int j=0; j<mt_global; j++){
            Avrows[i] += tilerank[i*mt_global+j];
            Aucols[j] += tilerank[i*mt_global+j];
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    real_t **Av_batch_pointers, **yv_batch_pointers;
    real_t **Au_batch_pointers, **yu_batch_pointers;
    Av_batch_pointers = (real_t**)malloc(sizeof(real_t*)*nt_local);
    yv_batch_pointers = (real_t**)malloc(sizeof(real_t*)*nt_local);
    Au_batch_pointers = (real_t**)malloc(sizeof(real_t*)*mt_global);
    yu_batch_pointers = (real_t**)malloc(sizeof(real_t*)*mt_global);
    Av_batch_pointers[0] = Av;
    yv_batch_pointers[0] = yv;
    for(int i=1; i<nt_local; i++){
        Av_batch_pointers[i] = Av_batch_pointers[i-1] + Avrows[i-1]*nb;
        yv_batch_pointers[i] = yv_batch_pointers[i-1] + Avrows[i-1];
    }
    Au_batch_pointers[0] = Au;
    yu_batch_pointers[0] = yu;
    for(int i=1; i<mt_global; i++){
        Au_batch_pointers[i] = Au_batch_pointers[i-1] + Aucols[i-1]*nb; 
        yu_batch_pointers[i] = yu_batch_pointers[i-1] + Aucols[i-1];
    }

    // copy Av
    real_t *tmpAv = Av, *tmpAu = Au;
    real_t *tmpAv_global, *tmpAu_global;
    if(mpirank == 0){
        tmpAv_global = Av_global;
    }
    for(int i=0; i<nt_global; i++){
        if(mpirank == 0){
            if(i % mpisize == 0){
                int offset = i / mpisize;
                memcpy(tmpAv, tmpAv_global, sizeof(real_t) * Avrows[offset]*nb);
                tmpAv += Avrows[offset]*nb;
                tmpAv_global += Avrows_global[i]*nb;
            }else{
                
                int dstrank = i % mpisize;
                int tag =  i/mpisize;
                MPI_Send(tmpAv_global, Avrows_global[i]*nb, MPI_C_COMPLEX, dstrank,
                tag, MPI_COMM_WORLD);
                tmpAv_global += Avrows_global[i]*nb;
            }
        }else{
            if(i % mpisize == mpirank){
                int tag = i / mpisize;
                MPI_Recv(tmpAv, Avrows[tag]*nb, MPI_C_COMPLEX, 0, tag, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
                tmpAv += Avrows[tag] * nb;
            }
        }
    }
    
#ifdef DEBUG
    // sleep(mpirank);
    if(mpirank == 0){
        printf("rank 0: global Av\n");
        tmpAv_global = Av_global;
        for(int i=0; i<nt_global; i++){
            print_float_matrix(tmpAv_global, Avrows_global[i], nb, Avrows_global[i]);    
            tmpAv_global += Avrows_global[i] *nb;
        }
    }
    printf("rank %d: print Av matrix \n", mpirank);
    tmpAv = Av;
    for(int i=0; i<nt_local; i++){
        print_float_matrix(tmpAv, Avrows[i], nb, Avrows[i]);
        tmpAv += Avrows[i] * nb;
    }
    fflush(stdout);
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    // copy Au
    if(mpirank == 0) tmpAu_global = Au_global;
    tmpAu = Au;
    for(int i=0; i<mt_global; i++){
        for(int j=0; j<nt_global; j++){
            if(mpirank == 0){
                int dst = j % mpisize;
                unsigned long int sendamount=(unsigned long int)nb * 
                gtilerank[j*mt_global+i];
                if(dst == 0){
                    memcpy(tmpAu, tmpAu_global, 
                    sizeof(real_t) * sendamount);
                    tmpAu += sendamount;
                }else{
                    int tag = j * mt_global + i;
#ifdef DEBUG
                    printf("send %d %d, %d tag %d amt %llu\n", i, j,
                    dst, tag, sendamount);
                    fflush(stdout);
#endif
                    MPI_Send(tmpAu_global, sendamount, MPI_C_COMPLEX, 
                    dst, tag, MPI_COMM_WORLD);
                }
                tmpAu_global += sendamount;
            }else{
                if(j % mpisize != mpirank) continue;
                int offset = j / mpisize;
                int tag = j*mt_global+i;
                unsigned long int sendamount = (unsigned long int) nb *
                tilerank[offset * mt_global + i];
#ifdef DEBUG
                printf("recv %d %d, tag %d amt %llu\n", i, j/mpisize, 
                tag, sendamount);
                fflush(stdout);
#endif
                MPI_Recv(tmpAu, sendamount, MPI_C_COMPLEX, 0, tag, 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                tmpAu += sendamount;
            }
        }
    }
#ifdef DEBUG
    // sleep(mpirank);
    if(mpirank == 0){
        printf("rank 0: global Au\n");
        tmpAu_global = Au_global;
        for(int i=0; i<mt_global; i++){
            print_float_matrix(tmpAu_global, nb, Aucols_global[i], nb);
            tmpAu_global += Aucols_global[i] *nb;
        }
    }
    printf("rank %d: print Au matrix \n", mpirank);
    tmpAu = Au;
    for(int i=0; i<mt_global; i++){
        print_float_matrix(tmpAu, nb, Aucols[i],nb);
        tmpAu += Aucols[i] * nb;
    }
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    real_t *tmpx, *tmpx_global;
    if(mpirank == 0) tmpx_global = x_global;
    tmpx = x;
    for(int i=0; i<nt_global; i++){
        if(mpirank == 0){
            if(i % mpisize == 0){
                memcpy(tmpx, tmpx_global, sizeof(real_t) * nb);
                tmpx += nb;
            }else{
                int dst = i % mpisize;
                int tag = i / mpisize;
                MPI_Send(tmpx_global, nb, MPI_C_COMPLEX, dst, 
                tag, MPI_COMM_WORLD);
            }
            tmpx_global += nb;
        }else{
            if(i % mpisize != mpirank) continue;
            int tag = i / mpisize;
            MPI_Recv(tmpx, nb, MPI_C_COMPLEX, 0, tag, MPI_COMM_WORLD, 
            MPI_STATUS_IGNORE);
            tmpx += nb;
        }
    }
#ifdef DEBUG
    // sleep(mpirank);
    if(mpirank == 0){
        printf("rank 0: global x\n");
        tmpx_global = x_global;
        for(int i=0; i<nt_global; i++){
            print_float_matrix(tmpx_global, nb, 1, nb);
            tmpx_global += nb;
        }
    }
    printf("rank %d: print x matrix \n", mpirank);
    tmpx = x;
    for(int i=0; i<nt_local; i++){
        print_float_matrix(tmpx, nb,1,nb);
        tmpx += nb;
    }
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    // preparing prefix for reshuffle
    int *tilerank_prefix_vbase, *tilerank_prefix_ubase;
    tilerank_prefix_vbase = (int*)malloc(sizeof(int) * mt_global * nt_local);
    tilerank_prefix_ubase = (int*)malloc(sizeof(int) * mt_global * nt_local);
    for(int i=0; i<nt_local; i++){
        for(int j=0; j<mt_global; j++){
            if(i == 0 && j == 0){
                tilerank_prefix_vbase[0] = 0;
            }else{
                unsigned long int curidx = i*mt_global+j;
                tilerank_prefix_vbase[curidx] = tilerank_prefix_vbase[curidx-1] + 
                tilerank[curidx-1];
            }
        }
    }
    for(int i=0; i<mt_global; i++){
        for(int j=0; j<nt_local; j++){
            if(i == 0 && j == 0){
                tilerank_prefix_ubase[0] = 0;
            }else{
                unsigned long int previ, prevj;
                if(j == 0){
                    prevj = nt_local-1;
                    previ = i-1;
                }else{
                    prevj = j-1;
                    previ = i;
                }
                unsigned long int curidx = j*mt_global+i;
                unsigned long int previdx = prevj*mt_global+previ;
                tilerank_prefix_ubase[curidx] = tilerank_prefix_ubase[previdx] + 
                tilerank[previdx];
            }
        }
    }
#ifdef DEBUG
    // sleep(mpirank);
    // printf("rank %d show tilerank and tilerank prefix \n", mpirank);
    // if(mpirank == 0)
    // print_int_matrix(gtilerank, mt_global, nt_global, mt_global);
    // printf("rank %d tilerank \n", mpirank);
    // print_int_matrix(tilerank, mt_global, nt_local, mt_global);
    // printf("rank %d tilerank vbase \n", mpirank);
    // print_int_matrix(tilerank_prefix_vbase, mt_global, nt_local, mt_global);
    // printf("rank %d tilerank ubase \n", mpirank);
    // print_int_matrix(tilerank_prefix_ubase, mt_global, nt_local, mt_global);
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    real_t alpha = 1.0;
    real_t beta = 0.0;
    if(mpirank == 0)
    printf("\nTime and Bandwidth Report:\n");
    // MPI_Finalize();
    // return 0;

    /*********************
    *  Main Loop
    * *******************/
    int Nruns = 3000;
    for(int iter=0; iter<Nruns;iter++){
    MPI_Barrier(MPI_COMM_WORLD);
    // double stime = gettime();
    double stime = MPI_Wtime();
    // start real computation

    // ftrace_region_begin("phase#1");
    /*********************
    *  Phase 1
    * *******************/
    #pragma omp parallel for 
    for(unsigned long int j=0; j<nt_local; j++){
        cblas_cgemv(CblasColMajor, CblasNoTrans, 
        Avrows[j], nb, &alpha, Av_batch_pointers[j], Avrows[j], 
        x+j*nb, 1, &beta, yv_batch_pointers[j], 1);
    }
    double sp1 = MPI_Wtime();
    // ftrace_region_end("phase#1");
    // sleep(mpirank);
    // printf("after phase 1 rank %d \n", mpirank);
    
    // for(int i=0; i<nt_local;i++){
    //     printf("x vector \n");
    //     print_float_matrix(x+i*nb, nb, 1, nb);
    //     printf("Av \n");
    //     print_float_matrix(Av_batch_pointers[i], Avrows[i], nb, Avrows[i]);
    //     printf("yv \n");
    //     print_float_matrix(yv_batch_pointers[i], Avrows[i], 1, Avrows[i]);
    // }
    // ftrace_region_begin("reshuffle");
    /*********************
     *  Reshuffle
     * *******************/
    #pragma omp parallel for
    for(unsigned long int i=0; i<nt_local; i++){
        for(unsigned long int j=0; j<mt_global; j++){
            unsigned long int ubases = tilerank_prefix_ubase[i*mt_global+j];
            unsigned long int vbases = tilerank_prefix_vbase[i*mt_global+j];
            for(int k=0; k<tilerank[i*mt_global+j]; k++){
                *(yu + ubases + k) = *(yv + vbases + k);
            }
        }
    }
    // ftrace_region_end("reshuffle");
    // sleep(mpirank);
    // printf("rank %d yu yv \n", mpirank);
    // print_float_matrix(yu, totalrank, 1, totalrank); 
    // print_float_matrix(yv, totalrank, 1, totalrank);

    double sp2 = MPI_Wtime();
    // ftrace_region_begin("phase#2");
    /*********************
     *  Phase 2
     * *******************/
    #pragma omp parallel for
    for(unsigned long int j=0; j<mt_global; j++){
        cblas_cgemv(CblasColMajor, CblasNoTrans, 
        nb, Aucols[j], &alpha, Au_batch_pointers[j], nb, 
        yu_batch_pointers[j], 1, &beta, y+j*nb, 1);
    }
    // ftrace_region_end("phase#2");
    /*********************
     *  Final Reduction
     * *******************/
    MPI_Reduce(y, y_final, gM, MPI_C_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    // double etime = gettime();
    double etime = MPI_Wtime();
    double execution_time = etime - stime; // execution time 
    double global_execution_time = execution_time;
    // MPI_Reduce(&execution_time, &global_execution_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    // MPI_Barrier(MPI_COMM_WORLD);
    /*************************
     *  Calculating Bandwidth
     * ***********************/
    if(mpirank == 0){
        double bandwidth;
        unsigned long int granknumber=0;
        if(isconstrank){
            granknumber = mt_global*nt_global*constrank;
        }else{
            for(int i=0; i<mt_global*nt_global;i++){
                granknumber += gtilerank[i];
            }
        }
        unsigned long int phase1 = granknumber*nb + gN + granknumber;
        unsigned long int shuffle = 2 * granknumber;
        unsigned long int phase2 = granknumber*nb + granknumber + gM;
        bandwidth = sizeof(real_t) * (phase1 + shuffle + phase2) / (global_execution_time * 1e9);
        double bts = sizeof(real_t) * (phase1 + shuffle + phase2) / 1e6;
        double flops = 4.0 * global_totalrank * nb;
        flops = flops / global_execution_time * 1e-9;
        printf("TLRMVM: %d Total Bytes %.3f MB Execution time %f phase1 %f phase2 %f phase 3 %f Bandwidth %f GB/s OP %f GFLOPS/s\n", iter, bts, global_execution_time, 
        sp1 - stime, sp2 - sp1, etime - sp2,
        bandwidth, flops);
    }
    } // iteration loop end
    /**************************************************************
     *  Check results with the one generated by uncompressed data
     * ************************************************************/
    if(mpirank == 0){
        // print_float_matrix(y_check, gM, 1, gM);
        // print_float_matrix(y_final, gM, 1, gM);
        real_t *y_diff = (real_t*)malloc(sizeof(real_t)*gM);
        for(unsigned long int i=0; i<gM; i++){
            y_diff[i] = (y_final[i] - y_check[i]);
            // printf("y diff %d %f \n", i, cabs(y_diff[i]));
        }
        float epsilon = cblas_scnrm2(gM, y_diff, 1);
        epsilon /= gM;


        float max_diff = 0.0, max_y_check = 0.0;
        // new calculation of error: see https://www.netlib.org/lapack/lug/node75.html
        for(int i=0; i<gM; i++){
            if(cabs(y_diff[i]) > max_diff) max_diff = cabs(y_diff[i]);
            if(cabs(y_check[i]) > max_y_check) max_y_check = cabs(y_check[i]);
        }
        
        // printf("print first 10 element of y_final \n");
        // print_float_matrix(y_final, 1, 10, 1);
        // printf("print first 10 element of y_check \n");
        // print_float_matrix(y_check, 1, 10, 1);
        // printf("print first 10 element of y_diff \n");
        // print_float_matrix(y_diff, 1, 10, 1);
        printf("\nAccuracy check:\n");
        printf("  1. Error based on netlib calculation: %.6e\n", max_diff / max_y_check);
        printf("     check %s \n", "https://www.netlib.org/lapack/lug/node75.html");
        printf("  2. Average of different vector sqrt of norm square: %.6e\n", epsilon);
        printf("\n\n");
    }
    


    /*******************************
     *  Clean up resources and exit
     * *****************************/
    if(mpirank == 0){
        free(gtilerank);
        free(y_check);
        free(A_global);
        free(Au_global);
        free(Av_global);
        free(x_global);
        free(Avrows_global);
        free(Aucols_global);

    }
    free(Au);
    free(Av);
    free(yu);
    free(yv);
    free(x);
    free(y);
    free(y_final);
    free(Avrows);
    free(Aucols);
    free(tilerank);
    free(Av_batch_pointers);
    free(Au_batch_pointers);
    free(yv_batch_pointers);
    free(yu_batch_pointers);
    free(tilerank_prefix_ubase);
    free(tilerank_prefix_vbase);
    MPI_Finalize();
    return 0;

}
