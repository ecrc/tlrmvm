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
#include <complex.h>

#include <cblas.h>
#include "util.h"

#define complex_double double complex
#define complex_float float complex


void naive_impl(complex_float* A, complex_float* x, complex_float* y, int m, int n){
    for(int i=0; i<m; i++){
        y[i] = (complex_float)0.0;
        for(int j=0; j<n; j++){
            y[i] += A[(long)i + (long)j * (long)m] * x[j];
        }
    }
}

void Initdata_cpu(complex_float *A, complex_float* x, complex_float *y, int m, int n){
    memset(y, 0, sizeof(complex_float) * m);
    for(int i=0; i<n; i++){
        x[i] = 1.0 + 1.0*I;
    }
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            long tmp = (long)i * (long)n + (long)j;
            A[tmp] = 1.0 + 1.0 * I;
        }
    }
}


double checkcorrectness(complex_float *y, complex_float *ynaive, int dim){
    double rerr = 0.0;

    for(int i=0; i<dim; i++){
        complex_float tmp = y[i] - ynaive[i];
        rerr += creal(tmp) * creal(tmp) + cimag(tmp) * cimag(tmp);
    }

    return rerr;
}

int main(int argc, const char* argv[])
{
    int m,n;
    if(argc < 3){
        printf("usage ./bin m n\n");
        exit(1);
    }
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    
    complex_float * A = (complex_float*)malloc(sizeof(complex_float)*m*n);
    complex_float * x = (complex_float*)malloc(sizeof(complex_float)*n);
    complex_float * y = (complex_float*)malloc(sizeof(complex_float)*m);
    complex_float * y_naive = (complex_float*)malloc(sizeof(complex_float)*m);
    complex_float alpha = 1.0 + 0.0*I;
    complex_float beta = 0.0 + 0.0*I;
    Initdata_cpu(A,x,y,m,n);
    naive_impl(A,x,y_naive,m,n);
    int Nruns = 1000;
    for(int i=0; i<Nruns; i++){
        double stime = gettime();
        cblas_cgemv(CblasColMajor, CblasNoTrans, m, n, &alpha, A, m, x, 1, &beta, y, 1);
        double etime = gettime();
        double rerr = checkcorrectness(y,y_naive,m);
        double exetime = etime - stime;
        double bd = sizeof(complex_float)*((long)m*(long)n+m+n)/(exetime * 1e9);
        double flops = 2 * m * n;
        flops = flops / exetime * 1e-9;
        printf("i %d, time %f, bandwidth %f GB/s OP %f FLOPS/s\n", i, exetime, bd, flops);        
    }
    
    free(A);    
    free(x);
    free(y);

    return 0;
}
