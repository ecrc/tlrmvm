/**
 * @copyright (c) 2020- King Abdullah University of Science and
 *                      Technology (KAUST). All rights reserved.
 **/



// c
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
#include <cblas.h>

#include "util.h"
#include "ctlrmvm.h"


int main(int argc, char * argv[]){
    // MPI initialization
    MPI_Init(&argc, &argv);
    TLRMVMArgs args;
    ReadArgs(&args, argc, argv);
    GetPaddingSize(&args.paddingM, &args.paddingN, args.gM, args.gN, args.nb);
    ZigzagMapping(args.groupamount, args.groupid, args.matstartid, args.matendid, 
    args.zigzag, &args.workmatid, &args.workmatamount);
    //zigzag
    // if(args.grouprank == 0){
    //     sleep(args.groupid);
    //     PrintIntMatrix(args.workmatid, 1, args.workmatamount, 1);
    // }
    complex_t *** Au, *** Av, **yu, **yv, **x, **y, **yfinal;
    Au = (complex_t ***)malloc(sizeof(complex_t**) * args.workmatamount);
    Av = (complex_t ***)malloc(sizeof(complex_t**) * args.workmatamount);
    yu = (complex_t **)malloc(sizeof(complex_t*) * args.workmatamount);
    yv = (complex_t **)malloc(sizeof(complex_t*) * args.workmatamount);
    x = (complex_t **)malloc(sizeof(complex_t*) * args.workmatamount);
    y = (complex_t **)malloc(sizeof(complex_t*) * args.workmatamount);
    yfinal = (complex_t **)malloc(sizeof(complex_t*) * args.workmatamount);
    LoadSeismicData(&args, Au, Av, yu, yv, x, y, yfinal);
    // zigzag check
    // if(args.groupid == 1){
    //     sleep(args.worldrank);
    //     printf("matid on %d \n", args.groupid);
    //     for(int i=0; i<args.workmatamount; i++){
    //         printf("%d ", args.workmatid[i]);
    //     }
    //     printf("\n");
    //     int fqi = 0;
    //     if(args.grouprank == 0) PrintIntMatrix(args.gtilerank[0], args.mt_g, args.nt_g, args.mt_g);
    //     printf("\n");
    //     PrintIntMatrix(args.localtilerank[0], args.mt_g, args.nt_l, args.mt_g);
    //     printf("\n");
    //     printf("zigzag %d merge %d startid %d endid %d groupid %d groupsize %d workmatamount %d nruns %d mtg %d ntg %d ntl %d\n", 
    //     args.zigzag, args.mergephase, args.matstartid, args.matendid, 
    //     args.groupid, args.groupsize, args.workmatamount, args.nruns, args.mt_g, args.nt_g, args.nt_l);
    //     printf("======end======\n");
    // }
    PrepareOffset(&args);
    batchctlrmvm(&args, Au, Av, yu, yv, x, y, yfinal);
    MPI_Finalize();
    return 0;
}
