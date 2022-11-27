//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

/*
!  Content:
!      Intel(R) oneAPI Math Kernel Library (oneMKL) examples interface
!******************************************************************************/

#ifndef _MKL_EXAMPLE_H_
#define _MKL_EXAMPLE_H_

#include "mkl_types.h"
#include "mkl_cblas.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define COMMENTS ':'
#define MAX_STRING_LEN  512

#define MIN(a,b)  (((a) > (b)) ? (b) : (a))
#define MAX(a,b)  (((a) > (b)) ? (a) : (b))
#define ABS(a)    (((a) < 0) ? -(a) : (a))

#define GENERAL_MATRIX  0
#define UPPER_MATRIX    1
#define LOWER_MATRIX   -1

#define FULLPRINT  0
#define SHORTPRINT 1

#ifdef MKL_ILP64
#define INT_FORMAT "%lld"
#else
#define INT_FORMAT "%d"
#endif

/*
 * ===========================================
 * Prototypes for example program functions
 * ===========================================
 */

int GetIntegerParameters(FILE*, ...);
int GetIntegerParameters8(FILE*, ...);
int GetIntegerParameters16(FILE*, ...);
int GetIntegerParameters32(FILE*, ...);
int GetScalarsH(FILE*, ...);
int GetScalarsD(FILE*, ...);
int GetScalarsS(FILE*, ...);
int GetScalarsC(FILE*, ...);
int GetScalarsZ(FILE*, ...);
int GetCblasCharParameters(FILE *in_file, ...);
MKL_INT MaxValue(MKL_INT, MKL_INT*);
MKL_INT GetVectorI(FILE*, MKL_INT, MKL_INT*);
MKL_INT GetVectorI32(FILE*, MKL_INT, MKL_INT32*);
MKL_INT GetVectorS(FILE*, MKL_INT, float*, MKL_INT);
MKL_INT GetVectorC(FILE*, MKL_INT, MKL_Complex8*, MKL_INT);
MKL_INT GetVectorD(FILE*, MKL_INT, double*, MKL_INT);
MKL_INT GetVectorZ(FILE*, MKL_INT, MKL_Complex16*, MKL_INT);
MKL_INT GetArrayI8(FILE*, CBLAS_LAYOUT*, MKL_INT, MKL_INT*, MKL_INT*, MKL_INT8*, MKL_INT*);
MKL_INT GetArrayI16(FILE*, CBLAS_LAYOUT*, MKL_INT, MKL_INT*, MKL_INT*, MKL_INT16*, MKL_INT*);
MKL_INT GetArrayI32(FILE*, CBLAS_LAYOUT*, MKL_INT, MKL_INT*, MKL_INT*, MKL_INT32*, MKL_INT*);
MKL_INT GetArrayBF16(FILE*, CBLAS_LAYOUT*, MKL_INT, MKL_INT*, MKL_INT*, MKL_BF16*, MKL_INT*);
MKL_INT GetArrayH(FILE*, CBLAS_LAYOUT*, MKL_INT, MKL_INT*, MKL_INT*, MKL_F16*, MKL_INT*);
MKL_INT GetArrayS(FILE*, CBLAS_LAYOUT*, MKL_INT, MKL_INT*, MKL_INT*, float*, MKL_INT*);
MKL_INT GetArrayD(FILE*, CBLAS_LAYOUT*, MKL_INT, MKL_INT*, MKL_INT*, double*, MKL_INT*);
MKL_INT GetArrayC(FILE*, CBLAS_LAYOUT*, MKL_INT, MKL_INT*, MKL_INT*, MKL_Complex8*, MKL_INT*);
MKL_INT GetArrayZ(FILE*, CBLAS_LAYOUT*, MKL_INT, MKL_INT*, MKL_INT*, MKL_Complex16*, MKL_INT*);
MKL_INT GetBandArrayS(FILE*, CBLAS_LAYOUT*, MKL_INT, MKL_INT, MKL_INT, MKL_INT, float*, MKL_INT);
MKL_INT GetBandArrayD(FILE*, CBLAS_LAYOUT*, MKL_INT, MKL_INT, MKL_INT, MKL_INT, double*, MKL_INT);
MKL_INT GetBandArrayC(FILE*, CBLAS_LAYOUT*, MKL_INT, MKL_INT, MKL_INT, MKL_INT, MKL_Complex8*, MKL_INT);
MKL_INT GetBandArrayZ(FILE*, CBLAS_LAYOUT*, MKL_INT, MKL_INT, MKL_INT, MKL_INT, MKL_Complex16*, MKL_INT);
MKL_INT GetValuesI8(FILE*, MKL_INT8*, MKL_INT, MKL_INT, MKL_INT);
MKL_INT GetValuesI16(FILE*, MKL_INT16*, MKL_INT, MKL_INT, MKL_INT);
MKL_INT GetValuesI32(FILE*, MKL_INT32*, MKL_INT, MKL_INT, MKL_INT);
MKL_INT GetValuesI(FILE *, MKL_INT*, MKL_INT, MKL_INT);
MKL_INT GetValuesBF16(FILE *, MKL_BF16*, MKL_INT, MKL_INT, MKL_INT);
MKL_INT GetValuesH(FILE *, MKL_F16*, MKL_INT, MKL_INT, MKL_INT);
MKL_INT GetValuesC(FILE *, MKL_Complex8*, MKL_INT, MKL_INT, MKL_INT);
MKL_INT GetValuesZ(FILE *, MKL_Complex16*, MKL_INT, MKL_INT, MKL_INT);
MKL_INT GetValuesD(FILE *, double*, MKL_INT, MKL_INT, MKL_INT);
MKL_INT GetValuesS(FILE *, float*, MKL_INT, MKL_INT, MKL_INT);
float   GetValueH(MKL_F16);
MKL_F16 f2h(float x);
void PrintParameters( char*, ... );
void PrintVectorI(MKL_INT, MKL_INT*, char*);
void PrintVectorI32(MKL_INT, MKL_INT32*, char*);
void PrintVectorS(int, MKL_INT, float*, MKL_INT, char*);
void PrintVectorC(int, MKL_INT, MKL_Complex8*, MKL_INT, char*);
void PrintVectorD(int, MKL_INT, double*, MKL_INT, char*);
void PrintVectorZ(int, MKL_INT, MKL_Complex16*,  MKL_INT, char*);
void PrintArrayI8(CBLAS_LAYOUT*, int, int, MKL_INT*, MKL_INT*, MKL_INT8*, MKL_INT*, char*);
void PrintArrayI16(CBLAS_LAYOUT*, int, int, MKL_INT*, MKL_INT*, MKL_INT16*, MKL_INT*, char*);
void PrintArrayI32(CBLAS_LAYOUT*, int, int, MKL_INT*, MKL_INT*, MKL_INT32*, MKL_INT*, char*);
void PrintArrayBF16(CBLAS_LAYOUT*, int, int, MKL_INT*, MKL_INT*, MKL_BF16*, MKL_INT*, char*);
void PrintArrayH(CBLAS_LAYOUT*, int, int, MKL_INT*, MKL_INT*, MKL_F16*, MKL_INT*, char*);
void PrintArrayS(CBLAS_LAYOUT*, int, int, MKL_INT*, MKL_INT*, float*, MKL_INT*, char*);
void PrintArrayD(CBLAS_LAYOUT*, int, int, MKL_INT*, MKL_INT*, double*, MKL_INT*, char*);
void PrintArrayC(CBLAS_LAYOUT*, int, int, MKL_INT*, MKL_INT*, MKL_Complex8*, MKL_INT*, char*);
void PrintArrayZ(CBLAS_LAYOUT*, int, int, MKL_INT*, MKL_INT*, MKL_Complex16*, MKL_INT*, char*);
void PrintBandArrayS(CBLAS_LAYOUT*, int, MKL_INT, MKL_INT, MKL_INT, MKL_INT, float*, MKL_INT, char*);
void PrintBandArrayD(CBLAS_LAYOUT*, int, MKL_INT, MKL_INT, MKL_INT, MKL_INT, double*, MKL_INT, char*);
void PrintBandArrayC(CBLAS_LAYOUT*, int, MKL_INT, MKL_INT, MKL_INT, MKL_INT, MKL_Complex8*, MKL_INT, char*);
void PrintBandArrayZ(CBLAS_LAYOUT*, int, MKL_INT, MKL_INT, MKL_INT, MKL_INT, MKL_Complex16*, MKL_INT, char*);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _MKL_EXAMPLE_H_ */
