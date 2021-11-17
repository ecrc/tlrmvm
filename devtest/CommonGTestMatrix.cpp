#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <memory.h>
#include <mpi.h>
#include <complex>
#include <string>
#include <vector>

#include "common/Common.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace std;
using namespace tlrmat;
using std::complex;
#define SComplexMat Matrix<complex<float>> 

using ::testing::Pointwise;
using ::testing::NanSensitiveFloatNear;
using ::testing::NanSensitiveFloatEq;
vector<string> g_command_line_arg_vec;


/**********************************
*   Test Function for Matrix Class
***********************************/

TEST(MATRIX, Constructor){
    SComplexMat mat(1000,1000); 
    mat.Fill(1);
    mat.Datavec()[0] = complex<float>(2,0);
    mat.RawPtr()[1] = complex<float>(3,0);
    ASSERT_EQ(mat.GetElem(0,0).real(), 2);
    ASSERT_EQ(mat.GetElem(0,0).imag(), 0);
    ASSERT_EQ(mat.GetElem(1,0).real(), 3);
    ASSERT_EQ(mat.GetElem(1,0).imag(), 0);
}

TEST(MATRIX, DeConstructor){
    SComplexMat *mat = new SComplexMat(1000,1000);
    delete mat;
}



TEST(MATRIX, GetRealPart){
    SComplexMat mat(1000,1000);
    complex<float> val(1.0,1.0);
    mat.Fill(val);
    float * realptr;
    GetRealPart<float>(mat, &realptr);
    float * checkptr;
    checkptr = new float[1000 * 1000];
    for(int i=0; i<1000 * 1000; i++) checkptr[i] = 1.0;
    auto comparator = 
    Pointwise(NanSensitiveFloatEq(1e-6),
    vector<float>(checkptr, checkptr + 1000 * 1000));
    for(int i=0; i<1000*1000; i++){
        ASSERT_EQ(checkptr[i], realptr[i]);
    }
    delete[] realptr;
    delete[] checkptr;
}

TEST(MATRIX, GetImagPart){
    SComplexMat mat(1000,1000);
    complex<float> val(1.0,1.0);
    mat.Fill(val);
    float * imagptr;
    GetImagPart<float>(mat, &imagptr);
    float * checkptr;
    checkptr = new float[1000 * 1000];
    for(int i=0; i<1000 * 1000; i++) checkptr[i] = 1.0;
    auto comparator = 
    Pointwise(NanSensitiveFloatEq(1e-6),
    vector<float>(checkptr, checkptr + 1000 * 1000));
    for(int i=0; i<1000*1000; i++){
        ASSERT_EQ(checkptr[i], imagptr[i]);
    }
    delete[] imagptr;
    delete[] checkptr;
}

// TEST(MATRIX, CopyToPointer){
//     SComplexMat mat(1000,1000);
//     complex<float> val(1.0,4.0);
//     mat.Fill(val);
//     float * imagptr, *realptr;
//     GetRealPart<float>(mat, &realptr);
//     GetImagPart<float>(mat, &imagptr);
//     complex<float> * dstptr;
//     mat.CopyToPointer(&dstptr);
//     for(int i=0; i<1000 * 1000; i++){
//         ASSERT_EQ(realptr[i], dstptr[i].real());
//         ASSERT_EQ(imagptr[i], dstptr[i].imag());
//     }
//     delete[] imagptr;
//     delete[] realptr;
//     delete[] dstptr;
// }

// TEST(MATRIX, CopyFromPointer){
//     SComplexMat mat(1000,1000,1000);
//     complex<float> *valptr = new complex<float>[1000*1000];
//     for(int i=0; i<1000*1000; i++) valptr[i] = complex<float>(4.0,1.0);
//     mat.CopyFromPointer(valptr);
//     float *realptr, *imagptr;
//     GetRealPart<float>(mat, &realptr);
//     GetImagPart<float>(mat, &imagptr);
//     for(int i=0; i<1000 * 1000; i++){
//         ASSERT_EQ(realptr[i], valptr[i].real());
//         ASSERT_EQ(imagptr[i], valptr[i].imag());
//     }
//     delete[] imagptr;
//     delete[] realptr;
//     delete[] valptr;
// }

TEST(MATRIX, Sum){
    Matrix<int> mat(1000,1000);
    mat.Fill(1);
    int val = mat.Sum();
    ASSERT_EQ(val, 1000*1000);
}

TEST(MATRIX, RowSum){
    Matrix<int> mat(1000,1000);
    mat.Fill(1);
    vector<int> rsum = mat.RowSum();
    for(int i=0; i<1000; i++){
        ASSERT_EQ(rsum[i], 1000);
    }
}

TEST(MATRIX, ColSum){
    Matrix<int> mat(1000,1000);
    mat.Fill(1);
    vector<int> csum = mat.ColSum();
    for(int i=0; i<1000; i++){
        ASSERT_EQ(csum[i], 1000);
    }
}

TEST(MATRIX, Transpose){
    int M = 1024;
    int N = 512;
    Matrix<int> mat(M,N);
    for(int i=0; i<M*N; i++){
        mat.RawPtr()[i] = i;
    }
    Matrix<int> mat_trans = mat.Transpose();
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            ASSERT_EQ(mat_trans.RawPtr()[i * N + j], i + j * M);
        }
    }
    ASSERT_EQ(mat_trans.Lda(), N);
    ASSERT_EQ(mat_trans.Row(), N);
    ASSERT_EQ(mat_trans.Col(), M);
}

TEST(MATRIX, Add){
    int M = 1024;
    int N = 512;
    Matrix<int> m1(M,N);
    Matrix<int> m2(M,N);
    m1.Fill(1);
    m2.Fill(2);
    Matrix<int> ret = m1 + m2;
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            ASSERT_EQ(ret.GetElem(i,j), 3);
        }
    }
}

TEST(MATRIX, Subtract){
    int M = 1024;
    int N = 512;
    Matrix<int> m1(M,N);
    Matrix<int> m2(M,N);
    m1.Fill(1);
    m2.Fill(2);
    Matrix<int> ret = m1 - m2;
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            ASSERT_EQ(ret.GetElem(i,j), -1);
        }
    }
}


TEST(MATRIX, Multiply){
    int M = 256;
    int N = 128;
    Matrix<complex<float>> m1(M,N);
    Matrix<complex<float>> m2(N,M);
    #pragma omp parallel for
    for(size_t i=0; i < M*N; i++){
        float a = (float)rand()/(float)RAND_MAX;
        float b = (float)rand()/(float)RAND_MAX;
        m1.RawPtr()[i] = complex<float>(a,b);
    }
    #pragma omp parallel for 
    for(size_t i=0; i < M*N; i++){
        float a = (float)rand()/(float)RAND_MAX;
        float b = (float)rand()/(float)RAND_MAX;
        m2.RawPtr()[i] = complex<float>(a,b);
    }
    Matrix<complex<float>> m3(M,N);
    m3 = m1 * m2;
    Matrix<complex<float>> ret(M,M);
    complex<float> *C = ret.RawPtr();
    for(int i=0; i<M; i++){
        for(int j=0; j<M; j++){
            complex<float> curval(0.0,0.0);
            for(int k=0; k<N; k++){
                curval += m1.RawPtr()[k * M + i] * m2.RawPtr()[k + j * N];
            }
            ret.RawPtr()[i + j * M] = curval;
        }
    }
    double neterr = NetlibError(m3.RawPtr(), ret.RawPtr(), ret.Datavec().size());
    ASSERT_LT(neterr, 1e-6);
}

TEST(MATRIX, Tofile){
    int M = 1024;
    int N = 512;
    Matrix<int> m1(M,N);
    Matrix<complex<float>> m2(M,N);
    m1.Fill(1);
    complex<float> val(1.0,0.0);
    m2.Fill(val);
    m1.Tofile("intm1.bin");
    m2.Tofile("complexm2.bin");
}

TEST(MATRIX, Fromfile){
    int M = 1024;
    int N = 512;
    Matrix<int> m1(M,N);
    m1.Fill(1.0);
    Matrix<int> m2;
    m1.Tofile("intm1.bin");
    m2.Fromfile("intm1.bin", M, N);
    double neterr = NetlibError(m1.RawPtr(), m2.RawPtr(), m1.Datavec().size());
    ASSERT_LT(neterr, 1e-6);
}

/**************************************
*   Basic Setting
***************************************/

class MyTestEnvironment : public testing::Environment {
 public:
  explicit MyTestEnvironment(const vector<string> &command_line_arg) {
    g_command_line_arg_vec = command_line_arg;
  }
};

int main(int argc, char **argv) {
  vector<string> command_line_arg_vec;
  testing::InitGoogleTest(&argc, argv);
  for(int i=0; i<argc-1; i++){
      char tmp[200];
      sprintf(tmp, "%s", argv[i+1]);
      command_line_arg_vec.push_back(string(tmp));
  }
  testing::AddGlobalTestEnvironment(new MyTestEnvironment(command_line_arg_vec));
  return RUN_ALL_TESTS();
}