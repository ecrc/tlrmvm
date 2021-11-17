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
#include "tlrmvm/Tlrmvm.h"

using namespace std;
using namespace tlrmat;
using namespace tlrmvm;
using std::complex;
#define SComplexMat Matrix<complex<float>> 

using ::testing::Pointwise;
using ::testing::NanSensitiveFloatEq;

vector<string> g_command_line_arg_vec;


TEST(SEISMICPCMATRIX, Constructor){
    SeismicPCMatrix seismicpcmat("/datawaha/ecrc/hongy0a/seismic/compressdata", 
    "0.001", 256, 0, 9801, 9801);
    Matrix<complex<float>> densemat = seismicpcmat.GetDense();
    densemat.Tofile("cppdensemat.bin");
    Matrix<complex<float>> xmat = seismicpcmat.GetX();
    xmat.Tofile("xmat.bin");
    Matrix<complex<float>> yv = seismicpcmat.Phase1();
    yv.Tofile("yvout.bin");
    Matrix<complex<float>> yu = seismicpcmat.Phase2();
    Matrix<complex<float>> y = seismicpcmat.Phase3();
    y.Tofile("algoy.bin");
    xmat.Tofile("xres.bin");
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