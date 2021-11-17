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


TEST(ASTROPCMATRIX, Constructor){
    AstronomyPCMatrix astropcmat("/datawaha/ecrc/hongy0a/astronomy/mavis/output/", 
    "0.0001", 128, "000", 4802, 19078);
    Matrix<float> densemat = astropcmat.GetDense();
    densemat.Tofile("cppdensemat.bin");
    Matrix<float> xmat = astropcmat.GetX();
    xmat.Tofile("xmat.bin");
    // Matrix<float> cppres = densemat * xmat;
    Matrix<float> yv = astropcmat.Phase1();
    yv.Tofile("yvout.bin");
    Matrix<float> yu = astropcmat.Phase2();
    // yu.Tofile("yuout.bin");
    Matrix<float> y = astropcmat.Phase3();
    y.Tofile("algoy.bin");
    xmat.Tofile("xres.bin");
    // cppres.Tofile("cppres.bin");
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