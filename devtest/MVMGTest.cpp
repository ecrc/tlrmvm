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
#include "common/AppUtil.h"
#include "tlrmvm/Tlrmvm.h"
#include "benchmark/benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

using namespace std;
using namespace tlrmat;

using ::testing::Pointwise;
using ::testing::NanSensitiveFloatNear;

vector<string> g_command_line_arg_vec;

/******************************************************************************//**
 * Matrix Vector Multiplication(MVM) on Platform Test Suite
 * The file evaluates Matrix vector Multiplication on several Platforms.
 * We have several implementation for different architectures to achieve the best
 * perfomrance. We use Sustained Bandwidth as mertics here since MVM is a Memory-Bound
 * Algorithms.
 * Command example:
 * $./MVM_GTest  
 *******************************************************************************/


TEST(MVM,Float){

}

TEST(MVM,Double){
    
}

TEST(MVM,ComplexFloat){
    
}

TEST(MVM,ComplexDouble){
    
}



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
