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
using ::testing::NanSensitiveFloatEq;

vector<string> g_command_line_arg_vec;
bool Printinfo = true;
void ReadArgs(string& datafolder, size_t &nb, string& acc, int &idstart, int &idend)
{
    assert(g_command_line_arg_vec.size() == 5);
    datafolder = g_command_line_arg_vec[0];
    nb = atol(g_command_line_arg_vec[1].c_str());
    acc = g_command_line_arg_vec[2];
    idstart = atoi(g_command_line_arg_vec[3].c_str());
    idend = atoi(g_command_line_arg_vec[4].c_str());
    if(Printinfo){
        cout << "==========Input Info==========" << endl;
        cout << "datafolder " << datafolder << endl;
        cout << "nb " << nb << endl;
        cout << "error threshold " << acc << endl;
        cout << "idstart " << idstart << endl;
        cout << "idend " << idend << endl;
        cout << "==============================" << endl;
        Printinfo = false;
    }
}


/**********************************
*   Test Function for TLRMVM CPU
***********************************/




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