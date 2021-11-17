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
    int mpirank, mpisize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    if(Printinfo && mpirank == 0){
        cout << "==========Input Info==========" << endl;
        cout << "mpisize " << mpisize << endl;
        cout << "datafolder " << datafolder << endl;
        cout << "nb " << nb << endl;
        cout << "error threshold " << acc << endl;
        cout << "idstart " << idstart << endl;
        cout << "idend " << idend << endl;
        cout << "==============================" << endl;
        Printinfo = false;
    }
}

/**************************************
*   Test Function for TileMatrix Class
***************************************/

TEST(TileMatrix, Constructor){
    string datafolder, acc;
    size_t nb;
    int idstart, idend;
    ReadArgs(datafolder, nb, acc, idstart, idend);
    unordered_map<string,int> idmaps;
    for(int i=idstart; i<idend; i++){
        idmaps["TestRank"+to_string(i)] = i;
    }
    TileMatrix<complex<float>> tmat(DatasetType::Seismology, 9801, 9801, 
    datafolder, acc, nb, idmaps);
    // MPI_Finalize();
}

TEST(TileMatrix, ZigZagScheme_dividable){
    string datafolder, acc;
    size_t nb;
    int idstart, idend;
    ReadArgs(datafolder, nb, acc, idstart, idend);
    // change idstart and idend;
    idstart = 0;
    idend = 128;
    unordered_map<string,int> idmaps;
    for(int i=idstart; i<idend; i++){
        idmaps["TestRank"+to_string(i)] = i;
    }
    // MPI_Init(NULL,NULL);
    TileMatrix<complex<float>> tmat(DatasetType::Seismology, 9801, 9801, 
    datafolder, acc, nb, idmaps);
    tmat.LoadRank();
    tmat.ResourceInit(RAScheme::FIXED, 16, 8, 512.0, false, false);
    ASSERT_EQ(tmat.localids.size(), 128 / 16 * 8);
    for(int i=0; i<tmat.localids.size(); i++){
        ASSERT_EQ(tmat.localids[i], i + tmat.groupid * tmat.localids.size());
    }
}

TEST(TileMatrix, ZigZagScheme_undividable){
    string datafolder, acc;
    size_t nb;
    int idstart, idend;
    ReadArgs(datafolder, nb, acc, idstart, idend);
    // change idstart and idend;
    idstart = 0;
    idend = 35;
    unordered_map<string,int> idmaps;
    for(int i=idstart; i<idend; i++){
        idmaps["TestRank"+to_string(i)] = i;
    }
    // MPI_Init(NULL,NULL);
    TileMatrix<complex<float>> tmat(DatasetType::Seismology, 9801, 9801, 
    datafolder, acc, nb, idmaps);
    tmat.LoadRank();
    tmat.ResourceInit(RAScheme::FIXED, 16, 8, 512.0, false, false);
    ASSERT_EQ(tmat.localids.size(), 128 / 16 * 8);
    for(int i=0; i<tmat.localids.size(); i++){
        ASSERT_EQ(tmat.localids[i], i + tmat.groupid * tmat.localids.size());
    }
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
    MPI_Init(NULL,NULL);
    int mpirank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    for(int i=0; i<argc-1; i++){
        char tmp[200];
        sprintf(tmp, "%s", argv[i+1]);
        command_line_arg_vec.push_back(string(tmp));
    }
    testing::AddGlobalTestEnvironment(new MyTestEnvironment(command_line_arg_vec));
    ::testing::TestEventListeners& listeners =
    ::testing::UnitTest::GetInstance()->listeners();
    if (mpirank != 0) {
        delete listeners.Release(listeners.default_result_printer());
    }
    int ret = RUN_ALL_TESTS();
    MPI_Finalize();
    return ret;
}