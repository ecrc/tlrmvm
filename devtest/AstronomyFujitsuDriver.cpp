#include "common.h"
#include "mpi.h"
#include <iostream>
using namespace std;
int main(){
    tlrmat::Matrix<float>(100,100,100);
    MPI_Init(NULL,NULL);
    int mpisize;
    int mpirank;
    MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    cout << "hello from " << mpirank << " worldsize " << mpisize << endl;
    MPI_Finalize();
    return 0;
}
