#include "tlrmvm_common.h"
#include "mpi.h"

int main(){
    tlrmvm::Matrix<float>(100,100,100);
    MPI_Init(NULL,NULL);
    MPI_Finalize();
    return 0;
}