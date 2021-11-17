#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=debug
#SBATCH -J GTest_matrix
#SBATCH -t 00:01:00

#OpenMP settings:
export OMP_NUM_THREADS=1

module swap PrgEnv-cray PrgEnv-gnu

########################################################################
## Command to Compile the Coce : 
## CC=cc CXX=CC cmake -DBUILD_BENCHMARK=ON -DUSE_COMPILER_BLAS=ON ..
#########################################################################

cd /project/k1524/users/hongy0a/tlrmvm-dev/build

#run the application:
srun --hint=nomultithread --ntasks=1 --ntasks-per-node=1 --ntasks-per-socket=1 \
--cpus-per-task=1 --ntasks-per-core=1 --mem-bind=v,local --cpu-bind=threads \
./GTest_matrix
