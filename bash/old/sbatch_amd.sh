#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=64
#SBATCH --partition=batch
#SBATCH -J MyJob
#SBATCH -o MyJob.%J.out
#SBATCH -e MyJob.%J.err
#SBATCH --time=00:05:00
#SBATCH --mem=200G
#SBATCH --constraint=[rome]

source bash/config_amd.sh
export OMP_NUM_THREADS=64
#run the application:
mpirun -np 2 ./amdbin random 5000 20000 40 const 10 > constrank2mpi_64threads.log

