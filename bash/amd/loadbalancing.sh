#!/bin/bash
#SBATCH -N 2
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch
#SBATCH -J LB
#SBATCH -o LB.%J.out
#SBATCH -e LB.%J.err
#SBATCH --time=00:05:00
#SBATCH --mem=200G
#SBATCH --constraint=[rome]
cd /home/hongy0a/work/tlrmvm-dev
module load gcc openmpi/3.0.0/gcc-6.4.0
export OMP_NUM_THREADS=8
which mpirun

mpirun -np 32 --map-by L3cache --rank-by core \
/home/hongy0a/work/tlrmvm-dev/bin/seismicbin 149 150 0.001 256 1000 32 0 0 /home/hongy0a/scratch/seismic_data/compressdata/

