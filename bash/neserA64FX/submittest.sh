#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --partition=kestrel
#SBATCH --account=i1002
#SBATCH -J test
#SBATCH -o test.%J.out
#SBATCH --time=00:05:00
source /etc/profile
module load cray-mvapich2_nogpu_sve/2.3.5
export LD_LIBRARY_PATH=/opt/slurm/default/lib:$LD_LIBRARY_PATH

cd /project/k1524/users/hongy0a/tlrmvm-dev/
echo "==========================="

# srun -n 2 lscpu
srun -n 2 ./build/fujitsu_astrodriver
