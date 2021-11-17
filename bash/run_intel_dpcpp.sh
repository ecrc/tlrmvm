#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=20
#SBATCH --partition=batch
#SBATCH -J Intel
#SBATCH -o Intel.%J.out
#SBATCH --time=00:05:00
#SBATCH --mem=100G
#SBATCH --constraint=[cascadelake]
#SBATCH --mail-user=hongyx1993@gmail.com
#SBATCH --mail-type=END

. /home/hongy0a/work/tlrmvm-dev/bash/ibexinteloneapi.sh 

cd /home/hongy0a/work/tlrmvm-dev/build
export MAVISDATASET=/home/hongy0a/scratch/dataset/mavis/output
# mpirun -np 2 ls 
mpirun -np 2 --map-by socket:PE=20 --rank-by core ./DPCPP --M=4802 --N=19078 --nb=256 --acc=0.0001 --mavisid=000 --datafolder=$MAVISDATASET