#
#  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
#                      All rights reserved.
#

# run on AMD Milan 2 sockets 1 cpu/socket

OMP_NUM_THREADS=8 mpirun -np 16 --map-by L3cache:PE=8 \
./install/test/ex2mpitlrmvm_float --M=4802 --N=19078 \
--errorthreshold=0.0001 --problemname=Sytheticfloat \
--datafolder=$WORK_ROOT --nb=256 --loopsize=1000

OMP_NUM_THREADS=8 mpirun -np 16 --map-by L3cache:PE=8 \
./install/test/ex2mpitlrmvm_complexfloat --M=9801 --N=9801 \
--errorthreshold=0.001 --problemname=Sytheticcomplex \
--datafolder=$WORK_ROOT --nb=256 --loopsize=1000


