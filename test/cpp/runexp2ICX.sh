#
#  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
#                      All rights reserved.
#

OMP_NUM_THREADS=28 mpirun -np 2 --map-by L3cache:PE=28 \
./install/test/ex2mpitlrmvm_complexfloat --M=9801 --N=9801 \
--errorthreshold=0.001 --problemname=SeismicFreq100 \
--datafolder=$WORK_ROOT --nb=256 --loopsize=5000
# 859us, 312.75 GB/s

# compile the repo with parallel intel mkl and ONLY run this binary.
# since only this binary uses threaded mkl.
# cmake command -DBLAS_VENDOR=Intel10_64lp
OMP_NUM_THREADS=56 ./install/test/ex0basic_densemvm 10000 10000 complex 5000
# on Intel Ice Lake (qaysar)
# median time 6880us
# Bandwidth 116.278 GB/s

OMP_NUM_THREADS=28 mpirun -np 2 --map-by L3cache:PE=28 \
./install/test/ex2mpitlrmvm_float --M=4802 --N=19078 \
--errorthreshold=0.0001 --problemname=mavis_000_R \
--datafolder=$WORK_ROOT --nb=256 --loopsize=5000
#400 us 384.0 GB/s

