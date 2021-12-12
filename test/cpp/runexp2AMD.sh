# run on AMD Milan 2 sockets 1 cpu/socket

OMP_NUM_THREADS=8 mpirun -np 16 --map-by L3cache:PE=8 \
./install/test/ex2mpitlrmvm_complexfloat --M=9801 --N=9801 \
--errorthreshold=0.001 --problemname=SeismicFreq100 \
--datafolder=$WORK_ROOT --nb=256 --loopsize=5000
# 379us, 708.84 GB/s


OMP_NUM_THREADS=8 mpirun -np 16 --map-by L3cache:PE=8 \
./install/test/ex2mpitlrmvm_float --M=4802 --N=19078 \
--errorthreshold=0.0001 --problemname=mavis_000_R \
--datafolder=$WORK_ROOT --nb=256 --loopsize=5000
#115 us 1335.66 GB/s