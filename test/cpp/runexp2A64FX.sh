# for NEC, for some reason the proof of concept matrix is running very slow,
# one can try to by pass the logic so that you can get speed up.

OMP_NUM_THREADS=8 mpirun -ve 0-1 -np 2 \
./install/test/ex2mpitlrmvm_float --M=4802 --N=19078 \
--errorthreshold=0.0001 --problemname=mavis_000_R \
--datafolder=$WORK_ROOT --nb=256 --loopsize=5000
#135 us 1138.03 GB/s

OMP_NUM_THREADS=8 mpirun -ve 0-1 -np 2 \
./install/test/ex2mpitlrmvm_complexfloat --M=9801 --N=9801 \
--errorthreshold=0.001 --problemname=SeismicFreq100 \
--datafolder=$WORK_ROOT --nb=256 --loopsize=5000
# 262us, 1025.71 GB/s


