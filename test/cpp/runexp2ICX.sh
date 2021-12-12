OMP_NUM_THREADS=28 mpirun -np 2 --map-by L3cache:PE=28 \
./install/test/ex2mpitlrmvm_complexfloat --M=9801 --N=9801 \
--errorthreshold=0.001 --problemname=SeismicFreq100 \
--datafolder=$WORK_ROOT --nb=256 --loopsize=5000
# 859us, 312.75 GB/s


OMP_NUM_THREADS=28 mpirun -np 2 --map-by L3cache:PE=28 \
./install/test/ex2mpitlrmvm_float --M=4802 --N=19078 \
--errorthreshold=0.0001 --problemname=mavis_000_R \
--datafolder=$WORK_ROOT --nb=256 --loopsize=5000
#400 us 384.0 GB/s