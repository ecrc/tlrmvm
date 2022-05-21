# for NEC, for some reason the proof of concept matrix is running very slow,
# one can try to by pass the logic so that you can get speed up.


# 1. Dense MVM Benchmark
OMP_NUM_THREADS=8 ve_exec --node=1 ./install/test/ex0basic_densemvm 10000 10000 complex 100
#Results,DenseMVM,complex,NEC,median time 3478 us,Bandwidth 230.017 GB/s

# 2. TLR-MVM Benchmark [ ordering type : No ordering, NEC]
OMP_NUM_THREADS=8 mpirun -ve 1 -np 1 \
./install/test/ex2mpitlrmvm_complexfloat --M=9801 --N=9801 \
--errorthreshold=0.001 --problemname=Mck_freqslice_100 \
--datafolder=$WORK_ROOT --nb=256 --loopsize=5000
#Results,TLR-MVM,complex,normal,NEC,median time 326 us,Bandwidth 811.532 GB/s

# 2. TLR-MVM Benchmark [ ordering type : hilbert, NEC]
OMP_NUM_THREADS=8 mpirun -ve 1 -np 1 \
./install/test/ex2mpitlrmvm_complexfloat --M=9801 --N=9801 \
--errorthreshold=0.001 --problemname=Mode1_Orderhilbert_Mck_freqslice_100 \
--datafolder=$WORK_ROOT --nb=256 --loopsize=5000
#Results,TLR-MVM,complex,normal,NEC,median time 105 us,Bandwidth 575 GB/s

OMP_NUM_THREADS=8 mpirun -ve 0-1 -np 2 \
./install/test/ex2mpitlrmvm_complexfloat --M=9801 --N=9801 \
--errorthreshold=0.001 --problemname=SeismicFreq100 \
--datafolder=$WORK_ROOT --nb=256 --loopsize=5000
# 262us, 1025.71 GB/s


OMP_NUM_THREADS=8 mpirun -ve 0-1 -np 2 \
./install/test/ex2mpitlrmvm_float --M=4802 --N=19078 \
--errorthreshold=0.0001 --problemname=mavis_000_R \
--datafolder=$WORK_ROOT --nb=256 --loopsize=5000
#135 us 1138.03 GB/s