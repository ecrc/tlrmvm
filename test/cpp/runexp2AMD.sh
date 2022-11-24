#
#  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
#                      All rights reserved.
#

# AMD CPU Experiments

################################################
# Seismic Redatuming, datatype = single complex
################################################

# ENV Setting

#export MPI_HOME=/home/hongy0a/installation/openmpi-4.1.1/install
#export MPI_ROOT=/home/hongy0a/installation/openmpi-4.1.1/install
#export BLIS_ROOT=/home/hongy0a/installation/blis/install
#export LD_LIBRARY_PATH=${MPI_ROOT}/lib:${BLIS_ROOT}/lib:$LD_LIBRARY_PATH
#export PATH=${MPI_ROOT}/bin:$PATH
#export WORK_ROOT=/home/hongy0a/datawaha/seismic/compresseddata

# 1. Dense MVM Benchmark
OMP_NUM_THREADS=56 ./install/test/ex0basic_densemvm 10000 10000 complex 100
#Results,DenseMVM,complex,rome,median time 25366.6 us,Bandwidth 31.5376 GB/s
#Results,DenseMVM,complex,milan,median time 23170 us,Bandwidth 34.5 GB/s
#TODO:Results,DenseMVM,complex,milan-x,median time ? us,Bandwidth ? GB/s

# 2. TLR-MVM Benchmark [ ordering type : No ordering, rome]
# rome run: sbatch bash/benchmark_script/ibexrome_seismic.sh
#Results,TLR-MVM,complex,normal,rome,median time 317 us,Bandwidth 834.572 GB/s

# 2. TLR-MVM Benchmark [ ordering type : No ordering]
OMP_NUM_THREADS=8 mpirun -np 16 --map-by L3cache:PE=8 \
./install/test/ex2mpitlrmvm_complexfloat --M=9801 --N=9801 \
--errorthreshold=0.001 --problemname=Mck_freqslice_100 \
--datafolder=$WORK_ROOT --nb=256 --loopsize=5000
#Results,TLR-MVM,complex,normal,milan,median time 362 us,Bandwidth 730 GB/s
#TODO:Results,TLR-MVM,complex,normal,milan-x,median time ? us,Bandwidth ? GB/s

# 2. TLR-MVM Benchmark [ ordering type : hilbert, rome]
# rome run: sbatch bash/benchmark_script/ibexrome_seismic.sh
#Results,TLR-MVM,complex,hilbert,rome,median time 215 us,Bandwidth 280.819 GB/s

# 2. TLR-MVM Benchmark [ ordering type : hilbert]
OMP_NUM_THREADS=8 mpirun -np 16 --map-by L3cache:PE=8 \
./install/test/ex2mpitlrmvm_complexfloat --M=9801 --N=9801 \
--errorthreshold=0.001 --problemname=Mode1_Orderhilbert_Mck_freqslice_100 \
--datafolder=$WORK_ROOT --nb=256 --loopsize=5000
#Results,TLR-MVM,complex,hilbert,milan,median time 211 us, Bandwidth 286.143 GB/s
#TODO:Results,TLR-MVM,complex,hilbert,milan-x,median time ? us, Bandwidth ? GB/s


OMP_NUM_THREADS=8 mpirun -np 16 --map-by L3cache:PE=8 \
./install/test/ex2mpitlrmvm_float --M=4802 --N=19078 \
--errorthreshold=0.0001 --problemname=mavis_000_R \
--datafolder=$WORK_ROOT --nb=256 --loopsize=5000
#115 us 1335.66 GB/s