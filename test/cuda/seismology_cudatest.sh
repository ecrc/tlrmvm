#!/bin/bash

#
#  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
#                      All rights reserved.
#

# 1. Dense MVM Benchmark
./install/test/ex5dense 10000 10000 1000 complex
#Results,DenseMVM,complex,V100,median time 1000.42 us,Bandwidth 799.827 GB/s
#Results,DenseMVM,complex,A10040g,median time 578.56 us,Bandwidth 1383.02 GB/s
#Results,DenseMVM,complex,A10080g,median time 478.208 us,Bandwidth 1673.25 GB/s

# 2. TLR-MVM Benchmark [ ordering type : No ordering, NVIDIA]
./install/test/ex4cudagraph_csingle --M=9801 --N=9801 \
--errorthreshold=0.001 --problemname=Mode4_Ordernormal_Mck_freqslice_100 \
--datafolder=$WORK_ROOT/compresseddata --nb=256 --streamsize=20 --loopsize=5000
#Results,TLR-MVM,complex,normal,V100,median time 329.248 us,Bandwidth 803.526 GB/s
#Results,TLR-MVM,complex,normal,A10040g,median time 223.808 us,Bandwidth 1182.08 GB/s
#TODO:Check,Results,TLR-MVM,complex,normal,A10080g,median time 230.048 us,Bandwidth 1150.02 GB/s

# 2. TLR-MVM Benchmark [ ordering type : hilbert, NVIDIA]
./install/test/ex4cudagraph_csingle --M=9801 --N=9801 \
--errorthreshold=0.001 --problemname=Mode4_Orderhilbert_Mck_freqslice_100 \
--datafolder=$WORK_ROOT/compresseddata --nb=256 --streamsize=20 --loopsize=5000
#Results,TLR-MVM,complex,hilbert,V100,median time 107.328 us,Bandwidth 562.538 GB/s
#Results,TLR-MVM,complex,hilbert,A10040g,median time 95.264 us,Bandwidth 633.776 GB/s
#TODO:Check,Results,TLR-MVM,complex,hilbert,A10080g,median time 92.672 us,Bandwidth 651.503 GB/s
