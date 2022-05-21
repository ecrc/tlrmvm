#!/bin/bash

# 2. TLR-MVM Benchmark [ ordering type : No ordering, NVIDIA]
./install/test/ex4cudagraph_csingle_mp --M=9801 --N=9801 \
--errorthreshold=0.001 --problemname=Mode4_Mck_freqslice_100 \
--datafolder=$WORK_ROOT/compresseddata --nb=256 --streamsize=20 --loopsize=5000
#Results,TLR-MVM,complex,normal,V100,median time 216.544 us,Bandwidth 635.049 GB/s
#Results,TLR-MVM,complex,normal,A10040g,median time 172.48 us,Bandwidth 797.287 GB/s
#TODO:Check,Results,TLR-MVM,complex,normal,A10080g,median time 169.216 us,Bandwidth 814.52 GB/s

# 2. TLR-MVM Benchmark [ ordering type : hilbert, NVIDIA]
./install/test/ex4cudagraph_csingle --M=9801 --N=9801 \
--errorthreshold=0.001 --problemname=Mode4_Orderhilbert_Mck_freqslice_100 \
--datafolder=$WORK_ROOT/compresseddata --nb=256 --streamsize=20 --loopsize=5000
#Results,TLR-MVM,complex,hilbert,V100,median time 98.976 us,Bandwidth 348.879 GB/s
#Results,TLR-MVM,complex,hilbert,A10040g,median time 85.856 us,Bandwidth 402.193 GB/s
#TODO:Check,Results,TLR-MVM,complex,hilbert,A10080g,median time 90.144 us,Bandwidth 383.061 GB/s

