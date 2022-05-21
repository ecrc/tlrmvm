#!/bin/bash

./install/test/Bench_hip_constrank --M=9801 --N=9801 --threshold=0.001 \
--datafolder=$WORK_ROOT/compresseddata --nb=256 \
--problem=Mode4_Ordernormal_Mck_freqslice_100 --streams=10 --loopsize=200