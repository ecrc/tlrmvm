#
#  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
#                      All rights reserved.
#

#/bin/bash
./install/test/ex4cudagraph_float --M=4802 --N=19078 \
--errorthreshold=0.0001 --problemname=mavis_000_R \
--datafolder=$WORK_ROOT --nb=256 --streamsize=20 --loopsize=100
# 151us 1013 GB/s
./install/test/ex5dense 5000 20000 1000 float
# 259us 1544 GB/s

