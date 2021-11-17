#/bin/bash
./install/test/ex4cudagraph_float --M=4802 --N=19078 \
--errorthreshold=0.0001 --problemname=mavis_000_R \
--datafolder=$WORK_ROOT --nb=256 --streamsize=20 --loopsize=100
# 151us 1013 GB/s
./install/test/ex5dense 5000 20000 1000 float
# 259us 1544 GB/s

./install/test/ex4cudagraph_csingle --M=9801 --N=9801 \
--errorthreshold=0.001 --problemname=SeismicFreq100 \
--datafolder=$WORK_ROOT --nb=256 --streamsize=20 --loopsize=100
# 249us 1078 GB/s
./install/test/ex5dense 10000 10000 1000 complex
# 478us 1673 GB/s