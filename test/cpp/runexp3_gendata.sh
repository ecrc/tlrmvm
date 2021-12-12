./install/test/ex3_gendata --M=4802 --N=19078 \
--errorthreshold=0.0001 --problemname=Sytheticfloat \
--datafolder=$WORK_ROOT --nb=256 --constrank=100 --dtype=float

./install/test/ex3_gendata --M=9801 --N=9801 \
--errorthreshold=0.001 --problemname=Sytheticcomplex \
--datafolder=$WORK_ROOT --nb=256 --constrank=100 --dtype=complexfloat

