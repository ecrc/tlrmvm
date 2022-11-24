#
#  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
#                      All rights reserved.
#

# float
python install/test/python/generateinput.py \
--nb=256 --error_threshold=0.0001 --problemname=mavis_000_R \
--datatype=float --TLRMVM_ROOT=$(pwd)/install \
--WORK_ROOT=$WORK_ROOT
