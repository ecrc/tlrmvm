#
#  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
#                      All rights reserved.
#

# complex single
python install/test/python/generateinput.py \
--nb=256 --error_threshold=0.001 --problemname=SeismicFreq100 \
--datatype=csingle --TLRMVM_ROOT=$(pwd)/install \
--WORK_ROOT=$WORK_ROOT