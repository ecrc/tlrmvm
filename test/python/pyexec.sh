# complex single 
python install/test/python/generateinput.py \
--nb=256 --error_threshold=0.001 --problemname=SeismicFreq100 \
--datatype=csingle --TLRMVM_ROOT=$(pwd)/install \
--WORK_ROOT=$WORK_ROOT

# float
python install/test/python/generateinput.py \
--nb=256 --error_threshold=0.0001 --problemname=mavis_000_R \
--datatype=float --TLRMVM_ROOT=$(pwd)/install \
--WORK_ROOT=$WORK_ROOT
