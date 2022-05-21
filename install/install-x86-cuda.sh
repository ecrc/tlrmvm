#!/usr/bin/bash
cmake -DCMAKE_CUDA_COMPILER:PATH=$(which nvcc) -DCMAKE_INSTALL_PREFIX=./install -DCMAKE_C_COMPILER=$(which gcc) -DCMAKE_CXX_COMPILER=$(which g++) \
-DUSE_MPI=ON -DUSE_MKL=ON -DBUILD_CUDA=ON -DBUILD_TEST=ON -DBUILD_PYTHON=ON -DPYTHON_EXECUTABLE=$(which python) 
#-DPython_ROOT_DIR=/home/hongy0a/miniconda3/envs/a100env
