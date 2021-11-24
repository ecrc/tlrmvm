# Tile Low-Rank Matrix Vector Multiplication

## 1. Introduction

Matrix-Vector Multiplication (MVM) is a fundamental memory-bound Level-2 BLAS operation. The kernel drives the performance of various scientific applications, 
including 1) seismic imaging to reveal the subsurface layers for better monitoring the permafrost degradation or mitigating exploration and drilling risks for oil 
and gas industries, and 2) ground-based computational astronomy for supporting real-time simulations necessary to outsmart the atmospheric turbulence and help 
identifying exoplanets. We further leverage the inherent data sparsity structure of the resulting covariance matrices using Tile Low-Rank (TLR) matrix 
approximations. Our TLR-MVM outperforms its dense counterpart on many vendor architectures with high productivity in mind and maintains the numerical robustness of 
the applications.  

## 2. Dependencies  

We have tested TLR-MVM on Intel, AMD, NEC Aurora, Fujitsu A64FX and NVIDIA GPUs. We use cmake system to build the code.

A single-threaded BLAS (matrix vector multiplication) implementation is required. One can use MKL, OpenBLAS or BLAS that comes with compilers, set `MKL_ROOT` to let library find it.

MPI is optional but strongly recommended, set `MPI_ROOT` to let library find it.

NCCL is required to build the library with NVIDIA GPU, set `NCCL_ROOT` to let library find it. 

## 3. Compilation

    mkdir build && cd build

### 3.1 Intel

    CC=gcc CXX=g++ cmake -DCMAKE_INSTALL_PREFIX=$(pwd)/install -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=OFF -DUSE_MKL=ON -DBUILD_TEST=ON -DBUILD_PYTHON=OFF -DBUILD_CUDA=OFF -DBUILD_DOC=OFF -DUSE_OPENBLAS=OFF -DUSE_COMPILER_BLAS=OFF -DUSE_BLIS=OFF -DUSE_NVTX=OFF -DUSE_NCCL=OFF ..

One can omit the cmake options that are `OFF`.

### 3.2 AMD EPYC

    CC=gcc CXX=g++ cmake -DCMAKE_INSTALL_PREFIX=$(pwd)/install -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON -DUSE_BLIS=ON -DBUILD_PYTHON=OFF -DBUILD_TEST=ON ..

### 3.3 NEC Aurora
    
    CC=mpincc CXX=mpinc++ cmake -DCMAKE_INSTALL_PREFIX=$(pwd)/install -DCMAKE_BUILD_TYPE=Release -DUSE_COMPILER_BLAS=ON -DBUILD_PYTHON=OFF -DBUILD_TEST=ON -DUSE_MPI=ON ..

### 3.5 CUDA

    CC=gcc CXX=g++ cmake -DCMAKE_INSTALL_PREFIX=$(pwd)/install -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER:PATH=$CUDAToolkit_ROOT/bin/nvcc -DUSE_MPI=ON -DUSE_MKL=ON -DBUILD_CUDA=ON -DBUILD_TEST=ON -DBUILD_PYTHON=OFF ..

Then do 

    make install -j

### 4 Test
You also need to download the dataset to run the experiments.
dataset download url:
https://drive.google.com/drive/folders/1_DSyloFjlScXGTlA1_ybJnTne59tUpgR?usp=sharing

after download, put it in a seperate folder and set `WORK_ROOT` to that folder.

in install/test folder, you can try to launch bash file. 
These are the test files.

### 5 Handout

![alt text](https://github.com/ecrc/tlrmvm-dev/blob/master/doxygen/handsout.png)

If you have any troubles, please create an issue or 
send email to yuxi.hong@kaust.edu.sa / hatem.ltaief@kaust.edu.sa.
