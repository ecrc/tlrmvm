# Tile Low-Rank Matrix Vector Multiplication

## 1. Introduction

Matrix-Vector Multiplication (MVM) is a fundamental memory-bound Level-2 BLAS operation. 
The kernel drives the performance of various scientific applications, 
including 1) seismic imaging to reveal the subsurface layers for better monitoring the 
permafrost degradation or mitigating exploration and drilling risks for oil 
and gas industries, and 2) ground-based computational astronomy for supporting real-time 
simulations necessary to outsmart the atmospheric turbulence and help 
identifying exoplanets. We further leverage the inherent data sparsity structure of the 
resulting covariance matrices using Tile Low-Rank (TLR) matrix 
approximations. Our TLR-MVM outperforms its dense counterpart on many vendor 
architectures with high productivity in mind and maintains the numerical robustness of 
the applications.  



## 2. Dependencies and Create build directory

### A. x86_64 systems
We strongly recommend using [spack](https://spack.readthedocs.io/en/latest/index.html)
to install TLR-MVM dependencies for x86_64 systems.

This includes `Intel CPU`, `AMD EPYC CPU`, `NVIDIA GPU`, `AMD GPU` in the following
installation sections.

We use `cmake@3.21.0`, `MKL` or `BLIS` or `OpenBLAS` or `cuBLAS` or `rocBLAS`, 
`openmpi@4.1.2` to 
build the library.

- `MKL` or `oneAPI MKL` can be used to install `INTEL CPU`
- `BLIS` is used to install 'AMD EPYC CPU'
- `cuBLAS` is used to install 'NVIDIA GPU'
- `rocBLAS` us used to install 'AMD GPU'

Check `install` folder `sapck.yaml` and use the file to install dependencies.


### B. NEC Aurora and Fujitsu A64FX

A single-threaded BLAS (matrix vector multiplication) implementation is required.
One can use MKL, OpenBLAS or BLAS that comes with compilers, set `MKL_ROOT` to
let library find it.

MPI is optional but strongly recommended, set `MPI_ROOT` to let library find it.

NCCL is required to build the library with NVIDIA GPU, set `NCCL_ROOT` to let library
find it.

    mkdir build && cd build


## 3. Installation


### A. Intel CPU (GCC Compiler Toolkit)

    CC=gcc CXX=g++ cmake -DCMAKE_INSTALL_PREFIX=./install \
    -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON -DUSE_MKL=ON -DBUILD_TEST=ON ..

### B. Intel CPU (DPCPP Compiler Toolkit with oneAPI)

    CC=gcc CXX=g++ cmake -DCMAKE_INSTALL_PREFIX=./install \
    -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON -DUSE_MKL=ON -DBUILD_TEST=ON ..

### C. AMD EPYC CPU 

    CC=gcc CXX=g++ cmake -DCMAKE_INSTALL_PREFIX=./install \
    -DCMAKE_BUILD_TYPE=Release -DUSE_MPI=ON -DUSE_BLIS=ON -DBUILD_TEST=ON ..

### D. NEC Aurora Vector Engine
    
    CC=mpincc CXX=mpinc++ cmake -DCMAKE_INSTALL_PREFIX=./install \
    -DCMAKE_BUILD_TYPE=Release -DUSE_COMPILER_BLAS=ON -DBUILD_TEST=ON \
    -DUSE_MPI=ON ..

### E. Fujitsu A64FX FX1000

    CC=mpincc CXX=mpinc++ cmake -DCMAKE_INSTALL_PREFIX=./install \
    -DCMAKE_BUILD_TYPE=Release -DUSE_COMPILER_BLAS=ON -DBUILD_TEST=ON \
    -DUSE_MPI=ON ..


### F. NVIDIA GPU

    CC=gcc CXX=g++ cmake -DCMAKE_INSTALL_PREFIX=./install \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER:PATH=$CUDAToolkit_ROOT/bin/nvcc \
    -DUSE_MKL=ON -DBUILD_CUDA=ON -DBUILD_TEST=ON ..

### G. AMD GPU

    CC=gcc CXX=g++ cmake -DCMAKE_INSTALL_PREFIX=./install \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER:PATH=$CUDAToolkit_ROOT/bin/nvcc \
    -DUSE_MKL=ON -DBUILD_CUDA=ON -DBUILD_TEST=ON ..


Compile and install

    make install -j


## 4. Test
You also need to download the dataset to run the experiments.
dataset download url:
1. seismic redatuming dataset
https://zenodo.org/record/6582600
2. MAVIS AO system matrcies dataset
https://zenodo.org/record/7305622

after download, put it in a seperate folder and set `WORK_ROOT` to that folder.

in install/test folder, you can try to launch bash file. 
These are the test files.

## 5. Benchmark

`benchmark` folder offers dense matrix vector multiplication benchmark tools.
We offer `Makefile` to compile the benchmarks since TLR-MVM requires single-threaded
BLAS while `benchmark` use threaded version BLAS.

Please check the corresponding environment variables in the `Makefile` and compile.

## 6. Python Support

Currently, we suggest one to use our python library with NVIDIA GPU.
To install it, 

    BUILD_CUDA=ON python setup.py build

This will create a build directory and build library inside it.
After installation, add the library path (build/libxxx) to your 

## 7 References 
Y. Hong, M. Ravasi, H. Ltaief, D. Keyes, Can tile low-rank compression live up to expectation? An application to 3D multi-dimensional deconvolution, 2023, SEG IMAGE 2023 International Meeting for Applied Geoscience & Energy (extended abstract).

Y. Hong, H. Ltaief, M. Ravasi,  D. Keyes, HPC Seismic Redatuming by Inversion with Algebraic Compression and Multiple Precisions, 2023, KAUST Repo Preprint. 

H. Ltaief, Y. Hong, L. Wilson, M. Jacquelin, M. Ravasi and D. Keyes, Scaling the “Memory Wall” for Multi-Dimensional Seismic Processing with Algebraic Compression on Cerebras CS-2 Systems, 2023, Proceedings of the International Conference for High Performance Computing, Networking, Storage, and Analysis (SC’23), IEEE Computer Society (ACM Gordon Bell Finalist).

H. Ltaief, Y. Hong, A. Dabah, R. Alomairy, S. Abdulah, C. Goreczny, P. Gepner, M. Ravasi, D. Gratadour and D. Keyes, Steering Customized AI Architectures for HPC Scientific Applications, 2023, Springer International Supercomputing Conference (ISC’23) (A. Bhatele et al., eds.), Lecture Notes in Computer Science, Vol. 13948, pp. 125–143, doi 10.1007/978-3-031-32041-5 7.

M. Ravasi, Y. Hong, H. Ltaief and D. Keyes, Tile-Low Rank Compressed Multi-Dimensional Convolution and Its Application to Seismic Redatuming Problems, 2022, 83rd EAGE Annual Conference, doi 10.3997/2214-4609.202210253 (extended abstract).

M. Ravasi, Y. Hong, H. Ltaief, D. Keyes and D. Vargas, Large-Scale Marchenko Imaging with Distance-Aware Matrix Reordering, Tile Low-Rank Compression, and Mixed-Precision Computations, 2022, SEG IMAGE 2022 International Meeting for Applied Geoscience & Energy (extended abstract).

H. Ltaief, J. Cranney, D. Gratadour, Y. Hong, L. Gatineau, and D. Keyes, Meeting the Real-Time Challenges of Ground-Based Telescopes Using Low-Rank Matrix Computations, 2021, ACM/IEEE International Conference for High Performance Computing, Networking, Storage, and Analysis (SC’21), doi 10.1145/3458817.3476225.

Y. Hong, H. Ltaief, M. Ravasi, L. Gatineau and D. E. Keyes, Accelerating Seismic Redatuming Using Tile Low-Rank Approximations on NEC SX-Aurora TSUBASA, 2021, Supercomputing Frontiers and Innovations 8:6–26, doi 0.14529/jsfi210201.

H. Zhang, J. Cranney, N. Doucet, Y. Hong, D. Gratadour, H. Ltaief, D. Keyes and F. Rigaut, Predictive Learn and Apply: MAVIS application – Learn, 2020, in Proceedings of SPIE 11448, Adaptive Optics Systems VII, 114482L (extended abstract).

## 8 Handout

![alt text](https://github.com/ecrc/tlrmvm/blob/master/doxygen/handsout.png)

If you have any troubles, please create an issue or 
send email to yuxi.hong@kaust.edu.sa / hatem.ltaief@kaust.edu.sa.
