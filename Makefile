CC ?= gcc 
CXX ?= g++
CUDAROOT ?= /usr/local/cuda-11.0
MPIROOT ?= /usr/local/
NCCLROOT ?= /usr/local/
MKLLINK = -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl

nec:
	mpincc -O3 dgemv.c -DUSE_NEC -o bin/necsingle -lcblas -fopenmp -lblas_sequential

cublas:
	nvcc clbastlrmv.cu -I${MPIROOT}/include -L${MPIROOT}/lib \
	-DUSE_NVIDIA -I${CUDADIR}/include -L${CUDADIR}/lib64 \
	-I${CUBLASROOT}/include -L${CUBLASROOT}/lib64 \
	-I${NCCLROOT}/include -L${NCCLROOT}/lib -lnccl \
	-lcuda -lcudart -lmpi -lcublas -lgomp -o cublasgemm

gmatrix:
	g++ -std=c++11 generate_random.cc -o generatedata

intel:
	gcc -O3 tlrmvm.c -lcblas -fopenmp 

clean:
	rm -rf magma_* cublas_*
	rm -rf *.nvprof
	rm -rf *.qdrep
	rm -rf *.bin
	rm -rf *.out *.err
	rm -rf core.*
	rm -rf *.log
