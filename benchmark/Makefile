###########################################################################
# Description:
# This Makefile is used for dense matrix vector multiplication benchmark
# Since TLR-MVM use single-threaded BLAS library and dense MVM use
# multi-threaded BLAS library, we seperate the installation.
#
# Author: Yuxi Hong
#
# Email: yuxi.hong@kaust.edu.sa
###########################################################################
MKL_single:
	icpx -qmkl=parallel -DUSE_MKL -o mklsingle cpp/bench_mvm_fp32.cpp -Wl,-rpath,$MKLROOT/lib/intel64

MKL_complex:
	icpx -qmkl=parallel -DUSE_MKL -o mklcomplex cpp/bench_mvm_singlecomplex.cpp -Wl,-rpath,$MKLROOT/lib/intel64

BLIS_single:
	g++ -O3 -DUSE_CBLAS -fopenmp -I${BLIS_MT_ROOT}/include cpp/bench_mvm_fp32.cpp -o amdcpusingle \
    		-L${BLIS_MT_ROOT}/lib -lblis-mt -Wl,-rpath,${BLIS_MT_ROOT}/lib

BLIS_complex:
	g++ -O3 -DUSE_CBLAS -fopenmp -I${BLIS_MT_ROOT}/include cpp/bench_mvm_singlecomplex.cpp -o amdcpucomplex \
    		-L${BLIS_MT_ROOT}/lib -lblis-mt -Wl,-rpath,${BLIS_MT_ROOT}/lib

NEC_single:
	. /opt/nec/ve/nlc/2.3.0/bin/nlcvars.sh
	nc++ -O3 -DUSE_CBLAS cpp/bench_mvm_fp32.cpp -o necsingle -lcblas -fopenmp -lblas_openmp

NEC_complex:
	. /opt/nec/ve/nlc/2.3.0/bin/nlcvars.sh
	nc++ -O3 -DUSE_CBLAS cpp/bench_mvm_fp32.cpp -o neccomplex -lcblas -fopenmp -lblas_openmp

FUJITSU_single:
	FCC -O3 -DUSE_CBLAS -Kfast,openmp cpp/bench_mvm_fp32.cpp -o fujitsusingle -SSL2BLAMP

FUJITSU_complex:
	FCC -O3 -DUSE_CBLAS -Kfast,openmp cpp/bench_mvm_singlecomplex.cpp -o fujitsucomplex -SSL2BLAMP

CUDA_single:
	g++ -O3 -I${CUDA_ROOT}/include cuda/bench_mvm_cuda_fp32.cpp -o cudasingle \
		-L${CUDA_ROOT}/lib64 -lcudart -lcublas  -Wl,-rpath,${CUDA_ROOT}/lib64

CUDA_complex:
	g++ -O3 -I${CUDA_ROOT}/include cuda/bench_mvm_cuda_singlecomplex.cpp -o cudacomplex \
    	-L${CUDA_ROOT}/lib64 -lcudart -lcublas  -Wl,-rpath,${CUDA_ROOT}/lib64

HIP_single:
	echo "Not Implemented!"

HIP_complex:
	echo "Not Implemented!"
