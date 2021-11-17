export OMP_NUM_THREADS=1 OMP_PROC_BIND=true OMP_DISPLAY_ENV=true && ./intelbin fixed 5000 20000 100 intel > amdtmp/csl_mkl_core1.log 2>&1
export OMP_NUM_THREADS=2 OMP_PROC_BIND=true OMP_DISPLAY_ENV=true && ./intelbin fixed 5000 20000 100 intel > amdtmp/csl_mkl_core2.log 2>&1
export OMP_NUM_THREADS=4 OMP_PROC_BIND=true OMP_DISPLAY_ENV=true && ./intelbin fixed 5000 20000 100 intel > amdtmp/csl_mkl_core4.log 2>&1
export OMP_NUM_THREADS=8 OMP_PROC_BIND=true OMP_DISPLAY_ENV=true && ./intelbin fixed 5000 20000 100 intel > amdtmp/csl_mkl_core8.log 2>&1
export OMP_NUM_THREADS=16 OMP_PROC_BIND=true OMP_DISPLAY_ENV=true && ./intelbin fixed 5000 20000 100 intel > amdtmp/csl_mkl_core16.log 2>&1
export OMP_NUM_THREADS=32 OMP_PROC_BIND=true OMP_DISPLAY_ENV=true && ./intelbin fixed 5000 20000 100 intel > amdtmp/csl_mkl_core32.log 2>&1
export OMP_NUM_THREADS=40 OMP_PROC_BIND=true OMP_DISPLAY_ENV=true && ./intelbin fixed 5000 20000 100 intel > amdtmp/csl_mkl_core40.log 2>&1
