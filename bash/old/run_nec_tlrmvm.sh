# constant rank M = 5000 N = 20000 nb = 100 constant rank is 25
make nec && export OMP_NUM_THREADS=8; mpirun -v -np 1 -ve 0 ./necbin random 5000 20000 100 const 25 
# mavis system M = 4864 N = 19200 nb = 128
make nec && export OMP_NUM_THREADS=8; mpirun -v -np 1 -ve 0 ./necbin mavis 000 0.0001 4864 19200 128