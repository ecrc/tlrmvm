
# constant rank only for nvidia 
# M = 5000 N = 20000 nb = 100, Nruns = 50
mpirun -np 1 cudabin fixed 5000 20000 100 50 cuda
