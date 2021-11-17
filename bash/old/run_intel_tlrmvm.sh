for ompthread in 20 40
do
export OMP_NUM_THREADS=${ompthread}
for nb in 40 100 200 500
do
mpirun --bind-to none -np 1 ./intelbin random 5000 20000 ${nb} const $((nb/4)) > intel_const_1MPI_OMP${ompthread}_M5000_N2000_nb${nb}.log
done 
for mavis in 000 010 015 020 025 030 035 040 045 050 055 060 065 070
do 
# M = 4864 N = 19200 nb = 128
mpirun --bind-to none -np 1 ./intelbin mavis ${mavis} 0.0001 4864 19200 128 > intel_mavis_${mavis}_1MPI_OMP${ompthread}.log
done 
done


