export OMP_NUM_THREADS=32 OMP_PROC_BIND=true OMP_DISPLAY_ENV=true OMP_PLACES="{0},{2},{4},{6},{8},{10},{12},{14},{16},{18},{20},{22},{24},{26},{28},{30},{32},{34},{36},{38},{40},{42},{44},{46},{48},{50},{52},{54},{56},{58},{60},{62}"
for nb in 40 100 200 500
do
mpirun --bind-to none -np 1 ./amdbin random 5000 20000 ${nb} const $((nb/4)) > amd_const_1MPI_OMP32_M5000_N2000_nb${nb}.log
done 
for mavis in 000 010 015 020 025 030 035 040 045 050 055 060 065 070
do 
mpirun --bind-to none -np 1 ./amdbin mavis ${mavis} 0.0001 4864 19200 128 > amd_mavis_${mavis}_1MPI_OMP32.log
done 

export OMP_NUM_THREADS=64 OMP_PROC_BIND=true OMP_DISPLAY_ENV=true OMP_PLACES="{0},{2},{4},{6},{8},{10},{12},{14},{16},{18},{20},{22},{24},{26},{28},{30},{32},{34},{36},{38},{40},{42},{44},{46},{48},{50},{52},{54},{56},{58},{60},{62},{64},{66},{68},{70},{72},{74},{76},{78},{80},{82},{84},{86},{88},{90},{92},{94},{96},{98},{100},{102},{104},{106},{108},{110},{112},{114},{116},{118},{120},{122},{124},{126}"
for nb in 40 100 200 500
do
mpirun --bind-to none -np 1 ./amdbin random 5000 20000 ${nb} const $((nb/4)) > amd_const_1MPI_OMP64_M5000_N2000_nb${nb}.log
done 
for mavis in 000 010 015 020 025 030 035 040 045 050 055 060 065 070
do 
mpirun --bind-to none -np 1 ./amdbin mavis ${mavis} 0.0001 4864 19200 128 > amd_mavis_${mavis}_1MPI_OMP64.log
done 
