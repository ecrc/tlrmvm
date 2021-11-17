import sys
for nb in [40,64,128,256,512]:
    for i in [1,2,4,8,16,32,64,128]:
        j = 128 // i
        print("OMP_NUM_THREADS={} mpirun -np {} --map-by slot:PE={} --rank-by core ./AstronomyGTestCPUMPI --gtest_filter=*Phase2_Correctness {} 0.0001 000 {} 4802 19078 >> {}_nb{}_OMP{}_MPI{}.log".format(j, i, j,sys.argv[1], nb, sys.argv[2], nb, j, i))
