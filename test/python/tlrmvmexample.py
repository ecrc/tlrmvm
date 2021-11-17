"""
This example gives you a brief introduction of how you can use TLRMVM in python.
We provide only TLRMVMCPU interface here, but one can easily follow the same way
to provide interface to others.
"""
import numpy as np 
import sys
import os
import time

if 'TLRMVM_ROOT' not in os.environ:
    print("TLRMVM_ROOT not set!") 
    exit()

if 'WORK_ROOT' not in os.environ:
    print("WORK_ROOT not set!") 
    exit()

# your TLRMVM_ROOT should point to your installation path.
TLRMVM_ROOT = os.environ['TLRMVM_ROOT']
sys.path.append( os.path.join( os.environ['TLRMVM_ROOT'],"python") )
WORK_ROOT = os.environ["WORK_ROOT"]

from tlrmvmpy import *

# get your data matrix in numpy format
# you can download the matrix used in this example at 
# https://drive.google.com/file/d/1KY4eeVSMm2mWOOKVU7QjsAf6tOREv-99/view?usp=sharing
import scipy.io as sio 
A = sio.loadmat(  os.path.join(WORK_ROOT, "Mck_freqslice100_1_sub1.mat") )['Rfreq']
randomx = np.random.rand(A.shape[1]) + 1j * np.random.rand(A.shape[1])
randomx = randomx.astype(np.csingle)
"""
Below is needed for creating input of tlrmvm.
They are parameters Tile size (nb) and Accuracy Threshold (error_threshold) 
you can play with to get decent performance and numerical accuracy.
"""
m = A.shape[0]
n = A.shape[1]
nb = 256
error_threshold = '0.001' # we use string for easy concatnating.
workplacefolder = WORK_ROOT
datasetname = 'SeismicFreq100'

# create tlrmvm util class 
tlrmvmutil = TLRMVM_Util(A, nb, workplacefolder, error_threshold, datasetname)
# compute svd and save 
tlrmvmutil.computesvd()
# create input of tlrmvm
tlrmvmutil.saveUV()
# get compression info
tlrmvmutil.printdatainfo()

tlrmvmutil.saveX(randomx)

# create config of tlrmvm
print("Rank file: ", tlrmvmutil.rankfile)
tlrmvmconfig = TlrmvmConfig(m, n, nb, tlrmvmutil.rankfile, 
workplacefolder, error_threshold, datasetname)
# maskmat is used in mpienv, if you launch two mpi processes where 
# they worked on 1 TRLMVM. The maskmat indicated which tile each process
# should work on.
# the summation of two maskmat should be equal to a one-like matrix, size is Mtiles x Ntiles.
# the elements inside maskmat is either 1 or 0.
maskmat = np.ones( (tlrmvmconfig.Mtg,tlrmvmconfig.Ntg) )
tlrmvmconfig.setmaskmat(maskmat)

"""
tlrmvm compute
"""
# we need to select correct data type by passing numpy data type 
# currently support float32, float64, csingle, cdouble
tlrmvminstance = Tlrmvm(tlrmvmconfig, np.csingle)
tlrmvminstance.MemoryInit()



print("TLR-MVM Benchmarking ...")
rawtime = []
for i in range(10):
    start = time.time()
    tlrmvminstance.MVM(randomx)
    end = time.time()
    rawtime.append(end - start)

print("TLR-MVM Correctness Check")
tlrmat = TLRMat(m, n, nb, error_threshold, workplacefolder, 
datasetname, dtype=np.csingle)
yreal = tlrmat.matvec(x = randomx)

pyphase1 = tlrmat.yv
cphase1 = tlrmvminstance.yv
print("Precision of Phase1: {:.3e}".format( neterr(cphase1, pyphase1) ))

pyphase2 = tlrmat.yu
cphase2 = tlrmvminstance.yu
print("Precision of Phase2: {:.3e}".format( neterr(cphase2, pyphase2) ))

pyphase3 = tlrmat.y
cphase3 = tlrmvminstance.y
print("Precision of Phase3 (Output): {:.3e}".format( neterr(cphase3, pyphase3) ))

rawtime = np.array(rawtime)
mediantime = np.median(rawtime)

elerw = elementsRW(tlrmvminstance.paddingM, tlrmvminstance.paddingN, tlrmvminstance.granksum, tlrmvminstance.nb)
# we are operating complex single 
bd = elerw * 8 / mediantime
print("TLR-MVM Meidan time : {:.3f} us. Sustained baidwidth: {:.3f} GB/s. ".format(mediantime * 1e6, bd * 1e-9))

tlrmvminstance.MemoryFree()


mvmrawtime = []
for i in range(10):
    start = time.time()
    tmpy = A @ randomx
    end = time.time()
    mvmrawtime.append(end - start)

npbd = (A.shape[0] * A.shape[1] + A.shape[1] + A.shape[0]) * 8  / np.median(mvmrawtime) * 1e-9
print("Numpy MVM Median time: {:.3f} us.  Sustained baidwidth: {:.3f} GB/s. ".format(np.median(mvmrawtime) * 1e6 , npbd ) )
