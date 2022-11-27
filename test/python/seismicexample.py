#   @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
#                       All rights reserved.

import numpy as np
import sys
import os
from scipy.io import loadmat

# sys.path.append("/home/hongy0a/Repos/tlrmvm-dev/")

from pytlrmvm.tlrmvmtools import *
A = loadmat('/home/hongy0a/Repos/datasets/seismic/Mck_freqslices/Mck_freqslice100_sub1.mat')['Rfreq']
nb = 256
error_threshold = '0.001'
datasetname = 'Mck_freqslice_100'
# datasetname = 'Mck_freqslice_100'
workplacefolder = '/home/hongy0a/Repos/datasets/seismic/'
tlrmvmutil = TLRMVM_Util(A, nb, workplacefolder, error_threshold, datasetname)
tlrmvmutil.computesvd()
# create input of tlrmvm
tlrmvmutil.saveUV()
# get compression info
tlrmvmutil.printdatainfo()
