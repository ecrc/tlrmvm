#   @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
#                       All rights reserved.

import numpy as np
import sys
import os
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--TLRMVM_ROOT', type=str, help='installation dir')
parser.add_argument('--WORK_ROOT', type=str, help='workspace dir')
parser.add_argument('--nb', type=int, help='nb')
parser.add_argument('--error_threshold', type=str,help='error threshold')
parser.add_argument('--problemname', type=str, help='problem name')
parser.add_argument('--datatype', type=str, help='datatype of dataset')

args = parser.parse_args()

print("Your installation path: ", args.TLRMVM_ROOT)
print("Your workspace path: ", args.WORK_ROOT)

# your TLRMVM_ROOT should point to your installation path.
TLRMVM_ROOT = args.TLRMVM_ROOT
sys.path.append( os.path.join( args.TLRMVM_ROOT,"python") )
WORK_ROOT = args.WORK_ROOT
print("Downloading dataset to path: {}".format( WORK_ROOT ))
if not os.path.exists(WORK_ROOT):
    os.mkdir(WORK_ROOT)
problemname = args.problemname

from tlrmvmtools import *
dtype = None
datatype = args.datatype
if datatype == 'float':
    dtype = np.float32
elif datatype == 'double':
    dtype = np.float64 
elif datatype == 'csingle':
    dtype = np.csingle
elif datatype == 'cdouble':
    dtype = np.cdouble
else:
    print("Not support datatype.")
    exit(1)

# get your data matrix in numpy format
# you can download the matrix used in this example at 
# https://drive.google.com/file/d/1KY4eeVSMm2mWOOKVU7QjsAf6tOREv-99/view?usp=sharing
A = np.load(  os.path.join(WORK_ROOT, "{}.npy".format(problemname)) ).astype(dtype)

if datatype in ['csingle', 'cdouble']:
    randomx = np.random.rand(A.shape[1]) + 1j * np.random.rand(A.shape[1])
    randomx = randomx.astype(dtype)
else:
    randomx = np.random.rand(A.shape[1])
    randomx = randomx.astype(dtype)

"""
Below is needed for creating input of tlrmvm.
They are parameters Tile size (nb) and Accuracy Threshold (error_threshold) 
you can play with to get decent performance and numerical accuracy.
"""
m = A.shape[0]
n = A.shape[1]
nb = args.nb
error_threshold = args.error_threshold # we use string for easy concatnating.
workplacefolder = WORK_ROOT
datasetname = args.problemname

# create tlrmvm util class 
tlrmvmutil = TLRMVM_Util(A, nb, workplacefolder, error_threshold, datasetname)
# compute svd and save 
tlrmvmutil.computesvd()
# create input of tlrmvm
tlrmvmutil.saveUV()
# get compression info
tlrmvmutil.printdatainfo()

tlrmvmutil.saveX(randomx)