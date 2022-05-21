##################################################################
# @copyright (c) 2021- King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# Author: Yuxi Hong, 2021.10.27
# Description: Convert matlab .mat file to numpy npy.
##################################################################
import os
from os.path import join
from scipy.io import loadmat
import numpy as np
import pickle 
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, help='matfilename')
parser.add_argument('--mat_root', type=str, help='matfile dir')
parser.add_argument('--work_root', type=str, help='workspace dir')
args = parser.parse_args()

matname = join(args.mat_root, args.filename + '.mat')
npyname = join(args.work_root, args.filename + '.npy')
work_root = args.work_root
A = loadmat(matname)['Rfreq']
with open(npyname, 'wb') as f:
    np.save(f, A)
