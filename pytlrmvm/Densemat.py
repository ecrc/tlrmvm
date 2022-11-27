#   @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
#                       All rights reserved.

from os.path import join
from scipy.io import loadmat
import numpy as np


class DenseMat:
    """
    A simple dense mat vec impl class
    """
    def __init__(self, freq_list, datafolder) -> None:
        self.denseA = {}
        for freq in freq_list:
            curAfile = join(datafolder, 'Mck_freqslices','Mck_freqslice{}_sub1.mat'.format(freq))
            curA = loadmat(curAfile)
            self.denseA[freq] = curA['Rfreq']
        self.ymap = None

    def matvec(self, xlist, trans=False, conj=False):
        assert(len(self.denseA.keys()) == len(xlist))
        self.ymap = {}
        cnt = 0
        for k,v in self.denseA.items():
            if not trans and not conj:
                self.ymap[k] = v @ xlist[cnt]
            elif trans and not conj:
                self.ymap[k] = v.T @ xlist[cnt]
            elif not trans and conj:
                self.ymap[k] = np.conjugate(v) @ xlist[cnt]
            elif trans and conj:
                self.ymap[k] = np.conjugate(v).T @ xlist[cnt]
            cnt += 1
    
    def generateUVR(self):
        """
        Generate 
        """
        pass