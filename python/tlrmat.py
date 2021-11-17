import time
import pickle 
import os
import numpy as np
from enum import IntEnum


def neterr(yreal,yapprox):
    return np.max(np.abs(yreal-yapprox)) / np.max(np.abs(yreal))

def elementsRW(paddingM, paddingN, grank, nb):
    p1 = grank * nb + paddingN + grank
    p2 = 2 * grank 
    p3 = grank * nb + grank + paddingM 
    return (p1+p2+p3)


class TLRMat:
    def __init__(self, m, n, nb, acc, datafolder, problemname, dtype):
        self.m = m
        self.n = n
        self.problemname = problemname
        self.nb = nb
        self.acc = acc 
        self.datafolder = datafolder
        self.dtype = dtype
        if m % nb != 0:
            self.pm = ( m // nb + 1 ) * nb
        else:
            self.pm = ( m // nb ) * nb
        if m % nb != 0:
            self.pn = ( n // nb + 1 ) * nb
        else:
            self.pn = ( n // nb ) * nb
        self.mtg = self.pm // nb
        self.ntg = self.pn // nb
        self.loaddata()

    def loadX(self):
        xvec = np.fromfile(os.path.join(self.datafolder, 
        "{}_x.bin".format(self.problemname)),dtype=self.dtype)
        self.xvec = xvec

    def loaddata(self):
        datafolder = self.datafolder
        nb = self.nb 
        acc = self.acc 
        ntg = self.ntg 
        mtg = self.mtg 
        ufile = os.path.join(datafolder,'{}_Ubases_nb{}_acc{}.bin'.format(self.problemname, nb, acc))
        vfile = os.path.join(datafolder,'{}_Vbases_nb{}_acc{}.bin'.format(self.problemname, nb, acc))
        rfile = os.path.join(datafolder,'{}_Rmat_nb{}_acc{}.bin'.format(self.problemname, nb, acc))
        self.u = np.fromfile(ufile, dtype=self.dtype)
        self.v = np.fromfile(vfile, dtype=self.dtype)
        self.rank = np.fromfile(rfile, dtype=np.int32).reshape(ntg, mtg).T
        u = self.u 
        v = self.v 
        rank = self.rank
        self.numpyulist = [[] for i in range(mtg)]
        self.numpyvlist = [[] for i in range(ntg)]

        rankcolsum = np.sum(rank, axis=0)
        rankrowsum = np.sum(rank, axis=1)
        umat = u.reshape(-1,nb).T
        prev = 0
        post = 0
        for i in range(mtg):
            post += rankrowsum[i]
            rowumat = umat[:, prev:post]
            innerprev = 0
            innerpost = 0
            for j in range(ntg):
                innerpost += rank[i,j]
                curblock = rowumat[:, innerprev:innerpost]
                self.numpyulist[i].append(curblock)
                innerprev += rank[i,j]
            prev += rankrowsum[i]
        prev = 0
        post = 0
        #recover v list
        for i in range(ntg):
            post += rankcolsum[i]
            colvmat = v[prev*nb:post*nb].reshape(nb, rankcolsum[i]).T
            innerprev = 0
            innerpost = 0
            for j in range(mtg):
                innerpost += rank[j,i]
                curblock = colvmat[innerprev:innerpost,:]
                self.numpyvlist[j].append(curblock)
                innerprev += rank[j, i]
            prev += rankcolsum[i]

    ####################################################
    # matvec interface
    ####################################################
    def matvec(self, transpose = False, conjugate = False, x=None):
        if x is None:
            self.loadX()
            x = self.xvec
        else:
            assert(x.shape[0] == self.n)
        if not transpose and not conjugate:
            return self._matveckernel(x)
        elif transpose and not conjugate:
            return self._matvectranskernel(x)
        elif not transpose and conjugate:
            return np.conjugate(self._matveckernel(np.conjugate(x)))
        else:
            return np.conjugate(self._matvectranskernel(np.conjugate(x)))

    def _matveckernel(self, x):
        return self._tlrmvmsingleprecision(x)

    def _matvectranskernel(self, x):
        return self._tlrmvm_trans_singleprecision(x)

    def _tlrmvm_trans_singleprecision(self,x):
        nb = self.nb 
        ntg = self.ntg 
        mtg = self.mtg 
        u = self.u 
        v = self.v 
        rank = self.rank
        rankcolsum = np.sum(rank, axis=0)
        rankrowsum = np.sum(rank, axis=1)
        # need to transpose rank!!
        trank = rank.T
        paddingx = np.zeros(self.pm,dtype=np.csingle)
        paddingx[:x.shape[0]] = x 
        tmpy = []
        voffset = 0
        uoffset = 0
        for i in range(mtg):
            tmpu = u[(uoffset*nb) : (uoffset+rankrowsum[i])*nb].reshape(-1,nb)
            tmpx = paddingx[i*nb : (i+1)*nb]
            tmpy.append(tmpu @ tmpx)
            uoffset += rankrowsum[i]
        res = []
        for i in range(ntg):
            tmpyv = []
            for j in range(mtg):
                beginpos = np.sum(trank[:i, j])
                stoppos = np.sum(trank[:(i+1), j])
                tmpyv.append(tmpy[j][beginpos:stoppos])
            curyv = np.concatenate(tmpyv)
            res.append(curyv)
        tlrmvmyout = []
        voffset = 0
        for i in range(ntg):
            tlrmvmyout.append(v[(voffset*nb) : (voffset+rankcolsum[i])*nb].reshape(-1,rankcolsum[i]) @ res[i])
            voffset += rankcolsum[i]
        tlrmvmout = np.concatenate(tlrmvmyout)
        return tlrmvmout[:x.shape[0]]


    def _tlrmvmsingleprecision(self, x):
        nb = self.nb 
        ntg = self.ntg 
        mtg = self.mtg 
        u = self.u
        v = self.v
        rank = self.rank
        rankcolsum = np.sum(rank, axis=0)
        rankrowsum = np.sum(rank, axis=1)
        paddingx = np.zeros(self.pn,dtype=np.csingle)
        paddingx[:x.shape[0]] = x 
        tmpy = []
        voffset=0
        for i in range(ntg):
            t1 = v[ (voffset*nb) : (voffset + rankcolsum[i])*nb].reshape(nb,rankcolsum[i]).T
            t2 = paddingx[i*nb : (i+1)*nb]
            tmpy.append(t1 @ t2) 
            voffset += rankcolsum[i]

        self.yv = np.concatenate(tmpy)
        ## phase 2 reshuffle
        res = []
        for i in range(mtg):
            tmpyu = []
            for j in range(ntg):
                beginpos = np.sum(rank[:i,j])
                stoppos = np.sum(rank[:(i+1),j])
                tmpyu.append(tmpy[j][beginpos:stoppos])
            curyu = np.concatenate(tmpyu)
            res.append(curyu)
        self.yu = np.concatenate(res)
        ## phase 3 
        tlrmvmyout = []
        uoffset = 0
        for i in range(mtg):
            tlrmvmyout.append(u[(uoffset*nb) : (uoffset+rankrowsum[i])*nb].reshape(-1,nb).T @ res[i])
            uoffset += rankrowsum[i]
        tlrmvmout = np.concatenate(tlrmvmyout)
        self.y = tlrmvmout[:x.shape[0]]
        return tlrmvmout[:x.shape[0]]
