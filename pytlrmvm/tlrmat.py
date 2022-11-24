#   @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
#                       All rights reserved.

import time
import pickle 
import os
from os.path import join
import numpy as np
from enum import IntEnum

def neterr(yreal,yapprox):
    return np.max(np.abs(yreal-yapprox)) / np.max(np.abs(yreal))

def elementsRW(paddingM, paddingN, grank, nb):
    p1 = grank * nb + paddingN + grank
    p2 = 2 * grank 
    p3 = grank * nb + grank + paddingM 
    return (p1+p2+p3)

def seismic_dataloader(filename_prefix, datafolder, nb, acc, ntg, mtg,fake_value=None):
    rank_file = join(datafolder,
                     '{}_Rmat_nb{}_acc{}.bin'.format(filename_prefix, nb, acc))
    ubase_file = join(datafolder,
                      '{}_Ubases_nb{}_acc{}.bin'.format(filename_prefix, nb, acc))
    vbase_file = join(datafolder,
                      '{}_Vbases_nb{}_acc{}.bin'.format(filename_prefix, nb, acc))
    rank = np.fromfile(rank_file, dtype=np.int32).reshape(ntg, mtg).T
    u = np.fromfile(ubase_file, dtype=np.csingle)
    v = np.fromfile(vbase_file, dtype=np.csingle)
    if fake_value is not None:
        u.fill(fake_value)
        v.fill(fake_value)
    return rank, u, v

def transpose_2dlist(list2d : list):
    """transpose a 2d list"""
    res = []
    for i in range(len(list2d)):
        curlist = []
        for j in range(len(list2d[i])):
            curlist.append(list2d[j][i])
        res.append(curlist)
    return res

def merge_2dlist_alongrow(list2d: list):
    """merge 2d list from row"""
    res = []
    for i in range(len(list2d)):
        squeeze = []
        for j in range(len(list2d[i])):
            if list2d[i][j] is not None:
                squeeze.append(list2d[i][j])
        if len(squeeze) == 0:
            res.append(None)
        else:
            curnp = squeeze[0]
            for j in range(1, len(squeeze)):
                curnp = np.concatenate((curnp, squeeze[j]), axis=1)
            res.append(curnp)
    return res
    
def merge_2dlist_alongcolumn(list2d: list):
    """merge 2d list from col"""
    res = []
    for i in range(len(list2d[0])):
        squeeze = []
        for j in range(len(list2d)):
            if list2d[j][i] is not None:
                squeeze.append(list2d[j][i])
        if len(squeeze) == 0:
            res.append(None)
        else:
            curnp = squeeze[0]
            for j in range(1, len(squeeze)):
                curnp = np.concatenate((curnp,squeeze[j]), axis=0)
            res.append(curnp)
    return res

def merge_1dlist(list1d: list, nb):
    """merge 1d tuple list, when meet None, file nb zeros, for phase3."""
    tmpres = []
    for x in list1d:
        if x is None:
            tmpres.append(np.zeros(nb, dtype=np.complex64))
        else:
            tmpres.append(x[0].astype(np.float32) + 1j * x[1].astype(np.float32))
    cuval = tmpres[0]
    for i in range(1,len(tmpres)):
        cuval = np.concatenate((cuval, tmpres[i]), axis=0)
    return cuval

class TLRMVM:
    """tlrmvm implementation for various precision"""
    def __init__(self, Us, Vs, rank, dtype, nb, m, n) -> None:
        """TLRMVM init function.

        Args:
            Us : 2D-list, each element is a complex64 ndarray.
            Vs : 2D-list, each element is a complex64 ndarray.
            rank : rank distribution of current data type.
            It's not necessarily the origin rank, somewhere can be zero. 
            dtype : the precision of current TLRMVM.
            nb : tile size.
            pm : m after padding.
            pn : n after padding.
        """
        self.m = m
        self.n = n
        self.mtg = m // nb if m % nb == 0 else m // nb + 1
        self.ntg = n // nb if n % nb == 0 else n // nb + 1
        self.pm = self.mtg * nb
        self.pn = self.ntg * nb
        self.nb = nb
        self.dtype = dtype
        self.ureal = [[None for _ in range(self.ntg)] for _ in range(self.mtg)]
        self.uimag = [[None for _ in range(self.ntg)] for _ in range(self.mtg)]
        self.vreal = [[None for _ in range(self.ntg)] for _ in range(self.mtg)]
        self.vimag = [[None for _ in range(self.ntg)] for _ in range(self.mtg)]
        self.rank = rank
        if dtype == np.float32:
            self.make_fp32(Us,Vs,rank)
        elif dtype == np.float16:
            self.make_fp16(Us,Vs,rank)
        elif dtype == np.int8:
            self.make_int8(Us,Vs,rank)
        else:
            print("TLRMVM dtype not supported!")
            raise NotImplementedError
        
    def make_fp32(self,Us,Vs,rank):
        """generate tlrmvm format input for fp32"""
        for i in range(self.mtg):
            for j in range(self.ntg):
                if rank[i][j] != 0:
                    self.ureal[i][j] = Us[i][j].real
                    self.uimag[i][j] = Us[i][j].imag
                    self.vreal[i][j] = Vs[i][j].real
                    self.vimag[i][j] = Vs[i][j].imag

    def make_fp16(self,Us,Vs,rank):
        """generate tlrmvm format input for fp16"""
        for i in range(self.mtg):
            for j in range(self.ntg):
                if rank[i][j] != 0:
                    self.ureal[i][j] = Us[i][j].real.astype(np.float16)
                    self.uimag[i][j] = Us[i][j].imag.astype(np.float16)
                    self.vreal[i][j] = Vs[i][j].real.astype(np.float16)
                    self.vimag[i][j] = Vs[i][j].imag.astype(np.float16)

    def getmaxelem(self,bases: list):
        res = [None for _ in range(len(bases))]
        for i in range(len(bases)):
            if bases[i] is not None:
                res[i] = np.max(np.abs(bases[i]))
        return res

    def make_int8(self,Us,Vs,rank):
        """generate tlrmvm format input for int8"""
        for i in range(self.mtg):
            for j in range(self.ntg):
                if rank[i][j] != 0:
                    self.ureal[i][j] = Us[i][j].real
                    self.uimag[i][j] = Us[i][j].imag
                    self.vreal[i][j] = Vs[i][j].real
                    self.vimag[i][j] = Vs[i][j].imag
        # special processing
        # step 1, get max value for each stack u bases
        # and stack v bases.
        vreal = merge_2dlist_alongcolumn(self.vreal)
        vimag = merge_2dlist_alongcolumn(self.vimag)
        ureal = merge_2dlist_alongrow(self.ureal)
        uimag = merge_2dlist_alongrow(self.uimag)
        self.int8_vrealmax = self.getmaxelem(vreal)
        self.int8_vimagmax = self.getmaxelem(vimag)
        self.int8_urealmax = self.getmaxelem(ureal)
        self.int8_uimagmax = self.getmaxelem(uimag)
        # remove fp32 info
        self.i8_tlrvreal = [None for _ in range(self.ntg)]
        self.i8_tlrvimag = [None for _ in range(self.ntg)]
        self.i8_tlrureal = [None for _ in range(self.mtg)]
        self.i8_tlruimag = [None for _ in range(self.mtg)]
        for i in range(self.ntg):
            if self.int8_vrealmax[i] is not None:
                if self.int8_vrealmax[i] > 1e-10:
                    self.i8_tlrvreal[i] = \
                        (vreal[i] / self.int8_vrealmax[i] * 126).astype(np.int32)
                else:
                    self.i8_tlrvreal[i] = vreal[i].astype(np.int32) # set 0
            if self.int8_vimagmax[i] is not None:
                if self.int8_vimagmax[i] > 1e-10:
                    self.i8_tlrvimag[i] = \
                        (vimag[i] / self.int8_vimagmax[i] * 126).astype(np.int32)
                else:
                    self.i8_tlrvimag[i] = vimag[i].astype(np.int32) # set 0
        for i in range(self.mtg):
            if self.int8_urealmax[i] is not None:
                if self.int8_urealmax[i] > 1e-10:
                    self.i8_tlrureal[i] = \
                        (ureal[i] / self.int8_urealmax[i] * 126).astype(np.int32)
                else:
                    self.i8_tlrureal[i] = ureal[i].astype(np.int32) # set 0
            if self.int8_uimagmax[i] is not None:
                if self.int8_uimagmax[i] > 1e-10:
                    self.i8_tlruimag[i] = \
                        (uimag[i] / self.int8_uimagmax[i] * 126).astype(np.int32)
                else:
                    self.i8_tlruimag[i] = uimag[i].astype(np.int32) # set 0
        
    def compute(self, x: np.ndarray, trans=False):
        """a general compute interface for tlrmvm"""
        if self.dtype == np.int8:
            return self.int8_compute(x,trans)
        else:
            return self.float_compute(x,trans)
    
    def int8_compute(self,x:np.ndarray,trans=False):
        assert(x.shape[0] == self.pn)
        if trans:
            return self.int8_trans_compute(x)
        # normal int8 computation
        # make x 
        xreal = None
        ximag = None
        xlist = []
        xrealmax = []
        ximagmax = []
        for i in range(self.ntg):
            xreal = x[i*self.nb:(i+1)*self.nb].real
            ximag = x[i*self.nb:(i+1)*self.nb].imag
            xrealmax.append(np.max(np.abs(xreal)))
            ximagmax.append(np.max(np.abs(ximag)))
            if xrealmax[i] > 1e-10:
                xreal = (xreal / xrealmax[i] * 126.).astype(np.int32)
            else:
                xreal = xreal.astype(np.int32) # set 0
            if ximagmax[i] > 1e-10:
                ximag = (ximag / ximagmax[i] * 126.).astype(np.int32)
            else:
                ximag = ximag.astype(np.int32) # set 0
            xlist.append((xreal,ximag))
        p1 = []
        for i in range(self.ntg):
            if self.int8_vrealmax[i] is not None:
                p1.append(self.int8_basic_complex_compute(
                    self.i8_tlrvreal[i],self.i8_tlrvimag[i],
                    xlist[i][0],xlist[i][1],
                    self.int8_vrealmax[i],self.int8_vimagmax[i],
                    xrealmax[i],ximagmax[i]
                ))
            else:
                p1.append(None)
        #split
        p2_real = []
        p2_imag = []
        for i in range(self.ntg):
            cur_real = []
            cur_imag = []
            acc = 0
            for j in range(self.mtg):
                if self.rank[j][i] == 0:
                    cur_real.append(None)
                    cur_imag.append(None)
                else:
                    cur_real.append(p1[i][0][acc : acc + self.rank[j][i]])
                    cur_imag.append(p1[i][1][acc : acc + self.rank[j][i]])
                    acc += self.rank[j][i]
            p2_real.append(cur_real)
            p2_imag.append(cur_imag)
        p3_in_real = merge_2dlist_alongcolumn(p2_real)
        p3_in_imag = merge_2dlist_alongcolumn(p2_imag)
        p3_out = [None for _ in range(self.mtg)]
        ureal = self.recoverfloat(self.i8_tlrureal, self.int8_urealmax)
        uimag = self.recoverfloat(self.i8_tlruimag, self.int8_uimagmax)
        for i in range(self.mtg):
            if ureal[i] is not None:
                p3_out[i] = self.float_basic_complex_compute(
                    ureal[i], uimag[i], p3_in_real[i], p3_in_imag[i]
                )
        p3final = merge_1dlist(p3_out, self.nb)
        return p3final[:self.m]

    def int8_trans_compute(self,x:np.ndarray):
        rank = self.rank.T
        # make x 
        xreal = None
        ximag = None
        xlist = []
        xrealmax = []
        ximagmax = []
        for i in range(self.ntg):
            xreal = x[i*self.nb:(i+1)*self.nb].real
            ximag = x[i*self.nb:(i+1)*self.nb].imag
            xrealmax.append(np.max(np.abs(xreal)))
            ximagmax.append(np.max(np.abs(ximag)))
            if xrealmax[i] > 1e-10:
                xreal = (xreal / xrealmax[i] * 126.).astype(np.int32)
            else:
                xreal = xreal.astype(np.int32) # set 0
            if ximagmax[i] > 1e-10:
                ximag = (ximag / ximagmax[i] * 126.).astype(np.int32)
            else:
                ximag = ximag.astype(np.int32) # set 0
            xlist.append((xreal,ximag))
        p1 = []
        for i in range(self.mtg):
            if self.int8_urealmax[i] is not None:
                p1.append(self.int8_basic_complex_compute(
                    self.i8_tlrureal[i].T, self.i8_tlruimag[i].T,
                    xlist[i][0],xlist[i][1],
                    self.int8_urealmax[i],self.int8_uimagmax[i],
                    xrealmax[i],ximagmax[i]
                ))
            else:
                p1.append(None)
        # split
        p2_real = []
        p2_imag = []
        for i in range(self.mtg):
            cur_real = []
            cur_imag = []
            acc = 0
            for j in range(self.ntg):
                if rank[j][i] == 0:
                    cur_real.append(None)
                    cur_imag.append(None)
                else:
                    cur_real.append(p1[i][0][acc : acc + rank[j][i]])
                    cur_imag.append(p1[i][1][acc : acc + rank[j][i]])
                    acc += rank[j][i]
            p2_real.append(cur_real)
            p2_imag.append(cur_imag)
        p3_in_real = merge_2dlist_alongcolumn(p2_real)
        p3_in_imag = merge_2dlist_alongcolumn(p2_imag)
        p3_out = [None for _ in range(self.mtg)]
        # v bases
        vreal = self.recoverfloat(self.i8_tlrvreal, self.int8_vrealmax)
        vimag = self.recoverfloat(self.i8_tlrvimag, self.int8_vimagmax)
        for i in range(self.ntg):
            if vreal[i] is not None:
                p3_out[i] = self.float_basic_complex_compute(
                    vreal[i].T, vimag[i].T, p3_in_real[i],p3_in_imag[i]
                )
        p3final = merge_1dlist(p3_out, self.nb)
        return p3final[:self.n]

    def recoverfloat(self, i8list, i8max):
        res = [None for _ in range(len(i8list))]
        for i in range(len(i8list)):
            if i8list[i] is not None:
                res[i] = i8list[i].astype(np.float32) * i8max[i] / 126.
        return res

    def int8_basic_complex_compute(self,r1,i1,r2,i2,r1max,i1max,r2max,i2max):
        m1 = r1@r2
        m2 = r1@i2
        m3 = i1@r2
        m4 = i1@i2
        m1 = m1.astype(np.float32) * r1max * r2max / 126. / 126.
        m2 = m2.astype(np.float32) * r1max * i2max / 126. / 126.
        m3 = m3.astype(np.float32) * i1max * r2max / 126. / 126.
        m4 = m4.astype(np.float32) * i1max * i2max / 126. / 126.
        return m1-m4,m2+m3

    def float_compute(self, x: np.ndarray, trans=False):
        """float compute support fp32 and fp16"""
        assert(x.shape[0] == self.pn)
        if trans:
            return self.float_trans_compute(x)
        # normal tlrmvm computation
        # make x 
        xreal = None
        ximag = None
        xlist = []
        for i in range(self.ntg):
            xreal = x[i*self.nb:(i+1)*self.nb].real.astype(self.dtype)
            ximag = x[i*self.nb:(i+1)*self.nb].imag.astype(self.dtype)
            xlist.append((xreal,ximag))
        # v bases        
        vreal = merge_2dlist_alongcolumn(self.vreal)
        vimag = merge_2dlist_alongcolumn(self.vimag)
        p1 = []
        for i in range(self.ntg):
            if vreal[i] is not None:
                p1.append(self.float_basic_complex_compute(
                    vreal[i],vimag[i],
                xlist[i][0],xlist[i][1]))
            else:
                p1.append(None)
        # split
        p2_real = []
        p2_imag = []
        for i in range(self.ntg):
            cur_real = []
            cur_imag = []
            acc = 0
            for j in range(self.mtg):
                if self.rank[j][i] == 0:
                    cur_real.append(None)
                    cur_imag.append(None)
                else:
                    cur_real.append(p1[i][0][acc : acc + self.rank[j][i]])
                    cur_imag.append(p1[i][1][acc : acc + self.rank[j][i]])
                    acc += self.rank[j][i]
            p2_real.append(cur_real)
            p2_imag.append(cur_imag)
        p3_in_real = merge_2dlist_alongcolumn(p2_real)
        p3_in_imag = merge_2dlist_alongcolumn(p2_imag)
        p3_out = [None for _ in range(self.mtg)]
        ureal = merge_2dlist_alongrow(self.ureal)
        uimag = merge_2dlist_alongrow(self.uimag)
        for i in range(self.mtg):
            if ureal[i] is not None:
                p3_out[i] = self.float_basic_complex_compute(
                ureal[i], uimag[i],p3_in_real[i], p3_in_imag[i])
        p3final = merge_1dlist(p3_out, self.nb)
        return p3final[:self.m]

    def float_trans_compute(self, x: np.ndarray):
        rank = self.rank.T # rank transpose
        # make x 
        xreal = None
        ximag = None
        xlist = []
        for i in range(self.mtg):
            xreal = x[i*self.nb:(i+1)*self.nb].real.astype(self.dtype)
            ximag = x[i*self.nb:(i+1)*self.nb].imag.astype(self.dtype)
            xlist.append((xreal,ximag))
        # u bases
        ureal = merge_2dlist_alongrow(self.ureal)
        uimag = merge_2dlist_alongrow(self.uimag)
        p1 = []
        for i in range(self.mtg):
            if ureal[i] is not None:
                p1.append(self.float_basic_complex_compute(
                    ureal[i].T, uimag[i].T, xlist[i][0], xlist[i][1]
                ))
            else:
                p1.append(None)
        # split
        p2_real = []
        p2_imag = []
        for i in range(self.mtg):
            cur_real = []
            cur_imag = []
            acc = 0
            for j in range(self.ntg):
                if rank[j][i] == 0:
                    cur_real.append(None)
                    cur_imag.append(None)
                else:
                    cur_real.append(p1[i][0][acc : acc + rank[j][i]])
                    cur_imag.append(p1[i][1][acc : acc + rank[j][i]])
                    acc += rank[j][i]
            p2_real.append(cur_real)
            p2_imag.append(cur_imag)
        p3_in_real = merge_2dlist_alongcolumn(p2_real)
        p3_in_imag = merge_2dlist_alongcolumn(p2_imag)
        p3_out = [None for _ in range(self.mtg)]
        # v bases        
        vreal = merge_2dlist_alongcolumn(self.vreal)
        vimag = merge_2dlist_alongcolumn(self.vimag)
        for i in range(self.ntg):
            if vreal[i] is not None:
                p3_out[i] = self.float_basic_complex_compute(
                    vreal[i].T, vimag[i].T, p3_in_real[i], p3_in_imag[i]
                )
        p3final = merge_1dlist(p3_out, self.nb)
        return p3final[:self.n]

    def float_basic_complex_compute(self,r1,i1,r2,i2):
        m1 = r1@r2
        m2 = r1@i2
        m3 = i1@r2
        m4 = i1@i2
        return m1-m4,m2+m3


class TLRMat_new:
    """
    Tile Low-Rank Matrix Class.
    1. mimic a single matrix behavior. 
    2. Its MVM is doing TLR-MVM. 
    3. support mixed precision.
    4. support float point and complex data type.
    """
    def __init__(self, m:int, n:int, nb:int) -> None:
        """
        m: original matrix row size.
        n: original matrix col size.
        nb: tile size.
        """
        self.m = m
        self.n = n
        self.nb = nb
        self.inited = False
        # mtg: number of tiles along rows
        # ntg: number of tiles along cols
        # pm: padding row size
        # pn: padding col size
        self.mtg = m // nb if m % nb == 0 else m // nb + 1
        self.ntg = n // nb if n % nb == 0 else n // nb + 1
        self.rank = np.zeros((self.mtg, self.ntg)).astype(np.int32)
        self.pm = self.mtg * nb
        self.pn = self.ntg * nb

    def loadrealdata(self):
        """Load real data from dataset."""
        pass

    def randomrank(self, rank:np.ndarray):
        """Set rank as input random rank. This will also set random data."""
        pass

    def precisionmask(self, fp64=None, fp32=None, fp16=None, int8=None, zero=None):
        """Set precision mask array of each tiles."""
        pass


    


class TLRMat:
    def __init__(self, m, n, filename_prefix, freq_list,
                nb, acc, datafolder, fake_value=None, fp32_mask=None, 
                fp16_mask=None, int8_mask=None):
        self.m = m
        self.n = n
        self.freq_list = freq_list
        self.total_freq = len(self.freq_list)
        self.nb = nb
        self.acc = acc
        self.filename_prefix = filename_prefix
        self.datafolder = datafolder
        m = self.m
        n = self.n
        self.mtg = m // nb if m % nb == 0 else m // nb + 1
        self.ntg = n // nb if n % nb == 0 else n // nb + 1
        self.pm = self.mtg * nb
        self.pn = self.ntg * nb

        # for load function
        self.ubase_map = {}
        self.vbase_map = {}
        self.rank_map = {}
        self.ubase_origin_block = {}
        self.vbase_origin_block = {}
        self.load(fake_value)
        # create TLRMVM class
        self.fp16_mask = fp16_mask
        self.fp32_mask = fp32_mask
        self.int8_mask = int8_mask
        self.tlrmvm_map_list = []
        if fp32_mask is not None:
            self.tlrmvm_fp32_map = {}
            for freq in self.freq_list:
                rank = self.getrank(freq, fp32_mask)
                self.tlrmvm_fp32_map[freq] = TLRMVM(
                    self.ubase_origin_block[freq],
                    self.vbase_origin_block[freq],rank,np.float32,self.nb,
                    self.m,self.n)
            self.tlrmvm_map_list.append(self.tlrmvm_fp32_map)

        if fp16_mask is not None:
            self.tlrmvm_fp16_map = {}
            for freq in self.freq_list:
                rank = self.getrank(freq, fp16_mask)
                self.tlrmvm_fp16_map[freq] = TLRMVM(
                    self.ubase_origin_block[freq],
                    self.vbase_origin_block[freq],rank,np.float16,self.nb,
                    self.m,self.n)
            self.tlrmvm_map_list.append(self.tlrmvm_fp16_map)
        
        if int8_mask is not None:
            self.tlrmvm_int8_map = {}
            for freq in self.freq_list:
                rank = self.getrank(freq, int8_mask)
                self.tlrmvm_int8_map[freq] = TLRMVM(
                    self.ubase_origin_block[freq],
                    self.vbase_origin_block[freq],rank,np.int8,self.nb,
                    self.m,self.n)
            self.tlrmvm_map_list.append(self.tlrmvm_int8_map)
        
        # tile-based computation
        self.tile_y = {}
        # tlrmvm computation
        self.tlrmvm_y = {}

    def resetmask(self,fp32_mask, fp16_mask, int8_mask):
        self.fp16_mask = fp16_mask
        self.fp32_mask = fp32_mask
        self.int8_mask = int8_mask
        self.tlrmvm_map_list = []
        if fp32_mask is not None:
            self.tlrmvm_fp32_map = {}
            for freq in self.freq_list:
                rank = self.getrank(freq, fp32_mask)
                self.tlrmvm_fp32_map[freq] = TLRMVM(
                    self.ubase_origin_block[freq],
                    self.vbase_origin_block[freq],rank,np.float32,self.nb,
                    self.m,self.n)
            self.tlrmvm_map_list.append(self.tlrmvm_fp32_map)

        if fp16_mask is not None:
            self.tlrmvm_fp16_map = {}
            for freq in self.freq_list:
                rank = self.getrank(freq, fp16_mask)
                self.tlrmvm_fp16_map[freq] = TLRMVM(
                    self.ubase_origin_block[freq],
                    self.vbase_origin_block[freq],rank,np.float16,self.nb,
                    self.m,self.n)
            self.tlrmvm_map_list.append(self.tlrmvm_fp16_map)
        
        if int8_mask is not None:
            self.tlrmvm_int8_map = {}
            for freq in self.freq_list:
                rank = self.getrank(freq, int8_mask)
                self.tlrmvm_int8_map[freq] = TLRMVM(
                    self.ubase_origin_block[freq],
                    self.vbase_origin_block[freq],rank,np.int8,self.nb,
                    self.m,self.n)
            self.tlrmvm_map_list.append(self.tlrmvm_int8_map)
        
        # tile-based computation
        self.tile_y = {}
        # tlrmvm computation
        self.tlrmvm_y = {}

    def load(self,fake_value=None):
        # load data and generate origin U,V block tiles
        filename_prefix = self.filename_prefix
        datafolder = join(self.datafolder, 'compresseddata')
        nb = self.nb
        acc = self.acc
        ntg = self.ntg
        mtg = self.mtg
        # load data
        data = []
        for freq in self.freq_list:
            curprefix = "{}{}".format(filename_prefix,freq)
            data.append(seismic_dataloader(curprefix, datafolder, nb, acc, ntg, mtg,
            fake_value))
        cnt = 0
        for freq in self.freq_list:
            self.rank_map[freq] = data[cnt][0]
            self.ubase_map[freq] = data[cnt][1]
            self.vbase_map[freq] = data[cnt][2]
            # print("loading dtype , ", self.ubase_map[freq].dtype): complex64
            cnt += 1
        # generate several precision
        cnt = 0
        for freq in self.freq_list:
            u = self.ubase_map[freq]
            v = self.vbase_map[freq]
            rank = self.rank_map[freq]
            rankcolsum = np.sum(rank, axis=0)
            rankrowsum = np.sum(rank, axis=1)
            # recover u list
            self.ubase_origin_block[freq] = [[] for i in range(mtg)]
            self.vbase_origin_block[freq] = [[] for i in range(mtg)]
            umat = u.reshape(-1, nb).T
            prev = 0
            post = 0
            for i in range(mtg):
                post += rankrowsum[i]
                rowumat = umat[:, prev:post]
                innerprev = 0
                innerpost = 0
                for j in range(ntg):
                    innerpost += rank[i, j]
                    curblock = rowumat[:, innerprev:innerpost]
                    self.ubase_origin_block[freq][i].append(curblock)
                    innerprev += rank[i, j]
                prev += rankrowsum[i]
            prev = 0
            post = 0
            # recover v list
            for i in range(ntg):
                post += rankcolsum[i]
                colvmat = v[prev * nb:post * nb].reshape(nb, rankcolsum[i]).T
                innerprev = 0
                innerpost = 0
                for j in range(mtg):
                    innerpost += rank[j, i]
                    curblock = colvmat[innerprev:innerpost, :]
                    self.vbase_origin_block[freq][j].append(curblock)
                    innerprev += rank[j, i]
                prev += rankcolsum[i]
            cnt += 1
        # print("ubase origin block dtype ",self.ubase_origin_block[100][0][0].dtype)
        # : complex64

    def getrank(self,freq: int, mask : np.ndarray):
        """get rank of a certain precision based on mask array."""
        rank = self.rank_map[freq]
        retrank = np.copy(rank)
        for i in range(self.mtg):
            for j in range(self.ntg):
                if not mask[i][j]:
                    retrank[i][j] = 0
        return retrank

    def matvec(self, xlist: list, trans=False, conj=False):
        """mvm interface"""
        self.ymap = {}
        for idx, freq in enumerate(self.freq_list):
            x = xlist[idx]
            padding_x = np.zeros(self.pn, dtype=np.csingle)
            padding_x[:self.n] = x
            if conj:
                padding_x = np.conj(padding_x)
            y = np.zeros(self.m).astype(np.complex64)
            for idx,tlrmvm in enumerate(self.tlrmvm_map_list):
                out = tlrmvm[freq].compute(padding_x, trans) 
                y += out 
            if conj:
                y = np.conj(y)
            self.ymap[freq] = y

    # main entry of computation
    def tlrmvm_compute(self, xlist: list, trans=False, conj=False):
        """Compute the mvm using tlrmvm algorithms."""
        for idx, freq in enumerate(self.freq_list):
            x = xlist[idx]
            padding_x = np.zeros(self.pn, dtype=np.csingle)
            padding_x[:self.n] = x
            if conj:
                padding_x = np.conj(padding_x)
            y = np.zeros(self.m).astype(np.complex64)
            for tlrmvm in self.tlrmvm_map_list:
                y += tlrmvm[freq].compute(padding_x, trans)
            if conj:
                y = np.conj(y)
            self.tlrmvm_y[freq] = y

    # tile compute, for sanity check
    def tile_compute(self, xlist: list, trans=False, conj=False):
        """Compute the mvm on a tile bases using origin block. \n
        This is for sanity check.\n"""
        for idx, freq in enumerate(self.freq_list):
            x = xlist[idx]
            if conj:
                x = np.conj(x)
            if trans:
                y = self.tile_compute_trans_impl(freq, x)
            else:
                y = self.tile_compute_impl(freq, x)
            if conj:
                y = np.conj(y)
            self.tile_y[freq] = y

    def tile_compute_trans_impl(self, freq: int, x: np.ndarray):
        padding_x = np.zeros(self.pn, dtype=np.csingle)
        padding_x[:self.n] = x
        y = np.zeros(self.pm, dtype=np.csingle)
        for cj in range(self.ntg):
            cur_block = np.zeros(self.nb, dtype=np.csingle)
            for ri in range(self.mtg):
                if self.fp32_mask is not None and self.fp32_mask[ri][cj]:
                    a = self.vbase_origin_block[freq][ri][cj].T @ \
                        self.ubase_origin_block[freq][ri][cj].T
                    cur_block += a @ padding_x[ri * self.nb:(ri + 1) * self.nb]
                elif self.fp16_mask is not None and self.fp16_mask[ri][cj]:
                    m1real=self.vbase_origin_block[freq][ri][cj].T \
                        .real.astype(np.float16).astype(np.float32)
                    m1imag=self.vbase_origin_block[freq][ri][cj].T \
                        .imag.astype(np.float16).astype(np.float32)
                    m2real=self.ubase_origin_block[freq][ri][cj].T \
                        .real.astype(np.float16).astype(np.float32)
                    m2imag=self.ubase_origin_block[freq][ri][cj].T \
                        .imag.astype(np.float16).astype(np.float32)
                    x1real = padding_x[ri * self.nb:(ri + 1) * self.nb].real \
                        .astype(np.float16).astype(np.float32)
                    x1imag = padding_x[ri * self.nb:(ri + 1) * self.nb].imag \
                        .astype(np.float16).astype(np.float32)
                    m1 = m1real + 1j * m1imag
                    m2 = m2real + 1j * m2imag
                    x = x1real + 1j * x1imag
                    out = m1 @ m2 @ x
                    cur_block += out
            y[cj*self.nb:(cj+1)*self.nb] = cur_block
        return y[:self.m]

    def tile_compute_impl(self, freq: int, x: np.ndarray):
        padding_x = np.zeros(self.pn, dtype=np.csingle)
        padding_x[:self.n] = x
        y = np.zeros(self.pm, dtype=np.csingle)
        for ri in range(self.mtg):
            cur_block = np.zeros(self.nb, dtype=np.csingle)
            for cj in range(self.ntg):
                if self.fp32_mask is not None and self.fp32_mask[ri][cj]:
                    a = self.ubase_origin_block[freq][ri][cj] @ \
                        self.vbase_origin_block[freq][ri][cj]
                    cur_block += a @ padding_x[cj * self.nb:(cj + 1) * self.nb]
                elif self.fp16_mask is not None and self.fp16_mask[ri][cj]:
                    m1real=self.ubase_origin_block[freq][ri][cj] \
                        .real.astype(np.float16).astype(np.float32)
                    m1imag=self.ubase_origin_block[freq][ri][cj] \
                        .imag.astype(np.float16).astype(np.float32)
                    m2real=self.vbase_origin_block[freq][ri][cj] \
                        .real.astype(np.float16).astype(np.float32)
                    m2imag=self.vbase_origin_block[freq][ri][cj] \
                        .imag.astype(np.float16).astype(np.float32)
                    x1real = padding_x[cj * self.nb:(cj + 1) * self.nb].real \
                        .astype(np.float16).astype(np.float32)
                    x1imag = padding_x[cj * self.nb:(cj + 1) * self.nb].imag \
                        .astype(np.float16).astype(np.float32)
                    m1 = m1real + 1j * m1imag
                    m2 = m2real + 1j * m2imag
                    x = x1real + 1j * x1imag
                    out = m1 @ m2 @ x
                    cur_block += out
            y[ri*self.nb:(ri+1)*self.nb] = cur_block
        return y[:self.m]


class TLRMat_old:
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
        self.rank,self.u,self.v = seismic_dataloader(self.problemname, self.datafolder, 
        nb,acc,ntg,mtg)
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
        self.phase1out = tmpy
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
        self.phase2out = res
        ## phase 3 
        tlrmvmyout = []
        uoffset = 0
        for i in range(mtg):
            tlrmvmyout.append(u[(uoffset*nb) : (uoffset+rankrowsum[i])*nb].reshape(-1,nb).T @ res[i])
            uoffset += rankrowsum[i]
        tlrmvmout = np.concatenate(tlrmvmyout)
        self.y = tlrmvmout[:x.shape[0]]
        return tlrmvmout[:x.shape[0]]
