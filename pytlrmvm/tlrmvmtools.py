#   @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
#                       All rights reserved.

##################################################################
#
# Author: Yuxi Hong
# Description: A tools for generating compressed U and V bases
# which are input of TLR-MVM.
##################################################################
import os
from os.path import join, exists
from tqdm import tqdm
import numpy as np
import pickle
from scipy.linalg import svd


class TLRMVM_Util:
    """A TLR-MVM Utility class
    1. compute svd for input of TLR-MVM
    3. save U and V bases
    4. save Dense matrix
    """

    def __init__(self, denseAarray, nb, datafolder, error_threshold, problemname, rankmodule) -> None:
        self.denseA = denseAarray
        self.dtype = denseAarray.dtype
        self.m = denseAarray.shape[0]
        self.n = denseAarray.shape[1]

        self.nb = nb
        self.mtg = self.m // nb if self.m % nb == 0 else self.m // nb + 1
        self.ntg = self.n // nb if self.n % nb == 0 else self.n // nb + 1
        self.paddingm = self.mtg * nb
        self.paddingn = self.ntg * nb
        self.datafolder = datafolder
        if not exists(self.datafolder):
            print("Folder {} not exists!".format(self.datafolder))
        self.error_threshold = error_threshold
        self.problemname = problemname
        self.rankfile = join(self.datafolder, 'compresseddata',
                             '{}_Rmat_nb{}_acc{}.bin'.format(self.problemname, self.nb, self.error_threshold))
        self.rankmodule = rankmodule

    def computesvd(self):
        A = self.denseA
        padding_m = self.paddingm
        padding_n = self.paddingn
        m = self.m
        n = self.n
        mtiles = self.mtg
        ntiles = self.ntg
        svdsavepath = join(self.datafolder, 'SVDinfo')
        if not exists(svdsavepath):
            os.mkdir(svdsavepath)
        nb = self.nb
        svdname = join(svdsavepath, '{}_nb{}.pickle'.format(self.problemname, nb))
        if exists(svdname):
            print("svd {} exists.".format(svdname))
            return
        else:
            print("save svd to {}. ".format(svdname))
        bigmap = dict()
        padA = np.zeros((padding_m, padding_n), dtype=self.dtype)
        padA[:m, :n] = A
        for j in tqdm(range(ntiles)):
            for i in range(mtiles):
                curblock = padA[i * nb:(i + 1) * nb, j * nb:(j + 1) * nb]
                [u, s, v] = svd(curblock)
                bigmap['{}_{}'.format(i, j)] = [u, s, v]
        with open(svdname, 'wb') as f:
            pickle.dump(bigmap, f)

    def saveX(self, xvec):
        xfile = join(self.datafolder, '{}_x.bin'.format(self.problemname))
        xvec.tofile(xfile)

    def saveUV(self):
        svdname = join(self.datafolder, 'SVDinfo', '{}_nb{}.pickle'.format(self.problemname, self.nb))
        if not exists(svdname):
            print("please do svd to matrix first!")
        with open(svdname, 'rb') as f:
            bigmap = pickle.load(f)
        nb = self.nb
        acc = self.error_threshold
        uvsavepath = join(self.datafolder, 'compresseddata')
        if not exists(uvsavepath):
            os.mkdir(uvsavepath)
        ufile = uvsavepath + '/{}_Ubases_nb{}_acc{}.bin'.format(self.problemname, nb, acc)
        vfile = uvsavepath + '/{}_Vbases_nb{}_acc{}.bin'.format(self.problemname, nb, acc)
        rfile = uvsavepath + '/{}_Rmat_nb{}_acc{}.bin'.format(self.problemname, nb, acc)

        print("generate uvr file to {}.".format(uvsavepath))
        padding_m = self.paddingm
        padding_n = self.paddingn
        m = self.m
        n = self.n
        ntiles = self.ntg
        mtiles = self.mtg
        nb = self.nb
        tmpacc = self.error_threshold
        acc = tmpacc if isinstance(tmpacc, float) else float(tmpacc)
        ApproximateA = np.zeros((padding_m, padding_n), dtype=self.dtype)
        originpadA = np.zeros((padding_m, padding_n), dtype=self.dtype)
        originpadA[:m, :n] = self.denseA
        normA = np.linalg.norm(self.denseA, 'fro')
        ranklist = np.zeros((mtiles, ntiles), dtype=np.int32)
        print("rankmat shape, ", ranklist.shape)
        ulist = [[] for _ in range(mtiles)]
        vlist = [[] for _ in range(mtiles)]

        p = mtiles
        for i in tqdm(range(mtiles - 1)):
            for j in range(ntiles - 1):
                curblock = originpadA[i * nb:(i + 1) * nb, j * nb:(j + 1) * nb]
                normblock = np.linalg.norm(curblock, 'fro')
                [u, s, v] = bigmap['{}_{}'.format(i, j)]
                srk = 0
                erk = nb
                while srk != erk:
                    midrk = (srk + erk) // 2
                    tmpu = u[:, :midrk]
                    tmps = s[:midrk]
                    tmpv = v[:midrk, :]
                    if np.linalg.norm(curblock - (tmpu * tmps) @ tmpv, ord='fro') < normA * acc:
                        erk = midrk
                    else:
                        srk = midrk + 1
                if srk == 0:
                    srk = 1
                tmpu = u[:, :srk]
                tmps = s[:srk]
                tmpv = v[:srk, :]
                ApproximateA[i * nb:(i + 1) * nb, j * nb:(j + 1) * nb] = (tmpu * tmps) @ tmpv
                us = tmpu * tmps
                vt = tmpv
                if srk == 0:
                    ranklist[i, j] = 1
                    ulist[i].append(np.zeros((nb, 1), dtype=self.dtype))
                    vlist[i].append(np.zeros((1, nb), dtype=self.dtype))
                else:
                    ranklist[i, j] = srk
                    ulist[i].append(us)
                    vlist[i].append(vt)

        def getsrk(normA, nb, acc, u, s, v):
            srk = 0
            erk = nb
            while srk != erk:
                midrk = (srk + erk) // 2
                tmpu = u[:, :midrk]
                tmps = s[:midrk]
                tmpv = v[:midrk, :]
                if np.linalg.norm(curblock - (tmpu * tmps) @ tmpv, ord='fro') < normA * acc:
                    erk = midrk
                else:
                    srk = midrk + 1
            return srk

        for i in tqdm(range(mtiles)):
            for j in range(ntiles):
                if i < mtiles - 1 and j < ntiles - 1:
                    continue
                curblock = originpadA[i * nb:(i + 1) * nb, j * nb:(j + 1) * nb]
                normblock = np.linalg.norm(curblock, 'fro')
                [u, s, v] = bigmap['{}_{}'.format(i, j)]
                if i < mtiles - 1 or j < ntiles - 1:
                    if i == mtiles - 1:
                        presum = np.sum(ranklist[:, j])
                        srk = getsrk(normA, nb, acc, u, s, v)
                        while srk < nb and (srk + presum) % self.rankmodule != 0:
                            srk += 1

                        if srk == nb and (srk + presum) % self.rankmodule != 0:
                            print("can't find a solution! i = mtiles")
                            exit()
                        else:
                            ranklist[i, j] = srk
                    elif j == ntiles - 1:
                        presum = np.sum(ranklist[i, :])
                        srk = getsrk(normA, nb, acc, u, s, v)
                        while srk < nb and (srk + presum) % self.rankmodule != 0:
                            srk += 1
                        if srk == nb and (srk + presum) % self.rankmodule != 0:
                            print("can't find a solution! j = ntiles")
                            exit()
                        else:
                            ranklist[i, j] = srk
                elif i == mtiles - 1 and j == ntiles - 1:
                    srk = 0
                    while srk < nb and (srk + np.sum(ranklist[i, :])) % self.rankmodule != 0 and \
                            (srk + np.sum(ranklist[:, j])) % self.rankmodule != 0:
                        srk += 1
                    if srk == nb:
                        print("can't find a solution!")
                        exit()
                    else:
                        ranklist[i, j] = srk
                if srk == 0:
                    srk = self.rankmodule
                tmpu = u[:, :srk]
                tmps = s[:srk]
                tmpv = v[:srk, :]
                ApproximateA[i * nb:(i + 1) * nb, j * nb:(j + 1) * nb] = (tmpu * tmps) @ tmpv
                us = tmpu * tmps
                vt = tmpv
                if srk == 0:
                    ranklist[i, j] = 1
                    ulist[i].append(np.zeros((nb, 1), dtype=self.dtype))
                    vlist[i].append(np.zeros((1, nb), dtype=self.dtype))
                else:
                    ranklist[i, j] = srk
                    ulist[i].append(us)
                    vlist[i].append(vt)
        tmpurow = []
        for x in ulist:
            tmpurow.append(np.concatenate(x, axis=1))
        finalu = np.concatenate(tmpurow, axis=1)
        finalu.T.tofile(ufile)
        tmpvcol = []
        npvlist = np.array(vlist, dtype=np.object)
        for i in range(npvlist.shape[1]):
            tmpvcol.append(np.concatenate(npvlist[:, i], axis=0))

        with open(vfile, 'wb') as f:
            for x in tmpvcol:
                x.T.tofile(f)
        ranklist.T.tofile(rfile)

    def printdatainfo(self):
        print("Description of Dataset: ")
        print("problem name : {} ".format(self.problemname))
        print("m is {} n {} nb is {} error threshold is {}.".format(self.m, self.n, self.nb, self.error_threshold))
        rankfile = join(self.datafolder, 'compresseddata',
                        '{}_Rmat_nb{}_acc{}.bin'.format(self.problemname, self.nb, self.error_threshold))
        self.ranklist = np.fromfile(rankfile, dtype=np.int32)
        mn = self.m * self.n
        rank = np.sum(self.ranklist)
        print("Global rank is {}, compression rate is {:.3f}%.".format(rank, 2 * rank * self.nb / mn * 100))


if __name__ == "__main__":
    import numpy as np
    import os
    import argparse
    from astropy.io.fits import open as fitsopen
    from scipy.io import loadmat
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb', type=int, help='nb')
    parser.add_argument('--error_threshold', type=str,help='error threshold.')
    parser.add_argument('--compressed_name', type=str, help='The file name for compressed U,V,and R.')
    parser.add_argument('--data_dir', type=str, help='your original data dir.')
    parser.add_argument('--data_type', type=str, help='datatype of dataset.')
    parser.add_argument('--data_name', type=str, help='The name of original matrix.')
    parser.add_argument('--matlabmat_name', type=str, default=None, help='The name of original matrix in .mat file.')
    parser.add_argument('--rank_module', type=int, help='rank module.')
    args = parser.parse_args()
    dtype = None
    datatype = args.data_type
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
    A = None
    if args.data_name.split('.')[-1] == 'npy':
        A = np.load(join(args.data_dir,args.data_name)).astype(dtype)
    elif args.data_name.split('.')[-1] == 'fits':
        A = fitsopen(join(args.data_dir,args.data_name))[0].data.astype(dtype)
    elif args.data_name.split('.')[-1] == 'mat':
        A = loadmat(join(args.data_dir,args.data_name))[args.matlabmat_name]
    else:
        A = pickle.load(open(join(args.data_dir,args.data_name))).astype(dtype)
    rankmodule = int(args.rank_module)
    if rankmodule == 0:
        print("not 0.")
        exit()
    tlrmvmutil = TLRMVM_Util(A, args.nb, args.data_dir, args.error_threshold, args.compressed_name, rankmodule)
    # compute svd and save
    tlrmvmutil.computesvd()
    # create input of tlrmvm
    tlrmvmutil.saveUV()
    # get compression info
    tlrmvmutil.printdatainfo()
