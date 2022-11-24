#   @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
#                       All rights reserved.

import TLRMVMpy as _cppimpl
import numpy as _np
from time import time as _time

def allclose(yreal,yapprox):
    minlen = min(yreal.shape[0], yapprox.shape[0])
    yreal = yreal[:minlen]
    yapprox = yapprox[:minlen]
    if _np.max(_np.abs(yreal)) == _np.max(_np.abs(yapprox)) and _np.max(_np.abs(yreal)) == 0:
        return 0
    return _np.max(_np.abs(yreal - yapprox)) / _np.max(_np.abs(yreal))



class TlrmvmConfig:
    def __init__(self, m, n, nb, datafolder, error_threshold, datasetname) -> None:
        self._cppinst = _cppimpl.TLRMVMConfig(m, n, nb, datafolder, error_threshold, datasetname)
        self.Mtg = self._cppinst.Mtg
        self.Ntg = self._cppinst.Ntg
        self.granksum = self._cppinst.granksum
        self.paddingM = self._cppinst.paddingM
        self.paddingN = self._cppinst.paddingN
        self.m = m
        self.n = n
        self.nb = nb
        self.datafolder = datafolder
        self.error_threshold = error_threshold
        self.datasetname = datasetname

    def setmaskmat(self, maskmat):
        _cppimpl.SetMaskmat(maskmat, self._cppinst)

    def printmaskmat(self):
        self._cppinst.PrintMaskmat()


class Tlrmvm:
    def __init__(self, tlrmvmconfig, dtype, ) -> None:
        if dtype == _np.float32:
            self._cppinst = _cppimpl.TLRMVMCPUFloat(tlrmvmconfig._cppinst)
        elif dtype == _np.csingle:
            self._cppinst = _cppimpl.TLRMVMCPUComplexFloat(tlrmvmconfig._cppinst)
        elif dtype == _np.float64:
            self._cppinst = _cppimpl.TLRMVMCPUDouble(tlrmvmconfig._cppinst)
        elif dtype == _np.cdouble:
            self._cppinst = _cppimpl.TLRMVMCPUComplexDouble(tlrmvmconfig._cppinst)
        else:
            print("unsupported datatype")
            exit()
        self.dtype = dtype
        self.granksum = self._cppinst.config.granksum
        self.originM = self._cppinst.config.originM
        self.originN = self._cppinst.config.originN
        self.paddingM = self._cppinst.config.paddingM
        self.paddingN = self._cppinst.config.paddingN
        self.nb = tlrmvmconfig.nb

        self.Av = _np.zeros(self.granksum * self.nb, dtype=dtype)
        self.Au = _np.zeros(self.granksum * self.nb, dtype=dtype)
        self.x = _np.zeros(self.paddingM, dtype=dtype)
        self.yu = _np.zeros(self.granksum, dtype=dtype)
        self.yv = _np.zeros(self.granksum, dtype=dtype)
        self.y = _np.zeros(self.originM, dtype=dtype)

    def MemoryInit(self):
        dtype = self.dtype
        self._cppinst.MemoryInit()

    def MemoryFree(self):
        self._cppinst.MemoryFree()

    def SetTransposeConjugate(self, transpose=False, conjugate=False):
        self._cppinst.SetTransposeConjugate(transpose, conjugate)

    def CopyBackResults(self):
        self._cppinst.CopyBackResults()

    def MVM(self, x):
        dtype = self.dtype

        if dtype == _np.float32:
            _cppimpl.Updatex_f(x, self._cppinst)
        elif dtype == _np.csingle:
            _cppimpl.Updatex_cf(x, self._cppinst)
        elif dtype == _np.float64:
            _cppimpl.Updatex_d(x, self._cppinst)
        elif dtype == _np.cdouble:
            _cppimpl.Updatex_cd(x, self._cppinst)

        self._cppinst.MVM()
        if dtype == _np.float32:
            _cppimpl.Updateyv_f(self.yv, self._cppinst, self.granksum)
            _cppimpl.Updateyu_f(self.yu, self._cppinst, self.granksum)
            _cppimpl.Updatey_f(self.y, self._cppinst, self.originM)
        elif dtype == _np.csingle:
            _cppimpl.Updateyv_cf(self.yv, self._cppinst, self.granksum)
            _cppimpl.Updateyu_cf(self.yu, self._cppinst, self.granksum)
            _cppimpl.Updatey_cf(self.y, self._cppinst, self.originM)
        elif dtype == _np.float64:
            _cppimpl.Updateyv_d(self.yv, self._cppinst, self.granksum)
            _cppimpl.Updateyu_d(self.yu, self._cppinst, self.granksum)
            _cppimpl.Updatey_d(self.y, self._cppinst, self.originM)
        elif dtype == _np.cdouble:
            _cppimpl.Updateyv_cd(self.yv, self._cppinst, self.granksum)
            _cppimpl.Updateyu_cd(self.yu, self._cppinst, self.granksum)
            _cppimpl.Updatey_cd(self.y, self._cppinst, self.originM)

    def Phase1(self):
        dtype = self.dtype
        self._cppinst.Phase1()
        if dtype == _np.float32:
            _cppimpl.Updateyv_f(self.yv, self._cppinst, self.granksum)
        elif dtype == _np.csingle:
            _cppimpl.Updateyv_cf(self.yv, self._cppinst, self.granksum)
        elif dtype == _np.float64:
            _cppimpl.Updateyv_d(self.yv, self._cppinst, self.granksum)
        elif dtype == _np.cdouble:
            _cppimpl.Updateyv_cd(self.yv, self._cppinst, self.granksum)

    def Phase2(self):
        dtype = self.dtype
        self._cppinst.Phase2()
        if dtype == _np.float32:
            _cppimpl.Updateyu_f(self.yu, self._cppinst, self.granksum)

        elif dtype == _np.csingle:
            _cppimpl.Updateyu_cf(self.yu, self._cppinst, self.granksum)

        elif dtype == _np.float64:
            _cppimpl.Updateyu_d(self.yu, self._cppinst, self.granksum)

        elif dtype == _np.cdouble:
            _cppimpl.Updateyu_cd(self.yu, self._cppinst, self.granksum)


    def Phase3(self):
        dtype = self.dtype
        self._cppinst.Phase3()
        if dtype == _np.float32:
            _cppimpl.Updatey_f(self.y, self._cppinst, self.originM)
        elif dtype == _np.csingle:
            _cppimpl.Updatey_cf(self.y, self._cppinst, self.originM)
        elif dtype == _np.float64:
            _cppimpl.Updatey_d(self.y, self._cppinst, self.originM)
        elif dtype == _np.cdouble:
            _cppimpl.Updatey_cd(self.y, self._cppinst, self.originM)



class TlrmvmGPU:
    def __init__(self, tlrmvmconfig, dtype, ) -> None:
        if dtype == _np.float32:
            self._cppinst = _cppimpl.TLRMVMGPUFloat(tlrmvmconfig._cppinst)
        elif dtype == _np.csingle:
            self._cppinst = _cppimpl.TLRMVMGPUComplexFloat(tlrmvmconfig._cppinst)
        elif dtype == _np.float64:
            self._cppinst = _cppimpl.TLRMVMGPUDouble(tlrmvmconfig._cppinst)
        elif dtype == _np.cdouble:
            self._cppinst = _cppimpl.TLRMVMGPUComplexDouble(tlrmvmconfig._cppinst)
        else:
            print("unsupported datatype")
            exit()
        self.dtype = dtype
        self.granksum = self._cppinst.config.granksum
        self.originM = self._cppinst.config.originM
        self.originN = self._cppinst.config.originN
        self.paddingM = self._cppinst.config.paddingM
        self.paddingN = self._cppinst.config.paddingN
        self.nb = tlrmvmconfig.nb

        self.Av = _np.zeros(self.granksum * self.nb, dtype=dtype)
        self.Au = _np.zeros(self.granksum * self.nb, dtype=dtype)
        self.x = _np.zeros(self.paddingM, dtype=dtype)
        self.yu = _np.zeros(self.granksum, dtype=dtype)
        self.yv = _np.zeros(self.granksum, dtype=dtype)
        self.y = _np.zeros(self.originM, dtype=dtype)

        self.transpose = False
        self.conjugate = False

    def MemoryInit(self):
        self._cppinst.MemoryInit()

    def MemoryFree(self):
        self._cppinst.MemoryFree()

    def StreamInit(self, ns):
        self._cppinst.StreamInit(ns)

    def StreamDestroy(self):
        self._cppinst.StreamDestroy()

    def Updatex(self,x):
        dtype = self.dtype
        if dtype == _np.float32:
            _cppimpl.Updatexgpu_f(x,self._cppinst)
        elif dtype == _np.csingle:
            _cppimpl.Updatexgpu_cf(x,self._cppinst)
        elif dtype == _np.float64:
            _cppimpl.Updatexgpu_d(x,self._cppinst)
        elif dtype == _np.cdouble:
            _cppimpl.Updatexgpu_cd(x,self._cppinst)


    def SetTransposeConjugate(self, transpose=False, conjugate=False):
        self.transpose = transpose
        self.conjugate = conjugate
        self._cppinst.SetTransposeConjugate(transpose, conjugate)

    def CopyBackResults(self):
        self._cppinst.CopyBackResults()

    def MVM(self, x):
        dtype = self.dtype
        if dtype == _np.float32:
            _cppimpl.Updatexgpu_f(x, self._cppinst)
        elif dtype == _np.csingle:
            _cppimpl.Updatexgpu_cf(x, self._cppinst)
        elif dtype == _np.float64:
            _cppimpl.Updatexgpu_d(x, self._cppinst)
        elif dtype == _np.cdouble:
            _cppimpl.Updatexgpu_cd(x, self._cppinst)
        self._cppinst.SetTransposeConjugate(self.transpose, self.conjugate)
        self._cppinst.TryConjugateXvec()
        self._cppinst.MVM()
        self._cppinst.TryConjugateResults()
        self._cppinst.CopyBackResults()
        if dtype == _np.float32:
            _cppimpl.Updateyvgpu_f(self.yv, self._cppinst, self.granksum)
            _cppimpl.Updateyugpu_f(self.yu, self._cppinst, self.granksum)
            _cppimpl.Updateygpu_f(self.y, self._cppinst, self.originM)
        elif dtype == _np.csingle:
            _cppimpl.Updateyvgpu_cf(self.yv, self._cppinst, self.granksum)
            _cppimpl.Updateyugpu_cf(self.yu, self._cppinst, self.granksum)
            _cppimpl.Updateygpu_cf(self.y, self._cppinst, self.originM)
        elif dtype == _np.float64:
            _cppimpl.Updateyvgpu_d(self.yv, self._cppinst, self.granksum)
            _cppimpl.Updateyugpu_d(self.yu, self._cppinst, self.granksum)
            _cppimpl.Updateygpu_d(self.y, self._cppinst, self.originM)
        elif dtype == _np.cdouble:
            _cppimpl.Updateyvgpu_cd(self.yv, self._cppinst, self.granksum)
            _cppimpl.Updateyugpu_cd(self.yu, self._cppinst, self.granksum)
            _cppimpl.Updateygpu_cd(self.y, self._cppinst, self.originM)


    def MVMGraph(self, x):
        print("ERROR")
        return

    def Phase1(self):
        dtype = self.dtype
        self._cppinst.Phase1()
        if dtype == _np.float32:
            _cppimpl.Updateyvgpu_f(self.yv, self._cppinst, self.granksum)
        elif dtype == _np.csingle:
            _cppimpl.Updateyvgpu_cf(self.yv, self._cppinst, self.granksum)
        elif dtype == _np.float64:
            _cppimpl.Updateyvgpu_d(self.yv, self._cppinst, self.granksum)
        elif dtype == _np.cdouble:
            _cppimpl.Updateyvgpu_cd(self.yv, self._cppinst, self.granksum)

    def Phase2(self):
        dtype = self.dtype
        self._cppinst.Phase2()
        if dtype == _np.float32:
            _cppimpl.Updateyugpu_f(self.yu, self._cppinst, self.granksum)

        elif dtype == _np.csingle:
            _cppimpl.Updateyugpu_cf(self.yu, self._cppinst, self.granksum)

        elif dtype == _np.float64:
            _cppimpl.Updateyugpu_d(self.yu, self._cppinst, self.granksum)

        elif dtype == _np.cdouble:
            _cppimpl.Updateyugpu_cd(self.yu, self._cppinst, self.granksum)


    def Phase3(self):
        self._cppinst.Phase3()
        if self.dtype == _np.float32:
            _cppimpl.Updateygpu_f(self.y, self._cppinst, self.originM)
        elif self.dtype == _np.csingle:
            _cppimpl.Updateygpu_cf(self.y, self._cppinst, self.originM)
        elif self.dtype == _np.float64:
            _cppimpl.Updateygpu_d(self.y, self._cppinst, self.originM)
        elif self.dtype == _np.cdouble:
            _cppimpl.Updateygpu_cd(self.y, self._cppinst, self.originM)


class BatchTlrmvmGPU:
    def __init__(self, configvec, dtype, bandlen = None) -> None:
        inconfig = []
        mtg,ntg = configvec[0].Ntg,configvec[0].Ntg
        print(mtg,ntg)
        masknp = None
        if bandlen is not None:
            masknp = _np.zeros((39,39),dtype=_np.int32)
            for i in range(mtg):
                for j in range(ntg):
                    if abs(i-j) >= bandlen:
                        masknp[i,j] = 0
                    else:
                        masknp[i,j] = 1
        print(masknp)
        for x in configvec:
            if masknp is not None:
                x.setmaskmat(masknp)
            inconfig.append(x._cppinst)
        if dtype != _np.csingle:
            print("only support csingle for now")
            exit(0)
        self._cppinst = _cppimpl.BatchTLRMVMGPUComplexFloat(inconfig)

        self.dtype = dtype

        self.originM = self._cppinst.config_vec[0].originM
        self.originN = self._cppinst.config_vec[0].originN
        self.paddingM = self._cppinst.config_vec[0].paddingM
        self.paddingN = self._cppinst.config_vec[0].paddingN
        self.nb = self._cppinst.config_vec[0].nb
        self.batchsize = len(configvec)

        self.x = _np.zeros(self.originN * self.batchsize, dtype=dtype)
        self.y = _np.zeros(self.originM * self.batchsize, dtype=dtype)

        self.transpose = False
        self.conjugate = False
        self.executiontime = 0
        self.KeyMVM = 0

    def MemoryInit(self):
        self._cppinst.MemoryInit()

    def MemoryFree(self):
        self._cppinst.MemoryFree()

    def StreamInit(self, ns):
        self._cppinst.StreamInit(ns)

    def Updatex(self,x):
        dtype = self.dtype
        if dtype == _np.float32:
            _cppimpl.Updatexgpu_f(x,self._cppinst)
        elif dtype == _np.csingle:
            _cppimpl.Updatexgpu_cf(x,self._cppinst)
        elif dtype == _np.float64:
            _cppimpl.Updatexgpu_d(x,self._cppinst)
        elif dtype == _np.cdouble:
            _cppimpl.Updatexgpu_cd(x,self._cppinst)

    def SetTransposeConjugate(self, transpose=False, conjugate=False):
        self.transpose = transpose
        self.conjugate = conjugate
        self._cppinst.SetTransposeConjugate(transpose, conjugate)

    def CopyBackResults(self):
        self._cppinst.CopyBackResults()

    def MVM(self, x):
        t0 = _time()
        dtype = self.dtype
        if dtype == _np.float32:
            _cppimpl.BatchUpdatexgpu_f(x, self._cppinst)
        elif dtype == _np.csingle:
            _cppimpl.BatchUpdatexgpu_cf(x, self._cppinst)
        elif dtype == _np.float64:
            _cppimpl.BatchUpdatexgpu_d(x, self._cppinst)
        elif dtype == _np.cdouble:
            _cppimpl.BatchUpdatexgpu_cd(x, self._cppinst)
        self._cppinst.SetTransposeConjugate(self.transpose, self.conjugate)
        self._cppinst.TryConjugateXvec()
        kt0 = _time()
        self._cppinst.MVM()
        kt1 = _time()
        self._cppinst.TryConjugateResults()
        self._cppinst.CopyBackResults()
        if dtype == _np.csingle:
            _cppimpl.BatchUpdateygpu_cf(self.y, self._cppinst, self.originM * self.batchsize)
        t1 = _time()
        self.executiontime += t1 - t0
        self.KeyMVM += kt1 - kt0
        return self.y


class BatchTlrmvmGPUFP16:
    def __init__(self, configvec, dtype) -> None:
        inconfig = []
        for x in configvec:
            inconfig.append(x._cppinst)
        if dtype != _np.csingle:
            print("only support csingle for now")
            exit(0)
        self._cppinst = _cppimpl.BatchTLRMVMGPUFP16(inconfig)
        self.dtype = dtype

        self.originM = self._cppinst.config_vec[0].originM
        self.originN = self._cppinst.config_vec[0].originN
        self.paddingM = self._cppinst.config_vec[0].paddingM
        self.paddingN = self._cppinst.config_vec[0].paddingN
        self.nb = self._cppinst.config_vec[0].nb
        self.batchsize = len(configvec)

        self.x = _np.zeros(self.originN * self.batchsize, dtype=dtype)
        self.y = _np.zeros(self.originM * self.batchsize, dtype=dtype)

        self.transpose = False
        self.conjugate = False
        self.executiontime = 0
        self.KeyMVM = 0

    def MemoryInit(self):
        self._cppinst.MemoryInit()

    def MemoryFree(self):
        self._cppinst.MemoryFree()

    def StreamInit(self, ns):
        self._cppinst.StreamInit(ns)

    def SetTransposeConjugate(self, transpose=False, conjugate=False):
        self.transpose = transpose
        self.conjugate = conjugate
        self._cppinst.SetTransposeConjugate(transpose, conjugate)

    def CopyBackResults(self):
        self._cppinst.CopyBackResults()

    def MVM(self, x):
        t0 = _time()
        dtype = self.dtype
        _cppimpl.BatchUpdatexgpu_FP16_cf(x, self._cppinst)
        self._cppinst.SetTransposeConjugate(self.transpose, self.conjugate)
        self._cppinst.TryConjugateXvec()
        kt0 = _time()
        self._cppinst.MVM()
        kt1 = _time()
        self._cppinst.TryConjugateResults()
        self._cppinst.CopyBackResults()
        _cppimpl.BatchUpdateygpu_FP16_cf(self.y, self._cppinst, self.originM * self.batchsize)
        t1 = _time()
        self.executiontime += t1 - t0
        self.KeyMVM += kt1 - kt0
        return self.y

class BatchTlrmvmGPUBF16:
    def __init__(self, configvec, dtype) -> None:
        inconfig = []
        for x in configvec:
            inconfig.append(x._cppinst)
        if dtype != _np.csingle:
            print("only support csingle for now")
            exit(0)
        self._cppinst = _cppimpl.BatchTLRMVMGPUBF16(inconfig)
        self.dtype = dtype

        self.originM = self._cppinst.config_vec[0].originM
        self.originN = self._cppinst.config_vec[0].originN
        self.paddingM = self._cppinst.config_vec[0].paddingM
        self.paddingN = self._cppinst.config_vec[0].paddingN
        self.nb = self._cppinst.config_vec[0].nb
        self.batchsize = len(configvec)

        self.x = _np.zeros(self.originN * self.batchsize, dtype=dtype)
        self.y = _np.zeros(self.originM * self.batchsize, dtype=dtype)

        self.transpose = False
        self.conjugate = False
        self.executiontime = 0
        self.KeyMVM = 0

    def MemoryInit(self):
        self._cppinst.MemoryInit()

    def MemoryFree(self):
        self._cppinst.MemoryFree()

    def StreamInit(self, ns):
        self._cppinst.StreamInit(ns)

    def SetTransposeConjugate(self, transpose=False, conjugate=False):
        self.transpose = transpose
        self.conjugate = conjugate
        self._cppinst.SetTransposeConjugate(transpose, conjugate)

    def CopyBackResults(self):
        self._cppinst.CopyBackResults()

    def MVM(self, x):
        t0 = _time()
        dtype = self.dtype
        _cppimpl.BatchUpdatexgpu_BF16_cf(x, self._cppinst)
        self._cppinst.SetTransposeConjugate(self.transpose, self.conjugate)
        self._cppinst.TryConjugateXvec()
        kt0 = _time()
        self._cppinst.MVM()
        kt1 = _time()
        self._cppinst.TryConjugateResults()
        self._cppinst.CopyBackResults()
        _cppimpl.BatchUpdateygpu_BF16_cf(self.y, self._cppinst, self.originM * self.batchsize)
        t1 = _time()
        self.executiontime += t1 - t0
        self.KeyMVM += kt1 - kt0
        return self.y


class BatchTlrmvmGPUINT8:
    def __init__(self, configvec, dtype) -> None:
        inconfig = []
        for x in configvec:
            inconfig.append(x._cppinst)
        if dtype != _np.csingle:
            print("only support csingle for now")
            exit(0)
        self._cppinst = _cppimpl.BatchTLRMVMGPUINT8(inconfig)
        self.dtype = dtype

        self.originM = self._cppinst.config_vec[0].originM
        self.originN = self._cppinst.config_vec[0].originN
        self.paddingM = self._cppinst.config_vec[0].paddingM
        self.paddingN = self._cppinst.config_vec[0].paddingN
        self.nb = self._cppinst.config_vec[0].nb
        self.batchsize = len(configvec)

        self.x = _np.zeros(self.originN * self.batchsize, dtype=dtype)
        self.y = _np.zeros(self.originM * self.batchsize, dtype=dtype)

        self.transpose = False
        self.conjugate = False
        self.executiontime = 0
        self.KeyMVM = 0

    def MemoryInit(self):
        self._cppinst.MemoryInit()

    def MemoryFree(self):
        self._cppinst.MemoryFree()

    def StreamInit(self, ns):
        self._cppinst.StreamInit(ns)

    def SetTransposeConjugate(self, transpose=False, conjugate=False):
        self.transpose = transpose
        self.conjugate = conjugate
        self._cppinst.SetTransposeConjugate(transpose, conjugate)

    def CopyBackResults(self):
        self._cppinst.CopyBackResults()

    def MVM(self, x):
        t0 = _time()
        dtype = self.dtype
        _cppimpl.BatchUpdatexgpu_INT8_cf(x, self._cppinst)
        self._cppinst.SetTransposeConjugate(self.transpose, self.conjugate)
        self._cppinst.TryConjugateXvec()
        kt0 = _time()
        # self._cppinst.MVM()
        if self.transpose:
            self._cppinst.Phase1Transpose()
            self._cppinst.Phase2Transpose()
            self._cppinst.Phase3Transpose()
        else:
            self._cppinst.Phase1()
            self._cppinst.Phase2()
            self._cppinst.Phase3()
        kt1 = _time()
        self._cppinst.TryConjugateResults()
        self._cppinst.CopyBackResults()
        _cppimpl.BatchUpdateygpu_INT8_cf(self.y, self._cppinst, self.originM * self.batchsize)
        t1 = _time()
        self.executiontime += t1 - t0
        self.KeyMVM += kt1 - kt0
        return self.y


def Setbandlen(config16, configint8, bandlen):
        _cppimpl.Setbandlen([ x._cppinst for x in config16],[x._cppinst for x in configint8], bandlen)