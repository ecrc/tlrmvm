from os import error
from .TLRMVMpy import *
import numpy as np

class TlrmvmConfig:
    def __init__(self, m, n, nb, rankfile, datafolder, error_threshold, datasetname) -> None:
        self._cppinst = TlrmvmConfigCpp(m, n, nb, 
        rankfile, datafolder, error_threshold, datasetname)
        self.Mtg = self._cppinst.Mtg
        self.Ntg = self._cppinst.Ntg
        self.m = m
        self.n = n 
        self.nb = nb 
        self.rankfile = rankfile 
        self.datafolder =  datafolder 
        self.error_threshold = error_threshold
        self.datasetname = datasetname
    
    def setmaskmat(self, maskmat):
        SetMaskmat(maskmat, self._cppinst)
    

class Tlrmvm:
    def __init__(self, tlrmvmconfig, dtype) -> None:
        if dtype == np.float32:
            self._cppinst = Tlrmvm_float(tlrmvmconfig._cppinst)
        elif dtype == np.csingle:
            self._cppinst = Tlrmvm_complexfloat(tlrmvmconfig._cppinst)
        elif dtype == np.float64:
            self._cppinst = Tlrmvm_double(tlrmvmconfig._cppinst)
        elif dtype == np.cdouble:
            self._cppinst = Tlrmvm_complexdouble(tlrmvmconfig._cppinst)
        else:
            print("unsupported datatype")
            exit()
        self.dtype = dtype 
        self.granksum = self._cppinst.granksum
        self.originM = self._cppinst.originM
        self.originN = self._cppinst.originN
        self.paddingM = self._cppinst.paddingM
        self.paddingN = self._cppinst.paddingN
        self.nb = tlrmvmconfig.nb

        self.Av = np.zeros(self.granksum * self.nb, dtype=dtype)
        self.Au = np.zeros(self.granksum * self.nb, dtype=dtype)
        self.x = np.zeros(self.paddingM, dtype=dtype)
        self.yu = np.zeros(self.granksum, dtype=dtype)
        self.yv = np.zeros(self.granksum, dtype=dtype)
        self.y = np.zeros(self.originM, dtype=dtype)

    def MemoryInit(self):
        dtype = self.dtype
        self._cppinst.MemoryInit()
        if dtype == np.float32:
            UpdateAuAv_f(self.Au, self.Av, self._cppinst)
        elif dtype == np.csingle:
            UpdateAuAv_cf(self.Au, self.Av, self._cppinst)
        elif dtype == np.float64:
            UpdateAuAv_d(self.Au, self.Av, self._cppinst)
        elif dtype == np.cdouble:
            UpdateAuAv_cd(self.Au, self.Av, self._cppinst)

    def MemoryFree(self):
        self._cppinst.MemoryFree()

    def MVM(self, x):
        dtype = self.dtype

        if dtype == np.float32:
            Updatex_f(x, self._cppinst)
        elif dtype == np.csingle:
            Updatex_cf(x, self._cppinst)
        elif dtype == np.float64:
            Updatex_d(x, self._cppinst)
        elif dtype == np.cdouble:
            Updatex_cd(x, self._cppinst)

        self._cppinst.MVM()
        if dtype == np.float32:
            Updateyv_f(self.yv, self._cppinst, self.granksum)
            Updateyu_f(self.yu, self._cppinst, self.granksum)
            Updatey_f(self.y, self._cppinst, self.originM)
        elif dtype == np.csingle:
            Updateyv_cf(self.yv, self._cppinst, self.granksum)
            Updateyu_cf(self.yu, self._cppinst, self.granksum)
            Updatey_cf(self.y, self._cppinst, self.originM)
        elif dtype == np.float64:
            Updateyv_d(self.yv, self._cppinst, self.granksum)
            Updateyu_d(self.yu, self._cppinst, self.granksum)
            Updatey_d(self.y, self._cppinst, self.originM)
        elif dtype == np.cdouble:
            Updateyv_cd(self.yv, self._cppinst, self.granksum)
            Updateyu_cd(self.yu, self._cppinst, self.granksum)
            Updatey_cd(self.y, self._cppinst, self.originM)

    def Phase1(self):
        dtype = self.dtype
        self._cppinst.Phase1()
        if dtype == np.float32:
            Updateyv_f(self.yv, self._cppinst, self.granksum)
        elif dtype == np.csingle:
            Updateyv_cf(self.yv, self._cppinst, self.granksum)
        elif dtype == np.float64:
            Updateyv_d(self.yv, self._cppinst, self.granksum)
        elif dtype == np.cdouble:
            Updateyv_cd(self.yv, self._cppinst, self.granksum)
        
    def Phase2(self):
        dtype = self.dtype
        self._cppinst.Phase2()
        if dtype == np.float32:
            Updateyu_f(self.yu, self._cppinst, self.granksum)
            
        elif dtype == np.csingle:
            Updateyu_cf(self.yu, self._cppinst, self.granksum)
            
        elif dtype == np.float64:
            Updateyu_d(self.yu, self._cppinst, self.granksum)
            
        elif dtype == np.cdouble:
            Updateyu_cd(self.yu, self._cppinst, self.granksum)
            

    def Phase3(self):
        dtype = self.dtype
        self._cppinst.Phase3()
        if dtype == np.float32:
            Updatey_f(self.y, self._cppinst, self.originM)
        elif dtype == np.csingle:
            Updatey_cf(self.y, self._cppinst, self.originM)
        elif dtype == np.float64:
            Updatey_d(self.y, self._cppinst, self.originM)
        elif dtype == np.cdouble:            
            Updatey_cd(self.y, self._cppinst, self.originM)


