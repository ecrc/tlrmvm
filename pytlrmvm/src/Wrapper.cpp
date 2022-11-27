//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <common/Common.hpp>
#include <tlrmvm/Tlrmvm.hpp>
#include "wrapperdef.h"
namespace py = pybind11;

// interface to accept maskmat using numpy
template<typename T>
Matrix<T> Convertnpy2Matrix(py::array_t<T> npyin){
    py::buffer_info buf1 = npyin.request();
    T *ptr1 = static_cast<T *>(buf1.ptr);
    if(buf1.shape[1] == 0){
        return Matrix<T>(ptr1, buf1.shape[0],1);
    }else{
        return Matrix<T>(ptr1, buf1.shape[0], buf1.shape[1]);
    }
}

void SetMaskmat(py::array_t<int> &maskmat, TlrmvmConfig &tlrmvmconfig){
    auto maskmatc = Convertnpy2Matrix<int>(maskmat);
    tlrmvmconfig.UpdateMaskmat(maskmatc);
}


void Setbandlen(vector<TlrmvmConfig> &configfp16, vector<TlrmvmConfig> &configint8, int bandlength){
    auto & config0 = configfp16[0];
    Matrix<int> maskmat1(config0.Mtg, config0.Ntg);
    Matrix<int> maskmat2(config0.Mtg, config0.Ntg);
    maskmat1.Fill(0);
    maskmat2.Fill(0);
    for(int i=0; i<config0.Mtg; i++){
        for(int j=0; j<config0.Ntg; j++){
            // bandlength = 0, full int8
            // bandlength = 39, full fp16
            if (abs(i-j) < bandlength)
                maskmat1.SetElem(i,j,1);
            if (abs(i-j) >= bandlength)
                maskmat2.SetElem(i,j,1);
        }
    }
    for(auto & config : configfp16) config.UpdateMaskmat(maskmat1);
    for(auto & config : configint8) config.UpdateMaskmat(maskmat2);
}

PYBIND11_MODULE(TLRMVMpy, m) {
    py::class_<TlrmvmConfig>(m, "TLRMVMConfig")
            .def(py::init<int,int,int,string,string,string>())
            .def("UpdateMaskmat", &TlrmvmConfig::UpdateMaskmat)
            .def("PrintMaskmat", &TlrmvmConfig::PrintMaskmat)
            .def_readwrite("originM",&TlrmvmConfig::originM)
            .def_readwrite("originN",&TlrmvmConfig::originN)
            .def_readwrite("Mtg",&TlrmvmConfig::Mtg)
            .def_readwrite("Ntg",&TlrmvmConfig::Ntg)
            .def_readwrite("nb",&TlrmvmConfig::nb)
            .def_readwrite("datafolder",&TlrmvmConfig::datafolder)
            .def_readwrite("granksum",&TlrmvmConfig::granksum)
            .def_readwrite("paddingN",&TlrmvmConfig::paddingN)
            .def_readwrite("paddingM",&TlrmvmConfig::paddingM)
            .def_readwrite("acc",&TlrmvmConfig::acc)
            .def_readwrite("Maskmat",&TlrmvmConfig::Maskmat)
    ;

    addtlrmvmcpu<float>(m);
    addtlrmvmcpu<double>(m);
    addtlrmvmcpu<complex<float>>(m);
    addtlrmvmcpu<complex<double>>(m);
#ifdef USE_HIP
    addtlrmvmgpu<float,float>(m);
    addtlrmvmgpu<double,double>(m);
    addtlrmvmgpu<complex<float>,hipComplex>(m);
    addtlrmvmgpu<complex<double>,hipDoubleComplex>(m);
#endif
#ifdef USE_CUDA
    addtlrmvmgpu<float,float>(m);
    addtlrmvmgpu<double,double>(m);
    addtlrmvmgpu<complex<float>,cuComplex>(m);
    addtlrmvmgpu<complex<double>,cuDoubleComplex>(m);
#endif


    m.def("SetMaskmat", &SetMaskmat);

    // cpu update
    m.def("Updateyv_f",&Updateyv<float>);
    m.def("Updateyv_cf",&Updateyv<complex<float>>);
    m.def("Updateyv_d",&Updateyv<double>);
    m.def("Updateyv_cd",&Updateyv<complex<double>>);

    m.def("Updateyu_f",&Updateyu<float>);
    m.def("Updateyu_cf",&Updateyu<complex<float>>);
    m.def("Updateyu_d",&Updateyu<double>);
    m.def("Updateyu_cd",&Updateyu<complex<double>>);

    m.def("Updatey_f",&Updatey<float>);
    m.def("Updatey_cf",&Updatey<complex<float>>);
    m.def("Updatey_d",&Updatey<double>);
    m.def("Updatey_cd",&Updatey<complex<double>>);

    m.def("Updatex_f",&Updatex<float>);
    m.def("Updatex_cf",&Updatex<complex<float>>);
    m.def("Updatex_d",&Updatex<double>);
    m.def("Updatex_cd",&Updatex<complex<double>>);

    // GPU update
    m.def("Updateyugpu_f",&Updateyu<float,float>);
    m.def("Updateyugpu_cf",&Updateyu<complex<float>,GPUcomplex>);
    m.def("Updateyugpu_d",&Updateyu<double,double>);
    m.def("Updateyugpu_cd",&Updateyu<complex<double>,GPUDoublecomplex>);

    m.def("Updateyvgpu_f",&Updateyv<float,float>);
    m.def("Updateyvgpu_cf",&Updateyv<complex<float>,GPUcomplex>);
    m.def("Updateyvgpu_d",&Updateyv<double,double>);
    m.def("Updateyvgpu_cd",&Updateyv<complex<double>,GPUDoublecomplex>);

    m.def("Updateygpu_f",&Updatey<float,float>);
    m.def("Updateygpu_cf",&Updatey<complex<float>,GPUcomplex>);
    m.def("Updateygpu_d",&Updatey<double,double>);
    m.def("Updateygpu_cd",&Updatey<complex<double>,GPUDoublecomplex>);

    m.def("Updatexgpu_f",&Updatex<float,float>);
    m.def("Updatexgpu_cf",&Updatex<complex<float>,GPUcomplex>);
    m.def("Updatexgpu_d",&Updatex<double,double>);
    m.def("Updatexgpu_cd",&Updatex<complex<double>,GPUDoublecomplex>);

    addCommonWrapper<float>(m);
    addCommonWrapper<double>(m);
    addCommonWrapper<complex<float>>(m);
    addCommonWrapper<complex<double>>(m);
    addbatchtlrmvmgpu<float,float>(m);
    addbatchtlrmvmgpu<double,double>(m);
    addbatchtlrmvmgpu<complex<float>,GPUcomplex>(m);
    addbatchtlrmvmgpu<complex<double>,GPUDoublecomplex>(m);

    m.def("BatchUpdateygpu_f",&BatchUpdatey<float,float>);
    m.def("BatchUpdateygpu_cf",&BatchUpdatey<complex<float>,GPUcomplex>);
    m.def("BatchUpdateygpu_d",&BatchUpdatey<double,double>);
    m.def("BatchUpdateygpu_cd",&BatchUpdatey<complex<double>,GPUDoublecomplex>);

    m.def("BatchUpdatexgpu_f",&BatchUpdatex<float,float>);
    m.def("BatchUpdatexgpu_cf",&BatchUpdatex<complex<float>,GPUcomplex>);
    m.def("BatchUpdatexgpu_d",&BatchUpdatex<double,double>);
    m.def("BatchUpdatexgpu_cd",&BatchUpdatex<complex<double>,GPUDoublecomplex>);

    addbatchtlrmvmgpufp16(m);
    m.def("BatchUpdatexgpu_FP16_cf", &BatchUpdatex_FP16);
    m.def("BatchUpdateygpu_FP16_cf", &BatchUpdatey_FP16);
    addbatchtlrmvmgpubf16(m);
    m.def("BatchUpdatexgpu_BF16_cf", &BatchUpdatex_BF16);
    m.def("BatchUpdateygpu_BF16_cf", &BatchUpdatey_BF16);

    addbatchtlrmvmgpuint8(m);

    m.def("BatchUpdatexgpu_INT8_cf", &BatchUpdatex_INT8);
    m.def("BatchUpdateygpu_INT8_cf", &BatchUpdatey_INT8);


    m.def("SetMaskmat", &SetMaskmat);
}
