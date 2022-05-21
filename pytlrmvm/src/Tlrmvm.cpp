//
// Created by Yuxi Hong on 12/03/2022.
//

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <common/Common.hpp>
#include <tlrmvm/Tlrmvm.hpp>
#include "wrapperdef.h"
namespace py = pybind11;


template<typename T>
void addtlrmvmcpu(py::module m){
    string name = "TLRMVMCPU";
    string precision;

    using tlrmvminst = TlrmvmCPU<T>;
    if(typeid(T)==typeid(double)){
        precision = "Double";
    }
    if(typeid(T)==typeid(float)){
        precision = "Float";
    }
    if(typeid(T)==typeid(complex<double>)){
        precision = "ComplexDouble";
    }
    if(typeid(T)==typeid(complex<float>)){
        precision = "ComplexFloat";
    }
    py::class_<tlrmvminst>(m, (name+precision).c_str())
            .def(py::init<TlrmvmConfig>())
            .def("MemoryInit", &tlrmvminst::MemoryInit)
            .def("MemoryFree", &tlrmvminst::MemoryFree)
            .def("SetTransposeConjugate", &tlrmvminst::SetTransposeConjugate)
            .def("MVM", &tlrmvminst::MVM)
            .def("setX", &tlrmvminst::setX)
            .def("CopyToFinalresults", &tlrmvminst::CopyToFinalresults)
            .def_readwrite("config", &tlrmvminst::config)
            .def_readwrite("transpose", &tlrmvminst::transpose)
            .def_readwrite("conjugate", &tlrmvminst::conjugate)
            .def_readwrite("finalresults", &tlrmvminst::finalresults)
            .def_readwrite("p1ptrs", &tlrmvminst::p1ptrs)
            .def_readwrite("p3ptrs", &tlrmvminst::p3ptrs)
            .def_readwrite("p1transptrs", &tlrmvminst::p3transptrs)
            .def_readwrite("p3transptrs", &tlrmvminst::p3transptrs)
            ;
}

template void addtlrmvmcpu<float>(py::module m);
template void addtlrmvmcpu<double>(py::module m);
template void addtlrmvmcpu<complex<float>>(py::module m);
template void addtlrmvmcpu<complex<double>>(py::module m);


template<typename T>
void Updatex(py::array_t<T> inx, TlrmvmCPU<T> *tlrmvminst){
    py::buffer_info buf1 = inx.request();
    T *ptr1 = static_cast<T *>(buf1.ptr);
    int paddingN = tlrmvminst->config.paddingN;
    int originN = tlrmvminst->config.originN;
    CopyData(tlrmvminst->p1ptrs.x, ptr1, originN);
}

template void Updatex(py::array_t<float> inx, TlrmvmCPU<float> *tlrmvminst);
template void Updatex(py::array_t<double> inx, TlrmvmCPU<double> *tlrmvminst);
template void Updatex(py::array_t<complex<float>> inx, TlrmvmCPU<complex<float>> *tlrmvminst);
template void Updatex(py::array_t<complex<double>> inx, TlrmvmCPU<complex<double>> *tlrmvminst);


template<typename T>
void Updateyv(py::array_t<T> outarray, TlrmvmCPU<T> *tlrmvminst, size_t vectorlength){
    py::buffer_info buf1 = outarray.request();
    T *ptr1 = static_cast<T *>(buf1.ptr);
    memcpy(ptr1, tlrmvminst->p1ptrs.y, sizeof(T) * vectorlength);
}

template void Updateyv<float>(py::array_t<float>, TlrmvmCPU<float>*, size_t);
template void Updateyv<double>(py::array_t<double>, TlrmvmCPU<double>*, size_t);
template void Updateyv<complex<float>>(py::array_t<complex<float>>, TlrmvmCPU<complex<float>>*, size_t);
template void Updateyv<complex<double>>(py::array_t<complex<double>>, TlrmvmCPU<complex<double>>*, size_t);


template<typename T>
void Updateyu(py::array_t<T> outarray, TlrmvmCPU<T> *tlrmvminst, size_t vectorlength){
    py::buffer_info buf1 = outarray.request();
    T *ptr1 = static_cast<T *>(buf1.ptr);
    memcpy(ptr1, tlrmvminst->p3ptrs.x, sizeof(T) * vectorlength);
}


template void Updateyu<float>(py::array_t<float>, TlrmvmCPU<float>*, size_t);
template void Updateyu<double>(py::array_t<double>, TlrmvmCPU<double>*, size_t);
template void Updateyu<complex<float>>(py::array_t<complex<float>>, TlrmvmCPU<complex<float>>*, size_t);
template void Updateyu<complex<double>>(py::array_t<complex<double>>, TlrmvmCPU<complex<double>>*, size_t);

template<typename T>
void Updatey(py::array_t<T> outarray, TlrmvmCPU<T> *tlrmvminst, size_t vectorlength){
    py::buffer_info buf1 = outarray.request();
    T *ptr1 = static_cast<T *>(buf1.ptr);
    memcpy(ptr1, tlrmvminst->finalresults, sizeof(T) * vectorlength);
}

template void Updatey<float>(py::array_t<float>, TlrmvmCPU<float>*, size_t);
template void Updatey<double>(py::array_t<double>, TlrmvmCPU<double>*, size_t);
template void Updatey<complex<float>>(py::array_t<complex<float>>, TlrmvmCPU<complex<float>>*, size_t);
template void Updatey<complex<double>>(py::array_t<complex<double>>, TlrmvmCPU<complex<double>>*, size_t);

template<typename HostType, typename DeviceType>
void addtlrmvmgpu(py::module m){
    string name = "TLRMVMGPU";
    string precision;
    using tlrmvminst = GPUinst<HostType, DeviceType>;
    if(typeid(HostType)==typeid(double)){
        precision = "Double";
    }
    if(typeid(HostType)==typeid(float)){
        precision = "Float";
    }
    if(typeid(HostType)==typeid(complex<double>)){
        precision = "ComplexDouble";
    }
    if(typeid(HostType)==typeid(complex<float>)){
        precision = "ComplexFloat";
    }

    py::class_<tlrmvminst>(m, (name+precision).c_str())
            .def(py::init<TlrmvmConfig>())
            .def("MemoryInit", &tlrmvminst::MemoryInit)
            .def("MemoryFree", &tlrmvminst::MemoryFree)
            .def("StreamInit", &tlrmvminst::StreamInit)
            .def("StreamDestroy", &tlrmvminst::StreamDestroy)
            .def("SetTransposeConjugate", &tlrmvminst::SetTransposeConjugate)
            .def("TryConjugateXvec", &tlrmvminst::TryConjugateXvec)
            .def("TryConjugateResults", &tlrmvminst::TryConjugateResults)
            .def("MVM", &tlrmvminst::MVM)
            .def("MVMGraph", &tlrmvminst::MVMGraph)
            .def("setX", &tlrmvminst::setX)
            .def("CopyBackResults", &tlrmvminst::CopyBackResults)
            .def_readwrite("config", &tlrmvminst::config)
            .def_readwrite("transpose", &tlrmvminst::transpose)
            .def_readwrite("conjugate", &tlrmvminst::conjugate)
            .def_readwrite("tlrmvmcpu", &tlrmvminst::tlrmvmcpu)
            ;
}

template void addtlrmvmgpu<float,float>(py::module m);
template void addtlrmvmgpu<double,double>(py::module m);
template void addtlrmvmgpu<complex<float>,GPUcomplex>(py::module m);
template void addtlrmvmgpu<complex<double>,GPUDoublecomplex>(py::module m);

// for check correctness
template<typename Hosttype, typename Devicetype>
void Updateyv(py::array_t<Hosttype> outarray, GPUinst<Hosttype,Devicetype> *tlrmvminst, size_t vectorlength){
    py::buffer_info buf1 = outarray.request();
    Hosttype *ptr1 = static_cast<Hosttype *>(buf1.ptr);
    memcpy(ptr1, tlrmvminst->tlrmvmcpu->p1ptrs.y, sizeof(Hosttype) * vectorlength);
}
template void Updateyv<float,float>(py::array_t<float>, GPUinst<float,float>*, size_t);
template void Updateyv<double,double>(py::array_t<double>, GPUinst<double,double>*, size_t);
template void Updateyv<complex<float>,GPUcomplex>
        (py::array_t<complex<float>>, GPUinst<complex<float>,GPUcomplex>*, size_t);
template void Updateyv<complex<double>,GPUDoublecomplex>
        (py::array_t<complex<double>>, GPUinst<complex<double>,GPUDoublecomplex>*, size_t);


// for check correctness
template<typename Hosttype, typename Devicetype>
void Updateyu(py::array_t<Hosttype> outarray, GPUinst<Hosttype,Devicetype> *tlrmvminst, size_t vectorlength){
    py::buffer_info buf1 = outarray.request();
    Hosttype *ptr1 = static_cast<Hosttype *>(buf1.ptr);
    memcpy(ptr1, tlrmvminst->tlrmvmcpu->p3ptrs.x, sizeof(Hosttype) * vectorlength);
}

template void Updateyu<float,float>(py::array_t<float>, GPUinst<float,float>*, size_t);
template void Updateyu<double,double>(py::array_t<double>, GPUinst<double,double>*, size_t);
template void Updateyu<complex<float>,GPUcomplex>
        (py::array_t<complex<float>>, GPUinst<complex<float>,GPUcomplex>*, size_t);
template void Updateyu<complex<double>,GPUDoublecomplex>
        (py::array_t<complex<double>>, GPUinst<complex<double>,GPUDoublecomplex>*, size_t);


// for return results
template<typename Hosttype, typename Devicetype>
void Updatey(py::array_t<Hosttype> outarray, GPUinst<Hosttype,Devicetype> *tlrmvminst, size_t vectorlength){
    py::buffer_info buf1 = outarray.request();
    Hosttype *ptr1 = static_cast<Hosttype *>(buf1.ptr);
    memcpy(ptr1, tlrmvminst->tlrmvmcpu->finalresults, sizeof(Hosttype) * vectorlength);
}

template void Updatey<float,float>(py::array_t<float>, GPUinst<float,float>*, size_t);
template void Updatey<double,double>(py::array_t<double>, GPUinst<double,double>*, size_t);
template void Updatey<complex<float>,GPUcomplex>
        (py::array_t<complex<float>>, GPUinst<complex<float>,GPUcomplex>*, size_t);
template void Updatey<complex<double>,GPUDoublecomplex>
        (py::array_t<complex<double>>, GPUinst<complex<double>,GPUDoublecomplex>*, size_t);


// for updating x
template<typename Hosttype, typename Devicetype>
void Updatex(py::array_t<Hosttype> inx, GPUinst<Hosttype,Devicetype> *tlrmvminst){
    py::buffer_info buf1 = inx.request();
    Hosttype *ptr1 = static_cast<Hosttype *>(buf1.ptr);
    int paddingN = tlrmvminst->config.paddingN;
    int originN = tlrmvminst->config.originN;
    tlrmvminst->setX(ptr1, originN);
}

template void Updatex<float,float>(py::array_t<float>, GPUinst<float,float>*);
template void Updatex<double,double>(py::array_t<double>, GPUinst<double,double>*);
template void Updatex<complex<float>,GPUcomplex>
        (py::array_t<complex<float>>, GPUinst<complex<float>,GPUcomplex>*);
template void Updatex<complex<double>,GPUDoublecomplex>
        (py::array_t<complex<double>>, GPUinst<complex<double>,GPUDoublecomplex>*);

