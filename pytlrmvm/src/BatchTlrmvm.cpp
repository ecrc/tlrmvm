//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

//
// Created by Yuxi Hong on 12/03/2022.
//

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <common/Common.hpp>
#include <tlrmvm/Tlrmvm.hpp>

#include "wrapperdef.h"
namespace py = pybind11;

template<typename HostType, typename DeviceType>
void addbatchtlrmvmgpu(py::module m){
    string name = "BatchTLRMVMGPU";
    string precision;
    using tlrmvminst = BatchGPUinst<HostType, DeviceType>;

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
            .def(py::init<vector<TlrmvmConfig>>())
            .def("MemoryInit", &tlrmvminst::MemoryInit)
            .def("MemoryFree", &tlrmvminst::MemoryFree)
            .def("StreamInit", &tlrmvminst::StreamInit)
            .def("SetTransposeConjugate", &tlrmvminst::SetTransposeConjugate)
            .def("TryConjugateXvec", &tlrmvminst::TryConjugateXvec)
            .def("TryConjugateResults", &tlrmvminst::TryConjugateResults)
            .def("MVM", &tlrmvminst::MVM_SingleGraph)
            .def("setX", &tlrmvminst::setX)
            .def("CopyBackResults", &tlrmvminst::CopyBackResults)
            .def_readwrite("transpose", &tlrmvminst::transpose)
            .def_readwrite("conjugate", &tlrmvminst::conjugate)
            .def_readwrite("config_vec", &tlrmvminst::config_vec)
            .def_readwrite("tlrmvmcpu_vec", &tlrmvminst::cpuinstvec)
            ;
}


template void addbatchtlrmvmgpu<float,float>(py::module m);
template void addbatchtlrmvmgpu<double,double>(py::module m);
template void addbatchtlrmvmgpu<complex<float>,GPUcomplex>(py::module m);
template void addbatchtlrmvmgpu<complex<double>,GPUDoublecomplex>(py::module m);


void addbatchtlrmvmgpufp16(py::module m){
    py::class_<BatchGPUinstFP16>(m, "BatchTLRMVMGPUFP16")
            .def(py::init<vector<TlrmvmConfig>>())
            .def("MemoryInit", &BatchGPUinstFP16::MemoryInit)
            .def("MemoryFree", &BatchGPUinstFP16::MemoryFree)
            .def("StreamInit", &BatchGPUinstFP16::StreamInit)
            .def("SetTransposeConjugate", &BatchGPUinstFP16::SetTransposeConjugate)
            .def("TryConjugateXvec", &BatchGPUinstFP16::TryConjugateXvec)
            .def("TryConjugateResults", &BatchGPUinstFP16::TryConjugateResults)
            .def("MVM", &BatchGPUinstFP16::MVM_SingleGraph)
            .def("setX", &BatchGPUinstFP16::setX)
            .def("CopyBackResults", &BatchGPUinstFP16::CopyBackResults)
            .def_readwrite("transpose", &BatchGPUinstFP16::transpose)
            .def_readwrite("conjugate", &BatchGPUinstFP16::conjugate)
            .def_readwrite("config_vec", &BatchGPUinstFP16::config_vec)
            .def_readwrite("tlrmvmcpu_vec", &BatchGPUinstFP16::cpuinstvec)
            ;
}

void addbatchtlrmvmgpubf16(py::module m){
    py::class_<BatchGPUinstBF16>(m, "BatchTLRMVMGPUBF16")
            .def(py::init<vector<TlrmvmConfig>>())
            .def("MemoryInit", &BatchGPUinstBF16::MemoryInit)
            .def("MemoryFree", &BatchGPUinstBF16::MemoryFree)
            .def("StreamInit", &BatchGPUinstBF16::StreamInit)
            .def("SetTransposeConjugate", &BatchGPUinstBF16::SetTransposeConjugate)
            .def("TryConjugateXvec", &BatchGPUinstBF16::TryConjugateXvec)
            .def("TryConjugateResults", &BatchGPUinstBF16::TryConjugateResults)
            .def("MVM", &BatchGPUinstBF16::MVM_SingleGraph)
            .def("setX", &BatchGPUinstBF16::setX)
            .def("CopyBackResults", &BatchGPUinstBF16::CopyBackResults)
            .def_readwrite("transpose", &BatchGPUinstBF16::transpose)
            .def_readwrite("conjugate", &BatchGPUinstBF16::conjugate)
            .def_readwrite("config_vec", &BatchGPUinstBF16::config_vec)
            .def_readwrite("tlrmvmcpu_vec", &BatchGPUinstBF16::cpuinstvec)
            ;
}

void addbatchtlrmvmgpuint8(py::module m){
    py::class_<BatchGPUinstINT8>(m, "BatchTLRMVMGPUINT8")
            .def(py::init<vector<TlrmvmConfig>>())
            .def("MemoryInit", &BatchGPUinstINT8::MemoryInit)
            .def("MemoryFree", &BatchGPUinstINT8::MemoryFree)
            .def("StreamInit", &BatchGPUinstINT8::StreamInit)
            .def("SetTransposeConjugate", &BatchGPUinstINT8::SetTransposeConjugate)
            .def("TryConjugateXvec", &BatchGPUinstINT8::TryConjugateXvec)
            .def("TryConjugateResults", &BatchGPUinstINT8::TryConjugateResults)
            .def("MVM", &BatchGPUinstINT8::MVM_MultiGraph)
            .def("Phase1", &BatchGPUinstINT8::Phase1)
            .def("Phase2", &BatchGPUinstINT8::Phase2)
            .def("Phase3", &BatchGPUinstINT8::Phase3)
            .def("Phase1Transpose", &BatchGPUinstINT8::Phase1Transpose)
            .def("Phase2Transpose", &BatchGPUinstINT8::Phase2Transpose)
            .def("Phase3Transpose", &BatchGPUinstINT8::Phase3Transpose)
            .def("setX", &BatchGPUinstINT8::setX)
            .def("CopyBackResults", &BatchGPUinstINT8::CopyBackResults)
            .def_readwrite("transpose", &BatchGPUinstINT8::transpose)
            .def_readwrite("conjugate", &BatchGPUinstINT8::conjugate)
            .def_readwrite("config_vec", &BatchGPUinstINT8::config_vec)
            .def_readwrite("tlrmvmcpu_vec", &BatchGPUinstINT8::cpuinstvec)
            ;
}

// for updating x
template<typename Hosttype, typename Devicetype>
void BatchUpdatex(py::array_t<Hosttype> inx, BatchGPUinst<Hosttype,Devicetype> *tlrmvminst){
    py::buffer_info buf1 = inx.request();
    Hosttype *ptr1 = static_cast<Hosttype *>(buf1.ptr);
    int totallen = tlrmvminst->config_vec[0].originN * tlrmvminst->config_vec.size();
    tlrmvminst->setX(ptr1, totallen);
}

template void BatchUpdatex<float,float>(py::array_t<float>, BatchGPUinst<float,float>*);
template void BatchUpdatex<double,double>(py::array_t<double>, BatchGPUinst<double,double>*);
template void BatchUpdatex<complex<float>,GPUcomplex>
        (py::array_t<complex<float>>, BatchGPUinst<complex<float>,GPUcomplex>*);
template void BatchUpdatex<complex<double>,GPUDoublecomplex>
        (py::array_t<complex<double>>, BatchGPUinst<complex<double>,GPUDoublecomplex>*);

// for return results
template<typename Hosttype, typename Devicetype>
void BatchUpdatey(py::array_t<Hosttype> outarray, BatchGPUinst<Hosttype,Devicetype> *tlrmvminst, size_t vectorlength){
    py::buffer_info buf1 = outarray.request();
    Hosttype *ptr1 = static_cast<Hosttype *>(buf1.ptr);
    memcpy(ptr1, tlrmvminst->finalresults.data(), sizeof(Hosttype) * vectorlength);
}

template void BatchUpdatey<float,float>(py::array_t<float>, BatchGPUinst<float,float>*, size_t);
template void BatchUpdatey<double,double>(py::array_t<double>, BatchGPUinst<double,double>*, size_t);
template void BatchUpdatey<complex<float>,GPUcomplex>
        (py::array_t<complex<float>>, BatchGPUinst<complex<float>,GPUcomplex>*, size_t);
template void BatchUpdatey<complex<double>,GPUDoublecomplex>
        (py::array_t<complex<double>>, BatchGPUinst<complex<double>,GPUDoublecomplex>*, size_t);

// for updating x
void BatchUpdatex_FP16(py::array_t<complex<float>> inx, BatchGPUinstFP16 *tlrmvminst){
    py::buffer_info buf1 = inx.request();
    complex<float> *ptr1 = static_cast<complex<float> *>(buf1.ptr);
    int totallen = tlrmvminst->config_vec[0].originN * tlrmvminst->config_vec.size();
    tlrmvminst->setX(ptr1, totallen);
}
// for return results
void BatchUpdatey_FP16(py::array_t<complex<float>> outarray, BatchGPUinstFP16 *tlrmvminst, size_t vectorlength){
    py::buffer_info buf1 = outarray.request();
    complex<float> *ptr1 = static_cast<complex<float> *>(buf1.ptr);
    memcpy(ptr1, tlrmvminst->finalresults.data(), sizeof(complex<float>) * vectorlength);
}

// for updating x
void BatchUpdatex_BF16(py::array_t<complex<float>> inx, BatchGPUinstBF16 *tlrmvminst){
    py::buffer_info buf1 = inx.request();
    complex<float> *ptr1 = static_cast<complex<float> *>(buf1.ptr);
    int totallen = tlrmvminst->config_vec[0].originN * tlrmvminst->config_vec.size();
    tlrmvminst->setX(ptr1, totallen);
}
// for return results
void BatchUpdatey_BF16(py::array_t<complex<float>> outarray, BatchGPUinstBF16 *tlrmvminst, size_t vectorlength){
    py::buffer_info buf1 = outarray.request();
    complex<float> *ptr1 = static_cast<complex<float> *>(buf1.ptr);
    memcpy(ptr1, tlrmvminst->finalresults.data(), sizeof(complex<float>) * vectorlength);
}

// for updating x
void BatchUpdatex_INT8(py::array_t<complex<float>> inx, BatchGPUinstINT8 *tlrmvminst){
    py::buffer_info buf1 = inx.request();
    complex<float> *ptr1 = static_cast<complex<float> *>(buf1.ptr);
    int totallen = tlrmvminst->config_vec[0].originN * tlrmvminst->config_vec.size();
    tlrmvminst->setX(ptr1, totallen);
}


// for return results
void BatchUpdatey_INT8(py::array_t<complex<float>> outarray, BatchGPUinstINT8 *tlrmvminst, size_t vectorlength){
    py::buffer_info buf1 = outarray.request();
    complex<float> *ptr1 = static_cast<complex<float> *>(buf1.ptr);
    memcpy(ptr1, tlrmvminst->finalresults.data(), sizeof(complex<float>) * vectorlength);
}
