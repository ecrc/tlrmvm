//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

//
// Created by Yuxi Hong on 12/03/2022.
//

#pragma once

#include <pybind11/pybind11.h>
namespace py = pybind11;
#ifdef USE_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#define GPUinst  hiptlrmvm::Tlrmvmhip
#define BatchGPUinst hiptlrmvm::BatchTlrmvmhip
#define GPUcomplex  hipComplex
#define GPUDoublecomplex  hipDoubleComplex
#endif
#ifdef USE_CUDA
#include <cuComplex.h>
#define GPUinst  cudatlrmvm::Tlrmvmcuda
#define BatchGPUinst cudatlrmvm::BatchTlrmvmcuda
#define BatchGPUinstFP16 cudatlrmvm::BatchTlrmvmcudaFP16<complex<float>, cuHalfComplex>
#define BatchGPUinstBF16 cudatlrmvm::BatchTlrmvmcudaFP16<complex<float>, cubfComplex>
#define BatchGPUinstINT8 cudatlrmvm::BatchTlrmvmcudaINT8<complex<float>, cuInt8Complex>
#define GPUcomplex  cuComplex
#define GPUDoublecomplex  cuDoubleComplex
#endif

template<typename T>
void addCommonWrapper(py::module m);

template<typename T>
void addtlrmvmcpu(py::module m);

template<typename T>
void Updateyv(py::array_t<T> outarray, TlrmvmCPU<T> *tlrmvminst, size_t vectorlength);

template<typename T>
void Updateyu(py::array_t<T> outarray, TlrmvmCPU<T> *tlrmvminst, size_t vectorlength);

template<typename T>
void Updatey(py::array_t<T> outarray, TlrmvmCPU<T> *tlrmvminst, size_t vectorlength);

template<typename T>
void Updatex(py::array_t<T> inx, TlrmvmCPU<T> *tlrmvminst);

template<typename H, typename D>
void addtlrmvmgpu(py::module m);

template<typename Hosttype, typename Devicetype>
void Updateyv(py::array_t<Hosttype> outarray, GPUinst<Hosttype,Devicetype> *tlrmvminst, size_t vectorlength);

template<typename Hosttype, typename Devicetype>
void Updateyu(py::array_t<Hosttype> outarray, GPUinst<Hosttype,Devicetype> *tlrmvminst, size_t vectorlength);

template<typename Hosttype, typename Devicetype>
void Updatey(py::array_t<Hosttype> outarray, GPUinst<Hosttype,Devicetype> *tlrmvminst, size_t vectorlength);

template<typename Hosttype, typename Devicetype>
void Updatex(py::array_t<Hosttype> inx, GPUinst<Hosttype,Devicetype> *tlrmvminst);

template<typename HostType, typename DeviceType>
void addbatchtlrmvmgpu(py::module m);



template<typename Hosttype, typename Devicetype>
void BatchUpdatey(py::array_t<Hosttype> outarray, BatchGPUinst<Hosttype,Devicetype> *tlrmvminst, size_t vectorlength);

template<typename Hosttype, typename Devicetype>
void BatchUpdatex(py::array_t<Hosttype> inx, BatchGPUinst<Hosttype,Devicetype> *tlrmvminst);

void addbatchtlrmvmgpufp16(py::module m);
void addbatchtlrmvmgpubf16(py::module m);

void BatchUpdatex_FP16(py::array_t<complex<float>> inx, BatchGPUinstFP16 *tlrmvminst);
void BatchUpdatey_FP16(py::array_t<complex<float>> outarray, BatchGPUinstFP16 *tlrmvminst, size_t vectorlength);
void BatchUpdatex_BF16(py::array_t<complex<float>> inx, BatchGPUinstBF16 *tlrmvminst);
void BatchUpdatey_BF16(py::array_t<complex<float>> outarray, BatchGPUinstBF16 *tlrmvminst, size_t vectorlength);


void addbatchtlrmvmgpuint8(py::module m);

void BatchUpdatex_INT8(py::array_t<complex<float>> inx, BatchGPUinstINT8 *tlrmvminst);

void BatchUpdatey_INT8(py::array_t<complex<float>> outarray, BatchGPUinstINT8 *tlrmvminst, size_t vectorlength);

void addbatchtlrmvmgpufp16int8(py::module m);
