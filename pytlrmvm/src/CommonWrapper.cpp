//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

//
// Created by Yuxi Hong on 11/03/2022.
//
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>
#include <common/Common.hpp>
#include <tlrmvm/Tlrmvm.hpp>

namespace py = pybind11;

template<typename T>
void addCommonWrapper(py::module m){
    string name = "Matrix";
    string precision;

    using matrixinst = Matrix<T>;
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
    py::class_<matrixinst>(m, (name+precision).c_str(),py::buffer_protocol())
            .def(py::init<>())
            .def(py::init<T*, size_t, size_t>())
            .def(py::init<vector<T>, size_t, size_t>())
            .def(py::init<size_t, size_t>())
            .def(py::init<const matrixinst&>())
            .def("Transpose", &matrixinst::Transpose)
            .def("Conjugate", &matrixinst::Conjugate)
            .def("allclose", &matrixinst::allclose)
            .def("ApplyMask", &matrixinst::ApplyMask)
            .def(py::self += py::self)
            .def(py::self -= py::self)
            .def(py::self *= py::self)
            .def_buffer([](matrixinst &m) -> py::buffer_info {
                return py::buffer_info(
                        m.RawPtr(),
                        sizeof(T),
                        py::format_descriptor<T>::format(),
                        2,
                        {m.Row(), m.Col()},
                        {sizeof(T) * m.Col(), sizeof(T)}
                );
            });
}

template void addCommonWrapper<double>(py::module m);
template void addCommonWrapper<float>(py::module m);
template void addCommonWrapper<complex<double>>(py::module m);
template void addCommonWrapper<complex<float>>(py::module m);

