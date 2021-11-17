#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <tlrmvm/Tlrmvm.h>
using namespace tlrmvm;
namespace py = pybind11;


template<typename T>
void add_tlrmvmcpu(py::module m){
    string name = "Tlrmvm";
    string prec;

    using tlrmvminst = TlrmvmCPU<T>;
    if(typeid(T)==typeid(double)){
        prec = "_double";
    }
    if(typeid(T)==typeid(float)){
        prec = "_float";
    }
    if(typeid(T)==typeid(complex<double>)){
        prec = "_complexdouble";
    }
    if(typeid(T)==typeid(complex<float>)){
        prec = "_complexfloat";
    }

    py::class_<tlrmvminst>(m, (name+prec).c_str())
    .def(py::init<TlrmvmConfig>())
    .def("InitData", &tlrmvminst::InitData)
    .def("FreeData", &tlrmvminst::FreeData)
    .def("MemoryInit", &tlrmvminst::MemoryInit)
    .def("MemoryFree", &tlrmvminst::MemoryFree)
    .def("MVM", &tlrmvminst::MVM)
    .def("Phase1", &tlrmvminst::Phase1)
    .def("Phase1GetMembuffer", &tlrmvminst::Phase1GetMembuffer)
    .def("AllocatePhase1Buffer", &tlrmvminst::AllocatePhase1Buffer)
    .def("Phase1CopyData", &tlrmvminst::Phase1CopyData)
    .def("Phase2", &tlrmvminst::Phase2)
    .def("Phase2Prepare", &tlrmvminst::Phase2Prepare)
    .def("Phase3", &tlrmvminst::Phase3)
    .def("AllocatePhase3Buffer", &tlrmvminst::AllocatePhase3Buffer)
    .def("Phase3CopyData", &tlrmvminst::Phase3CopyData)
    .def_readwrite("tlrmvmconfig", &tlrmvminst::tlrmvmconfig)
    .def_readwrite("originM", &tlrmvminst::originM)
    .def_readwrite("originN", &tlrmvminst::originN)
    .def_readwrite("paddingM", &tlrmvminst::paddingM)
    .def_readwrite("paddingN", &tlrmvminst::paddingN)
    .def_readwrite("Mtg", &tlrmvminst::Mtg)
    .def_readwrite("Ntg", &tlrmvminst::Ntg)
    .def_readwrite("nb", &tlrmvminst::nb)
    .def_readwrite("granksum", &tlrmvminst::granksum)
    ;

}

template<typename T>
void add_Matrix(py::module m){

}

// interface to accept maskmat using numpy
template<typename T>
Matrix<T> Convertnpy2Matrix(py::array_t<T> npyin){
    py::buffer_info buf1 = npyin.request();
    T *ptr1 = static_cast<T *>(buf1.ptr);
    return Matrix<T>(ptr1, buf1.shape[0], buf1.shape[1]);
}
template Matrix<float> Convertnpy2Matrix(py::array_t<float>);
template Matrix<double> Convertnpy2Matrix(py::array_t<double>);
template Matrix<complex<float>> Convertnpy2Matrix(py::array_t<complex<float>>);
template Matrix<complex<double>> Convertnpy2Matrix(py::array_t<complex<double>>);
template Matrix<int> Convertnpy2Matrix(py::array_t<int>);


void SetMaskmat(py::array_t<int> &maskmat, TlrmvmConfig &tlrmvmconfig){
    tlrmvmconfig.Maskmat = Convertnpy2Matrix<int>(maskmat);
}

template<typename T>
void UpdateAuAv(py::array_t<T> outAu,py::array_t<T> outAv, TlrmvmCPU<T> *tlrmvminst){
    py::buffer_info buf1 = outAu.request();
    T *ptr1 = static_cast<T *>(buf1.ptr);
    py::buffer_info buf2 = outAv.request();
    T *ptr2 = static_cast<T *>(buf2.ptr);
    CopyData(ptr1, tlrmvminst->h_Au, tlrmvminst->granksum * tlrmvminst->nb);
    CopyData(ptr2, tlrmvminst->h_Av, tlrmvminst->granksum * tlrmvminst->nb);
}

template void UpdateAuAv<float>(py::array_t<float> outAu,py::array_t<float> outAv, TlrmvmCPU<float>*);
template void UpdateAuAv<double>(py::array_t<double> outAu,py::array_t<double> outAv, TlrmvmCPU<double>*);
template void UpdateAuAv<complex<float>>
(py::array_t<complex<float>> outAu,py::array_t<complex<float>> outAv, TlrmvmCPU<complex<float>>*);
template void UpdateAuAv<complex<double>>
(py::array_t<complex<double>> outAu,py::array_t<complex<double>> outAv, TlrmvmCPU<complex<double>>*);


template<typename T>
void Updatex(py::array_t<T> inx, TlrmvmCPU<T> *tlrmvminst){
    py::buffer_info buf1 = inx.request();
    T *ptr1 = static_cast<T *>(buf1.ptr);
    memset(tlrmvminst->h_x, 0, sizeof(T) * tlrmvminst->paddingN);
    CopyData(tlrmvminst->h_x, ptr1, tlrmvminst->originN);
}

template void Updatex<float>(py::array_t<float> inx, TlrmvmCPU<float> *tlrmvminst);
template void Updatex<double>(py::array_t<double> inx, TlrmvmCPU<double> *tlrmvminst);
template void Updatex<complex<float>>
(py::array_t<complex<float>> inx, TlrmvmCPU<complex<float>> *tlrmvminst);
template void Updatex<complex<double>>
(py::array_t<complex<double>> inx, TlrmvmCPU<complex<double>> *tlrmvminst);



template<typename T>
void Updateyv(py::array_t<T> outarray, TlrmvmCPU<T> *tlrmvminst, size_t vectorlength){
    py::buffer_info buf1 = outarray.request();
    T *ptr1 = static_cast<T *>(buf1.ptr);
    memcpy(ptr1, tlrmvminst->h_yv, sizeof(T) * vectorlength);
}

template void Updateyv<float>(py::array_t<float>, TlrmvmCPU<float>*, size_t);
template void Updateyv<double>(py::array_t<double>, TlrmvmCPU<double>*, size_t);
template void Updateyv<complex<float>>(py::array_t<complex<float>>, TlrmvmCPU<complex<float>>*, size_t);
template void Updateyv<complex<double>>(py::array_t<complex<double>>, TlrmvmCPU<complex<double>>*, size_t);


template<typename T>
void Updateyu(py::array_t<T> outarray, TlrmvmCPU<T> *tlrmvminst, size_t vectorlength){
    py::buffer_info buf1 = outarray.request();
    T *ptr1 = static_cast<T *>(buf1.ptr);
    memcpy(ptr1, tlrmvminst->h_yu, sizeof(T) * vectorlength);
}

template void Updateyu<float>(py::array_t<float>, TlrmvmCPU<float>*, size_t);
template void Updateyu<double>(py::array_t<double>, TlrmvmCPU<double>*, size_t);
template void Updateyu<complex<float>>(py::array_t<complex<float>>, TlrmvmCPU<complex<float>>*, size_t);
template void Updateyu<complex<double>>(py::array_t<complex<double>>, TlrmvmCPU<complex<double>>*, size_t);


template<typename T>
void Updatey(py::array_t<T> outarray, TlrmvmCPU<T> *tlrmvminst, size_t vectorlength){
    py::buffer_info buf1 = outarray.request();
    T *ptr1 = static_cast<T *>(buf1.ptr);
    memcpy(ptr1, tlrmvminst->h_y, sizeof(T) * vectorlength);
}

template void Updatey<float>(py::array_t<float>, TlrmvmCPU<float>*, size_t);
template void Updatey<double>(py::array_t<double>, TlrmvmCPU<double>*, size_t);
template void Updatey<complex<float>>(py::array_t<complex<float>>, TlrmvmCPU<complex<float>>*, size_t);
template void Updatey<complex<double>>(py::array_t<complex<double>>, TlrmvmCPU<complex<double>>*, size_t);



PYBIND11_MODULE(TLRMVMpy, m) {
    py::class_<TlrmvmConfig>(m, "TlrmvmConfigCpp")
    .def(py::init<int,int,int,string,string,string>())
    .def_readwrite("originM",&TlrmvmConfig::originM)
    .def_readwrite("originN",&TlrmvmConfig::originN)
    .def_readwrite("Mtg",&TlrmvmConfig::Mtg)
    .def_readwrite("Ntg",&TlrmvmConfig::Ntg)
    .def_readwrite("nb",&TlrmvmConfig::nb)
    .def_readwrite("datafolder",&TlrmvmConfig::datafolder)
    .def_readwrite("acc",&TlrmvmConfig::acc)
    .def_readwrite("Maskmat",&TlrmvmConfig::Maskmat)
    ;
    
    add_tlrmvmcpu<float>(m);
    add_tlrmvmcpu<double>(m);
    add_tlrmvmcpu<complex<float>>(m);
    add_tlrmvmcpu<complex<double>>(m);  

    m.def("SetMaskmat", &SetMaskmat);

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

    m.def("UpdateAuAv_f",&UpdateAuAv<float>);
    m.def("UpdateAuAv_cf",&UpdateAuAv<complex<float>>);
    m.def("UpdateAuAv_d",&UpdateAuAv<double>);
    m.def("UpdateAuAv_cd",&UpdateAuAv<complex<double>>);
    
    m.def("Updatex_f",&Updatex<float>);
    m.def("Updatex_cf",&Updatex<complex<float>>);
    m.def("Updatex_d",&Updatex<double>);
    m.def("Updatex_cd",&Updatex<complex<double>>);
    
}
