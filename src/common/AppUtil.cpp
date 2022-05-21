#include <cassert>

#include "AppUtil.hpp"
#include "cpu/Util.hpp"

template<typename T>
void ReadBinary(string absfilepath, T * outbuffer, size_t length){
    FILE *f = fopen(absfilepath.c_str(), "rb");
    int ret = fread(outbuffer, sizeof(T), length, f); 
    assert(ret == length);
    fclose(f);
}

template void ReadBinary<int>(string, int*, size_t);
template void ReadBinary<float>(string, float*, size_t);
template void ReadBinary<double>(string, double*, size_t);


void ReadAstronomyBinary(string prefix, int ** outbuffer, size_t length, string acc, int nb, string id){
    char postfix[200];
    sprintf(postfix, "/R_id%s_nb%d_acc%s.bin", id.c_str(), nb, acc.c_str());
    string pstr(postfix);
    int * retptr;
    retptr = new int[length];
    ReadBinary(prefix + postfix, retptr, length);
    outbuffer[0] = retptr;
}
void ReadAstronomyBinary(string prefix, float ** outbuffer, size_t length, string acc, int nb, string id){
    char postfix[200];
    float *dataptr;
    sprintf(postfix, "_id%s_nb%d_acc%s.bin", id.c_str(), nb, acc.c_str());
    dataptr = new float[length];
    ReadBinary(prefix+postfix, dataptr, length);
    outbuffer[0] = dataptr;
}

void ReadSeismicBinary(string prefix, int ** outbuffer, size_t length, string acc, int nb, int id){
    char postfix[200];
    sprintf(postfix, "R-Mck_freqslice%d_nb%d_acc%s.bin", id, nb, acc.c_str());
    string pstr(postfix);
    int * retptr;
    retptr = new int[length];
    string abspath = PathJoin({prefix, postfix});
    ReadBinary(abspath, retptr, length);
    outbuffer[0] = retptr;
}
void ReadSeismicBinary(string prefix, complex<float> ** outptr, size_t length, 
string acc, int nb, int id){
    char postfix[200];
    vector<string> UVstr = {"_real_Mck_freqslice", "_imag_Mck_freqslice"};
    float *dataptr[4];
    for(int i=0; i<2; i++){
        sprintf(postfix, "%s%d_nb%d_acc%s.bin", UVstr[i].c_str(), id, nb, acc.c_str());
        dataptr[i] = new float[length];
        ReadBinary(prefix+postfix, dataptr[i], length);
    }
    complex<float> * tmpptr = new complex<float>[length];
    for(size_t i=0; i<length; i++){
        tmpptr[i] = complex<float>(dataptr[0][i], dataptr[1][i]);
    }
    outptr[0] = tmpptr;
    for(int i=0; i<2; i++) delete[] dataptr[i];
}

void ReadSeismicBinaryX(string prefix, complex<float> ** outX, size_t length, string acc, int nb, int id){
    char postfix[200];
    float *dataptr[2];
    string xstr = "xvector/fdplus_";
    sprintf(postfix, "%s%d_real.bin", xstr.c_str(), id);
    dataptr[0] = new float[length];
    ReadBinary(PathJoin({prefix, postfix}), dataptr[0], length);
    sprintf(postfix, "%s%d_imag.bin", xstr.c_str(), id);
    dataptr[1] = new float[length];
    ReadBinary(PathJoin({prefix, postfix}), dataptr[1], length);
    complex<float> * Xptr = new complex<float>[length];

    for(size_t i=0; i<length; i++){
        Xptr[i] = complex<float>(dataptr[0][i], dataptr[1][i]);
    }
    outX[0] = Xptr;
    for(int i=0; i<2; i++) delete[] dataptr[i];
}

void RandomX(float * xvector, int length){
    for(int i=0; i<length; i++){
        xvector[i] = (float)rand() / (float)2147483647;
    }
}
void RandomX(complex<float> *xvector, int length){
    for(int i=0; i<length; i++){
//        xvector[i] = complex<float> ( (float)rand() / (float)2147483647 , (float)rand() / (float)2147483647);
        xvector[i] = complex<float> ( 1.0 , 3.0);
    }
}

void RandomX(double * xvector, int length){
    for(int i=0; i<length; i++){
        xvector[i] = (double)rand() / (float)2147483647;
    }
}
void RandomX(complex<double> *xvector, int length){
    for(int i=0; i<length; i++){
        xvector[i] = complex<double> ((double)rand() / (double)2147483647 , (double)rand() / (double)2147483647);
    }
}


