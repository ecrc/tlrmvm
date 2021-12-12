#include <string>
#include <vector>
#include <chrono>

#include <algorithm>
#include <mpi.h>
// include common component
#include <common/Common.h>
using namespace tlrmat;

// include tlrmvm component
#include <tlrmvm/Tlrmvm.h>
using namespace tlrmvm;
using namespace std;

// This App is used to generate synthetic dataset as input of tlrmvm.
// The rank is constant.

struct Params{
    int originM;
    int originN;
    int nb;
    int constrank;
    string acc;
    string datafolder;
    string problemname;
    string rankfile;
    string Ufile;
    string Vfile;
    string dtype;
    Params(){}
};



template<typename T>
void generate_data(Params & pm){
    char rpath[100]; 
    sprintf(rpath, "%s/%s_Rmat_nb%d_acc%s.bin", pm.datafolder.c_str(), pm.problemname.c_str(), pm.nb, pm.acc.c_str());
    int nb = pm.nb;
    int mtiles = pm.originM / nb;
    if(pm.originM % nb != 0) mtiles++;
    int ntiles = pm.originN / nb;
    if(pm.originN % nb != 0) ntiles++;
    int paddingM = mtiles * nb;
    int paddingN = ntiles * nb;
    int grank = mtiles * ntiles * pm.constrank;
    T* uvec = new T[grank * nb];
    T* vvec = new T[grank * nb];
    T* xvec = new T[paddingN];
    for(int i=0; i<grank*nb; i++) uvec[i] = (T)1.0;
    for(int i=0; i<grank*nb; i++) vvec[i] = (T)1.0;
    for(int i=0; i<paddingN; i++) xvec[i] = (T)1.0;
    int * rvec = new int[mtiles * ntiles];
    for(int i=0; i<mtiles * ntiles; i++) rvec[i] = pm.constrank;
    char upath[100]; 
    sprintf(upath, "%s/%s_Ubases_nb%d_acc%s.bin", pm.datafolder.c_str(), 
    pm.problemname.c_str(), pm.nb, pm.acc.c_str());
    char vpath[100]; 
    sprintf(vpath, "%s/%s_Vbases_nb%d_acc%s.bin", pm.datafolder.c_str(), 
    pm.problemname.c_str(), pm.nb, pm.acc.c_str());
    char xpath[100]; 
    sprintf(xpath, "%s/%s_x.bin", pm.datafolder.c_str(), 
    pm.problemname.c_str());
    auto umat = Matrix<T>(uvec, grank, nb);
    umat.Tofile(upath);
    auto vmat = Matrix<T>(vvec, grank, nb);
    vmat.Tofile(vpath);
    auto rmat = Matrix<int>(rvec, mtiles, ntiles);
    rmat.Tofile(rpath);
    auto xmat = Matrix<T>(xvec, paddingN, 1);
    xmat.Tofile(xpath);
    delete[] rvec;
    delete[] uvec;
    delete[] vvec;
}

int main(int argc, char** argv){
    Params pm = Params();
    vector<double> timestat;
    vector<double> bandstat;
    double bytesprocessed;
    size_t granksum;
    auto argparser = ArgsParser(argc, argv);
    pm.originM = argparser.getint("M");
    pm.originN = argparser.getint("N");
    pm.nb = argparser.getint("nb");
    pm.acc = argparser.getstring("errorthreshold");
    pm.problemname = argparser.getstring("problemname");
    pm.datafolder = argparser.getstring("datafolder");
    pm.constrank = argparser.getint("constrank");
    pm.dtype = argparser.getstring("dtype");
    if(pm.dtype == "float"){
        generate_data<float>(pm);
    }else if(pm.dtype == "complexfloat"){
        generate_data<complex<float>>(pm);
    }
    return 0;
}   
