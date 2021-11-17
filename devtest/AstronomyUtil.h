#ifndef ASTRONOMYUTIL_H
#define ASTRONOMYUTIL_H

#include <string>
#include <vector>
using std::string;
using std::vector;





class AstronomyTestCPUOnly{

    AstronomyTestCPUOnly();
    void AllocatePointers();
    void DestroyPointers();

protected:

    // host pointer
    float *hAv;
    float *hx;
    float *hyv;
    vector<size_t> AvMs;
    vector<size_t> AvKs;
    vector<size_t> AvNs;
    float **hAvbp;
    float **hxbp;
    float **hyvbp;
    size_t Avtotalelems;
    size_t xtotalelems;
    size_t yvtotalelems;
    float *hAu;
    float *hyu;
    float *hy; 
    float **hAubp;
    float **hyubp;
    float **hybp;
    unsigned long int* offsetphase2_h;
    float alpha;
    float beta;


};




#endif // ASTRONOMYUTIL_H



