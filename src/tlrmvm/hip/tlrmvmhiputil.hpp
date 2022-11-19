#pragma once

#include <vector>
using std::vector;

#include "../cpu/TlrmvmCPU.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace hiptlrmvm {

    struct SingleGraph{
        SingleGraph();
        void StreamInit(int streamsize);
        int streamsize;
        hipGraph_t graph;
        bool graphCreated;
        hipGraphExec_t instance;
        hipEvent_t* events;
        hipEvent_t event_start;
        hipEvent_t event_phase1finish;
        hipEvent_t event_phase2finish;
        void syncallstreams(hipEvent_t * events, hipStream_t * stream,int streamsize);
        void syncstream0(hipEvent_t * events, hipStream_t * stream,int streamsize);
        void syncotherstreams(hipEvent_t event, hipStream_t * stream,int streamsize);
    };

    struct MultiGraph{
        MultiGraph();
        void StreamInit(int batchsize, int streamsize);
        int batchsize;
        int streamsize;
        vector<hipGraph_t> graph;
        vector<bool> graphCreated;
        vector<hipGraphExec_t> instance;
        hipEvent_t* *events;
        vector<hipEvent_t> event_start;
        vector<hipEvent_t> event_phase2finish;
    };

    // BatchTlrmvmcudaINT8

    struct CUDAI8basesPointers{
        CUDAI8basesPointers();
        size_t Acnt;
        size_t Xcnt;
        size_t Ycnt;
        vector<size_t> Ms;
        vector<size_t> Ks;
        vector<size_t> Ns;

        hipInt8Complex * Abuffer; // real data buffer
        vector<hipComplex> maxA;
        hipComplex * maxA_device; // used to scale up to fp16
        vector<size_t> Aelems; // each gemv A elems
        vector<size_t> Aelemsoffset; // each gemv A elems, prefix
        size_t * Aelemsoffset_device; // used to scale up to fp16
        hipHalfComplex * ybuffer; // y buffer, alway a half buffer

        vector<size_t> xelems; // each gemv x elems
        vector<size_t> xelemsoffset; // each gemv x elems, prefix

        vector<size_t> yelems; // each gemv y elems
        vector<size_t> yelemsoffset; // each gemv y elems, prefix

    };

    struct CUDAI8XPointers{
        CUDAI8XPointers();
        hipInt8Complex * xbuffer;
        vector<hipComplex> maxx;
        hipComplex * maxx_device; // used to scale up to fp16
        vector<size_t> xelems; // each gemv x elems
        size_t* xelems_device; // each gemv x elems
        vector<size_t> xelemsoffset; // each gemv x elems, prefix
        size_t* xelemsoffset_device; // each gemv x elems, prefix
        hipComplex *p3xreductionbuffer_device;
    };

    struct CBMaxInfo{
        CBMaxInfo();
        size_t maxA;
        size_t maxx;
        size_t maxy;
        size_t maxbatchsize;
    };

    void getcomplexvectormax(complex<float> *hy, size_t xlength);

}

