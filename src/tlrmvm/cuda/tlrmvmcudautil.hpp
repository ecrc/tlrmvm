#pragma once

#include <vector>
using std::vector;

#include "../cpu/TlrmvmCPU.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace cudatlrmvm {

    struct SingleGraph{
        SingleGraph();
        void StreamInit(int streamsize);
        int streamsize;
        cudaGraph_t graph;
        bool graphCreated;
        cudaGraphExec_t instance;
        cudaEvent_t* events;
        cudaEvent_t event_start;
        cudaEvent_t event_phase1finish;
        cudaEvent_t event_phase2finish;
        void syncallstreams(cudaEvent_t * events, cudaStream_t * stream,int streamsize);
        void syncstream0(cudaEvent_t * events, cudaStream_t * stream,int streamsize);
        void syncotherstreams(cudaEvent_t event, cudaStream_t * stream,int streamsize);
    };

    struct MultiGraph{
        MultiGraph();
        void StreamInit(int batchsize, int streamsize);
        int batchsize;
        int streamsize;
        vector<cudaGraph_t> graph;
        vector<bool> graphCreated;
        vector<cudaGraphExec_t> instance;
        cudaEvent_t* *events;
        vector<cudaEvent_t> event_start;
        vector<cudaEvent_t> event_phase2finish;
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

        cuInt8Complex * Abuffer; // real data buffer
        vector<cuComplex> maxA;
        cuComplex * maxA_device; // used to scale up to fp16
        vector<size_t> Aelems; // each gemv A elems
        vector<size_t> Aelemsoffset; // each gemv A elems, prefix
        size_t * Aelemsoffset_device; // used to scale up to fp16
        cuHalfComplex * ybuffer; // y buffer, alway a half buffer

        vector<size_t> xelems; // each gemv x elems
        vector<size_t> xelemsoffset; // each gemv x elems, prefix

        vector<size_t> yelems; // each gemv y elems
        vector<size_t> yelemsoffset; // each gemv y elems, prefix

    };

    struct CUDAI8XPointers{
        CUDAI8XPointers();
        cuInt8Complex * xbuffer;
        vector<cuComplex> maxx;
        cuComplex * maxx_device; // used to scale up to fp16
        vector<size_t> xelems; // each gemv x elems
        size_t* xelems_device; // each gemv x elems
        vector<size_t> xelemsoffset; // each gemv x elems, prefix
        size_t* xelemsoffset_device; // each gemv x elems, prefix
        cuComplex *p3xreductionbuffer_device;
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

