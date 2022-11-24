//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

//
// Created by Yuxi Hong on 25/03/2022.
//

#include "tlrmvmcudautil.hpp"
#include <cuda_runtime_api.h>
#include <cuda.h>

namespace cudatlrmvm{

    SingleGraph::SingleGraph() {}

    void SingleGraph::StreamInit(int streamsize) {
        // single graph creation
        this->streamsize = streamsize;
        CUDACHECK(cudaEventCreate(&event_start));
        CUDACHECK(cudaEventCreate(&event_phase2finish));
        CUDACHECK(cudaEventCreate(&event_phase1finish));
        events = new cudaEvent_t[4*streamsize];
        for(int i=0; i<4*streamsize; i++) CUDACHECK(cudaEventCreate(&events[i]));
        graphCreated = false;
    }

    void SingleGraph::syncallstreams(cudaEvent_t *eventsptr, cudaStream_t *streamptr, int stream_size) {
        for(int streami=1; streami < stream_size; streami++){
            cudaEventRecord(eventsptr[streami], streamptr[streami]);
        }
        for(int streami=1; streami < stream_size; streami++){
            cudaStreamWaitEvent(streamptr[0], eventsptr[streami]);
        }
        cudaEventRecord(eventsptr[0], streamptr[0]);
        for(int streami=1; streami < stream_size; streami++){
            cudaStreamWaitEvent(streamptr[streami], eventsptr[0]);
        }
    }

    void SingleGraph::syncstream0(cudaEvent_t *eventsptr, cudaStream_t *streamptr, int stream_size) {
        for(int streami=1; streami < stream_size; streami++){
            cudaEventRecord(eventsptr[streami], streamptr[streami]);
        }
        for(int streami=1; streami < stream_size; streami++){
            cudaStreamWaitEvent(streamptr[0], eventsptr[streami]);
        }
    }

    void SingleGraph::syncotherstreams(cudaEvent_t event, cudaStream_t * streamptr, int stream_size){
        cudaEventRecord(event, streamptr[0]);
        for(int streami=1; streami<stream_size; streami++){
            cudaStreamWaitEvent(streamptr[streami], event);
        }
    }

    MultiGraph::MultiGraph() {}

    void MultiGraph::StreamInit(int batchsize, int streamsize) {
        this->batchsize = batchsize;
        this->streamsize = streamsize;
        // multi graph creation
        event_start.resize(batchsize);
        event_phase2finish.resize(batchsize);
        graphCreated.resize(batchsize);
        instance.resize(batchsize);
        graph.resize(batchsize);
        events = new cudaEvent_t*[batchsize];
        for(int bi=0; bi<batchsize; bi++){
            CUDACHECK(cudaEventCreate(&event_start[bi]));
            CUDACHECK(cudaEventCreate(&event_phase2finish[bi]));
            events[bi] = new cudaEvent_t[4*streamsize];
            for(int i=0; i<4*streamsize; i++) CUDACHECK(cudaEventCreate(&events[bi][i]));
            graphCreated[bi] = false;
        }
    }

    CUDAI8basesPointers::CUDAI8basesPointers() {}
    CUDAI8XPointers::CUDAI8XPointers(){}

    CBMaxInfo::CBMaxInfo() {maxA=maxx=maxy=maxbatchsize=0;}

    void getcomplexvectormax(complex<float> *hy, size_t xlength){
        double rmax = 0;
        double imax = 0;
        for(int i=0; i<xlength; i++){
            rmax = fmax(rmax, abs(hy[i].real()));
            imax = fmax(imax, abs(hy[i].imag()));
        }
        cout << "rmax " << rmax << ", imax " << imax << endl;
    }

}

