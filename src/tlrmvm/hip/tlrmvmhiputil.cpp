//  @Copyright (c) 2022 King Abdullah University of Science and Technology (KAUST).
//                      All rights reserved.

//
// Created by Yuxi Hong on 08/04/2022.
//

#include "tlrmvmhiputil.hpp"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hipblas.h>

namespace hiptlrmvm{

    SingleGraph::SingleGraph() {}

    void SingleGraph::StreamInit(int streamsize) {
        // single graph creation
        this->streamsize = streamsize;
        HIPCHECK(hipEventCreate(&event_start));
        HIPCHECK(hipEventCreate(&event_phase2finish));
        HIPCHECK(hipEventCreate(&event_phase1finish));
        events = new hipEvent_t[4*streamsize];
        for(int i=0; i<4*streamsize; i++) HIPCHECK(hipEventCreate(&events[i]));
        graphCreated = false;
    }

    void SingleGraph::syncallstreams(hipEvent_t *eventsptr, hipStream_t *streamptr, int stream_size) {
        for(int streami=1; streami < stream_size; streami++){
            hipEventRecord(eventsptr[streami], streamptr[streami]);
        }
        for(int streami=1; streami < stream_size; streami++){
            hipStreamWaitEvent(streamptr[0], eventsptr[streami],0);
        }
        hipEventRecord(eventsptr[0], streamptr[0]);
        for(int streami=1; streami < stream_size; streami++){
            hipStreamWaitEvent(streamptr[streami], eventsptr[0],0);
        }
    }

    void SingleGraph::syncstream0(hipEvent_t *eventsptr, hipStream_t *streamptr, int stream_size) {
        for(int streami=1; streami < stream_size; streami++){
            hipEventRecord(eventsptr[streami], streamptr[streami]);
        }
        for(int streami=1; streami < stream_size; streami++){
            hipStreamWaitEvent(streamptr[0], eventsptr[streami],0);
        }
    }

    void SingleGraph::syncotherstreams(hipEvent_t event, hipStream_t * streamptr, int stream_size){
        hipEventRecord(event, streamptr[0]);
        for(int streami=1; streami<stream_size; streami++){
            hipStreamWaitEvent(streamptr[streami], event,0);
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
        events = new hipEvent_t*[batchsize];
        for(int bi=0; bi<batchsize; bi++){
            HIPCHECK(hipEventCreate(&event_start[bi]));
            HIPCHECK(hipEventCreate(&event_phase2finish[bi]));
            events[bi] = new hipEvent_t[4*streamsize];
            for(int i=0; i<4*streamsize; i++) HIPCHECK(hipEventCreate(&events[bi][i]));
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

