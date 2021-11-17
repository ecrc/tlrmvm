 #include "common/Common.h"
#include "common/AppUtil.h"

#include "tlrmvm/cpu/TlrmvmCPU.h"

#include "TlrmvmGPU.h"
#include "TlrmvmKernel.cuh"

#include "common/cuda/cublasInterface.h"
#include "common/cuda/Util.h"

#include <cassert>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h> // if you need CUBLAS v2, include before magma.h

using namespace tlrmvm;
using namespace cudatlrmat;

namespace cudatlrmvm
{

void initialphabeta(cuComplex &alpha, cuComplex &beta){
    alpha.x = 1.0;
    alpha.y = 0.0;
    beta.x = 0.0;
    beta.y = 0.0;
}

void initialphabeta(cuDoubleComplex &alpha, cuDoubleComplex &beta){
    alpha.x = 1.0;
    alpha.y = 0.0;
    beta.x = 0.0;
    beta.y = 0.0;
}

void initialphabeta(float &alpha, float &beta){
    alpha = 1.0;
    beta = 0.0;
}

void initialphabeta(double &alpha, double &beta){
    alpha = 1.0;
    beta = 0.0;
}


template<typename HostType, typename DeviceType>
TlrmvmGPU<HostType, DeviceType>::TlrmvmGPU(TlrmvmConfig tlrmvmconfig)
: TlrmvmCPU<HostType>::TlrmvmCPU(tlrmvmconfig){}


template<typename HostType, typename DeviceType>
void TlrmvmGPU<HostType,DeviceType>::setX(HostType * xvector, size_t xlength){
    TlrmvmBase<HostType>::setX(xvector, xlength);
    CopyDataB2HD((HostType*)this->d_x, this->h_x, xlength);
}

template<typename HostType, typename DeviceType>
void TlrmvmGPU<HostType,DeviceType>::StreamInit(int streamsize){
    
    this->streamsize = streamsize;
    streamptr = new cudaStream_t[streamsize];
    cublashandleptr = new cublasHandle_t[streamsize];
    for(int i=0; i<streamsize; i++) cudaStreamCreateWithFlags(&streamptr[i], cudaStreamNonBlocking);
    for(int i=0; i<streamsize; i++) cublasCreate_v2(&cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cublasSetStream_v2(cublashandleptr[i], streamptr[i]);

    CUDACHECK(cudaEventCreate(&event_start));
    CUDACHECK(cudaEventCreate(&event_phase2finish));
    events = new cudaEvent_t[2*streamsize];
    for(int i=0; i<2*streamsize; i++) CUDACHECK(cudaEventCreate(&events[i]));
    // graph
    graphCreated = false;
    initialphabeta(alpha, beta);
}

template<typename HostType, typename DeviceType>
void TlrmvmGPU<HostType,DeviceType>::StreamDestroy(){
    
    for(int i=0; i<streamsize; i++) cublasDestroy(cublashandleptr[i]);
    for(int i=0; i<streamsize; i++) cudaStreamDestroy(streamptr[i]);
    delete[] cublashandleptr;
    delete[] streamptr;
    CUDACHECK(cudaEventDestroy(event_start));
    CUDACHECK(cudaEventDestroy(event_phase2finish));
    for(int i=0; i<2*streamsize; i++) CUDACHECK(cudaEventDestroy(events[i]));
    delete[] events;
    // graph
    if(graphCreated){
        cudaGraphExecDestroy(instance);
        cudaGraphDestroy(graph);
    }
    

}


template<typename HostType, typename DeviceType>
void TlrmvmGPU<HostType,DeviceType>::MemoryInit(){
    InitData();
    Phase1GetMembuffer();
    AllocatePhase1Buffer();
    Phase1CopyData();
    Phase2Prepare();
    Phase3GetMembuffer();
    AllocatePhase3Buffer();
    Phase3CopyData();
    // FreeData();
}

template<typename HostType, typename DeviceType>
void TlrmvmGPU<HostType,DeviceType>::MemoryFree(){
    TlrmvmCPU<HostType>::MemoryFree();
    // GPU free
    FreecuHostMemory(d_Avbp);
    FreecuHostMemory(d_xbp);
    FreecuHostMemory(d_yvbp);
    FreecuHostMemory(d_yvoutbp);

    FreeDeviceMemory(d_Av);
    FreeDeviceMemory(d_x);
    FreeDeviceMemory(d_yv);
    FreeDeviceMemory(d_yvout);

    FreecuHostMemory(d_Aubp);
    FreecuHostMemory(d_yubp);
    FreecuHostMemory(d_ybp);
    FreecuHostMemory(d_youtbp);

    FreeDeviceMemory(d_Au);
    FreeDeviceMemory(d_yu);
    FreeDeviceMemory(d_y);
    FreeDeviceMemory(d_yout);

}


template<typename HostType, typename DeviceType>
void TlrmvmGPU<HostType,DeviceType>::Phase1(){
    cudaDeviceSynchronize();
    for(int i=0; i<this->Ntg; i++){
        if(this->AvMs[i] != 0){
            cublasgemv(cublashandleptr[i%streamsize], CUBLAS_OP_N,
            this->AvMs[i], this->AvKs[i],
            &alpha, (const DeviceType*)d_Avbp[i], this->AvMs[i], 
            (const DeviceType*)d_xbp[i], 1, &beta, 
            d_yvbp[i], 1);
        }
    }
    cudaDeviceSynchronize();
}

template<typename HostType, typename DeviceType>
void TlrmvmGPU<HostType,DeviceType>::Phase1GetMembuffer(){
    TlrmvmCPU<HostType>::Phase1GetMembuffer();
    // GPU
    // Device Memory phase 1
    int batchsize = this->AvMs.size();
    
    GetcuHostMemory(&d_Avbp, batchsize);
    GetcuHostMemory(&d_xbp, batchsize);
    GetcuHostMemory(&d_yvbp, batchsize);
    GetcuHostMemory(&d_yvoutbp, batchsize);

}


template<typename HostType, typename DeviceType>
void TlrmvmGPU<HostType,DeviceType>::AllocatePhase1Buffer(){
    TlrmvmCPU<HostType>::AllocatePhase1Buffer();
    GetDeviceMemory(&d_Av, this->phase1Acnt);
    GetDeviceMemory(&d_x, this->phase1Xcnt);
    GetDeviceMemory(&d_yv, this->phase1Ycnt);
    GetDeviceMemory(&d_yvout, this->phase1Ycnt);
    d_Avbp[0] = d_Av;
    d_yvbp[0] = d_yv;
    d_xbp[0] = d_x;
    d_yvoutbp[0] = d_yvout;
}


template<typename HostType, typename DeviceType>
void TlrmvmGPU<HostType,DeviceType>::Phase1CopyData(){
    TlrmvmCPU<HostType>::Phase1CopyData();
    auto AvMs = this->AvMs;
    auto AvNs = this->AvNs;
    auto AvKs = this->AvKs;
    for(int i=1; i<this->Ntg; i++){
        size_t AvMK = AvMs[i-1] * AvKs[i-1];
        size_t AvKN = AvKs[i-1] * AvNs[i-1];
        size_t AvMN = AvMs[i-1] * AvNs[i-1];
        d_Avbp[i] = d_Avbp[i-1] + AvMK;
        d_xbp[i] = d_xbp[i-1] + AvKN;
        d_yvbp[i] = d_yvbp[i-1] + AvMN;
    }
    CopyDataB2HD((HostType*)d_Av, this->h_Av, this->phase1Acnt);
    CopyDataB2HD((HostType*)d_x, this->h_x, this->phase1Xcnt);
}

template<typename HostType, typename DeviceType>
void TlrmvmGPU<HostType,DeviceType>::Phase2(){
    cudaDeviceSynchronize();
    phase2_nosplit<DeviceType>(d_yv, d_phase2mapping, d_yu, this->granksum, streamptr[0]);
    cudaDeviceSynchronize();
}

template<typename HostType, typename DeviceType>
void TlrmvmGPU<HostType,DeviceType>::Phase2Prepare(){
    TlrmvmCPU<HostType>::Phase2Prepare();
    GetDeviceMemory(&d_phase2mapping, this->h_phase2mapping.size());
    CopyDataB2HD(d_phase2mapping, this->h_phase2mapping.data(), 
    this->h_phase2mapping.size());
}

template<typename HostType, typename DeviceType>
void TlrmvmGPU<HostType,DeviceType>::Phase3(){
    cudaDeviceSynchronize();
    for(int i=0; i<this->Mtg; i++){
        cublasgemv(cublashandleptr[i%streamsize], CUBLAS_OP_N,
        this->AuMs[i], this->AuKs[i],
        &alpha, (const DeviceType*)d_Aubp[i], this->AuMs[i], 
        (const DeviceType*)d_yubp[i], 1, &beta, 
        d_ybp[i], 1);
    }
    cudaDeviceSynchronize();
}

template<typename HostType, typename DeviceType>
void TlrmvmGPU<HostType,DeviceType>::Phase3GetMembuffer(){
    TlrmvmCPU<HostType>::Phase3GetMembuffer();
    int batchsize = this->AuMs.size();
    GetcuHostMemory(&d_Aubp, batchsize);
    GetcuHostMemory(&d_yubp, batchsize);
    GetcuHostMemory(&d_ybp, batchsize);
    GetcuHostMemory(&d_youtbp, batchsize);
}

template<typename HostType, typename DeviceType>
void TlrmvmGPU<HostType,DeviceType>::AllocatePhase3Buffer(){
    TlrmvmCPU<HostType>::AllocatePhase3Buffer();
    GetDeviceMemory(&d_Au, this->phase3Acnt);
    GetDeviceMemory(&d_yu, this->phase3Xcnt);
    GetDeviceMemory(&d_y, this->phase3Ycnt);
    GetDeviceMemory(&d_yout, this->phase3Ycnt);
    d_Aubp[0] = d_Au;
    d_yubp[0] = d_yu;
    d_ybp[0] = d_y;
    d_youtbp[0] = d_yout;
}

template<typename HostType, typename DeviceType>
void TlrmvmGPU<HostType,DeviceType>::Phase3CopyData(){
    TlrmvmCPU<HostType>::Phase3CopyData();
    auto AuMs = this->AuMs;
    auto AuNs = this->AuNs;
    auto AuKs = this->AuKs;
    for(int i=1; i<this->Mtg; i++){
        size_t AuMK = AuMs[i-1] * AuKs[i-1];
        size_t AuKN = AuKs[i-1] * AuNs[i-1];
        size_t AuMN = AuMs[i-1] * AuNs[i-1];
        d_Aubp[i] = d_Aubp[i-1] + AuMK;
        d_yubp[i] = d_yubp[i-1] + AuKN;
        d_ybp[i] = d_ybp[i-1] + AuMN;
        d_youtbp[i] = d_youtbp[i-1] + AuMN;
    }
    CopyDataB2HD((HostType*)d_Au, this->h_Au, this->phase3Acnt);
}

template<typename HostType, typename DeviceType>
void TlrmvmGPU<HostType,DeviceType>::MVM(){
    if(!graphCreated){
        cudaStreamBeginCapture(streamptr[0],cudaStreamCaptureModeGlobal);
        cudaEventRecord(event_start, streamptr[0]);
        for(int streami=1; streami<streamsize; streami++){
            cudaStreamWaitEvent(streamptr[streami], event_start);
        }
        // phase 1
        for(int i=0; i<this->Ntg; i++){
            int streamid = (i) % (streamsize);
            if(this->AvMs[i] != 0){
                CUBLASCHECK(cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                this->AvMs[i], this->AvNs[i], this->AvKs[i], 
                &alpha, this->d_Avbp[i], this->AvMs[i], 
                this->d_xbp[i], this->AvKs[i], 
                &beta, this->d_yvbp[i], this->AvMs[i]));
            }
        }
        for(int streami=1; streami < streamsize; streami++){
            cudaEventRecord(events[streami], streamptr[streami]);
        }
        for(int streami=1; streami < streamsize; streami++){
            cudaStreamWaitEvent(streamptr[0], events[streami]);
        }
        // phase 2
        phase2_nosplit<DeviceType>(this->d_yv, this->d_phase2mapping, this->d_yu, 
        this->granksum, streamptr[0]);
        cudaEventRecord(events[0], streamptr[0]);
        for(int streami=1; streami < streamsize; streami++){
            cudaStreamWaitEvent(streamptr[streami], events[0]);
        }
        // phase 3
        for(int i=0; i<this->Mtg; i++){
            int streamid = (i) % (streamsize);
            if(this->AuMs[i] != 0){
                CUBLASCHECK(
                    cublasgemm(cublashandleptr[streamid],CUBLAS_OP_N, CUBLAS_OP_N,
                    this->AuMs[i], this->AuNs[i], this->AuKs[i], 
                    &alpha, this->d_Aubp[i], this->AuMs[i], 
                    this->d_yubp[i], this->AuKs[i], 
                    &beta, this->d_ybp[i], this->AuMs[i])
                );
            }
        }
        // final merge
        for(int streami=1; streami < streamsize; streami++){
            cudaEventRecord(events[streamsize + streami], streamptr[streami]);
        }
        for(int streami=1; streami < streamsize; streami++){
            cudaStreamWaitEvent(streamptr[0], events[streamsize + streami]);
        }
        cudaStreamEndCapture(streamptr[0], &graph);
        cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
        graphCreated = true;
    }

    cudaGraphLaunch(instance, streamptr[0]);
    cudaStreamSynchronize(streamptr[0]);

}

template<typename HostType, typename DeviceType>
void TlrmvmGPU<HostType, DeviceType>::CopyBackResults(){
    CopyDataB2HD(this->h_yv, (HostType*)d_yv, this->granksum);
    CopyDataB2HD(this->h_yu, (HostType*)d_yu, this->granksum);
    CopyDataB2HD(this->h_y, (HostType*)d_y, this->originM);
}

template<typename HostType, typename DeviceType>
void TlrmvmGPU<HostType,DeviceType>::FreeData(){
    TlrmvmCPU<HostType>::FreeData();
}

template class TlrmvmGPU<float, float>;
template class TlrmvmGPU<double, double>;
template class TlrmvmGPU<complex<float>, cuComplex>;
template class TlrmvmGPU<complex<double>, cuDoubleComplex>;



ComplexPtr::ComplexPtr(int M, int N, int nb, Matrix<int> OrgRmat, Matrix<int> ComplexRmat)
:M(M),N(N),nb(nb),Mtg(M/nb),Ntg(N/nb),OrgRmat(OrgRmat),ComplexRmat(ComplexRmat)
{
    if(Mtg != 39){cout << "something wrong" << endl; exit(1);}
    colsum = ComplexRmat.ColSum();
    Maskmat = ComplexRmat;
    for(int i=0; i<Maskmat.Col(); i++){
        for(int j=0; j<Maskmat.Row(); j++){
            if(Maskmat.GetElem(j, i) != 0){
                Maskmat.SetElem(j,i,1);
            }
        }
    }
}

void ComplexPtr::InitData(string datafolder, string acc, int freqx, size_t originN){
    complex<float> *DataAv;
    complex<float> *DataAu;
    complex<float> *Datax_originN;
    complex<float> *Datax = new complex<float>[N];
    for(int i=0; i<N; i++) Datax[i] = complex<float>(0.0,0.0);
    size_t granksum = OrgRmat.Sum();
    ReadSeismicBinary(datafolder+"/V", &DataAv, granksum * nb, acc ,nb, freqx);
    ReadSeismicBinary(datafolder+"/U", &DataAu, granksum * nb, acc ,nb, freqx);
    ReadSeismicBinaryX(datafolder, &Datax_originN, originN, acc, nb, freqx);
    for(int i=0; i<originN; i++) Datax[i] = Datax_originN[i];
    delete[] Datax_originN;
    this->originN = originN;
    complex<float> alpha = 1.0;
    complex<float> beta = 0.0;

    for(int i=0; i<39; i++){
        AvMs.push_back(colsum[i]);
        AvKs.push_back(nb);
        AvNs.push_back(1);
    }
    complexgranksum = ComplexRmat.Sum();
    
    size_t totalAcnt = 0;
    size_t totalXcnt = 0;
    size_t totalYcnt = 0;

    for(int i=0; i<AvMs.size(); i++){
        totalAcnt += AvMs[i] * AvKs[i];
        totalXcnt += AvKs[i] * AvNs[i];
        totalYcnt += AvMs[i] * AvNs[i];
    }

    // host memory  phase 1
    GetHostMemory(&h_Av, totalAcnt); // Av
        Fillval(h_Av, beta, totalAcnt);
    h_Avbp[0] = h_Av;
    GetHostMemory(&h_x, totalXcnt); // x 
        Fillval(h_x, beta, totalXcnt);
    h_xbp[0] = h_x;
    GetHostMemory(&h_yv, totalYcnt); // yv rr_ri
    h_yvbp[0] = h_yv;
    GetHostMemory(&h_yvout, complexgranksum); // yv real
    h_yvoutbp[0] = h_yvout;



    // Device Memory phase 1
    GetDeviceMemory(&d_Av, totalAcnt); // Av real
    d_Avbp[0] = d_Av;
    GetDeviceMemory(&d_x, totalXcnt); // x 
    d_xbp[0] = d_x;
    GetDeviceMemory(&d_yv, totalYcnt); // yv rr_ri
    d_yvbp[0] = d_yv;
    GetDeviceMemory(&d_yvout, complexgranksum); // yv real
    d_yvoutbp[0] = d_yvout;
    GetDeviceMemory(&d_complexcolrank, colsum.size()); // colrank fp32

    for(int i=1; i<39; i++){
        size_t AvMK = AvMs[i-1] * AvKs[i-1];
        size_t AvKN = AvKs[i-1] * AvNs[i-1];
        size_t AvMN = AvMs[i-1] * AvNs[i-1];
        h_Avbp[i] = h_Avbp[i-1] + AvMK;
        h_xbp[i] = h_xbp[i-1] + AvKN;
        h_yvbp[i] = h_yvbp[i-1] + AvMN;
        d_Avbp[i] = d_Avbp[i-1] + AvMK;
        d_xbp[i] = d_xbp[i-1] + AvKN;
        d_yvbp[i] = d_yvbp[i-1] + AvMN;
    }
    // move data from DataAv to h_Av 
    gcolsum = OrgRmat.ColSum();
    complex<float> * Avwalkptr = DataAv;
    for(int i=0; i<Ntg; i++){
        // column start pointers
        complex<float> *colptr = h_Avbp[i];
        
        size_t lda = gcolsum[i];
        for(int nbi = 0; nbi < nb; nbi++){
            for(int j=0; j < Mtg; j++){
                int currank = OrgRmat.GetElem(j,i);
               if(ComplexRmat.GetElem(j,i) == OrgRmat.GetElem(j,i)){
                   for(int k=0; k<currank; k++){
                       *(colptr+k) = *(Avwalkptr + k);
                   }
                   colptr += currank;
                }
                Avwalkptr += currank;
            }
        }
    }

    // move data from Datax to hx
    complex<float> * xwalkptr = Datax;
    size_t offset = 0;
    for(int i=0; i<Ntg; i++){
        for(int j=0; j<nb; j++){
            *(h_x + offset + j) = *(xwalkptr + i*nb + j);
        }
        offset += nb;
    }





    // phase 2
    vector<vector<vector<vector<int>>>> phase2record;
    phase2record.resize(Mtg, vector<vector<vector<int>>>()); // Mtg row
    for(int i=0; i<Mtg; i++) phase2record[i].resize(Ntg, vector<vector<int>>()); // Ntg col
    for(int i=0; i<Mtg; i++){
        for(int j=0; j<Ntg; j++){
            phase2record[i][j].resize(2, vector<int>());
        }
    }

    size_t p2walker = 0;
    for(int i=0; i<Mtg; i++){
        for(int j=0; j<Ntg; j++){
            if(ComplexRmat.GetElem(i,j) != 0){
                int currank = ComplexRmat.GetElem(i,j);
                for(int k=0; k<currank; k++){
                    phase2record[i][j][0].push_back(p2walker++);
                }
            }
        }
        for(int j=0; j<Ntg; j++){
            if(ComplexRmat.GetElem(i,j) != 0){
                int currank = ComplexRmat.GetElem(i,j);
                for(int k=0; k<currank; k++){
                    // phase2record[i][j][1].push_back(p2walker++);
                }
            }
        }
    }
    // unfold
    for(int i=0; i<Ntg; i++){
        for(int j=0; j<Mtg; j++){
            if(ComplexRmat.GetElem(j,i) != 0){
                int currank = ComplexRmat.GetElem(j,i);
                for(int k=0; k<currank; k++){
                    h_phase2mapping.push_back(phase2record[j][i][0][k]);
                }
            }
        }
    }
    for(int i=0; i<Ntg; i++){
        for(int j=0; j<Mtg; j++){
            if(ComplexRmat.GetElem(j,i) != 0){
                int currank = ComplexRmat.GetElem(j,i);
                for(int k=0; k<currank; k++){
                    // h_phase2mapping.push_back(phase2record[j][i][1][k]);
                }
            }
        }
    }
    GetDeviceMemory(&d_phase2mapping, 2*complexgranksum);


    // phase 3
    rowsum = ComplexRmat.RowSum();
    for(int i=0; i<39; i++){
        AuMs.push_back(nb);
        AuKs.push_back(rowsum[i]);
        AuNs.push_back(1);
    }
    complexgranksum = ComplexRmat.Sum();
    
    totalAcnt = 0;
    totalXcnt = 0;
    totalYcnt = 0;

    for(int i=0; i<AuMs.size(); i++){
        totalAcnt += AuMs[i] * AuKs[i];
        totalXcnt += AuKs[i] * AuNs[i];
        totalYcnt += AuMs[i] * AuNs[i];
    }

    GetHostMemory(&h_Au, totalAcnt); // Au real
        Fillval(h_Au, alpha, totalAcnt);
    h_Aubp[0] = h_Au;
    GetHostMemory(&h_yu, totalXcnt); // x 
        Fillval(h_yu, alpha, totalXcnt);
    h_yubp[0] = h_yu;
    GetHostMemory(&h_y, totalYcnt); // y rr_ri
    h_ybp[0] = h_y;
    GetHostMemory(&h_yout, M); // yout real
    h_youtbp[0] = h_yout;
    GetHostMemory(&h_finaly, M); // yout imag

    // Device Memory phase 3
    GetDeviceMemory(&d_Au, totalAcnt); // Au real
    d_Aubp[0] = d_Au;
    GetDeviceMemory(&d_yu, totalXcnt);
    d_yubp[0] = d_yu;
    GetDeviceMemory(&d_y, totalYcnt); // y rr_ri
    d_ybp[0] = d_y;
    GetDeviceMemory(&d_yout, complexgranksum); // yout real
    d_youtbp[0] = d_yout;
    GetDeviceMemory(&d_complexrowrank, rowsum.size()); // rowrank fp32
    GetDeviceMemory(&d_finaly, M); // final output
    
    for(int i=1; i<39; i++){
        size_t AuMK = AuMs[i-1] * AuKs[i-1];
        size_t AuKN = AuKs[i-1] * AuNs[i-1];
        size_t AuMN = AuMs[i-1] * AuNs[i-1];

        h_Aubp[i] = h_Aubp[i-1] + AuMK;
        h_yubp[i] = h_yubp[i-1] + AuKN;
        h_ybp[i] = h_ybp[i-1] + AuMN;

        d_Aubp[i] = d_Aubp[i-1] + AuMK;
        d_yubp[i] = d_yubp[i-1] + AuKN;
        d_ybp[i] = d_ybp[i-1] + AuMN;

    }

    // move data Au to memory buffer
    {
        complex<float> *colptr = h_Au;
        complex<float> *dataauwalker = DataAu;
        for(int i=0; i<Mtg; i++)
        {
            for(int j=0; j<Ntg; j++){
                int currank = OrgRmat.GetElem(i,j);
                if(Maskmat.GetElem(i, j) == 1){
                    for(size_t k=0; k<currank*nb; k++){
                        *(colptr) = *(dataauwalker+k);
                        colptr++;
                    }
                }
                dataauwalker += currank * nb;
            }
        }
    }
    delete[] DataAu; delete[] DataAv; delete[] Datax;
}



void ComplexPtr::CopyData2GPU(){

    CopyDataB2HD(d_Av, h_Av, complexgranksum * nb);
    CopyDataB2HD(d_x, h_x, nb * Ntg);
    
    CopyDataB2HD(d_phase2mapping, h_phase2mapping.data(), 2*complexgranksum);

    CopyDataB2HD(d_Au, h_Au, complexgranksum * nb);


}

void ComplexPtr::FreeData(){
    FreeDeviceMemory(d_Av);
    FreeDeviceMemory(d_x);
}





}



