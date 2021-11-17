#include "TlrmvmMixedPrec.h"
#include <common/AppUtil.h>

using namespace cudatlrmat;

namespace cudatlrmvm
{



Float32Ptr::Float32Ptr(int M, int N, int nb, Matrix<int> OrgRmat, Matrix<int> FP32Rmat)
:M(M),N(N),nb(nb),Mtg(M/nb),Ntg(N/nb),OrgRmat(OrgRmat),FP32Rmat(FP32Rmat)
{

    if(Mtg != 39){cout << "something wrong" << endl; exit(1);}
    colsum = FP32Rmat.ColSum();
    Maskmat = FP32Rmat;
    for(int i=0; i<Maskmat.Col(); i++){
        for(int j=0; j<Maskmat.Row(); j++){
            if(Maskmat.GetElem(j, i) != 0){
                Maskmat.SetElem(j,i,1);
            }
        }
    }
}



void Float32Ptr::InitData(string datafolder, string acc, int freqx, size_t originN){
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
    float alpha = 1.0;
    float beta = 0.0;

    for(int i=0; i<39; i++){
        AvMs.push_back(colsum[i]);
        AvKs.push_back(nb);
        AvNs.push_back(2);
    }
    fp32granksum = FP32Rmat.Sum();
    
    size_t totalAcnt = 0;
    size_t totalXcnt = 0;
    size_t totalYcnt = 0;

    for(int i=0; i<AvMs.size(); i++){
        totalAcnt += AvMs[i] * AvKs[i];
        totalXcnt += AvKs[i] * AvNs[i];
        totalYcnt += AvMs[i] * AvNs[i];
    }

    // host memory  phase 1
    GetHostMemory(&h_Av[0], totalAcnt); // Av real
        Fillval(h_Av[0], beta, totalAcnt);
    h_Avbp[0][0] = h_Av[0];
    GetHostMemory(&h_Av[1], totalAcnt); // Av imag
        Fillval(h_Av[1], beta, totalAcnt);
    h_Avbp[0][1] = h_Av[1];
    GetHostMemory(&h_x, totalXcnt); // x 
        Fillval(h_x, beta, totalXcnt);
    h_xbp[0] = h_x;
    GetHostMemory(&h_yv[0], totalYcnt); // yv rr_ri
    h_yvbp[0][0] = h_yv[0];
    GetHostMemory(&h_yv[1], totalYcnt); // yv ir_ii
    h_yvbp[0][1] = h_yv[1];
    if(fp32granksum != totalYcnt/2) cout << "somethin is wrong fp32grank totalYcnt" << endl;
    if(totalXcnt != Ntg * nb * 2) cout << "someting is wrong with xdim " << endl;
    GetHostMemory(&h_yvout[0], fp32granksum); // yv real
    h_yvoutbp[0][0] = h_yvout[0];
    GetHostMemory(&h_yvout[1], fp32granksum); // yv imag
    h_yvoutbp[0][1] = h_yvout[1];



    // Device Memory phase 1
    GetDeviceMemory(&d_Av[0], totalAcnt); // Av real
    d_Avbp[0][0] = d_Av[0];
    GetDeviceMemory(&d_Av[1], totalAcnt); // Av imag
    d_Avbp[0][1] = d_Av[1];
    GetDeviceMemory(&d_x, totalXcnt); // x 
    d_xbp[0] = d_x;
    GetDeviceMemory(&d_yv[0], totalYcnt); // yv rr_ri
    d_yvbp[0][0] = d_yv[0];
    GetDeviceMemory(&d_yv[1], totalYcnt); // yv ir_ii
    d_yvbp[0][1] = d_yv[1];
    GetDeviceMemory(&d_yvout[0], fp32granksum); // yv real
    d_yvoutbp[0][0] = d_yvout[0];
    GetDeviceMemory(&d_yvout[1], fp32granksum); // yv imag
    d_yvoutbp[0][1] = d_yvout[1];
    GetDeviceMemory(&d_fp32colrank, colsum.size()); // colrank fp32

    for(int i=1; i<39; i++){
        
        size_t AvMK = AvMs[i-1] * AvKs[i-1];
        size_t AvKN = AvKs[i-1] * AvNs[i-1];
        size_t AvMN = AvMs[i-1] * AvNs[i-1];

        h_Avbp[i][0] = h_Avbp[i-1][0] + AvMK;
        h_Avbp[i][1] = h_Avbp[i-1][1] + AvMK;
        h_xbp[i] = h_xbp[i-1] + AvKN;
        h_yvbp[i][0] = h_yvbp[i-1][0] + AvMN;
        h_yvbp[i][1] = h_yvbp[i-1][1] + AvMN;
        h_yvoutbp[i][0] = h_yvoutbp[i-1][0] + colsum[i-1];
        h_yvoutbp[i][1] = h_yvoutbp[i-1][1] + colsum[i-1];
        
        d_Avbp[i][0] = d_Avbp[i-1][0] + AvMK;
        d_Avbp[i][1] = d_Avbp[i-1][1] + AvMK;
        d_xbp[i] = d_xbp[i-1] + AvKN;
        d_yvbp[i][0] = d_yvbp[i-1][0] + AvMN;
        d_yvbp[i][1] = d_yvbp[i-1][1] + AvMN;
        d_yvoutbp[i][0] = d_yvoutbp[i-1][0] + colsum[i-1];
        d_yvoutbp[i][1] = d_yvoutbp[i-1][1] + colsum[i-1];

    }
    // move data from DataAv to h_Av 
    gcolsum = OrgRmat.ColSum();
    complex<float> * Avwalkptr = DataAv;
    for(int i=0; i<Ntg; i++){
        // column start pointers
        float *realcolptr = h_Avbp[i][0];
        float *imagcolptr = h_Avbp[i][1];
        
        size_t lda = gcolsum[i];
        for(int nbi = 0; nbi < nb; nbi++){
            for(int j=0; j < Mtg; j++){
                int currank = OrgRmat.GetElem(j,i);
               if(FP32Rmat.GetElem(j,i) == OrgRmat.GetElem(j,i)){
                   for(int k=0; k<currank; k++){
                       *(realcolptr+k) = (Avwalkptr + k)->real();
                       *(imagcolptr+k) = (Avwalkptr + k)->imag();
                   }
                   realcolptr += currank;
                   imagcolptr += currank;
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
            *(h_x + offset + j) = (xwalkptr + i*nb + j)->real();
            *(h_x + offset + nb + j) = (xwalkptr + i*nb + j)->imag();
        }
        offset += 2 * nb;
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
            if(FP32Rmat.GetElem(i,j) != 0){
                int currank = FP32Rmat.GetElem(i,j);
                for(int k=0; k<currank; k++){
                    phase2record[i][j][0].push_back(p2walker++);
                }
            }
        }
        for(int j=0; j<Ntg; j++){
            if(FP32Rmat.GetElem(i,j) != 0){
                int currank = FP32Rmat.GetElem(i,j);
                for(int k=0; k<currank; k++){
                    phase2record[i][j][1].push_back(p2walker++);
                }
            }
        }
    }
    if(p2walker != 2*fp32granksum) cout << "p2waler phase 2 is wrong "<< p2walker << endl;
    // unfold
    for(int i=0; i<Ntg; i++){
        for(int j=0; j<Mtg; j++){
            if(FP32Rmat.GetElem(j,i) != 0){
                int currank = FP32Rmat.GetElem(j,i);
                for(int k=0; k<currank; k++){
                    h_phase2mapping.push_back(phase2record[j][i][0][k]);
                }
            }
        }
    }
    for(int i=0; i<Ntg; i++){
        for(int j=0; j<Mtg; j++){
            if(FP32Rmat.GetElem(j,i) != 0){
                int currank = FP32Rmat.GetElem(j,i);
                for(int k=0; k<currank; k++){
                    h_phase2mapping.push_back(phase2record[j][i][1][k]);
                }
            }
        }
    }
    if(h_phase2mapping.size() != 2 * fp32granksum) cout << "h_mapping phase2 is wrong" << endl;
    GetDeviceMemory(&d_phase2mapping, 2*fp32granksum);


    // phase 3
    rowsum = FP32Rmat.RowSum();
    for(int i=0; i<39; i++){
        AuMs.push_back(nb);
        AuKs.push_back(rowsum[i]);
        AuNs.push_back(2);
    }
    fp32granksum = FP32Rmat.Sum();
    
    totalAcnt = 0;
    totalXcnt = 0;
    totalYcnt = 0;

    for(int i=0; i<AuMs.size(); i++){
        totalAcnt += AuMs[i] * AuKs[i];
        totalXcnt += AuKs[i] * AuNs[i];
        totalYcnt += AuMs[i] * AuNs[i];
    }

    GetHostMemory(&h_Au[0], totalAcnt); // Au real
        Fillval(h_Au[0], alpha, totalAcnt);
    h_Aubp[0][0] = h_Au[0];
    GetHostMemory(&h_Au[1], totalAcnt); // Au imag
        Fillval(h_Au[1], alpha, totalAcnt);
    h_Aubp[0][1] = h_Au[1];
    GetHostMemory(&h_yu, totalXcnt); // x 
        Fillval(h_yu, alpha, totalXcnt);
    h_yubp[0] = h_yu;
    GetHostMemory(&h_y[0], totalYcnt); // y rr_ri
    h_ybp[0][0] = h_y[0];
    GetHostMemory(&h_y[1], totalYcnt); // y ir_ii
    h_ybp[0][1] = h_y[1];
    GetHostMemory(&h_yout[0], M); // yout real
    h_youtbp[0][0] = h_yout[0];
    GetHostMemory(&h_yout[1], M); // yout imag
    h_youtbp[0][1] = h_yout[1];
    GetHostMemory(&h_finaly, M); // yout imag

    // Device Memory phase 3
    GetDeviceMemory(&d_Au[0], totalAcnt); // Au real
    d_Aubp[0][0] = d_Au[0];
    GetDeviceMemory(&d_Au[1], totalAcnt); // Au imag
    d_Aubp[0][1] = d_Au[1];
    GetDeviceMemory(&d_yu, totalXcnt);
    d_yubp[0] = d_yu;
    GetDeviceMemory(&d_y[0], totalYcnt); // y rr_ri
    d_ybp[0][0] = d_y[0];
    GetDeviceMemory(&d_y[1], totalYcnt); // y ir_ii
    d_ybp[0][1] = d_y[1];
    GetDeviceMemory(&d_yout[0], fp32granksum); // yout real
    d_youtbp[0][0] = d_yout[0];
    GetDeviceMemory(&d_yout[1], fp32granksum); // yout imag
    d_youtbp[0][1] = d_yout[1];
    GetDeviceMemory(&d_fp32rowrank, rowsum.size()); // rowrank fp32
    GetDeviceMemory(&d_finaly, M); // final output
    
    for(int i=1; i<39; i++){
        size_t AuMK = AuMs[i-1] * AuKs[i-1];
        size_t AuKN = AuKs[i-1] * AuNs[i-1];
        size_t AuMN = AuMs[i-1] * AuNs[i-1];

        h_Aubp[i][0] = h_Aubp[i-1][0] + AuMK;
        h_Aubp[i][1] = h_Aubp[i-1][1] + AuMK;
        h_yubp[i] = h_yubp[i-1] + AuKN;
        h_ybp[i][0] = h_ybp[i-1][0] + AuMN;
        h_ybp[i][1] = h_ybp[i-1][1] + AuMN;
        h_youtbp[i][0] = h_youtbp[i-1][0] + rowsum[i-1];
        h_youtbp[i][1] = h_youtbp[i-1][1] + rowsum[i-1];
        
        d_Aubp[i][0] = d_Aubp[i-1][0] + AuMK;
        d_Aubp[i][1] = d_Aubp[i-1][1] + AuMK;
        d_yubp[i] = d_yubp[i-1] + AuKN;
        d_ybp[i][0] = d_ybp[i-1][0] + AuMN;
        d_ybp[i][1] = d_ybp[i-1][1] + AuMN;
        d_youtbp[i][0] = d_youtbp[i-1][0] + rowsum[i-1];
        d_youtbp[i][1] = d_youtbp[i-1][1] + rowsum[i-1];
    }



    // move data Au to memory buffer
    {
        float *realptr = h_Au[0];
        float *imagptr = h_Au[1];
        complex<float> *dataauwalker = DataAu;
        for(int i=0; i<Mtg; i++)
        {
            for(int j=0; j<Ntg; j++){
                int currank = OrgRmat.GetElem(i,j);
                if(Maskmat.GetElem(i, j) == 1){
                    for(size_t k=0; k<currank*nb; k++){
                        *(realptr) = (dataauwalker+k)->real();
                        *(imagptr) = (dataauwalker+k)->imag();
                        realptr++;
                        // offsetreal++;
                        imagptr++;   
                        // offsetimag++;
                    }
                }
                dataauwalker += currank * nb;
            }
        }
    }

    delete[] DataAu; delete[] DataAv; delete[] Datax;


}

void Float32Ptr::LoadRealDataset(complex<float>* DataAv, complex<float> *DataAu, complex<float>*Datax){

}

void Float32Ptr::CopyData2GPU(){

    CopyDataB2HD(d_Av[0], h_Av[0], fp32granksum * nb);
    CopyDataB2HD(d_Av[1], h_Av[1], fp32granksum * nb);
    CopyDataB2HD(d_x, h_x, nb * Ntg * 2);
    
    CopyDataB2HD(d_fp32colrank, colsum.data(), colsum.size());
    CopyDataB2HD(d_phase2mapping, h_phase2mapping.data(), 2*fp32granksum);

    CopyDataB2HD(d_Au[0], h_Au[0], fp32granksum * nb);
    CopyDataB2HD(d_Au[1], h_Au[1], fp32granksum * nb);
    CopyDataB2HD(d_yu, h_yu, 2 * fp32granksum);
    CopyDataB2HD(d_fp32rowrank, rowsum.data(), rowsum.size());


}

void Float32Ptr::FreeData(){

    delete[] h_Av[0];
    delete[] h_Av[1];
    delete[] h_x;
    delete[] h_yv[0];
    delete[] h_yv[1];
    delete[] h_yvout[0];
    delete[] h_yvout[1];

    FreeDeviceMemory(d_Av[0]);
    FreeDeviceMemory(d_Av[1]);
    FreeDeviceMemory(d_x);
    FreeDeviceMemory(d_yv[0]);
    FreeDeviceMemory(d_yv[1]);
    FreeDeviceMemory(d_yvout[0]);
    FreeDeviceMemory(d_yvout[1]);
    FreeDeviceMemory(d_fp32colrank);
    FreeDeviceMemory(d_phase2mapping);


    delete[] h_Au[0];
    delete[] h_Au[1];
    delete[] h_yu;
    delete[] h_y[0];
    delete[] h_y[1];
    delete[] h_yout[0];
    delete[] h_yout[1];


    FreeDeviceMemory(d_Au[0]);
    FreeDeviceMemory(d_Au[1]);
    FreeDeviceMemory(d_yu);
    FreeDeviceMemory(d_y[0]);
    FreeDeviceMemory(d_y[1]);
    FreeDeviceMemory(d_yout[0]);
    FreeDeviceMemory(d_yout[1]);
    FreeDeviceMemory(d_fp32rowrank);

}



Float16Ptr::Float16Ptr(int M, int N, int nb, Matrix<int> OrgRmat, Matrix<int> FP16Rmat)
:M(M),N(N),nb(nb),Mtg(M/nb),Ntg(N/nb),OrgRmat(OrgRmat),FP16Rmat(FP16Rmat)
{

    if(Mtg != 39){cout << "something wrong" << endl; exit(1);}
    colsum = FP16Rmat.ColSum();
    Maskmat = FP16Rmat;
    for(int i=0; i<Maskmat.Col(); i++){
        for(int j=0; j<Maskmat.Row(); j++){
            if(Maskmat.GetElem(j, i) != 0){
                Maskmat.SetElem(j,i,1);
            }
        }
    }

}


void Float16Ptr::InitData(string datafolder, string acc, int freqx, size_t originN){
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
    half alpha = (half)1.0;
    half beta = (half)0.0;

    for(int i=0; i<39; i++){
        AvMs.push_back(colsum[i]);
        AvKs.push_back(nb);
        AvNs.push_back(2);
    }
    fp16granksum = FP16Rmat.Sum();
    
    size_t totalAcnt = 0;
    size_t totalXcnt = 0;
    size_t totalYcnt = 0;

    for(int i=0; i<AvMs.size(); i++){
        totalAcnt += AvMs[i] * AvKs[i];
        totalXcnt += AvKs[i] * AvNs[i];
        totalYcnt += AvMs[i] * AvNs[i];
    }

    // host memory  phase 1
    GetcuHostMemory(&h_Av[0], totalAcnt); // Av real
        Fillval(h_Av[0], beta, totalAcnt);
    h_Avbp[0][0] = h_Av[0];
    GetcuHostMemory(&h_Av[1], totalAcnt); // Av imag
        Fillval(h_Av[1], beta, totalAcnt);
    h_Avbp[0][1] = h_Av[1];
    GetcuHostMemory(&h_x, totalXcnt); // x 
        Fillval(h_x, beta, totalXcnt);
    h_xbp[0] = h_x;
    GetcuHostMemory(&h_yv[0], totalYcnt); // yv rr_ri
    h_yvbp[0][0] = h_yv[0];
    GetcuHostMemory(&h_yv[1], totalYcnt); // yv ir_ii
    h_yvbp[0][1] = h_yv[1];
    if(fp16granksum != totalYcnt/2) cout << "somethin is wrong fp32grank totalYcnt" << endl;
    if(totalXcnt != Ntg * nb * 2) cout << "someting is wrong with xdim " << endl;
    GetcuHostMemory(&h_yvout[0], fp16granksum); // yv real
    h_yvoutbp[0][0] = h_yvout[0];
    GetcuHostMemory(&h_yvout[1], fp16granksum); // yv imag
    h_yvoutbp[0][1] = h_yvout[1];



    // Device Memory phase 1
    GetDeviceMemory(&d_Av[0], totalAcnt); // Av real
    d_Avbp[0][0] = d_Av[0];
    GetDeviceMemory(&d_Av[1], totalAcnt); // Av imag
    d_Avbp[0][1] = d_Av[1];
    GetDeviceMemory(&d_x, totalXcnt); // x 
    d_xbp[0] = d_x;
    GetDeviceMemory(&d_yv[0], totalYcnt); // yv rr_ri
    d_yvbp[0][0] = d_yv[0];
    GetDeviceMemory(&d_yv[1], totalYcnt); // yv ir_ii
    d_yvbp[0][1] = d_yv[1];
    GetDeviceMemory(&d_yvout[0], fp16granksum); // yv real
    d_yvoutbp[0][0] = d_yvout[0];
    GetDeviceMemory(&d_yvout[1], fp16granksum); // yv imag
    d_yvoutbp[0][1] = d_yvout[1];
    GetDeviceMemory(&d_fp16colrank, colsum.size()); // colrank fp32

    for(int i=1; i<39; i++){
        
        size_t AvMK = AvMs[i-1] * AvKs[i-1];
        size_t AvKN = AvKs[i-1] * AvNs[i-1];
        size_t AvMN = AvMs[i-1] * AvNs[i-1];

        h_Avbp[i][0] = h_Avbp[i-1][0] + AvMK;
        h_Avbp[i][1] = h_Avbp[i-1][1] + AvMK;
        h_xbp[i] = h_xbp[i-1] + AvKN;
        h_yvbp[i][0] = h_yvbp[i-1][0] + AvMN;
        h_yvbp[i][1] = h_yvbp[i-1][1] + AvMN;
        h_yvoutbp[i][0] = h_yvoutbp[i-1][0] + colsum[i-1];
        h_yvoutbp[i][1] = h_yvoutbp[i-1][1] + colsum[i-1];
        
        d_Avbp[i][0] = d_Avbp[i-1][0] + AvMK;
        d_Avbp[i][1] = d_Avbp[i-1][1] + AvMK;
        d_xbp[i] = d_xbp[i-1] + AvKN;
        d_yvbp[i][0] = d_yvbp[i-1][0] + AvMN;
        d_yvbp[i][1] = d_yvbp[i-1][1] + AvMN;
        d_yvoutbp[i][0] = d_yvoutbp[i-1][0] + colsum[i-1];
        d_yvoutbp[i][1] = d_yvoutbp[i-1][1] + colsum[i-1];

    }
    // move data from DataAv to h_Av 
    gcolsum = OrgRmat.ColSum();
    complex<float> * Avwalkptr = DataAv;
    for(int i=0; i<Ntg; i++){
        // column start pointers
        half *realcolptr = h_Avbp[i][0];
        half *imagcolptr = h_Avbp[i][1];
        
        size_t lda = gcolsum[i];
        for(int nbi = 0; nbi < nb; nbi++){
            for(int j=0; j < Mtg; j++){
                int currank = OrgRmat.GetElem(j,i);
               if(FP16Rmat.GetElem(j,i) == OrgRmat.GetElem(j,i)){
                   for(int k=0; k<currank; k++){
                       *(realcolptr+k) = (half)(Avwalkptr + k)->real();
                       *(imagcolptr+k) = (half)(Avwalkptr + k)->imag();
                   }
                   realcolptr += currank;
                   imagcolptr += currank;
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
            *(h_x + offset + j) = (half)(xwalkptr + i*nb + j)->real();
            *(h_x + offset + nb + j) = (half)(xwalkptr + i*nb + j)->imag();
        }
        offset += 2 * nb;
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
            if(FP16Rmat.GetElem(i,j) != 0){
                int currank = FP16Rmat.GetElem(i,j);
                for(int k=0; k<currank; k++){
                    phase2record[i][j][0].push_back(p2walker++);
                }
            }
        }
        for(int j=0; j<Ntg; j++){
            if(FP16Rmat.GetElem(i,j) != 0){
                int currank = FP16Rmat.GetElem(i,j);
                for(int k=0; k<currank; k++){
                    phase2record[i][j][1].push_back(p2walker++);
                }
            }
        }
    }
    if(p2walker != 2*fp16granksum) cout << "p2waler phase 2 is wrong "<< p2walker << endl;
    // unfold
    for(int i=0; i<Ntg; i++){
        for(int j=0; j<Mtg; j++){
            if(FP16Rmat.GetElem(j,i) != 0){
                int currank = FP16Rmat.GetElem(j,i);
                for(int k=0; k<currank; k++){
                    h_phase2mapping.push_back(phase2record[j][i][0][k]);
                }
            }
        }
    }
    for(int i=0; i<Ntg; i++){
        for(int j=0; j<Mtg; j++){
            if(FP16Rmat.GetElem(j,i) != 0){
                int currank = FP16Rmat.GetElem(j,i);
                for(int k=0; k<currank; k++){
                    h_phase2mapping.push_back(phase2record[j][i][1][k]);
                }
            }
        }
    }
    if(h_phase2mapping.size() != 2 * fp16granksum) cout << "h_mapping phase2 is wrong" << endl;
    GetDeviceMemory(&d_phase2mapping, 2*fp16granksum);


    // phase 3
    rowsum = FP16Rmat.RowSum();
    for(int i=0; i<39; i++){
        AuMs.push_back(nb);
        AuKs.push_back(rowsum[i]);
        AuNs.push_back(2);
    }
    fp16granksum = FP16Rmat.Sum();
    
    totalAcnt = 0;
    totalXcnt = 0;
    totalYcnt = 0;

    for(int i=0; i<AuMs.size(); i++){
        totalAcnt += AuMs[i] * AuKs[i];
        totalXcnt += AuKs[i] * AuNs[i];
        totalYcnt += AuMs[i] * AuNs[i];
    }

    GetcuHostMemory(&h_Au[0], totalAcnt); // Au real
        Fillval(h_Au[0], alpha, totalAcnt);
    h_Aubp[0][0] = h_Au[0];
    GetcuHostMemory(&h_Au[1], totalAcnt); // Au imag
        Fillval(h_Au[1], alpha, totalAcnt);
    h_Aubp[0][1] = h_Au[1];
    GetcuHostMemory(&h_yu, totalXcnt); // x 
        Fillval(h_yu, alpha, totalXcnt);
    h_yubp[0] = h_yu;
    GetcuHostMemory(&h_y[0], totalYcnt); // y rr_ri
    h_ybp[0][0] = h_y[0];
    GetcuHostMemory(&h_y[1], totalYcnt); // y ir_ii
    h_ybp[0][1] = h_y[1];
    GetcuHostMemory(&h_yout[0], M); // yout real
    h_youtbp[0][0] = h_yout[0];
    GetcuHostMemory(&h_yout[1], M); // yout imag
    h_youtbp[0][1] = h_yout[1];
    GetHostMemory(&h_finaly, M); // yout imag complex

    // Device Memory phase 3
    GetDeviceMemory(&d_Au[0], totalAcnt); // Au real
    d_Aubp[0][0] = d_Au[0];
    GetDeviceMemory(&d_Au[1], totalAcnt); // Au imag
    d_Aubp[0][1] = d_Au[1];
    GetDeviceMemory(&d_yu, totalXcnt);
    d_yubp[0] = d_yu;
    GetDeviceMemory(&d_y[0], totalYcnt); // y rr_ri
    d_ybp[0][0] = d_y[0];
    GetDeviceMemory(&d_y[1], totalYcnt); // y ir_ii
    d_ybp[0][1] = d_y[1];
    GetDeviceMemory(&d_yout[0], fp16granksum); // yout real
    d_youtbp[0][0] = d_yout[0];
    GetDeviceMemory(&d_yout[1], fp16granksum); // yout imag
    d_youtbp[0][1] = d_yout[1];
    GetDeviceMemory(&d_fp16rowrank, rowsum.size()); // rowrank fp32
    GetDeviceMemory(&d_finaly, M); // final output
    
    for(int i=1; i<39; i++){
        size_t AuMK = AuMs[i-1] * AuKs[i-1];
        size_t AuKN = AuKs[i-1] * AuNs[i-1];
        size_t AuMN = AuMs[i-1] * AuNs[i-1];

        h_Aubp[i][0] = h_Aubp[i-1][0] + AuMK;
        h_Aubp[i][1] = h_Aubp[i-1][1] + AuMK;
        h_yubp[i] = h_yubp[i-1] + AuKN;
        h_ybp[i][0] = h_ybp[i-1][0] + AuMN;
        h_ybp[i][1] = h_ybp[i-1][1] + AuMN;
        h_youtbp[i][0] = h_youtbp[i-1][0] + rowsum[i-1];
        h_youtbp[i][1] = h_youtbp[i-1][1] + rowsum[i-1];
        
        d_Aubp[i][0] = d_Aubp[i-1][0] + AuMK;
        d_Aubp[i][1] = d_Aubp[i-1][1] + AuMK;
        d_yubp[i] = d_yubp[i-1] + AuKN;
        d_ybp[i][0] = d_ybp[i-1][0] + AuMN;
        d_ybp[i][1] = d_ybp[i-1][1] + AuMN;
        d_youtbp[i][0] = d_youtbp[i-1][0] + rowsum[i-1];
        d_youtbp[i][1] = d_youtbp[i-1][1] + rowsum[i-1];
    }



    // move data Au to memory buffer
    {
        half *realptr = h_Au[0];
        half *imagptr = h_Au[1];
        complex<float> *dataauwalker = DataAu;
        for(int i=0; i<Mtg; i++)
        {
            for(int j=0; j<Ntg; j++){
                int currank = OrgRmat.GetElem(i,j);
                if(Maskmat.GetElem(i, j) == 1){
                    for(size_t k=0; k<currank*nb; k++){
                        *(realptr) = (half)(dataauwalker+k)->real();
                        *(imagptr) = (half)(dataauwalker+k)->imag();
                        realptr++;
                        imagptr++;   
                    }
                }
                dataauwalker += currank * nb;
            }
        }
    }

    delete[] DataAu; delete[] DataAv; delete[] Datax;


}

void Float16Ptr::CopyData2GPU(){

    CopyDataB2HD(d_Av[0], h_Av[0], fp16granksum * nb);
    CopyDataB2HD(d_Av[1], h_Av[1], fp16granksum * nb);
    CopyDataB2HD(d_x, h_x, nb * Ntg * 2);

    CopyDataB2HD(d_fp16colrank, colsum.data(), colsum.size());
    CopyDataB2HD(d_phase2mapping, h_phase2mapping.data(), 2*fp16granksum);

    CopyDataB2HD(d_Au[0], h_Au[0], fp16granksum * nb);
    CopyDataB2HD(d_Au[1], h_Au[1], fp16granksum * nb);
    CopyDataB2HD(d_yu, h_yu, 2 * fp16granksum);
    CopyDataB2HD(d_fp16rowrank, rowsum.data(), rowsum.size());

}

void Float16Ptr::FreeData(){

    FreecuHostMemory(h_Av[0]);
    FreecuHostMemory(h_Av[1]);
    FreecuHostMemory(h_x);
    FreecuHostMemory(h_yv[0]);
    FreecuHostMemory(h_yv[1]);
    FreecuHostMemory(h_yvout[0]);
    FreecuHostMemory(h_yvout[1]);
    

    FreeDeviceMemory(d_Av[0]);
    FreeDeviceMemory(d_Av[1]);
    FreeDeviceMemory(d_x);
    FreeDeviceMemory(d_yv[0]);
    FreeDeviceMemory(d_yv[1]);
    FreeDeviceMemory(d_yvout[0]);
    FreeDeviceMemory(d_yvout[1]);
    FreeDeviceMemory(d_fp16colrank);

    FreeDeviceMemory(d_phase2mapping);


    FreecuHostMemory(h_Au[0]);
    FreecuHostMemory(h_Au[1]);
    FreecuHostMemory(h_yu);
    FreecuHostMemory(h_y[0]);
    FreecuHostMemory(h_y[1]);
    FreecuHostMemory(h_yout[0]);
    FreecuHostMemory(h_yout[1]);


    FreeDeviceMemory(d_Au[0]);
    FreeDeviceMemory(d_Au[1]);
    FreeDeviceMemory(d_yu);
    FreeDeviceMemory(d_y[0]);
    FreeDeviceMemory(d_y[1]);
    FreeDeviceMemory(d_yout[0]);
    FreeDeviceMemory(d_yout[1]);

}







Int8Ptr::Int8Ptr(int M, int N, int nb, Matrix<int> OrgRmat, Matrix<int> Int8Rmat)
:M(M),N(N),nb(nb),Mtg(M/nb),Ntg(N/nb),OrgRmat(OrgRmat),Int8Rmat(Int8Rmat)
{

    if(Mtg != 39){cout << "something wrong" << endl; exit(1);}
    colsum = Int8Rmat.ColSum();
    Maskmat = Int8Rmat;
    for(int i=0; i<Maskmat.Col(); i++){
        for(int j=0; j<Maskmat.Row(); j++){
            if(Maskmat.GetElem(j, i) != 0){
                Maskmat.SetElem(j,i,1);
            }
        }
    }
}

size_t RoundTo16x(size_t colsum){
    if(colsum % 16 == 0) return colsum;
    return (colsum / 16 + 1) * 16;
}

void Int8Ptr::InitData(string datafolder, string acc, int freqx, size_t originN){
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
    int8_t beta = 0;
    int8granksum_withpadding = 0;
    for(int i=0; i<39; i++){
        // AvMs.push_back(colsum[i]);
        size_t roundcolsum = RoundTo16x(colsum[i]);
        colsum_withpadding.push_back(roundcolsum);
        int8granksum_withpadding += roundcolsum;
        AvMs.push_back(roundcolsum);
        AvKs.push_back(nb);
        AvNs.push_back(2);
    }
    int8granksum = Int8Rmat.Sum();
    size_t totalAcnt = 0;
    size_t totalXcnt = 0;
    size_t totalYcnt = 0;
    
    for(int i=0; i<AvMs.size(); i++){
        totalAcnt += AvMs[i] * AvKs[i];
        totalXcnt += AvKs[i] * AvNs[i];
        totalYcnt += AvMs[i] * AvNs[i];
    }
    // host memory phase 1
    GetHostMemory(&h_Av[0], totalAcnt); // Av real
        Fillval(h_Av[0], beta, totalAcnt);
    h_Avbp[0][0] = h_Av[0];
    GetHostMemory(&h_Av[1], totalAcnt); // Av imag
        Fillval(h_Av[1], beta, totalAcnt);
    h_Avbp[0][1] = h_Av[1];
    GetHostMemory(&h_x, totalXcnt); // x 
        Fillval(h_x, beta, totalXcnt);
    h_xbp[0] = h_x;
    GetHostMemory(&h_yv[0], totalYcnt); // yv rr_ri
    h_yvbp[0][0] = h_yv[0];
    GetHostMemory(&h_yv[1], totalYcnt); // yv ir_ii
    h_yvbp[0][1] = h_yv[1];
    // if(int8granksum != totalYcnt/2) cout << "somethin is wrong fp32grank totalYcnt" << endl;
    // if(totalXcnt != Ntg * nb * 2) cout << "someting is wrong with xdim " << endl;
    GetHostMemory(&h_yvout[0], int8granksum_withpadding); // yv real
    h_yvoutbp[0][0] = h_yvout[0];
    GetHostMemory(&h_yvout[1], int8granksum_withpadding); // yv imag
    h_yvoutbp[0][1] = h_yvout[1];


    // Device Memory phase 1
    GetDeviceMemory(&d_Av[0], totalAcnt); // Av real
    d_Avbp[0][0] = d_Av[0];
    GetDeviceMemory(&d_Av[1], totalAcnt); // Av imag
    d_Avbp[0][1] = d_Av[1];
    GetDeviceMemory(&d_x, totalXcnt); // x 
    d_xbp[0] = d_x;
    GetDeviceMemory(&d_yv[0], totalYcnt); // yv rr_ri
    d_yvbp[0][0] = d_yv[0];
    GetDeviceMemory(&d_yv[1], totalYcnt); // yv ir_ii
    d_yvbp[0][1] = d_yv[1];
    GetDeviceMemory(&d_yvout[0], int8granksum_withpadding); // yv real
    d_yvoutbp[0][0] = d_yvout[0];
    GetDeviceMemory(&d_yvout[1], int8granksum_withpadding); // yv imag
    d_yvoutbp[0][1] = d_yvout[1];
    GetDeviceMemory(&d_int8colrank, colsum.size()); // colrank int8

    for(int i=1; i<39; i++){
        
        size_t AvMK = AvMs[i-1] * AvKs[i-1];
        size_t AvKN = AvKs[i-1] * AvNs[i-1];
        size_t AvMN = AvMs[i-1] * AvNs[i-1];

        h_Avbp[i][0] = h_Avbp[i-1][0] + AvMK;
        h_Avbp[i][1] = h_Avbp[i-1][1] + AvMK;
        h_xbp[i] = h_xbp[i-1] + AvKN;
        h_yvbp[i][0] = h_yvbp[i-1][0] + AvMN;
        h_yvbp[i][1] = h_yvbp[i-1][1] + AvMN;
        h_yvoutbp[i][0] = h_yvoutbp[i-1][0] + colsum[i-1];
        h_yvoutbp[i][1] = h_yvoutbp[i-1][1] + colsum[i-1];
        
        d_Avbp[i][0] = d_Avbp[i-1][0] + AvMK;
        d_Avbp[i][1] = d_Avbp[i-1][1] + AvMK;
        d_xbp[i] = d_xbp[i-1] + AvKN;
        d_yvbp[i][0] = d_yvbp[i-1][0] + AvMN;
        d_yvbp[i][1] = d_yvbp[i-1][1] + AvMN;
        d_yvoutbp[i][0] = d_yvoutbp[i-1][0] + colsum[i-1];
        d_yvoutbp[i][1] = d_yvoutbp[i-1][1] + colsum[i-1];

    }
    // calculate xtilemax
    for(int i=0; i<39; i++){
        for(int j=0; j<nb; j++){
            auto curxreal = Datax[i * nb + j].real();
            auto curximag = Datax[i * nb + j].imag();
            if(j == 0){
                xtilemax[i][0] = fabs(curxreal) ;
                xtilemax[i][1] = fabs(curximag) ;
            }else{
                xtilemax[i][0] = fmax( xtilemax[i][0] , fabs(curxreal) )  ;
                xtilemax[i][1] = fmax( xtilemax[i][1] , fabs(curximag) )  ;
            }
        }
    }

    // move data from Datax to hx
    complex<float> * xwalkptr = Datax;
    size_t offset = 0;
    for(int i=0; i<Ntg; i++){
        for(int j=0; j<nb; j++){
            *(h_x + offset + j) = (int8_t)((xwalkptr + i*nb + j)->real() / xtilemax[i][0] * MAXINT8 );
            *(h_x + offset + nb + j) = (int8_t)((xwalkptr + i*nb + j)->imag() / xtilemax[i][1] * MAXINT8 );
        }
        offset += 2 * nb;
    }


    // move data from DataAv to h_Av 
    gcolsum = OrgRmat.ColSum();
    complex<float> * Avwalkptr = DataAv;
    for(int i=0; i<Ntg; i++){
        // column start pointers
        int8_t *realcolptr = h_Avbp[i][0];
        int8_t *imagcolptr = h_Avbp[i][1];
        
        size_t lda = gcolsum[i];
        for(int nbi = 0; nbi < nb; nbi++){
            for(int j=0; j < Mtg; j++){
                int currank = OrgRmat.GetElem(j,i);
               if(Int8Rmat.GetElem(j,i) == OrgRmat.GetElem(j,i)){
                    // init cur tile max val
                    if(nbi == 0)
                    {
                        Avtilemax[j][i][0] = fabs( Avwalkptr->real() );
                        Avtilemax[j][i][1] = fabs( Avwalkptr->imag() );
                    }

                    for(int k=0; k<currank; k++){
                        auto curreal = (Avwalkptr + k)->real();
                        auto curimag = (Avwalkptr + k)->imag();
                        Avtilemax[j][i][0] = fmax( Avtilemax[j][i][0] , curreal );
                        Avtilemax[j][i][1] = fmax( Avtilemax[j][i][1] , curimag );
                    }

                }
                Avwalkptr += currank;
            }
        }
        for(int nbi = 0; nbi < nb; nbi++){
            for(int j=0; j < Mtg; j++){
                int currank = OrgRmat.GetElem(j,i);
               if(Int8Rmat.GetElem(j,i) == OrgRmat.GetElem(j,i)){
                   for(int k=0; k<currank; k++){
                       *(realcolptr+k) = (int8_t)((Avwalkptr + k)->real() / Avtilemax[j][i][0] * MAXINT8 );
                       *(imagcolptr+k) = (int8_t)((Avwalkptr + k)->imag() / Avtilemax[j][i][1] * MAXINT8 );
                   }
                   realcolptr += currank;
                   imagcolptr += currank;
                }
                Avwalkptr += currank;
            }
        }
    }
    for(int i=0; i<39; i++){
        for(int j=0; j<39; j++){
            size_t curidx = i * 39 + j;
            h_Avtilemax[curidx*2] = Avtilemax[j][i][0];
            h_Avtilemax[curidx*2+1] = Avtilemax[j][i][1];
        }
    }


    // phase 2
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
            if(Int8Rmat.GetElem(i,j) != 0){
                int currank = Int8Rmat.GetElem(i,j);
                for(int k=0; k<currank; k++){
                    phase2record[i][j][0].push_back(p2walker++);
                }
            }
        }
        for(int j=0; j<Ntg; j++){
            if(Int8Rmat.GetElem(i,j) != 0){
                int currank = Int8Rmat.GetElem(i,j);
                for(int k=0; k<currank; k++){
                    phase2record[i][j][1].push_back(p2walker++);
                }
            }
        }
    }
    // if(p2walker != 2*fp32granksum) cout << "p2waler phase 2 is wrong "<< p2walker << endl;
    // unfold
    for(int i=0; i<Ntg; i++){
        for(int j=0; j<Mtg; j++){
            if(Int8Rmat.GetElem(j,i) != 0){
                int currank = Int8Rmat.GetElem(j,i);
                for(int k=0; k<currank; k++){
                    h_phase2mapping.push_back(phase2record[j][i][0][k]);
                }
            }
        }
    }
    for(int i=0; i<Ntg; i++){
        for(int j=0; j<Mtg; j++){
            if(Int8Rmat.GetElem(j,i) != 0){
                int currank = Int8Rmat.GetElem(j,i);
                for(int k=0; k<currank; k++){
                    h_phase2mapping.push_back(phase2record[j][i][1][k]);
                }
            }
        }
    }
    // if(h_phase2mapping.size() != 2 * fp32granksum) cout << "h_mapping phase2 is wrong" << endl;
    GetDeviceMemory(&d_phase2mapping, 2*int8granksum);


}


void Int8Ptr::CopyData2GPU(){

    CopyDataB2HD(d_Av[0], h_Av[0], int8granksum * nb);
    CopyDataB2HD(d_Av[1], h_Av[1], int8granksum * nb);
    CopyDataB2HD(d_x, h_x, nb * Ntg * 2);
    CopyDataB2HD(d_Avtilemax, h_Avtilemax, 39 * 39 * 2);
    CopyDataB2HD(d_int8colrank, colsum.data(), colsum.size());
    CopyDataB2HD(d_int8colrank_withpadding, colsum_withpadding.data(), colsum_withpadding.size());

    
}


// void Int8Ptr::InitData(string datafolder, string acc, int freqx, size_t originN){
//     complex<float> *DataAv;
//     complex<float> *DataAu;
//     complex<float> *Datax_originN;
//     complex<float> *Datax = new complex<float>[N];
//     for(int i=0; i<N; i++) Datax[i] = complex<float>(0.0,0.0);
//     size_t granksum = OrgRmat.Sum();
//     ReadSeismicBinary(datafolder+"/V", &DataAv, granksum * nb, acc ,nb, freqx);
//     ReadSeismicBinary(datafolder+"/U", &DataAu, granksum * nb, acc ,nb, freqx);
//     ReadSeismicBinaryX(datafolder, &Datax_originN, originN, acc, nb, freqx);
//     for(int i=0; i<originN; i++) Datax[i] = Datax_originN[i];
//     cout << endl;
//     delete[] Datax_originN;
//     this->originN = originN;

//     for(int i=0; i<39; i++){
//         AvMs.push_back(colsum[i]);
//         AvKs.push_back(nb);
//         AvNs.push_back(1);
//     }
//     int8granksum = Int8Rmat.Sum();
//     size_t totalAcnt = 0;
//     size_t totalXcnt = 0;
//     size_t totalYcnt = 0;

//     for(int i=0; i<AvMs.size(); i++){
//         totalAcnt += AvMs[i] * AvKs[i];
//         totalXcnt += AvKs[i] * AvNs[i];
//         totalYcnt += AvMs[i] * AvNs[i];
//     }
//     // host memory  phase 1 real memory
//     GetHostMemory(&h_Av, 2 * totalAcnt); // Av
//     h_Avbp[0] = h_Av;
//     GetHostMemory(&h_Av_complex, totalAcnt); // Av float
//     GetHostMemory(&h_x, 2 * totalXcnt); // x 
//     h_xbp[0] = h_x;
//     GetHostMemory(&h_yv, 2 * totalYcnt); // yv
//     h_yvbp[0] = h_yv;

//     if(int8granksum != totalYcnt) cout << "somethin is wrong fp32grank totalYcnt" << endl;
//     if(totalXcnt != Ntg * nb) cout << "someting is wrong with xdim " << endl;

//     // Device Memory phase 1
//     GetDeviceMemory(&d_Av, 2 * totalAcnt); // Av real
//     d_Avbp[0] = d_Av;
//     GetDeviceMemory(&d_x, 2 * totalXcnt); // x 
//     d_xbp[0] = d_x;
//     GetDeviceMemory(&d_yv, 2 * totalYcnt); // yv 
//     d_yvbp[0] = d_yv;

//     GetDeviceMemory(&d_int8colrank, colsum.size()); // colrank fp32


//     for(int i=1; i<39; i++){
        
//         size_t AvMK = AvMs[i-1] * AvKs[i-1];
//         size_t AvKN = AvKs[i-1] * AvNs[i-1];
//         size_t AvMN = AvMs[i-1] * AvNs[i-1];
        
//         // phase 1 cpu batch pointers
//         h_Avbp[i] = h_Avbp[i-1] + 2 * AvMK;
//         h_xbp[i] = h_xbp[i-1] + 2 * AvKN;
//         h_yvbp[i] = h_yvbp[i-1] + 2 * AvMN;        
//         // phase 1 gpu batch pointers
//         d_Avbp[i] = d_Avbp[i-1] + 2 * AvMK;
//         d_xbp[i] = d_xbp[i-1] + 2 * AvKN;
//         d_yvbp[i] = d_yvbp[i-1] + 2 * AvMN;

//     }


//     // calculate xtilemax
//     for(int i=0; i<39; i++){
//         for(int j=0; j<nb; j++){
//             auto curxreal = Datax[i * nb + j].real();
//             auto curximag = Datax[i * nb + j].imag();
//             if(j == 0){
//                 xtilemax[i] = fmax( fabs(curxreal), fabs(curximag) );
//             }else{
//                 auto max1 = fmax( fabs(curxreal), fabs(curximag) );
//                 xtilemax[i] = fmax( max1, xtilemax[i] );
//             }
//         }
//     }
//     // copy float x to int8 x
//     for(int i=0; i<39; i++){
//         for(int j=0; j<nb; j++){
//             h_x[2 * (i * nb + j) ] = (int8_t) ( Datax[i * nb + j].real() / xtilemax[i] * (float)MAXINT8 );
//             h_x[2 * (i * nb + j) + 1] = (int8_t) ( Datax[i * nb + j].imag() / xtilemax[i] * (float)MAXINT8 );
//         }
//     }

//     // first copy Av 
//     gcolsum = OrgRmat.ColSum();
//     {
//         complex<float> * Avwalkptr = DataAv;
//         complex<float> * i8walkptr = h_Av_complex;
//         for(int i=0; i<Ntg; i++){
//             //column start pointers
//             size_t lda = gcolsum[i];
//             for(int nbi=0; nbi < nb; nbi++){
//                 for(int j=0; j<Mtg; j++){
//                     int currank = OrgRmat.GetElem(j,i);
//                     if(Int8Rmat.GetElem(j,i) == OrgRmat.GetElem(j,i)){
//                         for(int k=0; k<currank; k++){
//                             *(i8walkptr + k) = *(Avwalkptr + k);
//                         }
//                         i8walkptr += currank;
//                     }
//                     Avwalkptr += currank;
//                 }
//             }
//         }
//     }
//     // for(int i=0; i<10; i++){
//     //     cout << DataAv[i] << endl;
//     // }
//     // then get Av col max
//     {
//         size_t offset = 0;
//         complex<float> * Avwalkptr = h_Av_complex;
//         for(int i=0; i<Ntg; i++){
//             size_t curelem = colsum[i] * nb;
//             float maxcolelem = 0.0;
//             for(int j=offset; j<offset+curelem; j++){
//                 maxcolelem = fmax( maxcolelem, fabs(Avwalkptr->real()) );
//                 maxcolelem = fmax( maxcolelem, fabs(Avwalkptr->imag()) );
//                 Avwalkptr++;
//             }
//             Avcolmax[i] = maxcolelem;
//         }
//     }
//     // then copy float Av to int8 Av
//     {
//         size_t offset = 0;
//         complex<float> * Avwalkptr = h_Av_complex;
//         for(int i=0; i<Ntg; i++){
//             size_t curelem = colsum[i] * nb;
//             for(int j=0; j<offset + curelem; j++){
//                 h_Av[2*j] = (int8_t) ( Avwalkptr->real() / Avcolmax[i] * MAXINT8 );
//                 h_Av[2*j + 1] = (int8_t) ( Avwalkptr->imag() / Avcolmax[i] * MAXINT8 );
//                 Avwalkptr++;
//             }
//         }
//     }


// }






} // namespace cudatlrmvm
