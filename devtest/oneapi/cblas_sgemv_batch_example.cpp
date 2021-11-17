#include <string.h>
#include <cstdlib>
#include <mkl.h>
#include <oneapi/mkl.hpp>
#include <CL/sycl.hpp>

using namespace cl;
#define real_t float _Complex

int main(){
    auto queue = sycl::queue(sycl::cpu_selector());
    // int batchsize = 40;
    // int *m_array = malloc(sizeof(int) * batchsize);
    // int *n_array = malloc(sizeof(int) * batchsize);
    // int *curmsize = malloc(sizeof(int) * batchsize);
    // for(int i=0; i<batchsize; i++) curmsize[i] = 256 + i;
    // int curm = 0;
    // for(int i=0; i<batchsize; i++) curm += curmsize[i];
    // int curn = 256;
    // real_t * Aptr = malloc(sizeof(real_t) * curm * curn);
    // for(int i=0; i< curm * curn; i++) Aptr[i] = 1.0;
    // real_t * Bptr = malloc(sizeof(real_t) * batchsize * curn);
    // for(int i=0; i<batchsize * curn; i++) Bptr[i] = 1.0;
    // real_t * Cptr = malloc(sizeof(real_t) * curm);

    // real_t ** Abatch = malloc(sizeof(real_t*) * batchsize);
    // Abatch[0] = Aptr;
    // real_t ** Bbatch = malloc(sizeof(real_t*) * batchsize);
    // Bbatch[0] = Bptr;
    // real_t ** Cbatch = malloc(sizeof(real_t*) * batchsize);
    // Cbatch[0] = Cptr;
    // for(int i=0; i<batchsize; i++){
    //     m_array[i] = curmsize[i];
    //     n_array[i] = curn;
    //     if(i >= 1){
    //         Abatch[i] = Abatch[i-1] + curmsize[i-1] * curn;
    //         Bbatch[i] = Bbatch[i-1] + curn;
    //         Cbatch[i] = Cbatch[i-1] + curmsize[i-1];
    //     }
    // }

    // CBLAS_TRANSPOSE* trans_array;
    // trans_array = malloc(sizeof(CBLAS_TRANSPOSE) * batchsize);
    // for(int i=0; i<batchsize; i++) trans_array[i] = CblasNoTrans;
    
    // real_t *alpha_array =  malloc(sizeof(real_t) * batchsize);
    // int *lda_array = malloc(sizeof(int) * batchsize);
    // int *incx_array = malloc(sizeof(int) * batchsize);
    // real_t *beta_array =  malloc(sizeof(real_t) * batchsize);
    // int *incy_array = malloc(sizeof(int) * batchsize);
    // int group_count = batchsize;
    // int *group_size_array = malloc(sizeof(int) * batchsize);

    // for(int i=0; i<batchsize; i++){
    //     lda_array[i] = m_array[i];
    //     alpha_array[i] = (real_t)1.0;
    //     beta_array[i] = (real_t)0.0;
    //     incy_array[i] = 1;
    //     incx_array[i] = 1;
    //     group_size_array[i] = 1;
    // }

    // cblas_cgemv_batch(CblasColMajor,trans_array,
    // m_array, n_array, alpha_array, 
    // (const void**)Abatch, lda_array, (const void**)Bbatch, incx_array,
    // beta_array, (void **)Cbatch, incy_array, 1, &batchsize);
    
}