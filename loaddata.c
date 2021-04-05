#include <stdlib.h>
#include <stdio.h>

#define real_t float

int main(int argc, char * argv[]){
    FILE *f;
    char filename[100] = "testa.bin";
    printf("loading %s \n", filename);
    f = fopen(filename,"rb");
    unsigned long int fnum = 20 * 20;
    real_t * fval = (real_t*)malloc(sizeof(real_t) * fnum);
    fread((fval), sizeof(real_t), fnum, f);
    int i;
    
    for(int i=0; i<100; i++){
        printf("%d %f \n", i, fval[i]);
    }
    fclose(f);
}