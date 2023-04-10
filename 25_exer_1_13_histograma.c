
/*
 Programa que imprime el histograma de las longitudes de las palabras
 de su entrada. El programa usa barras horizontales para imprimir
 el histograma.
 */

#include <stdio.h>

#define EXIT_SUCCESS    0

int main() {
    int c, i, nover, wsize, totwords;
    int nwordsize[10];
    nover=wsize=totwords=0;
    int j=0;

    for(i=0; i<10; ++i)
        nwordsize[i] = 0; // rellena de 0 a 9 con ceros en el arreglo

    while((c=getchar())!=EOF) {
        if(c!=' ' && c!='\n' && c!='\t') {
            ++wsize;
        } else if(wsize<=10) {
            ++nwordsize[wsize-1];
            ++totwords;
            wsize=0;
        } else {
            ++nover;
            wsize=0;
        } // end if-else
    } // end while

    printf("\nPalabras Totales = %d",totwords+nover);
    printf("\nEstadísticas: ");

    for(i=9;i>=0;--i){
        printf("\n\t[%d]\t|", i+1);
        for(int j=0;j<nwordsize[i];++j) printf("*");
    }

    printf("\nPalabras mayores a 10: %d", nover);

    return EXIT_SUCCESS;

} // end main
