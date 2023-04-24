
/*
 * Author: AaronPI
 * Mail: aaron3.1416@gmail.com
 * Problema 1-14: Escriba un programa que imprima el histograma de las
 * frecuencias con que se presentan diferentes caracteres leídos a la entrada.
 */

#include <stdio.h>

#define EXIT_SUCCESS            0
#define MAX_ALPHABET_LENGTH     25  /* De A=65 a Z=90, y de a=87 hasta z=122 */
#define MAX_DIGITS_LENGTH       10  /* De 0=48 a 9=57 */

int main() {

    int nnums[MAX_DIGITS_LENGTH];
    char nupper[MAX_ALPHABET_LENGTH];
    char nlower[MAX_ALPHABET_LENGTH];
    int nwhite = 0;
    int nothers = 0;
    char c;

    for(int a=0;a<MAX_DIGITS_LENGTH;++a) nnums[a] = 0;
    for(int b=0;b<=MAX_ALPHABET_LENGTH;++b) nupper[b] = nlower[b] = 0;

    while((c=getchar()) != EOF) {
        if(c==' ' || c=='\n' || c=='\t') {
          ++nwhite;
        } else if(c >= '0' && c<='9') {
            ++nnums[c-'0'];
        } else if(c >= 'A' && c<='Z') {
            ++nupper[c-'A'];
        } else if(c >= 'a' && c<='z') {
            ++nlower[c-'a'];
        } else {
          ++nothers;
        } // end if-else
    } // end while

    printf("\nEstadísticas: ");

    int i, j;
    i = j = 0;
    printf("\n");
    for(i=0;i<MAX_DIGITS_LENGTH;++i) {
        printf("\n\t[%d]\t|", i);
        for(j=0;j<nnums[i];++j) printf("*");
    } // end for

    printf("\n");
    for(i=0;i<MAX_ALPHABET_LENGTH + 1;++i) {
        if(nupper[i] != 0) {
            printf("\n\t[%c]\t|", i + 'A');
            for(j=0;j<nupper[i];++j) printf("*");
        } // end if
    } // end for

    printf("\n");
    for(i=0;i<MAX_ALPHABET_LENGTH + 1;++i) {
        if(nlower[i] != 0) {
            printf("\n\t[%c]\t|", i + 'a');
            for(j=0;j<nlower[i];++j) printf("*");
        } // end if
    } // end for

    printf("\n\nEspacios en blanco: %d", nwhite);
    printf("\nOtros caracteres: %d", nothers);

    return EXIT_SUCCESS;

} // end main

