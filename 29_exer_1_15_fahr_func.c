/*Imprime la tabla Fahrenheit-Celsius para fahr=0,20,...,300, usando una función*/
#include <stdio.h>

#define EXIT_SUCCESS    0

int to_celsius(int);

int main() {
    int fahr,lower,upper,step;

    lower=0;    /*Límite inferior*/
    upper=300;  /*Límite superior*/
    step=20;    /*Tamaño del incremento*/
    fahr=lower;

    while(fahr<=upper) {
        printf("%d\t%d\n", fahr,to_celsius(fahr));
        fahr=fahr+step;
    } // end while

    return EXIT_SUCCESS;
} // end main

int to_celsius(int fahr) {
    return 5*(fahr-32)/9;
} // end to_celsius function









 
