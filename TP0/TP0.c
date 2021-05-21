#define M_PI 3.14159265358979323846

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

//#include <libloaderapi.h>
#include <string.h>


FILE *fnames;
FILE *fdata;


float gaussian(float x, float u, float sigma){
    return (1/(sqrt(2*M_PI)*sigma))*exp(-1/(2*pow(sigma,2))*pow(x-u,2));
}

void test_gaussian(){
    for(float i=0;i<10;i=i+0.1){
        printf("%f, %f\n",i,gaussian(i,5,0.75));
    }
}

float rand_max(float x){
    float r= (float)rand()/(float)(RAND_MAX/x);
    //printf("%f\n", r);
    return r;
}
//a) y b)
void gen_datos(int d, int n, float c, char ej, FILE *names, FILE *data){
    fprintf(names,"0, 1. \n\n");
    for(int i=0;i<d;i++){
        fprintf(names,"d%d: continuous.\n",i);
    }

    
    for(int i=0;i<n;i++){
        for (int j=0; j < d; j++){

            //initializations
            int origin=0;
            float sigma=0;

            if (ej=='a'){
                origin= (i<n/2) ? 1 : -1;
                sigma= c*sqrt(d);
            }
            else if(ej=='b'){
                origin= (i<n/2) && (j==0) ? 1 : ((i>=n/2) && (j==0) ? -1 : 0);
                sigma= c;
            }
            else{
                printf("Error, ingrese un ejercicio que acepte su cantidad de argumentos\n");
                return;
            }

            //tengo que declarar el x e y afuera del do while para poder usarlos en la condicion
            float x=0;
            float y=0;
            do{
                x=rand_max(10*sigma)-5*sigma+origin;//va de -5 sigma a 5 sigma
                y= rand_max(1/(sqrt(2*M_PI)*sigma));
            }while(y>gaussian(x,origin,sigma));


            //printf("origen= %d, ",origin);
            //printf("%f, ",x);//guardo coordenada
            fprintf(data,"%f, ",x);
        }

        //printf("\n");
        //printf("%d\n",(i<n/2) ? 1 : 0);
        fprintf(data,"%d\n",(i<n/2) ? 1 : 0);

    }
}

//c)
//---------------------------------------------------------------------------
float l1(float theta){
    return (theta+M_PI)/(4*M_PI);
}

float l2(float theta){
    return (theta)/(4*M_PI);
}

int identificar_espiral(float theta, float r){
    if ((l2(theta)<= r) && (r <=l1(theta))){
        //printf("salio por 1: %f <= %f <= %f para theta=%f\n",l2(theta),r,l1(theta),theta);
        return 0;
    }
    else if ((l1(theta) <= r) && (r <= l2(theta + 2*M_PI))){
        //printf("salio por 2\n");
        return 1;
    }
    else if ((l2(theta + 2*M_PI) <= r) && (r <= l1(theta + 2*M_PI))){
        //printf("salio por 3\n");
        return 0;
    }
    else if (l1(theta + 2*M_PI) <= r){
        //printf("salio por 4\n");
        return 1;
    }
    else if (M_PI<=theta && r <= l1(theta - 2*M_PI)){
        //printf("salio por 6\n");
        return 0;
    }
    else if (r <= l2(theta)){
        //printf("salio por 5\n");
        return 1;
    }
    else return 2; //no deberia llegar nunca aca
}

//Funciones para loopear los radianes entre 0 y 2*PI
/* wrap x -> [0,max) */
double wrapMax(double x, double max)
{
    /* integer math: `(max + x % max) % max` */
    return fmod(max + fmod(x, max), max);
}
/* wrap x -> [min,max) */
double wrapMinMax(double x, double min, double max)
{
    return min + wrapMax(x - min, max - min);
}

/*
void generar_espirales(int n, FILE *names, FILE *data){
    fprintf(names,"0, 1. \n\nx: continuous. \ny: continuous.");
    for(int i=0;i<n;i++){
        float r = rand_max(1);
        float theta= rand_max(2*M_PI);
        int color=identificar_espiral(theta*M_PI,r);

        //conversion coords polares a coords cartesianas
        float x=r*cos(theta);
        float y=r*sin(theta);

        //printf("%f, %f, %d\n",x,y,color);
        printf("%f, %f\n",theta,r);
        fprintf(data,"%f, %f, %d\n",x,y,color);
    }

}
*/


void generar_espirales(int n, FILE *names, FILE *data){
    fprintf(names,"0, 1. \n\nx: continuous. \ny: continuous.");
    for(int i=0;i<n;i++){
        float x = rand_max(2)-1;
        float y = rand_max(2)-1;

        float r = sqrt(x*x+y*y);
        while(r>1){
            x = rand_max(2)-1;
            y = rand_max(2)-1;

            r = sqrt(x*x+y*y);
        }
        float theta = wrapMinMax(atan2(y,x), 0, 2*M_PI);


        int color=identificar_espiral(theta,r);

        //conversion coords polares a coords cartesianas
        printf("x=%f, y=%f, angulo=%f, r=%f\n",x,y,theta,r);
        //printf("%f, %f, %d\n",x,y,color);
        fprintf(data,"%f, %f, %d\n",x,y,color);
    }

}

//---------------------------------------------------------------------------

//No es necesario pero puede venir util a futuro
//--------------------------------------------------------
/*
void get_paths(char *names, char *data){
    char absolutePath[256];
    size_t len = sizeof(absolutePath);

    //Solo funciona en windows, si se corre en linux usar una funcion como readlink(), o no utilizar la funcion get_paths y escribir manualmente los paths de names y data que quiera utilizar
    int bytes = GetModuleFileName(NULL, absolutePath, len);

    for(int i=strlen(absolutePath);i>0;i--){
        if (absolutePath[i]=='\\'){
            absolutePath[i+1]='\0';
            break;
        }
    }

    strcpy(names, absolutePath);
    strcpy(data, absolutePath);
    strncat(names, "ej.names", 9);//ver si lo hago con 8 sino
    strncat(data, "ej.data", 8);//ver si lo hago con 7 sino

    printf("namespath: %s \ndatapath: %s\n",names,data);

    return;
}
*/
//--------------------------------------------------------

//gcc TP0.c -lm

//./main ej n d c
//./main c n

//Donde:
//- ej es un caracter que indica el ejercicio, es decir es igual a 'a','b' o 'c'
//- n es el numero de muestras generadas
//- d es la dimension de las muestras
//- c es el valor usado en la desviascion estandar

int main(int argc, char *argv[]) {

    char ej='a';
    int n=200;
    int d=2;
    float c=0.75;

    srand((unsigned)time(NULL));


    //char namesPath[256];
    //char dataPath[256];
    //get_paths(namesPath, dataPath);

    char namesPath[9]="ej.names";
    char dataPath[8]="ej.data";


    //printf("Ejercicio %c ejecutado con los parametros n=%d, d=%d, c=%f\n",ej,n,d,c);

    if (argc==5){
        fnames = fopen(namesPath,"w");
        fdata = fopen(dataPath,"w");

        ej=argv[1][0];
        n=atoi(argv[2]);
        d=atoi(argv[3]);
        c=atof(argv[4]);

        gen_datos(d,n,c,ej,fnames, fdata);

        fclose(fnames);
        fclose(fdata);

    }
    else if(argc==3){
        fnames = fopen(namesPath,"w");
        fdata = fopen(dataPath,"w");

        ej=argv[1][0];
        n=atoi(argv[2]);

        if(ej=='c'){
            generar_espirales(n, fnames, fdata);
        }
        else{
            printf("Error, se pasaron %d en lugar de los 2 argumentos requeridos\n",argc-1);
        }

        fclose(fnames);
        fclose(fdata);
    }
    else{
        printf("Error, se pasaron %d en lugar de los 4 argumentos requeridos\n",argc-1);
    }

}