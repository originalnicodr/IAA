/*
k_nn.c : Clasificación de k-primeros-vecinos
Formato de datos: c4.5
La clase a predecir tiene que ser un numero comenzando de 0: por ejemplo, para 3 clases, las clases deben ser 0,1,2

PMG - Ultima revision: 
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define LOW 1.e-14                 /*Minimo valor posible para una probabilidad*/
#define PI  3.141592653

/*defino una estructura para el clasificador naive bayes*/
struct KF {
    int N_IN;           /*Total numbre of inputs*/
    int N_Class;        /*Total number of classes (outputs)*/

    int SEED;           /* semilla para la funcion rand(). Los posibles valores son:*/
                    /* SEED: -1: No mezclar los patrones: usar los primeros PR para entrenar
                                 y el resto para validar.Toma la semilla del rand con el reloj.
                              0: Seleccionar semilla con el reloj, y mezclar los patrones.
                             >0: Usa el numero leido como semilla, y mezcla los patrones. */

    int N_K;        /* Number of neighbours */



    
};

struct DATOS {
    char *filename;      /* nombre del archivo para leer los datos */
    int N_IN;             /* N1: dimensiones de entrada */
    int PTOT;           /* cantidad TOTAL de patrones en el archivo .data */
    int PR;             /* cantidad de patrones de ENTRENAMIENTO */
                        /* cantidad de patrones de VALIDACION: PTOT - PR*/
    int PTEST;          /* cantidad de patrones de TEST (archivo .test) */

    /*matrices de datos*/
    float **data;                     /* train data */
    float **test;                     /* test  data */
    int *pred;                        /* salidas predichas ES UNA CLASE*/
    int *seq;      	       	      /* indice de acceso con la sequencia de presentacion de los patrones*/
};

int CONTROL;        /* nivel de verbosity */


/* -------------------------------------------------------------------------- */
/*define_matrix_datos: reserva espacio en memoria para todas las matrices de datos declaradas.
  Todas las dimensiones son leidas del archivo .net en la funcion arquitec()  */
/* -------------------------------------------------------------------------- */
int define_matrix_datos(struct DATOS *datos){
  int i,max;
    
  datos->seq=(int *)calloc(datos->PTOT,sizeof(int));
  for(i=0;i<datos->PTOT;i++) datos->seq[i]=i;  /* inicializacion del indice de acceso a los datos */
  
  datos->data=(float **)calloc(datos->PTOT,sizeof(float *));
  if(datos->PTEST) datos->test=(float **)calloc(datos->PTEST,sizeof(float *));

  if(datos->PTOT>datos->PTEST) max=datos->PTOT;
  else max=datos->PTEST;
  datos->pred=(int *)calloc(max,sizeof(int));

  if(datos->seq==NULL||datos->data==NULL||(datos->PTEST&&datos->test==NULL)||datos->pred==NULL) return 1;
  
  for(i=0;i<datos->PTOT;i++){
    datos->data[i]=(float *)calloc(datos->N_IN+1,sizeof(float));
	if(datos->data[i]==NULL) return 1;
  }
  for(i=0;i<datos->PTEST;i++){
    datos->test[i]=(float *)calloc(datos->N_IN+1,sizeof(float));
	if(datos->test[i]==NULL) return 1;
  }

  return 0;
}

/* ---------------------------------------------------------------------------------- */
/*arquitec: Lee el archivo .kf e inicializa el algoritmo en funcion de los valores leidos
  filename es el nombre del archivo .kf (sin la extension) */
/* ---------------------------------------------------------------------------------- */
int arquitec(char *filename,struct KF *kf,struct DATOS *datos, int k){
  FILE *b;
  char filepat[100];
  int i,j;
  /*bandera de error*/
  int error;

  time_t t;

  /*Paso 1:leer el archivo con la configuracion*/
  sprintf(filepat,"%s.kf",filename);
  b=fopen(filepat,"r");
  error=(b==NULL);
  if(error){
    printf("Error al abrir el archivo de parametros\n");
    return 1;
  }

  /* Dimensiones */
  fscanf(b,"%d",&kf->N_IN);
  fscanf(b,"%d",&kf->N_Class);
  datos->N_IN=kf->N_IN;
  kf->N_K=k;

  /* Archivo de patrones: datos para train y para validacion */
  fscanf(b,"%d",&datos->PTOT);
  fscanf(b,"%d",&datos->PR);
  fscanf(b,"%d",&datos->PTEST);

  /* Semilla para la funcion rand()*/
  fscanf(b,"%d",&kf->SEED);

  /* Nivel de verbosity*/
  fscanf(b,"%d",&CONTROL);

  fclose(b);


  /*Paso 2: Definir matrices para datos (e inicializarlos)*/
  error=define_matrix_datos(datos);
  if(error){
    printf("Error en la definicion de matrices de datos\n");
    return 1;
  }

  
  /* Chequear semilla para la funcion rand() */
  if(kf->SEED==0) kf->SEED=time(&t);

  /*Imprimir control por pantalla*/
  printf("\nNaive Bayes con distribuciones normales:\nCantidad de entradas:%d",kf->N_IN);
  printf("\nCantidad de clases:%d",kf->N_Class);
  printf("\nArchivo de patrones: %s",filename);
  printf("\nCantidad total de patrones: %d",datos->PTOT);
  printf("\nCantidad de patrones de entrenamiento: %d",datos->PR);
  printf("\nCantidad de patrones de validacion: %d",datos->PTOT-datos->PR);
  printf("\nCantidad de patrones de test: %d",datos->PTEST);
  printf("\nSemilla para la funcion rand(): %d",kf->SEED); 

  return 0;
}
/* -------------------------------------------------------------------------------------- */
/*read_data: lee los datos de los archivos de entrenamiento(.data) y test(.test)
  filenamepasado es el nombre de los archivos (sin extension)
  La cantidad de datos y la estructura de los archivos fue leida en la funcion arquitec()
  Los registros en el archivo pueden estar separados por blancos ( o tab ) o por comas    */
/* -------------------------------------------------------------------------------------- */
int read_data(char *filenamepasado,struct DATOS *datos){

  FILE *fpat;
  char filepat[100];
  float valor;
  int i,k,separador;
  /*bandera de error*/
  int error;

  sprintf(filepat,"%s.data",filenamepasado);
  fpat=fopen(filepat,"r");
  error=(fpat==NULL);
  if(error){
    printf("Error al abrir el archivo de datos\n");
    return 1;
  }

  datos->filename=filenamepasado;
  
  if(CONTROL>1) printf("\n\nDatos de entrenamiento:");

  for(k=0;k<datos->PTOT;k++){
	 if(CONTROL>1) printf("\nP%d:\t",k);
 	 for(i=0;i<datos->N_IN+1;i++){
	   fscanf(fpat,"%f",&valor);
	   datos->data[k][i]=valor;
	   if(CONTROL>1) printf("%f\t",datos->data[k][i]);
	   separador=getc(fpat);
	   if(separador!=',') ungetc(separador,fpat);
  	 }
  }
  fclose(fpat);

  if(!datos->PTEST) return 0;

  sprintf(filepat,"%s.test",filenamepasado);
  fpat=fopen(filepat,"r");
  error=(fpat==NULL);
  if(error){
    printf("Error al abrir el archivo de test\n");
    return 1;
  }

  if(CONTROL>1) printf("\n\nDatos de test:");

  for(k=0;k<datos->PTEST;k++){
	 if(CONTROL>1) printf("\nP%d:\t",k);
         for(i=0;i<datos->N_IN+1;i++){
	   fscanf(fpat,"%f",&valor);
	   datos->test[k][i]=valor;
	   if(CONTROL>1) printf("%f\t",datos->test[k][i]);
	   separador=getc(fpat);
	   if(separador!=',') ungetc(separador,fpat);
         }
  }
  fclose(fpat);

  return 0;

}

/* ------------------------------------------------------------ */
/* shuffle: mezcla el vector seq al azar.
   El vector seq es un indice para acceder a los patrones.
   Los patrones mezclados van desde seq[0] hasta seq[hasta-1]
   Esto permite separar la parte de validacion de la de train   */
/* ------------------------------------------------------------ */
void shuffle(int hasta,struct DATOS *datos){
   float x;
   int tmp;
   int top,select;

   top=hasta-1;
   while (top > 0) {
	x = (float)rand();
	x /= (RAND_MAX+1.0);
	x *= (top+1);
	select = (int)x;
	tmp = datos->seq[top];
	datos->seq[top] = datos->seq[select];
	datos->seq[select] = tmp;
	top --;
   }
  if(CONTROL>3) {printf("End shuffle\n");fflush(NULL);}
}

/* ------------------------------------------------------------------- */
/*Prob:Calcula la probabilidad de obtener el valor x para el input feature y la clase
  Aproxima las probabilidades por distribuciones normales */
/* ------------------------------------------------------------------- */
float prob(struct KF *kf,float x,int feature,int clase)  { //feature es a que coordenada del input el x pertenece

  return 1.0;
  //return fmax(LOW,prob);  
}

/* ------------------------------------------------------------------- */
/*dist:Calcula la distancia euclediana dados dos puntos*/
/* ------------------------------------------------------------------- */
float dist(int d, float* x1, float* x2){
  float res=0;
  for(int i=0;i<d;i++){
    res+= pow(x1[i]-x2[i],2);
  }
  return sqrt(res);
}


//Bubble sort
//---------------------------------------------------------------------
void swap(int *a, int *b){ 
  int temp = *a; 
  *a = *b; 
  *b = temp; 
}  

void bubbleSort(int array[], int n, float **datos, float* input, int d){ 
  int i, j; 
  for (i = 0; i < n-1; i++) for (j = 0; j < n-i-1; j++) if (dist(d,datos[array[j]],input) > dist(d,datos[array[j+1]],input)) 
  swap(&array[j], &array[j+1]); 
}   
//---------------------------------------------------------------------


//Merge sort
//---------------------------------------------------------------------

// Merge Function
void merge(int arr[], int l, int m, int r, float **datos, float* input, int d){
  int i, j, k;
  int n1 = m - l + 1;
  int n2 = r - m;
  // Create temp arrays
  int L[n1], R[n2];
  // Copy data to temp array
  for (i = 0; i < n1; i++) L[i] = arr[l + i];
  for (j = 0; j < n2; j++) R[j] = arr[m + 1+ j];
  // Merge the temp arrays
  i = 0;
  j = 0;
  k = l;
  while (i < n1 && j < n2){
    if (dist(d,datos[L[i]],input) <= dist(d,datos[R[j]],input)){
      arr[k] = L[i];
      i++;
    }
    else{
      arr[k] = R[j];
      j++;
    }
    k++;
  }
  // Copy the remaining elements of L[]
  while (i < n1){
    arr[k] = L[i];
    i++;
    k++;
  }
  // Copy the remaining elements of R[]
  while (j < n2){
    arr[k] = R[j];
    j++;
    k++;
  }
}

void mergeSort(int arr[], int l, int r, float **datos, float* input, int d){
  if (l < r){
    // Finding mid element
    int m = l+(r-l)/2;
    // Recursively sorting both the halves
    mergeSort(arr, l, m, datos, input, d);
    mergeSort(arr, m+1, r, datos, input, d);
  
    // Merge the array
    merge(arr, l, m, r, datos, input, d);
  }
}

//---------------------------------------------------------------------

void sorttest(struct KF *kf,struct DATOS *datos){
  float test[2]={0, 0};

  int orden[datos->PR];

  //inicializacion
  for(int k=0;k<datos->PR;k++){
    orden[k]=k;
  }
  
  mergeSort(orden,0,datos->PR-1,datos->data,test,kf->N_IN);
  //bubbleSort(orden, datos->PR-1,datos->data, test, kf->N_IN);

  for(int i=0;i<datos->PR;i++){
    printf("distancia: %f\n",dist(kf->N_IN,datos->data[orden[i]],test));
  }
}




/* ------------------------------------------------------------------------------ */
/*output: calcula la probabilidad de cada clase dado un vector de entrada *input
  devuelve la clase con mayor probabilidad                                               */
/* ------------------------------------------------------------------------------ */
int output(struct KF *kf,struct DATOS *datos,float *input){
  
  int i,k; 	
  float prob_de_clase;
  float max_prob=-1e40;

  int orden[datos->PR];

  //inicializacion
  for(k=0;k<datos->PR;k++){
    orden[k]=k;
  }
  
  float cont[kf->N_Class];

  //inicializacion
  for(k=0;k<kf->N_Class;k++){
    cont[k]=0;
  }
  
  //bubbleSort(orden, datos->PR-1,datos->data, input, kf->N_IN);
  mergeSort(orden,0,datos->PR-1,datos->data,input,kf->N_IN);

  

  for(i=0;i<kf->N_K;i++){
    cont[(int)datos->data[orden[i]][datos->N_IN]]++;
  }

  int maxprob=0;
  for(k=0;k<kf->N_Class;k++){
    cont[k]/=kf->N_K;
    if(cont[maxprob]<cont[k]) maxprob=k;
  }
  
  return maxprob;
}
/* ------------------------------------------------------------------------------ */
/*evaluar: calcula las clases predichas para un conjunto de datos
  la matriz S tiene que tener el formato adecuado ( definido en arquitec() )
  pat_ini y pat_fin son los extremos a tomar en la matriz
  usar_seq define si se accede a los datos directamente o a travez del indice seq
  los resultados (las propagaciones) se guardan en la matriz seq                  */
/* ------------------------------------------------------------------------------ */
float evaluar(struct KF *kf,struct DATOS *datos,float **S,int pat_ini,int pat_fin,int usar_seq){



  float mse=0.0;
  int nu;
  int patron;
  
  for (patron=pat_ini; patron < pat_fin; patron ++) {

   /*nu tiene el numero del patron que se va a presentar*/
    if(usar_seq) nu = datos->seq[patron];
    else         nu = patron;

    /*clase MAP para el patron nu*/
    datos->pred[nu]=output(kf,datos,S[nu]);

    /*actualizar error*/
    if(S[nu][kf->N_IN]!=(float)datos->pred[nu]) mse+=1.;
  }
    

  mse /= ( (float)(pat_fin-pat_ini));

  if(CONTROL>3) {printf("End prop\n");fflush(NULL);}

  return mse;
}

/* --------------------------------------------------------------------------------------- */
/*train: ajusta los parametros del algoritmo a los datos de entrenamiento
         guarda los parametros en un archivo de control
         calcula porcentaje de error en ajuste y test                                      */
/* --------------------------------------------------------------------------------------- */
int train(struct KF *kf,struct DATOS *datos){

  //no hay entrenamiento, pero tengo que devolver error de entrenamiento

  float train_error;
  float test_error;
  float valid_error;
  FILE *fpredic;
  char filepat[100];
  int error;
  

  /*calcular error de entrenamiento*/
  train_error=evaluar(kf,datos,datos->data,0,datos->PR,1);
  /*calcular error de validacion; si no hay, usar mse_train*/
  if(datos->PR==datos->PTOT) valid_error=train_error;
  else         valid_error=evaluar(kf,datos,datos->data,datos->PR,datos->PTOT,1);
  /*calcular error de test (si hay)*/
  if (datos->PTEST>0) test_error =evaluar(kf,datos,datos->test,0,datos->PTEST,0);
  else         test_error =0.;
  /*mostrar errores*/
  printf("\nFin del entrenamiento.\n\n");
  printf("Errores:\nEntrenamiento:%f%%\n",train_error*100.);
  printf("Validacion:%f%%\nTest:%f%%\n",valid_error*100.,test_error*100.);
  if(CONTROL) fflush(NULL);

  /* archivo de predicciones */
  sprintf(filepat,"%s.predic",datos->filename);
  fpredic=fopen(filepat,"w");
  error=(fpredic==NULL);
  if(error){
    printf("Error al abrir archivo para guardar predicciones\n");
    return 1;
  }
  for(int k=0; k < datos->PTEST ; k++){
    for(int i = 0 ;i< datos->N_IN;i++) fprintf(fpredic,"%f\t",datos->test[k][i]);
    fprintf(fpredic,"%d\n",datos->pred[k]);
  }
  fclose(fpredic);

  return 0;
}

/* ----------------------------------------------------------------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */
int main(int argc, char **argv){

  /*bandera de error*/
  int error;
 
  if(argc!=3){
    printf("Modo de uso: kf <filename> <k>\ndonde filename es el nombre del archivo (sin extension) y k el numero de vecinos\n");
    return 0;
  }

  struct KF kf;
  struct DATOS datos;

  int k=atoi(argv[2]);
  
  /* defino la estructura*/
  error=arquitec(argv[1],&kf,&datos,k);
  if(error){
    printf("Error en la definicion del clasificador\n");
    return 1;
  }

  /* leo los datos */
  error=read_data(argv[1],&datos);                 
  if(error){
    printf("Error en la lectura de datos\n");
    return 1;
  }

  //sorttest(&kf,&datos);
  // entreno la red
  error=train(&kf,&datos);                  
  if(error){
    printf("Error en el entrenamiento\n");
    return 1;
  }

  return 0;

}
/* ----------------------------------------------------------------------------------------------------- */

