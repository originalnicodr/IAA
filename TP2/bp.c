/*
BP: Entrenamiento de redes neuronales feedforward de 3 capas (entrada - oculta - salida).
Algoritmo: Backpropagation estocastico.
Capa intermedia con unidades sigmoideas.
Salidas con unidades lineales.

PMG - Ultima revision: 04/2019
*/

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

/*Defino la funcion sigmoidea*/
#define sigmoid(h) 1.0/(1.0+exp(-h))

/*defino una estructura para la red neuronal de 3 capas, entrada, oculta, salida*/
struct MLP {

    /*parametros de la red y el entrenamiento*/
    int N1;             /* N1: NEURONAS EN CAPA DE ENTRADA */
    int N2;             /* N2: NEURONAS EN CAPA INTERMEDIA */
    int N3;             /* N3: NEURONAS EN CAPA DE SALIDA  */

    int ITER;           /* Total de Iteraciones*/
    float ETA;          /* learning rate */
    float u;            /* Momentum */
    
    int NERROR;         /* guarda mediciones de error cada NERROR iteraciones */

    int WTS;            /* numero de archivo de sinapsis inicial
                           WTS=0 implica empezar la red con valores de sinapsis al azar
                           WTS>0 implica leer los pesos desde un archivo con un entrenamiento anterior*/

    int SEED;           /* semilla para la funcion rand(). Los posibles valores son:*/
                        /* SEED: -1: No mezclar los patrones: usar los primeros PR para entrenar
                                 y el resto para validar.Toma la semilla del rand con el reloj.
                              0: Seleccionar semilla con el reloj, y mezclar los patrones.
                             >0: Usa el numero leido como semilla, y mezcla los patrones. */


    /*matrices de la red*/
    float **w1,**w2;	      	  /* pesos entre neuronas */
    float *grad2,*grad3;	  /* gradiente en cada unidad */
    float **dw1,**dw2;	          /* correccion a cada peso   */
    float *x1,*x2,*x3; 	       	  /* activaciones de cada capa   */
    float *target;    	          /* valor correcto de la salida */
};

struct DATOS {
    char *filename;      /* nombre del archivo para leer los datos */
    int N1;             /* N1: dimensiones de entrada */
    int N3;             /* N3: dimensiones de SALIDA  */
    int PTOT;           /* cantidad TOTAL de patrones en el archivo .data */
    int PR;             /* cantidad de patrones de ENTRENAMIENTO */
                        /* cantidad de patrones de VALIDACION: PTOT - PR*/
    int PTEST;          /* cantidad de patrones de TEST (archivo .test) */

    /*matrices de datos*/
    float **data;                     /* train data */
    float **test;                     /* test  data */
    float **pred;                     /* salidas predichas */
    int *seq;      	       	      /* indice de acceso con la sequencia de presentacion de los patrones*/
};


int CONTROL;        /* nivel de verbosity: 0 -> solo resumen, 1 -> 0 + pesos, 2 -> 1 + datos*/


/* ------------------------------------------------------------------------------- */
/*unif_rnd:         
  genero un numero al azar en [-max , max]                                         */
/* ------------------------------------------------------------------------------- */

float unif_rnd(float max){
    return  (rand()/(RAND_MAX/2.0) - 1.0) * max;
}

/* ------------------------------------------------------------------------------- */
/*sinapsis_rnd: Inicializa al azar los valores de todas las sinapsis                 
  Los valores se toman entre [-max , max]
  SEED contiene la semilla del rand(). Si SEED es <=0 se toma como semilla el reloj*/
/* ------------------------------------------------------------------------------- */
void sinapsis_rnd(float max,struct MLP *nn){
  int i,j,k;
  float x;
  time_t t; 

  /* Semilla para la funcion rand() */
  if(nn->SEED<0) srand((unsigned) time(&t));   
  else{
    if(nn->SEED==0) nn->SEED=time(&t);
    srand((unsigned)nn->SEED);
  }  

  /*primer capa*/
  for (j = 1; j <= nn->N2; j ++) for (i = 0; i <= nn->N1; i ++) nn->w1[j][i] = unif_rnd(max);


  /*segunda capa*/
  for (j = 1; j <= nn->N3; j ++) for (i = 0; i <= nn->N2; i ++) nn->w2[j][i] = unif_rnd(max);

}
/* ------------------------------------------------------------------- */
/*sinapsis_save: guarda valores de las sinapsis en un archivo.
  El nombre del archivo de salida es (numero).wts, donde numero es un entero.*/
/* ------------------------------------------------------------------- */
int sinapsis_save(int numero,struct MLP *nn)  {
  FILE *fp;
  int i,j,largo;
  char p[13];
  
  sprintf(p, "%d", numero);
  largo=strlen(p);
  p[largo]='.';
  p[largo+1]='w';
  p[largo+2]='t';
  p[largo+3]='s';
  p[largo+4]='\0';
  
  if((fp=fopen(p,"w"))==NULL) return 1;
  
  /*primer capa*/
  for(j=1;j<=nn->N2;j++)  for(i=0;i<=nn->N1;i++) fprintf(fp,"%f\n",nn->w1[j][i]);
  
  /*segunda capa*/
  for (i=1;i<=nn->N3;i++) for(j=0;j<=nn->N2;j++) fprintf(fp,"%f\n",nn->w2[i][j]);
  
  fclose(fp);

  return 0;
}
/* --------------------------------------------------------------------- */
/*sinapsis_read: lee valores de las sinapsis desde un archivo.
  El nombre del archivo de entrada es (numero).wts , donde numero es un entero.*/
/* --------------------------------------------------------------------- */
int sinapsis_read(int numero,struct MLP *nn)  {
  FILE *fp;
  int i,j,largo;
  char p[13];

  sprintf(p, "%d", numero);
  largo=strlen(p);
  p[largo]='.';
  p[largo+1]='w';
  p[largo+2]='t';
  p[largo+3]='s';
  p[largo+4]='\0';
  if((fp=fopen(p,"r"))==NULL) return 1;
  
  /*primera capa*/
  for(j=1;j<=nn->N2;j++) for(i=0;i<=nn->N1;i++) fscanf(fp,"%f",&nn->w1[j][i]);
  
  /*segunda capa*/
  for (i=1;i<=nn->N3;i++) for(j=0;j<=nn->N2;j++) fscanf(fp,"%f",&nn->w2[i][j]);
  
  fclose(fp);

  return 0;
}
/* -------------------------------------------------------------------------- */
/*define_matrix_mlp: reserva espacio en memoria para todas las matrices declaradas
  en la estructura de la red. Inicializa a 0 los valores necesarios.
  Todas las dimensiones son leidas del archivo .net en la funcion arquitec()  */
/* -------------------------------------------------------------------------- */
int define_matrix_mlp(struct MLP *nn){

  int i,j,k;
  
  nn->x1=(float *)calloc(nn->N1+1,sizeof(float));
  nn->x2=(float *)calloc(nn->N2+1,sizeof(float));
  nn->x3=(float *)calloc(nn->N3+1,sizeof(float));
  nn->grad2=(float *)calloc(nn->N2+1,sizeof(float));
  nn->grad3=(float *)calloc(nn->N3+1,sizeof(float));
  nn->target=(float *)calloc(nn->N3+1,sizeof(float));
  if(nn->x1==NULL||nn->x2==NULL||nn->x3==NULL||nn->grad2==NULL||nn->grad3==NULL||nn->target==NULL) return 1;
  
  nn->w1= (float **)calloc(nn->N2+1,sizeof(float *));
  nn->w2= (float **)calloc(nn->N3+1,sizeof(float *));
  nn->dw1= (float **)calloc(nn->N2+1,sizeof(float *));
  nn->dw2= (float **)calloc(nn->N3+1,sizeof(float *));

  for(i=0;i<=nn->N2;i++){
    nn->w1[i]=(float *)calloc(nn->N1+1,sizeof(float));
    nn->dw1[i]=(float *)calloc(nn->N1+1,sizeof(float));
	if(nn->w1[i]==NULL||nn->dw1[i]==NULL) return 1;
  }
  for(i=0;i<=nn->N3;i++){
    nn->w2[i]=(float *)calloc(nn->N2+1,sizeof(float));
    nn->dw2[i]=(float *)calloc(nn->N2+1,sizeof(float));
	if(nn->w2[i]==NULL||nn->dw2[i]==NULL) return 1;
  }
  
  /* Inicializacion de todas las matrices necesarias */
  for (j = 1; j <= nn->N2; j++) {
    nn->grad2[j] = 0.0;
    for (k = 0; k <= nn->N1; k++) nn->dw1[j][k]=0.0;
  }
  for (i = 1; i <= nn->N3; i ++) {
    nn->grad3[i] =0.0;
    for (j = 0; j <= nn->N2; j++) nn->dw2[i][j]=0.0;
  }
  nn->x1[0]=-1.0;                    /* bias de las unidades de cada hilera             */
  nn->x2[0]=-1.0;


  return 0;
}

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
  datos->pred=(float **)calloc(max,sizeof(float *));

  if(datos->seq==NULL||datos->data==NULL||(datos->PTEST&&datos->test==NULL)||datos->pred==NULL) return 1;
  
  for(i=0;i<datos->PTOT;i++){
    datos->data[i]=(float *)calloc(datos->N1+datos->N3+1,sizeof(float));
	if(datos->data[i]==NULL) return 1;
  }
  for(i=0;i<datos->PTEST;i++){
    datos->test[i]=(float *)calloc(datos->N1+datos->N3+1,sizeof(float));
	if(datos->test[i]==NULL) return 1;
  }
  for(i=0;i<max;i++){
    datos->pred[i]=(float *)calloc(datos->N3+1,sizeof(float));
	if(datos->pred[i]==NULL) return 1;
  }

  return 0;
}
/* ---------------------------------------------------------------------------------- */
/*arquitec: Lee el archivo .net e inicializa la red en funcion de los valores leidos
  filename es el nombre del archivo .net (sin la extension) */
/* ---------------------------------------------------------------------------------- */
int arquitec(char *filename,struct MLP *nn,struct DATOS *datos){
  FILE *b;
  char filepat[100];
  int i,j;
  /*bandera de error*/
  int error;

  /*Paso 1:leer el archivo con la configuracion*/
  sprintf(filepat,"%s.net",filename);
  b=fopen(filepat,"r");
  error=(b==NULL);
  if(error){
    printf("Error al abrir el archivo de parametros\n");
    return 1;
  }

  /* Estructura de la red */
  fscanf(b,"%d",&nn->N1);
  fscanf(b,"%d",&nn->N2);
  fscanf(b,"%d",&nn->N3);
  datos->N1=nn->N1;
  datos->N3=nn->N3;
  
  /* Archivo de patrones: datos para train y para validacion */
  fscanf(b,"%d",&datos->PTOT);
  fscanf(b,"%d",&datos->PR);
  fscanf(b,"%d",&datos->PTEST);

  /* Parametros para el entrenamiento */
  fscanf(b,"%d",&nn->ITER);
  fscanf(b,"%f",&nn->ETA);
  fscanf(b,"%f",&nn->u);

  /* Datos para la salida de resultados */
  fscanf(b,"%d",&nn->NERROR);

  /* Inicializacion de las sinapsis - Azar o Archivo */
  fscanf(b,"%d",&nn->WTS);

  /* Semilla para la funcion rand()*/
  fscanf(b,"%d",&nn->SEED);

  /* Nivel de verbosity - variable gloabal*/
  fscanf(b,"%d",&CONTROL);

  fclose(b);



  /*Paso 2: Definir matrices para datos y pesos*/
  error=define_matrix_mlp(nn);
  if(error){
    printf("Error en la definicion de matrices de la red\n");
    return 1;
  }
  error=define_matrix_datos(datos);
  if(error){
    printf("Error en la definicion de matrices de datos\n");
    return 1;
  }



  /*Paso 3:leer sinapsis desde archivo o iniciar al azar*/
  if(nn->WTS!=0) error=sinapsis_read(nn->WTS,nn);
  else sinapsis_rnd(0.1,nn);
  if(error){
    printf("Error en la lectura de los pesos desde archivo\n");
    return 1;
  }


  /*Imprimir control por pantalla*/
  printf("\nArquitectura de la red: %d:%d:%d",nn->N1,nn->N2,nn->N3);
  printf("\nArchivo de patrones: %s",filename);
  printf("\nCantidad total de patrones: %d",datos->PTOT);
  printf("\nCantidad de patrones de entrenamiento: %d",datos->PR);
  printf("\nCantidad de patrones de validacion: %d",datos->PTOT-datos->PR);
  printf("\nCantidad de patrones de test: %d",datos->PTEST);
  printf("\nEpocas: %d",nn->ITER);
  printf("\nLearning Rate: %f",nn->ETA);
  printf("\nMomentum: %f",nn->u);
  printf("\nFrecuencia para grabar resultados: %d EPOCAS",nn->NERROR);
  printf("\nArchivo con sinapsis iniciales: %d.WTS",nn->WTS); 
  printf("\nSemilla para la funcion rand(): %d",nn->SEED); 
  if(CONTROL){
    printf("\nSINAPSIS:\n");
    for(j=1;j<=nn->N2;j++) {
      for(i=0;i<=nn->N1;i++) printf("%f\n",nn->w1[j][i]);
    }
    for(j=1;j<=nn->N3;j++) {
      for(i=0;i<=nn->N2;i++) printf("%f\n",nn->w2[j][i]);
    }   
  }

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
	 datos->data[k][0]=-1.0;
 	 for(i=1;i<=datos->N1+datos->N3;i++){
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
	 datos->test[k][0]=-1.0;
         for(i=1;i<=datos->N1+datos->N3;i++){
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
/*forward: propaga el vector de valores de entrada X1[] en la red
  En los vectores x2[] y x3[] quedan las activaciones correspondientes */
/* ------------------------------------------------------------------- */
void forward(struct MLP *nn){
  int i,j,k;
  float h;
  
  /*calcular los X2*/
  for(j=1; j<=nn->N2; j++){
    h = 0.0;
    for(k=0; k<=nn->N1; k++) h += nn->w1[j][k] * nn->x1[k];
    nn->x2[j] = sigmoid(h);
  }
  
  /*calcular los x3*/
  for(i=1; i<=nn->N3; i++){
    h = 0.0;
    for(j=0; j<=nn->N2; j++) h += nn->w2[i][j] * nn->x2[j];
    nn->x3[i] = h ;  /* activacion lineal*/
  }
}
/* ------------------------------------------------------------------------------ */
/*evaluar: calcula los valores de salida de la red para un conjunto de datos
  la matriz S es alguno de los datos validos
  pat_ini y pat_fin son los extremos a tomar en la matriz de datos
  usar_seq define si se accede a los datos directamente o a travez del indice seq
  los resultados (las propagaciones) se guardan en la matriz pred de los datos
  mse devuelve el error continuo
  discrete_error devuelve el error de clasificacion, error discreto                     */
/* ------------------------------------------------------------------------------ */
int evaluar(struct MLP *nn,struct DATOS *datos,float **S,int pat_ini,int pat_fin,int usar_seq, float *mse, float *discrete_error){

  int i,patron,nu;
  
  *discrete_error=*mse=0.0;
  
  for (patron=pat_ini; patron<pat_fin; patron++) { 

   /*nu tiene el numero del patron que se va a presentar*/
    if(usar_seq) nu = datos->seq[patron];
    else         nu = patron;
        
    /*cargar el patron en X1*/
    for (i = 0; i <= nn->N1; i ++) nn->x1[i] = S[nu][i];
    
    /*propagar la red*/
    forward(nn);
    
    /*generar matriz de predicciones*/
    for(i = 1; i <= nn->N3; i++) datos->pred[nu][i]=nn->x3[i];
    
    
    /*actualizar errores estimados*/
    for(i = 1; i <= nn->N3; i++) {
        *mse += (nn->x3[i]-S[nu][nn->N1+i])*(nn->x3[i]-S[nu][nn->N1+i]);                           /*mse se calcula para todas las salidas*/
        if(nn->N3==1) if((nn->x3[i]-0.5)*(S[nu][nn->N1+i]-0.5)<0.) *discrete_error+=1.0;    /*calcula error discreto para una sola salida - problema binario*/
    }
    
  }
  
  /*promedio los errores con la cantidad de patrones*/
  *mse /= ( (float)(pat_fin-pat_ini));
  *discrete_error /= (float) (pat_fin-pat_ini);
  
  if(CONTROL>3) {printf("End prop\n");fflush(NULL);}
  return 1;
}
/* --------------------------------------------------------------------------------------- */
/*train: entrena la red
  Algoritmo: Stochastic Back propagation
  Las salidas son las curvas de entrenamiento (mse estocastico,mse al final de la epoca,
  mse validacion, mse test) en el archivo .mse y la matriz final de sinapsis en el archivo
  (WTS+1).wts
  Los resultados finales y la matriz resultante corresponden al minimo de validacion
  si no hay validacion, corresponde al minimo de mse al final de la epoca                   */
/* ---------------------------------------------------------------------------------------- */
int train(struct MLP *nn,struct DATOS *datos){

  FILE *ferror,*fpredic;
  char filepat[100];

  int nu, nu1;
  int iter,epocas_del_minimo;
  int i,j,k;
  float h;
  float eta,suma;
  float mse,mse_train,mse_valid,mse_test,minimo_valid;
  float disc_train,disc_valid,disc_test;

  /*bandera de error*/
  int error;

  /* Inicializar archivos de salida, predicciones y curvas de error */
  sprintf(filepat,"%s.predic",datos->filename);
  fpredic=fopen(filepat,"w");
  error=(fpredic==NULL);
  if(error){
    printf("Error al abrir archivo para guardar predicciones\n");
    return 1;
  }
  sprintf(filepat,"%s.mse",datos->filename);
  ferror=fopen(filepat,"a");
  error=(ferror==NULL);
  if(error){
    printf("Error al abrir archivo para guardar curvas\n");
    return 1;
  }
  
  minimo_valid=1.0E20;

  /* inicializo eta: la nueva variable es para poder variar el learning rate durante el entrenamiento si se desea */
  eta=nn->ETA;

  /*efectuar shuffle inicial de los datos de entrenamiento si SEED != -1, para dividir al azar entrenamiento y validacion*/
  if(nn->SEED>-1){
    srand((unsigned)nn->SEED);    
    shuffle(datos->PTOT,datos);
  }

  /* for principal: ITER iteraciones -- iter es la epoca */
  for (iter=1; iter<=nn->ITER; iter++) {

    mse = 0.0;

    shuffle(datos->PR,datos); /*reordeno al azar SOLO los patrones de entrenamiento*/

    /*barrido sobre los patrones de entrenamiento*/
    for(nu1=0;nu1<datos->PR;nu1++) {

          nu = datos->seq[nu1];  /*nu tiene el numero del patron que se va a presentar, lo toma del indice*/

	  /*cargar el patron en X1, entradas de la red*/
	  for( k=1 ;k<=nn->N1 ;k++) nn->x1[k] = datos->data[nu][k];
	  
	  /*propagar*/
	  forward(nn);
	  
	  /*cargar los targets, los valores correctos de las salidas*/
	  for( k=1 ;k<=nn->N3 ;k++) nn->target[k] = datos->data[nu][k+nn->N1];

	  /*calcular delta (gradiente) en 3 hilera, capa de salida*/
	  for( i=1; i<=nn->N3; i++){
		 nn->grad3[i] = (nn->target[i] - nn->x3[i]);  /*corresponde a lineal*/
	  }

	  /*calcular delta (gradiente) en 2 hilera, capa oculta*/
	  for( j=1 ;j<=nn->N2; j++){
		 suma = 0.0;
		 for( i=1 ;i<=nn->N3; i++)
			suma += nn->grad3[i] * nn->w2[i][j];
		 nn->grad2[j] = suma * (1.0- nn->x2[j]) * nn->x2[j];
	  }

	  /*calcular dw2 y corregir w2*/
	  for( i=1; i<=nn->N3; i++) for( j=0; j<=nn->N2; j++){
		nn->dw2[i][j]= nn->u*nn->dw2[i][j] + eta*nn->grad3[i]*nn->x2[j];   /*termino de momentum + correcccion actual*/
		nn->w2[i][j] += nn->dw2[i][j];                               /*correccion*/
	  }

	  /*calcular dw1 y corregir w1*/
	  for( j=1; j<=nn->N2; j++) for( k=0; k<=nn->N1; k++){
		nn->dw1[j][k]= nn->u*nn->dw1[j][k] + eta*nn->grad2[j]*nn->x1[k];   /*termino de momentum + correcccion actual*/
		nn->w1[j][k] += nn->dw1[j][k];                               /*correccion*/
	  }

	  /*actualizar el mse sumando el error en el patron presentado*/
	  for( i=1 ;i<=nn->N3; i++) mse += (nn->x3[i]-nn->target[i]) * (nn->x3[i]-nn->target[i]);
	  

    }/* next nu1 - barrido sobre patrones*/


    /* controles: grabar error cada NERROR iteraciones*/
    if ((iter % nn->NERROR) == 0) {

      /*mse es el errro estocastico sobre el conjunto de entrenamiento*/
      mse /= ((float)datos->PR);
      
      /*calcular el error real sobre el conjunto de entrenamiento propagando con los pesos en el momento actual*/
      evaluar(nn,datos,datos->data,0,datos->PR,1,&mse_train,&disc_train);

      /*calcular mse de validacion; si no hay validacion, usar mse_train*/
      if(datos->PR==datos->PTOT){
          mse_valid=mse_train;
          disc_valid=disc_train;
      }else{
          evaluar(nn,datos,datos->data,datos->PR,datos->PTOT,1,&mse_valid,&disc_valid);
      }

      /*calcular mse de test (si hay)*/
      if (datos->PTEST>0){
          evaluar(nn,datos,datos->test,0,datos->PTEST,0,&mse_test,&disc_test);
      }else  mse_test = disc_test = 0.;

      /*guardo todos los errores en el archivo .mse*/
      fprintf(ferror,"%f\t%f\t%f\t%f\t",mse,mse_train,mse_valid,mse_test);
      fprintf(ferror,"%f\t%f\t%f\n",disc_train,disc_valid,disc_test);
      if(CONTROL) fflush(NULL);

      /*conservo los pesos de la red que da el minimo error continuo de validacion*/ 
      if(mse_valid<minimo_valid){
	sinapsis_save(nn->WTS+1,nn);
	minimo_valid=mse_valid;
	epocas_del_minimo=iter;
      }
    }
    
    if(CONTROL>2) {printf("Iteracion %d\n",iter);fflush(NULL);}
    
  }/*next iter - fin lazo de iteraciones*/

  /*mostrar resumen del entrenamiento - valores en la ultima epoca evaluada*/
  printf("\nFin del entrenamiento.\n\n");
  printf("Error final:\nEntrenamiento(est):%f\nEntrenamiento(med):%f\n",mse,mse_train);
  printf("Validacion:%f\nTest:%f\n",mse_valid,mse_test);
  
  /* Calcular y guardar predicciones sobre el archivo de test */
  sinapsis_read(nn->WTS+1,nn);                                  /*leer pesos del minimo de validacion desde archivo*/
  evaluar(nn,datos,datos->test,0,datos->PTEST,0,&mse_test,&disc_test);    /*propagar la mejor red sobre todo el conjunto de test*/
  for(k=0; k<datos->PTEST; k++){
    for(i=1 ;i<=datos->N1; i++) fprintf(fpredic,"%f\t",datos->test[k][i]);    /*guardo las entradas de cada patron de test*/
    for(i=1; i<=datos->N3; i++) fprintf(fpredic,"%f\t",datos->pred[k][i]);    /*guardo la prediccion (continua) de la red para ese patron*/
    fprintf(fpredic,"\n");
  }
  /*muestro en pantalla el error de la mejor red*/
  printf("\nError minimo en validacion:\nEpoca:%d\nValidacion:%f\nTest:%f\nTest discreto:%f%%\n\n",epocas_del_minimo,minimo_valid,mse_test,100.0*disc_test);
  
  /*cierro los archivos de salida*/
  fclose(fpredic);
  fclose(ferror);

  return 0;

}


/* ----------------------------------------------------------------------------------------------------- 
Funcion principal.
Llama a las funciones que crean la red, leen los datos, entrena la red
 ----------------------------------------------------------------------------------------------------- */
int main(int argc, char **argv){

  /*bandera de error*/
  int error;
    
  if(argc!=2){
    printf("Modo de uso: bp <filename>\ndonde filename es el nombre del archivo (sin extension)\n");
    return 0;
  }
  
  struct MLP nn;
  struct DATOS datos;
  
  /* defino la red e inicializo */
  error=arquitec(argv[1],&nn,&datos);
  if(error){
    printf("Error en la definicion de la red\n");
    return 1;
  }

  /* leo los datos */
  error=read_data(argv[1],&datos);                 
  if(error){
    printf("Error en la lectura de datos\n");
    return 1;
  }

  /* entreno la red*/
  error=train(&nn,&datos);                  
  if(error){
    printf("Error en el entrenamiento\n");
    return 1;
  }

  return 0;

}
/* ----------------------------------------------------------------------------------------------------- */












