import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

import re
import subprocess
import os

def clase(data):
    return data[len(data)-1]

def leer_datos(nombre):
    data=pd.read_csv(nombre,header=None).to_records(index=False)#.T.to_dict()
    #print([clase(x) for x in data])
    return data
#def svm_presicion(prediction_results):
    #probablemente tenga que cambiarlo
    #return (prediction_results['macro avg']['precision'] + prediction_results['macro avg']['recall'])/2
    """
    The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false   positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is   negative.
    The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives.   The recall is intuitively the ability of the classifier to find all the positive samples.
    The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score  reaches its best value at 1 and worst score at 0.
    The F-beta score weights recall more than precision by a factor of beta. beta == 1.0 means recall and precision are     equally important.
    The support is the number of occurrences of each class in y_true.
    """
def svm_accuracy(confussion_matrix):#solo funciona cuando tengo dos clases
    #print(confussion_matrix)
    #tn, fp, fn, tp= confussion_matrix.ravel()

    numerador=0
    denominador=0
    for i in range(0,len(confussion_matrix)):
        for j in range(0,len(confussion_matrix[0])):
            if i==j:
                numerador+=confussion_matrix[i][j]
            denominador+=confussion_matrix[i][j]
    r=float(numerador)/denominador
    return r

def training_folds(clases,data_name,algorithm):
    datos=leer_datos(data_name)

    #filterclass
    class_data=[]# lista de diccionarios donde:
        #'data':elementos de esa clase, 'n-folder':numero de elementos de esta clase en cada folder,'index':indice para separar los datos de esta clase en folders}
    for y in clases:
        elem_class={}
        elem_class['data']=[x for x in datos if clase(x)==y]
        elem_class['n-folder']=int((float(len(elem_class['data']))/len(datos))*float(len(datos))/10)
        elem_class['index']=0
        class_data.append(elem_class)
        #print("elem_class['n-folder']=(len(elem_class['data'])/len(datos))*len(datos)/10 = "+str(len(elem_class['data']))+"/"+str(len(datos))+"*"+str(len(datos))+"/10 = "+str(elem_class['n-folder']))

    #filetest=open("folders-test.txt", "w")

    

    #-------------Preparo los datos para utilizarse en el entrenamiento---------------------------
    datos=[]#lista con pares de folders para cada k

    
    for k in range(0,10):
        validation_folder={'data':[],'class':[]}
        training_folder={'data':[],'class':[]}
        for i in range(0,len(clases)):

            print("carpeta numero "+str(k)+" de la clase "+str(clases[i]))
            index=class_data[i]['index']
            n=class_data[i]['n-folder']
            print("index= " + str(index))
            print("n= "+str(n))
            validation_folder['data']+=[list(x)[:-1] for x in class_data[i]['data'][index:index+n]]#saco las clases de los datos
            validation_folder['class']+=[clases[i] for x in range(0,class_data[i]['n-folder'])]



            """
            for x in validation_folder['data']:
                for j in x:
                    filetest.write(str(j)+", ")
            filetest.write("\n")
            """
            

            temp_list=[list(x)[:-1] for x in (class_data[i]['data'][:index] + class_data[i]['data'][index+n:])]
            training_folder['data']+=temp_list
            training_folder['class']+=[clases[i] for x in temp_list]
            class_data[i]['index']+=n
        datos.append((training_folder,validation_folder))
        #print((training_folder,validation_folder))

    #filetest.close
    #-------------------------------------------------------------------------------------------
    
    if algorithm=='linear':
        print("Ejecutando el algoritmo SVM linear")
        return svm_linear(datos)
    if algorithm=='gaussian':
        print("Ejecutando el algoritmo SVM rbf (gaussian)")
        return svm_gaussian(datos)
    if algorithm=='trees':
        print("Ejecutando el algoritmo trees")
        #crear_archivos(datos, "ejercicio_final", 0, clases)
        return trees(datos,data_name+"-tree",clases)
    if algorithm=='bayes':
        print("Ejecutando el algoritmo naieve bayes")
        return naieve_bayes(datos,data_name+"-naieve-bayes",clases)
    
    
    



def svm_linear(datos):
    mejor_ajuste={'c':1.0,'presicion':(0,0)}#para que tenga para comparar al principio
    presicion_resultado=[]#donde se van a guardar los valores de presicion de cada fold
    mejor_fold=0 #
    for k in range(0,10):
        training_folder=datos[k][0]
        validation_folder=datos[k][1]

        #-----------Formateo de datos para usarlos con la svm------------------------------
        validation_folder['class']=pd.DataFrame(validation_folder['class'])
        validation_folder['data']=pd.DataFrame(validation_folder['data'])
        training_folder['class']=pd.DataFrame(training_folder['class'])
        training_folder['data']=pd.DataFrame(training_folder['data'])
        #---------------------------------------------------------------------------------
        
        #--------Buscar los valores de config optimos en el primer fold--------------
        if k==0:
            #valores_posibles=[10**x for x in range(-5,2)] #c=0.001 el mejor
            valores_posibles=[0.0001,0.0004,0.0007,0.001,0.004,0.007,0.01]

            for c in valores_posibles:
                print("Resultados de la carpeta numero " + str(k) + " con c= " + str(c))

                svclassifier = SVC(kernel='linear',C=c)

                # Training
                svclassifier.fit(training_folder['data'], training_folder['class'].values.ravel())

                # Making Predictions
                validacion_prediccion = svclassifier.predict(validation_folder['data'])

                # Evaluating the Algorithm
                validation_results=classification_report(validation_folder['class'], validacion_prediccion, output_dict=bool)
               
                validation_confusion_matrix=confusion_matrix(validation_folder['class'], validacion_prediccion)
                #print(validation_results)
                print("Validation acurracy:" + str(svm_accuracy(validation_confusion_matrix)))

                #-----------Training accurracy (se puede sacar)-------------------------
                # Calculando error training
                training_prediccion = svclassifier.predict(training_folder['data'])
                # Evaluating the Algorithm
                training_results=classification_report(training_folder['class'], training_prediccion, output_dict=bool)

                training_confusion_matrix=confusion_matrix(training_folder['class'], training_prediccion)
                #print(trainig_results)
                
                print("Training acurracy:" + str(svm_accuracy(training_confusion_matrix)))
                #-----------------------------------------------------------------------

                #---Optimizacion de los valores-----------
                if mejor_ajuste['presicion'][1]<svm_accuracy(validation_confusion_matrix):
                    mejor_ajuste={'c':c,'presicion':(svm_accuracy(training_confusion_matrix),svm_accuracy(validation_confusion_matrix))}
                #------------------------------------------ 



            presicion_resultado.append(mejor_ajuste['presicion'])
        #------------------------------------------------------------------------------

        else:
            print("Resultados de la carpeta numero " + str(k))
            svclassifier = SVC(kernel='linear',C=mejor_ajuste['c'])
            
            # Training
            svclassifier.fit(training_folder['data'], training_folder['class'].values.ravel())
            # Making Predictions
            validacion_prediccion = svclassifier.predict(validation_folder['data'])
            # Evaluating the Algorithm
            validation_results=classification_report(validation_folder['class'], validacion_prediccion, output_dict=bool)
            
            validation_confusion_matrix=confusion_matrix(validation_folder['class'], validacion_prediccion)

            #print(validation_results)
            print("Validation acurracy:" + str(svm_accuracy(validation_confusion_matrix)))

            #-----------Training accurracy (se puede sacar)-------------------------
            # Calculando error training
            training_prediccion = svclassifier.predict(training_folder['data'])
            # Evaluating the Algorithm
            training_results=classification_report(training_folder['class'], training_prediccion, output_dict=bool)
            
            training_confusion_matrix=confusion_matrix(training_folder['class'], training_prediccion)
            #print(trainig_results)
            
            print("Training acurracy:" + str(svm_accuracy(training_confusion_matrix)))
            #-----------------------------------------------------------------------

            presicion_resultado.append((svm_accuracy(training_confusion_matrix),svm_accuracy(validation_confusion_matrix)))

    for i in range(0,len(presicion_resultado)): #Lo hago asi por que quiero el valor de la carpeta
        if presicion_resultado[mejor_fold][1]<presicion_resultado[i][1]:
            mejor_fold=i
    print("La mejor solucion surgio en la carpeta n" + str(mejor_fold) + " con un valor de c="+str(mejor_ajuste['c']) +"y una presicion de validacion de "+str(presicion_resultado[mejor_fold][1]) +" y de entrenamiento de "+str(presicion_resultado[mejor_fold][0]))
    #return presicion_resultado[mejor_fold]
    return presicion_resultado


def svm_gaussian(data):
    mejor_ajuste={'c':1.0,'gamma':0,'presicion':(0,0)}#para que tenga para comparar al principio
    presicion_resultado=[]#donde se van a guardar los valores de presicion de cada fold
    mejor_fold=0 #
    for k in range(0,10):
        training_folder=data[k][0]
        validation_folder=data[k][1]

        #-----------Formateo de datos para usarlos con la svm------------------------------
        validation_folder['class']=pd.DataFrame(validation_folder['class'])
        validation_folder['data']=pd.DataFrame(validation_folder['data'])
        training_folder['class']=pd.DataFrame(training_folder['class'])
        training_folder['data']=pd.DataFrame(training_folder['data'])
        #---------------------------------------------------------------------------------
        
        #--------Buscar los valores de config optimos en el primer fold--------------
        if k==0:
            valores_posibles_c=[10**x for x in range(-5,6)]#deberia ser 6 en lugar de 2
            for c in valores_posibles_c:
                for gamma in ['scale','auto']+[float(x)/100 for x in range(1,100)]: #puede que usar todos esos valores posibles esten de mas, sino recortar (poner que gamma pruebe con valores negativos?)
                    print("Resultados de la carpeta numero " + str(k) + " con c= " + str(c) + " y gamma= "+str(gamma))

                    svclassifier = SVC(kernel='rbf',C=c, gamma=gamma)

                    # Training
                    svclassifier.fit(training_folder['data'], training_folder['class'].values.ravel())

                    # Making Predictions
                    validacion_prediccion = svclassifier.predict(validation_folder['data'])

                    # Evaluating the Algorithm
                    validation_results=classification_report(validation_folder['class'], validacion_prediccion, output_dict=bool)

                    validation_confusion_matrix=confusion_matrix(validation_folder['class'], validacion_prediccion)
                    #print(validation_results)
                    print("Validation acurracy:" + str(svm_accuracy(validation_confusion_matrix)))

                    #-----------Training accurracy (se puede sacar)-------------------------
                    # Calculando error training
                    training_prediccion = svclassifier.predict(training_folder['data'])
                    # Evaluating the Algorithm
                    training_results=classification_report(training_folder['class'], training_prediccion, output_dict=bool)

                    training_confusion_matrix=confusion_matrix(training_folder['class'], training_prediccion)
                    #print(trainig_results)

                    print("Training acurracy:" + str(svm_accuracy(training_confusion_matrix)))
                    #-----------------------------------------------------------------------

                    #---Optimizacion de los valores-----------
                    if mejor_ajuste['presicion'][1]<svm_accuracy(validation_confusion_matrix):
                        mejor_ajuste={'c':c,'gamma':gamma,'presicion':(svm_accuracy(training_confusion_matrix),svm_accuracy(validation_confusion_matrix))}
                    #------------------------------------------ 


            presicion_resultado.append(mejor_ajuste['presicion'])
        #------------------------------------------------------------------------------

        else:
            print("Resultados de la carpeta numero " + str(k))
            svclassifier = SVC(kernel='rbf',C=mejor_ajuste['c'], gamma=mejor_ajuste['gamma'])

            # Training
            svclassifier.fit(training_folder['data'], training_folder['class'].values.ravel())
            # Making Predictions
            validacion_prediccion = svclassifier.predict(validation_folder['data'])
            # Evaluating the Algorithm
            validation_results=classification_report(validation_folder['class'], validacion_prediccion, output_dict=bool)
            
            validation_confusion_matrix=confusion_matrix(validation_folder['class'], validacion_prediccion)

            #print(validation_results)
            print("Validation acurracy:" + str(svm_accuracy(validation_confusion_matrix)))

            #-----------Training accurracy (se puede sacar)-------------------------
            # Calculando error training
            training_prediccion = svclassifier.predict(training_folder['data'])
            # Evaluating the Algorithm
            training_results=classification_report(training_folder['class'], training_prediccion, output_dict=bool)
            
            training_confusion_matrix=confusion_matrix(training_folder['class'], training_prediccion)
            #print(trainig_results)
            
            print("Training acurracy:" + str(svm_accuracy(training_confusion_matrix)))
            #-----------------------------------------------------------------------

            presicion_resultado.append((svm_accuracy(training_confusion_matrix),svm_accuracy(validation_confusion_matrix)))

    for i in range(0,len(presicion_resultado)): #Lo hago asi por que quiero el valor de la carpeta
        if presicion_resultado[mejor_fold][1]<presicion_resultado[i][1]:
            mejor_fold=i
    print("La mejor solucion surgio en la carpeta n" + str(mejor_fold) + " con un valor de c="+str(mejor_ajuste['c']) +", un valor de gamma="+str(mejor_ajuste['gamma'])+"y una presicion de validacion de "+str(presicion_resultado[mejor_fold][1]) +" y de entrenamiento de "+str(presicion_resultado[mejor_fold][0]))
    #return presicion_resultado[mejor_fold]
    return presicion_resultado

def crear_archivos(data, nombre, k, clases):
    crear_archivo_data_tree(data,k,nombre)
    crear_archivo_test_tree(data,k,nombre)
    crear_archivo_names(data,nombre, clases)

def crear_archivo_data_tree(data,k,nombre): #el k es el fold que estoy usando para los datos
    f=open(nombre+".data", "w")
    

    training_data=data[k][0]['data']
    training_class=data[k][0]['class']

    """
    validation_data=data[k][1]['data']
    validation_class=data[k][1]['class']
    for d in range(0,len(validation_data)):#itero sobre datos
        write_value=''
        for i in validation_data[d]:#itero sobre componente de cada dato
            write_value+=str(i)+', '
        f.write(write_value+str(validation_class[d])+'\n')
    """

    for d in range(0,len(training_data)):#itero sobre datos
        write_value=''
        for i in training_data[d]:#itero sobre componente de cada dato
            write_value+=str(i)+', '
        f.write(write_value+str(training_class[d])+'\n')
    
    f.close

def crear_archivo_test_tree(data,k,nombre): 
    f=open(nombre+".test", "w")

    validation_data=data[k][1]['data']
    validation_class=data[k][1]['class']
    for d in range(0,len(validation_data)):#itero sobre datos
        write_value=''
        for i in validation_data[d]:#itero sobre componente de cada dato
            write_value+=str(i)+', '
        f.write(write_value+str(validation_class[d])+'\n')
    f.close

def crear_archivo_names(data,nombre, clases):
    f=open(nombre+".names", "w")

    for c in clases:
        if c != clases[0]:
            f.write(", ") #para no poner una coma adelante, lo tengo que hacer asi por que sino me queda 0, 1, .
        f.write(str(c))
    f.write(".\n\n")

    #No puede trabajar con variables que no sean continuas, habria que editar esto de aca para que funcione
    for i in range(0,len(data[0][1]['data'][0])):
        f.write("d"+str(i)+": continuous.\n")

    f.close

    

def get_tree_values(s):
    """
    Devuelve una tupla de lista de valores, donde la primera lista son los valores de training y la segunda
    son los valores de testing. Ambos se guardan en la lista como:
    [BeforePruningSize, BeforePruningErrors, BeforePruningErrors%, AfterPruningSize, AfterPruningErrors, AfterPruningErrors%, Estimate%]
    """
    trainingline_ = re.search('\n(.*)<<', s)
    trainingline=trainingline_.group(1)
    #print(trainingline)
    testline_ = re.search('\n(.*)<<', s.split('<<', 1)[1])
    testline=testline_.group(1)
    #print(testline)

    trainingv= re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", trainingline)
    testingv= re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", testline)

    return (map(lambda x:float(x),trainingv),map(lambda x:float(x),testingv))

def trees(data,nombre,clases):
    crear_archivo_names(data,nombre,clases)
    
    resultados_folds=[]
    for k in range(0,10):
        crear_archivo_data_tree(data,k,nombre)
        crear_archivo_test_tree(data,k,nombre)

        print("Entrenando arbol con fold n "+str(k))
        output= subprocess.check_output("./c4.5 -f "+nombre+" -g -u", shell=True, universal_newlines=True)
        #print(output)
        result=get_tree_values(output)
        TrainingAfterPruningError=result[0][5]
        ValidationAfterPruningError=result[1][5]

        print(result)

        resultados_folds.append((TrainingAfterPruningError,ValidationAfterPruningError))
        
    
    os.remove(nombre+".data")
    os.remove(nombre+".test")
    os.remove(nombre+".names")
    return resultados_folds #(TrainigError,ValidationError)



def crear_archivo_data_naieve_bayes(data,k,nombre): #el k es el fold que estoy usando para los datos
    f=open(nombre+".data", "w")
    
    validation_data=data[k][1]['data']
    validation_class=data[k][1]['class']
    training_data=data[k][0]['data']
    training_class=data[k][0]['class']

    for d in range(0,len(training_data)):#itero sobre datos
        write_value=''
        for i in training_data[d]:#itero sobre componente de cada dato
            write_value+=str(i)+', '
        f.write(write_value+str(training_class[d])+'\n')

    for d in range(0,len(validation_data)):#itero sobre datos
        write_value=''
        for i in validation_data[d]:#itero sobre componente de cada dato
            write_value+=str(i)+', '
        f.write(write_value+str(validation_class[d])+'\n')

    f.close

def crear_nb_naieve_bayes(data,nombre,clases):
    f=open(nombre+".nb", "w")

    validation_data=data[0][1]['data']
    validation_class=data[0][1]['class']
    training_data=data[0][0]['data']
    training_class=data[0][0]['class']


    dimension=len(validation_data[0])
    cantidad_datos_entrenamiento=len(training_data)
    cantidad_datos=len(validation_data)+cantidad_datos_entrenamiento
    cantidad_datos_test=0
    f.write(str(dimension)+'\n'+str(len(clases))+'\n'+str(cantidad_datos)+'\n'+str(cantidad_datos_entrenamiento)+'\n'+str(cantidad_datos_test)+'\n-1\n0') #

    f.close

def get_values_nb(s):
    return map(lambda x: float(x),re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s))

def naieve_bayes(data,nombre,clases):
    crear_archivo_names(data,nombre,clases)
    crear_nb_naieve_bayes(data,nombre,clases)

    resultados_folds=[]

    for k in range(0,10):
        crear_archivo_data_naieve_bayes(data,k,nombre)

        print("Entrenando arbol con fold n "+str(k))
        output= subprocess.check_output("./nb_n.out "+nombre, shell=True, universal_newlines=True)
        #print(output)
        s= output.split('Errores:', 1)[1]
        valores=get_values_nb(s)

        print("Errores: "+s)
        print(valores)
        resultados_folds.append((valores[0],valores[1]))

    os.remove(nombre+".data")
    os.remove(nombre+".nb")
    os.remove(nombre+".names")
    return resultados_folds #(TrainigError,ValidationError)

def desviacion_estandar(lst):
    d=average(lst)

    sd=0.0
    for x in lst:
        sd+=(x-d)**2
    sd=(float(sd)/(len(lst)-1))**(1.0/2)
    return sd

def average(lst):
    return float(sum(lst)) / len(lst)


def t_test(errores1,errores2):
    validation_error1=[x[1] for x in errores1]
    validation_error2=[x[1] for x in errores2]

    d=average(validation_error1)-average(validation_error2)
    #d=average(validation_error1+validation_error2)

    #Fijarse si en el calculo de la desviacion estandar para sd tengo que usar el d que calcule arriba o solamente el promedio de los errores
    sd=desviacion_estandar(validation_error1+validation_error2)

    """
    sd=0.0
    for x in validation_error1+validation_error2:
        sd+=(x-d)**2
    sd=(float(sd)/(len(validation_error1+validation_error2)-1))**(1.0/2)
    """
    

    t=float((d-0))/(float(sd)/(len(errores1+errores2)**(1/2)))

    return t

def main():
    
    #----------------Arboles------------------------
    filetree=open("TreeErrores.txt", "w")
    filetree.write("TrainingError ValidationError\n")
    error_trees=training_folds([0,1],"BBBs.data","trees")
    error_trees=[(x/100,y/100) for (x,y) in error_trees]
    for (training,validation) in error_trees:
        filetree.write(str(training) + "  " + str(validation) + "\n")
    filetree.close
    #-----------------------------------------------

    
    #----------------Naieve-Bayes-------------------
    filebayes=open("BayesErrores.txt", "w")
    filebayes.write("TrainingError ValidationError\n")
    error_bayes=training_folds([0,1],"BBBs.data","bayes")
    error_bayes=[(x/100,y/100) for (x,y) in error_bayes]
    for (training,validation) in error_bayes:
        filebayes.write(str(training) + "  " + str(validation) + "\n")
    filebayes.close
    #-----------------------------------------------
    

    
    #------------------SVM-Linear-------------------
    error_svm_linear=training_folds([0,1],"BBBs.data","linear")
    print(error_svm_linear)
    error_svm_linear=[(1-x,1-y) for (x,y) in error_svm_linear]#cambio los valores para que sean errores

    filesvmlinear=open("SVMLinearErrores.txt", "w")
    filesvmlinear.write("TrainingError ValidationError\n")

    for (training,validation) in error_svm_linear:
        filesvmlinear.write(str(training) + "  " + str(validation) + "\n")
    filesvmlinear.close
    #-----------------------------------------------
    
    
    
    
    #------------------SVM-Gaussian-------------------
    error_svm_gauss=training_folds([0,1],"BBBs.data","gaussian")
    print(error_svm_gauss)
    error_svm_gauss=[(1-x,1-y) for (x,y) in error_svm_gauss]#cambio los valores para que sean errores

    filesvmgauss=open("SVMGaussianErrores.txt", "w")
    filesvmgauss.write("TrainingError ValidationError\n")

    for (training,validation) in error_svm_gauss:
        filesvmgauss.write(str(training) + "  " + str(validation) + "\n")
    filesvmgauss.close
    #-----------------------------------------------

    print("Promedio error validacion bayes: "+str(average([x[1] for x in error_bayes])))
    print("Promedio error training bayes: "+str(average([x[0] for x in error_bayes])))
    print("Desviacion estandar bayes: "+str(desviacion_estandar([x[1] for x in error_bayes])))
    print("Promedio error validacion trees: "+str(average([x[1] for x in error_trees])))
    print("Promedio error training trees: "+str(average([x[0] for x in error_trees])))
    print("Desviacion estandar trees: "+str(desviacion_estandar([x[1] for x in error_trees])))
    print("Promedio error validacion svm linear: "+str(average([x[1] for x in error_svm_linear])))
    print("Promedio error training svm linear: "+str(average([x[0] for x in error_svm_linear])))
    print("Desviacion estandar svm linear: "+str(desviacion_estandar([x[1] for x in error_svm_linear])))
    print("Promedio error validacion svm gauss: "+str(average([x[1] for x in error_svm_gauss])))
    print("Promedio error training svm gauss: "+str(average([x[0] for x in error_svm_gauss])))
    print("Desviacion estandar svm gauss: "+str(desviacion_estandar([x[1] for x in error_svm_gauss])))

    """
    Promedio error validacion bayes: 0.607499999
    Promedio error training bayes: 0.561866668
    Desviacion estandar bayes: 0.0425734706778
    Promedio error validacion trees: 0.21
    Promedio error training trees: 0.0223
    Desviacion estandar trees: 0.0529674952736
    Promedio error validacion svm linear: 0.32
    Promedio error training svm linear: 0.2056
    Desviacion estandar svm linear: 0.0586893895389
    Promedio error validacion svm gauss: 0.285
    Promedio error training svm gauss: 0.0456
    Desviacion estandar svm gauss: 0.05027701043

    t-test mejor y peor: 1.95379483587
    t-test mejor y segundo mejor: 3.91010437919
    """


    #print(t_test(error_trees,error_bayes))#mejor y peor
    #print(t_test(error_trees,error_svm_gauss))#mejor y segundo mejor
    print(t_test(error_bayes,error_trees))#mejor y peor
    print(t_test(error_svm_gauss,error_trees))#mejor y segundo mejor





main()
#print(desviacion_estandar([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))






