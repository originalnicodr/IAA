import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


def test1():
    bankdata = pd.read_csv("bill_authentication.csv")
    #bankdata.shape
    #bankdata.head()

    X = bankdata.drop('Class', axis=1)
    y = bankdata['Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)

    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))

def test2():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Assign colum names to the dataset
    colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
    #colnames = ['Variance', 'Skewness', 'Curtosis', 'Entropy', 'Class']

    # Read dataset to pandas dataframe
    irisdata = pd.read_csv(url, names=colnames)

    # Preprocessing
    X = irisdata.drop('Class', axis=1)
    y = irisdata['Class']

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


    #----------------------------------------

    # Polynomial kernel
    #svclassifier = SVC(kernel='poly', degree=8)

    # Gaussian Kernel
    #svclassifier = SVC(kernel='rbf')

    # Sigmoid Kernel
    svclassifier = SVC(kernel='sigmoid')

    #------------------------------------------

    svclassifier.fit(X_train, y_train)

    # Making Predictions
    y_pred = svclassifier.predict(X_test)

    # Evaluating the Algorithm
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def clase(data):
    return data[len(data)-1]

def leer_datos(nombre):
    data=pd.read_csv(nombre,header=None).to_records(index=False)#.T.to_dict()
    #print([clase(x) for x in data])
    return data



#leer_datos("bill_authentication.csv")



def training_folds(clases,data,algorithm):
    datos=leer_datos(data)

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

    

    #-------------Preparo los datos para utilizarse en el entrenamiento---------------------------
    datos=[]#lista con pares de folders para cada k
    for k in range(0,10):
        validation_folder={'data':[],'class':[]}
        training_folder={'data':[],'class':[]}
        for i in range(0,len(clases)):
            #print("carpeta numero "+str(k))
            index=class_data[i]['index']
            n=class_data[i]['n-folder']
            #print("index= " + str(index))
            #print("n= "+str(n))
            validation_folder['data']+=class_data[i]['data'][index:index+n]
            training_folder['data']+=class_data[i]['data'][:index] + class_data[i]['data'][index+n:]
            class_data[i]['index']+=n
            datos.append((training_folder,validation_folder))
    #-------------------------------------------------------------------------------------------
    if algorithm=='linear':
        print("Ejecutando el algoritmo SVM linear")
        return svm_linear(datos)
    if algorithm=='gaussian':
        print("Ejecutando el algoritmo SVM rbf (gaussian)")
        return svm_gaussian(datos)
    if algorithm=='trees':
        print("Ejecutando el algoritmo trees")

    if algorithm=='bayes':
        print("Ejecutando el algoritmo naieve bayes")


def svm_linear(datos):
    mejor_ajuste={'c':1.0,'presicion':0}#para que tenga para comparar al principio
    presicion_resultado=[]#donde se van a guardar los valores de presicion de cada fold
    mejor_fold=0 #
    for k in range(0,10):
        training_folder=datos[k][0]
        validation_folder=datos[k][1]

        #-----------Formateo de datos para usarlos con la svm------------------------------
        validation_folder['class']=pd.DataFrame([list(x)[len(validation_folder['data'][0])-1] for x in validation_folder['data']], columns =[len(validation_folder['data'][0])])
        validation_folder['data']=pd.DataFrame([list(x)[:len(validation_folder['data'][0])-1] for x in validation_folder['data']], columns =range(0,len(validation_folder['data'][0])-1))
        training_folder['class']=pd.DataFrame([list(x)[len(training_folder['data'][0])-1] for x in training_folder['data']], columns =[len(training_folder['data'][0])])
        training_folder['data']=pd.DataFrame([list(x)[:len(training_folder['data'][0])-1] for x in training_folder['data']], columns =range(0,len(training_folder['data'][0])-1))
        #---------------------------------------------------------------------------------
        
        #--------Buscar los valores de config optimos en el primer fold--------------
        if k==0:
            valores_posibles=[10**x for x in range(-5,6)]

            for c in valores_posibles:
                print("Resultados de la carpeta numero " + str(k) + " con c= " + str(c))

                svclassifier = SVC(kernel='linear',C=c)
                # Training
                svclassifier.fit(training_folder['data'], training_folder['class'].values.ravel())

                # Making Predictions
                prediccion = svclassifier.predict(validation_folder['data'])

                # Evaluating the Algorithm
                presicion=classification_report(validation_folder['class'], prediccion, output_dict=bool)

                print("presicion - macro avg:" + str(presicion['macro avg']))

                #---Optimizacion de los valores-----------
                if mejor_ajuste['presicion']<(presicion['macro avg']['precision'] + presicion['macro avg']['recall'])/2:
                    mejor_ajuste={'c':c,'presicion':(presicion['macro avg']['precision'] + presicion['macro avg']['recall'])/2}
                #------------------------------------------ 

            presicion_resultado.append(mejor_ajuste['presicion'])
        #------------------------------------------------------------------------------

        else:
            print("Resultados de la carpeta numero " + str(k))
            svclassifier = SVC(kernel='linear',C=mejor_ajuste['c'])
            # Training
            svclassifier.fit(training_folder['data'], training_folder['class'].values.ravel())

            # Making Predictions
            prediccion = svclassifier.predict(validation_folder['data'])
            
            # Evaluating the Algorithm
            presicion=classification_report(validation_folder['class'], prediccion, output_dict=bool)

            print("presicion - macro avg:" + str(presicion['macro avg']))

            presicion_resultado.append((presicion['macro avg']['precision'] + presicion['macro avg']['recall'])/2)

        for i in range(0,len(presicion_resultado)): #Lo hago asi por que quiero el valor de la carpeta
            if presicion_resultado[mejor_fold]<presicion_resultado[i]:
                mejor_fold=i
        print("La mejor solucion surgio en la carpeta n" + str(mejor_fold) + " con un valor de c="+str(mejor_ajuste['c']) +"y una presicion de "+str(presicion_resultado[mejor_fold]))
        return presicion_resultado[mejor_fold]


def svm_gaussiana(data):
    mejor_ajuste={'c':1.0,'gamma':0,'presicion':0}#para que tenga para comparar al principio
    presicion_resultado=[]#donde se van a guardar los valores de presicion de cada fold
    mejor_fold=0 #
    for k in range(0,10):
        training_folder=datos[k][0]
        validation_folder=datos[k][1]

        #-----------Formateo de datos para usarlos con la svm------------------------------
        validation_folder['class']=pd.DataFrame([list(x)[len(validation_folder['data'][0])-1] for x in validation_folder['data']], columns =[len(validation_folder['data'][0])])
        validation_folder['data']=pd.DataFrame([list(x)[:len(validation_folder['data'][0])-1] for x in validation_folder['data']], columns =range(0,len(validation_folder['data'][0])-1))
        training_folder['class']=pd.DataFrame([list(x)[len(training_folder['data'][0])-1] for x in training_folder['data']], columns =[len(training_folder['data'][0])])
        training_folder['data']=pd.DataFrame([list(x)[:len(training_folder['data'][0])-1] for x in training_folder['data']], columns =range(0,len(training_folder['data'][0])-1))
        #---------------------------------------------------------------------------------
        
        #--------Buscar los valores de config optimos en el primer fold--------------
        if k==0:
            valores_posibles=[10**x for x in range(-5,6)]

            for c in valores_posibles:
                for gamma in ['scale','auto']+valores_posibles: #puede que usar todos esos valores posibles esten de mas, sino recortar (poner que gamma pruebe con valores negativos?)
                    print("Resultados de la carpeta numero " + str(k) + " con c= " + str(c))

                    svclassifier = SVC(kernel='rbf',C=c, gamma=gamma)
                    # Training
                    svclassifier.fit(training_folder['data'], training_folder['class'].values.ravel())

                    # Making Predictions
                    prediccion = svclassifier.predict(validation_folder['data'])

                    # Evaluating the Algorithm
                    presicion=classification_report(validation_folder['class'], prediccion, output_dict=bool)

                    print("presicion - macro avg:" + str(presicion['macro avg']))

                    #---Optimizacion de los valores-----------
                    if mejor_ajuste['presicion']<(presicion['macro avg']['precision'] + presicion['macro avg']['recall'])/2:
                        mejor_ajuste={'c':c, 'gamma':gamma, 'presicion':(presicion['macro avg']['precision'] + presicion['macro avg']['recall'])/2}
                    #------------------------------------------ 

            presicion_resultado.append(mejor_ajuste['presicion'])
        #------------------------------------------------------------------------------

        else:
            print("Resultados de la carpeta numero " + str(k))
            svclassifier = SVC(kernel='rbf',C=mejor_ajuste['c'], gamma=mejor_ajuste['gamma'])
            # Training
            svclassifier.fit(training_folder['data'], training_folder['class'].values.ravel())

            # Making Predictions
            prediccion = svclassifier.predict(validation_folder['data'])
            
            # Evaluating the Algorithm
            presicion=classification_report(validation_folder['class'], prediccion, output_dict=bool)

            print("presicion - macro avg:" + str(presicion['macro avg']))

            presicion_resultado.append((presicion['macro avg']['precision'] + presicion['macro avg']['recall'])/2)

        for i in range(0,len(presicion_resultado)): #Lo hago asi por que quiero el valor de la carpeta
            if presicion_resultado[mejor_fold]<presicion_resultado[i]:
                mejor_fold=i
        print("La mejor solucion surgio en la carpeta n" + str(mejor_fold) + " con un valor de c="+str(mejor_ajuste['c']) +"y una presicion de "+str(presicion_resultado[mejor_fold]))
        return presicion_resultado[mejor_fold]


training_folds([0,1],"BBBs.data","linear")













a={u'1': {'recall': 0.8148148148148148,
        'f1-score': 0.7719298245614035,
        'support': 27,
        'precision': 0.7333333333333333},
u'0': {'recall': 0.38461538461538464, 'f1-score': 0.4347826086956522, 'support': 13, 'precision': 0.5},
'weighted avg': {'recall': 0.675, 'f1-score': 0.6623569794050344, 'support': 40, 'precision': 0.6575},
'micro avg': {'recall': 0.675, 'f1-score': 0.675, 'support': 40, 'precision': 0.675},
'macro avg': {'recall': 0.5997150997150997, 'f1-score': 0.6033562166285278, 'support': 40, 'precision': 0.6166666666666667}}


#The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
#The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.
#The F-beta score can be interpreted as a weighted harmonic mean of the precision and recall, where an F-beta score reaches its best value at 1 and worst score at 0.
#The F-beta score weights recall more than precision by a factor of beta. beta == 1.0 means recall and precision are equally important.
#The support is the number of occurrences of each class in y_true.


#Resultados de la carpeta numero 0 con c= 1e-05
#presicion - macro avg:{'recall': 0.48148148148148145, 'f1-score': 0.3939393939393939, 'support': 40, 'precision': 0.3333333333333333}
#Resultados de la carpeta numero 0 con c= 0.0001
#Resultados de la carpeta numero 0 con c= 0.0001
#presicion - macro avg:{'recall': 0.5997150997150997, 'f1-score': 0.6033562166285278, 'support': 40, 'precision': 0.6166666666666667}
#Resultados de la carpeta numero 0 con c= 0.001
#Resultados de la carpeta numero 0 con c= 0.001
#presicion - macro avg:{'recall': 0.6182336182336182, 'f1-score': 0.6238244514106583, 'support': 40, 'precision': 0.6487455197132617}
#Resultados de la carpeta numero 0 con c= 0.01
#Resultados de la carpeta numero 0 con c= 0.01
#presicion - macro avg:{'recall': 0.5997150997150997, 'f1-score': 0.6033562166285278, 'support': 40, 'precision': 0.6166666666666667}
#Resultados de la carpeta numero 0 con c= 0.1
#Resultados de la carpeta numero 0 con c= 0.1
#presicion - macro avg:{'recall': 0.5997150997150997, 'f1-score': 0.6033562166285278, 'support': 40, 'precision': 0.6166666666666667}
#Resultados de la carpeta numero 0 con c= 1
#Resultados de la carpeta numero 0 con c= 1
#presicion - macro avg:{'recall': 0.5427350427350428, 'f1-score': 0.5423340961098397, 'support': 40, 'precision': 0.55}
#Resultados de la carpeta numero 0 con c= 10
#Resultados de la carpeta numero 0 con c= 10
#presicion - macro avg:{'recall': 0.6196581196581197, 'f1-score': 0.6218181818181819, 'support': 40, 'precision': 0.625}
#Resultados de la carpeta numero 0 con c= 100
#Resultados de la carpeta numero 0 con c= 100
#presicion - macro avg:{'recall': 0.5242165242165242, 'f1-score': 0.5238095238095238, 'support': 40, 'precision': 0.5266457680250785}
#Resultados de la carpeta numero 0 con c= 1000
#Resultados de la carpeta numero 0 con c= 1000
#presicion - macro avg:{'recall': 0.5256410256410257, 'f1-score': 0.5248078266946191, 'support': 40, 'precision': 0.5247252747252747}
#Resultados de la carpeta numero 0 con c= 10000
