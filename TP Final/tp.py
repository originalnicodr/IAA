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



datos=leer_datos("BBBs.data")

#filterclass
clases=[0,1]
class_data=[]
for y in clases:
    elem_class={}#'data':[], 'n-folder':0,'index':0} #por alguna razon me pide inicializarlo
    elem_class['data']=[x for x in datos if clase(x)==y]
    
    elem_class['n-folder']=(len(elem_class['data'])/len(datos))*len(datos)/10
    elem_class['index']=0
    class_data.append(elem_class)
    print(len(datos)/10)
    #print("elem_class['n-folder']=(len(elem_class['data'])/len(datos))*len(datos)/10 = "+str(len(elem_class['data']))+"/"+str(len(datos))+"*"+str(len(datos))+"/10 = "+str(elem_class['n-folder']))

for k in range(0,10):
    folder=[]
    for i in range(0,len(clases)):
        #print("carpeta numero "+str(k))
        index=class_data[i]['index']
        n=elem_class['n-folder']
        #print("index= " + str(index))
        #print("n= "+str(n))
        folder+=class_data[i]['data'][index:index+n]
        class_data[i]['index']+=n

#prepararlo para poder obtener lista SIN los elementos que uso para testear

    print(folder)

    if k==0:
        pass
    

