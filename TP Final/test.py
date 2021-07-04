import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

import re
import subprocess
import os

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