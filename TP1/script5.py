import re
from functools import reduce
import subprocess
import time
import os

def get_training_and_testing_values(s):
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

    """
    print("Guardando valores de training")
    trainingv=[]

    for i in range(0,3):
        trainingv.append(float(trainingline[trainingline.find("(")+2:trainingline.find(')')-1]))#si quiero el % sacar el -1
        trainingline=trainingline.split(')', 1)[1]

    #print(trainingv)

    print("Guardando valores de testing")
    testingv=[]

    for i in range(0,3):
        testingv.append(float(testline[testline.find("(")+2:testline.find(')')-1]))#si quiero el % sacar el -1
        testline=testline.split(')', 1)[1]

    #print(testingv)
    """

    trainingv= re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", trainingline)
    testingv= re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", testline)

    return (map(lambda x:float(x),trainingv),map(lambda x:float(x),testingv))

def sum(a, b):
    #print(f"a={a}, b={b}, {a} + {b} ={a+b}")
    return a + b


def ej5():
    f = open("ej5.txt", "a")

    f.write("TrainingSetSize    trainingAfterPrunningError    testingAfterPrunningError    trainingAfterPrunningSize    testingAfterPrunningSize\n")

    valoresn=[125, 250, 500, 1000, 2000, 4000]
    for n in valoresn:
        valores_guardados=[]
        for i in range(0,20):
            #estoy corriendo python 2 aca
            subprocess.check_output("./TP0.out b "+str(n)+" 2 0.78", shell=True, universal_newlines=True)
            output= subprocess.check_output("./c4.5 -f ej -g -u", shell=True, universal_newlines=True)
            valores_guardados.append(get_training_and_testing_values(output))

        #BeforePruningSize, BeforePruningErrors, BeforePruningErrors%, AfterPruningSize, AfterPruningErrors, AfterPruningErrors%, Estimate%]

        trainingAfterPSize=reduce(sum,map(lambda x: x[0][3],valores_guardados))/len(valores_guardados)
        trainingAfterPError=reduce(sum,map(lambda x: x[0][5],valores_guardados))/len(valores_guardados)

        testingAfterPSize=reduce(sum,map(lambda x: x[1][3],valores_guardados))/len(valores_guardados)
        testingAfterPError=reduce(sum,map(lambda x: x[1][5],valores_guardados))/len(valores_guardados)

        
        f.write(str(n) + "    " + str(trainingAfterPError) + "    " + str(testingAfterPError) + "    " + str(trainingAfterPSize) + "    " + str(testingAfterPSize) + "\n")

        time.sleep(1)



    f.close()




def ej6():
    f = open("ej6.txt", "a")

    f.write("CValue    TrainingBeforePrunningError    TrainingAfterPrunningError    TestingBeforePrunningError    TestingAfterPrunningError\n")

    valoresc=[0.5, 1, 1.5, 2, 2.5]
    for c in valoresc:
        valores_guardados=[]
        subprocess.check_output("./TP0.out a 10000 5 "+str(c), shell=True, universal_newlines=True)
        os.rename('ej.data', 'ej.test')
        for i in range(0,20):
            #estoy corriendo python 2 aca
            subprocess.check_output("./TP0.out a 250 5 "+str(c), shell=True, universal_newlines=True)
            output= subprocess.check_output("./c4.5 -f ej -g -u", shell=True, universal_newlines=True)
            valores_guardados.append(get_training_and_testing_values(output))

        #BeforePruningSize, BeforePruningErrors, BeforePruningErrors%, AfterPruningSize, AfterPruningErrors, AfterPruningErrors%, Estimate%]

        trainingBeforePError=reduce(sum,map(lambda x: x[0][2],valores_guardados))/len(valores_guardados)
        trainingAfterPError=reduce(sum,map(lambda x: x[0][5],valores_guardados))/len(valores_guardados)

        testingBeforePError=reduce(sum,map(lambda x: x[1][2],valores_guardados))/len(valores_guardados)
        testingAfterPError=reduce(sum,map(lambda x: x[1][5],valores_guardados))/len(valores_guardados)

        
        f.write(str(c) + "    " + str(trainingAfterPError) + "    " + str(trainingAfterPError) + "    " + str(testingBeforePError) + "    " + str(testingAfterPError) + "\n")

        time.sleep(1)



    f.close()


def get_values_data(s,n):
    listofstrings = re.split('\n', s)
    #print(listofstrings)
    print(re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", listofstrings[n]))

def ej6_1():
    f = open("ej6-1.txt", "a")

    f.write("CValue    ErrorMinimo\n")

    def predict_parallel(coord):
        if (float(coord[0])>0):
            return 1
        else:
            return 0

    def predict_diagonal(coord):
        #if (-float(coord[0])<float(coord[1]) and -float(coord[0])<float(coord[2]) and -float(coord[0])<float(coord[3]) and -float(coord[0])<float(coord[4])):
        #if (-float(coord[0])<float(coord[1]) or -float(coord[0])<float(coord[2]) or -float(coord[0])<float(coord[3]) or -float(coord[0])<float(coord[4])):
        if ((-float(coord[0]))<float(coord[1])):
            return 1  #etiqueto 1, pero tengo que ver que pasa con las otras componentes si comparo con coord[0]
        else:
            return 0

    valoresc=[0.5, 1, 1.5, 2, 2.5]
    for c in valoresc:
        correct=0
        subprocess.check_output("./TP0.out a 10000 5 "+str(c), shell=True, universal_newlines=True)
        os.rename('ej.data', 'minnimunerror.data')

        file = open('minnimunerror.data', 'r')
        
        for i in range(0,10000):
            valores_leidos=re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", file.readline())
            
            #Si estamos en ejercicio a)diagonal
            if (float(valores_leidos[5])==predict_diagonal(valores_leidos[:5])):#fijarse si estoy haciendolo bien
                correct+=1
            

            #Si estamos en ejercicio b)paralelo
            """
            if (float(valores_leidos[5])==predict_parallel(valores_leidos[:5])):#fijarse si estoy haciendolo bien
                correct+=1
                #print(str(valores_leidos[5])+"=="+str(predict_parallel(valores_leidos[:3])))
            """
        
        file.close()
        
        #print(correct)
        #print(float(correct)/10000)

        f.write(str(c) + "    " + str(100-float(correct)/100) + "\n")

        time.sleep(1)



    f.close()

def ej7():
    f = open("ej7.txt", "a")

    f.write("Dimensiones    TrainingBeforePrunningError    TrainingAfterPrunningError    TestingBeforePrunningError    TestingAfterPrunningError\n")

    valoresd=[2, 4, 8, 16, 32]
    for d in valoresd:
        valores_guardados=[]
        subprocess.check_output("./TP0.out b 10000 "+str(d)+" 0.78", shell=True, universal_newlines=True)
        os.rename('ej.data', 'ej.test')
        for i in range(0,20):
            #estoy corriendo python 2 aca
            subprocess.check_output("./TP0.out b 250 "+str(d)+" 0.78", shell=True, universal_newlines=True)
            output= subprocess.check_output("./c4.5 -f ej -g -u", shell=True, universal_newlines=True)
            valores_guardados.append(get_training_and_testing_values(output))

        #BeforePruningSize, BeforePruningErrors, BeforePruningErrors%, AfterPruningSize, AfterPruningErrors, AfterPruningErrors%, Estimate%]

        trainingBeforePError=reduce(sum,map(lambda x: x[0][2],valores_guardados))/len(valores_guardados)
        trainingAfterPError=reduce(sum,map(lambda x: x[0][5],valores_guardados))/len(valores_guardados)

        testingBeforePError=reduce(sum,map(lambda x: x[1][2],valores_guardados))/len(valores_guardados)
        testingAfterPError=reduce(sum,map(lambda x: x[1][5],valores_guardados))/len(valores_guardados)

        
        f.write(str(d) + "    " + str(trainingAfterPError) + "    " + str(trainingAfterPError) + "    " + str(testingBeforePError) + "    " + str(testingAfterPError) + "\n")

        time.sleep(1)



    f.close()

#ej6()
#ej6_1()
ej7()