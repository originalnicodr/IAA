import re
from functools import reduce
import subprocess
import time

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



s = """Evaluation on training data (3000 items):

	 Before Pruning           After Pruning
	----------------   ---------------------------
	Size      Errors   Size      Errors   Estimate

	 207   33( 1.1%)    193   38( 1.3%)    ( 5.6%)   <<

Evaluation on test data (10000 items):

	 Before Pruning           After Pruning
	----------------   ---------------------------
	Size      Errors   Size      Errors   Estimate

	 207  603( 6.0%)    193  602( 6.0%)    ( 5.6%)   <<


	  (a)  (b)	<-classified as
	 ---- ----
	 3489  312	(a): class 0
	  290 5909	(b): class 1"""
