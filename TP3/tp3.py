import re
from functools import reduce
import subprocess
import time
import os
import math

def get_values(s):
    return map(lambda x: float(x),re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s))

def ej1():
    
    f = open("ej1.txt", "w")
    f.write("ErrorEntrenamiento ErrorValidacion ErrorTest\n")

    valoresd=[2, 4, 8, 16, 32]
    for d in valoresd:
        nb = open("ej-"+str(d)+".nb", "w")
        nb.write(str(d)+'\n2\n250\n250\n1000\n0\n0')
        nb.close()

        valores_guardados=[]
        subprocess.check_output("./TP0.out a 10000 "+str(d)+" 0.78", shell=True, universal_newlines=True)
        os.rename('ej.data', 'ej-'+str(d)+'.test')
        subprocess.check_output("./TP0.out a 250 "+str(d)+" 0.78", shell=True, universal_newlines=True)
        os.rename('ej.data', 'ej-'+str(d)+'.data')

        output= subprocess.check_output("./nb_n.out ej-"+str(d), shell=True, universal_newlines=True)
        #print(output)
        s= output.split('Errores:', 1)[1]
        valores=get_values(s)
        #print(valores)

        f.write(str(valores[0])+" "+str(valores[1])+" "+str(valores[2])+"\n")
    f.close()
    
ej1()
"""
Naive Bayes con distribuciones normales:
Cantidad de entradas:2
Cantidad de clases:2
Archivo de patrones: ej-2
Cantidad total de patrones: 325
Cantidad de patrones de entrenamiento: 250
Cantidad de patrones de validacion: 75
Cantidad de patrones de test: 1000
Semilla para la funcion rand(): 1621954578
Fin del entrenamiento.

Errores:
Entrenamiento:64.800000%
Validacion:0.000000%
Test:100.000000%
"""



