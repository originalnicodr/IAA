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
        nb.write(str(d)+'\n2\n250\n188\n1000\n0\n0')
        nb.close()

        valores_guardados=[]
        subprocess.check_output("./TP0.out b 10000 "+str(d)+" 0.78", shell=True, universal_newlines=True)
        os.rename('ej.data', 'ej-'+str(d)+'.test')
        subprocess.check_output("./TP0.out b 250 "+str(d)+" 0.78", shell=True, universal_newlines=True)
        os.rename('ej.data', 'ej-'+str(d)+'.data')

        output= subprocess.check_output("./nb_n.out ej-"+str(d), shell=True, universal_newlines=True)
        #print(output)
        s= output.split('Errores:', 1)[1]
        valores=get_values(s)
        #print(valores)

        f.write(str(valores[0])+" "+str(valores[1])+" "+str(valores[2])+"\n")
    f.close()


def ej1_2():
    
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


        for i in range(0,10):
            #estoy corriendo python 2 aca

            subprocess.check_output("./TP0.out a 250 "+str(d)+" 0.78", shell=True, universal_newlines=True)
            os.rename('ej.data', 'ej-'+str(d)+'.data')

            print("Entrenando red numero " + str(i))
            output= subprocess.check_output("./nb_n.out ej-"+str(d), shell=True, universal_newlines=True)
            s= output.split('Errores:', 1)[1]
            #print(output)
            s2=get_values(s)
            #print(s2)
            valores_guardados.append(s2)
            time.sleep(1)

        r=[]
        #print(valores_guardados)
        for l in range(0,len(valores_guardados[0])):
            v_list=map(lambda x: x[l],valores_guardados)
            #print(v_list)
            v=sum(v_list,0)/len(valores_guardados)
            r.append(v)

        dif=[]
        m=0 #voy a tomar el indice de la red con la diferencia del error de Test en "Error minimo" mas chica para decidir cual es el que mejor se aproxima a la mediana
        for v in valores_guardados:
            dp=[]
            for i in range(0,len(r)):
                dp.append(abs(v[i]-r[i]))
            dif.append(dp)
            if dp[2]<dif[m][2]:
                    m=len(dif)-1


        for i in range(0,len(dif)):
            print("\nDiferencia de errores de la pasada n "+str(i)+": "+str(dif[i][2]))

        print("Nos quedamos con la red n "+str(m)+" por que es la mas cercana a la media segun el error de Test en \'Error minimo\'")
        


        f.write(str(valores_guardados[m][0])+" "+str(valores_guardados[m][1])+" "+str(valores_guardados[m][2])+"\n")
    f.close()

#ej1()
ej1_2()
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



