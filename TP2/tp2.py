import re
from functools import reduce
import subprocess
import time
import os
import math

def get_values(s):
    return map(lambda x: float(x),re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s))

def test():
    
    output= subprocess.check_output("./bp.o espirales", shell=True, universal_newlines=True)
    s= output.split('Error final:', 1)[1]

    """
    Error final:
    s[0]= Entrenamiento(est)
    s[1]= Entrenamiento(med)
    s[2]= Validacion
    s[3]= Test

    Error minimo en validacion:
    s[4]= Epoca
    s[5]= Validacion
    s[6]= Test
    s[7]= Test discreto
    """
    print(get_values(s))


def imprimir_errores(eList):
    s="""
Error final:
Entrenamiento(est): """ + str(eList[0]) + """
Entrenamiento(med): """ + str(eList[1]) + """
Validacion: """ + str(eList[2]) + """
Test: """ + str(eList[3]) + """

Error minimo en validacion:
Epoca: """ + str(eList[4]) + """
Validacion: """ + str(eList[5]) + """
Test: """ + str(eList[6]) + """
Test discreto: """ + str(eList[7])
    return s

def ej3():
    valores_guardados=[]
    for i in range(0,10):
        #estoy corriendo python 2 aca
        print("Entrenando red numero " + str(i))
        output= subprocess.check_output("./bp.o dos_elipses", shell=True, universal_newlines=True)
        s= output.split('Error final:', 1)[1]
        valores_guardados.append(get_values(s))
        os.rename('dos_elipses.mse', 'dos_elipses_'+str(i)+'.mse')
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
        if dp[6]<dif[m][6]:
                m=len(dif)-1
    
    
    for i in range(0,len(dif)):
        print("\nDiferencia de errores de la pasada n "+str(i))
        print(imprimir_errores(dif[i]))

    print("Nos quedamos con la red n "+str(m)+" por que es la mas cercana a la media segun el error de Test en \'Error minimo\'")
    

def ej4():
    valores_guardados=[]
    for i in range(0,10):
        #estoy corriendo python 2 aca
        print("Entrenando red numero " + str(i))
        output= subprocess.check_output("./bp.o ikeda", shell=True, universal_newlines=True)
        s= output.split('Error final:', 1)[1]
        valores_guardados.append(get_values(s))
        os.rename('ikeda.mse', 'ikeda_'+str(i)+'.mse')
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
        if dp[6]<dif[m][6]:
                m=len(dif)-1
    
    
    for i in range(0,len(dif)):
        print("\nDiferencia de errores de la pasada n "+str(i))
        print(imprimir_errores(dif[i]))

    print("Nos quedamos con la red n "+str(m)+" por que es la mas cercana a la media segun el error de Test en \'Error minimo\'")
    






#ej3()
ej4()
#output="""Error final:
#Entrenamiento(est):0.191583
#Entrenamiento(med):0.191829
#Validacion:0.191084
#Test:0.191472"""
#s= output.split('Error final:', 1)[1]
#print(get_values(s))