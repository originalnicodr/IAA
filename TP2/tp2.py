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
    
def ej5():
    valores_guardados=[]
    for i in range(0,10):
        #estoy corriendo python 2 aca
        print("Entrenando red numero " + str(i))
        output= subprocess.check_output("./bp-mod1.o ssp", shell=True, universal_newlines=True)
        s= output.split('Error final:', 1)[1]
        #print(s)
        valores_guardados.append(get_values(s))
        os.rename('ssp.mse', 'ssp_'+str(i)+'.mse')
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
        if dp[1]<dif[m][1]:
                m=len(dif)-1
    
    
    for i in range(0,len(dif)):
        print("\nDiferencia de errores de la pasada n "+str(i))
        print(imprimir_errores(dif[i]))

    print("Nos quedamos con la red n "+str(m)+" por que es la mas cercana a la media segun el error de Test en \'Error minimo\'")
    
def ej6():
    f = open("ej6.txt", "a")
    f.write("Error(est) Error(med) ValidacionFinal TestFinal Epoca Archivo ValidacionMinima TestMinimo TestDiscreto\n")

    valoresd=[2, 4, 8, 16, 32]
    for d in valoresd:
        valores_guardados=[]
        subprocess.check_output("./TP0.out b 10000 "+str(d)+" 0.78", shell=True, universal_newlines=True)
        os.rename('ej.data', 'ej-'+str(d)+'.test')
        subprocess.check_output("./TP0.out b 250 "+str(d)+" 0.78", shell=True, universal_newlines=True)
        os.rename('ej.data', 'ej-'+str(d)+'.data')


        for i in range(0,10):
            #estoy corriendo python 2 aca
            print("Entrenando red numero " + str(i))
            output= subprocess.check_output("./bp.o ej-"+str(d), shell=True, universal_newlines=True)
            s= output.split('Error final:', 1)[1]
            #print(output)
            s2=get_values(s)
            #print(s2)
            valores_guardados.append(s2)
            os.rename('ej-'+str(d)+'.mse', 'ej-'+str(d)+'_'+str(i)+'.mse')
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
            if dp[1]<dif[m][1]:
                    m=len(dif)-1


        for i in range(0,len(dif)):
            print("\nDiferencia de errores de la pasada n "+str(i))
            print(imprimir_errores(dif[i]))

        print("Nos quedamos con la red n "+str(m)+" por que es la mas cercana a la media segun el error de Test en \'Error minimo\'")
        #Error(est) Error(med)  ValidacionFinal  TestFinal    ValidacionMinima  TestMinimo  TestDiscreto
        
        f.write(str(valores_guardados[m][0])+" "+str(valores_guardados[m][1])+" "+str(valores_guardados[m][2])+" "+str(valores_guardados[m][3])+" "+str(valores_guardados[m][4])+" "+str(m)+" "+str(valores_guardados[m][5])+" "+str(valores_guardados[m][6])+" "+str(valores_guardados[m][7])+"\n")


#ej3()
#ej4()
#ej5()
ej6()
