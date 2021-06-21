import subprocess
import re
import os
import sys
import time
import numpy as np

def get_values(s):
    return map(lambda x: float(x),re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s))

def correr_hasta(nombreDatos,max_dist, step):
    valores=[]
    min=step
    step_list=np.arange(step,max_dist,step)
    for n in step_list:
        
        output= subprocess.check_output("./k-nn-d.out "+nombreDatos+" "+str(n), shell=True, universal_newlines=True)
        #print("Comando: ./k-nn-d.out "+nombreDatos+" "+str(n))
        os.rename(nombreDatos+".predic", nombreDatos+"-"+str(n)+".predic")
        #print("Genere: "+nombreDatos+".predic", nombreDatos+"-"+str(n)+".predic")
        s= output.split('Errores:', 1)[1]
        print(output)
        valores.append(get_values(s))
        #print(n)
        
        if valores[int((n-step)/step)][1]<valores[int((min-step)/step)][1]: #comparo errores de validacion
            min=n
    #print("Cambio: "+nombreDatos+"-"+str(min)+".predic")
    os.rename(nombreDatos+"-"+str(min)+".predic", nombreDatos+".predic")
    
    for n in step_list:
        if n != min: #para evitar que intente eliminar al que cambie de nombre
            os.remove(nombreDatos+"-"+str(n)+".predic")
    

    print("Distancia maxima optima: "+str(min)+"\nEntrenamiento:"+str(valores[int((min-step)/step)][0])+"%\nValidacion:"+str(valores[int((min-step)/step)][1])+"%\nTest:"+str(valores[int((min-step)/step)][2])+"%")

    return valores[int((min-step)/step)]

def correr_d(nombreDatos,max_dist):

    output= subprocess.check_output("./k-nn-d.out "+nombreDatos+" "+str(max_dist), shell=True, universal_newlines=True)
    #print(output)
    s= output.split('Errores:', 1)[1]
    valores=get_values(s)
    #print(n)
    os.remove(nombreDatos+".predic")

    print("Distancia maxima optima: "+str(max_dist)+"\nEntrenamiento:"+str(valores[0])+"%\nValidacion:"+str(valores[1])+"%\nTest:"+str(valores[2])+"%")
    return valores

def ejd(ejercicio,max_dist,step): #ejercicio= a o b
    
    f = open("ejd.txt", "w")
    f.write("ErrorEntrenamiento ErrorValidacion ErrorTest\n")

    valoresd=[2, 4, 8, 16, 32]
    for d in valoresd:
        print("Ejecutando con d: "+str(d))
        nb = open("ej-"+str(d)+".kf", "w")
        nb.write(str(d)+'\n2\n250\n200\n10000\n0\n0')
        nb.close()

        valores_guardados=[]
        subprocess.check_output("./TP0.out "+ejercicio+" 10000 "+str(d)+" 0.78", shell=True, universal_newlines=True)
        #print("Comando: ./TP0.out "+ejercicio+" 10000 "+str(d)+" 0.78")
        os.rename('ej.data', 'ej-'+str(d)+'.test')


        subprocess.check_output("./TP0.out "+ejercicio+" 250 "+str(d)+" 0.78", shell=True, universal_newlines=True)
        os.rename('ej.data', 'ej-'+str(d)+'.data')
        #print("Comando: ./TP0.out "+ejercicio+" 250 "+str(d)+" 0.78")

        if max_dist==step:
            #print("max_dist==step?")
            v=correr_d('ej-'+str(d),max_dist)
        else:
            #print("max_dist!=step")
            v=correr_hasta('ej-'+str(d),max_dist,step)
        
        f.write(str(v[0])+" "+str(v[1])+" "+str(v[2])+"\n")
    f.close()



def test():
    output= subprocess.check_output("./k-nn-d.out test 8", shell=True, universal_newlines=True)
    f = open("testeando sorter.txt", "w")
    f.write(output)
    f.close()

def main():
    # print command line arguments
    if len(sys.argv) != 3:
        print("Modo de uso: python k-nn-d.py <ejercicio> <max_dist>\ndonde ejercicio puede ser a o b\ny max_dist es la maxima distancia utilizada")
        return
    #correr_hasta(sys.argv[1],int(sys.argv[2]))
    ejd(sys.argv[1],float(sys.argv[2]),0.1)
    #correr_hasta(sys.argv[1],0.5, 0.1)


if __name__ == "__main__":
    main()

