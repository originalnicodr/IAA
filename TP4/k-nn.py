import subprocess
import re
import os
import sys
import time

def get_values(s):
    return map(lambda x: float(x),re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s))

def correr_hasta(nombreDatos,neighbours):
    valores=[]
    min=1
    for n in range(1,neighbours):
        output= subprocess.check_output("./k-nn.out "+nombreDatos+" "+str(n), shell=True, universal_newlines=True)
        os.rename(nombreDatos+".predic", nombreDatos+"-"+str(n)+".predic")
        #print(output)
        s= output.split('Errores:', 1)[1]
        valores.append(get_values(s))
        #print(n)
        if valores[n-1][1]<valores[min-1][1]: #comparo errores de validacion
            min=n
        print("Cantidad de Vecinos: "+str(n)+"\nEntrenamiento:"+str(valores[n-1][0])+"%\nValidacion:"+str(valores[n-1][1])+"%\nTest:"+str(valores[n-1][2])+"%\n\n")

    os.rename(nombreDatos+"-"+str(min)+".predic", nombreDatos+".predic")
    for n in range(1,neighbours):
        if n != min: #para evitar que intente eliminar al que cambie de nombre
            os.remove(nombreDatos+"-"+str(n)+".predic")

    print("Cantidad de Vecinos Optima: "+str(min)+"\nEntrenamiento:"+str(valores[min-1][0])+"%\nValidacion:"+str(valores[min-1][1])+"%\nTest:"+str(valores[min-1][2])+"%")

    """
    f = open("Errores.txt", "w")
    f.write("ErrorEntrenamiento ErrorValidacion ErrorTest\n")
    for i in valores:
        f.write(str(i[0])+" "+str(i[1])+" "+str(i[2])+"\n")
    f.close()
    """

    return valores[min-1]

def correr_k(nombreDatos,neighbours):

    output= subprocess.check_output("./k-nn.out "+nombreDatos+" "+str(neighbours), shell=True, universal_newlines=True)
    #print(output)
    s= output.split('Errores:', 1)[1]
    valores=get_values(s)
    #print(n)
    os.remove(nombreDatos+".predic")

    print("Cantidad de Vecinos Optima: "+str(neighbours)+"\nEntrenamiento:"+str(valores[0])+"%\nValidacion:"+str(valores[1])+"%\nTest:"+str(valores[2])+"%")
    return valores

def ejc(ejercicio,neighbours): #ejercicio= a o b
    
    f = open("ejc.txt", "w")
    f.write("ErrorEntrenamiento ErrorValidacion ErrorTest\n")

    valoresd=[2, 4, 8, 16, 32]
    for d in valoresd:
        nb = open("ej-"+str(d)+".kf", "w")
        nb.write(str(d)+'\n2\n250\n188\n10000\n0\n0')
        nb.close()


        subprocess.check_output("./TP0.out "+ejercicio+" 10000 "+str(d)+" 0.78", shell=True, universal_newlines=True)
        os.rename('ej.data', 'ej-'+str(d)+'.test')

        subprocess.check_output("./TP0.out "+ejercicio+" 250 "+str(d)+" 0.78", shell=True, universal_newlines=True)
        os.rename('ej.data', 'ej-'+str(d)+'.data')

        if neighbours==1:
            v=correr_k('ej-'+str(d),neighbours)
        else:
            v=correr_hasta('ej-'+str(d),neighbours)
            
        f.write(str(v[0])+" "+str(v[1])+" "+str(v[2])+"\n")
    f.close()



def test():
    output= subprocess.check_output("./k-nn.out test 8", shell=True, universal_newlines=True)
    f = open("testeando sorter.txt", "w")
    f.write(output)
    f.close()





def main():
    # print command line arguments
    if len(sys.argv) != 3:
        print("Modo de uso: python k-nn.py <filename> <neighbours>\ndonde filename es el nombre del archivo (sin extension)\ny neighbours es el numero maximo de vecinos usados")
        return
    #correr_hasta(sys.argv[1],int(sys.argv[2]))
    ejc(sys.argv[1],int(sys.argv[2]))


if __name__ == "__main__":
    main()

