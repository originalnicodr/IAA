import subprocess
import re
import os
import sys

def get_values(s):
    return map(lambda x: float(x),re.findall(r"[+-]? *(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", s))

def ej4(nombreDatos,bins):
    valores=[]
    min=1
    for n in range(1,bins):
        output= subprocess.check_output("./nb_n_4.out "+nombreDatos+" "+str(n), shell=True, universal_newlines=True)
        os.rename(nombreDatos+".predic", nombreDatos+"-"+str(n)+".predic")
        #print(output)
        s= output.split('Errores:', 1)[1]
        valores.append(get_values(s))
        #print(n)
        if valores[n-1][1]<valores[min-1][1]: #comparo errores de validacion
            min=n

    os.rename(nombreDatos+"-"+str(min)+".predic", nombreDatos+".predic")
    for n in range(1,bins):
        if n != min: #para evitar que intente eliminar al que cambie de nombre
            os.remove(nombreDatos+"-"+str(n)+".predic")

    print("Bin optimo: "+str(min)+"\nEntrenamiento:"+str(valores[min-2][0])+"%\nValidacion:"+str(valores[min-2][1])+"%\nTest:"+str(valores[min-2][2])+"%")

    f = open("ej4.txt", "w")
    f.write("ErrorEntrenamiento ErrorValidacion ErrorTest\n")
    for i in valores:
        f.write(str(i[0])+" "+str(i[1])+" "+str(i[2])+"\n")
    f.close()


def main():
    # print command line arguments
    if len(sys.argv) != 3:
        print("Modo de uso: python ej4.py <filename> <binnum>\ndonde filename es el nombre del archivo (sin extension)\ny binnum es el numero maximo de bins usado")
        return
    ej4(sys.argv[1],int(sys.argv[2]))


if __name__ == "__main__":
    main()

