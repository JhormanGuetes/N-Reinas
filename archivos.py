import numpy as np

def leerArchivo(nombreArchivo):
        ruta_archivo = './dimensiones/'+nombreArchivo
        matrix = np.loadtxt(ruta_archivo, skiprows=0)
        return matrix

def numeroDeColumnas(nombre_archivo):
        matrix = leerArchivo(nombre_archivo)
        return matrix.shape[0]

#print(numeroDeColumnas("4x4.txt"))