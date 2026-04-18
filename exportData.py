import numpy as np


def export_3nubes_3d(entradas, resultado_esperado1, resultado_esperado2):
    """
    Exporta los puntos de entrenamiento a un archivo CSV.

    Columnas del CSV:
      x, y, z          → coordenadas del punto en 3D
      resultado_esperado1, resultado_esperado2 → etiquetas de clase (one-hot parcial)
    """
    # column_stack une arrays como columnas de una sola matriz
    data = np.column_stack((entradas, resultado_esperado1, resultado_esperado2))

    # savetxt guarda la matriz como archivo de texto separado por comas
    np.savetxt('./datosExportados/entradas_3d_3nubes.csv', data,
               delimiter=',',
               header='x,y,z,resultado_esperado1,resultado_esperado2',
               comments='')   # comments='' evita que escriba '#' al inicio del header


def export_epocas_3nubes_3d(error, pesos, bias, nombre):
    """
    Exporta el historial de entrenamiento (error, pesos y bias por época) a CSV.

    Parámetros
    ----------
    error  : array con el error de cada época
    pesos  : array con los valores de los pesos en cada época
    bias   : array con los valores de los bias en cada época
    nombre : nombre del archivo CSV a crear (sin extensión)
    """
    data = np.column_stack((error, pesos, bias))

    np.savetxt(f'./datosExportados/{nombre}.csv', data,
               delimiter=',',
               header='error,peso1,peso2,peso3,bias',
               comments='')
