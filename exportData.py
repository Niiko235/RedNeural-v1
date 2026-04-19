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


def export_para_unity(entradas, valor_esperado, W1, W2, b1, b2):
    """
    Exporta todos los datos necesarios para que Unity reconstruya la visualización.
    Genera 5 archivos CSV en la carpeta datosExportados/:
      puntos.csv → coordenadas de los 300 puntos con su clase (0, 1 o 2)
      W1.csv     → pesos entrada→oculta  (3 × N)
      W2.csv     → pesos oculta→salida   (N × 3)
      b1.csv     → bias capa oculta      (1 × N)
      b2.csv     → bias capa salida      (1 × 3)
    """
    import os
    os.makedirs('./datosExportados', exist_ok=True)

    # Convertir etiquetas one-hot a índice de clase (0, 1 o 2)
    clases = np.argmax(valor_esperado, axis=1)

    # Puntos: x, y, z, clase
    puntos = np.column_stack((entradas, clases))
    # fmt mixto: 3 decimales para x,y,z y entero para clase
    np.savetxt('./datosExportados/puntos.csv', puntos,
               delimiter=',', header='x,y,z,clase', comments='',
               fmt=['%.6f', '%.6f', '%.6f', '%d'])

    # Pesos y bias (Unity los lee fila por fila)
    np.savetxt('./datosExportados/W1.csv', W1, delimiter=',', fmt='%.8f')
    np.savetxt('./datosExportados/W2.csv', W2, delimiter=',', fmt='%.8f')
    np.savetxt('./datosExportados/b1.csv', b1, delimiter=',', fmt='%.8f')
    np.savetxt('./datosExportados/b2.csv', b2, delimiter=',', fmt='%.8f')

    print("✓ Datos exportados para Unity en ./datosExportados/")


def export_historial_unity(epocas_hist, errores_hist, norma_W1_hist, norma_W2_hist):
    """
    Exporta el historial del entrenamiento para que Unity pueda animar las gráficas.
    Columnas: epoca, error, norma_W1, norma_W2
    norma_W1/W2 es la magnitud total de cada matriz de pesos (un número por época).
    """
    import os
    os.makedirs('./datosExportados', exist_ok=True)

    data = np.column_stack((epocas_hist, errores_hist, norma_W1_hist, norma_W2_hist))
    np.savetxt('./datosExportados/historial.csv', data,
               delimiter=',', header='epoca,error,norma_W1,norma_W2',
               comments='', fmt='%.6f')
    print("✓ Historial de entrenamiento exportado para Unity.")


def export_pesos_individuales_unity(epocas_hist, hist_W1, hist_W2, hist_b1, hist_b2):
    """
    Exporta el valor de cada peso y bias por época para que Unity muestre
    todas las líneas individuales (igual que Python en la gráfica de pesos).

    Columnas del CSV:
      epoca, w1_0, w1_1, ..., w2_0, w2_1, ..., b1_0, ..., b2_0, ...
    """
    import os
    os.makedirs('./datosExportados', exist_ok=True)

    arr_W1 = np.array(hist_W1)   # (épocas, n_W1)
    arr_W2 = np.array(hist_W2)   # (épocas, n_W2)
    arr_b1 = np.array(hist_b1)   # (épocas, n_b1)
    arr_b2 = np.array(hist_b2)   # (épocas, n_b2)

    n_W1, n_W2 = arr_W1.shape[1], arr_W2.shape[1]
    n_b1, n_b2 = arr_b1.shape[1], arr_b2.shape[1]

    headers = (['epoca']
               + [f'w1_{i}' for i in range(n_W1)]
               + [f'w2_{i}' for i in range(n_W2)]
               + [f'b1_{i}' for i in range(n_b1)]
               + [f'b2_{i}' for i in range(n_b2)])

    data = np.column_stack((epocas_hist, arr_W1, arr_W2, arr_b1, arr_b2))
    np.savetxt('./datosExportados/pesos_historial.csv', data,
               delimiter=',', header=','.join(headers), comments='', fmt='%.6f')
    print(f"✓ Pesos individuales exportados: W1({n_W1}) W2({n_W2}) b1({n_b1}) b2({n_b2}) por época.")


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
