import numpy as np
from generarDatos import generar_datos_3d_3Nubes
from redNeural import redNeural, predecir, matriz_confusion
from graficas import graficar  # ← importas la función

x1,y1,z1, x2,y2,z2, x3,y3,z3, valor_esperado = generar_datos_3d_3Nubes()

entradas_1 = np.column_stack((x1, y1, z1))
entradas_2 = np.column_stack((x2, y2, z2))
entradas_3 = np.column_stack((x3, y3, z3))
entradas   = np.vstack((entradas_1, entradas_2, entradas_3))

# Entrenar
W1, W2, b1, b2 = redNeural(entradas, valor_esperado, epocas=1000, taza_aprendizaje=0.01, numero_neuronas_capa_oculta=16)

# Evaluar
matriz_confusion(entradas, valor_esperado, W1, W2, b1, b2)

# Graficar ← solo agregas esta línea
graficar(entradas, W1, W2, b1, b2)