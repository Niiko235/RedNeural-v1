import numpy as np
from generarDatos import generar_datos_3d_3Nubes
from redNeural import redNeural, predecir, matriz_confusion
from graficas import graficar

# ── Parámetros ajustables ──────────────────────────────────────────────────────
EPOCAS               = 1000   # máximo de vueltas de entrenamiento
TAZA_APRENDIZAJE     = 0.01   # qué tan grandes son los pasos al ajustar pesos
NEURONAS_CAPA_OCULTA = 16     # neuronas en la capa intermedia

# TOLERANCIA: umbral de error para detener el entrenamiento antes de agotar épocas.
# La red suma el error de los 300 puntos en cada época. Cuando esa suma baja
# del valor de TOLERANCIA, el entrenamiento se detiene automáticamente.
#
# Ejemplos de comportamiento:
#   TOLERANCIA = 0.1    → para muy rápido (~50 épocas),  red poco entrenada, baja precisión
#   TOLERANCIA = 0.01   → para rápido     (~200 épocas), precisión aceptable
#   TOLERANCIA = 0.001  → balance bueno   (~500 épocas), buena precisión       ← ACTUAL
#   TOLERANCIA = 0.0001 → entrena más     (~900 épocas), mayor precisión, más lento
#   TOLERANCIA = 0.00001→ casi nunca para antes de agotar las 1000 épocas
#
# Si el error nunca baja de TOLERANCIA, la red igual termina al llegar a EPOCAS.
TOLERANCIA           = 0.001

# ── Generar datos ──────────────────────────────────────────────────────────────
x1,y1,z1, x2,y2,z2, x3,y3,z3, valor_esperado = generar_datos_3d_3Nubes()

entradas_1 = np.column_stack((x1, y1, z1))
entradas_2 = np.column_stack((x2, y2, z2))
entradas_3 = np.column_stack((x3, y3, z3))
entradas   = np.vstack((entradas_1, entradas_2, entradas_3))

# ── Entrenar ───────────────────────────────────────────────────────────────────
W1, W2, b1, b2 = redNeural(
    entradas, valor_esperado,
    epocas=EPOCAS,
    taza_aprendizaje=TAZA_APRENDIZAJE,
    numero_neuronas_capa_oculta=NEURONAS_CAPA_OCULTA,
    tolerancia=TOLERANCIA
)

# ── Evaluar ────────────────────────────────────────────────────────────────────
matriz_confusion(entradas, valor_esperado, W1, W2, b1, b2)

# ── Graficar ───────────────────────────────────────────────────────────────────
graficar(entradas, W1, W2, b1, b2)
