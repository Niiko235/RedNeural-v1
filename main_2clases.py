import numpy as np

from generarDatos import generar_datos_3d_2Nubes
from redNeural import redNeural
from exportData import (export_para_unity_2clases,
                        export_historial_unity_2clases,
                        export_pesos_individuales_unity_2clases)

# ── Parámetros ─────────────────────────────────────────────────────────────────
EPOCAS               = 1000
TAZA_APRENDIZAJE     = 0.01
NEURONAS_CAPA_OCULTA = 5
TOLERANCIA           = 0.01

# ── Generar datos: 2 nubes, 3 entradas (x, y, z) ──────────────────────────────
x1, y1, z1, x2, y2, z2, valor_esperado = generar_datos_3d_2Nubes()

entradas_1 = np.column_stack((x1, y1, z1))
entradas_2 = np.column_stack((x2, y2, z2))
entradas   = np.vstack((entradas_1, entradas_2))   # 200 puntos × 3 coords

print(f"Datos generados: {entradas.shape[0]} puntos, {entradas.shape[1]} entradas, 2 clases\n")

# ── Historial para exportar después ───────────────────────────────────────────
epocas_hist  = []
errores_hist = []
hist_W1, hist_W2, hist_b1, hist_b2 = [], [], [], []

def guardar_historial(epoca, error_total, W1, W2, b1, b2):
    epocas_hist.append(epoca)
    errores_hist.append(error_total)
    hist_W1.append(W1.flatten().copy())
    hist_W2.append(W2.flatten().copy())
    hist_b1.append(b1.flatten().copy())
    hist_b2.append(b2.flatten().copy())

# ── Entrenar ───────────────────────────────────────────────────────────────────
W1, W2, b1, b2 = redNeural(
    entradas, valor_esperado,
    epocas=EPOCAS,
    taza_aprendizaje=TAZA_APRENDIZAJE,
    numero_neuronas_capa_oculta=NEURONAS_CAPA_OCULTA,
    tolerancia=TOLERANCIA,
    numero_clases=2,
    callback=guardar_historial
)

# ── Exportar a datosExportados2Clases/ ────────────────────────────────────────
export_para_unity_2clases(entradas, valor_esperado, W1, W2, b1, b2)

norma_W1_hist = [np.linalg.norm(w.reshape(3, -1)) for w in hist_W1]
norma_W2_hist = [np.linalg.norm(w.reshape(-1, 2)) for w in hist_W2]
export_historial_unity_2clases(epocas_hist, errores_hist, norma_W1_hist, norma_W2_hist)

export_pesos_individuales_unity_2clases(epocas_hist, hist_W1, hist_W2, hist_b1, hist_b2)

print("\n=== LISTO ===")
print("Archivos generados en ./datosExportados2Clases/:")
print("  puntos.csv        → 200 puntos con x,y,z,clase")
print("  W1.csv            → pesos entrada→oculta  (3 × 5)")
print("  W2.csv            → pesos oculta→salida   (5 × 2)")
print("  b1.csv            → bias capa oculta      (1 × 5)")
print("  b2.csv            → bias capa salida      (1 × 2)")
print("  historial.csv     → error por época")
print("  pesos_historial.csv → todos los pesos por época")
