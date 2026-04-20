import numpy as np

from generarDatos import generar_datos_3d_3Nubes
from redNeural    import redNeural
from graficasEnVivo import GraficasEnVivo
from graficas     import graficar
from exportData   import (export_para_unity,
                          export_historial_unity,
                          export_pesos_individuales_unity)

# ── Parámetros ─────────────────────────────────────────────────────────────────
EPOCAS               = 1000
TAZA_APRENDIZAJE     = 0.01
NEURONAS_CAPA_OCULTA = 3
TOLERANCIA           = 0.5

# ── Generar datos ──────────────────────────────────────────────────────────────
x1,y1,z1, x2,y2,z2, x3,y3,z3, valor_esperado = generar_datos_3d_3Nubes()

entradas = np.vstack([
    np.column_stack((x1, y1, z1)),
    np.column_stack((x2, y2, z2)),
    np.column_stack((x3, y3, z3)),
])

# ── Gráficas en tiempo real ────────────────────────────────────────────────────
graficas = GraficasEnVivo(tolerancia=TOLERANCIA)

# ── Entrenar ───────────────────────────────────────────────────────────────────
W1, W2, b1, b2 = redNeural(
    entradas, valor_esperado,
    epocas=EPOCAS,
    taza_aprendizaje=TAZA_APRENDIZAJE,
    numero_neuronas_capa_oculta=NEURONAS_CAPA_OCULTA,
    tolerancia=TOLERANCIA,
    callback=graficas.actualizar,
)

graficas.cerrar_modo_interactivo()

# ── Exportar datos para Unity ──────────────────────────────────────────────────
export_para_unity(entradas, valor_esperado, W1, W2, b1, b2)
export_historial_unity(graficas.epocas, graficas.errores,
                       graficas.norma_W1, graficas.norma_W2)
export_pesos_individuales_unity(graficas.epocas,
                                graficas.hist_W1, graficas.hist_W2,
                                graficas.hist_b1, graficas.hist_b2)

# ── Mostrar fronteras de decisión ─────────────────────────────────────────────
graficar(entradas, W1, W2, b1, b2)
