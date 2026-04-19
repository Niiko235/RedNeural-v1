import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from generarDatos import generar_datos_3d_3Nubes
from redNeural import redNeural
from graficas import graficar
from exportData import export_para_unity, export_historial_unity, export_pesos_individuales_unity

# ── Parámetros ajustables ──────────────────────────────────────────────────────
EPOCAS               = 50
TAZA_APRENDIZAJE     = 0.01
NEURONAS_CAPA_OCULTA = 3

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
TOLERANCIA           = 0.5

# ── Generar datos ──────────────────────────────────────────────────────────────
x1,y1,z1, x2,y2,z2, x3,y3,z3, valor_esperado = generar_datos_3d_3Nubes()

entradas_1 = np.column_stack((x1, y1, z1))
entradas_2 = np.column_stack((x2, y2, z2))
entradas_3 = np.column_stack((x3, y3, z3))
entradas   = np.vstack((entradas_1, entradas_2, entradas_3))

# ── Gráficas en tiempo real ────────────────────────────────────────────────────
FONDO = '#0f0f1a'

# Activar modo interactivo: permite actualizar gráficas sin bloquear el programa
plt.ion()

fig_live = plt.figure(figsize=(13, 5))
fig_live.patch.set_facecolor(FONDO)
fig_live.suptitle('Entrenamiento en tiempo real', color='white', fontsize=12)

gs = gridspec.GridSpec(1, 2, figure=fig_live,
                       left=0.08, right=0.97, top=0.88, bottom=0.12,
                       wspace=0.35)

ax_error  = fig_live.add_subplot(gs[0, 0])   # gráfica de error por época
ax_pesos  = fig_live.add_subplot(gs[0, 1])   # gráfica de pesos por época

# Estilo oscuro para ambos paneles
for ax in [ax_error, ax_pesos]:
    ax.set_facecolor('#12122a')
    ax.tick_params(colors='#aaaacc', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

ax_error.set_title('Error cuadrático por época',  color='white', fontsize=10)
ax_error.set_xlabel('Época',       color='#aaaacc', fontsize=9)
ax_error.set_ylabel('Error total', color='#aaaacc', fontsize=9)

ax_pesos.set_title('Magnitud de pesos por época', color='white', fontsize=10)
ax_pesos.set_xlabel('Época',        color='#aaaacc', fontsize=9)
ax_pesos.set_ylabel('Norma (||W||)', color='#aaaacc', fontsize=9)

plt.show()   # mostrar la ventana vacía antes de entrenar

# Historial acumulado durante el entrenamiento
# Cada fila = una época, cada columna = un peso individual
epocas_hist  = []
errores_hist = []
hist_W1 = []   # (épocas, 48)  — 3×16 pesos entrada→oculta
hist_W2 = []   # (épocas, 48)  — 16×3 pesos oculta→salida
hist_b1 = []   # (épocas, 16)  — bias capa oculta
hist_b2 = []   # (épocas,  3)  — bias capa salida

# Actualizar cada 5 épocas para que la animación sea fluida sin ser lenta
INTERVALO_REFRESCO = 5


def actualizar_graficas(epoca, error_total, W1, W2, b1, b2):
    """
    Callback que se llama al final de cada época.
    Guarda el valor individual de cada peso y redibuja ambas gráficas.
    Cada peso se muestra como su propia línea en la gráfica de pesos.
    """
    epocas_hist.append(epoca)
    errores_hist.append(error_total)

    # Guardar copia aplanada de todos los pesos (flatten los convierte en 1D)
    hist_W1.append(W1.flatten().copy())
    hist_W2.append(W2.flatten().copy())
    hist_b1.append(b1.flatten().copy())   # 16 valores
    hist_b2.append(b2.flatten().copy())   #  3 valores

    # Solo redibujar cada INTERVALO_REFRESCO épocas para no ir lento
    if epoca % INTERVALO_REFRESCO != 0:
        return

    arr_W1 = np.array(hist_W1)   # (épocas, 48)
    arr_W2 = np.array(hist_W2)   # (épocas, 48)
    arr_b1 = np.array(hist_b1)   # (épocas, 16)
    arr_b2 = np.array(hist_b2)   # (épocas,  3)

    # ── Gráfica 1: error cuadrático ───────────────────────────────────────────
    ax_error.clear()
    ax_error.set_facecolor('#12122a')
    ax_error.plot(epocas_hist, errores_hist, color='tomato', linewidth=1.8)
    ax_error.axhline(y=TOLERANCIA, color='yellow', linewidth=0.9,
                     linestyle='--', alpha=0.8, label=f'Tolerancia ({TOLERANCIA})')
    ax_error.set_title('Error cuadrático por época', color='white', fontsize=10)
    ax_error.set_xlabel('Época',       color='#aaaacc', fontsize=9)
    ax_error.set_ylabel('Error total', color='#aaaacc', fontsize=9)
    ax_error.tick_params(colors='#aaaacc', labelsize=8)
    ax_error.legend(facecolor='#1a1a2e', edgecolor='#444466',
                    labelcolor='white', fontsize=8)
    for spine in ax_error.spines.values():
        spine.set_edgecolor('#333355')

    # ── Gráfica 2: todos los pesos individuales ───────────────────────────────
    ax_pesos.clear()
    ax_pesos.set_facecolor('#12122a')

    # W1: 48 pesos (entrada→oculta) — azul, líneas finas semitransparentes
    for j in range(arr_W1.shape[1]):
        ax_pesos.plot(epocas_hist, arr_W1[:, j],
                      color='royalblue', linewidth=0.6, alpha=0.35)

    # W2: 48 pesos (oculta→salida) — verde
    for j in range(arr_W2.shape[1]):
        ax_pesos.plot(epocas_hist, arr_W2[:, j],
                      color='limegreen', linewidth=0.6, alpha=0.35)

    # b1: 16 bias (capa oculta) — cian
    for j in range(arr_b1.shape[1]):
        ax_pesos.plot(epocas_hist, arr_b1[:, j],
                      color='cyan', linewidth=0.6, alpha=0.45)

    # b2: 3 bias (capa salida) — naranja
    for j in range(arr_b2.shape[1]):
        ax_pesos.plot(epocas_hist, arr_b2[:, j],
                      color='orange', linewidth=0.9, alpha=0.7)

    # Leyenda con una línea representativa por grupo
    from matplotlib.lines import Line2D
    leyenda = [
        Line2D([0],[0], color='royalblue', linewidth=1.5, label=f'W1 — {arr_W1.shape[1]} pesos (entrada→oculta)'),
        Line2D([0],[0], color='limegreen', linewidth=1.5, label=f'W2 — {arr_W2.shape[1]} pesos (oculta→salida)'),
        Line2D([0],[0], color='cyan',      linewidth=1.5, label=f'b1 — {arr_b1.shape[1]} bias (oculta)'),
        Line2D([0],[0], color='orange',    linewidth=1.5, label=f'b2 — {arr_b2.shape[1]} bias (salida)'),
    ]
    ax_pesos.legend(handles=leyenda, facecolor='#1a1a2e', edgecolor='#444466',
                    labelcolor='white', fontsize=7)

    ax_pesos.set_title('Pesos individuales por época', color='white', fontsize=10)
    ax_pesos.set_xlabel('Época',  color='#aaaacc', fontsize=9)
    ax_pesos.set_ylabel('Valor del peso', color='#aaaacc', fontsize=9)
    ax_pesos.tick_params(colors='#aaaacc', labelsize=8)
    for spine in ax_pesos.spines.values():
        spine.set_edgecolor('#333355')

    # Redibujar la ventana sin bloquear el entrenamiento
    fig_live.canvas.draw()
    fig_live.canvas.flush_events()


# ── Entrenar ───────────────────────────────────────────────────────────────────
W1, W2, b1, b2 = redNeural(
    entradas, valor_esperado,
    epocas=EPOCAS,
    taza_aprendizaje=TAZA_APRENDIZAJE,
    numero_neuronas_capa_oculta=NEURONAS_CAPA_OCULTA,
    tolerancia=TOLERANCIA,
    callback=actualizar_graficas   # pasa la función para actualizar en tiempo real
)

# Desactivar modo interactivo: las gráficas quedan fijas al terminar
plt.ioff()


# ── Exportar datos para Unity ──────────────────────────────────────────────────
export_para_unity(entradas, valor_esperado, W1, W2, b1, b2)

# Calcular norma (magnitud) de W1 y W2 por época (un número por época)
norma_W1_hist = [np.linalg.norm(w.reshape(3, -1)) for w in hist_W1]
norma_W2_hist = [np.linalg.norm(w.reshape(-1, 3)) for w in hist_W2]
export_historial_unity(epocas_hist, errores_hist, norma_W1_hist, norma_W2_hist)

# Exportar cada peso individual por época para la gráfica detallada en Unity
export_pesos_individuales_unity(epocas_hist, hist_W1, hist_W2, hist_b1, hist_b2)

# ── Mostrar fronteras de decisión (abre una segunda ventana) ──────────────────
graficar(entradas, W1, W2, b1, b2)
