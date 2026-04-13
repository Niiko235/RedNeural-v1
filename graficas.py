import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

from generarDatos import generar_datos_3d_3Nubes
from redNeural import redNeural, sigmoid


# ── 1. Generar datos y entrenar ────────────────────────────────────────────────
x1,y1,z1, x2,y2,z2, x3,y3,z3, valor_esperado = generar_datos_3d_3Nubes()

entradas_1 = np.column_stack((x1, y1, z1))
entradas_2 = np.column_stack((x2, y2, z2))
entradas_3 = np.column_stack((x3, y3, z3))
entradas   = np.vstack((entradas_1, entradas_2, entradas_3))

print("Entrenando red neuronal...")
W1, W2, b1, b2 = redNeural(entradas, valor_esperado, epocas=1000, taza_aprendizaje=0.01)
print("Entrenamiento completo.\n")


# ── 2. Función de predicción vectorizada para la malla ─────────────────────────
def predecir_malla(puntos, W1, W2, b1, b2):
    """
    puntos: array (N, 3)
    retorna: array (N,) con la clase predicha (0, 1 o 2)
    """
    oculta = sigmoid(np.dot(puntos, W1) + b1)   # (N, 5)
    salida = sigmoid(np.dot(oculta,   W2) + b2)  # (N, 3)
    return np.argmax(salida, axis=1)             # (N,)


# ── 3. Colores por clase ───────────────────────────────────────────────────────
COLORES_CLASE  = ['royalblue', 'limegreen', 'tomato']
NOMBRES_CLASE  = ['Grupo Azul  (3,3,z)', 'Grupo Verde (9,9,z)', 'Grupo Rojo  (3,9,z)']
CMAP           = plt.cm.colors.ListedColormap(COLORES_CLASE)

colores_reales = (
    ['royalblue'] * len(x1) +
    ['limegreen'] * len(x2) +
    ['tomato']    * len(x3)
)


# ── 4. Malla 2D para los cortes (XY) ──────────────────────────────────────────
RESOLUCION = 60
x_lin = np.linspace(0, 12, RESOLUCION)
y_lin = np.linspace(0, 12, RESOLUCION)
xx, yy = np.meshgrid(x_lin, y_lin)


def calcular_corte(z_val):
    """Predice la clase en el plano XY para un Z fijo."""
    zz_plano = np.full(xx.shape, z_val)
    puntos   = np.column_stack((xx.ravel(), yy.ravel(), zz_plano.ravel()))
    pred     = predecir_malla(puntos, W1, W2, b1, b2)
    return pred.reshape(xx.shape)


# ── 5. Figura principal ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 9))
fig.patch.set_facecolor('#0f0f1a')

# Espacio para el slider abajo
ax3d = fig.add_axes([0.05, 0.15, 0.92, 0.80], projection='3d')
ax3d.set_facecolor('#0f0f1a')

# Estilo de ejes
for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
    pane.fill = False
    pane.set_edgecolor('#333355')

ax3d.tick_params(colors='#aaaacc', labelsize=8)
ax3d.xaxis.label.set_color('#aaaacc')
ax3d.yaxis.label.set_color('#aaaacc')
ax3d.zaxis.label.set_color('#aaaacc')
ax3d.set_xlabel('X', labelpad=8)
ax3d.set_ylabel('Y', labelpad=8)
ax3d.set_zlabel('Z', labelpad=8)
ax3d.set_title('Fronteras de Decisión — Red Neuronal 3D',
               color='white', fontsize=13, pad=14)


# ── 6. Dibujar puntos reales ───────────────────────────────────────────────────
scatter = ax3d.scatter(
    entradas[:, 0], entradas[:, 1], entradas[:, 2],
    c=colores_reales, s=18, alpha=0.85, edgecolors='none', zorder=5
)

# Leyenda manual
from matplotlib.lines import Line2D
leyenda = [Line2D([0],[0], marker='o', color='w',
                  markerfacecolor=c, markersize=9, label=n)
           for c, n in zip(COLORES_CLASE, NOMBRES_CLASE)]
ax3d.legend(handles=leyenda, loc='upper left',
            facecolor='#1a1a2e', edgecolor='#444466',
            labelcolor='white', fontsize=9)


# ── 7. Cortes transversales iniciales ─────────────────────────────────────────
Z_CORTES_INICIALES = [3.0, 6.0, 9.0]
corte_artists = []

for z_val in Z_CORTES_INICIALES:
    pred_plano = calcular_corte(z_val)
    cf = ax3d.contourf(
        xx, yy, pred_plano,
        levels=[-0.5, 0.5, 1.5, 2.5],
        zdir='z', offset=z_val,
        colors=COLORES_CLASE, alpha=0.22
    )
    # línea de contorno para marcar la frontera
    cl = ax3d.contour(
        xx, yy, pred_plano,
        levels=[0.5, 1.5],
        zdir='z', offset=z_val,
        colors='white', linewidths=0.6, alpha=0.5
    )
    corte_artists.append((cf, cl))
    ax3d.text(12.2, 0, z_val, f'z={z_val:.1f}',
              color='#aaaacc', fontsize=7, va='center')

ax3d.set_xlim(0, 12)
ax3d.set_ylim(0, 12)
ax3d.set_zlim(0, 12)


# ── 8. Slider para corte dinámico ─────────────────────────────────────────────
ax_slider = fig.add_axes([0.15, 0.04, 0.70, 0.025])
ax_slider.set_facecolor('#1a1a2e')

slider = Slider(
    ax=ax_slider,
    label='Corte Z',
    valmin=0.0, valmax=12.0,
    valinit=6.0, valstep=0.2,
    color='#5566ff'
)
slider.label.set_color('white')
slider.valtext.set_color('white')

# Artistas del corte dinámico (se borran y redibujan al mover el slider)
corte_dinamico = {'cf': None, 'cl': None, 'txt': None}

def actualizar_slider(val):
    z_val = slider.val

    # Borrar corte anterior
    if corte_dinamico['cf'] is not None:
        for col in corte_dinamico['cf'].collections:
            col.remove()
    if corte_dinamico['cl'] is not None:
        for col in corte_dinamico['cl'].collections:
            col.remove()
    if corte_dinamico['txt'] is not None:
        corte_dinamico['txt'].remove()

    # Dibujar nuevo corte
    pred_plano = calcular_corte(z_val)

    corte_dinamico['cf'] = ax3d.contourf(
        xx, yy, pred_plano,
        levels=[-0.5, 0.5, 1.5, 2.5],
        zdir='z', offset=z_val,
        colors=COLORES_CLASE, alpha=0.35
    )
    corte_dinamico['cl'] = ax3d.contour(
        xx, yy, pred_plano,
        levels=[0.5, 1.5],
        zdir='z', offset=z_val,
        colors='white', linewidths=1.2, alpha=0.8
    )
    corte_dinamico['txt'] = ax3d.text(
        12.2, 0, z_val, f'z={z_val:.1f}',
        color='yellow', fontsize=8, va='center', fontweight='bold'
    )
    fig.canvas.draw_idle()

slider.on_changed(actualizar_slider)
actualizar_slider(6.0)   # dibujar corte inicial del slider


# ── 9. Mostrar ────────────────────────────────────────────────────────────────
plt.show()