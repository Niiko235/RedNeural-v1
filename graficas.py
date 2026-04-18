import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (activa proyección 3D)

from redNeural import sigmoid


def graficar(entradas, W1, W2, b1, b2):
    """
    Recibe los datos de entrenamiento y los pesos ya entrenados,
    y muestra la figura con el panel 3D y los 3 paneles de frontera 2D.
    """

    # ── Predicción vectorizada para la malla ──────────────────────────────────
    def predecir_malla(puntos):
        """
        Predice la clase de miles de puntos a la vez (en lote).
        A diferencia de predecir(), que procesa un punto por vez, esta función
        recibe una matriz (N, 3) y devuelve un array (N,) con la clase de cada punto.
        Esto es necesario para colorear las regiones de la gráfica eficientemente.
        """
        oculta = sigmoid(np.dot(puntos, W1) + b1)   # forward capa oculta (N, neuronas)
        salida = sigmoid(np.dot(oculta,  W2) + b2)  # forward capa salida  (N, 3)
        return np.argmax(salida, axis=1)             # clase con mayor probabilidad (N,)

    # ── Paleta de colores ──────────────────────────────────────────────────────
    COLORES = ['royalblue', 'limegreen', 'tomato']
    NOMBRES = ['Grupo 1 — media (3, 3, 3)',
               'Grupo 2 — media (9, 9, 9)',
               'Grupo 3 — media (3, 9, 9)']

    # ListedColormap mapea el valor 0→azul, 1→verde, 2→rojo en los contourf
    CMAP = ListedColormap(COLORES)

    # LEVELS define los bordes entre colores en contourf:
    #   valor < 0.5  → clase 0 (azul)
    #   0.5 – 1.5   → clase 1 (verde)
    #   valor > 1.5 → clase 2 (rojo)
    LEVELS = [-0.5, 0.5, 1.5, 2.5]

    # Lista de colores paralela a `entradas`: primeros 100 azul, siguientes verde, etc.
    N_pts = len(entradas) // 3
    colores_reales = COLORES[0:1]*N_pts + COLORES[1:2]*N_pts + COLORES[2:3]*N_pts

    # ── Volumen 3D de regiones (baja resolución) ───────────────────────────────
    # Se crea una cuadrícula 3D con RES_VOL puntos por eje → RES_VOL³ puntos en total.
    # Baja resolución (20) para que no sea lento; solo sirve como fondo difuso.
    RES_VOL = 20
    xg = np.linspace(0, 12, RES_VOL)
    yg = np.linspace(0, 12, RES_VOL)
    zg = np.linspace(0, 12, RES_VOL)

    # meshgrid convierte 3 arrays 1D en una cuadrícula 3D completa
    XG, YG, ZG = np.meshgrid(xg, yg, zg)

    # ravel() aplana cada cuadrícula a 1D para poder pasarlos como columnas
    grid_pts   = np.column_stack([XG.ravel(), YG.ravel(), ZG.ravel()])  # (8000, 3)
    pred_vol   = predecir_malla(grid_pts)                                # (8000,)
    vol_colors = np.array(COLORES)[pred_vol]   # convierte índice de clase a color

    # ── Mallas 2D para las fronteras (alta resolución) ────────────────────────
    # Alta resolución (250) para que la línea blanca de la frontera se vea nítida.
    RES_2D = 250
    lin = np.linspace(0, 12, RES_2D)
    A, B = np.meshgrid(lin, lin)   # A y B son matrices (250, 250) que cubren el plano

    def frontera_xy(z_fijo=6.0):
        """Clasifica todos los puntos del plano XY con Z fijo."""
        # np.full crea un array del mismo tamaño que A pero con un valor constante
        pts = np.column_stack([A.ravel(), B.ravel(), np.full(A.size, z_fijo)])
        return predecir_malla(pts).reshape(A.shape)   # vuelve a forma (250, 250)

    def frontera_xz(y_fijo=6.0):
        """Clasifica todos los puntos del plano XZ con Y fijo."""
        pts = np.column_stack([A.ravel(), np.full(A.size, y_fijo), B.ravel()])
        return predecir_malla(pts).reshape(A.shape)

    def frontera_yz(x_fijo=6.0):
        """Clasifica todos los puntos del plano YZ con X fijo."""
        pts = np.column_stack([np.full(A.size, x_fijo), A.ravel(), B.ravel()])
        return predecir_malla(pts).reshape(A.shape)

    # ── Layout de la figura ────────────────────────────────────────────────────
    FONDO    = '#0f0f1a'   # color de fondo oscuro de toda la figura
    FONDO_2D = '#12122a'   # fondo levemente más claro para los paneles 2D

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(FONDO)

    # GridSpec divide la figura en 3 filas × 2 columnas
    # La columna izquierda (gs[:, 0]) la ocupa el 3D completo
    # La columna derecha tiene 3 paneles 2D apilados (gs[0,1], gs[1,1], gs[2,1])
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           left=0.03, right=0.98,
                           top=0.93, bottom=0.05,
                           wspace=0.28, hspace=0.55)

    ax3d  = fig.add_subplot(gs[:, 0], projection='3d')   # panel 3D (toda la columna)
    ax_xy = fig.add_subplot(gs[0, 1])                    # panel 2D superior
    ax_xz = fig.add_subplot(gs[1, 1])                    # panel 2D medio
    ax_yz = fig.add_subplot(gs[2, 1])                    # panel 2D inferior

    # ── Panel 3D ──────────────────────────────────────────────────────────────
    ax3d.set_facecolor(FONDO)

    # Quitar el relleno de las caras del cubo y dejar solo el borde
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor('#222244')

    # Fondo difuso: nube de puntos de la cuadrícula 3D coloreados por clase predicha.
    # alpha muy bajo (0.04) para que sea casi transparente y no tape los puntos reales.
    # zorder=1 → se dibuja primero (queda atrás)
    ax3d.scatter(grid_pts[:, 0], grid_pts[:, 1], grid_pts[:, 2],
                 c=vol_colors, alpha=0.04, s=25, edgecolors='none', zorder=1)

    # ── Planos de corte con la frontera de decisión ───────────────────────────
    # Cada plano se dibuja en z=6, y=6 o x=6 (punto medio del espacio 0–12).
    # contourf → rellena las regiones con color según la clase predicha
    # contour  → dibuja solo la línea blanca en el borde entre clases
    # zdir     → indica el eje perpendicular al plano ('z' = plano horizontal XY, etc.)
    # offset   → posición del plano a lo largo del eje indicado por zdir

    # Plano XY en z=6 (corte horizontal)
    bxy_3d = frontera_xy(6.0)
    ax3d.contourf(A, B, bxy_3d, levels=LEVELS, zdir='z', offset=6,
                  cmap=CMAP, alpha=0.30)
    ax3d.contour( A, B, bxy_3d, levels=[0.5, 1.5], zdir='z', offset=6,
                  colors='white', linewidths=1.5, alpha=0.9)
    ax3d.text(12.3, 0, 6.2, 'z=6', color='white', fontsize=7, alpha=0.8)

    # Plano XZ en y=6 (corte frontal — A representa X, B representa Z)
    bxz_3d = frontera_xz(6.0)
    ax3d.contourf(A, B, bxz_3d, levels=LEVELS, zdir='y', offset=6,
                  cmap=CMAP, alpha=0.30)
    ax3d.contour( A, B, bxz_3d, levels=[0.5, 1.5], zdir='y', offset=6,
                  colors='white', linewidths=1.5, alpha=0.9)
    ax3d.text(12.3, 6, 0, 'y=6', color='white', fontsize=7, alpha=0.8)

    # Plano YZ en x=6 (corte lateral — A representa Y, B representa Z)
    byz_3d = frontera_yz(6.0)
    ax3d.contourf(A, B, byz_3d, levels=LEVELS, zdir='x', offset=6,
                  cmap=CMAP, alpha=0.30)
    ax3d.contour( A, B, byz_3d, levels=[0.5, 1.5], zdir='x', offset=6,
                  colors='white', linewidths=1.5, alpha=0.9)
    ax3d.text(6, 12.3, 0, 'x=6', color='white', fontsize=7, alpha=0.8)

    # Puntos reales encima de todo (zorder=5 → se dibuja último, queda al frente)
    ax3d.scatter(entradas[:, 0], entradas[:, 1], entradas[:, 2],
                 c=colores_reales, s=25, alpha=0.95,
                 edgecolors='white', linewidths=0.3, zorder=5)

    # Leyenda manual con círculos de colores (Line2D con marker='o')
    leyenda = [Line2D([0], [0], marker='o', color='w',
                      markerfacecolor=c, markersize=9, label=n)
               for c, n in zip(COLORES, NOMBRES)]
    ax3d.legend(handles=leyenda, loc='upper left',
                facecolor='#1a1a2e', edgecolor='#444466',
                labelcolor='white', fontsize=8)

    ax3d.set_xlim(0, 12); ax3d.set_ylim(0, 12); ax3d.set_zlim(0, 12)
    ax3d.set_xlabel('X', color='#aaaacc', labelpad=6)
    ax3d.set_ylabel('Y', color='#aaaacc', labelpad=6)
    ax3d.set_zlabel('Z', color='#aaaacc', labelpad=6)
    ax3d.tick_params(colors='#aaaacc', labelsize=7)
    ax3d.set_title('Vista 3D — Nubes + planos de decisión (x=6, y=6, z=6)',
                   color='white', fontsize=10, pad=10)

    # ── Función auxiliar para estilizar los paneles 2D ────────────────────────
    def estilo_2d(ax, titulo, xlabel, ylabel):
        """Aplica fondo oscuro, títulos y colores de texto a un panel 2D."""
        ax.set_facecolor(FONDO_2D)
        ax.set_title(titulo, color='white', fontsize=9, pad=5)
        ax.set_xlabel(xlabel, color='#aaaacc', fontsize=8)
        ax.set_ylabel(ylabel, color='#aaaacc', fontsize=8)
        ax.tick_params(colors='#aaaacc', labelsize=7)
        ax.set_xlim(0, 12); ax.set_ylim(0, 12)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    # ── Panel XY — vista desde arriba (corte z = 6) ───────────────────────────
    # contourf: pinta el fondo según la región de clase (azul / verde / rojo)
    # contour:  dibuja la línea blanca exacta donde cambia de clase
    # scatter:  proyecta los puntos reales en este plano (usando columnas x e y)
    bxy = frontera_xy(6.0)
    ax_xy.contourf(lin, lin, bxy, levels=LEVELS, cmap=CMAP, alpha=0.55)
    ax_xy.contour( lin, lin, bxy, levels=[0.5, 1.5],
                   colors='white', linewidths=2.0, alpha=0.9)
    ax_xy.scatter(entradas[:, 0], entradas[:, 1],
                  c=colores_reales, s=12, alpha=0.8, edgecolors='none')
    estilo_2d(ax_xy, 'Plano XY  (corte z = 6)', 'X', 'Y')

    # ── Panel XZ — vista lateral (corte y = 6) ────────────────────────────────
    # Proyecta columna 0 (X) vs columna 2 (Z) de las entradas
    bxz = frontera_xz(6.0)
    ax_xz.contourf(lin, lin, bxz, levels=LEVELS, cmap=CMAP, alpha=0.55)
    ax_xz.contour( lin, lin, bxz, levels=[0.5, 1.5],
                   colors='white', linewidths=2.0, alpha=0.9)
    ax_xz.scatter(entradas[:, 0], entradas[:, 2],
                  c=colores_reales, s=12, alpha=0.8, edgecolors='none')
    estilo_2d(ax_xz, 'Plano XZ  (corte y = 6)', 'X', 'Z')

    # ── Panel YZ — vista frontal (corte x = 6) ────────────────────────────────
    # Proyecta columna 1 (Y) vs columna 2 (Z) de las entradas
    byz = frontera_yz(6.0)
    ax_yz.contourf(lin, lin, byz, levels=LEVELS, cmap=CMAP, alpha=0.55)
    ax_yz.contour( lin, lin, byz, levels=[0.5, 1.5],
                   colors='white', linewidths=2.0, alpha=0.9)
    ax_yz.scatter(entradas[:, 1], entradas[:, 2],
                  c=colores_reales, s=12, alpha=0.8, edgecolors='none')
    estilo_2d(ax_yz, 'Plano YZ  (corte x = 6)', 'Y', 'Z')

    # ── Mostrar la figura ──────────────────────────────────────────────────────
    plt.suptitle('Red Neuronal — Fronteras de Decisión en 3 Ejes',
                 color='white', fontsize=13, y=0.98)
    plt.show()


# Permite ejecutar graficas.py de forma independiente para pruebas
if __name__ == '__main__':
    from generarDatos import generar_datos_3d_3Nubes
    from redNeural import redNeural

    x1,y1,z1, x2,y2,z2, x3,y3,z3, valor_esperado = generar_datos_3d_3Nubes()
    entradas = np.vstack([
        np.column_stack((x1, y1, z1)),
        np.column_stack((x2, y2, z2)),
        np.column_stack((x3, y3, z3)),
    ])
    W1, W2, b1, b2 = redNeural(entradas, valor_esperado,
                                epocas=1000, taza_aprendizaje=0.01)
    graficar(entradas, W1, W2, b1, b2)
