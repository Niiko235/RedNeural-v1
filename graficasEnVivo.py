import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


class GraficasEnVivo:
    """
    Encapsula la figura matplotlib de entrenamiento en tiempo real.
    Guarda el historial de errores y pesos para exportar a Unity después.
    """

    FONDO = '#0f0f1a'

    def __init__(self, tolerancia, intervalo_refresco=5):
        self.tolerancia          = tolerancia
        self.intervalo_refresco  = intervalo_refresco

        # Historial acumulado
        self.epocas   = []
        self.errores  = []
        self.hist_W1  = []
        self.hist_W2  = []
        self.hist_b1  = []
        self.hist_b2  = []

        self._crear_figura()

    # ── Callback que recibe redNeural() al final de cada época ────────────────
    def actualizar(self, epoca, error_total, W1, W2, b1, b2):
        self.epocas.append(epoca)
        self.errores.append(error_total)
        self.hist_W1.append(W1.flatten().copy())
        self.hist_W2.append(W2.flatten().copy())
        self.hist_b1.append(b1.flatten().copy())
        self.hist_b2.append(b2.flatten().copy())

        if epoca % self.intervalo_refresco != 0:
            return

        self._redibujar()

    # ── Propiedades útiles para las funciones de exportación ──────────────────
    @property
    def norma_W1(self):
        return [np.linalg.norm(w.reshape(3, -1)) for w in self.hist_W1]

    @property
    def norma_W2(self):
        return [np.linalg.norm(w.reshape(-1, 3)) for w in self.hist_W2]

    # ── Creación de la figura ─────────────────────────────────────────────────
    def _crear_figura(self):
        plt.ion()

        self._fig = plt.figure(figsize=(13, 5))
        self._fig.patch.set_facecolor(self.FONDO)
        self._fig.suptitle('Entrenamiento en tiempo real', color='white', fontsize=12)

        gs = gridspec.GridSpec(1, 2, figure=self._fig,
                               left=0.08, right=0.97, top=0.88, bottom=0.12,
                               wspace=0.35)

        self._ax_error = self._fig.add_subplot(gs[0, 0])
        self._ax_pesos = self._fig.add_subplot(gs[0, 1])

        for ax in [self._ax_error, self._ax_pesos]:
            ax.set_facecolor('#12122a')
            ax.tick_params(colors='#aaaacc', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#333355')

        self._ax_error.set_title('Error cuadrático por época',  color='white', fontsize=10)
        self._ax_error.set_xlabel('Época',       color='#aaaacc', fontsize=9)
        self._ax_error.set_ylabel('Error total', color='#aaaacc', fontsize=9)

        self._ax_pesos.set_title('Magnitud de pesos por época', color='white', fontsize=10)
        self._ax_pesos.set_xlabel('Época',         color='#aaaacc', fontsize=9)
        self._ax_pesos.set_ylabel('Norma (||W||)', color='#aaaacc', fontsize=9)

        plt.show()

    # ── Redibujo cada N épocas ────────────────────────────────────────────────
    def _redibujar(self):
        arr_W1 = np.array(self.hist_W1)
        arr_W2 = np.array(self.hist_W2)
        arr_b1 = np.array(self.hist_b1)
        arr_b2 = np.array(self.hist_b2)

        # Gráfica 1: error
        self._ax_error.clear()
        self._ax_error.set_facecolor('#12122a')
        self._ax_error.plot(self.epocas, self.errores, color='tomato', linewidth=1.8)
        self._ax_error.axhline(y=self.tolerancia, color='yellow', linewidth=0.9,
                               linestyle='--', alpha=0.8,
                               label=f'Tolerancia ({self.tolerancia})')
        self._ax_error.set_title('Error cuadrático por época', color='white', fontsize=10)
        self._ax_error.set_xlabel('Época',       color='#aaaacc', fontsize=9)
        self._ax_error.set_ylabel('Error total', color='#aaaacc', fontsize=9)
        self._ax_error.tick_params(colors='#aaaacc', labelsize=8)
        self._ax_error.legend(facecolor='#1a1a2e', edgecolor='#444466',
                              labelcolor='white', fontsize=8)
        for spine in self._ax_error.spines.values():
            spine.set_edgecolor('#333355')

        # Gráfica 2: pesos individuales
        self._ax_pesos.clear()
        self._ax_pesos.set_facecolor('#12122a')

        for j in range(arr_W1.shape[1]):
            self._ax_pesos.plot(self.epocas, arr_W1[:, j],
                                color='royalblue', linewidth=0.6, alpha=0.35)
        for j in range(arr_W2.shape[1]):
            self._ax_pesos.plot(self.epocas, arr_W2[:, j],
                                color='limegreen', linewidth=0.6, alpha=0.35)
        for j in range(arr_b1.shape[1]):
            self._ax_pesos.plot(self.epocas, arr_b1[:, j],
                                color='cyan', linewidth=0.6, alpha=0.45)
        for j in range(arr_b2.shape[1]):
            self._ax_pesos.plot(self.epocas, arr_b2[:, j],
                                color='orange', linewidth=0.9, alpha=0.7)

        leyenda = [
            Line2D([0],[0], color='royalblue', linewidth=1.5,
                   label=f'W1 — {arr_W1.shape[1]} pesos (entrada→oculta)'),
            Line2D([0],[0], color='limegreen', linewidth=1.5,
                   label=f'W2 — {arr_W2.shape[1]} pesos (oculta→salida)'),
            Line2D([0],[0], color='cyan',      linewidth=1.5,
                   label=f'b1 — {arr_b1.shape[1]} bias (oculta)'),
            Line2D([0],[0], color='orange',    linewidth=1.5,
                   label=f'b2 — {arr_b2.shape[1]} bias (salida)'),
        ]
        self._ax_pesos.legend(handles=leyenda, facecolor='#1a1a2e',
                              edgecolor='#444466', labelcolor='white', fontsize=7)
        self._ax_pesos.set_title('Pesos individuales por época', color='white', fontsize=10)
        self._ax_pesos.set_xlabel('Época',         color='#aaaacc', fontsize=9)
        self._ax_pesos.set_ylabel('Valor del peso', color='#aaaacc', fontsize=9)
        self._ax_pesos.tick_params(colors='#aaaacc', labelsize=8)
        for spine in self._ax_pesos.spines.values():
            spine.set_edgecolor('#333355')

        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

    def cerrar_modo_interactivo(self):
        plt.ioff()
