import numpy as np


# ── Funciones de activación ────────────────────────────────────────────────────

def sigmoid(x):
    # Aplasta cualquier número al rango (0, 1)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivada(x):
    # Recibe x = sigmoid(z), entonces σ'(z) = σ(z) · (1 - σ(z))
    return x * (1 - x)

def error_cuadratico(y_pred, y_real):
    # Error cuadrático medio: mide qué tan lejos está la predicción del valor real
    return np.sum((y_real - y_pred) ** 2) / 2


# ── Red neuronal ───────────────────────────────────────────────────────────────

def redNeural(entradas, valor_esperado, epocas, taza_aprendizaje,
              numero_neuronas_capa_oculta=5, tolerancia=0.001, numero_clases=3, callback=None):
    """
    Entrena una red neuronal de una capa oculta para clasificar 3 clases.

    Arquitectura:  entrada(3)  →  oculta(N)  →  salida(3)
    Activación:    sigmoid en capa oculta y en capa de salida
    Pérdida:       error cuadrático medio (ECM)
    Parada:        máximo de épocas  O  cuando error total < tolerancia

    Parámetros
    ----------
    epocas                    : número máximo de pasadas sobre todos los datos
    taza_aprendizaje          : tamaño del paso al ajustar los pesos (ej. 0.01)
    numero_neuronas_capa_oculta: capacidad de la red para aprender patrones
    tolerancia                : si el error total cae por debajo de este valor
                                el entrenamiento se detiene anticipadamente
    """

    # ── Pesos iniciales aleatorios pequeños ───────────────────────────────────
    # Se multiplican por 0.1 para que sigmoid no sature desde el inicio
    W1 = np.random.rand(3, numero_neuronas_capa_oculta) * 0.1          # entrada → oculta
    W2 = np.random.rand(numero_neuronas_capa_oculta, numero_clases) * 0.1  # oculta → salida
    b1 = np.zeros((1, numero_neuronas_capa_oculta))                    # bias capa oculta
    b2 = np.zeros((1, numero_clases))                                  # bias capa salida

    print("=== Pesos iniciales ===")
    print("W1 (entrada→oculta):\n", W1)
    print("W2 (oculta→salida):\n",  W2)
    print(f"\nTolerancia configurada: {tolerancia}%")
    print(f"Máximo de épocas:       {epocas}\n")

    # ── Bucle de entrenamiento ─────────────────────────────────────────────────
    for epoca in range(epocas):
        error_total = 0   # acumula el error de todos los 300 puntos en esta época

        for i in range(len(entradas)):
            x = entradas[i].reshape(1, -1)        # punto de entrada      (1, 3)
            y = valor_esperado[i].reshape(1, -1)  # etiqueta one-hot real (1, 3)

            # ── FORWARD PASS: calcular la predicción ──────────────────────────

            # Capa oculta: multiplicar entrada por pesos y aplicar sigmoid
            z1     = np.dot(x, W1) + b1   # suma ponderada (1, N)
            h      = sigmoid(z1)           # activación     (1, N)

            # Capa de salida: multiplicar oculta por pesos y aplicar sigmoid
            z2     = np.dot(h, W2) + b2   # suma ponderada (1, 3)
            y_pred = sigmoid(z2)           # predicción     (1, 3)

            # Acumular el error de este punto
            error_total += error_cuadratico(y_pred, y)

            # ── BACKPROPAGATION: ajustar pesos según el error ─────────────────

            # Error en la capa de salida:
            # cuánto contribuyó cada neurona de salida al error total
            delta_salida = (y - y_pred) * sigmoid_derivada(y_pred)   # (1, 3)

            # Error en la capa oculta:
            # se propaga el error de salida hacia atrás a través de W2
            delta_oculta = np.dot(delta_salida, W2.T) * sigmoid_derivada(h)  # (1, N)

            # Actualizar pesos de la capa de salida (oculta → salida)
            W2 += taza_aprendizaje * np.dot(h.T, delta_salida)
            b2 += taza_aprendizaje * delta_salida

            # Actualizar pesos de la capa oculta (entrada → oculta)
            W1 += taza_aprendizaje * np.dot(x.T, delta_oculta)
            b1 += taza_aprendizaje * delta_oculta

        # ── Reporte cada 100 épocas ───────────────────────────────────────────
        if (epoca + 1) % 1 == 0:
            print(f"Época {epoca + 1:4d} | Error total: {error_total:.6f}")

        # ── Actualizar gráficas en tiempo real (si se pasó un callback) ───────
        # El callback recibe la época actual, el error y los pesos para graficarlos
        if callback is not None:
            callback(epoca + 1, error_total, W1, W2, b1, b2)

        # ── Parada anticipada por tolerancia ──────────────────────────────────
        # Si el error acumulado de los 300 puntos es menor que la tolerancia,
        # la red ya aprendió suficientemente bien → no tiene sentido seguir
        error_total /= len(entradas)  # promedio de error por punto
        if error_total < tolerancia / 100:
            print(f"\n✓ Tolerancia alcanzada en época {epoca + 1}"
                  f" — Error total: {error_total:.6f}"
                  f" < Tolerancia: {tolerancia}%")
            break

    print("\n=== Pesos finales ===")
    print("W1 (entrada→oculta):\n", W1)
    print("W2 (oculta→salida):\n",  W2)

    return W1, W2, b1, b2


