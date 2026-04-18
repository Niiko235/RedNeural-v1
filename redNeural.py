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
              numero_neuronas_capa_oculta=5, tolerancia=0.001):
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
    W1 = np.random.rand(3, numero_neuronas_capa_oculta) * 0.1   # entrada → oculta
    W2 = np.random.rand(numero_neuronas_capa_oculta, 3) * 0.1   # oculta  → salida
    b1 = np.zeros((1, numero_neuronas_capa_oculta))              # bias capa oculta
    b2 = np.zeros((1, 3))                                        # bias capa salida

    print("=== Pesos iniciales ===")
    print("W1 (entrada→oculta):\n", W1)
    print("W2 (oculta→salida):\n",  W2)
    print(f"\nTolerancia configurada: {tolerancia}")
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
        if (epoca + 1) % 100 == 0:
            print(f"Época {epoca + 1:4d} | Error total: {error_total:.6f}")

        # ── Parada anticipada por tolerancia ──────────────────────────────────
        # Si el error acumulado de los 300 puntos es menor que la tolerancia,
        # la red ya aprendió suficientemente bien → no tiene sentido seguir
        if error_total < tolerancia:
            print(f"\n✓ Tolerancia alcanzada en época {epoca + 1}"
                  f" — Error total: {error_total:.6f}"
                  f" < Tolerancia: {tolerancia}")
            break

    print("\n=== Pesos finales ===")
    print("W1 (entrada→oculta):\n", W1)
    print("W2 (oculta→salida):\n",  W2)

    return W1, W2, b1, b2


# ── Predicción de un solo punto ────────────────────────────────────────────────

def predecir(entrada, W1, W2, b1, b2):
    x = np.array(entrada).reshape(1, -1)

    # Forward pass del punto a predecir
    h      = sigmoid(np.dot(x, W1) + b1)
    y_pred = sigmoid(np.dot(h, W2) + b2)

    print("Probabilidades por clase:", y_pred)

    # La clase predicha es la neurona con mayor probabilidad
    resultado = np.argmax(y_pred[0])
    nombres   = {0: 'Grupo azul', 1: 'Grupo verde', 2: 'Grupo rojo'}
    return nombres.get(resultado, 'No se pudo predecir')


# ── Evaluación con matriz de confusión ─────────────────────────────────────────

def matriz_confusion(entradas, valor_esperado, W1, W2, b1, b2):
    """
    Compara la clase real vs la clase predicha para cada punto.
    La diagonal de la matriz son los aciertos; fuera de la diagonal, los errores.
    """
    n_clases = 3
    confusion = np.zeros((n_clases, n_clases), dtype=int)

    for i in range(len(entradas)):
        # Forward pass para obtener la predicción
        h      = sigmoid(np.dot(entradas[i].reshape(1, -1), W1) + b1)
        y_pred = sigmoid(np.dot(h, W2) + b2)

        pred = np.argmax(y_pred)           # clase que predijo la red
        real = np.argmax(valor_esperado[i]) # clase real del punto

        confusion[real][pred] += 1   # fila = real, columna = predicho

    precision = np.trace(confusion) / np.sum(confusion)

    print("\n=== Matriz de Confusión ===")
    print("             Pred_Azul  Pred_Verde  Pred_Rojo")
    nombres_filas = ["Real_Azul  ", "Real_Verde ", "Real_Rojo  "]
    for i, fila in enumerate(confusion):
        print(f"  {nombres_filas[i]}  {fila}")
    print(f"\n  Precisión: {precision * 100:.2f}%")

    return confusion, precision
