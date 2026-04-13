
import numpy as np



# Red nueral que se encargará de entrennarse para diferenciar entre 3 nubes de puntos en un plano 3D, cada nube de puntos representará una clase diferente. La red tendrá una capa de entrada con 3 neuronas (correspondientes a las coordenadas x, y, z), una capa oculta con 5 neuronas y una capa de salida con 3 neuronas (correspondientes a las 3 clases). La función de activación utilizada será la función sigmoide.

# La red se entrenará utilizando el algoritmo de retropropagación y el error cuadrático medio como función de pérdida. El conjunto de datos de entrenamiento consistirá en 300 puntos, con 100 puntos para cada clase, generados aleatoriamente dentro de un rango específico para cada clase.

# la rede devolvera una matriz de confusión y la precisión del modelo después de entrenar.

# las etiquetas de las clases serán codificadas utilizando one-hot encoding, es decir, la clase 1 se representará como [1, 0, 0], la clase 2 como [0, 1, 0] y la clase 3 como [0, 0, 1].


# Red neural que clasifica 3 nubes de puntos en 3D.
# Arquitectura: entrada(3) → oculta(5) → salida(3)
# Activación: sigmoide | Pérdida: error cuadrático medio

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivada(x):
    return x * (1 - x)  # x ya es sigmoid(z)

def error_cuadratico(resultado, valor_esperado):
    return np.sum((valor_esperado - resultado) ** 2) / 2


def redNeural(entradas, valor_esperado, epocas, taza_aprendizaje, numero_neuronas_capa_oculta = 5):

    # Inicializar pesos y sesgos
    pesos_entrada_oculta = np.random.rand(3, numero_neuronas_capa_oculta) * 0.1   # (3, 5)
    pesos_oculta_salida  = np.random.rand(numero_neuronas_capa_oculta, 3) * 0.1   # (5, 3)
    bias_oculta  = np.zeros((1, numero_neuronas_capa_oculta))                      # (1, 5)
    bias_salida  = np.zeros((1, 3))                      # (1, 3)

    print("=== Pesos iniciales ===")
    print("W1 (entrada→oculta):\n", pesos_entrada_oculta)
    print("W2 (oculta→salida):\n",  pesos_oculta_salida)

    # ── Entrenamiento ──────────────────────────────────────────
    for epoca in range(epocas):
        error_total = 0

        for i in range(len(entradas)):
            entrada_i = entradas[i].reshape(1, -1)          # (1, 3)
            esperado_i = valor_esperado[i].reshape(1, -1)   # (1, 3)

            # ── Forward pass ──────────────────────────────────
            # BUG CORREGIDO: usaba entradas[0] siempre en vez de entradas[i]
            capa_oculta_sum = np.dot(entrada_i, pesos_entrada_oculta) + bias_oculta  # (1, 5)
            capa_oculta_sal = sigmoid(capa_oculta_sum)                                # (1, 5)

            capa_salida_sum = np.dot(capa_oculta_sal, pesos_oculta_salida) + bias_salida  # (1, 3)
            capa_salida_res = sigmoid(capa_salida_sum)                                     # (1, 3)

            error = error_cuadratico(capa_salida_res, esperado_i)
            error_total += error

            # ── Backpropagation ───────────────────────────────
            # BUG CORREGIDO: signo y dimensiones de las actualizaciones

            # Delta capa de salida: (1, 3)
            delta_salida = (esperado_i - capa_salida_res) * sigmoid_derivada(capa_salida_res)

            # Delta capa oculta: (1, 5)
            delta_oculta = np.dot(delta_salida, pesos_oculta_salida.T) * sigmoid_derivada(capa_oculta_sal)

            # Actualizar pesos W2 (oculta→salida): (5, 3)
            # BUG CORREGIDO: antes restaba en vez de sumar, y faltaba transponer
            pesos_oculta_salida += taza_aprendizaje * np.dot(capa_oculta_sal.T, delta_salida)
            bias_salida         += taza_aprendizaje * delta_salida

            # Actualizar pesos W1 (entrada→oculta): (3, 5)
            # BUG CORREGIDO: usaba entradas[1] fijo en vez de la entrada actual
            pesos_entrada_oculta += taza_aprendizaje * np.dot(entrada_i.T, delta_oculta)
            bias_oculta          += taza_aprendizaje * delta_oculta

        if (epoca + 1) % 100 == 0:
            print(f"Época {epoca + 1:4d} | Error total: {error_total:.4f}")

    print("\n=== Pesos finales ===")
    print("W1 (entrada→oculta):\n", pesos_entrada_oculta)
    print("W2 (oculta→salida):\n",  pesos_oculta_salida)

    # BUG CORREGIDO: la función ahora retorna los pesos para poder usar predecir()
    return pesos_entrada_oculta, pesos_oculta_salida, bias_oculta, bias_salida


def predecir(entrada, pesos_entrada_oculta, pesos_oculta_salida, bias_oculta, bias_salida):
    entrada = np.array(entrada).reshape(1, -1)

    capa_oculta_sal = sigmoid(np.dot(entrada, pesos_entrada_oculta) + bias_oculta)
    capa_salida_res = sigmoid(np.dot(capa_oculta_sal, pesos_oculta_salida) + bias_salida)

    print("Probabilidades:", capa_salida_res)

    # BUG CORREGIDO: antes accedía a capa_salida_resultado[i] (1D incorrecto)
    resultado = np.argmax(capa_salida_res[0])

    nombres = {0: 'Grupo azul', 1: 'Grupo verde', 2: 'Grupo rojo'}
    return nombres.get(resultado, 'No se pudo predecir')


def matriz_confusion(entradas, valor_esperado,
                     pesos_entrada_oculta, pesos_oculta_salida,
                     bias_oculta, bias_salida):
    """Evalúa el modelo y muestra la matriz de confusión y precisión."""
    n_clases = 3
    confusion = np.zeros((n_clases, n_clases), dtype=int)

    for i in range(len(entradas)):
        oculta = sigmoid(np.dot(entradas[i].reshape(1, -1), pesos_entrada_oculta) + bias_oculta)
        salida = sigmoid(np.dot(oculta, pesos_oculta_salida) + bias_salida)

        pred  = np.argmax(salida)
        real  = np.argmax(valor_esperado[i])
        confusion[real][pred] += 1

    precision = np.trace(confusion) / np.sum(confusion)

    print("\n=== Matriz de Confusión ===")
    print("         Pred_Azul  Pred_Verde  Pred_Rojo")
    nombres = ["Real_Azul ", "Real_Verde", "Real_Rojo "]
    for i, fila in enumerate(confusion):
        print(f"{nombres[i]}  {fila}")
    print(f"\nPrecisión: {precision * 100:.2f}%")

    return confusion, precision