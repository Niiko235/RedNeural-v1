using System.IO;
using System.Globalization;
using UnityEngine;

/// <summary>
/// Reconstruye la red neuronal entrenada en Python usando los pesos exportados.
/// Implementa el mismo forward pass: entrada(3) → sigmoid → oculta(N) → sigmoid → salida(3)
/// </summary>
public class RedNeuronal
{
    private float[,] W1;   // pesos entrada → oculta  (3 × N)
    private float[,] W2;   // pesos oculta  → salida  (N × 3)
    private float[]  b1;   // bias capa oculta         (N)
    private float[]  b2;   // bias capa salida          (3)

    private int nEntrada;
    private int nOculta;
    private int nSalida;

    /// <summary>
    /// Carga los pesos desde los CSV exportados por Python.
    /// </summary>
    public RedNeuronal(string carpeta)
    {
        W1 = LeerMatriz(Path.Combine(carpeta, "W1.csv"));
        W2 = LeerMatriz(Path.Combine(carpeta, "W2.csv"));
        b1 = LeerVector(Path.Combine(carpeta, "b1.csv"));
        b2 = LeerVector(Path.Combine(carpeta, "b2.csv"));

        nEntrada = W1.GetLength(0);   // 3
        nOculta  = W1.GetLength(1);   // N neuronas ocultas
        nSalida  = W2.GetLength(1);   // 3 clases
    }

    /// <summary>
    /// Constructor directo con matrices ya construidas.
    /// Usado por GraficadorEntrenamiento para reconstruir la red en cada checkpoint.
    /// </summary>
    public RedNeuronal(float[,] W1, float[,] W2, float[] b1, float[] b2)
    {
        this.W1 = W1;  this.W2 = W2;
        this.b1 = b1;  this.b2 = b2;
        nEntrada = W1.GetLength(0);
        nOculta  = W1.GetLength(1);
        nSalida  = W2.GetLength(1);
    }

    // ── Función de activación ─────────────────────────────────────────────────
    // Misma sigmoid que en Python: 1 / (1 + e^(-x))
    private float Sigmoid(float x) => 1f / (1f + Mathf.Exp(-x));

    /// <summary>
    /// Dado un punto (x, y, z) en el espacio 3D, retorna la clase predicha (0, 1 o 2).
    /// Replica exactamente el forward pass de redNeural.py
    /// </summary>
    public int Predecir(float x, float y, float z)
    {
        float[] entrada = { x, y, z };

        // ── Capa oculta: z1 = x·W1 + b1,  h = sigmoid(z1) ───────────────────
        float[] h = new float[nOculta];
        for (int j = 0; j < nOculta; j++)
        {
            float suma = b1[j];
            for (int i = 0; i < nEntrada; i++)
                suma += entrada[i] * W1[i, j];
            h[j] = Sigmoid(suma);
        }

        // ── Capa de salida: z2 = h·W2 + b2,  y = sigmoid(z2) ────────────────
        float[] salida = new float[nSalida];
        for (int k = 0; k < nSalida; k++)
        {
            float suma = b2[k];
            for (int j = 0; j < nOculta; j++)
                suma += h[j] * W2[j, k];
            salida[k] = Sigmoid(suma);
        }

        // ── Clase con mayor probabilidad (argmax) ─────────────────────────────
        int   claseMax = 0;
        float valorMax = salida[0];
        for (int k = 1; k < nSalida; k++)
            if (salida[k] > valorMax) { valorMax = salida[k]; claseMax = k; }

        return claseMax;
    }

    // ── Lectura de archivos CSV ───────────────────────────────────────────────

    /// <summary>Lee un CSV de varias filas y columnas y retorna una matriz float[,]</summary>
    private float[,] LeerMatriz(string ruta)
    {
        string[] lineas = File.ReadAllLines(ruta);
        int filas = lineas.Length;
        int cols  = lineas[0].Split(',').Length;

        float[,] mat = new float[filas, cols];
        for (int i = 0; i < filas; i++)
        {
            string[] vals = lineas[i].Split(',');
            for (int j = 0; j < cols; j++)
                mat[i, j] = float.Parse(vals[j].Trim(), CultureInfo.InvariantCulture);
        }
        return mat;
    }

    /// <summary>Lee una fila de CSV y retorna un vector float[]</summary>
    private float[] LeerVector(string ruta)
    {
        // b1 y b2 vienen como una sola fila separada por comas
        string[] vals = File.ReadAllText(ruta).Trim().Split(',');
        float[] vec = new float[vals.Length];
        for (int i = 0; i < vals.Length; i++)
            vec[i] = float.Parse(vals[i].Trim(), CultureInfo.InvariantCulture);
        return vec;
    }
}
