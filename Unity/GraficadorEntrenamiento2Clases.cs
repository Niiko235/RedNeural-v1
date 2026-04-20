using System.Collections;
using System.IO;
using System.Globalization;
using UnityEngine;

/// <summary>
/// Dibuja las dos gráficas del entrenamiento de 2 clases y sincroniza la frontera 3D:
///   - Gráfica inferior: Error cuadrático por época (línea roja)
///   - Gráfica superior: Todos los pesos individuales por época
///                       W1=azul, W2=verde, b1=cian, b2=naranja
///
/// SETUP EN UNITY:
///   1. Crear GameObject vacío llamado "Graficas".
///   2. Posicionarlo donde quieras las gráficas en la escena (ej: X=18.5, Y=0, Z=0).
///   3. Agregar este script al GameObject "Graficas".
///   4. Asegurarse de que "Visualizador" tiene el script Visualizador2Clases.
/// </summary>
public class GraficadorEntrenamiento2Clases : MonoBehaviour
{
    [Header("Datos")]
    public string rutaDatos = @"D:\Trabajos de la u\Septimo semestre\computacional\Segunda entrega\Python red\datosExportados2Clases";

    [Header("Animación")]
    public float duracionAnimacion = 20f;
    public float retardoInicio     = 3f;

    [Header("Posición y tamaño de las gráficas")]
    public float ancho     = 9f;
    public float alto      = 4.5f;
    public float separacion = 1.5f;

    [Header("Fondo oscuro")]
    public bool crearFondoDesdeCode = true;

    [Header("Sincronización con la escena 3D")]
    public bool sincronizarFrontera = true;
    public int  numCheckpoints      = 8;

    private Visualizador2Clases visualizador;

    void Start()
    {
        if (sincronizarFrontera)
        {
            visualizador = FindObjectOfType<Visualizador2Clases>();
            if (visualizador == null)
                Debug.LogWarning("GraficadorEntrenamiento2Clases: no se encontró Visualizador2Clases.");
        }

        StartCoroutine(DibujarGraficas());
    }

    IEnumerator DibujarGraficas()
    {
        yield return new WaitForSeconds(retardoInicio);

        // ── 1. Cargar historial.csv ───────────────────────────────────────────
        string[] lineasHist = File.ReadAllLines(Path.Combine(rutaDatos, "historial.csv"));
        int nE = lineasHist.Length - 1;

        float[] epocasE = new float[nE];
        float[] errores = new float[nE];
        float maxError  = 0f;

        for (int i = 0; i < nE; i++)
        {
            string[] c = lineasHist[i + 1].Split(',');
            epocasE[i] = float.Parse(c[0], CultureInfo.InvariantCulture);
            errores[i] = float.Parse(c[1], CultureInfo.InvariantCulture);
            if (errores[i] > maxError) maxError = errores[i];
        }
        float maxEpocaE = epocasE[nE - 1];

        // ── 2. Cargar pesos_historial.csv ─────────────────────────────────────
        string[] lineasP = File.ReadAllLines(Path.Combine(rutaDatos, "pesos_historial.csv"));
        string[] headers = lineasP[0].Split(',');

        int nW1 = 0, nW2 = 0, nb1 = 0, nb2 = 0;
        for (int c = 1; c < headers.Length; c++)
        {
            string h = headers[c].Trim();
            if      (h.StartsWith("w1_")) nW1++;
            else if (h.StartsWith("w2_")) nW2++;
            else if (h.StartsWith("b1_")) nb1++;
            else if (h.StartsWith("b2_")) nb2++;
        }
        int totalLineas = nW1 + nW2 + nb1 + nb2;
        int nP = lineasP.Length - 1;

        float[] epocasP = new float[nP];
        float[,] pesos  = new float[nP, totalLineas];
        float minPeso = float.MaxValue, maxPeso = float.MinValue;

        for (int i = 0; i < nP; i++)
        {
            string[] cols = lineasP[i + 1].Split(',');
            epocasP[i] = float.Parse(cols[0], CultureInfo.InvariantCulture);
            for (int j = 0; j < totalLineas; j++)
            {
                float v = float.Parse(cols[j + 1], CultureInfo.InvariantCulture);
                pesos[i, j] = v;
                if (v < minPeso) minPeso = v;
                if (v > maxPeso) maxPeso = v;
            }
        }
        float maxEpocaP = epocasP[nP - 1];

        Debug.Log($"Pesos cargados: W1={nW1} W2={nW2} b1={nb1} b2={nb2} | Épocas={nP}");

        // ── 3. Posiciones de las dos gráficas ─────────────────────────────────
        Vector3 origenError = transform.position;
        Vector3 origenPesos = origenError + Vector3.up * (alto + separacion);

        // ── 4. Fondos oscuros ─────────────────────────────────────────────────
        if (crearFondoDesdeCode)
        {
            CrearFondo(origenError, ancho, alto);
            CrearFondo(origenPesos, ancho, alto);
        }

        // ── 5. Marcos de los ejes ─────────────────────────────────────────────
        DibujarMarco(origenError, ancho, alto);
        DibujarMarco(origenPesos, ancho, alto);

        // ── 6. Etiquetas ──────────────────────────────────────────────────────
        CrearEtiqueta("Error cuadratico por epoca",
                      origenError + new Vector3(ancho / 2f, alto + 0.45f, 0f), 0.35f, Color.white);
        CrearEtiqueta("Pesos individuales por epoca",
                      origenPesos + new Vector3(ancho / 2f, alto + 0.45f, 0f), 0.35f, Color.white);

        CrearEtiqueta($"W1 ({nW1} pesos)",
                      origenPesos + new Vector3(1.2f, alto + 0.05f, 0f), 0.22f, new Color(0.3f, 0.55f, 1f));
        CrearEtiqueta($"W2 ({nW2} pesos)",
                      origenPesos + new Vector3(3.5f, alto + 0.05f, 0f), 0.22f, new Color(0.2f, 0.9f, 0.3f));
        CrearEtiqueta($"b1 ({nb1} bias)",
                      origenPesos + new Vector3(5.8f, alto + 0.05f, 0f), 0.22f, new Color(0f, 0.9f, 0.9f));
        CrearEtiqueta($"b2 ({nb2} bias)",
                      origenPesos + new Vector3(7.8f, alto + 0.05f, 0f), 0.22f, new Color(1f, 0.6f, 0f));

        // ── 7. Gráfica de error (línea roja) ──────────────────────────────────
        LineRenderer lrError = CrearLineaRenderer(new Color(1f, 0.3f, 0.3f), 0.08f, 1f);
        StartCoroutine(AnimarLineaSimple(lrError,
            epocasE, errores, nE, maxEpocaE, 0f, maxError,
            origenError, ancho, alto, duracionAnimacion));

        // ── 8. Gráfica de pesos (una línea por peso) ──────────────────────────
        LineRenderer[] lrsPesos = new LineRenderer[totalLineas];
        for (int k = 0; k < totalLineas; k++)
        {
            Color color; float alpha, grosor;
            if      (k < nW1)             { color = new Color(0.3f, 0.55f, 1f);  alpha = 0.40f; grosor = 0.05f; }
            else if (k < nW1 + nW2)       { color = new Color(0.2f, 0.9f, 0.3f); alpha = 0.40f; grosor = 0.05f; }
            else if (k < nW1 + nW2 + nb1) { color = new Color(0f,   0.9f, 0.9f); alpha = 0.50f; grosor = 0.05f; }
            else                           { color = new Color(1f,   0.6f, 0f);   alpha = 0.80f; grosor = 0.07f; }

            lrsPesos[k] = CrearLineaRenderer(color, grosor, alpha);
        }

        yield return StartCoroutine(AnimarTodosPesos(lrsPesos,
            epocasP, pesos, nP, totalLineas, maxEpocaP, minPeso, maxPeso,
            origenPesos, ancho, alto, duracionAnimacion,
            nW1, nW2, nb1, nb2));
    }

    // ── Anima todos los pesos y sincroniza la frontera 3D ─────────────────────
    IEnumerator AnimarTodosPesos(LineRenderer[] lrs,
        float[] xs, float[,] valores, int n, int totalLineas,
        float maxX, float minY, float maxY,
        Vector3 origen, float w, float h, float duracion,
        int nW1, int nW2, int nb1, int nb2)
    {
        float tiempoPorPunto = duracion / n;
        float rangoY = (maxY - minY > 0f) ? maxY - minY : 1f;

        for (int k = 0; k < totalLineas; k++) lrs[k].positionCount = 0;

        int nextCp = 0;

        for (int i = 0; i < n; i++)
        {
            // Actualizar frontera 3D en los checkpoints
            if (sincronizarFrontera && visualizador != null)
            {
                float progreso = (float)i / Mathf.Max(n - 1, 1);
                float umbral   = (float)nextCp / numCheckpoints;
                if (nextCp <= numCheckpoints && progreso >= umbral)
                {
                    RedNeuronal redEpoca = ReconstruirRed(valores, i, nW1, nW2, nb1, nb2);
                    visualizador.ActualizarFronteraSincrono(redEpoca);
                    Debug.Log($"Frontera actualizada — época {(int)xs[i]} ({progreso:P0})");
                    nextCp++;
                }
            }

            // Dibujar el punto actual en cada línea de pesos
            for (int k = 0; k < totalLineas; k++)
            {
                lrs[k].positionCount = i + 1;
                float px = origen.x + (xs[i] / maxX) * w;
                float py = origen.y + ((valores[i, k] - minY) / rangoY) * h;
                lrs[k].SetPosition(i, new Vector3(px, py, origen.z));
            }

            yield return new WaitForSeconds(tiempoPorPunto);
        }
    }

    // ── Anima la línea de error de izquierda a derecha ────────────────────────
    IEnumerator AnimarLineaSimple(LineRenderer lr,
        float[] xs, float[] ys, int n,
        float maxX, float minY, float maxY,
        Vector3 origen, float w, float h, float duracion)
    {
        float tiempoPorPunto = duracion / n;
        float rangoY = (maxY - minY > 0f) ? maxY - minY : 1f;
        lr.positionCount = 0;

        for (int i = 0; i < n; i++)
        {
            lr.positionCount = i + 1;
            float px = origen.x + (xs[i] / maxX) * w;
            float py = origen.y + ((ys[i] - minY) / rangoY) * h;
            lr.SetPosition(i, new Vector3(px, py, origen.z));
            yield return new WaitForSeconds(tiempoPorPunto);
        }
    }

    // ── Reconstruye la red con los pesos de una época ─────────────────────────
    RedNeuronal ReconstruirRed(float[,] pesos, int epocaIdx,
                               int nW1, int nW2, int nb1, int nb2)
    {
        int nOculta = nW1 / 3;           // 3 entradas → W1 es [3 × nOculta]
        int nSalida = nW2 / nOculta;     // dinámico: 2 para este proyecto

        float[,] W1mat = new float[3, nOculta];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < nOculta; c++)
                W1mat[r, c] = pesos[epocaIdx, r * nOculta + c];

        float[,] W2mat = new float[nOculta, nSalida];
        for (int r = 0; r < nOculta; r++)
            for (int c = 0; c < nSalida; c++)
                W2mat[r, c] = pesos[epocaIdx, nW1 + r * nSalida + c];

        float[] b1vec = new float[nb1];
        for (int j = 0; j < nb1; j++)
            b1vec[j] = pesos[epocaIdx, nW1 + nW2 + j];

        float[] b2vec = new float[nb2];
        for (int j = 0; j < nb2; j++)
            b2vec[j] = pesos[epocaIdx, nW1 + nW2 + nb1 + j];

        return new RedNeuronal(W1mat, W2mat, b1vec, b2vec);
    }

    // ── Quad oscuro de fondo ──────────────────────────────────────────────────
    void CrearFondo(Vector3 origen, float w, float h)
    {
        GameObject fondo = GameObject.CreatePrimitive(PrimitiveType.Quad);
        fondo.transform.parent     = transform;
        fondo.transform.position   = origen + new Vector3(w / 2f, h / 2f, 0.05f);
        fondo.transform.localScale = new Vector3(w, h, 1f);
        Destroy(fondo.GetComponent<Collider>());
        Material mat = new Material(Shader.Find("Sprites/Default"));
        mat.color = new Color(0.04f, 0.04f, 0.09f, 1f);
        fondo.GetComponent<Renderer>().material = mat;
    }

    // ── Marco rectangular de los ejes ────────────────────────────────────────
    void DibujarMarco(Vector3 origen, float w, float h)
    {
        LineRenderer marco = CrearLineaRenderer(new Color(0.4f, 0.4f, 0.5f), 0.04f, 1f);
        marco.positionCount = 5;
        marco.SetPosition(0, origen);
        marco.SetPosition(1, origen + Vector3.right * w);
        marco.SetPosition(2, origen + new Vector3(w, h, 0f));
        marco.SetPosition(3, origen + Vector3.up * h);
        marco.SetPosition(4, origen);
    }

    LineRenderer CrearLineaRenderer(Color color, float grosor, float alpha)
    {
        color.a = alpha;
        GameObject go = new GameObject("Linea");
        go.transform.parent = transform;
        LineRenderer lr = go.AddComponent<LineRenderer>();
        lr.material      = new Material(Shader.Find("Sprites/Default"));
        lr.startColor    = color;
        lr.endColor      = color;
        lr.startWidth    = grosor;
        lr.endWidth      = grosor;
        lr.useWorldSpace = true;
        lr.positionCount = 0;
        return lr;
    }

    void CrearEtiqueta(string texto, Vector3 posicion, float tamaño, Color color)
    {
        GameObject go = new GameObject("Etiqueta");
        go.transform.parent   = transform;
        go.transform.position = posicion;
        TextMesh tm      = go.AddComponent<TextMesh>();
        tm.text          = texto;
        tm.fontSize      = 24;
        tm.characterSize = tamaño * 0.1f;
        tm.color         = color;
        tm.anchor        = TextAnchor.MiddleCenter;
        tm.alignment     = TextAlignment.Center;
    }
}
