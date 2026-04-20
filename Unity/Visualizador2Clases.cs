using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Globalization;
using UnityEngine;

/// <summary>
/// Visualiza la red neuronal de 2 clases entrenada en Python:
///   - Carga puntos desde puntos.csv y los muestra con animación de aparición.
///   - Crea voxels mostrando la frontera de decisión.
///   - La frontera se sincroniza con GraficadorEntrenamiento2Clases.
///
/// SETUP EN UNITY:
///   1. GameObject vacío llamado "Visualizador" → agregar este script.
///   2. Crear otro GameObject "Graficas" → agregar GraficadorEntrenamiento2Clases.
///   3. Cámara sugerida: posición (6, 10, -16), rotación (25, 0, 0).
/// </summary>
public class Visualizador2Clases : MonoBehaviour
{
    [Header("Ruta de los archivos CSV")]
    public string rutaDatos = @"D:\Trabajos de la u\Septimo semestre\computacional\Segunda entrega\Python red\datosExportados2Clases";

    [Header("Frontera de decisión")]
    public int   resolucion       = 22;
    public float tamañoVoxel      = 0.45f;
    public float duracionMovimiento = 1.8f;

    [Header("Puntos de datos")]
    public float tamañoPunto       = 0.22f;
    public float velocidadAparecer = 4f;

    [Header("Sincronización con gráficas")]
    public bool sincronizarConGraficas = true;

    // clase 0 → cian,  clase 1 → rojo
    private readonly Color[] coloresClase =
    {
        new Color(0.25f, 0.85f, 0.95f),
        new Color(1.00f, 0.28f, 0.20f),
    };

    private RedNeuronal red;
    private GameObject  padreFrontera;

    void Start()
    {
        red = new RedNeuronal(rutaDatos);
        Debug.Log($"Red 2 clases cargada desde: {rutaDatos}");
        StartCoroutine(SecuenciaVisualizacion());
    }

    IEnumerator SecuenciaVisualizacion()
    {
        yield return StartCoroutine(AnimarPuntos());

        if (!sincronizarConGraficas)
            yield return StartCoroutine(AnimarFrontera());
    }

    // ── Llamado desde GraficadorEntrenamiento2Clases en cada checkpoint ────────
    public void ActualizarFronteraSincrono(RedNeuronal nuevaRed)
    {
        if (padreFrontera != null) Destroy(padreFrontera);

        padreFrontera = new GameObject("Frontera de Decisión");
        padreFrontera.transform.parent = transform;

        float paso = 12f / resolucion;

        int[,,] clases = new int[resolucion, resolucion, resolucion];
        for (int xi = 0; xi < resolucion; xi++)
        for (int yi = 0; yi < resolucion; yi++)
        for (int zi = 0; zi < resolucion; zi++)
            clases[xi, yi, zi] = nuevaRed.Predecir(
                (xi + 0.5f) * paso, (yi + 0.5f) * paso, (zi + 0.5f) * paso);

        for (int xi = 0; xi < resolucion; xi++)
        for (int yi = 0; yi < resolucion; yi++)
        for (int zi = 0; zi < resolucion; zi++)
        {
            if (!EsFrontera(clases, xi, yi, zi)) continue;
            float x = (xi + 0.5f) * paso;
            float y = (yi + 0.5f) * paso;
            float z = (zi + 0.5f) * paso;

            GameObject voxel = GameObject.CreatePrimitive(PrimitiveType.Cube);
            voxel.transform.parent     = padreFrontera.transform;
            voxel.transform.localScale = Vector3.one * tamañoVoxel;
            voxel.transform.position   = new Vector3(x, y, z);
            Destroy(voxel.GetComponent<Collider>());
            voxel.GetComponent<Renderer>().material =
                CrearMaterial(new Color(1f, 1f, 0.2f), 0.32f);   // amarillo semitransparente
        }
    }

    // ── 1. Puntos: aparecen escalando de 0 al tamaño final ────────────────────
    IEnumerator AnimarPuntos()
    {
        string[] lineas = File.ReadAllLines(Path.Combine(rutaDatos, "puntos.csv"));

        GameObject padre = new GameObject("Nubes de Puntos");
        padre.transform.parent = transform;

        for (int i = 1; i < lineas.Length; i++)
        {
            string[] cols = lineas[i].Split(',');
            float px    = float.Parse(cols[0], CultureInfo.InvariantCulture);
            float py    = float.Parse(cols[1], CultureInfo.InvariantCulture);
            float pz    = float.Parse(cols[2], CultureInfo.InvariantCulture);
            int   clase = (int)float.Parse(cols[3], CultureInfo.InvariantCulture);

            GameObject esfera = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            esfera.transform.parent     = padre.transform;
            esfera.transform.position   = new Vector3(px, py, pz);
            esfera.transform.localScale = Vector3.zero;
            esfera.name = $"Punto_C{clase}_{i}";
            Destroy(esfera.GetComponent<Collider>());
            esfera.GetComponent<Renderer>().material =
                CrearMaterial(coloresClase[clase], alpha: 1f);

            float retardo = (i - 1) * 0.008f;
            StartCoroutine(EscalarHasta(esfera, tamañoPunto, retardo));
        }

        yield return new WaitForSeconds(lineas.Length * 0.008f + 0.3f);
        Debug.Log($"Puntos creados: {lineas.Length - 1}");
    }

    IEnumerator EscalarHasta(GameObject go, float tamaño, float retardo)
    {
        yield return new WaitForSeconds(retardo);
        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime * velocidadAparecer;
            go.transform.localScale = Vector3.one * Mathf.SmoothStep(0f, tamaño, Mathf.Clamp01(t));
            yield return null;
        }
        go.transform.localScale = Vector3.one * tamaño;
    }

    // ── 2. Frontera: voxels parten de un plano plano y se mueven a posición ────
    IEnumerator AnimarFrontera()
    {
        float paso   = 12f / resolucion;
        float centro = 6f;

        int[,,] clases = new int[resolucion, resolucion, resolucion];
        for (int xi = 0; xi < resolucion; xi++)
        for (int yi = 0; yi < resolucion; yi++)
        for (int zi = 0; zi < resolucion; zi++)
            clases[xi, yi, zi] = red.Predecir(
                (xi + 0.5f) * paso, (yi + 0.5f) * paso, (zi + 0.5f) * paso);

        padreFrontera = new GameObject("Frontera de Decisión");
        padreFrontera.transform.parent = transform;

        int totalVoxels = 0;
        for (int xi = 0; xi < resolucion; xi++)
        for (int yi = 0; yi < resolucion; yi++)
        for (int zi = 0; zi < resolucion; zi++)
        {
            if (!EsFrontera(clases, xi, yi, zi)) continue;

            float x = (xi + 0.5f) * paso;
            float y = (yi + 0.5f) * paso;
            float z = (zi + 0.5f) * paso;

            GameObject voxel = GameObject.CreatePrimitive(PrimitiveType.Cube);
            voxel.transform.parent     = padreFrontera.transform;
            voxel.transform.localScale = Vector3.one * tamañoVoxel;
            Destroy(voxel.GetComponent<Collider>());
            voxel.GetComponent<Renderer>().material =
                CrearMaterial(new Color(1f, 1f, 0.2f), 0.32f);

            float retardo = (xi * resolucion * resolucion + yi * resolucion + zi) * 0.0005f;
            StartCoroutine(MoverVoxel(voxel,
                new Vector3(x, centro, z),
                new Vector3(x, y,      z), retardo));
            totalVoxels++;
        }

        Debug.Log($"Frontera animando: {totalVoxels} voxels");
        yield return new WaitForSeconds(duracionMovimiento + 0.5f);
    }

    IEnumerator MoverVoxel(GameObject voxel, Vector3 origen, Vector3 destino, float retardo)
    {
        voxel.transform.position = origen;
        yield return new WaitForSeconds(retardo);
        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime / duracionMovimiento;
            voxel.transform.position =
                Vector3.Lerp(origen, destino, Mathf.SmoothStep(0f, 1f, t));
            yield return null;
        }
        voxel.transform.position = destino;
    }

    bool EsFrontera(int[,,] clases, int xi, int yi, int zi)
    {
        int clase = clases[xi, yi, zi];
        int[][] vecinos = {
            new[]{ xi-1,yi,zi }, new[]{ xi+1,yi,zi },
            new[]{ xi,yi-1,zi }, new[]{ xi,yi+1,zi },
            new[]{ xi,yi,zi-1 }, new[]{ xi,yi,zi+1 },
        };
        foreach (int[] v in vecinos)
        {
            if (v[0] < 0 || v[1] < 0 || v[2] < 0) continue;
            if (v[0] >= resolucion || v[1] >= resolucion || v[2] >= resolucion) continue;
            if (clases[v[0], v[1], v[2]] != clase) return true;
        }
        return false;
    }

    Material CrearMaterial(Color color, float alpha)
    {
        color.a = alpha;
        Shader shURP = Shader.Find("Universal Render Pipeline/Lit");
        if (shURP != null)
        {
            Material mat = new Material(shURP);
            mat.SetColor("_BaseColor", color);
            if (alpha < 1f)
            {
                mat.SetFloat("_Surface", 1f); mat.SetFloat("_Blend", 0f); mat.SetFloat("_ZWrite", 0f);
                mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
                mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
                mat.renderQueue = 3000;
            }
            return mat;
        }
        Material matStd = new Material(Shader.Find("Standard"));
        matStd.color = color;
        if (alpha < 1f)
        {
            matStd.SetFloat("_Mode", 3);
            matStd.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
            matStd.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
            matStd.SetInt("_ZWrite", 0);
            matStd.EnableKeyword("_ALPHABLEND_ON");
            matStd.renderQueue = 3000;
        }
        return matStd;
    }
}
