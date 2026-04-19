using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Globalization;
using UnityEngine;

/// <summary>
/// Visualiza la red neuronal entrenada en Python:
///   - Carga puntos y pesos desde CSV (ruta configurable en el Inspector).
///   - Crea esferas coloreadas por clase con animación de aparición.
///   - Crea voxels semitransparentes mostrando la frontera de decisión,
///     que aparecen progresivamente mientras "se entrena" visualmente.
///
/// SETUP EN UNITY:
///   1. GameObject vacío → agregar este script.
///   2. En el Inspector, el campo "Ruta Datos" ya apunta a StreamingAssets.
///      Copiar datosExportados/ dentro de Assets/StreamingAssets/.
///   3. Cámara sugerida: posición (6, 10, -16), rotación (25, 0, 0).
/// </summary>
public class Visualizador : MonoBehaviour
{
    [Header("Ruta de los archivos CSV")]
    [Tooltip("Ruta absoluta a la carpeta datosExportados del proyecto Python.\nEjemplo: D:\\Trabajos de la u\\Septimo semestre\\computacional\\Segunda entrega\\Python red\\datosExportados\nDejar vacío solo usa StreamingAssets (requiere copiar archivos).")]
    public string rutaDatos = "";

    [Header("Frontera de decisión")]
    [Tooltip("Voxels por eje. Más alto = más detalle pero más lento.")]
    public int resolucion = 22;
    public float tamañoVoxel = 0.45f;
    [Tooltip("Segundos que tarda cada voxel en moverse del plano inicial a su posición final")]
    public float duracionMovimiento = 1.8f;

    [Header("Puntos de datos")]
    public float tamañoPunto = 0.22f;
    [Tooltip("Velocidad con que cada punto escala de 0 a su tamaño final")]
    public float velocidadAparecer = 4f;

    // Colores por clase — mismo orden que Python: azul, verde, rojo
    private readonly Color[] coloresClase =
    {
        new Color(0.25f, 0.45f, 1.00f),   // clase 0 → azul  (Grupo 1)
        new Color(0.20f, 0.85f, 0.25f),   // clase 1 → verde (Grupo 2)
        new Color(1.00f, 0.28f, 0.20f),   // clase 2 → rojo  (Grupo 3)
    };

    [Header("Sincronización con gráficas")]
    [Tooltip("Activar cuando GraficadorEntrenamiento controla la frontera en sync con las gráficas.\nCuando está activo, este script solo anima los puntos; la frontera la maneja GraficadorEntrenamiento.")]
    public bool sincronizarConGraficas = true;

    private RedNeuronal red;
    // Referencia al padre de los voxels de frontera — permite destruir y reemplazar
    private GameObject padreFrontera;

    void Start()
    {
        if (string.IsNullOrEmpty(rutaDatos))
            rutaDatos = Path.Combine(Application.streamingAssetsPath, "datosExportados");

        red = new RedNeuronal(rutaDatos);
        Debug.Log($"Red neuronal cargada desde: {rutaDatos}");

        StartCoroutine(SecuenciaVisualizacion());
    }

    // ── Secuencia de animación ─────────────────────────────────────────────────
    IEnumerator SecuenciaVisualizacion()
    {
        yield return StartCoroutine(AnimarPuntos());

        // Si sincronizarConGraficas está activo, GraficadorEntrenamiento
        // se encarga de actualizar la frontera en sync con las gráficas.
        if (!sincronizarConGraficas)
            yield return StartCoroutine(AnimarFrontera());
    }

    // ── Actualiza la frontera 3D con nuevos pesos (llamado desde GraficadorEntrenamiento) ──
    public void ActualizarFronteraSincrono(RedNeuronal nuevaRed)
    {
        if (padreFrontera != null)
            Destroy(padreFrontera);

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
            int   c = clases[xi, yi, zi];

            GameObject voxel = GameObject.CreatePrimitive(PrimitiveType.Cube);
            voxel.transform.parent     = padreFrontera.transform;
            voxel.transform.localScale = Vector3.one * tamañoVoxel;
            voxel.transform.position   = new Vector3(x, y, z);
            Destroy(voxel.GetComponent<Collider>());
            voxel.GetComponent<Renderer>().material = CrearMaterial(coloresClase[c], 0.32f);
        }
    }

    // ── 1. Esferas de datos — aparecen escalando de 0 al tamaño final ─────────
    IEnumerator AnimarPuntos()
    {
        string[] lineas = File.ReadAllLines(Path.Combine(rutaDatos, "puntos.csv"));

        GameObject padre = new GameObject("Nubes de Puntos");
        padre.transform.parent = transform;

        // Crear todas las esferas con escala 0 y luego animarlas
        var esferas = new List<(GameObject go, float delay)>();

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
            esfera.transform.localScale = Vector3.zero;   // empieza invisible
            esfera.name = $"Punto_C{clase}_{i}";
            Destroy(esfera.GetComponent<Collider>());
            esfera.GetComponent<Renderer>().material =
                CrearMaterial(coloresClase[clase], alpha: 1f);

            // Animar escala: aparece suavemente
            float retardo = (i - 1) * 0.008f;   // cada punto aparece 8ms después del anterior
            StartCoroutine(EscalarHasta(esfera, tamañoPunto, retardo));
        }

        // Esperar a que todos los puntos terminen de aparecer antes de la frontera
        yield return new WaitForSeconds(lineas.Length * 0.008f + 0.3f);
        Debug.Log($"Puntos creados: {lineas.Length - 1}");
    }

    // Escala un GameObject de 0 a 'tamaño' después de 'retardo' segundos
    IEnumerator EscalarHasta(GameObject go, float tamaño, float retardo)
    {
        yield return new WaitForSeconds(retardo);
        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime * velocidadAparecer;
            float escala = Mathf.SmoothStep(0f, tamaño, Mathf.Clamp01(t));
            go.transform.localScale = Vector3.one * escala;
            yield return null;
        }
        go.transform.localScale = Vector3.one * tamaño;
    }

    // ── 2. Voxels de frontera — parten de un plano plano y se mueven a su ───────
    //       posición real, igual que la línea del perceptrón que giraba/subía
    IEnumerator AnimarFrontera()
    {
        float paso   = 12f / resolucion;
        float centro = 6f;   // punto medio del espacio de datos (0-12)

        // Evaluar la red en toda la cuadrícula
        int[,,] clases = new int[resolucion, resolucion, resolucion];
        for (int xi = 0; xi < resolucion; xi++)
        for (int yi = 0; yi < resolucion; yi++)
        for (int zi = 0; zi < resolucion; zi++)
            clases[xi, yi, zi] = red.Predecir(
                (xi + 0.5f) * paso,
                (yi + 0.5f) * paso,
                (zi + 0.5f) * paso);

        padreFrontera = new GameObject("Frontera de Decisión");
        padreFrontera.transform.parent = transform;

        int totalVoxels = 0;

        for (int xi = 0; xi < resolucion; xi++)
        for (int yi = 0; yi < resolucion; yi++)
        for (int zi = 0; zi < resolucion; zi++)
        {
            if (!EsFrontera(clases, xi, yi, zi)) continue;

            // Posición final: donde la red aprendió que está la frontera
            float x = (xi + 0.5f) * paso;
            float y = (yi + 0.5f) * paso;
            float z = (zi + 0.5f) * paso;
            int   c = clases[xi, yi, zi];

            GameObject voxel = GameObject.CreatePrimitive(PrimitiveType.Cube);
            voxel.transform.parent = padreFrontera.transform;
            voxel.transform.localScale = Vector3.one * tamañoVoxel;
            voxel.name = $"Frontera_C{c}";
            Destroy(voxel.GetComponent<Collider>());
            voxel.GetComponent<Renderer>().material =
                CrearMaterial(coloresClase[c], alpha: 0.32f);

            // Posición inicial: plano plano en Y=centro (como red sin entrenar)
            // El voxel está en la posición correcta en X y Z pero al mismo Y
            Vector3 posInicial = new Vector3(x, centro, z);
            Vector3 posFinal   = new Vector3(x, y,      z);

            // Retardo escalonado para que no todos salgan al mismo tiempo
            float retardo = (xi * resolucion * resolucion + yi * resolucion + zi) * 0.0005f;

            // Mover desde el plano inicial hasta la posición aprendida
            StartCoroutine(MoverVoxel(voxel, posInicial, posFinal, retardo));
            totalVoxels++;
        }

        Debug.Log($"Frontera animando: {totalVoxels} voxels");
        yield return new WaitForSeconds(duracionMovimiento + 0.5f);
    }

    // Mueve un voxel suavemente desde 'origen' hasta 'destino'
    // con el mismo efecto que la línea del perceptrón al ajustarse
    IEnumerator MoverVoxel(GameObject voxel, Vector3 origen, Vector3 destino, float retardo)
    {
        voxel.transform.position = origen;
        yield return new WaitForSeconds(retardo);

        float t = 0f;
        while (t < 1f)
        {
            t += Time.deltaTime / duracionMovimiento;
            // SmoothStep da una aceleración suave al inicio y al final
            voxel.transform.position = Vector3.Lerp(origen, destino, Mathf.SmoothStep(0f, 1f, t));
            yield return null;
        }
        voxel.transform.position = destino;
    }

    // Retorna true si algún vecino directo tiene clase distinta
    bool EsFrontera(int[,,] clases, int xi, int yi, int zi)
    {
        int clase = clases[xi, yi, zi];
        int[][] vecinos = {
            new[]{ xi-1, yi, zi }, new[]{ xi+1, yi, zi },
            new[]{ xi, yi-1, zi }, new[]{ xi, yi+1, zi },
            new[]{ xi, yi, zi-1 }, new[]{ xi, yi, zi+1 },
        };
        foreach (int[] v in vecinos)
        {
            if (v[0] < 0 || v[1] < 0 || v[2] < 0) continue;
            if (v[0] >= resolucion || v[1] >= resolucion || v[2] >= resolucion) continue;
            if (clases[v[0], v[1], v[2]] != clase) return true;
        }
        return false;
    }

    // ── Material compatible con URP y Built-in pipeline ───────────────────────
    Material CrearMaterial(Color color, float alpha)
    {
        color.a = alpha;

        Shader shaderURP = Shader.Find("Universal Render Pipeline/Lit");
        if (shaderURP != null)
        {
            Material mat = new Material(shaderURP);
            mat.SetColor("_BaseColor", color);
            if (alpha < 1f)
            {
                mat.SetFloat("_Surface", 1f);
                mat.SetFloat("_Blend",   0f);
                mat.SetFloat("_ZWrite",  0f);
                mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
                mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
                mat.renderQueue = 3000;
            }
            return mat;
        }

        Shader shaderStd = Shader.Find("Standard");
        if (shaderStd != null)
        {
            Material mat = new Material(shaderStd);
            mat.color = color;
            if (alpha < 1f)
            {
                mat.SetFloat("_Mode", 3);
                mat.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
                mat.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                mat.SetInt("_ZWrite",   0);
                mat.EnableKeyword("_ALPHABLEND_ON");
                mat.renderQueue = 3000;
            }
            return mat;
        }

        Debug.LogError("Shader no encontrado. Verificar Render Pipeline del proyecto.");
        return new Material(Shader.Find("Hidden/InternalErrorShader"));
    }
}
