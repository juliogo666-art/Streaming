# 🔍 Revisión Exhaustiva — Proyecto 4 Streaming

## Índice
1. [Estado Actual del Proyecto](#1-estado-actual)
2. [Puntos Mejorables](#2-puntos-mejorables)
3. [Errores a Arreglar](#3-errores-a-arreglar)
4. [Qué Añadiría](#4-qué-añadiría)
5. [Análisis vs Rúbrica del Enunciado](#5-análisis-vs-rúbrica)
6. [NCF / Two Towers vs Mejoras en Modelos Existentes](#6-ncf--two-towers)
7. [Propuesta de CSV de Registro de Métricas](#7-csv-de-registro)

---

## 1. Estado Actual

### Arquitectura del Proyecto
```
Proyecto 4 - Streaming
├── main.py                    # Lanzador Backend + Frontend
├── src/
│   ├── api/
│   │   ├── main_api.py        # FastAPI (711 líneas) — 5 endpoints de recomendación
│   │   ├── database.py        # Pool MySQL
│   │   └── etl.py             # Pipeline datos CSV → MySQL
│   ├── frontend/
│   │   ├── app_ui.py          # Landing Streamlit
│   │   └── pages/
│   │       ├── 1_Administrador.py  # Panel admin + EDA + Métricas
│   │       └── 2_Usuario.py   # Login/Registro + Catálogo + Recomendaciones IA
│   ├── models/
│   │   ├── jj/                # 5 modelos de producción
│   │   ├── nil/               # NCF-Lite + Benchmark (BPR-MF, NCF, LightGCN)
│   │   └── tx/                # SVD+KNN Rerank con géneros
│   ├── networks/dl/rn_mlp.py  # Arquitectura Wide&Deep (PyTorch)
│   ├── utils/
│   │   ├── evaluacion_ranking.py  # Evaluador comparativo
│   │   └── exportar_onnx.py   # Conversor PKL→Joblib + PTH→ONNX
│   └── data/                  # Raw, Clean, Ready + SQL
```

### Métricas Actuales (Top-10, 300 usuarios)

| Modelo | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 | Coverage@10 |
|--------|:---:|:---:|:---:|:---:|:---:|
| **Implicit BPR** | **24.6%** | **24.8%** | **0.3267** | **85.7%** | 7.6% |
| SVD | 7.8% | 5.7% | 0.0953 | 45.3% | 3.2% |
| KNN | 5.5% | 2.5% | 0.0567 | 26.3% | 7.6% |
| TF-IDF | 0.9% | 0.6% | 0.0148 | 8.3% | 8.2% |
| Wide&Deep | 0.2% | 0.2% | 0.0016 | 1.7% | 0.3% |

> [!IMPORTANT]
> **Implicit BPR domina abrumadoramente** en todas las métricas de ranking. Wide&Deep tiene un rendimiento prácticamente nulo en ranking — esto es un problema grave que abordo más abajo.

---

## 2. Puntos Mejorables

### 🔴 Críticos (Impacto Alto)

#### 2.1 Wide&Deep está prácticamente roto para ranking
- **NDCG = 0.0016** (frente a 0.33 de Implicit) indica que no aprende el orden correcto.
- **Causa probable**: La función `forward()` en `rn_mlp.py` usa `torch.sigmoid(prediction) * 5.5` que comprime TODAS las predicciones al rango [0, 5.5]. Combinado con el filtro de películas `MIN_RATINGS_PELICULA=5` (demasiado permisivo para una red neuronal), muchos embeddings quedan sin entrenar y sus predicciones aleatorias saturan el Top-10.
- **Solución**: Aumentar `MIN_RATINGS_PELICULA` a 50-100, probar con más épocas (20-30) y quitar o reducir el clamp del forward.

#### 2.2 README.md con contenido duplicado 3 veces
- Las instrucciones "PREPARACIÓN DB", "PREPARACIÓN BACKEND" y "PREPARACIÓN FRONTEND" se repiten **tres veces literalmente** (líneas 51-98, 90-132, 133-174). Esto da **muy mala impresión** para la rúbrica.
- Además expone credenciales en claro (`usuario: txema`, `contraseña: root`).

#### 2.3 Admin bypasea autenticación en producción
- [1_Administrador.py L32](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/frontend/pages/1_Administrador.py#L32): `st.session_state["admin_autenticado"] = True` hardcodeado con el comentario "añadido para pruebas, eliminar luego". **Esto debe quitarse antes de la entrega.**

#### 2.4 `@app.on_event("startup")` está deprecado
- FastAPI v0.93+ deprecó `on_event`. La alternativa moderna es usar **lifespan handlers** con `@asynccontextmanager`. Considerar migración.

### 🟡 Importantes (Impacto Medio)

#### 2.5 Archivos `.pkl` duplicados — desperdicio de disco
- Hay versiones `.pkl` Y `.joblib` de los mismos modelos en `src/models/jj/`:
  - `modelo_1_SVD.pkl` (717 MB) + `modelo_1_SVD.joblib` (327 MB)
  - `modelo_2_knn_cs.pkl` (902 MB) + `modelo_2_knn_cs.joblib` (515 MB)
- **Total desperdicio**: ~1.6 GB de archivos .pkl que ya no se usan en producción.

#### 2.6 Sin validación de inputs en el frontend
- El registro de usuario no valida formato de email, longitud mínima de contraseña, ni caracteres especiales.
- El `id_simulado` del sidebar permite simular cualquier usuario sin restricciones — aceptable para demo, pero debería documentarse.

#### 2.7 Importaciones inline repetidas
- `from sklearn.metrics.pairwise import cosine_similarity` se importa **dentro del endpoint** de Content-Based (línea 619 de main_api.py). Debería estar al inicio del archivo.

#### 2.8 Sin sistema de logging estructurado
- Todo usa `print()` en lugar de `logging`. Para la rúbrica de monitorización, perder los logs al cerrar la consola es un punto débil serio.

#### 2.9 Recomendaciones de Series no implementadas
- La pestaña de Series solo muestra catálogo y dice "en desarrollo". Los modelos solo trabajan con películas.

### 🟢 Menores (Pulido)

#### 2.10 Coherencia de estilos CSS
- `2_Usuario.py` tiene estilos inline premium (Azul Marino + Oro), pero `1_Administrador.py` no tiene estilos custom → inconsistencia visual entre páginas.

#### 2.11 `server_error.log` en el root del proyecto
- Debería estar en una carpeta `logs/` o ser generado por el sistema de logging.

#### 2.12 Carpeta `src/schemas/` vacía
- Debería eliminarse o usarse para los esquemas Pydantic que están dentro de main_api.py.

#### 2.13 Archivo `rules_cleaning.yaml` tiene solo 4 bytes
- Solo contiene "null" — archivo vacío/placeholder que debería completarse o eliminarse.

---

## 3. Errores a Arreglar

### 🐛 Bug 1: Bypass de admin autenticación
**Archivo**: [1_Administrador.py:32](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/frontend/pages/1_Administrador.py#L32)
```python
st.session_state["admin_autenticado"] = True  # añadido para pruebas, eliminar luego
```
**Fix**: Eliminar esta línea.

### 🐛 Bug 2: `set_page_config` se llama dos veces
**Archivo**: [2_Usuario.py:5](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/frontend/pages/2_Usuario.py#L5) y [app_ui.py:3](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/frontend/app_ui.py#L3)
Streamlit solo permite un `set_page_config` por sesión. Si las dos páginas la definen, puede dar error en producción. Solo la landing page (`app_ui.py`) debería definirla.

### 🐛 Bug 3: Variable `top_tf` referenciada sin garantía
**Archivo**: [evaluacion_ranking.py:421](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/utils/evaluacion_ranking.py#L421)
```python
if (top_svd or top_knn or top_wnd or top_tf or ...):
```
Si el modelo TFIDF no está cargado, `top_tf` nunca se define. Esto causará un `NameError` si no existe TFIDF pero sí el resto.
**Fix**: Inicializar `top_tf = []` antes del bloque de evaluación.

### 🐛 Bug 4: Cookie naming inconsistente
**Archivo**: [1_Administrador.py:27](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/frontend/pages/1_Administrador.py#L27)
```python
saved_cookie = cookies.get(cookie="disney_admin_session")
```
El nombre "disney_admin_session" debería ser "spire_admin_session" para coherencia con la marca SPIRE del frontend.

### 🐛 Bug 5: Imagen `width="stretch"` inválida en Streamlit
**Archivo**: [2_Usuario.py:409-414](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/frontend/pages/2_Usuario.py#L409)
```python
st.image(..., width="stretch")  # Parámetro inválido
```
`st.image` no acepta `width="stretch"`. El parámetro correcto es `use_container_width=True`.

### 🐛 Bug 6: Bare `except` sin tipo en varios archivos
**Archivos**: `etl.py:17`, `modelo_2_knn+cs.py:235`, `1_Administrador.py:28`
Los `except:` sin tipo pueden ocultar errores importantes. Deberían ser `except Exception:` como mínimo.

---

## 4. Qué Añadiría

### 📌 Para la rúbrica (prioritario)

1. **Documento de Especificación del Proyecto** (Imprescindible para Nivel 4-5)
   - Arquitectura del sistema con diagrama Mermaid
   - Descripción de escalabilidad (pool de conexiones, modelos en RAM, cache)
   - Análisis de alta disponibilidad (replicación MySQL, healthcheck endpoints)
   - Sistema de ficheros distribuidos (HDFS o justificación de alternativa)
   - Plan de contingencia ante fallos

2. **Sistema de Monitorización** (M5RA4)
   - Dashboard con métricas de sistema: latencia endpoints, uso RAM, tiempos de carga de modelos
   - Alertas cuando un modelo falla en cargar
   - Logs estructurados (`logging` con formateo JSON) persistidos en archivo

3. **Manual de Instalación y Despliegue en Inglés** (Requisito explícito)
   - Docker Compose para MySQL + Backend + Frontend
   - Instrucciones de despliegue paso a paso
   - Requisitos de infraestructura (RAM ≥16GB para los modelos)

4. **Manual de Usuario Final**
   - Guía visual del flujo login → catálogo → recomendaciones
   - FAQ y troubleshooting

### 📌 Para mejorar el producto

5. **Endpoint de healthcheck ampliado**
   - `/health` que reporte qué modelos están cargados, cuánto tardaron, y estado de MySQL

6. **Sistema de feedback del usuario**
   - Botón 👍/👎 en cada recomendación para retroalimentar modelos futuros
   - Rating explícito (1-5 estrellas) guardado en BD

7. **Cache de recomendaciones**
   - Redis o `lru_cache` para evitar recalcular recomendaciones en cada petición

8. **Tests unitarios**
   - Tests para endpoints, modelos y ETL — la rúbrica valora "diversos casos de prueba"

9. **Recomendaciones para Series**
   - El dataset de series ya existe (`dataset_final_shows.csv`, 28MB). Falta entrenar modelos homólogos o adaptar un modelo genérico.

---

## 5. Análisis vs Rúbrica

### Mapeo de Entregables vs Estado Actual

| Entregable | Rúbrica | Estado | Nivel Estimado | Gaps |
|---|---|---|:---:|---|
| Documento Especificación | M5RA1 | ❌ No existe | 1 | Falta documento completo |
| Escalabilidad | M5RA2 | ⚠️ Parcial (pool MySQL) | 2 | Sin análisis formal |
| Alta Disponibilidad | M5RA3 | ⚠️ Parcial | 2 | Sin redundancia ni failover |
| Sistema Ficheros Distribuidos | M5RA3 | ❌ No descrito | 1 | Sin HDFS ni alternativa justificada |
| Monitorización | M5RA4 | ❌ Solo consola | 1 | Sin dashboard ni alertas |
| Informes Rendimiento | M5RA4 | ✅ `metricas_ranking.csv` | 3 | Falta interpretación formal |
| Algoritmos Big Data | M5RA5 | ✅ 5 modelos implementados | 4 | Buena documentación en código |
| Código + Control Versiones | M5RA1-5 | ✅ Git + Comentarios | 4 | README pobre y duplicado |
| Vídeo Funcionamiento | M5RA1-5 | ❓ No verificable | ? | Aún no hecho (se presume) |
| Vídeo Monitorización EN INGLÉS | M5RA4 | ❓ No verificable | ? | Requiere sistema de monitoring |
| Manual Instalación EN INGLÉS | M5RA2,3,5 | ❌ No existe | 1 | Crítico para Nivel ≥3 |
| Manual Usuario | M5RA1 | ❌ No existe | 1 | Necesario para la entrega |

> [!CAUTION]
> **Los 4 mayores gaps son: Documento de Especificación, Monitorización, Manual de Instalación (EN INGLÉS), y Manual de Usuario.** Sin estos, el proyecto difícilmente pasará del Nivel 2 en esos criterios aunque el código sea bueno.

---

## 6. NCF / Two Towers vs Mejoras en Modelos Existentes

### 6.1 ¿Qué dice el Benchmark de Nil?

El `benchmark_modelos.py` ya probó 3 arquitecturas con K-Core(500,500), 2 épocas:

| Modelo | Precision@10 | NDCG@10 | Recall@10 | Tiempo/Época | RAM |
|---|:---:|:---:|:---:|:---:|:---:|
| **NCF-Lite** | **0.0332** | **0.1972** | **0.3320** | 1581s | 828 MB |
| BPR-MF | 0.0330 | 0.1878 | 0.3300 | 48s | ~8 MB |
| LightGCN | 0.0316 | 0.1796 | 0.3160 | 15139s | 1433 MB |

### 6.2 Mi Análisis y Recomendación

#### ¿Hacer un NCF para producción?
**Sí, pero con matices.**

El NCF de Nil ya está hecho y exportado a ONNX (`ncf_model.onnx`, 5.6MB). Tiene sentido **integrarlo como 6º modelo** en el backend y el evaluador:

- **Pros**: Añade diversidad al pipeline, NCF es una referencia clásica en papers de Deep Learning para RecSys, y ya tiene los artefactos (ONNX + JSON mappings).
- **Contras**: Con los filtros actuales (K-Core 200/100), el recall es del 33% — bueno, pero no supera al 24.8% de Implicit BPR con filtros más permisivos. **Las métricas no son directamente comparables** porque usan filtros K-Core diferentes.

#### ¿Hacer un Two Towers (Retrieval + Ranking)?
**Lo veo excesivo para el alcance del proyecto**, pero sería impresionante para la rúbrica si se hace bien:

```
Two Towers = Torre de Usuario + Torre de Item
    ↓              ↓
Embeddings separados → Product Dot → Score de relevancia
```

- Es la arquitectura de YouTube/Google/TikTok para retrieval a escala.
- Requiere más tiempo de desarrollo que integrar el NCF existente.
- **Mi recomendación**: Si queda tiempo, hacerlo como "extra". Pero el ROI de tiempo es mejor en **arreglar el Wide&Deep** y **escribir la documentación** que falta para la rúbrica.

#### ¿O mejor mejorar los modelos existentes?
**Sí, esta es la ruta que más valor aporta:**

1. **Wide&Deep** → Arreglar el rendimiento de ranking (NDCG 0.0016 es inaceptable):
   - Aumentar `MIN_RATINGS_PELICULA` de 5 a 50-100
   - Entrenar con más épocas (20+)
   - Considerar reemplazar `sigmoid * 5.5` por salida lineal + MSELoss
   - Alternativamente, cambiar la loss a BPR/BCE como hace NCF-Lite

2. **Implicit BPR** → Ya es el mejor. Podría mejorar aún más con:
   - Tuning de factores (probar 128, 256)
   - Probar ALS en lugar de BPR

3. **KNN** → Su Coverage es buena (7.6%) pero su precision es baja:
   - Probar `KNNWithMeans` o `KNNWithZScore` en lugar de `KNNBasic`

### 6.3 Plan Recomendado (Priorizado)

| Prioridad | Acción | Impacto en Rúbrica | Esfuerzo |
|:---:|---|:---:|:---:|
| 🔴 P1 | Arreglar bugs (admin bypass, etc.) | Alto | 30 min |
| 🔴 P1 | Limpiar README.md | Alto | 1 hora |
| 🔴 P1 | Escribir Documento de Especificación | **Crítico** | 3-4 horas |
| 🔴 P1 | Manual de Instalación EN INGLÉS | **Crítico** | 2 horas |
| 🔴 P1 | Manual de Usuario | **Crítico** | 2 horas |
| 🟡 P2 | Sistema de Monitorización + Dashboard | Alto | 3-4 horas |
| 🟡 P2 | Integrar NCF de Nil como 6º modelo | Medio | 2-3 horas |
| 🟡 P2 | Arreglar Wide&Deep (re-entrenar) | Medio | 2-3 horas |
| 🟢 P3 | Two Towers (si queda tiempo) | Impresionante | 6-8 horas |
| 🟢 P3 | Tests unitarios | Medio | 3 horas |
| 🟢 P3 | Recomendaciones para Series | Bajo | 4-5 horas |

---

## 7. CSV de Registro de Métricas

### Tu idea: ¿Merece la pena?

> [!TIP]
> **Sí, es una buena idea** y encaja perfectamente con la rúbrica de "Informes de monitorización y análisis de rendimiento" (M5RA4, Nivel 4-5).

### Diseño Propuesto

Un CSV acumulativo que registre cada ejecución de evaluación con sus hiperparámetros:

```csv
timestamp,modelo,n_factores,n_epocas,learning_rate,regularizacion,min_ratings_user,min_ratings_item,MAE,RMSE,NDCG_10,Precision_10,Recall_10,Hit_Rate_10,Coverage_10,train_time_s,dataset_size
2026-04-03 18:30,SVD,100,20,0.005,0.02,20,1,0.68,0.87,0.0953,0.078,0.057,0.453,0.032,245.3,25769933
2026-04-03 18:30,SVD,150,30,0.003,0.03,20,1,0.65,0.84,0.1102,0.089,0.064,0.512,0.028,412.7,25769933
...
```

### Columnas clave:
- **Identificación**: `timestamp`, `modelo`, `dataset_size`
- **Hiperparámetros**: `n_factores`, `n_epocas`, `learning_rate`, `regularizacion`, `min_ratings_user`, `min_ratings_item`
- **Métricas de Error**: `MAE`, `RMSE` (para SVD/KNN/W&D que predicen ratings)
- **Métricas de Ranking**: `NDCG_10`, `Precision_10`, `Recall_10`, `Hit_Rate_10`, `Coverage_10`
- **Rendimiento**: `train_time_s`

### Implementación

Se modificaría cada script de entrenamiento de modelo para que, al finalizar, **appenda una fila** al CSV:

```python
import csv
from datetime import datetime

def registrar_metricas(modelo_nombre, hiperparams, metricas, ruta_csv="src/utils/historial_metricas.csv"):
    """Añade una fila al registro acumulativo de métricas."""
    fila = {
        "timestamp": datetime.now().isoformat(),
        "modelo": modelo_nombre,
        **hiperparams,
        **metricas,
    }
    
    existe = os.path.exists(ruta_csv)
    with open(ruta_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fila.keys())
        if not existe:
            writer.writeheader()
        writer.writerow(fila)
```

### Conexión con el evaluador

El `evaluacion_ranking.py` ya genera un CSV puntual (`metricas_ranking.csv`). Con este historial acumulativo:
1. Cada vez que re-entrenas un modelo con distintos hiperparámetros → se registra.
2. Luego ejecutas `evaluacion_ranking.py` → sus resultados también se registran.
3. En el panel Admin, una nueva pestaña **"Historial de Experimentos"** podría mostrar la evolución temporal de cada modelo.

### ¿Es innecesario?
No es innecesario, pero **no es urgente**. Es un "nice to have" que:
- Suma puntos para M5RA4 (Nivel 4-5: "Informe completo con análisis de puntos críticos y recomendaciones")
- Facilita comparar diferentes configuraciones de hiperparámetros de forma organizada
- Es útil para el vídeo de demostración (mostrar evolución de métricas)

**Mi recomendación**: Implementarlo DESPUÉS de los documentos obligatorios, como una mejora de la fase de monitorización.

---

## Preguntas clave antes de proceder

> [!IMPORTANT]
> Antes de ejecutar cualquier cambio, necesito que confirmes:

1. **¿Quieres que priorice la documentación (especificación, manuales) o las mejoras técnicas (modelos, bugs)?** Los dos bloques son necesarios, pero el tiempo es limitado.

2. **¿Integro el NCF de Nil como sexto modelo** ahora, o prefieres dejarlo como está y centrarnos en arreglar el Wide&Deep primero?

3. **¿El vídeo de demostración y el vídeo de monitorización en inglés ya los tenéis planificados?** Porque el sistema de monitorización es prerequisito para el segundo vídeo.

4. **¿Quieres que haga el Two Towers** como modelo experimental, o lo descartamos para centrarnos en lo prioritario de la rúbrica?

5. **¿Elimino los ficheros .pkl duplicados** (~1.6 GB) que ya están convertidos a .joblib?
