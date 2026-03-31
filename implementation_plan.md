# Plan de Implementación: Modelos Adicionales y Mejoras del Sistema de Recomendación

## Estado Actual del Proyecto

Tras revisar todo el código y los documentos del proyecto, este es el inventario:

| # | Modelo | Tipo | Estado | Archivo | Tamaño | Integrado API |
|---|--------|------|--------|---------|--------|---------------|
| 1 | **SVD** | CF Explícito | ✅ Entrenado | `modelo_1_SVD.pkl` | 930 MB | ✅ `/recomendar/{user_id}` |
| 2 | **KNN+Cosine** | CF Memoria | ✅ Entrenado | `modelo_2_knn_cs.pkl` | 900 MB | ✅ `/recomendar/knn/{user_id}` |
| 3 | **Wide&Deep** | DL Híbrido | ✅ Entrenado | `modelo_3_wnd.pth` | 45 MB | ✅ `/recomendar/wnd/{user_id}` |
| 4 | **TF-IDF Content** | Content-Based | ✅ Entrenado | `modelo_4_*.pkl` | ~17 MB | ✅ `/recomendar/content/{user_id}` |

Además existe un script de evaluación de ranking ([evaluacion_ranking.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/utils/evaluacion_ranking.py)) que ya calcula **Precision@K**, **NDCG@K** y **Hit Rate@K**.

---

## 1. ¿Hacer LightFM? — SÍ, RECOMENDADO

### Veredicto: ✅ **SÍ, implementar**

LightFM es el modelo que más valor aportaría al proyecto ahora mismo por estas razones:

| Aspecto | Justificación |
|---------|---------------|
| **Cold Start** | Es el ÚNICO modelo capaz de recomendar a usuarios nuevos Y películas nuevas simultáneamente, usando features de ambos (género, idioma, año para películas; y features del usuario si las tuvieras) |
| **Híbrido real** | Vuestros 4 modelos actuales son: 2 de CF puro (SVD, KNN), 1 DL que solo usa IDs (Wide&Deep), y 1 de contenido puro (TF-IDF). LightFM es el único que **combina** CF + Content de forma nativa en una sola matriz de factores |
| **Eficiencia** | Entrena en minutos (no horas), consume poca RAM, y el modelo resultante pesa < 10 MB |
| **WARP loss** | Soporta la función de pérdida WARP (Weighted Approximate-Rank Pairwise) que es específica para **ranking**, no para predecir ratings. Es exactamente lo que los PDFs del curso piden |
| **Datos disponibles** | Tenéis `genre_ids`, `overview`, `original_language`, `popularity`, `vote_average` en el catálogo  — perfecto como item features |

### Qué features usar
```
Item Features: genre_ids (one-hot), original_language, década de release
User Features: (no tenéis datos demográficos, pero LightFM funciona sin ellos)
Interacciones: ratings ≥ 3.5 → positivo, resto → ignorar
Loss: WARP (optimiza ranking directamente)
```

### Fichero: `modelo_5_lightfm.py`

---

## 2. ¿Hacer Two-Tower? — NO en esta fase

### Veredicto: ⚠️ **No por ahora** — Lo que tenéis ya lo cubre parcialmente

| Argumento | Detalle |
|-----------|---------|
| **Complejidad** | Requiere TensorFlow Recommenders o implementación manual en PyTorch. Duplicáis esfuerzo con Wide&Deep |
| **Infraestructura** | La arquitectura Two-Tower brilla en producción con millones de usuarios en tiempo real (Google, YouTube). Vuestro proyecto es académico con serving síncrono |
| **El PDF lo menciona** | Sí, el PDF de Retrieval+Ranking Architectures lo explica, pero como **concepto teórico a conocer**, no como requisito de implementación |
| **Alternativa mejor** | Con LightFM + un índice FAISS sobre los embeddings de SVD/W&D, lográis la misma separación Retrieval → Ranking sin montar una segunda red neuronal completa |

> [!TIP]
> Si el profesor pregunta por Two-Tower, la respuesta correcta es: "Implementamos la separación Retrieval/Ranking usando embeddings precalculados + FAISS para candidatos, y luego un modelo de ranking (Wide&Deep o LightFM) para re-ordenar los candidatos". Esto es esencialmente lo que hace un Two-Tower, pero más pragmático.

---

## 3. ¿Hacer ALS? — NO, sería redundante

### Veredicto: ❌ **No**

| Argumento | Detalle |
|-----------|---------|
| **Redundancia** | ALS (Alternating Least Squares) resuelve exactamente el mismo problema que SVD: factorización de la matriz usuario-ítem. SVD de Surprise y ALS de PySpark son **algoritmos hermanos** — ambos descomponen la misma matriz |
| **Diferencia real** | ALS brilla con **datos implícitos** (clics, tiempo de visualización, compras sin rating). Vosotros tenéis **ratings explícitos** (0.5-5.0 estrellas), donde SVD es superior |
| **PySpark** | ALS es nativo en PySpark, pero vuestro proyecto NO usa Spark en producción (solo lo usaste en otro proyecto). Añadir PySpark como dependencia solo para un modelo redundante no tiene sentido |
| **Performance** | Con 33M de ratings, SVD de Surprise ya entrena razonablemente. ALS no aportaría una mejora significativa en calidad |

> [!NOTE]
> Si se quiere demostrar conocimiento de ALS en la documentación, se puede **mencionar** en el documento de especificación como "alternativa considerada y descartada por ser redundante con SVD dado que nuestros datos son explícitos". Eso demuestra criterio técnico, que vale más que implementar todo.

---

## 4. ¿Reemplazar Wide&Deep por NCF? — NO, mejor mejorar el Wide&Deep

### Veredicto: ⚠️ **No reemplazar, sino mejorar**

| Aspecto | Wide&Deep actual | NCF (He et al. 2017) |
|---------|-----------------|----------------------|
| **Arquitectura** | Wide (embeddings directos) + Deep (MLP 64→32) | GMF (producto elemento a elemento) + MLP |
| **Diferencia real** | La principal diferencia es que NCF usa `element-wise product` en la parte "GMF" en vez de concatenación. Es una variante, no una revolución |
| **El paper NCF** | He et al. demostraron mejora sobre MF puro, pero **no** sobre Wide&Deep con features. Wide&Deep de Google (2016) y NCF de He (2017) están al mismo nivel |
| **Esfuerzo** | Reescribir la red desde cero y re-entrenar vs. mejorar lo existente |

### Lo que SÍ mejoraría el Wide&Deep actual:

1. **Añadir features del catálogo** al componente Wide (actualmente solo usa user_id × movie_id, no usa géneros, popularidad, idioma)
2. **Corregir el rango de predicción**: `sigmoid × 5.5` debería ser `sigmoid × 4.5 + 0.5` para mapear exactamente a [0.5, 5.0]
3. **Añadir Dropout** en las capas Deep para regularización
4. **Batch Normalization** entre capas

> [!IMPORTANT]
> Mejorar el Wide&Deep existente da más impacto con menos esfuerzo que reescribir un NCF desde cero. Además, podéis explicar en la documentación por qué escogisteis Wide&Deep sobre NCF (mayor madurez, soporte de features, arquitectura de Google en producción).

---

## 5. Exportar modelos a ONNX — PARCIALMENTE posible

### Análisis por modelo:

| Modelo | ¿ONNX posible? | Detalle |
|--------|----------------|---------|
| **SVD (Surprise)** | ❌ **No directamente** | Surprise no es un framework de tensores. El modelo SVD de Surprise almacena matrices NumPy internas (`pu`, `qi`, `bu`, `bi`). Se puede exportar manualmente: extraer las matrices y crear un operador ONNX custom, pero es complejo y sin beneficio real (el predict de Surprise es una multiplicación de vectores instantánea) |
| **KNN (Surprise)** | ❌ **No viable** | KNN almacena una matriz de similitudes completa. No tiene un "forward pass" que ONNX pueda representar |
| **Wide&Deep (PyTorch)** | ✅ **SÍ** | PyTorch tiene `torch.onnx.export()` nativo. Es 3 líneas de código. El modelo ya está en `.pth`, convertir a `.onnx` es trivial |
| **TF-IDF (sklearn)** | ⚠️ **Posible con skl2onnx** | La librería `skl2onnx` puede convertir TfidfVectorizer, pero la similitud coseno post-hoc no se puede empaquetar en ONNX fácilmente |
| **LightFM (nuevo)** | ❌ **No viable** | LightFM tiene su propia representación interna, no compatible con ONNX |

### Recomendación práctica:

```python
# Wide&Deep → ONNX (3 líneas)
dummy_user = torch.tensor([0], dtype=torch.long)
dummy_movie = torch.tensor([0], dtype=torch.long)
torch.onnx.export(model, (dummy_user, dummy_movie), "modelo_3_wnd.onnx",
                  input_names=["user_id", "movie_id"],
                  output_names=["rating"],
                  dynamic_axes={"user_id": {0: "batch"}, "movie_id": {0: "batch"}})
```

> [!WARNING]
> ONNX es el mejor formato para **deployment en producción multiplataforma** (mobile, C++, JavaScript), pero para un proyecto académico de Python, el beneficio real es mínimo. Lo recomiendo **solo para el Wide&Deep** como demostración técnica, y mantener el resto en sus formatos nativos.

### Acción concreta:
- ✅ Exportar Wide&Deep a `.onnx` (fácil, 3 líneas)
- ✅ Añadir script de exportación `exportar_onnx.py`
- ❌ No forzar ONNX en SVD/KNN/LightFM (más complejidad que beneficio)

---

## 6. ¿Qué métricas implementar? — TODAS las que ya tenéis + 3 más

### Estado actual de métricas:

| Métrica | Tipo | ¿Implementada? | Script |
|---------|------|----------------|--------|
| RMSE | Rating | ✅ En cada modelo | modelo_1/2/3 |
| MAE | Rating | ✅ En cada modelo | modelo_1/2/3 |
| Precision@K | Ranking | ✅ | evaluacion_ranking.py |
| NDCG@K | Ranking | ✅ | evaluacion_ranking.py |
| Hit Rate@K | Ranking | ✅ | evaluacion_ranking.py |
| **Recall@K** | Ranking | ❌ Falta | Añadir |
| **MAP** | Ranking | ❌ Falta | Añadir |
| **Coverage** | Diversidad | ❌ Falta | Añadir |
| **Diversity** | Diversidad | ❌ Falta | Opcional |

### Recomendación: Implementar **7 métricas** (las 5 actuales + Recall@K + Coverage)

| Métrica | Por qué |
|---------|---------|
| **Recall@K** | Complementa directamente a Precision@K. Es trivial de implementar (2 líneas) y los PDFs la mencionan |
| **Coverage** | El enunciado pide "algoritmos y técnicas Big Data". Demostrar que vuestro sistema no tiene "popularity bias" (que solo recomienda las 50 mismas películas) es importante. Coverage mide qué % del catálogo total aparece en las recomendaciones |
| **MAP** | Opcional. Si ya tenéis NDCG, MAP es redundante (ambas miden calidad del ranking). Solo la añadiría si el PDF la pide explícitamente |
| **Diversity** | Opcional. Interesante pero no prioritaria |

### Código a añadir en `evaluacion_ranking.py`:

```python
def recall_at_k(recomendadas, relevantes):
    """De todas las películas relevantes, ¿qué % captura el top-K?"""
    if not relevantes:
        return 0.0
    aciertos = len(set(recomendadas) & set(relevantes))
    return aciertos / len(relevantes)

def coverage(todas_recomendaciones, catalogo_total):
    """¿Qué % del catálogo aparece en alguna recomendación?"""
    pelis_recomendadas = set()
    for recs in todas_recomendaciones:
        pelis_recomendadas.update(recs)
    return len(pelis_recomendadas) / len(catalogo_total)
```

---

## 7. Negative Sampling — SÍ, implementar

### ¿Por qué es necesario?

Actualmente todos los modelos entrenan **solo con ratings explícitos positivos**. Esto crea un sesgo grave:

```
El modelo solo ve: "Al usuario 42 le gustó Matrix (5★), Inception (4.5★), Interstellar (4★)"
El modelo NUNCA ve: "Al usuario 42 NO le interesó Barbie, Frozen, Titanic"
→ El modelo no aprende a RECHAZAR, solo a ACEPTAR → Recomienda cualquier cosa popular
```

### Cómo implementarlo:

| Estrategia | Descripción | Modelo aplicable |
|-----------|-------------|-----------------|
| **Random Negatives** | Por cada rating positivo, muestrear 4 películas aleatorias que el usuario NO ha visto como "negativos" | Wide&Deep, LightFM |
| **Popularity-Based Negatives** | Muestrear negativos ponderados por popularidad (películas populares que el usuario ignoró son negativos más informativos) | Wide&Deep, LightFM |
| **Hard Negatives** | Usar el modelo actual para predecir, y elegir como negativos las películas con predicción alta que el usuario realmente no vio | Wide&Deep (2ª pasada) |

### Implementación concreta para Wide&Deep:

```python
# En cargar_y_preparar_datos() del modelo_3_wide&deep.py:

def generar_negativos(df, ratio=4):
    """Por cada interacción positiva, genera `ratio` negativos aleatorios."""
    all_movies = set(df['tmdb_id'].unique())
    negativos = []
    
    for user_id, group in df.groupby('userId'):
        pelis_vistas = set(group['tmdb_id'])
        pelis_no_vistas = list(all_movies - pelis_vistas)
        
        n_neg = min(len(group) * ratio, len(pelis_no_vistas))
        neg_sample = np.random.choice(pelis_no_vistas, n_neg, replace=False)
        
        for movie_id in neg_sample:
            negativos.append({
                'userId': user_id,
                'tmdb_id': movie_id,
                'rating': 0.0  # Negativo implícito
            })
    
    df_neg = pd.DataFrame(negativos)
    return pd.concat([df, df_neg], ignore_index=True)
```

### Para LightFM:
LightFM ya tiene negative sampling **integrado** cuando usas la loss WARP. No necesitas implementarlo manualmente — es una de sus grandes ventajas.

---

## Resumen de Decisiones

| Pregunta | Decisión | Razón |
|----------|----------|-------|
| **¿LightFM?** | ✅ **SÍ** | Híbrido real, resuelve Cold Start, WARP para ranking, rápido |
| **¿Two-Tower?** | ❌ **No** | Excesivo para proyecto académico. Separar Retrieval/Ranking con embeddings + FAISS es suficiente |
| **¿ALS?** | ❌ **No** | Redundante con SVD en datos explícitos |
| **¿NCF vs W&D?** | ⚠️ **Mejorar W&D** | Añadir features al Wide, corregir rango, no reescribir |
| **¿ONNX?** | ⚠️ **Solo W&D** | PyTorch → ONNX trivial. El resto no merece la complejidad |
| **¿Métricas?** | ✅ **7 métricas** | Las 5 actuales + Recall@K + Coverage |
| **¿Negative Sampling?** | ✅ **SÍ** | Para W&D manual, para LightFM automático con WARP |

---

## Tareas Ordenadas por Prioridad

### Fase 1: Modelo nuevo + Mejoras (Impacto alto)

- [ ] **Implementar LightFM** (`modelo_5_lightfm.py`)
  - Instalar `lightfm` 
  - Preparar item features (géneros one-hot)
  - Entrenar con WARP loss
  - Integrar endpoint `/recomendar/lightfm/{user_id}`

- [ ] **Mejorar Wide&Deep**
  - Corregir rango predicción: `sigmoid * 4.5 + 0.5`
  - Añadir Dropout (0.2) entre capas
  - Implementar Negative Sampling (ratio 4:1)

### Fase 2: Métricas + Evaluación

- [ ] **Añadir Recall@K y Coverage** a `evaluacion_ranking.py`
- [ ] **Re-evaluar todos los modelos** (ahora 5) con las 7 métricas
- [ ] **Generar tabla comparativa** final para la documentación

### Fase 3: Exportación y Polish

- [ ] **Exportar Wide&Deep a ONNX** (`exportar_onnx.py`)
- [ ] **Integrar LightFM en el selector de Streamlit**
- [ ] **Dashboard comparativo** en Admin

---

## Open Questions

> [!IMPORTANT]
> 1. **¿Habéis ejecutado ya `evaluacion_ranking.py`?** Necesito saber si tenéis resultados actuales de NDCG/Precision/HR para los 4 modelos antes de decidir si el Wide&Deep necesita mejoras urgentes o si ya funciona aceptablemente.
> 
> 2. **¿Qué resultados de RMSE/MAE habéis obtenido** hasta ahora para SVD, KNN y Wide&Deep? Esto me ayudará a calibrar las expectativas.
>
> 3. **¿Queréis que LightFM use los mismos datos filtrados** que los otros modelos (usuarios con ≥20 ratings) o que use todo el dataset (33M filas) aprovechando que es mucho más rápido?
>
> 4. **El archivo `rn_mlp.py`** que define la clase `WideAndDeepModel` parece no existir (solo queda en `__pycache__`). ¿Se borró accidentalmente? Lo necesito para mejorar la arquitectura y para la exportación ONNX.

## Verification Plan

### Automated Tests
- Entrenar LightFM y verificar que NDCG@10 > 0.05 (baseline razonable)
- Exportar Wide&Deep a ONNX y verificar que las predicciones ONNX == predicciones PyTorch (diferencia < 1e-5)
- Ejecutar `evaluacion_ranking.py` con los 5 modelos y generar CSV comparativo

### Manual Verification
- Probar el endpoint `/recomendar/lightfm/{user_id}` desde el navegador
- Verificar Cold Start con user_id inventado (999999) en LightFM vs TF-IDF
- Comprobar que el selector de modelos en Streamlit muestra los 5 modelos
