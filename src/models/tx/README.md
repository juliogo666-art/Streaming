# Modelos `tx/` - Guia de testeo

Este documento resume como se han testeado los modelos de `src/models/tx/`:

- `model_SVD_CS.py`
- `model_SVD_KNN_RERANK.py`
- `model_SVD_KNN_RERANK_con_generos.py`

## 1) Dataset y objetivo

- **Ratings**: `src/data/ready/ratings_finales_ia.csv`
- **Catalogo**: `src/data/ready/dataset_final_movies.csv`
- **Objetivo principal**: recomendacion **top-N** (ranking), no solo prediccion de nota.

## 2) Protocolo de evaluacion usado

La evaluacion global se ejecuta desde:

- `src/utils/evaluacion_ranking.py`

### Flujo general del benchmark

1. Se cargan interacciones de ratings y catalogo.
2. Se filtran peliculas candidatas (por popularidad de catalogo).
3. Se seleccionan usuarios de test con historial suficiente.
4. Para cada usuario:
   - Se oculta una parte de peliculas relevantes (holdout).
   - Se generan candidatas excluyendo lo ya visible.
   - Cada modelo devuelve su top-K.
5. Se comparan recomendaciones contra los ocultos.

## 3) Metricas utilizadas

En el benchmark se calculan:

- **Precision@K**: proporcion de recomendaciones relevantes en el top-K.
- **Recall@K**: proporcion de relevantes recuperados del total relevante.
- **HitRate@K**: si hubo al menos un acierto en el top-K.
- **NDCG@K**: calidad del orden en el ranking (aciertos altos puntuan mas).
- **MRR@K**: posicion del primer acierto relevante.
- **Coverage@K**: cobertura del catalogo recomendado.

Estas metricas se computan con el pipeline:

- `src/pipelines/evaluation_pipeline.py`
- modulos de `src/metrics/`

## 4) Como se testea cada modelo de `tx/`

## `model_SVD_CS.py`

- Entrena SVD y recomienda con score de **similitud coseno** en espacio latente.
- En benchmark, `evaluacion_ranking.py` carga el payload (`modelo` + `trainset`) y usa:
  - `cargar_modelo_guardado()`
  - `recomendar_con_coseno(...)`

## `model_SVD_KNN_RERANK.py`

- Pipeline hibrido:
  - reduccion latente (SVD),
  - candidatos por KNN en espacio latente,
  - rerank por senales de catalogo.
- Se testea con top-K sobre candidatas de evaluacion.

## `model_SVD_KNN_RERANK_con_generos.py`

- Igual base que el anterior, pero con senal adicional de **afinidad de generos** en rerank.
- Se testea en el mismo protocolo para comparabilidad directa.

## 5) Comandos de ejecucion

### Entrenar / generar artefactos del modelo SVD+CS

```bash
uv run .\src\models\tx\model_SVD_CS.py
```

### Ejecutar benchmark de ranking (incluye TX_SVD_CS)

```bash
uv run .\src\utils\evaluacion_ranking.py
```

## 6) Reproducibilidad y notas practicas

- Mantener semillas fijas en splits cuando aplique (`random_state`).
- No mezclar caches incompatibles entre experimentos.
- Comparar modelos con **mismo K**, mismo conjunto de usuarios y mismo protocolo.
- Para top-N, priorizar NDCG/Precision/Recall sobre RMSE/MAE.

## 7) Salida de resultados

- El resumen final del benchmark se guarda en:
  - `src/utils/metricas_ranking.csv`

Si se actualiza el protocolo (usuarios, K, umbrales o filtros), reflejarlo aqui para mantener trazabilidad de experimentos.
