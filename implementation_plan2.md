# Plan de Implementación: Reorganización de Artefactos + Evaluación Unificada

## Contexto

Reorganizar el proyecto para:
1. Mover todos los modelos entrenados a `artifacts/` con estructura clara
2. Actualizar todas las rutas en los scripts que generan y consumen los modelos
3. Mejorar el pipeline de evaluación para cubrir todos los modelos
4. Unificar el tracking de métricas en un CSV centralizado

## User Review Required

> [!IMPORTANT]
> **Estructura de `artifacts/`**: Se propone dividir en 4 subcarpetas + una de datos auxiliares:
> ```
> artifacts/
> ├── weights/       # Modelos clásicos (.pkl, .joblib) - SVD, KNN, Implicit, TF-IDF
> ├── exports/       # Modelos exportados (.onnx) - W&D, NCF, Two-Towers  
> ├── checkpoints/   # Pesos PyTorch (.pth) - W&D, Two-Towers
> └── mappings/      # Mapeos de IDs (.pkl, .json) - wnd_mappings, ncf_user2idx, etc.
> ```

> [!WARNING]
> **37 archivos de artefactos en `src/models/jj/`** que se moverán. Los scripts `.py` permanecen donde están (no se borra nada).

---

## Proposed Changes

### Componente 1: Mover artefactos a `artifacts/`

#### Archivos a mover (jj/)

| Archivo origen (`src/models/jj/`) | Destino (`artifacts/`) | Categoría |
|---|---|---|
| `modelo_1_SVD.pkl` | `weights/modelo_1_SVD.pkl` | Peso clásico |
| `modelo_1_SVD.joblib` | `weights/modelo_1_SVD.joblib` | Peso clásico |
| `modelo_2_knn_cs.pkl` | `weights/modelo_2_knn_cs.pkl` | Peso clásico |
| `modelo_2_knn_cs.joblib` | `weights/modelo_2_knn_cs.joblib` | Peso clásico |
| `modelo_2.5_knn_msd.pkl` | `weights/modelo_2.5_knn_msd.pkl` | Peso clásico |
| `modelo_2.5_knn_msd.joblib` | `weights/modelo_2.5_knn_msd.joblib` | Peso clásico |
| `modelo_4_tfidf.pkl` | `weights/modelo_4_tfidf.pkl` | Peso clásico |
| `modelo_4_tfidf.joblib` | `weights/modelo_4_tfidf.joblib` | Peso clásico |
| `modelo_4_matriz.pkl` | `weights/modelo_4_matriz.pkl` | Peso clásico |
| `modelo_4_matriz.joblib` | `weights/modelo_4_matriz.joblib` | Peso clásico |
| `modelo_4_indices.pkl` | `weights/modelo_4_indices.pkl` | Peso clásico |
| `modelo_4_indices.joblib` | `weights/modelo_4_indices.joblib` | Peso clásico |
| `modelo_5_implicit.pkl` | `weights/modelo_5_implicit.pkl` | Peso clásico |
| `modelo_5_implicit_dataset.pkl` | `weights/modelo_5_implicit_dataset.pkl` | Peso clásico |
| `modelo_5_metricas.json` | `weights/modelo_5_metricas.json` | Datos auxiliares |
| `modelo_3_wnd.pth` | `checkpoints/modelo_3_wnd.pth` | Checkpoint PyTorch |
| `modelo_7_twotowers.pth` | `checkpoints/modelo_7_twotowers.pth` | Checkpoint PyTorch |
| `modelo_3_wnd.onnx` | `exports/modelo_3_wnd.onnx` | Exportación ONNX |
| `modelo_3_wnd.onnx.data` | `exports/modelo_3_wnd.onnx.data` | Datos ONNX |
| `modelo_6_ncf.onnx` | `exports/modelo_6_ncf.onnx` | Exportación ONNX |
| `modelo_6_ncf.onnx.data` | `exports/modelo_6_ncf.onnx.data` | Datos ONNX |
| `modelo_7_twotowers.onnx` | `exports/modelo_7_twotowers.onnx` | Exportación ONNX |
| `modelo_7_twotowers.onnx.data` | `exports/modelo_7_twotowers.onnx.data` | Datos ONNX |
| `wnd_mappings.pkl` | `mappings/wnd_mappings.pkl` | Mapeo |
| `twotowers_mappings.pkl` | `mappings/twotowers_mappings.pkl` | Mapeo |
| `ncf_user2idx.json` | `mappings/ncf_user2idx.json` | Mapeo |
| `ncf_item2idx.json` | `mappings/ncf_item2idx.json` | Mapeo |

#### Archivos a mover (nil/)

| Archivo origen (`src/models/nil/`) | Destino (`artifacts/`) |
|---|---|
| `ncf_model.onnx` | `exports/nil_ncf_model.onnx` |

---

### Componente 2: Actualizar rutas en código fuente

Los scripts `.py` **no se mueven ni se borran**, solo se cambian las constantes de ruta.

#### [MODIFY] [main_api.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/api/main_api.py)
- Líneas 43-55: Actualizar las 13 rutas de modelos de `src/models/jj/...` a `artifacts/weights/...`, `artifacts/exports/...`, `artifacts/mappings/...`

#### [MODIFY] [evaluacion_ranking.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/utils/evaluacion_ranking.py)
- Líneas 19-32: Actualizar las 14 rutas de modelos

#### [MODIFY] [exportar_onnx.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/utils/exportar_onnx.py)
- Líneas 17-29: Actualizar rutas de modelos

#### [MODIFY] [modelo_1_SVD.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/models/jj/modelo_1_SVD.py)
- Línea 42: Ruta de guardado `.pkl` → `artifacts/weights/`

#### [MODIFY] [modelo_2_knn+cs.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/models/jj/modelo_2_knn+cs.py)
- Línea 54: Ruta de guardado

#### [MODIFY] [modelo_2.5_knn_msd.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/models/jj/modelo_2.5_knn_msd.py)
- Línea 47: Ruta de guardado

#### [MODIFY] [modelo_3_wide&deep.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/models/jj/modelo_3_wide&deep.py)
- Líneas 57-59: Rutas de guardado (.pth, mappings)

#### [MODIFY] [modelo_4_bcs_tf-idf.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/models/jj/modelo_4_bcs_tf-idf.py)
- Líneas 42-44: Rutas de guardado (tfidf, matriz, indices)

#### [MODIFY] [modelo_5_implicit.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/models/jj/modelo_5_implicit.py)
- Líneas 39-41: Rutas de guardado (modelo, dataset, métricas)

#### [MODIFY] [modelo_6_ncf.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/models/jj/modelo_6_ncf.py)
- Líneas 53-55: Rutas de guardado (onnx, user2idx, item2idx)

#### [MODIFY] [modelo_7_twotowers.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/models/jj/modelo_7_twotowers.py)
- Líneas 21-22: Rutas de guardado (.pth, mappings)

---

### Componente 3: Mejora del pipeline de evaluación

#### [MODIFY] [evaluation_pipeline.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/pipelines/evaluation_pipeline.py)
- Añadir método `registrar_resultados_en_csv()` que llame a `registrar_metricas()` con las métricas de ranking calculadas
- Conectar el pipeline con el `historial_metricas.csv` para que todo quede centralizado

#### [NEW] [mrr.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/metrics/mrr.py)
- Nueva métrica MRR@K (Mean Reciprocal Rank) siguiendo el MetricProtocol existente

#### [MODIFY] [evaluacion_ranking.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/utils/evaluacion_ranking.py)
- Añadir métrica MRR@K a la lista de métricas
- Llamar a `registrar_resultados_en_csv()` tras evaluar cada modelo
- (Futuro) Integrar modelos tx/ en la evaluación

---

### Componente 4: Crear subdirectorio `mappings/` en artifacts

#### [NEW] `artifacts/mappings/__init__.py`
- Archivo vacío para marcar el directorio

---

## Open Questions

> [!IMPORTANT]
> 1. **Caché de tx/**: Los modelos de tx/ guardan artefactos en `src/data/cache/` (`.joblib`, `.npy`, `.parquet`). ¿Los movemos también a `artifacts/` o los dejamos como sistema de caché local?
> 2. **Mover archivos**: ¿Quieres que lance un script PowerShell que mueva los archivos ahora mismo, o prefieres moverlos tú manualmente?

## Verification Plan

### Automated Tests
- Verificar que todas las rutas nuevas existen tras el movimiento
- Comprobar que `main_api.py` importa los modelos correctamente (dry-run)

### Manual Verification
- Ejecutar `python -m src.api.main_api` y comprobar que los modelos se cargan
- Ejecutar un modelo cualquiera (ej: `modelo_1_SVD.py`) y comprobar que guarda en la nueva ruta
