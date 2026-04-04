# Plan de Implementación — Mejoras Proyecto 4 Streaming

## Objetivo
Aplicar todas las correcciones, mejoras y nuevas funcionalidades acordadas sobre el proyecto.

---

## Fase 1: Correcciones Rápidas y Limpieza

### 1.1 README.md
- Eliminar las 3 secciones duplicadas (mantener solo 1 copia de cada sección)
- Añadir diagrama de arquitectura del proyecto con Mermaid
- Limpiar credenciales hardcodeadas (referenciar a `.env.sample`)

### 1.2 Bugs a corregir
- **Bug 2**: Eliminar `set_page_config` de `2_Usuario.py` (solo queda en `app_ui.py`)
- **Bug 3**: Inicializar `top_tf = []` en `evaluacion_ranking.py` antes del bloque
- **Bug 4**: Renombrar cookie `"disney_admin_session"` → `"spire_admin_session"` en `1_Administrador.py`
- **Bug 5**: Cambiar `width="stretch"` → `use_container_width=True` en `2_Usuario.py`
- **Bug 6**: Cambiar `except:` → `except Exception:` en `etl.py`, `1_Administrador.py`

### 1.3 Código deprecado/mejorable
- Migrar `@app.on_event("startup")` a **lifespan handler** en `main_api.py`
- Mover import inline `from sklearn.metrics.pairwise import cosine_similarity` al top de `main_api.py`

### 1.4 Organización de archivos
- Mover `LoginRequest`/`RegisterRequest` de `main_api.py` → `src/schemas/schemas.py`
- Rellenar `src/config/rules_cleaning.yaml` con reglas ETL reales
- Mover `server_error.log` a carpeta `logs/` y configurar logging del backend

### 1.5 Consistencia visual
- Añadir CSS premium (Azul Marino + Oro) a `1_Administrador.py` para igualar el estilo de `2_Usuario.py`

---

## Fase 2: Sistema de Registro CSV de Métricas

### 2.1 Crear `src/utils/registrar_metricas.py`
- Función `registrar_metricas(modelo, hiperparams, metricas)` que appenda al CSV
- Columnas: timestamp, modelo, hiperparámetros variables, MAE, RMSE, NDCG_10, Precision_10, Recall_10, Hit_Rate_10, Coverage_10, train_time_s, dataset_size
- Donde una métrica no aplique → "NA"

### 2.2 Integrar en cada script de modelo
- `modelo_1_SVD.py`, `modelo_2_knn+cs.py`, `modelo_3_wide&deep.py`, `modelo_4_bcs_tf-idf.py`, `modelo_5_implicit.py`
- Al finalizar entrenamiento → llamar a `registrar_metricas()`

### 2.3 Integrar en `evaluacion_ranking.py`
- Al guardar `metricas_ranking.csv`, también registrar cada modelo en el historial

---

## Fase 3: Re-entrenar Wide & Deep

### 3.1 Ajustar hiperparámetros
- Subir `MIN_RATINGS_PELICULA` de 5 → 50 (eliminar ruido de embeddings no entrenados)
- Aumentar épocas a 15-20
- Evaluar si el `sigmoid * 5.5` perjudica el ranking

### 3.2 Re-entrenar y registrar métricas

---

## Fase 4: NCF (Modelo 6)

### 4.1 Crear `src/models/jj/modelo_6_ncf.py`
- Arquitectura NCF-Lite (GMF + MLP) basada en el trabajo de Nil pero adaptada
- Entrenamiento con BCE pairwise + negative sampling
- Exportar a ONNX (`modelo_6_ncf.onnx`) + mappings JSON

### 4.2 Integrar en Backend
- Añadir endpoint `/recomendar/ncf/{user_id}` en `main_api.py`
- Cargar modelo ONNX en startup

### 4.3 Integrar en Frontend
- Añadir opción "NCF (Neural CF)" en el selector de modelos de `2_Usuario.py`

### 4.4 Integrar en Evaluador
- Añadir NCF al `evaluacion_ranking.py`

---

## Fase 5: Two Towers (Siguiente sesión)
- Arquitectura retrieval de dos torres (User Tower + Item Tower)
- Exportar a ONNX como `modelo_7_twotowers.onnx`
- Integrar en backend, frontend y evaluador
- *(Se hará en otra sesión)*

---

## Sobre el volcado de ratings a SQL (tu pregunta)

> [!TIP]
> **Recomendación para integrar nuevos usuarios sin pisar los ratings del modelo:**
> 
> 1. Crear tabla `user_ratings` en MySQL con (`user_id`, `tmdb_id`, `rating`, `timestamp`)
> 2. Los ratings del CSV de MovieLens se cargan como bloque inicial (INSERT IGNORE)
> 3. Cuando un usuario nuevo vota en la interfaz, se inserta en la misma tabla
> 4. Para re-entrenar modelos: `SELECT * FROM user_ratings` exporta un CSV fusionado
> 5. **Clave**: Usar el `id_usuario` de MySQL como `userId` de los modelos, con un offset (ej: IDs de MovieLens 1-283.000, tus usuarios nuevos desde 1.000.000) para que no colisionen

---

## Verificación
- Backend arranca sin errores con los modelos cargados
- Frontend carga ambas páginas con CSS consistente
- Evaluador ejecuta con todos los modelos (incluido NCF)
- CSV de historial se genera correctamente
