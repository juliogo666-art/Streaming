# Walkthrough: Fase 4 y 5 — Hub de Modelos, Recomendación Dinámica y Control de Acceso

## Resumen de Cambios

### 1. Repositorio Remoto de Modelos (HuggingFace)

#### [upload_models.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/scripts/upload_models.py) — REESCRITO
- Escanea automáticamente las 4 subcarpetas de `artifacts/` (`weights`, `exports`, `checkpoints`, `mappings`)
- Mantiene la estructura de carpetas en el repositorio HuggingFace
- Carga el token desde `.env` (variable `HF_TOKEN`)
- Muestra resumen previo con número de archivos y tamaño total
- Barra de progreso `[1/N]` por cada archivo subido

#### [download_models.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/utils/download_models.py) — NUEVO
- `obtener_archivos_faltantes()`: Comprueba qué modelos faltan localmente
- `verificar_y_descargar(callback)`: Descarga solo los que faltan desde HuggingFace (repo público, sin token)
- Acepta un `callback_progreso` para integrar con la UI de Streamlit
- Usable como módulo importable y como script directo: `python -m src.utils.download_models`

#### [main.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/main.py) — MODIFICADO
- Nueva función `verificar_modelos()` que se ejecuta **antes** de lanzar el backend y frontend
- Si faltan archivos, los descarga automáticamente antes de iniciar uvicorn/streamlit

---

### 2. Sistema de Recomendación Dinámico

#### [main_api.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/api/main_api.py) — NUEVO ENDPOINT `/recomendar/smart/{user_id}`

Umbrales basados en el análisis de `metricas_ranking.csv`:

| Rango valoraciones | Modelo seleccionado | NDCG@10 |
|---|---|---|
| 0-10 (Cold Start) | Content-Based (TF-IDF) / Popularidad | 0.022 |
| 11-99 (Intermedio) | SVD → KNN → BPR (cascada fallback) | ~0.32 |
| 100+ (Experto) | NCF-Lite → Wide&Deep → SVD (cascada) | 0.779 |

El endpoint devuelve un campo `selector` con el razonamiento: ej. `"Smart → NCF-Lite (Experto: 342 valoraciones)"`

#### [recommendation.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/schemas/recommendation.py) — MODIFICADO
- Añadido campo `selector: Optional[str]` al schema de respuesta

#### [2_Usuario.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/frontend/pages/2_Usuario.py) — MODIFICADO
- Nueva opción `"⚡ Smart (Auto)"` como primera y por defecto en el selector de modelo
- Muestra `🧠 Smart → ...` debajo de las recomendaciones para indicar qué modelo se usó

---

### 3. Control de Acceso por Roles

#### [migration_user_v3.sql](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/data/scripts_sql/migration_user_v3.sql) — NUEVO
- Añade columna `role ENUM('admin', 'user') DEFAULT 'user'` a la tabla `users`
- Asigna `role = 'admin'` al usuario `root`

#### [main_api.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/api/main_api.py) — MODIFICADO
- El query de `/login` ahora incluye `role` en el SELECT

#### [1_Administrador.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/frontend/pages/1_Administrador.py) — MODIFICADO
- ❌ Eliminado el bypass `st.session_state["admin_autenticado"] = True`
- ✅ Tras el login, verifica que `role == 'admin'` antes de dar acceso
- Corregido nombre de cookie: `disney_admin_session` → `spire_admin_session`

---

### 4. Mejoras Generales

- [.env.sample](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/.env.sample): Añadido `HF_TOKEN`
- [.gitignore](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/.gitignore): Añadido `*.onnx.data`, actualizado comentario

## Siguiente Paso

> [!IMPORTANT]
> **Para completar el punto 5**, necesitas ejecutar el script de subida para poblar tu repositorio HuggingFace:
> ```
> python -m src.scripts.upload_models
> ```
> Esto subirá ~3.5 GB de modelos a `JulioJ777/Streaming`. Una vez subidos, cualquier persona que clone el repo de GitHub y ejecute `main.py` descargará los modelos automáticamente.
