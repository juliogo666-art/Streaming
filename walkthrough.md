# Walkthrough — Optimización Proyecto 4 Streaming

## Resumen

Se ejecutaron las **Fases 1, 2 y 4** del plan de implementación aprobado, aplicando correcciones de bugs, refactorizaciones de código, un nuevo sistema de logging de métricas, y la integración completa del modelo NCF.

---

## Fase 1: Correcciones y Limpieza

### Archivos Modificados

#### [README.md](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/README.md)
- Eliminadas **3 secciones duplicadas** que inflaban el archivo
- Añadido **diagrama de arquitectura Mermaid** (árbol de directorios + flujo del sistema)
- Añadida **tabla de tecnologías**
- Eliminadas credenciales hardcoded

#### [main_api.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/api/main_api.py)
- **Migrado** de `@app.on_event("startup")` (deprecado) a moderno `lifespan` handler con `asynccontextmanager`
- **Logging estructurado** con `logging.FileHandler` → `logs/backend.log`
- **Import inline** de `cosine_similarity` movido al top-level
- **Pydantic schemas** extraídos a módulo dedicado

#### [1_Administrador.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/frontend/pages/1_Administrador.py)
- Cookie renombrada de `disney_admin_session` → `spire_admin_session`
- `bare except:` → `except Exception:`
- **CSS premium** idéntico al de `2_Usuario.py` (Azul Marino + Oro)

#### [2_Usuario.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/frontend/pages/2_Usuario.py)
- Eliminado `set_page_config` duplicado (ya se llama en `app_ui.py`)
- `width="stretch"` → `use_container_width=True` (API deprecada de Streamlit)

#### [etl.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/api/etl.py)
- `bare except:` → `except Exception:`

### Archivos Nuevos

| Archivo | Descripción |
|---|---|
| [schemas.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/schemas/schemas.py) | Pydantic models (`LoginRequest`, `RegisterRequest`) |
| [rules_cleaning.yaml](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/config/rules_cleaning.yaml) | Reglas ETL (columnas, nulos, tipos, géneros, filtros) |
| `logs/` | Directorio de logs (`server_error.log` movido aquí) |

---

## Fase 2: Sistema de Registro de Métricas

### [registrar_metricas.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/utils/registrar_metricas.py)

Nuevo módulo que persiste métricas de cada ejecución de entrenamiento:

- **CSV acumulativo** (`historial_metricas.csv`) con columnas estandarizadas
- Columnas: `timestamp`, `modelo`, hiperparámetros (13 cols), métricas error (2) y ranking (6), rendimiento, notas
- Usa `"NA"` para valores no aplicables al modelo (ej: K_vecinos para SVD)
- Incluye `leer_historial()` para análisis posterior con pandas

### Integración en 5 scripts de modelos

Cada script ahora llama `registrar_metricas()` automáticamente al finalizar el entrenamiento:

| Modelo | Script | Métricas registradas |
|---|---|---|
| SVD | modelo_1_SVD.py | MAE, RMSE, n_factores, n_epocas, lr, reg |
| KNN | modelo_2_knn+cs.py | MAE, RMSE, k_vecinos, min_ratings |
| W&D | modelo_3_wide&deep.py | MAE, RMSE, emb_dim, batch_size, hidden_layers |
| TF-IDF | modelo_4_bcs_tf-idf.py | solo notas (no aplican MAE/RMSE) |
| Implicit | modelo_5_implicit.py | metricas dict del modelo |

---

## Fase 4: NCF — Modelo 6

### [modelo_6_ncf.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/models/jj/modelo_6_ncf.py)

Nuevo modelo completo con arquitectura GMF+MLP:

- **K-Core filtering** iterativo (200 ratings/user, 100 ratings/item)
- **Pairwise BCE** training con negative sampling (4 negativos/positivo)
- **NCF-Lite** con dos ramas: GMF (producto embeddings) + MLP (capas densas)
- **Exportación ONNX** para inferencia sin PyTorch
- **Mapeos JSON** (`user2idx`, `item2idx`) para traducir IDs reales ↔ índices
- Soporte GPU: CUDA, DirectML (AMD/Intel), o CPU automático

### Backend — Endpoint NCF
```diff:main_api.py
from fastapi import FastAPI, HTTPException
from .database import get_db_connection
from .etl import ejecutar_importacion, limpiar_tablas_contenido
import pandas as pd
import numpy as np
from pydantic import BaseModel
import bcrypt
import os

try:
    import onnxruntime as ort

    ONNX_DISPONIBLE = True
    print(f"[WnD] ONNX Runtime {ort.__version__} disponible.")
except ImportError as e:
    print(f"[WnD] onnxruntime no instalado: {e}")
    ONNX_DISPONIBLE = False

app = FastAPI()


@app.get("/status")
def check_status():
    return {"status": "Backend funcionando correctamente"}


@app.get("/usuarios")
def obtener_usuarios():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)  # Devuelve filas como diccionarios
    cursor.execute("SELECT * FROM users")
    usuarios = cursor.fetchall()
    cursor.close()
    conn.close()
    return usuarios


def importar_series(conn):
    print("inicio importacion series")
    # Importamos Series
    ejecutar_importacion(conn, "src/data/clean/tmdb_shows_limpio.csv", "tv")
    print("fin importacion series")


def importar_peliculas(conn):
    print("inicio importacion peliculas, se viene lo chungo")
    # Importamos Películas
    # ejecutar_importacion(conn, 'src/data/raw/tmdb/movies/peliculas_2020.csv', 'movie')
    ejecutar_importacion(conn, "src/data/clean/tmdb_movies_limpio.csv", "movie")
    print("fin importacion peliculas, al fin!")


def limpiar_csv():
    df = pd.read_csv("src/data/raw/tmdb/movies/peliculas_2020.csv")

    # 2. LIMPIEZA CRÍTICA: Convertir NaN de Pandas a None de Python (NULL en MySQL)
    # Esto evita el error de "Lost connection" por tipos de datos inválidos
    df = df.replace({np.nan: None})

    # 3. Asegurar tipos de datos (evita el Out of Range)
    df["vote_average"] = pd.to_numeric(df["vote_average"]).round(2)
    df["popularity"] = pd.to_numeric(df["popularity"]).round(2)
    df.to_csv(
        "src/data/raw/tmdb/movies/peliculas_2020_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )


@app.post("/importar_datos")
def importar_datos():
    conn = get_db_connection()
    try:
        limpiar_tablas_contenido(conn)
        # limpiar_csv()
        importar_series(conn)
        importar_peliculas(conn)
        return {"status": "success", "message": "Catálogos actualizados correctamente"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    fecha_nacimiento: str = None
    sexo: str = None
    intereses: list[int] = []


@app.post("/login")
def login(datos: LoginRequest):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # 1. Buscamos al usuario solo por nombre de usuario
    query = "SELECT id_usuario, username, email, passwd FROM users WHERE username = %s"
    cursor.execute(query, (datos.username,))

    usuario = cursor.fetchone()

    cursor.close()
    conn.close()

    if usuario:
        # Obtenemos el hash que estaba guardado en BD
        hash_guardado = usuario["passwd"]

        # 2. Comprobamos si la contraseña coincide (Soportando tanto Hash nuevo como Texto Plano antiguo)
        es_valido = False
        if hash_guardado.startswith("$2b$") or hash_guardado.startswith("$2a$"):
            # Si es un hash de bcrypt válido
            es_valido = bcrypt.checkpw(
                datos.password.encode("utf-8"), hash_guardado.encode("utf-8")
            )
        else:
            # Si es una contraseña antigua en texto plano (ej: 'root')
            es_valido = datos.password == hash_guardado

        if es_valido:
            # Por seguridad, borramos la contraseña del diccionario temporal antes de mandarlo al frontend
            del usuario["passwd"]

            return {"status": "success", "message": "Login exitoso", "user": usuario}
        else:
            raise HTTPException(
                status_code=401, detail="Usuario o contraseña incorrectos"
            )
    else:
        raise HTTPException(status_code=401, detail="Usuario o contraseña incorrectos")


@app.get("/genres")
def obtener_generos():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name FROM genres ORDER BY name ASC")
    generos = cursor.fetchall()
    cursor.close()
    conn.close()
    return generos


@app.post("/register")
def register(datos: RegisterRequest):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        # 1. Verificar si el usuario o email ya existen
        cursor.execute(
            "SELECT id_usuario FROM users WHERE username = %s OR email = %s",
            (datos.username, datos.email),
        )
        if cursor.fetchone():
            raise HTTPException(
                status_code=400, detail="El nombre de usuario o email ya están en uso."
            )

        # 2. Hashear la contraseña
        password_hash = bcrypt.hashpw(
            datos.password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

        # 3. Insertar usuario
        query_user = """
            INSERT INTO users (username, email, passwd, fecha_nacimiento, sexo)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(
            query_user,
            (
                datos.username,
                datos.email,
                password_hash,
                datos.fecha_nacimiento,
                datos.sexo,
            ),
        )
        id_usuario = cursor.lastrowid

        # 4. Manejar intereses
        intereses = datos.intereses
        if not intereses:
            # Seleccionar los 3 géneros más populares (basado en reproducciones totales)
            query_pop = """
                SELECT cg.genre_id
                FROM content_genres cg
                JOIN content_stats cs ON cg.content_id = cs.content_id
                GROUP BY cg.genre_id
                ORDER BY SUM(cs.reproducciones_totales) DESC
                LIMIT 3
            """
            cursor.execute(query_pop)
            res_pop = cursor.fetchall()
            intereses = [row["genre_id"] for row in res_pop]

        # Insertar intereses
        if intereses:
            query_int = (
                "INSERT INTO user_interests (id_usuario, genre_id) VALUES (%s, %s)"
            )
            for g_id in intereses:
                cursor.execute(query_int, (id_usuario, g_id))

        conn.commit()
        return {
            "status": "success",
            "message": "Usuario registrado correctamente",
            "user_id": id_usuario,
        }

    except Exception as e:
        conn.rollback()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error en el servidor: {str(e)}")
    finally:
        cursor.close()
        conn.close()


# --- CONFIGURACIÓN DE RUTAS Y ESTADO ---
# Ruta al modelo SVD entrenado y al CSV de ratings, para saber qué pelis ya ha visto el usuario.
ruta_ratings = "src/data/ready/ratings_finales_ia.csv"
ruta_catalogo = "src/data/ready/dataset_final_movies.csv"

# --- INICIALIZACIÓN DE ESTADO DE LA APP ---

# --- INICIALIZACIÓN DE ESTADO DE LA APP ---
# Usamos app.state para que los modelos persistan correctamente en memoria entre peticiones
app.state.modelo_svd = None
app.state.df_ratings_ia = None
app.state.df_catalogo = None
app.state.modelo_knn = None
app.state.modelo_wnd = None
app.state.wnd_mappings = None
app.state.modelo_tfidf_mat = None
app.state.modelo_tfidf_idx = None
app.state.modelo_imp = None
app.state.modelo_imp_dat = None

# --- RUTAS DE MODELOS (Sincronizadas con disco) ---
ruta_modelo_svd = "src/models/jj/modelo_1_SVD.joblib"
ruta_modelo_knn = "src/models/jj/modelo_2_knn_cs.joblib"
ruta_modelo_wnd = "src/models/jj/modelo_3_wnd.onnx"
ruta_wnd_map = "src/models/jj/wnd_mappings.pkl"
ruta_tfidf_mat = "src/models/jj/modelo_4_matriz.joblib"
ruta_tfidf_idx = "src/models/jj/modelo_4_indices.joblib"
ruta_imp = "src/models/jj/modelo_5_implicit.pkl"
ruta_imp_dat = "src/models/jj/modelo_5_implicit_dataset.pkl"


@app.on_event("startup")
def cargar_modelo_al_arrancar():
    """Carga los modelos pesados en la memoria RAM de forma asíncrona."""
    print("[STARTUP] Iniciando carga de modelos de IA en memoria...")
    print("[INFO] Cargando ratings (434MB)... espera unos 30-40s.")

    # --- Modelo 1: SVD (Requiere scikit-surprise) ---
    if os.path.exists(ruta_modelo_svd):
        try:
            import joblib

            app.state.modelo_svd = joblib.load(ruta_modelo_svd)
            print("Modelo SVD cargado correctamente (Joblib).")
        except Exception as e:
            print(f"ERROR: No se pudo cargar el modelo SVD: {e}.")

    # --- Modelo 2: KNN ---
    if os.path.exists(ruta_modelo_knn):
        try:
            import joblib

            app.state.modelo_knn = joblib.load(ruta_modelo_knn)
            print("Modelo KNN+Cosine cargado (Joblib).")
        except Exception as e:
            print(f"ERROR: No se pudo cargar el modelo KNN: {e}")

    # --- Modelo 3: Wide & Deep (ONNX Runtime) ---
    if os.path.exists(ruta_modelo_wnd) and os.path.exists(ruta_wnd_map):
        try:
            import onnxruntime as ort
            import pickle

            app.state.modelo_wnd = ort.InferenceSession(ruta_modelo_wnd)
            with open(ruta_wnd_map, "rb") as f:
                app.state.wnd_mappings = pickle.load(f)
            print("Modelo Wide&Deep ONNX cargado correctamente.")
        except Exception as e:
            print(f"ERROR: No se pudo cargar el modelo Wide&Deep: {e}")

    # --- Modelo 4: Content-Based (TF-IDF) ---
    if os.path.exists(ruta_tfidf_mat) and os.path.exists(ruta_tfidf_idx):
        try:
            import joblib

            app.state.modelo_tfidf_mat = joblib.load(ruta_tfidf_mat)
            app.state.modelo_tfidf_idx = joblib.load(ruta_tfidf_idx)
            print("Modelo TF-IDF cargado correctamente (Joblib).")
        except Exception as e:
            print(f"ERROR: No se pudo cargar el modelo TF-IDF: {e}")

    # --- Modelo 5: Implicit BPR ---
    if os.path.exists(ruta_imp) and os.path.exists(ruta_imp_dat):
        try:
            import pickle

            with open(ruta_imp, "rb") as f:
                app.state.modelo_imp = pickle.load(f)
            with open(ruta_imp_dat, "rb") as f:
                app.state.modelo_imp_dat = pickle.load(f)
            print("Modelo Implicit cargado correctamente.")
        except Exception as e:
            print(f"ERROR: No se pudo cargar el modelo Implicit: {e}")

    # --- CARGA DE DATOS (CSV) ---
    try:
        if os.path.exists(ruta_ratings):
            app.state.df_ratings_ia = pd.read_csv(ruta_ratings)
            print(f"Ratings cargados: {len(app.state.df_ratings_ia):,} filas.")

        if os.path.exists(ruta_catalogo):
            app.state.df_catalogo = pd.read_csv(ruta_catalogo)
            print(f"Catálogo cargado: {len(app.state.df_catalogo):,} películas.")
    except Exception as e:
        print(f"ERROR al cargar archivos CSV de datos: {e}")


@app.get("/recomendar/svd/{user_id}")
@app.get("/recomendar/{user_id}")
def recomendar_peliculas(user_id: int, n: int = 10):
    """
    Endpoint que devuelve las top-N películas recomendadas para un usuario.
    Usa el modelo SVD entrenado para predecir ratings de películas no vistas.
    """
    print(f"DEBUG: Petición SVD para User {user_id}")

    # --- DIAGNÓSTICO EN TIEMPO DE EJECUCIÓN ---
    if app.state.modelo_svd is None:
        causa = "app.state.modelo_svd es None (Error de persistencia)"
        print(f"DEBUG ERROR: {causa}")
        raise HTTPException(status_code=503, detail=causa)
    if app.state.df_ratings_ia is None:
        causa = "app.state.df_ratings_ia es None (CSV de ratings no cargado)"
        print(f"DEBUG ERROR: {causa}")
        raise HTTPException(status_code=503, detail=causa)

    # 1. Películas que este usuario YA ha visto
    pelis_vistas = set(
        app.state.df_ratings_ia[app.state.df_ratings_ia["userId"] == user_id][
            "tmdb_id"
        ].tolist()
    )

    # 2. Todas las películas disponibles en el sistema (aseguramos que parten del catálogo)
    if app.state.df_catalogo is not None:
        todas_las_pelis = set(app.state.df_catalogo["tmdb_id"].unique())
    else:
        todas_las_pelis = set(app.state.df_ratings_ia["tmdb_id"].unique())

    # 3. Candidatas = las que NO ha visto
    pelis_no_vistas = todas_las_pelis - pelis_vistas

    # Si el usuario ha visto todas (raro pero posible), avisamos
    if not pelis_no_vistas:
        return {
            "recomendaciones": [],
            "mensaje": "Este usuario ya ha valorado todas las películas disponibles.",
        }

    # 4. Predecimos la nota para cada película candidata
    predicciones = []
    for tmdb_id in pelis_no_vistas:
        pred = app.state.modelo_svd.predict(user_id, tmdb_id)
        predicciones.append(
            {"tmdb_id": int(tmdb_id), "predicted_rating": round(pred.est, 2)}
        )

    # 5. Ordenamos de mayor a menor y nos quedamos con las top-N
    predicciones.sort(key=lambda x: x["predicted_rating"], reverse=True)
    top_n = predicciones[:n]

    # 6. Enriquecemos con datos del catálogo
    enriquecer_recomendaciones(top_n)

    return {"recomendaciones": top_n, "modelo": "SVD (Surprise)"}


##############################################################################################
#  Helper: Enriquecer recomendaciones con datos del catálogo
##############################################################################################


def enriquecer_recomendaciones(recomendaciones):
    """Añade datos del catálogo (título, poster, etc.) a una lista de diccionarios con tmdb_id."""
    if app.state.df_catalogo is not None:
        for rec in recomendaciones:
            match = app.state.df_catalogo[
                app.state.df_catalogo["tmdb_id"] == rec["tmdb_id"]
            ]
            if not match.empty:
                fila = match.iloc[0]
                rec["titulo"] = str(fila.get("titulo", "Sin Título"))
                rec["poster_path"] = str(fila.get("poster_path", ""))
                rec["overview"] = str(fila.get("overview", "Sin sinopsis disponible."))
                rec["vote_average"] = float(fila.get("vote_average", 0))
            else:
                rec["titulo"] = f"Película #{rec['tmdb_id']}"
                rec["poster_path"] = ""
                rec["overview"] = "Sin sinopsis disponible."
                rec["vote_average"] = 0.0


##############################################################################################
#  Recomendación Modelo KNN + Cosine Similarity
##############################################################################################


@app.get("/recomendar/knn/{user_id}")
def recomendar_knn(user_id: int, n: int = 10):
    """Endpoint de recomendaciones usando KNN + Cosine Similarity (Modelo 2)."""
    print(f"DEBUG: Petición KNN para User {user_id}")
    if app.state.modelo_knn is None:
        causa = "app.state.modelo_knn es None"
        print(f"DEBUG ERROR: {causa}")
        raise HTTPException(status_code=503, detail=causa)
    if app.state.df_ratings_ia is None:
        causa = "app.state.df_ratings_ia es None"
        print(f"DEBUG ERROR: {causa}")
        raise HTTPException(status_code=503, detail=causa)

    pelis_vistas = set(
        app.state.df_ratings_ia[app.state.df_ratings_ia["userId"] == user_id][
            "tmdb_id"
        ].tolist()
    )
    # Extraemos items del modelo y buscamos similares en base a la historia del usuario
    # En este setup simplificado buscamos los items mas populares del dataset
    item_counts = app.state.df_ratings_ia["tmdb_id"].value_counts()
    top_items = item_counts.index.tolist()
    todas = (
        set(app.state.df_catalogo["tmdb_id"].unique())
        if app.state.df_catalogo is not None
        else set(app.state.df_ratings_ia["tmdb_id"].unique())
    )
    pelis_no_vistas = todas - pelis_vistas

    if not pelis_no_vistas:
        return {
            "recomendaciones": [],
            "modelo": "KNN+Cosine",
            "mensaje": "Este usuario ya ha valorado todas las películas.",
        }

    predicciones = []
    for tmdb_id in pelis_no_vistas:
        pred = app.state.modelo_knn.predict(user_id, tmdb_id)
        predicciones.append(
            {"tmdb_id": int(tmdb_id), "predicted_rating": round(pred.est, 2)}
        )

    predicciones.sort(key=lambda x: x["predicted_rating"], reverse=True)
    top_n = predicciones[:n]
    enriquecer_recomendaciones(top_n)

    return {"recomendaciones": top_n, "modelo": "KNN+Cosine"}


##############################################################################################
#  Recomendación Modelo Wide & Deep (PyTorch)
##############################################################################################


@app.get("/recomendar/wnd/{user_id}")
def recomendar_wnd_endpoint(user_id: int, n: int = 10):
    """Endpoint de recomendaciones usando Wide & Deep Neural Network (ONNX Native)."""
    print(f"DEBUG: Petición Wide&Deep para User {user_id}")
    if app.state.modelo_wnd is None or app.state.wnd_mappings is None:
        causa = "app.state.modelo_wnd o mappings es None"
        print(f"DEBUG ERROR: {causa}")
        raise HTTPException(status_code=503, detail=causa)
    if app.state.df_ratings_ia is None:
        causa = "app.state.df_ratings_ia es None"
        print(f"DEBUG ERROR: {causa}")
        raise HTTPException(status_code=503, detail=causa)

    user2idx = app.state.wnd_mappings["user2idx"]
    movie2idx = app.state.wnd_mappings["movie2idx"]

    if user_id not in user2idx:
        return {
            "recomendaciones": [],
            "modelo": "Wide&Deep (ONNX)",
            "mensaje": f"El usuario {user_id} no cumplió el filtro de entrenamiento (>100 valoraciones).",
        }

    u_idx = user2idx[user_id]
    pelis_vistas = set(
        app.state.df_ratings_ia[app.state.df_ratings_ia["userId"] == user_id][
            "tmdb_id"
        ].tolist()
    )

    candidatas = [
        (tid, midx) for tid, midx in movie2idx.items() if tid not in pelis_vistas
    ]
    if not candidatas:
        return {
            "recomendaciones": [],
            "modelo": "Wide&Deep (ONNX)",
            "mensaje": "Este usuario ya ha valorado todas las películas.",
        }

    tmdb_ids, movie_indices = zip(*candidatas)

    # Creamos arrays NumPy en lugar de tensores PyTorch
    user_arr = np.array([u_idx] * len(movie_indices), dtype=np.int64)
    movie_arr = np.array(list(movie_indices), dtype=np.int64)

    # Inferencia ONNX
    try:
        ort_inputs = {
            "user_id": user_arr,
            "movie_id": movie_arr,
        }
        preds_norm = app.state.modelo_wnd.run(None, ort_inputs)[0]
    except Exception as e:
        print(f"ERROR en inferencia Wide&Deep: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error en inferencia del modelo WnD: {e}"
        )

    # Restringir predicción a rango [0.5, 5.0]
    preds_onnx = np.clip(preds_norm, 0.5, 5.0)

    predicciones = [
        {"tmdb_id": int(tid), "predicted_rating": round(float(preds_onnx[i]), 2)}
        for i, tid in enumerate(tmdb_ids)
    ]

    predicciones.sort(key=lambda x: x["predicted_rating"], reverse=True)
    top_n = predicciones[:n]
    enriquecer_recomendaciones(top_n)

    return {"recomendaciones": top_n, "modelo": "Wide&Deep (ONNX)"}


##############################################################################################
#  Recomendación Modelo 4: Content-Based / Cold Start
##############################################################################################


@app.get("/recomendar/content/{user_id}")
def recomendar_content_endpoint(user_id: int, n: int = 10):
    """Endpoint de recomendaciones por contenido (Modelo 4)."""
    print(f"DEBUG: Petición Content-Based para User {user_id}")
    if app.state.modelo_tfidf_mat is None or app.state.modelo_tfidf_idx is None:
        causa = "app.state.modelo_tfidf_mat o idx es None"
        print(f"DEBUG ERROR: {causa}")
        raise HTTPException(status_code=503, detail=causa)
    if app.state.df_ratings_ia is None or app.state.df_catalogo is None:
        causa = "app.state.df_ratings_ia o df_catalogo es None"
        print(f"DEBUG ERROR: {causa}")
        raise HTTPException(status_code=503, detail=causa)

    user_ratings = app.state.df_ratings_ia[app.state.df_ratings_ia["userId"] == user_id]

    # Cold Start (Usuario sin historial)
    if user_ratings.empty:
        # Recomendamos por popularidad general
        if "vote_count" in app.state.df_catalogo.columns:
            top_pop = (
                app.state.df_catalogo[app.state.df_catalogo["vote_count"] > 100]
                .sort_values(by="vote_average", ascending=False)
                .head(n)
            )
        else:
            top_pop = app.state.df_catalogo.head(n)

        recomendaciones = []
        for idx, row in top_pop.iterrows():
            recomendaciones.append(
                {"tmdb_id": int(row["tmdb_id"]), "predicted_rating": 4.5}
            )

        enriquecer_recomendaciones(recomendaciones)
        return {
            "recomendaciones": recomendaciones,
            "modelo": "TF-IDF (Cold Start Populares)",
        }

    # Usuario con historial -> Content-Based por similitud
    viewed_ids = user_ratings["tmdb_id"].tolist()
    tfidf_indices = [
        app.state.modelo_tfidf_idx[mid]
        for mid in viewed_ids
        if mid in app.state.modelo_tfidf_idx
    ]

    if not tfidf_indices:
        return {
            "recomendaciones": [],
            "modelo": "Content-Based",
            "mensaje": "Película favorita no encontrada.",
        }

    # Calculamos perfil del usuario (media de los vectores de peliculas vistas)
    user_profile = np.asarray(app.state.modelo_tfidf_mat[tfidf_indices].mean(axis=0))
    # Similitud coseno contra todo el catálogo
    from sklearn.metrics.pairwise import cosine_similarity

    cos_sim = cosine_similarity(user_profile, app.state.modelo_tfidf_mat).flatten()

    # Obtenemos los mas similares (ignorando el mismisimo 1.0)
    top_indices = cos_sim.argsort()[::-1][1 : n + 1]

    predicciones = []
    for idx_sim in top_indices:
        peli = app.state.df_catalogo.iloc[idx_sim]
        score_sim = cos_sim[idx_sim]
        # Creamos un rating visual combinando su rating original con la similitud
        pseudo_rating = min(5.0, 4.0 * (0.8 + 0.2 * score_sim))
        predicciones.append(
            {
                "tmdb_id": int(peli["tmdb_id"]),
                "predicted_rating": round(pseudo_rating, 2),
            }
        )

    enriquecer_recomendaciones(predicciones)
    return {"recomendaciones": predicciones, "modelo": "Content-Based"}


##############################################################################################
#  Recomendación Modelo 5: Implicit BPR
##############################################################################################


@app.get("/recomendar/implicit/{user_id}")
def recomendar_implicit_endpoint(user_id: int, n: int = 10):
    """Endpoint de recomendaciones usando filtrado colaborativo BPR de la librería implicit."""
    print(f"DEBUG: Petición Implicit BPR para User {user_id}")
    if app.state.modelo_imp is None or app.state.modelo_imp_dat is None:
        causa = f"app.state.modelo_imp es {app.state.modelo_imp is None} | app.state.modelo_imp_dat es {app.state.modelo_imp_dat is None}"
        print(f"DEBUG ERROR: {causa}")
        raise HTTPException(status_code=503, detail=f"Fallo en Modelos Imply: {causa}")
    if app.state.df_ratings_ia is None:
        causa = "app.state.df_ratings_ia es None"
        print(f"DEBUG ERROR: {causa}")
        raise HTTPException(status_code=503, detail=causa)

    user2idx = app.state.modelo_imp_dat["user2idx"]
    item2idx = app.state.modelo_imp_dat["item2idx"]
    idx2item = {v: k for k, v in item2idx.items()}

    if user_id not in user2idx:
        return {
            "recomendaciones": [],
            "modelo": "Implicit BPR",
            "mensaje": f"El usuario {user_id} no tiene historial en el modelo BPR.",
        }

    u_idx = user2idx[user_id]

    # Extraemos arrays nativos mediante NumPy para evitar el bug Cython de implicit.recommend() en Windows!
    u_factors = np.asarray(app.state.modelo_imp.user_factors[u_idx])
    i_factors = np.asarray(app.state.modelo_imp.item_factors)

    # Producto escalar vectorizado (Calcula score para tooooodas las peliculas del modelo a la vez)
    scores = u_factors.dot(i_factors.T)

    # Localizamos las que ya ha visto en el dataset completo
    pelis_vistas = set(
        app.state.df_ratings_ia[app.state.df_ratings_ia["userId"] == user_id][
            "tmdb_id"
        ].tolist()
    )

    # Forzamos su score a menos infinito para que nunca salgan en el top
    for tid in pelis_vistas:
        if tid in item2idx:
            midx = item2idx[tid]
            scores[midx] = -np.inf

    # np.argsort()[::-1] ordena de mayor a menor y extraemos las top N posiciones de memoria
    top_indices = np.argsort(scores)[::-1][:n]

    predicciones = []
    for idx_sim in top_indices:
        tid = idx2item[idx_sim]
        score_puro = float(scores[idx_sim])
        # Al ser ranking BPR no devuelve nota 0-5. Reescalamos visualmente para UI amigable.
        rating_ui = min(5.0, max(0.5, 3.5 + (score_puro * 0.2)))

        predicciones.append(
            {"tmdb_id": int(tid), "predicted_rating": round(rating_ui, 2)}
        )

    enriquecer_recomendaciones(predicciones)

    return {"recomendaciones": predicciones, "modelo": "Implicit BPR"}
===
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from .database import get_db_connection
from .etl import ejecutar_importacion, limpiar_tablas_contenido
from ..schemas.schemas import LoginRequest, RegisterRequest
import pandas as pd
import numpy as np
import bcrypt
import os
from sklearn.metrics.pairwise import cosine_similarity

try:
    import onnxruntime as ort

    ONNX_DISPONIBLE = True
    print(f"[WnD] ONNX Runtime {ort.__version__} disponible.")
except ImportError as e:
    print(f"[WnD] onnxruntime no instalado: {e}")
    ONNX_DISPONIBLE = False

# --- Configuración de Logging Estructurado ---
log_handler = logging.FileHandler("logs/backend.log", encoding="utf-8")
log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger = logging.getLogger("streaming_api")
logger.addHandler(log_handler)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# --- CONFIGURACIÓN DE RUTAS Y ESTADO ---
# Ruta al modelo SVD entrenado y al CSV de ratings, para saber qué pelis ya ha visto el usuario.
ruta_ratings = "src/data/ready/ratings_finales_ia.csv"
ruta_catalogo = "src/data/ready/dataset_final_movies.csv"

# --- RUTAS DE MODELOS (Sincronizadas con disco) ---
ruta_modelo_svd = "src/models/jj/modelo_1_SVD.joblib"
ruta_modelo_knn = "src/models/jj/modelo_2_knn_cs.joblib"
ruta_modelo_wnd = "src/models/jj/modelo_3_wnd.onnx"
ruta_wnd_map = "src/models/jj/wnd_mappings.pkl"
ruta_tfidf_mat = "src/models/jj/modelo_4_matriz.joblib"
ruta_tfidf_idx = "src/models/jj/modelo_4_indices.joblib"
ruta_imp = "src/models/jj/modelo_5_implicit.pkl"
ruta_imp_dat = "src/models/jj/modelo_5_implicit_dataset.pkl"
ruta_modelo_ncf = "src/models/jj/modelo_6_ncf.onnx"
ruta_ncf_user2idx = "src/models/jj/ncf_user2idx.json"
ruta_ncf_item2idx = "src/models/jj/ncf_item2idx.json"


@asynccontextmanager
async def lifespan(the_app: FastAPI):
    """Gestiona el ciclo de vida de la aplicación: carga modelos al arrancar, limpia al cerrar."""
    logger.info("[STARTUP] Iniciando carga de modelos de IA en memoria...")
    logger.info("[INFO] Cargando ratings (434MB)... espera unos 30-40s.")

    # --- Inicialización de estado ---
    the_app.state.modelo_svd = None
    the_app.state.df_ratings_ia = None
    the_app.state.df_catalogo = None
    the_app.state.modelo_knn = None
    the_app.state.modelo_wnd = None
    the_app.state.wnd_mappings = None
    the_app.state.modelo_tfidf_mat = None
    the_app.state.modelo_tfidf_idx = None
    the_app.state.modelo_imp = None
    the_app.state.modelo_imp_dat = None
    the_app.state.modelo_ncf = None
    the_app.state.ncf_user2idx = None
    the_app.state.ncf_item2idx = None

    # --- Modelo 1: SVD ---
    if os.path.exists(ruta_modelo_svd):
        try:
            import joblib
            the_app.state.modelo_svd = joblib.load(ruta_modelo_svd)
            logger.info("Modelo SVD cargado correctamente (Joblib).")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo SVD: {e}.")

    # --- Modelo 2: KNN ---
    if os.path.exists(ruta_modelo_knn):
        try:
            import joblib
            the_app.state.modelo_knn = joblib.load(ruta_modelo_knn)
            logger.info("Modelo KNN+Cosine cargado (Joblib).")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo KNN: {e}")

    # --- Modelo 3: Wide & Deep (ONNX Runtime) ---
    if os.path.exists(ruta_modelo_wnd) and os.path.exists(ruta_wnd_map):
        try:
            import onnxruntime as ort
            import pickle
            the_app.state.modelo_wnd = ort.InferenceSession(ruta_modelo_wnd)
            with open(ruta_wnd_map, "rb") as f:
                the_app.state.wnd_mappings = pickle.load(f)
            logger.info("Modelo Wide&Deep ONNX cargado correctamente.")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo Wide&Deep: {e}")

    # --- Modelo 4: Content-Based (TF-IDF) ---
    if os.path.exists(ruta_tfidf_mat) and os.path.exists(ruta_tfidf_idx):
        try:
            import joblib
            the_app.state.modelo_tfidf_mat = joblib.load(ruta_tfidf_mat)
            the_app.state.modelo_tfidf_idx = joblib.load(ruta_tfidf_idx)
            logger.info("Modelo TF-IDF cargado correctamente (Joblib).")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo TF-IDF: {e}")

    # --- Modelo 5: Implicit BPR ---
    if os.path.exists(ruta_imp) and os.path.exists(ruta_imp_dat):
        try:
            import pickle
            with open(ruta_imp, "rb") as f:
                the_app.state.modelo_imp = pickle.load(f)
            with open(ruta_imp_dat, "rb") as f:
                the_app.state.modelo_imp_dat = pickle.load(f)
            logger.info("Modelo Implicit cargado correctamente.")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo Implicit: {e}")

    # --- Modelo 6: NCF (ONNX) ---
    if os.path.exists(ruta_modelo_ncf):
        try:
            import onnxruntime as ort
            import json
            the_app.state.modelo_ncf = ort.InferenceSession(ruta_modelo_ncf)
            if os.path.exists(ruta_ncf_user2idx):
                with open(ruta_ncf_user2idx, "r") as f:
                    the_app.state.ncf_user2idx = {int(k): v for k, v in json.load(f).items()}
            if os.path.exists(ruta_ncf_item2idx):
                with open(ruta_ncf_item2idx, "r") as f:
                    the_app.state.ncf_item2idx = {int(k): v for k, v in json.load(f).items()}
            logger.info("Modelo NCF ONNX cargado correctamente.")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo NCF: {e}")

    # --- CARGA DE DATOS (CSV) ---
    try:
        if os.path.exists(ruta_ratings):
            the_app.state.df_ratings_ia = pd.read_csv(ruta_ratings)
            logger.info(f"Ratings cargados: {len(the_app.state.df_ratings_ia):,} filas.")
        if os.path.exists(ruta_catalogo):
            the_app.state.df_catalogo = pd.read_csv(ruta_catalogo)
            logger.info(f"Catálogo cargado: {len(the_app.state.df_catalogo):,} películas.")
    except Exception as e:
        logger.error(f"Error al cargar archivos CSV de datos: {e}")

    logger.info("[STARTUP] Carga de modelos completada.")
    yield  # La app está corriendo
    logger.info("[SHUTDOWN] Cerrando aplicación...")


app = FastAPI(lifespan=lifespan)


@app.get("/status")
def check_status():
    return {"status": "Backend funcionando correctamente"}


@app.get("/usuarios")
def obtener_usuarios():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)  # Devuelve filas como diccionarios
    cursor.execute("SELECT * FROM users")
    usuarios = cursor.fetchall()
    cursor.close()
    conn.close()
    return usuarios


def importar_series(conn):
    print("inicio importacion series")
    # Importamos Series
    ejecutar_importacion(conn, "src/data/clean/tmdb_shows_limpio.csv", "tv")
    print("fin importacion series")


def importar_peliculas(conn):
    print("inicio importacion peliculas, se viene lo chungo")
    # Importamos Películas
    # ejecutar_importacion(conn, 'src/data/raw/tmdb/movies/peliculas_2020.csv', 'movie')
    ejecutar_importacion(conn, "src/data/clean/tmdb_movies_limpio.csv", "movie")
    print("fin importacion peliculas, al fin!")


def limpiar_csv():
    df = pd.read_csv("src/data/raw/tmdb/movies/peliculas_2020.csv")

    # 2. LIMPIEZA CRÍTICA: Convertir NaN de Pandas a None de Python (NULL en MySQL)
    # Esto evita el error de "Lost connection" por tipos de datos inválidos
    df = df.replace({np.nan: None})

    # 3. Asegurar tipos de datos (evita el Out of Range)
    df["vote_average"] = pd.to_numeric(df["vote_average"]).round(2)
    df["popularity"] = pd.to_numeric(df["popularity"]).round(2)
    df.to_csv(
        "src/data/raw/tmdb/movies/peliculas_2020_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )


@app.post("/importar_datos")
def importar_datos():
    conn = get_db_connection()
    try:
        limpiar_tablas_contenido(conn)
        # limpiar_csv()
        importar_series(conn)
        importar_peliculas(conn)
        return {"status": "success", "message": "Catálogos actualizados correctamente"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        conn.close()


# LoginRequest y RegisterRequest importados de src.schemas.schemas


@app.post("/login")
def login(datos: LoginRequest):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # 1. Buscamos al usuario solo por nombre de usuario
    query = "SELECT id_usuario, username, email, passwd FROM users WHERE username = %s"
    cursor.execute(query, (datos.username,))

    usuario = cursor.fetchone()

    cursor.close()
    conn.close()

    if usuario:
        # Obtenemos el hash que estaba guardado en BD
        hash_guardado = usuario["passwd"]

        # 2. Comprobamos si la contraseña coincide (Soportando tanto Hash nuevo como Texto Plano antiguo)
        es_valido = False
        if hash_guardado.startswith("$2b$") or hash_guardado.startswith("$2a$"):
            # Si es un hash de bcrypt válido
            es_valido = bcrypt.checkpw(
                datos.password.encode("utf-8"), hash_guardado.encode("utf-8")
            )
        else:
            # Si es una contraseña antigua en texto plano (ej: 'root')
            es_valido = datos.password == hash_guardado

        if es_valido:
            # Por seguridad, borramos la contraseña del diccionario temporal antes de mandarlo al frontend
            del usuario["passwd"]

            return {"status": "success", "message": "Login exitoso", "user": usuario}
        else:
            raise HTTPException(
                status_code=401, detail="Usuario o contraseña incorrectos"
            )
    else:
        raise HTTPException(status_code=401, detail="Usuario o contraseña incorrectos")


@app.get("/genres")
def obtener_generos():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, name FROM genres ORDER BY name ASC")
    generos = cursor.fetchall()
    cursor.close()
    conn.close()
    return generos


@app.post("/register")
def register(datos: RegisterRequest):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        # 1. Verificar si el usuario o email ya existen
        cursor.execute(
            "SELECT id_usuario FROM users WHERE username = %s OR email = %s",
            (datos.username, datos.email),
        )
        if cursor.fetchone():
            raise HTTPException(
                status_code=400, detail="El nombre de usuario o email ya están en uso."
            )

        # 2. Hashear la contraseña
        password_hash = bcrypt.hashpw(
            datos.password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

        # 3. Insertar usuario
        query_user = """
            INSERT INTO users (username, email, passwd, fecha_nacimiento, sexo)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(
            query_user,
            (
                datos.username,
                datos.email,
                password_hash,
                datos.fecha_nacimiento,
                datos.sexo,
            ),
        )
        id_usuario = cursor.lastrowid

        # 4. Manejar intereses
        intereses = datos.intereses
        if not intereses:
            # Seleccionar los 3 géneros más populares (basado en reproducciones totales)
            query_pop = """
                SELECT cg.genre_id
                FROM content_genres cg
                JOIN content_stats cs ON cg.content_id = cs.content_id
                GROUP BY cg.genre_id
                ORDER BY SUM(cs.reproducciones_totales) DESC
                LIMIT 3
            """
            cursor.execute(query_pop)
            res_pop = cursor.fetchall()
            intereses = [row["genre_id"] for row in res_pop]

        # Insertar intereses
        if intereses:
            query_int = (
                "INSERT INTO user_interests (id_usuario, genre_id) VALUES (%s, %s)"
            )
            for g_id in intereses:
                cursor.execute(query_int, (id_usuario, g_id))

        conn.commit()
        return {
            "status": "success",
            "message": "Usuario registrado correctamente",
            "user_id": id_usuario,
        }

    except Exception as e:
        conn.rollback()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error en el servidor: {str(e)}")
    finally:
        cursor.close()
        conn.close()





@app.get("/recomendar/svd/{user_id}")
@app.get("/recomendar/{user_id}")
def recomendar_peliculas(user_id: int, n: int = 10):
    """
    Endpoint que devuelve las top-N películas recomendadas para un usuario.
    Usa el modelo SVD entrenado para predecir ratings de películas no vistas.
    """
    print(f"DEBUG: Petición SVD para User {user_id}")

    # --- DIAGNÓSTICO EN TIEMPO DE EJECUCIÓN ---
    if app.state.modelo_svd is None:
        causa = "app.state.modelo_svd es None (Error de persistencia)"
        print(f"DEBUG ERROR: {causa}")
        raise HTTPException(status_code=503, detail=causa)
    if app.state.df_ratings_ia is None:
        causa = "app.state.df_ratings_ia es None (CSV de ratings no cargado)"
        print(f"DEBUG ERROR: {causa}")
        raise HTTPException(status_code=503, detail=causa)

    # 1. Películas que este usuario YA ha visto
    pelis_vistas = set(
        app.state.df_ratings_ia[app.state.df_ratings_ia["userId"] == user_id][
            "tmdb_id"
        ].tolist()
    )

    # 2. Todas las películas disponibles en el sistema (aseguramos que parten del catálogo)
    if app.state.df_catalogo is not None:
        todas_las_pelis = set(app.state.df_catalogo["tmdb_id"].unique())
    else:
        todas_las_pelis = set(app.state.df_ratings_ia["tmdb_id"].unique())

    # 3. Candidatas = las que NO ha visto
    pelis_no_vistas = todas_las_pelis - pelis_vistas

    # Si el usuario ha visto todas (raro pero posible), avisamos
    if not pelis_no_vistas:
        return {
            "recomendaciones": [],
            "mensaje": "Este usuario ya ha valorado todas las películas disponibles.",
        }

    # 4. Predecimos la nota para cada película candidata
    predicciones = []
    for tmdb_id in pelis_no_vistas:
        pred = app.state.modelo_svd.predict(user_id, tmdb_id)
        predicciones.append(
            {"tmdb_id": int(tmdb_id), "predicted_rating": round(pred.est, 2)}
        )

    # 5. Ordenamos de mayor a menor y nos quedamos con las top-N
    predicciones.sort(key=lambda x: x["predicted_rating"], reverse=True)
    top_n = predicciones[:n]

    # 6. Enriquecemos con datos del catálogo
    enriquecer_recomendaciones(top_n)

    return {"recomendaciones": top_n, "modelo": "SVD (Surprise)"}


##############################################################################################
#  Helper: Enriquecer recomendaciones con datos del catálogo
##############################################################################################


def enriquecer_recomendaciones(recomendaciones):
    """Añade datos del catálogo (título, poster, etc.) a una lista de diccionarios con tmdb_id."""
    if app.state.df_catalogo is not None:
        for rec in recomendaciones:
            match = app.state.df_catalogo[
                app.state.df_catalogo["tmdb_id"] == rec["tmdb_id"]
            ]
            if not match.empty:
                fila = match.iloc[0]
                rec["titulo"] = str(fila.get("titulo", "Sin Título"))
                rec["poster_path"] = str(fila.get("poster_path", ""))
                rec["overview"] = str(fila.get("overview", "Sin sinopsis disponible."))
                rec["vote_average"] = float(fila.get("vote_average", 0))
            else:
                rec["titulo"] = f"Película #{rec['tmdb_id']}"
                rec["poster_path"] = ""
                rec["overview"] = "Sin sinopsis disponible."
                rec["vote_average"] = 0.0


##############################################################################################
#  Recomendación Modelo KNN + Cosine Similarity
##############################################################################################


@app.get("/recomendar/knn/{user_id}")
def recomendar_knn(user_id: int, n: int = 10):
    """Endpoint de recomendaciones usando KNN + Cosine Similarity (Modelo 2)."""
    print(f"DEBUG: Petición KNN para User {user_id}")
    if app.state.modelo_knn is None:
        causa = "app.state.modelo_knn es None"
        print(f"DEBUG ERROR: {causa}")
        raise HTTPException(status_code=503, detail=causa)
    if app.state.df_ratings_ia is None:
        causa = "app.state.df_ratings_ia es None"
        print(f"DEBUG ERROR: {causa}")
        raise HTTPException(status_code=503, detail=causa)

    pelis_vistas = set(
        app.state.df_ratings_ia[app.state.df_ratings_ia["userId"] == user_id][
            "tmdb_id"
        ].tolist()
    )
    # Extraemos items del modelo y buscamos similares en base a la historia del usuario
    # En este setup simplificado buscamos los items mas populares del dataset
    item_counts = app.state.df_ratings_ia["tmdb_id"].value_counts()
    top_items = item_counts.index.tolist()
    todas = (
        set(app.state.df_catalogo["tmdb_id"].unique())
        if app.state.df_catalogo is not None
        else set(app.state.df_ratings_ia["tmdb_id"].unique())
    )
    pelis_no_vistas = todas - pelis_vistas

    if not pelis_no_vistas:
        return {
            "recomendaciones": [],
            "modelo": "KNN+Cosine",
            "mensaje": "Este usuario ya ha valorado todas las películas.",
        }

    predicciones = []
    for tmdb_id in pelis_no_vistas:
        pred = app.state.modelo_knn.predict(user_id, tmdb_id)
        predicciones.append(
            {"tmdb_id": int(tmdb_id), "predicted_rating": round(pred.est, 2)}
        )

    predicciones.sort(key=lambda x: x["predicted_rating"], reverse=True)
    top_n = predicciones[:n]
    enriquecer_recomendaciones(top_n)

    return {"recomendaciones": top_n, "modelo": "KNN+Cosine"}


##############################################################################################
#  Recomendación Modelo Wide & Deep (PyTorch)
##############################################################################################


@app.get("/recomendar/wnd/{user_id}")
def recomendar_wnd_endpoint(user_id: int, n: int = 10):
    """Endpoint de recomendaciones usando Wide & Deep Neural Network (ONNX Native)."""
    print(f"DEBUG: Petición Wide&Deep para User {user_id}")
    if app.state.modelo_wnd is None or app.state.wnd_mappings is None:
        causa = "app.state.modelo_wnd o mappings es None"
        print(f"DEBUG ERROR: {causa}")
        raise HTTPException(status_code=503, detail=causa)
    if app.state.df_ratings_ia is None:
        causa = "app.state.df_ratings_ia es None"
        print(f"DEBUG ERROR: {causa}")
        raise HTTPException(status_code=503, detail=causa)

    user2idx = app.state.wnd_mappings["user2idx"]
    movie2idx = app.state.wnd_mappings["movie2idx"]

    if user_id not in user2idx:
        return {
            "recomendaciones": [],
            "modelo": "Wide&Deep (ONNX)",
            "mensaje": f"El usuario {user_id} no cumplió el filtro de entrenamiento (>100 valoraciones).",
        }

    u_idx = user2idx[user_id]
    pelis_vistas = set(
        app.state.df_ratings_ia[app.state.df_ratings_ia["userId"] == user_id][
            "tmdb_id"
        ].tolist()
    )

    candidatas = [
        (tid, midx) for tid, midx in movie2idx.items() if tid not in pelis_vistas
    ]
    if not candidatas:
        return {
            "recomendaciones": [],
            "modelo": "Wide&Deep (ONNX)",
            "mensaje": "Este usuario ya ha valorado todas las películas.",
        }

    tmdb_ids, movie_indices = zip(*candidatas)

    # Creamos arrays NumPy en lugar de tensores PyTorch
    user_arr = np.array([u_idx] * len(movie_indices), dtype=np.int64)
    movie_arr = np.array(list(movie_indices), dtype=np.int64)

    # Inferencia ONNX
    try:
        ort_inputs = {
            "user_id": user_arr,
            "movie_id": movie_arr,
        }
        preds_norm = app.state.modelo_wnd.run(None, ort_inputs)[0]
    except Exception as e:
        print(f"ERROR en inferencia Wide&Deep: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error en inferencia del modelo WnD: {e}"
        )

    # Restringir predicción a rango [0.5, 5.0]
    preds_onnx = np.clip(preds_norm, 0.5, 5.0)

    predicciones = [
        {"tmdb_id": int(tid), "predicted_rating": round(float(preds_onnx[i]), 2)}
        for i, tid in enumerate(tmdb_ids)
    ]

    predicciones.sort(key=lambda x: x["predicted_rating"], reverse=True)
    top_n = predicciones[:n]
    enriquecer_recomendaciones(top_n)

    return {"recomendaciones": top_n, "modelo": "Wide&Deep (ONNX)"}


##############################################################################################
#  Recomendación Modelo 4: Content-Based / Cold Start
##############################################################################################


@app.get("/recomendar/content/{user_id}")
def recomendar_content_endpoint(user_id: int, n: int = 10):
    """Endpoint de recomendaciones por contenido (Modelo 4)."""
    print(f"DEBUG: Petición Content-Based para User {user_id}")
    if app.state.modelo_tfidf_mat is None or app.state.modelo_tfidf_idx is None:
        causa = "app.state.modelo_tfidf_mat o idx es None"
        print(f"DEBUG ERROR: {causa}")
        raise HTTPException(status_code=503, detail=causa)
    if app.state.df_ratings_ia is None or app.state.df_catalogo is None:
        causa = "app.state.df_ratings_ia o df_catalogo es None"
        print(f"DEBUG ERROR: {causa}")
        raise HTTPException(status_code=503, detail=causa)

    user_ratings = app.state.df_ratings_ia[app.state.df_ratings_ia["userId"] == user_id]

    # Cold Start (Usuario sin historial)
    if user_ratings.empty:
        # Recomendamos por popularidad general
        if "vote_count" in app.state.df_catalogo.columns:
            top_pop = (
                app.state.df_catalogo[app.state.df_catalogo["vote_count"] > 100]
                .sort_values(by="vote_average", ascending=False)
                .head(n)
            )
        else:
            top_pop = app.state.df_catalogo.head(n)

        recomendaciones = []
        for idx, row in top_pop.iterrows():
            recomendaciones.append(
                {"tmdb_id": int(row["tmdb_id"]), "predicted_rating": 4.5}
            )

        enriquecer_recomendaciones(recomendaciones)
        return {
            "recomendaciones": recomendaciones,
            "modelo": "TF-IDF (Cold Start Populares)",
        }

    # Usuario con historial -> Content-Based por similitud
    viewed_ids = user_ratings["tmdb_id"].tolist()
    tfidf_indices = [
        app.state.modelo_tfidf_idx[mid]
        for mid in viewed_ids
        if mid in app.state.modelo_tfidf_idx
    ]

    if not tfidf_indices:
        return {
            "recomendaciones": [],
            "modelo": "Content-Based",
            "mensaje": "Película favorita no encontrada.",
        }

    # Calculamos perfil del usuario (media de los vectores de peliculas vistas)
    user_profile = np.asarray(app.state.modelo_tfidf_mat[tfidf_indices].mean(axis=0))
    # Similitud coseno contra todo el catálogo
    cos_sim = cosine_similarity(user_profile, app.state.modelo_tfidf_mat).flatten()

    # Obtenemos los mas similares (ignorando el mismisimo 1.0)
    top_indices = cos_sim.argsort()[::-1][1 : n + 1]

    predicciones = []
    for idx_sim in top_indices:
        peli = app.state.df_catalogo.iloc[idx_sim]
        score_sim = cos_sim[idx_sim]
        # Creamos un rating visual combinando su rating original con la similitud
        pseudo_rating = min(5.0, 4.0 * (0.8 + 0.2 * score_sim))
        predicciones.append(
            {
                "tmdb_id": int(peli["tmdb_id"]),
                "predicted_rating": round(pseudo_rating, 2),
            }
        )

    enriquecer_recomendaciones(predicciones)
    return {"recomendaciones": predicciones, "modelo": "Content-Based"}


##############################################################################################
#  Recomendación Modelo 5: Implicit BPR
##############################################################################################


@app.get("/recomendar/implicit/{user_id}")
def recomendar_implicit_endpoint(user_id: int, n: int = 10):
    """Endpoint de recomendaciones usando filtrado colaborativo BPR de la librería implicit."""
    print(f"DEBUG: Petición Implicit BPR para User {user_id}")
    if app.state.modelo_imp is None or app.state.modelo_imp_dat is None:
        causa = f"app.state.modelo_imp es {app.state.modelo_imp is None} | app.state.modelo_imp_dat es {app.state.modelo_imp_dat is None}"
        print(f"DEBUG ERROR: {causa}")
        raise HTTPException(status_code=503, detail=f"Fallo en Modelos Imply: {causa}")
    if app.state.df_ratings_ia is None:
        causa = "app.state.df_ratings_ia es None"
        print(f"DEBUG ERROR: {causa}")
        raise HTTPException(status_code=503, detail=causa)

    user2idx = app.state.modelo_imp_dat["user2idx"]
    item2idx = app.state.modelo_imp_dat["item2idx"]
    idx2item = {v: k for k, v in item2idx.items()}

    if user_id not in user2idx:
        return {
            "recomendaciones": [],
            "modelo": "Implicit BPR",
            "mensaje": f"El usuario {user_id} no tiene historial en el modelo BPR.",
        }

    u_idx = user2idx[user_id]

    # Extraemos arrays nativos mediante NumPy para evitar el bug Cython de implicit.recommend() en Windows!
    u_factors = np.asarray(app.state.modelo_imp.user_factors[u_idx])
    i_factors = np.asarray(app.state.modelo_imp.item_factors)

    # Producto escalar vectorizado (Calcula score para tooooodas las peliculas del modelo a la vez)
    scores = u_factors.dot(i_factors.T)

    # Localizamos las que ya ha visto en el dataset completo
    pelis_vistas = set(
        app.state.df_ratings_ia[app.state.df_ratings_ia["userId"] == user_id][
            "tmdb_id"
        ].tolist()
    )

    # Forzamos su score a menos infinito para que nunca salgan en el top
    for tid in pelis_vistas:
        if tid in item2idx:
            midx = item2idx[tid]
            scores[midx] = -np.inf

    # np.argsort()[::-1] ordena de mayor a menor y extraemos las top N posiciones de memoria
    top_indices = np.argsort(scores)[::-1][:n]

    predicciones = []
    for idx_sim in top_indices:
        tid = idx2item[idx_sim]
        score_puro = float(scores[idx_sim])
        # Al ser ranking BPR no devuelve nota 0-5. Reescalamos visualmente para UI amigable.
        rating_ui = min(5.0, max(0.5, 3.5 + (score_puro * 0.2)))

        predicciones.append(
            {"tmdb_id": int(tid), "predicted_rating": round(rating_ui, 2)}
        )

    enriquecer_recomendaciones(predicciones)

    return {"recomendaciones": predicciones, "modelo": "Implicit BPR"}


##############################################################################################
#  Recomendación Modelo 6: NCF (Neural Collaborative Filtering)
##############################################################################################


@app.get("/recomendar/ncf/{user_id}")
def recomendar_ncf_endpoint(user_id: int, n: int = 10):
    """
    Endpoint de recomendaciones usando NCF-Lite (GMF + MLP) via ONNX Runtime.
    El modelo produce logits de relevancia para cada item; se seleccionan los Top-N
    excluyendo items ya vistos por el usuario.
    """
    logger.info(f"Petición NCF para User {user_id}")

    if app.state.modelo_ncf is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo NCF no cargado. Ejecuta modelo_6_ncf.py para generar los artefactos."
        )
    if app.state.ncf_user2idx is None or app.state.ncf_item2idx is None:
        raise HTTPException(
            status_code=503,
            detail="Mapeos NCF (user2idx/item2idx) no cargados."
        )

    # Verificar que el usuario existe en el vocabulario del modelo
    if user_id not in app.state.ncf_user2idx:
        raise HTTPException(
            status_code=404,
            detail=f"Usuario {user_id} no encontrado en el vocabulario NCF. "
                   f"Solo usuarios con >= 200 ratings están cubiertos."
        )

    user_idx = app.state.ncf_user2idx[user_id]

    # Construir mapeos inversos
    idx2item = {v: k for k, v in app.state.ncf_item2idx.items()}
    n_items = len(app.state.ncf_item2idx)

    # Puntuar TODOS los items de una sola vez (batch inference via ONNX)
    user_ids_np = np.full(n_items, user_idx, dtype=np.int64)
    item_ids_np = np.arange(n_items, dtype=np.int64)

    scores = app.state.modelo_ncf.run(
        None,
        {"user_ids": user_ids_np, "item_ids": item_ids_np}
    )[0]  # [n_items]

    # Excluir items ya vistos
    if app.state.df_ratings_ia is not None:
        pelis_vistas = set(
            app.state.df_ratings_ia[app.state.df_ratings_ia["userId"] == user_id][
                "tmdb_id"
            ].tolist()
        )
        for tid in pelis_vistas:
            if tid in app.state.ncf_item2idx:
                midx = app.state.ncf_item2idx[tid]
                scores[midx] = -np.inf

    # Seleccionar Top-N
    top_indices = np.argsort(scores)[::-1][:n]

    predicciones = []
    for idx_item in top_indices:
        tid = idx2item[idx_item]
        score_puro = float(scores[idx_item])
        # Transformar logit a rating visual amigable (3.0 - 5.0 rango)
        rating_ui = min(5.0, max(0.5, 3.5 + (score_puro * 0.3)))
        predicciones.append(
            {"tmdb_id": int(tid), "predicted_rating": round(rating_ui, 2)}
        )

    enriquecer_recomendaciones(predicciones)
    return {"recomendaciones": predicciones, "modelo": "NCF-Lite"}

```

- Nuevo endpoint `GET /recomendar/ncf/{user_id}`
- Carga del modelo ONNX + JSON mappings en el lifespan
- Batch inference: puntúa todos los items en una sola llamada ONNX
- Excluye items ya vistos, reescala logits a rating visual 0.5-5.0

### Frontend — Selector
```diff:2_Usuario.py
import streamlit as st
import requests
import datetime

st.set_page_config(page_title="SPIRE Streaming - Usuario", layout="wide")

# Validar si el usuario está logueado en Streamlit Session State
if "usuario_autenticado" not in st.session_state:
    st.session_state["usuario_autenticado"] = False
    st.session_state["usuario_actual"] = None

# ---- ESTILOS PERSONALIZADOS (Azul Marino y Oro) ----
st.markdown(
    """
    <style>
    /* Fondo y colores principales */
    .stApp {
        background-color: #001220;
        color: #f0f0f0;
    }
    
    /* Títulos en Dorado Premium (Más profundo) */
    h1, h2, h3, .stSubheader {
        color: #B8860B !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Texto normal */
    p, span, label {
        color: #e0e0e0 !important;
    }
    
    /* Estilo para los Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(0, 31, 63, 0.5);
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: transparent;
        border-radius: 5px;
        color: #f0f0f0;
        border: 1px solid transparent;
        transition: all 0.2s;
    }
    /* Tab seleccionado: Oro oscuro con texto azul muy oscuro para contraste */
    .stTabs [aria-selected="true"] {
        background-color: #B8860B !important;
        color: #001220 !important;
        border-color: #B8860B !important;
        font-weight: bold;
    }
    
    /* Botones Premium */
    div.stButton > button {
        background-color: #B8860B !important;
        color: #001220 !important;
        font-weight: bold !important;
        border: 1px solid #B8860B !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div.stButton > button:hover {
        background-color: #001f3f !important;
        color: #B8860B !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 20px rgba(184, 134, 11, 0.4) !important;
    }
    
    /* Inputs y widgets estilizados */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stMultiSelect>div>div>div, .stDateInput>div>div>input {
        background-color: #001f3f !important;
        color: white !important;
        border: 1px solid #1a3a5f !important;
        border-radius: 8px !important;
    }
    
    /* Checkboxes personalizados (Grid de gustos) */
    .stCheckbox > label > div[data-testid="stMarkdownContainer"] > p {
        color: #f0f0f0 !important;
        font-size: 0.9rem;
    }
    /* Estilo del cuadro del checkbox */
    span[data-baseweb="checkbox"] > div {
        border-color: #B8860B !important;
    }
    /* Checkbox marcado: fondo oro oscuro */
    div[data-checked="true"] {
        background-color: #B8860B !important;
    }
    
    /* Mensajes de información y éxito */
    .stAlert {
        border-radius: 10px !important;
        border: 1px solid #B8860B !important;
        background-color: rgba(0, 31, 63, 0.8) !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not st.session_state["usuario_autenticado"]:
    st.title("BIENVENIDO A SPIRE STREAMING")
    st.markdown(
        "Accede a la experiencia definitiva de cine o crea tu cuenta exclusiva."
    )

    tab_login, tab_registro = st.tabs(["Iniciar Sesión", "Registrarse"])

    # --- PESTAÑA LOGIN ---
    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            submit_button = st.form_submit_button("Entrar")

            if submit_button:
                payload = {"username": username, "password": password}
                try:
                    response = requests.post(
                        "http://localhost:8000/login", json=payload
                    )

                    if response.status_code == 200:
                        datos_usuario = response.json()
                        st.session_state["usuario_autenticado"] = True
                        st.session_state["usuario_actual"] = datos_usuario["user"]
                        st.success("¡Login exitoso!")
                        st.rerun()  # Recarga la página para mostrar el catálogo
                    elif response.status_code == 401:
                        st.error(
                            "Credenciales incorrectas. Verifica tu usuario o regístrate en la otra pestaña."
                        )
                    else:
                        st.error(f"Error en el servidor: {response.status_code}")
                except Exception as e:
                    st.error(f"Error de conexión: {e}")

    # --- PESTAÑA REGISTRO ---
    with tab_registro:
        st.subheader("Crea tu cuenta")

        # Intentar cargar géneros para el selector de gustos
        opciones_generos = {}
        try:
            resp_gen = requests.get("http://localhost:8000/genres")
            if resp_gen.status_code == 200:
                for g in resp_gen.json():
                    opciones_generos[g["name"]] = g["id"]
        except Exception as e:
            st.error(f"Error cargando categorías: {e}")

        with st.form("register_form"):
            col1, col2 = st.columns(2)
            with col1:
                new_username = st.text_input(
                    "Usuario", placeholder="Tu nombre de usuario"
                )
                new_email = st.text_input("Email", placeholder="ejemplo@correo.com")
                new_password = st.text_input("Contraseña", type="password")

            with col2:
                # Ajustamos el calendario para cubrir desde 1900
                new_fecha_nac = st.date_input(
                    "Fecha de Nacimiento",
                    value=None,
                    min_value=datetime.date(1900, 1, 1),
                    max_value=datetime.date.today(),
                )
                new_sexo = st.selectbox(
                    "Sexo",
                    ["Hombre", "Mujer", "Otro"],
                    index=None,
                    placeholder="Selecciona...",
                )
                new_password_confirm = st.text_input(
                    "Confirmar Contraseña", type="password"
                )

            st.write("---")
            st.markdown("**¿Qué generos te gusta ver?**")

            # --- REFACTOR: Grid de Checkboxes para Categorías ---
            if not opciones_generos:
                st.caption("Cargando categorías o servidor no disponible...")

            # Mostramos las categorías en un grid de 4 columnas para que quepa bien
            if opciones_generos:
                generos_lista = list(opciones_generos.keys())
                num_cols = 4
                rows = [
                    generos_lista[i : i + num_cols]
                    for i in range(0, len(generos_lista), num_cols)
                ]

                for row_genres in rows:
                    cols = st.columns(num_cols)
                    for i, genre_name in enumerate(row_genres):
                        # Usamos key para persistir el valor en session_state y capturarlo al enviar
                        cols[i].checkbox(genre_name, key=f"genre_{genre_name}")

            btn_register = st.form_submit_button("REGISTRARSE AHORA")

            if btn_register:
                # Recopilamos los gustos marcados desde el session_state
                gustos_actuales = [
                    opciones_generos[name]
                    for name in opciones_generos
                    if st.session_state.get(f"genre_{name}")
                ]

                if not new_username or not new_password or not new_email:
                    st.warning(
                        "Por favor, rellena los campos obligatorios (Usuario, Email y Contraseña)."
                    )
                elif new_password != new_password_confirm:
                    st.error("Las contraseñas no coinciden.")
                else:
                    # Preparar payload
                    payload = {
                        "username": new_username,
                        "email": new_email,
                        "password": new_password,
                        "fecha_nacimiento": str(new_fecha_nac)
                        if new_fecha_nac
                        else None,
                        "sexo": new_sexo,
                        "intereses": gustos_actuales,
                    }

                    try:
                        response = requests.post(
                            "http://127.0.0.1:8000/register", json=payload
                        )
                        if response.status_code == 200:
                            st.success(
                                f"¡Bienvenido, {new_username}! Tu cuenta ha sido creada."
                            )
                            st.info(
                                "Ya puedes ir a la pestaña 'Iniciar Sesión' para entrar."
                            )
                        else:
                            error_detail = response.json().get(
                                "detail", "Error desconocido"
                            )
                            st.error(f"Error al registrar: {error_detail}")
                    except Exception as e:
                        st.error(f"No se pudo conectar con el servidor: {e}")
#################################################################################################
# ---- PANTALLA PRINCIPAL DEL USUARIO (Logueado) ----

else:
    import pandas as pd
    import os

    usuario = st.session_state["usuario_actual"]
    nombre_mostrar = usuario.get("username", f"Usuario #{usuario.get('id_usuario')}")

    # Barra superior con Logout
    col_titulo, col_logout = st.columns([8, 2])
    with col_titulo:
        st.title(f"Bienvenido a SPIRE, {nombre_mostrar}!")
    with col_logout:
        st.write("")  # Espacio
        if st.button("Cerrar Sesión"):
            st.session_state["usuario_autenticado"] = False
            st.session_state["usuario_actual"] = None
            st.rerun()

    st.markdown(
        "Descubre Películas y Series gracias a nuestro Recomendador de Inteligencia Artificial."
    )

    # --- BUSCADOR GLOBAL ---
    search_query = st.text_input("Busca por título o palabras clave...")

    # --- SELECTOR DE MODELO DE IA ---
    modelo_ia = st.sidebar.selectbox(
        "Motor de Recomendación",
        [
            "SVD (Rápido)",
            "KNN + Cosine (Explicable)",
            "Wide & Deep (Profundo)",
            "Content-Based (Cold-Start)",
            "Implicit BPR (Ranking Top)",
        ],
        index=4,  # Ponemos el BPR por defecto porque es el mejor
    )
    mapa_endpoints = {
        "SVD (Rápido)": "recomendar",
        "KNN + Cosine (Explicable)": "recomendar/knn",
        "Wide & Deep (Profundo)": "recomendar/wnd",
        "Content-Based (Cold-Start)": "recomendar/content",
        "Implicit BPR (Ranking Top)": "recomendar/implicit",
    }
    endpoint_ia = mapa_endpoints[modelo_ia]

    # --- DEV TOOL: Simular otro usuario ---
    id_simulado = st.sidebar.number_input(
        "Datos ratings - ID Usuario ",
        value=usuario.get("id_usuario", 1),
        step=1,
        help="Permite simular predicciones para IDs de súper-usuarios (ej. 9) que existen en el set de datos pero no en tu base de datos local.",
    )

    # --- CARGA DE DATOS ---
    @st.cache_data
    def load_catalog_data():
        movies_path = "src/data/ready/dataset_final_movies.csv"
        shows_path = "src/data/ready/dataset_final_shows.csv"

        df_movies = pd.DataFrame()
        df_shows = pd.DataFrame()

        if os.path.exists(movies_path):
            try:
                df_movies = pd.read_csv(
                    movies_path, on_bad_lines="skip", engine="python"
                )
            except Exception:
                pass

        if os.path.exists(shows_path):
            try:
                df_shows = pd.read_csv(shows_path, on_bad_lines="skip", engine="python")
            except Exception:
                pass

        return df_movies, df_shows

    df_movies, df_shows = load_catalog_data()

    # =====================================================================================
    # Dibuja un grid de postales con poster, título y sinopsis
    # =====================================================================================
    def render_cards(df, limit=8, key_prefix="card", date_col="fecha_estreno"):
        """Dibuja una fila de postales con poster, título y botón de sinopsis."""
        if df.empty:
            st.info("No hay datos disponibles para mostrar.")
            return

        cols = st.columns(4)
        for idx, (_, row) in enumerate(df.head(limit).iterrows()):
            with cols[idx % 4]:
                # Poster
                poster_url = "https://via.placeholder.com/300x450.png?text=Sin+Poster"
                if (
                    pd.notna(row.get("poster_path"))
                    and str(row.get("poster_path")) != ""
                ):
                    poster_url = f"https://image.tmdb.org/t/p/w500{row['poster_path']}"
                st.image(poster_url, use_container_width=True)

                # Título truncado
                titulo = str(row.get("titulo", "Sin Título"))
                if len(titulo) > 30:
                    titulo = titulo[:27] + "..."

                # Año
                year = str(row.get(date_col, ""))[:4]
                if year and year != "nan":
                    st.markdown(f"**{titulo}** ({year})")
                else:
                    st.markdown(f"**{titulo}**")

                # Nota
                nota = row.get("vote_average", 0)
                if nota:
                    st.caption(f"{nota}")

                # Botón de sinopsis
                if st.button("Ver sinopsis", key=f"{key_prefix}_{idx}"):
                    st.toast(str(row.get("overview", "Sin sinopsis disponible.")))

    # =====================================================================================
    # Renderiza las recomendaciones IA dentro de una pestaña
    # =====================================================================================
    def render_recomendaciones_ia(key_prefix="ia", endpoint="recomendar"):
        """Llama al Backend y pinta las recomendaciones del modelo SVD."""
        # Usamos el id_simulado del sidebar para pruebas con IA
        user_id_ia = id_simulado
        if not user_id_ia:
            st.info("Tu perfil no tiene un ID asociado para generar recomendaciones.")
            return

        try:
            resp_ia = requests.get(
                f"http://127.0.0.1:8000/{endpoint}/{user_id_ia}", params={"n": 8}
            )
            if resp_ia.status_code == 200:
                recomendaciones = resp_ia.json().get("recomendaciones", [])
                if recomendaciones:
                    cols_ia = st.columns(4)
                    for idx, rec in enumerate(recomendaciones):
                        with cols_ia[idx % 4]:
                            poster = rec.get("poster_path", "")
                            if poster and poster != "" and poster != "nan":
                                st.image(
                                    f"https://image.tmdb.org/t/p/w500{poster}",
                                    width="stretch",
                                )
                            else:
                                st.image(
                                    "https://via.placeholder.com/300x450.png?text=Sin+Poster",
                                    width="stretch",
                                )
                            titulo_rec = rec.get("titulo", "Sin Título")
                            if len(titulo_rec) > 30:
                                titulo_rec = titulo_rec[:27] + "..."
                            st.markdown(f"**{titulo_rec}**")
                            st.caption(
                                f" Predicción IA: {rec['predicted_rating']} / 5.0"
                            )
                            if st.button("Ver sinopsis", key=f"{key_prefix}_{idx}"):
                                st.toast(
                                    rec.get("overview", "Sin sinopsis disponible.")
                                )
                else:
                    st.info("No se encontraron recomendaciones para tu perfil.")
            elif resp_ia.status_code == 503:
                error_det = resp_ia.json().get("detail", "Error desconocido")
                st.warning(f"El Backend reporta: {error_det}")
            else:
                st.warning("No se pudieron obtener recomendaciones.")
        except requests.exceptions.ConnectionError:
            st.warning("No se pudo conectar con el Backend para recomendaciones.")

    # =====================================================================================
    # Contenido de una pestaña (recomendaciones + top rated + más vistos)
    # =====================================================================================
    def render_tab_content(df, search, is_movie=True, endpoint_ia="recomendar"):
        """Dibuja las 3 secciones dentro de una pestaña: IA, Top Rated, Más Visto."""
        prefix = "mov" if is_movie else "tv"
        date_col = "fecha_estreno" if is_movie else "first_air_date"

        # Si hay búsqueda activa, mostramos los resultados filtrados
        if search:
            st.subheader("Resultados de búsqueda")
            if not df.empty and "titulo" in df.columns:
                mask = df["titulo"].str.contains(search, case=False, na=False)
                resultados = df[mask]
                if not resultados.empty:
                    render_cards(
                        resultados,
                        limit=12,
                        key_prefix=f"{prefix}_search",
                        date_col=date_col,
                    )
                else:
                    st.info("No se encontraron resultados para tu búsqueda.")
            return

        # --- Sección 1: Recomendaciones IA ---
        st.subheader("Recomendado para ti")
        if is_movie:
            render_recomendaciones_ia(key_prefix=f"{prefix}_ia", endpoint=endpoint_ia)
        else:
            st.info(
                "Las recomendaciones de series están en desarrollo. De momento disfruta del catálogo."
            )

        st.divider()

        # --- Sección 2: Top mejor puntuadas ---
        st.subheader("Mejor puntuadas por la comunidad")
        if not df.empty and "vote_average" in df.columns and "vote_count" in df.columns:
            # Filtro mínimo de votos para que no salgan pelis con 1 voto y nota 10
            df_top = df[df["vote_count"] > 500].sort_values(
                by="vote_average", ascending=False
            )
            render_cards(df_top, limit=8, key_prefix=f"{prefix}_top", date_col=date_col)
        else:
            st.info("No hay datos suficientes para generar el ranking.")

        st.divider()

        # --- Sección 3: Lo más visto ---
        st.subheader("Lo más visto")
        if not df.empty and "vote_count" in df.columns:
            df_popular = df.sort_values(by="vote_count", ascending=False)
            render_cards(
                df_popular, limit=8, key_prefix=f"{prefix}_pop", date_col=date_col
            )
        else:
            st.info("No hay datos suficientes para generar lo más visto.")

    ###########################################################################################
    # --- PESTAÑAS PRINCIPALES ---
    ###########################################################################################

    tab_movies, tab_shows = st.tabs(["Películas", "Series"])

    with tab_movies:
        render_tab_content(
            df_movies, search_query, is_movie=True, endpoint_ia=endpoint_ia
        )

    with tab_shows:
        render_tab_content(
            df_shows, search_query, is_movie=False, endpoint_ia=endpoint_ia
        )
===
import streamlit as st
import requests
import datetime


# Validar si el usuario está logueado en Streamlit Session State
if "usuario_autenticado" not in st.session_state:
    st.session_state["usuario_autenticado"] = False
    st.session_state["usuario_actual"] = None

# ---- ESTILOS PERSONALIZADOS (Azul Marino y Oro) ----
st.markdown(
    """
    <style>
    /* Fondo y colores principales */
    .stApp {
        background-color: #001220;
        color: #f0f0f0;
    }
    
    /* Títulos en Dorado Premium (Más profundo) */
    h1, h2, h3, .stSubheader {
        color: #B8860B !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Texto normal */
    p, span, label {
        color: #e0e0e0 !important;
    }
    
    /* Estilo para los Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(0, 31, 63, 0.5);
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: transparent;
        border-radius: 5px;
        color: #f0f0f0;
        border: 1px solid transparent;
        transition: all 0.2s;
    }
    /* Tab seleccionado: Oro oscuro con texto azul muy oscuro para contraste */
    .stTabs [aria-selected="true"] {
        background-color: #B8860B !important;
        color: #001220 !important;
        border-color: #B8860B !important;
        font-weight: bold;
    }
    
    /* Botones Premium */
    div.stButton > button {
        background-color: #B8860B !important;
        color: #001220 !important;
        font-weight: bold !important;
        border: 1px solid #B8860B !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div.stButton > button:hover {
        background-color: #001f3f !important;
        color: #B8860B !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 20px rgba(184, 134, 11, 0.4) !important;
    }
    
    /* Inputs y widgets estilizados */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stMultiSelect>div>div>div, .stDateInput>div>div>input {
        background-color: #001f3f !important;
        color: white !important;
        border: 1px solid #1a3a5f !important;
        border-radius: 8px !important;
    }
    
    /* Checkboxes personalizados (Grid de gustos) */
    .stCheckbox > label > div[data-testid="stMarkdownContainer"] > p {
        color: #f0f0f0 !important;
        font-size: 0.9rem;
    }
    /* Estilo del cuadro del checkbox */
    span[data-baseweb="checkbox"] > div {
        border-color: #B8860B !important;
    }
    /* Checkbox marcado: fondo oro oscuro */
    div[data-checked="true"] {
        background-color: #B8860B !important;
    }
    
    /* Mensajes de información y éxito */
    .stAlert {
        border-radius: 10px !important;
        border: 1px solid #B8860B !important;
        background-color: rgba(0, 31, 63, 0.8) !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not st.session_state["usuario_autenticado"]:
    st.title("BIENVENIDO A SPIRE STREAMING")
    st.markdown(
        "Accede a la experiencia definitiva de cine o crea tu cuenta exclusiva."
    )

    tab_login, tab_registro = st.tabs(["Iniciar Sesión", "Registrarse"])

    # --- PESTAÑA LOGIN ---
    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            submit_button = st.form_submit_button("Entrar")

            if submit_button:
                payload = {"username": username, "password": password}
                try:
                    response = requests.post(
                        "http://localhost:8000/login", json=payload
                    )

                    if response.status_code == 200:
                        datos_usuario = response.json()
                        st.session_state["usuario_autenticado"] = True
                        st.session_state["usuario_actual"] = datos_usuario["user"]
                        st.success("¡Login exitoso!")
                        st.rerun()  # Recarga la página para mostrar el catálogo
                    elif response.status_code == 401:
                        st.error(
                            "Credenciales incorrectas. Verifica tu usuario o regístrate en la otra pestaña."
                        )
                    else:
                        st.error(f"Error en el servidor: {response.status_code}")
                except Exception as e:
                    st.error(f"Error de conexión: {e}")

    # --- PESTAÑA REGISTRO ---
    with tab_registro:
        st.subheader("Crea tu cuenta")

        # Intentar cargar géneros para el selector de gustos
        opciones_generos = {}
        try:
            resp_gen = requests.get("http://localhost:8000/genres")
            if resp_gen.status_code == 200:
                for g in resp_gen.json():
                    opciones_generos[g["name"]] = g["id"]
        except Exception as e:
            st.error(f"Error cargando categorías: {e}")

        with st.form("register_form"):
            col1, col2 = st.columns(2)
            with col1:
                new_username = st.text_input(
                    "Usuario", placeholder="Tu nombre de usuario"
                )
                new_email = st.text_input("Email", placeholder="ejemplo@correo.com")
                new_password = st.text_input("Contraseña", type="password")

            with col2:
                # Ajustamos el calendario para cubrir desde 1900
                new_fecha_nac = st.date_input(
                    "Fecha de Nacimiento",
                    value=None,
                    min_value=datetime.date(1900, 1, 1),
                    max_value=datetime.date.today(),
                )
                new_sexo = st.selectbox(
                    "Sexo",
                    ["Hombre", "Mujer", "Otro"],
                    index=None,
                    placeholder="Selecciona...",
                )
                new_password_confirm = st.text_input(
                    "Confirmar Contraseña", type="password"
                )

            st.write("---")
            st.markdown("**¿Qué generos te gusta ver?**")

            # --- REFACTOR: Grid de Checkboxes para Categorías ---
            if not opciones_generos:
                st.caption("Cargando categorías o servidor no disponible...")

            # Mostramos las categorías en un grid de 4 columnas para que quepa bien
            if opciones_generos:
                generos_lista = list(opciones_generos.keys())
                num_cols = 4
                rows = [
                    generos_lista[i : i + num_cols]
                    for i in range(0, len(generos_lista), num_cols)
                ]

                for row_genres in rows:
                    cols = st.columns(num_cols)
                    for i, genre_name in enumerate(row_genres):
                        # Usamos key para persistir el valor en session_state y capturarlo al enviar
                        cols[i].checkbox(genre_name, key=f"genre_{genre_name}")

            btn_register = st.form_submit_button("REGISTRARSE AHORA")

            if btn_register:
                # Recopilamos los gustos marcados desde el session_state
                gustos_actuales = [
                    opciones_generos[name]
                    for name in opciones_generos
                    if st.session_state.get(f"genre_{name}")
                ]

                if not new_username or not new_password or not new_email:
                    st.warning(
                        "Por favor, rellena los campos obligatorios (Usuario, Email y Contraseña)."
                    )
                elif new_password != new_password_confirm:
                    st.error("Las contraseñas no coinciden.")
                else:
                    # Preparar payload
                    payload = {
                        "username": new_username,
                        "email": new_email,
                        "password": new_password,
                        "fecha_nacimiento": str(new_fecha_nac)
                        if new_fecha_nac
                        else None,
                        "sexo": new_sexo,
                        "intereses": gustos_actuales,
                    }

                    try:
                        response = requests.post(
                            "http://127.0.0.1:8000/register", json=payload
                        )
                        if response.status_code == 200:
                            st.success(
                                f"¡Bienvenido, {new_username}! Tu cuenta ha sido creada."
                            )
                            st.info(
                                "Ya puedes ir a la pestaña 'Iniciar Sesión' para entrar."
                            )
                        else:
                            error_detail = response.json().get(
                                "detail", "Error desconocido"
                            )
                            st.error(f"Error al registrar: {error_detail}")
                    except Exception as e:
                        st.error(f"No se pudo conectar con el servidor: {e}")
#################################################################################################
# ---- PANTALLA PRINCIPAL DEL USUARIO (Logueado) ----

else:
    import pandas as pd
    import os

    usuario = st.session_state["usuario_actual"]
    nombre_mostrar = usuario.get("username", f"Usuario #{usuario.get('id_usuario')}")

    # Barra superior con Logout
    col_titulo, col_logout = st.columns([8, 2])
    with col_titulo:
        st.title(f"Bienvenido a SPIRE, {nombre_mostrar}!")
    with col_logout:
        st.write("")  # Espacio
        if st.button("Cerrar Sesión"):
            st.session_state["usuario_autenticado"] = False
            st.session_state["usuario_actual"] = None
            st.rerun()

    st.markdown(
        "Descubre Películas y Series gracias a nuestro Recomendador de Inteligencia Artificial."
    )

    # --- BUSCADOR GLOBAL ---
    search_query = st.text_input("Busca por título o palabras clave...")

    # --- SELECTOR DE MODELO DE IA ---
    modelo_ia = st.sidebar.selectbox(
        "Motor de Recomendación",
        [
            "SVD (Rápido)",
            "KNN + Cosine (Explicable)",
            "Wide & Deep (Profundo)",
            "Content-Based (Cold-Start)",
            "Implicit BPR (Ranking Top)",
            "NCF-Lite (Deep Learning)",
        ],
        index=4,  # Ponemos el BPR por defecto porque es el mejor
    )
    mapa_endpoints = {
        "SVD (Rápido)": "recomendar",
        "KNN + Cosine (Explicable)": "recomendar/knn",
        "Wide & Deep (Profundo)": "recomendar/wnd",
        "Content-Based (Cold-Start)": "recomendar/content",
        "Implicit BPR (Ranking Top)": "recomendar/implicit",
        "NCF-Lite (Deep Learning)": "recomendar/ncf",
    }
    endpoint_ia = mapa_endpoints[modelo_ia]

    # --- DEV TOOL: Simular otro usuario ---
    id_simulado = st.sidebar.number_input(
        "Datos ratings - ID Usuario ",
        value=usuario.get("id_usuario", 1),
        step=1,
        help="Permite simular predicciones para IDs de súper-usuarios (ej. 9) que existen en el set de datos pero no en tu base de datos local.",
    )

    # --- CARGA DE DATOS ---
    @st.cache_data
    def load_catalog_data():
        movies_path = "src/data/ready/dataset_final_movies.csv"
        shows_path = "src/data/ready/dataset_final_shows.csv"

        df_movies = pd.DataFrame()
        df_shows = pd.DataFrame()

        if os.path.exists(movies_path):
            try:
                df_movies = pd.read_csv(
                    movies_path, on_bad_lines="skip", engine="python"
                )
            except Exception:
                pass

        if os.path.exists(shows_path):
            try:
                df_shows = pd.read_csv(shows_path, on_bad_lines="skip", engine="python")
            except Exception:
                pass

        return df_movies, df_shows

    df_movies, df_shows = load_catalog_data()

    # =====================================================================================
    # Dibuja un grid de postales con poster, título y sinopsis
    # =====================================================================================
    def render_cards(df, limit=8, key_prefix="card", date_col="fecha_estreno"):
        """Dibuja una fila de postales con poster, título y botón de sinopsis."""
        if df.empty:
            st.info("No hay datos disponibles para mostrar.")
            return

        cols = st.columns(4)
        for idx, (_, row) in enumerate(df.head(limit).iterrows()):
            with cols[idx % 4]:
                # Poster
                poster_url = "https://via.placeholder.com/300x450.png?text=Sin+Poster"
                if (
                    pd.notna(row.get("poster_path"))
                    and str(row.get("poster_path")) != ""
                ):
                    poster_url = f"https://image.tmdb.org/t/p/w500{row['poster_path']}"
                st.image(poster_url, use_container_width=True)

                # Título truncado
                titulo = str(row.get("titulo", "Sin Título"))
                if len(titulo) > 30:
                    titulo = titulo[:27] + "..."

                # Año
                year = str(row.get(date_col, ""))[:4]
                if year and year != "nan":
                    st.markdown(f"**{titulo}** ({year})")
                else:
                    st.markdown(f"**{titulo}**")

                # Nota
                nota = row.get("vote_average", 0)
                if nota:
                    st.caption(f"{nota}")

                # Botón de sinopsis
                if st.button("Ver sinopsis", key=f"{key_prefix}_{idx}"):
                    st.toast(str(row.get("overview", "Sin sinopsis disponible.")))

    # =====================================================================================
    # Renderiza las recomendaciones IA dentro de una pestaña
    # =====================================================================================
    def render_recomendaciones_ia(key_prefix="ia", endpoint="recomendar"):
        """Llama al Backend y pinta las recomendaciones del modelo SVD."""
        # Usamos el id_simulado del sidebar para pruebas con IA
        user_id_ia = id_simulado
        if not user_id_ia:
            st.info("Tu perfil no tiene un ID asociado para generar recomendaciones.")
            return

        try:
            resp_ia = requests.get(
                f"http://127.0.0.1:8000/{endpoint}/{user_id_ia}", params={"n": 8}
            )
            if resp_ia.status_code == 200:
                recomendaciones = resp_ia.json().get("recomendaciones", [])
                if recomendaciones:
                    cols_ia = st.columns(4)
                    for idx, rec in enumerate(recomendaciones):
                        with cols_ia[idx % 4]:
                            poster = rec.get("poster_path", "")
                            if poster and poster != "" and poster != "nan":
                                st.image(
                                    f"https://image.tmdb.org/t/p/w500{poster}",
                                    use_container_width=True,
                                )
                            else:
                                st.image(
                                    "https://via.placeholder.com/300x450.png?text=Sin+Poster",
                                    use_container_width=True,
                                )
                            titulo_rec = rec.get("titulo", "Sin Título")
                            if len(titulo_rec) > 30:
                                titulo_rec = titulo_rec[:27] + "..."
                            st.markdown(f"**{titulo_rec}**")
                            st.caption(
                                f" Predicción IA: {rec['predicted_rating']} / 5.0"
                            )
                            if st.button("Ver sinopsis", key=f"{key_prefix}_{idx}"):
                                st.toast(
                                    rec.get("overview", "Sin sinopsis disponible.")
                                )
                else:
                    st.info("No se encontraron recomendaciones para tu perfil.")
            elif resp_ia.status_code == 503:
                error_det = resp_ia.json().get("detail", "Error desconocido")
                st.warning(f"El Backend reporta: {error_det}")
            else:
                st.warning("No se pudieron obtener recomendaciones.")
        except requests.exceptions.ConnectionError:
            st.warning("No se pudo conectar con el Backend para recomendaciones.")

    # =====================================================================================
    # Contenido de una pestaña (recomendaciones + top rated + más vistos)
    # =====================================================================================
    def render_tab_content(df, search, is_movie=True, endpoint_ia="recomendar"):
        """Dibuja las 3 secciones dentro de una pestaña: IA, Top Rated, Más Visto."""
        prefix = "mov" if is_movie else "tv"
        date_col = "fecha_estreno" if is_movie else "first_air_date"

        # Si hay búsqueda activa, mostramos los resultados filtrados
        if search:
            st.subheader("Resultados de búsqueda")
            if not df.empty and "titulo" in df.columns:
                mask = df["titulo"].str.contains(search, case=False, na=False)
                resultados = df[mask]
                if not resultados.empty:
                    render_cards(
                        resultados,
                        limit=12,
                        key_prefix=f"{prefix}_search",
                        date_col=date_col,
                    )
                else:
                    st.info("No se encontraron resultados para tu búsqueda.")
            return

        # --- Sección 1: Recomendaciones IA ---
        st.subheader("Recomendado para ti")
        if is_movie:
            render_recomendaciones_ia(key_prefix=f"{prefix}_ia", endpoint=endpoint_ia)
        else:
            st.info(
                "Las recomendaciones de series están en desarrollo. De momento disfruta del catálogo."
            )

        st.divider()

        # --- Sección 2: Top mejor puntuadas ---
        st.subheader("Mejor puntuadas por la comunidad")
        if not df.empty and "vote_average" in df.columns and "vote_count" in df.columns:
            # Filtro mínimo de votos para que no salgan pelis con 1 voto y nota 10
            df_top = df[df["vote_count"] > 500].sort_values(
                by="vote_average", ascending=False
            )
            render_cards(df_top, limit=8, key_prefix=f"{prefix}_top", date_col=date_col)
        else:
            st.info("No hay datos suficientes para generar el ranking.")

        st.divider()

        # --- Sección 3: Lo más visto ---
        st.subheader("Lo más visto")
        if not df.empty and "vote_count" in df.columns:
            df_popular = df.sort_values(by="vote_count", ascending=False)
            render_cards(
                df_popular, limit=8, key_prefix=f"{prefix}_pop", date_col=date_col
            )
        else:
            st.info("No hay datos suficientes para generar lo más visto.")

    ###########################################################################################
    # --- PESTAÑAS PRINCIPALES ---
    ###########################################################################################

    tab_movies, tab_shows = st.tabs(["Películas", "Series"])

    with tab_movies:
        render_tab_content(
            df_movies, search_query, is_movie=True, endpoint_ia=endpoint_ia
        )

    with tab_shows:
        render_tab_content(
            df_shows, search_query, is_movie=False, endpoint_ia=endpoint_ia
        )
```

- Añadido `"NCF-Lite (Deep Learning)"` al sidebar selectbox
- Mapeado a endpoint `recomendar/ncf`

### Evaluador
```diff:evaluacion_ranking.py
"""
#######################################################################################
# SCRIPT DE EVALUACIÓN DE RANKING (NDCG K, Precision K, Hit Rate)
# =======================================================================================
# Compara los modelos (SVD, KNN, W&D, Content-Based, LightFM) sobre un conjunto de usuarios.
# Oculta películas que el usuario ha valorado positivamente (relevantes) y comprueba
# si el modelo logra sugerirlas en su Top 10.
#######################################################################################
"""

import pandas as pd
import numpy as np
import pickle
import os
import torch
import math

############################################################################################

# ---- Rutas ----
RUTA_CATALOGO = "src/data/ready/dataset_final_movies.csv"
RUTA_RATINGS = "src/data/ready/ratings_finales_ia.csv"

# Modelos
RUTA_SVD = "src/models/jj/modelo_1_SVD.pkl"
RUTA_KNN = "src/models/jj/modelo_2_knn_cs.pkl"
RUTA_WND_PTH = "src/models/jj/modelo_3_wnd.pth"
RUTA_WND_MAP = "src/models/jj/wnd_mappings.pkl"
RUTA_TFIDF_MOD = "src/models/jj/modelo_4_tfidf.pkl"
RUTA_TFIDF_MAT = "src/models/jj/modelo_4_matriz.pkl"
RUTA_TFIDF_IDX = "src/models/jj/modelo_4_indices.pkl"
RUTA_IMP_MOD = "src/models/jj/modelo_5_implicit.pkl"
RUTA_IMP_DAT = "src/models/jj/modelo_5_implicit_dataset.pkl"

# Guardar Resultados
RUTA_RESULTADOS = "src/utils/metricas_ranking.csv"

# Configuración
K = 10  # Queremos evaluar el Top-10
NUM_USUARIOS = 300  # Aumentado a 300 usuarios para dar rigor estadístico al paper
UMBRAL_RELEVANTE = (
    4.0  # Una película es 'Relevante' si el usuario le dio >= 4 estrellas
)

############################################################################################

# ---- FUNCIONES DE MÉTRICAS MATEMÁTICAS ----


def precision_at_k(recomendadas, relevantes):
    """Porcentaje de recomendaciones (Top K) que son realmente películas relevantes."""
    if not recomendadas:
        return 0.0
    aciertos = len(set(recomendadas) & set(relevantes))
    return aciertos / len(recomendadas)


def recall_at_k(recomendadas, relevantes):
    """De todas las películas relevantes, ¿qué % logró capturar el Top K?"""
    if not relevantes:
        return 0.0
    aciertos = len(set(recomendadas) & set(relevantes))
    return aciertos / len(relevantes)


def coverage(recs_totales, n_catalogo):
    """Porcentaje del catálogo total que nuestro sistema es capaz de recomendar."""
    if n_catalogo == 0:
        return 0.0
    unicas_recomendadas = set()
    for lista in recs_totales:
        unicas_recomendadas.update(lista)
    return len(unicas_recomendadas) / n_catalogo


############################################################################################


def hit_rate(recomendadas, relevantes):
    """1 = si al menos recomendó 1 película relevante, 0 = si no acertó ni una."""
    return 1 if len(set(recomendadas) & set(relevantes)) > 0 else 0


############################################################################################


def ndcg_at_k(recomendadas, relevantes_escala):
    """
    Normalized Discounted Cumulative Gain.
    Recompensamos mas si el acierto ocurre en el Top 1 que en el Top 10.
    relevantes_escala es un dict: {tmdb_id: nota_real}
    """
    dcg = 0.0
    idcg = 0.0

    # Calcular DCG
    for i, peli in enumerate(recomendadas):
        if peli in relevantes_escala:
            relevancia = relevantes_escala[peli]  # Ej. 4.5 o 5.0
            dcg += relevancia / math.log2(i + 2)  # Penalización de posición

    # Calcular IDCG (El DCG Ideal si lo hubiese ordenado perfectamente)
    ideal_relevancias = sorted(list(relevantes_escala.values()), reverse=True)
    for i, rel in enumerate(ideal_relevancias[: len(recomendadas)]):
        idcg += rel / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


############################################################################################
# ---- LÓGICA PRINCIPAL ----


def cargar_modelos():
    print("=" * 60)
    print("  Cargando modelos de IA...")
    print("=" * 60)

    modelos = {}
    # SVD
    if os.path.exists(RUTA_SVD):
        with open(RUTA_SVD, "rb") as f:
            modelos["SVD"] = pickle.load(f)

    # KNN
    if os.path.exists(RUTA_KNN):
        with open(RUTA_KNN, "rb") as f:
            modelos["KNN"] = pickle.load(f)

    # Content-Based (TF-IDF)
    if os.path.exists(RUTA_TFIDF_MAT):
        with open(RUTA_TFIDF_MAT, "rb") as f:
            modelos["TFIDF_MAT"] = pickle.load(f)
        with open(RUTA_TFIDF_IDX, "rb") as f:
            modelos["TFIDF_IDX"] = pickle.load(f)

    # Wide&Deep
    if os.path.exists(RUTA_WND_PTH):
        try:
            import sys

            sys.path.insert(
                0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            )
            from src.networks.dl.rn_mlp import WideAndDeepModel

            with open(RUTA_WND_MAP, "rb") as f:
                wnd_maps = pickle.load(f)
            wnd = WideAndDeepModel(
                len(wnd_maps["user2idx"]), len(wnd_maps["movie2idx"]), 32, [64, 32]
            )
            wnd.load_state_dict(
                torch.load(RUTA_WND_PTH, map_location="cpu", weights_only=True)
            )
            wnd.eval()
            modelos["WND"] = wnd
            modelos["WND_MAPS"] = wnd_maps
        except Exception as e:
            print(f"  Error cargando W&D: {e}")

    # Implicit BPR
    if os.path.exists(RUTA_IMP_MOD) and os.path.exists(RUTA_IMP_DAT):
        try:
            with open(RUTA_IMP_MOD, "rb") as f:
                modelos["IMP"] = pickle.load(f)
            with open(RUTA_IMP_DAT, "rb") as f:
                modelos["IMP_DAT"] = pickle.load(f)
        except Exception as e:
            print(f"  Error cargando Implicit: {e}")

    print(f"  Modelos listos: {list(modelos.keys())}")
    return modelos


############################################################################################


def predecir_svd_knn(modelo, user_id, candidatas):
    preds = []
    for tmdb_id in candidatas:
        pred = modelo.predict(user_id, tmdb_id)
        preds.append((tmdb_id, pred.est))
    preds.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in preds[:K]]


############################################################################################


def predecir_wnd(modelo, mapas, user_id, candidatas):
    u2idx = mapas["user2idx"]
    m2idx = mapas["movie2idx"]

    if user_id not in u2idx:
        return []
    u_idx = u2idx[user_id]

    candidatas_validas = [(tid, m2idx[tid]) for tid in candidatas if tid in m2idx]
    if not candidatas_validas:
        return []

    tids, m_idxs = zip(*candidatas_validas)
    t_users = torch.tensor([u_idx] * len(m_idxs), dtype=torch.long)
    t_items = torch.tensor(list(m_idxs), dtype=torch.long)

    with torch.no_grad():
        # NO usamos torch.clamp(..., 0.5, 5.0) aquí porque destruye el orden relativo
        # de las películas favoritas si muchas de ellas se saturan al tope de la escala.
        # Para ranking, nos da igual que prediga 5.3 o 6.8, solo nos importa quién es mayor.
        preds = modelo(t_users, t_items)

    pares = [(tids[i], preds[i].item()) for i in range(len(tids))]
    pares.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in pares[:K]]


############################################################################################


def predecir_content(matriz, indices, df_movies, user_ratings_vistas, candidatas):
    from sklearn.metrics.pairwise import linear_kernel

    # Película con nota más alta del historial de "vistas"
    if user_ratings_vistas.empty:
        return []
    fav = user_ratings_vistas.sort_values(by="rating", ascending=False).iloc[0]
    tid_fav = int(fav["tmdb_id"])

    if tid_fav not in indices:
        return []
    idx_fav = indices[tid_fav]

    # Simulamos Similitud en bloque
    vector_fav = matriz[idx_fav]
    similitudes = linear_kernel(vector_fav, matriz).flatten()

    # Filtrar solo candidatos
    preds_sims = []
    for tid in candidatas:
        if tid in indices:
            preds_sims.append((tid, similitudes[indices[tid]]))

    preds_sims.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in preds_sims[:K]]


def predecir_implicit(modelo, dataset, user_id, test_vistas_ratings, candidatas):
    """
    Predicción usando el modelo de la librería `implicit`.
    """
    user2idx = dataset["user2idx"]
    item2idx = dataset["item2idx"]
    
    if user_id not in user2idx:
        return []
        
    u_idx = user2idx[user_id]
    
    # Preparamos las candidatas soportadas (que están en el catálogo entrenado)
    cands_validas = [(tid, item2idx[tid]) for tid in candidatas if tid in item2idx]
    if not cands_validas:
        return []
        
    # Implicit tiene el parámetro `item_idxs` en `recommend` en sus versiones recientes,
    # pero para mayor seguridad, vamos a usar `recommend` limitando o extrayendo a mano los scores
    # O, lo más estándar: pedir muchas y filtrar. Pero `predict` es más fácil:
    # `scores = model.user_factors[u] @ model.item_factors.T`
    
    tids, m_idxs = zip(*cands_validas)
    
    # En implicit, predecir el score es producto escalar del factor de usuario con el factor de item
    u_factors = np.asarray(modelo.user_factors)[u_idx]
    
    # Si las dimensiones son compatibles:
    if hasattr(modelo, "item_factors"):
        i_factors = np.asarray(modelo.item_factors)[list(m_idxs)]
        # Producto escalar para obtener los scores brutos
        scores = np.dot(i_factors, u_factors)
        
        pares = list(zip(tids, scores))
        pares.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in pares[:K]]
    else:
        return []


############################################################################################


def evaluar():
    print("\n" + "=" * 60)
    print(f"  Evaluacion del ranking (Top {K})")
    print("=" * 60)

    modelos = cargar_modelos()

    df = pd.read_csv(RUTA_RATINGS)
    df_movies = pd.read_csv(RUTA_CATALOGO)

    # Método Netflix:
    # Filtramos el catálogo candidato a películas con más de 50 votos globales.
    # ¿Por qué? Las Redes Neuronales (W&D) inicializan sus pesos (Embeddings) al azar.
    # Si una película tiene solo 1 o 2 votos, la red no la entrena, y su peso aleatorio puede
    # quedar inflado, provocando que recomiende "basura desconocida" en el Top 10 y saque 0% de acierto.
    if "vote_count" in df_movies.columns:
        pelis_conocidas = df_movies[df_movies["vote_count"] > 50]
        todas_pelis = set(pelis_conocidas["tmdb_id"].unique())
    else:
        todas_pelis = set(df_movies["tmdb_id"].unique())

    # 1. Muestrear usuarios
    conteos = df.groupby("userId").size()
    u_abundantes = conteos[conteos > 30].index.tolist()
    u_evaluar = np.random.choice(u_abundantes, NUM_USUARIOS, replace=False)

    # Inicializamos contadores por cada modelo
    # 'p' = precision, 'r' = recall, 'n' = ndcg, 'h' = hit_rate, 'recs' = lista todas recs (para coverage)
    metricas_base = {"p": 0, "r": 0, "n": 0, "h": 0, "recs": []}
    resultados = {
        "SVD": metricas_base.copy() if "SVD" in modelos else None,
        "KNN": metricas_base.copy() if "KNN" in modelos else None,
        "WND": metricas_base.copy() if "WND" in modelos else None,
        "TFIDF": metricas_base.copy() if "TFIDF_MAT" in modelos else None,
        "IMP": metricas_base.copy() if "IMP" in modelos else None,
    }
    # Reset lists inside dicts (importante por .copy() superficial)
    for k in resultados:
        if resultados[k] is not None:
            resultados[k] = {"p": 0, "r": 0, "n": 0, "h": 0, "recs": []}

    n_usuarios_final = 0

    print(f"\n  Iniciando evaluación sobre {NUM_USUARIOS} usuarios...")

    for u in u_evaluar:
        user_data = df[df["userId"] == u]

        # Ocultamos artificialmente el 20% de sus películas altas (Relevantes)
        relevantes = user_data[user_data["rating"] >= UMBRAL_RELEVANTE]
        if len(relevantes) < 5:
            continue  # Muy pocas favoritas para testear

        test_oculto = relevantes.sample(frac=0.2, random_state=42)
        vistas = user_data.drop(test_oculto.index)

        # El conjunto "Verdadero (Ground Truth)" para el test
        dict_relevantes = {
            row["tmdb_id"]: row["rating"] for _, row in test_oculto.iterrows()
        }
        pelis_relevantes = list(dict_relevantes.keys())

        # Conjunto Candidato: Todas las pelis menos las "Vistas"
        # Incluimos las ocultas para ver si las pilla
        candidatas = todas_pelis - set(vistas["tmdb_id"].unique())

        # Evaluate SVD
        top_svd = []
        if resultados["SVD"] is not None:
            top_svd = predecir_svd_knn(modelos["SVD"], u, candidatas)
            resultados["SVD"]["p"] += precision_at_k(top_svd, pelis_relevantes)
            resultados["SVD"]["r"] += recall_at_k(top_svd, pelis_relevantes)
            resultados["SVD"]["n"] += ndcg_at_k(top_svd, dict_relevantes)
            resultados["SVD"]["h"] += hit_rate(top_svd, pelis_relevantes)
            resultados["SVD"]["recs"].append(top_svd)

        # Evaluate KNN
        top_knn = []
        if resultados["KNN"] is not None:
            top_knn = predecir_svd_knn(modelos["KNN"], u, candidatas)
            resultados["KNN"]["p"] += precision_at_k(top_knn, pelis_relevantes)
            resultados["KNN"]["r"] += recall_at_k(top_knn, pelis_relevantes)
            resultados["KNN"]["n"] += ndcg_at_k(top_knn, dict_relevantes)
            resultados["KNN"]["h"] += hit_rate(top_knn, pelis_relevantes)
            resultados["KNN"]["recs"].append(top_knn)

        # Evaluate W&D
        top_wnd = []
        if resultados["WND"] is not None:
            top_wnd = predecir_wnd(modelos["WND"], modelos["WND_MAPS"], u, candidatas)
            if top_wnd:
                resultados["WND"]["p"] += precision_at_k(top_wnd, pelis_relevantes)
                resultados["WND"]["r"] += recall_at_k(top_wnd, pelis_relevantes)
                resultados["WND"]["n"] += ndcg_at_k(top_wnd, dict_relevantes)
                resultados["WND"]["h"] += hit_rate(top_wnd, pelis_relevantes)
                resultados["WND"]["recs"].append(top_wnd)

        # Evaluate Content-Based
        if "TFIDF_MAT" in modelos:
            top_tf = predecir_content(
                modelos["TFIDF_MAT"],
                modelos["TFIDF_IDX"],
                df_movies,
                vistas,
                candidatas,
            )
            resultados["TFIDF"]["p"] += precision_at_k(top_tf, pelis_relevantes)
            resultados["TFIDF"]["r"] += recall_at_k(top_tf, pelis_relevantes)
            resultados["TFIDF"]["n"] += ndcg_at_k(top_tf, dict_relevantes)
            resultados["TFIDF"]["h"] += hit_rate(top_tf, pelis_relevantes)
            resultados["TFIDF"]["recs"].append(top_tf)

        # Evaluate Implicit BPR
        top_imp = []
        if resultados["IMP"] is not None:
            top_imp = predecir_implicit(
                modelos["IMP"], modelos["IMP_DAT"], u, vistas, candidatas
            )
            if top_imp:
                resultados["IMP"]["p"] += precision_at_k(top_imp, pelis_relevantes)
                resultados["IMP"]["r"] += recall_at_k(top_imp, pelis_relevantes)
                resultados["IMP"]["n"] += ndcg_at_k(top_imp, dict_relevantes)
                resultados["IMP"]["h"] += hit_rate(top_imp, pelis_relevantes)
                resultados["IMP"]["recs"].append(top_imp)

        # Solo si procesamos un usuario válido, aumentamos el contador
        if (
            top_svd
            or top_knn
            or top_wnd
            or top_tf
            or (resultados.get("IMP") and top_imp)
        ):
            n_usuarios_final += 1

    # Convertir sumas a medias --> Dividimos entre N usuarios testados válidos
    n_catalogo = len(todas_pelis)
    print("\n" + "=" * 80)
    print(
        f"  RESULTADOS GLOBALES (Promedios para Top-{K} sobre {n_usuarios_final} usuarios):"
    )
    print("=" * 80)

    records = []
    for mod, vals in resultados.items():
        if vals is None or not vals["recs"]:
            continue

        prec = vals["p"] / n_usuarios_final
        rec = vals["r"] / n_usuarios_final
        ndcg = vals["n"] / n_usuarios_final
        hr = vals["h"] / n_usuarios_final
        cov = coverage(vals["recs"], n_catalogo)

        records.append(
            {
                "Modelo": mod,
                f"Precision_{K}": prec,
                f"Recall_{K}": rec,
                f"NDCG_{K}": ndcg,
                f"Hit_Rate_{K}": hr,
                f"Coverage_{K}": cov,
            }
        )
        print(
            f"  -> {mod:<8} | Precision: {prec * 100:4.1f}% | Recall: {rec * 100:4.1f}% | "
            f"NDCG: {ndcg:.4f} | HR: {hr * 100:4.1f}% | Cov: {cov * 100:4.1f}%"
        )

    # Guardar a CSV
    df_res = pd.DataFrame(records)
    df_res.to_csv(RUTA_RESULTADOS, index=False)
    print("\n  Reporte de métricas guardado en:", RUTA_RESULTADOS)
    print("=" * 60)


############################################################################################

if __name__ == "__main__":
    evaluar()
===
"""
#######################################################################################
# SCRIPT DE EVALUACIÓN DE RANKING (NDCG K, Precision K, Hit Rate)
# =======================================================================================
# Compara los modelos (SVD, KNN, W&D, Content-Based, LightFM) sobre un conjunto de usuarios.
# Oculta películas que el usuario ha valorado positivamente (relevantes) y comprueba
# si el modelo logra sugerirlas en su Top 10.
#######################################################################################
"""

import pandas as pd
import numpy as np
import pickle
import os
import torch
import math

############################################################################################

# ---- Rutas ----
RUTA_CATALOGO = "src/data/ready/dataset_final_movies.csv"
RUTA_RATINGS = "src/data/ready/ratings_finales_ia.csv"

# Modelos
RUTA_SVD = "src/models/jj/modelo_1_SVD.pkl"
RUTA_KNN = "src/models/jj/modelo_2_knn_cs.pkl"
RUTA_WND_PTH = "src/models/jj/modelo_3_wnd.pth"
RUTA_WND_MAP = "src/models/jj/wnd_mappings.pkl"
RUTA_TFIDF_MOD = "src/models/jj/modelo_4_tfidf.pkl"
RUTA_TFIDF_MAT = "src/models/jj/modelo_4_matriz.pkl"
RUTA_TFIDF_IDX = "src/models/jj/modelo_4_indices.pkl"
RUTA_IMP_MOD = "src/models/jj/modelo_5_implicit.pkl"
RUTA_IMP_DAT = "src/models/jj/modelo_5_implicit_dataset.pkl"
RUTA_NCF_ONNX = "src/models/jj/modelo_6_ncf.onnx"
RUTA_NCF_USER2IDX = "src/models/jj/ncf_user2idx.json"
RUTA_NCF_ITEM2IDX = "src/models/jj/ncf_item2idx.json"

# Guardar Resultados
RUTA_RESULTADOS = "src/utils/metricas_ranking.csv"

# Configuración
K = 10  # Queremos evaluar el Top-10
NUM_USUARIOS = 300  # Aumentado a 300 usuarios para dar rigor estadístico al paper
UMBRAL_RELEVANTE = (
    4.0  # Una película es 'Relevante' si el usuario le dio >= 4 estrellas
)

############################################################################################

# ---- FUNCIONES DE MÉTRICAS MATEMÁTICAS ----


def precision_at_k(recomendadas, relevantes):
    """Porcentaje de recomendaciones (Top K) que son realmente películas relevantes."""
    if not recomendadas:
        return 0.0
    aciertos = len(set(recomendadas) & set(relevantes))
    return aciertos / len(recomendadas)


def recall_at_k(recomendadas, relevantes):
    """De todas las películas relevantes, ¿qué % logró capturar el Top K?"""
    if not relevantes:
        return 0.0
    aciertos = len(set(recomendadas) & set(relevantes))
    return aciertos / len(relevantes)


def coverage(recs_totales, n_catalogo):
    """Porcentaje del catálogo total que nuestro sistema es capaz de recomendar."""
    if n_catalogo == 0:
        return 0.0
    unicas_recomendadas = set()
    for lista in recs_totales:
        unicas_recomendadas.update(lista)
    return len(unicas_recomendadas) / n_catalogo


############################################################################################


def hit_rate(recomendadas, relevantes):
    """1 = si al menos recomendó 1 película relevante, 0 = si no acertó ni una."""
    return 1 if len(set(recomendadas) & set(relevantes)) > 0 else 0


############################################################################################


def ndcg_at_k(recomendadas, relevantes_escala):
    """
    Normalized Discounted Cumulative Gain.
    Recompensamos mas si el acierto ocurre en el Top 1 que en el Top 10.
    relevantes_escala es un dict: {tmdb_id: nota_real}
    """
    dcg = 0.0
    idcg = 0.0

    # Calcular DCG
    for i, peli in enumerate(recomendadas):
        if peli in relevantes_escala:
            relevancia = relevantes_escala[peli]  # Ej. 4.5 o 5.0
            dcg += relevancia / math.log2(i + 2)  # Penalización de posición

    # Calcular IDCG (El DCG Ideal si lo hubiese ordenado perfectamente)
    ideal_relevancias = sorted(list(relevantes_escala.values()), reverse=True)
    for i, rel in enumerate(ideal_relevancias[: len(recomendadas)]):
        idcg += rel / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


############################################################################################
# ---- LÓGICA PRINCIPAL ----


def cargar_modelos():
    print("=" * 60)
    print("  Cargando modelos de IA...")
    print("=" * 60)

    modelos = {}
    # SVD
    if os.path.exists(RUTA_SVD):
        with open(RUTA_SVD, "rb") as f:
            modelos["SVD"] = pickle.load(f)

    # KNN
    if os.path.exists(RUTA_KNN):
        with open(RUTA_KNN, "rb") as f:
            modelos["KNN"] = pickle.load(f)

    # Content-Based (TF-IDF)
    if os.path.exists(RUTA_TFIDF_MAT):
        with open(RUTA_TFIDF_MAT, "rb") as f:
            modelos["TFIDF_MAT"] = pickle.load(f)
        with open(RUTA_TFIDF_IDX, "rb") as f:
            modelos["TFIDF_IDX"] = pickle.load(f)

    # Wide&Deep
    if os.path.exists(RUTA_WND_PTH):
        try:
            import sys

            sys.path.insert(
                0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            )
            from src.networks.dl.rn_mlp import WideAndDeepModel

            with open(RUTA_WND_MAP, "rb") as f:
                wnd_maps = pickle.load(f)
            wnd = WideAndDeepModel(
                len(wnd_maps["user2idx"]), len(wnd_maps["movie2idx"]), 32, [64, 32]
            )
            wnd.load_state_dict(
                torch.load(RUTA_WND_PTH, map_location="cpu", weights_only=True)
            )
            wnd.eval()
            modelos["WND"] = wnd
            modelos["WND_MAPS"] = wnd_maps
        except Exception as e:
            print(f"  Error cargando W&D: {e}")

    # Implicit BPR
    if os.path.exists(RUTA_IMP_MOD) and os.path.exists(RUTA_IMP_DAT):
        try:
            with open(RUTA_IMP_MOD, "rb") as f:
                modelos["IMP"] = pickle.load(f)
            with open(RUTA_IMP_DAT, "rb") as f:
                modelos["IMP_DAT"] = pickle.load(f)
        except Exception as e:
            print(f"  Error cargando Implicit: {e}")

    # NCF (ONNX Runtime)
    if os.path.exists(RUTA_NCF_ONNX):
        try:
            import onnxruntime as ort
            import json
            modelos["NCF"] = ort.InferenceSession(RUTA_NCF_ONNX)
            with open(RUTA_NCF_USER2IDX, "r") as f:
                modelos["NCF_USER2IDX"] = {int(k): v for k, v in json.load(f).items()}
            with open(RUTA_NCF_ITEM2IDX, "r") as f:
                modelos["NCF_ITEM2IDX"] = {int(k): v for k, v in json.load(f).items()}
            print(f"  NCF ONNX cargado.")
        except Exception as e:
            print(f"  Error cargando NCF: {e}")

    print(f"  Modelos listos: {list(modelos.keys())}")
    return modelos


############################################################################################


def predecir_svd_knn(modelo, user_id, candidatas):
    preds = []
    for tmdb_id in candidatas:
        pred = modelo.predict(user_id, tmdb_id)
        preds.append((tmdb_id, pred.est))
    preds.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in preds[:K]]


############################################################################################


def predecir_wnd(modelo, mapas, user_id, candidatas):
    u2idx = mapas["user2idx"]
    m2idx = mapas["movie2idx"]

    if user_id not in u2idx:
        return []
    u_idx = u2idx[user_id]

    candidatas_validas = [(tid, m2idx[tid]) for tid in candidatas if tid in m2idx]
    if not candidatas_validas:
        return []

    tids, m_idxs = zip(*candidatas_validas)
    t_users = torch.tensor([u_idx] * len(m_idxs), dtype=torch.long)
    t_items = torch.tensor(list(m_idxs), dtype=torch.long)

    with torch.no_grad():
        # NO usamos torch.clamp(..., 0.5, 5.0) aquí porque destruye el orden relativo
        # de las películas favoritas si muchas de ellas se saturan al tope de la escala.
        # Para ranking, nos da igual que prediga 5.3 o 6.8, solo nos importa quién es mayor.
        preds = modelo(t_users, t_items)

    pares = [(tids[i], preds[i].item()) for i in range(len(tids))]
    pares.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in pares[:K]]


############################################################################################


def predecir_content(matriz, indices, df_movies, user_ratings_vistas, candidatas):
    from sklearn.metrics.pairwise import linear_kernel

    # Película con nota más alta del historial de "vistas"
    if user_ratings_vistas.empty:
        return []
    fav = user_ratings_vistas.sort_values(by="rating", ascending=False).iloc[0]
    tid_fav = int(fav["tmdb_id"])

    if tid_fav not in indices:
        return []
    idx_fav = indices[tid_fav]

    # Simulamos Similitud en bloque
    vector_fav = matriz[idx_fav]
    similitudes = linear_kernel(vector_fav, matriz).flatten()

    # Filtrar solo candidatos
    preds_sims = []
    for tid in candidatas:
        if tid in indices:
            preds_sims.append((tid, similitudes[indices[tid]]))

    preds_sims.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in preds_sims[:K]]


def predecir_implicit(modelo, dataset, user_id, test_vistas_ratings, candidatas):
    """
    Predicción usando el modelo de la librería `implicit`.
    """
    user2idx = dataset["user2idx"]
    item2idx = dataset["item2idx"]
    
    if user_id not in user2idx:
        return []
        
    u_idx = user2idx[user_id]
    
    # Preparamos las candidatas soportadas (que están en el catálogo entrenado)
    cands_validas = [(tid, item2idx[tid]) for tid in candidatas if tid in item2idx]
    if not cands_validas:
        return []
        
    # Implicit tiene el parámetro `item_idxs` en `recommend` en sus versiones recientes,
    # pero para mayor seguridad, vamos a usar `recommend` limitando o extrayendo a mano los scores
    # O, lo más estándar: pedir muchas y filtrar. Pero `predict` es más fácil:
    # `scores = model.user_factors[u] @ model.item_factors.T`
    
    tids, m_idxs = zip(*cands_validas)
    
    # En implicit, predecir el score es producto escalar del factor de usuario con el factor de item
    u_factors = np.asarray(modelo.user_factors)[u_idx]
    
    # Si las dimensiones son compatibles:
    if hasattr(modelo, "item_factors"):
        i_factors = np.asarray(modelo.item_factors)[list(m_idxs)]
        # Producto escalar para obtener los scores brutos
        scores = np.dot(i_factors, u_factors)
        
        pares = list(zip(tids, scores))
        pares.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in pares[:K]]
    else:
        return []


def predecir_ncf(modelo_onnx, user2idx, item2idx, user_id, candidatas):
    """
    Predicción NCF via ONNX Runtime: puntua cada candidata y devuelve Top-K.
    """
    if user_id not in user2idx:
        return []

    u_idx = user2idx[user_id]

    # Filtrar candidatas que existan en el vocabulario NCF
    cands_validas = [(tid, item2idx[tid]) for tid in candidatas if tid in item2idx]
    if not cands_validas:
        return []

    tids, m_idxs = zip(*cands_validas)
    user_ids_np = np.full(len(m_idxs), u_idx, dtype=np.int64)
    item_ids_np = np.array(list(m_idxs), dtype=np.int64)

    scores = modelo_onnx.run(None, {"user_ids": user_ids_np, "item_ids": item_ids_np})[0]

    pares = list(zip(tids, scores.tolist()))
    pares.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in pares[:K]]


############################################################################################


def evaluar():
    print("\n" + "=" * 60)
    print(f"  Evaluacion del ranking (Top {K})")
    print("=" * 60)

    modelos = cargar_modelos()

    df = pd.read_csv(RUTA_RATINGS)
    df_movies = pd.read_csv(RUTA_CATALOGO)

    # Método Netflix:
    # Filtramos el catálogo candidato a películas con más de 50 votos globales.
    # ¿Por qué? Las Redes Neuronales (W&D) inicializan sus pesos (Embeddings) al azar.
    # Si una película tiene solo 1 o 2 votos, la red no la entrena, y su peso aleatorio puede
    # quedar inflado, provocando que recomiende "basura desconocida" en el Top 10 y saque 0% de acierto.
    if "vote_count" in df_movies.columns:
        pelis_conocidas = df_movies[df_movies["vote_count"] > 50]
        todas_pelis = set(pelis_conocidas["tmdb_id"].unique())
    else:
        todas_pelis = set(df_movies["tmdb_id"].unique())

    # 1. Muestrear usuarios
    conteos = df.groupby("userId").size()
    u_abundantes = conteos[conteos > 30].index.tolist()
    u_evaluar = np.random.choice(u_abundantes, NUM_USUARIOS, replace=False)

    # Inicializamos contadores por cada modelo
    # 'p' = precision, 'r' = recall, 'n' = ndcg, 'h' = hit_rate, 'recs' = lista todas recs (para coverage)
    metricas_base = {"p": 0, "r": 0, "n": 0, "h": 0, "recs": []}
    resultados = {
        "SVD": metricas_base.copy() if "SVD" in modelos else None,
        "KNN": metricas_base.copy() if "KNN" in modelos else None,
        "WND": metricas_base.copy() if "WND" in modelos else None,
        "TFIDF": metricas_base.copy() if "TFIDF_MAT" in modelos else None,
        "IMP": metricas_base.copy() if "IMP" in modelos else None,
        "NCF": metricas_base.copy() if "NCF" in modelos else None,
    }
    # Reset lists inside dicts (importante por .copy() superficial)
    for k in resultados:
        if resultados[k] is not None:
            resultados[k] = {"p": 0, "r": 0, "n": 0, "h": 0, "recs": []}

    n_usuarios_final = 0

    print(f"\n  Iniciando evaluación sobre {NUM_USUARIOS} usuarios...")

    for u in u_evaluar:
        user_data = df[df["userId"] == u]

        # Ocultamos artificialmente el 20% de sus películas altas (Relevantes)
        relevantes = user_data[user_data["rating"] >= UMBRAL_RELEVANTE]
        if len(relevantes) < 5:
            continue  # Muy pocas favoritas para testear

        test_oculto = relevantes.sample(frac=0.2, random_state=42)
        vistas = user_data.drop(test_oculto.index)

        # El conjunto "Verdadero (Ground Truth)" para el test
        dict_relevantes = {
            row["tmdb_id"]: row["rating"] for _, row in test_oculto.iterrows()
        }
        pelis_relevantes = list(dict_relevantes.keys())

        # Conjunto Candidato: Todas las pelis menos las "Vistas"
        # Incluimos las ocultas para ver si las pilla
        candidatas = todas_pelis - set(vistas["tmdb_id"].unique())

        # Evaluate SVD
        top_svd = []
        if resultados["SVD"] is not None:
            top_svd = predecir_svd_knn(modelos["SVD"], u, candidatas)
            resultados["SVD"]["p"] += precision_at_k(top_svd, pelis_relevantes)
            resultados["SVD"]["r"] += recall_at_k(top_svd, pelis_relevantes)
            resultados["SVD"]["n"] += ndcg_at_k(top_svd, dict_relevantes)
            resultados["SVD"]["h"] += hit_rate(top_svd, pelis_relevantes)
            resultados["SVD"]["recs"].append(top_svd)

        # Evaluate KNN
        top_knn = []
        if resultados["KNN"] is not None:
            top_knn = predecir_svd_knn(modelos["KNN"], u, candidatas)
            resultados["KNN"]["p"] += precision_at_k(top_knn, pelis_relevantes)
            resultados["KNN"]["r"] += recall_at_k(top_knn, pelis_relevantes)
            resultados["KNN"]["n"] += ndcg_at_k(top_knn, dict_relevantes)
            resultados["KNN"]["h"] += hit_rate(top_knn, pelis_relevantes)
            resultados["KNN"]["recs"].append(top_knn)

        # Evaluate W&D
        top_wnd = []
        if resultados["WND"] is not None:
            top_wnd = predecir_wnd(modelos["WND"], modelos["WND_MAPS"], u, candidatas)
            if top_wnd:
                resultados["WND"]["p"] += precision_at_k(top_wnd, pelis_relevantes)
                resultados["WND"]["r"] += recall_at_k(top_wnd, pelis_relevantes)
                resultados["WND"]["n"] += ndcg_at_k(top_wnd, dict_relevantes)
                resultados["WND"]["h"] += hit_rate(top_wnd, pelis_relevantes)
                resultados["WND"]["recs"].append(top_wnd)

        # Evaluate Content-Based
        if "TFIDF_MAT" in modelos:
            top_tf = predecir_content(
                modelos["TFIDF_MAT"],
                modelos["TFIDF_IDX"],
                df_movies,
                vistas,
                candidatas,
            )
            resultados["TFIDF"]["p"] += precision_at_k(top_tf, pelis_relevantes)
            resultados["TFIDF"]["r"] += recall_at_k(top_tf, pelis_relevantes)
            resultados["TFIDF"]["n"] += ndcg_at_k(top_tf, dict_relevantes)
            resultados["TFIDF"]["h"] += hit_rate(top_tf, pelis_relevantes)
            resultados["TFIDF"]["recs"].append(top_tf)

        # Evaluate Implicit BPR
        top_imp = []
        if resultados["IMP"] is not None:
            top_imp = predecir_implicit(
                modelos["IMP"], modelos["IMP_DAT"], u, vistas, candidatas
            )
            if top_imp:
                resultados["IMP"]["p"] += precision_at_k(top_imp, pelis_relevantes)
                resultados["IMP"]["r"] += recall_at_k(top_imp, pelis_relevantes)
                resultados["IMP"]["n"] += ndcg_at_k(top_imp, dict_relevantes)
                resultados["IMP"]["h"] += hit_rate(top_imp, pelis_relevantes)
                resultados["IMP"]["recs"].append(top_imp)

        # Evaluate NCF
        top_ncf = []
        if resultados["NCF"] is not None:
            top_ncf = predecir_ncf(
                modelos["NCF"], modelos["NCF_USER2IDX"], modelos["NCF_ITEM2IDX"],
                u, candidatas
            )
            if top_ncf:
                resultados["NCF"]["p"] += precision_at_k(top_ncf, pelis_relevantes)
                resultados["NCF"]["r"] += recall_at_k(top_ncf, pelis_relevantes)
                resultados["NCF"]["n"] += ndcg_at_k(top_ncf, dict_relevantes)
                resultados["NCF"]["h"] += hit_rate(top_ncf, pelis_relevantes)
                resultados["NCF"]["recs"].append(top_ncf)

        # Solo si procesamos un usuario válido, aumentamos el contador
        if (
            top_svd
            or top_knn
            or top_wnd
            or top_tf
            or (resultados.get("IMP") and top_imp)
            or (resultados.get("NCF") and top_ncf)
        ):
            n_usuarios_final += 1

    # Convertir sumas a medias --> Dividimos entre N usuarios testados válidos
    n_catalogo = len(todas_pelis)
    print("\n" + "=" * 80)
    print(
        f"  RESULTADOS GLOBALES (Promedios para Top-{K} sobre {n_usuarios_final} usuarios):"
    )
    print("=" * 80)

    records = []
    for mod, vals in resultados.items():
        if vals is None or not vals["recs"]:
            continue

        prec = vals["p"] / n_usuarios_final
        rec = vals["r"] / n_usuarios_final
        ndcg = vals["n"] / n_usuarios_final
        hr = vals["h"] / n_usuarios_final
        cov = coverage(vals["recs"], n_catalogo)

        records.append(
            {
                "Modelo": mod,
                f"Precision_{K}": prec,
                f"Recall_{K}": rec,
                f"NDCG_{K}": ndcg,
                f"Hit_Rate_{K}": hr,
                f"Coverage_{K}": cov,
            }
        )
        print(
            f"  -> {mod:<8} | Precision: {prec * 100:4.1f}% | Recall: {rec * 100:4.1f}% | "
            f"NDCG: {ndcg:.4f} | HR: {hr * 100:4.1f}% | Cov: {cov * 100:4.1f}%"
        )

    # Guardar a CSV
    df_res = pd.DataFrame(records)
    df_res.to_csv(RUTA_RESULTADOS, index=False)
    print("\n  Reporte de métricas guardado en:", RUTA_RESULTADOS)
    print("=" * 60)


############################################################################################

if __name__ == "__main__":
    evaluar()
```

- Función `predecir_ncf()` para scoring via ONNX
- NCF integrado en el loop de evaluación con Precision, Recall, NDCG, Hit Rate, Coverage

---

## Próximos Pasos

1. **Entrenar NCF**: `python src/models/jj/modelo_6_ncf.py`
2. **Evaluar ranking**: `python src/utils/evaluacion_ranking.py`
3. **Fase 3**: Re-entrenar Wide&Deep con hiperparámetros ajustados (MIN_RATINGS_PELICULA=50, EPOCHS=15)
4. **Fase 5** (otra sesión): Two-Tower model
