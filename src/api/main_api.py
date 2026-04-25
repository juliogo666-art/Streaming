from contextlib import asynccontextmanager
import logging
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .database import get_db_connection
from .etl import ejecutar_importacion, limpiar_tablas_contenido
from ..schemas.schemas import LoginRequest, RegisterRequest, RatingRequest

# [PASO 1: SCHEMAS] Definimos la salida estricta de la API para que frontend no falle
from src.schemas.recommendation import RecommendationResponse

# [PASO 2: TRACKING] Importamos el logger para guardar el historial de recomendaciones en JSONL
from src.tracking.logger import RecommendationLogger

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
os.makedirs("logs", exist_ok=True)
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

# --- RUTAS DE MODELOS (Centralizadas en artifacts/) ---
# Pesos de modelos clásicos (Surprise, sklearn, implicit)
ruta_modelo_svd = "artifacts/weights/modelo_1_SVD.joblib"
ruta_modelo_knn = "artifacts/weights/modelo_2.5_knn_msd.joblib"
ruta_tfidf_mat = "artifacts/weights/modelo_4_matriz.joblib"
ruta_tfidf_idx = "artifacts/weights/modelo_4_indices.joblib"
ruta_imp = "artifacts/weights/modelo_5_implicit.pkl"
ruta_imp_dat = "artifacts/weights/modelo_5_implicit_dataset.pkl"
# Modelos exportados a ONNX (inferencia sin PyTorch)
ruta_modelo_wnd = "artifacts/exports/modelo_3_wnd.onnx"
ruta_modelo_ncf = "artifacts/exports/modelo_6_ncf.onnx"
ruta_modelo_tt = "artifacts/exports/modelo_7_twotowers.onnx"
# Mapeos de IDs internos <-> reales (necesarios para ONNX y BPR)
ruta_wnd_map = "artifacts/mappings/wnd_mappings.pkl"
ruta_ncf_user2idx = "artifacts/mappings/ncf_user2idx.json"
ruta_ncf_item2idx = "artifacts/mappings/ncf_item2idx.json"
ruta_tt_map = "artifacts/mappings/twotowers_mappings.pkl"


@asynccontextmanager
async def lifespan(the_app: FastAPI):
    """Gestiona el ciclo de vida de la aplicación: carga modelos al arrancar, limpia al cerrar."""
    t_startup_total = time.perf_counter()
    logger.info("[STARTUP] Iniciando carga de modelos de IA en memoria...")
    logger.info("[INFO] Cargando ratings (~434MB)... espera unos 30-40s.")

    # Diccionario para acumular tiempos y generar el benchmark al final
    _bench: dict[str, float] = {}

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
    the_app.state.modelo_tt = None
    the_app.state.tt_mappings = None

    # --- Modelo 1: SVD ---
    _t0 = time.perf_counter()
    if os.path.exists(ruta_modelo_svd):
        try:
            import joblib

            the_app.state.modelo_svd = joblib.load(ruta_modelo_svd)
            logger.info("Modelo SVD cargado correctamente (Joblib).")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo SVD: {e}.")
    _bench["Modelo 1 · SVD (Joblib)"] = time.perf_counter() - _t0

    # --- Modelo 2: KNN ---
    _t0 = time.perf_counter()
    if os.path.exists(ruta_modelo_knn):
        try:
            import joblib

            the_app.state.modelo_knn = joblib.load(ruta_modelo_knn)
            logger.info("Modelo KNN+Cosine cargado (Joblib).")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo KNN: {e}")
    _bench["Modelo 2 · KNN+Cosine (Joblib)"] = time.perf_counter() - _t0

    # --- Modelo 3: Wide & Deep (ONNX Runtime) ---
    _t0 = time.perf_counter()
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
    _bench["Modelo 3 · Wide&Deep (ONNX)"] = time.perf_counter() - _t0

    # --- Modelo 4: Content-Based (TF-IDF) ---
    _t0 = time.perf_counter()
    if os.path.exists(ruta_tfidf_mat) and os.path.exists(ruta_tfidf_idx):
        try:
            import joblib

            the_app.state.modelo_tfidf_mat = joblib.load(ruta_tfidf_mat)
            the_app.state.modelo_tfidf_idx = joblib.load(ruta_tfidf_idx)
            logger.info("Modelo TF-IDF cargado correctamente (Joblib).")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo TF-IDF: {e}")
    _bench["Modelo 4 · TF-IDF Content-Based (Joblib)"] = time.perf_counter() - _t0

    # --- Modelo 5: Implicit BPR ---
    _t0 = time.perf_counter()
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
    _bench["Modelo 5 · Implicit BPR (Pickle)"] = time.perf_counter() - _t0

    # --- Modelo 6: NCF (ONNX) ---
    _t0 = time.perf_counter()
    if os.path.exists(ruta_modelo_ncf):
        try:
            import onnxruntime as ort
            import json

            the_app.state.modelo_ncf = ort.InferenceSession(ruta_modelo_ncf)
            if os.path.exists(ruta_ncf_user2idx):
                with open(ruta_ncf_user2idx, "r") as f:
                    the_app.state.ncf_user2idx = {
                        int(k): v for k, v in json.load(f).items()
                    }
            if os.path.exists(ruta_ncf_item2idx):
                with open(ruta_ncf_item2idx, "r") as f:
                    the_app.state.ncf_item2idx = {
                        int(k): v for k, v in json.load(f).items()
                    }
            logger.info("Modelo NCF ONNX cargado correctamente.")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo NCF: {e}")
    _bench["Modelo 6 · NCF (ONNX)"] = time.perf_counter() - _t0

    # --- Modelo 7: Two Towers (ONNX) ---
    _t0 = time.perf_counter()
    if os.path.exists(ruta_modelo_tt) and os.path.exists(ruta_tt_map):
        try:
            import onnxruntime as ort
            import pickle

            the_app.state.modelo_tt = ort.InferenceSession(ruta_modelo_tt)
            with open(ruta_tt_map, "rb") as f:
                the_app.state.tt_mappings = pickle.load(f)
            logger.info("Modelo TwoTowers ONNX cargado correctamente.")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo TwoTowers: {e}")
    _bench["Modelo 7 · TwoTowers (ONNX)"] = time.perf_counter() - _t0

    # --- CARGA DE DATOS (CSV) ---
    _t0 = time.perf_counter()
    try:
        if os.path.exists(ruta_ratings):
            the_app.state.df_ratings_ia = pd.read_csv(ruta_ratings)
            logger.info(
                f"Ratings cargados: {len(the_app.state.df_ratings_ia):,} filas."
            )
            # Pre-calcular conteos para velocidad O(1) en mensajes de progreso
            the_app.state.user_counts = (
                the_app.state.df_ratings_ia.groupby("userId").size().to_dict()
            )
    except Exception as e:
        logger.error(f"Error al cargar CSV de ratings: {e}")
    _bench["CSV · ratings_finales_ia (~434MB)"] = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    try:
        if os.path.exists(ruta_catalogo):
            the_app.state.df_catalogo = pd.read_csv(ruta_catalogo)
            logger.info(
                f"Catálogo cargado: {len(the_app.state.df_catalogo):,} películas."
            )
    except Exception as e:
        logger.error(f"Error al cargar CSV de catálogo: {e}")
    _bench["CSV · dataset_final_movies"] = time.perf_counter() - _t0

    # ── BENCHMARK RESUMEN ─────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_startup_total
    sep = "─" * 55
    logger.info(f"[BENCHMARK] {sep}")
    for nombre, segundos in _bench.items():
        # Marca los que superan 5 segundos como posibles cuellos de botella
        alerta = "LENTO" if segundos > 5.0 else ""
        logger.info(f"[BENCHMARK]   {nombre:<42} {segundos:>6.2f}s{alerta}")
    logger.info(f"[BENCHMARK] {sep}")
    logger.info(f"[BENCHMARK]   {'TOTAL ARRANQUE':<42} {t_total:>6.2f}s")
    logger.info(f"[BENCHMARK] {sep}")
    # ─────────────────────────────────────────────────────────────────────────

    logger.info("[STARTUP] Sistema listo para servir peticiones.")
    yield  # La app está corriendo
    logger.info("[SHUTDOWN] Cerrando aplicación...")


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instanciamos el logger de telemetría (se guardará en logs/recommendations.jsonl)
# Esto nos servirá para analizar a posteriori qué se ha estado recomendando y a quién.
telemetria = RecommendationLogger()


def _log_recomendacion_con_tiempo(
    user_id: int, modelo: str, recomendaciones_top_n: list, inicio: float
) -> None:
    """Registra telemetría de recomendación junto con su tiempo de respuesta."""
    tiempo_ms = round((time.perf_counter() - inicio) * 1000, 2)
    telemetria.log_recommendations(
        str(user_id),
        modelo,
        recomendaciones_top_n,
        tiempo_recomendacion_ms=tiempo_ms,
    )


@app.get("/status")
def check_status():
    return {"status": "Backend funcionando correctamente"}

# --- Heartbeat Watchdog ---
heartbeat_iniciado = False
tiempo_ultimo_heartbeat = time.time()

@app.post("/api/heartbeat")
def heartbeat():
    global heartbeat_iniciado, tiempo_ultimo_heartbeat
    heartbeat_iniciado = True
    tiempo_ultimo_heartbeat = time.time()
    return {"status": "ok"}

@app.get("/api/heartbeat_status")
def heartbeat_status():
    global heartbeat_iniciado, tiempo_ultimo_heartbeat
    if not heartbeat_iniciado:
        return {"seconds_since_last": 0}
    return {"seconds_since_last": time.time() - tiempo_ultimo_heartbeat}


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


def _adjuntar_gustos_perfil(cursor, usuario: dict) -> None:
    """Rellena gustos_top3 y gustos_source (prioridad ML; si no hay, user_selected)."""
    uid = usuario["id_usuario"]
    cursor.execute(
        """
        SELECT g.id, g.name
        FROM user_interests ui
        JOIN genres g ON g.id = ui.genre_id
        WHERE ui.id_usuario = %s
          AND ui.source = 'ml_inferred'
        ORDER BY g.name ASC
        LIMIT 3
        """,
        (uid,),
    )
    gustos_ml = cursor.fetchall()
    if gustos_ml:
        usuario["gustos_top3"] = [row["name"] for row in gustos_ml]
        usuario["gustos_source"] = "ml_inferred"
        return
    cursor.execute(
        """
        SELECT g.id, g.name
        FROM user_interests ui
        JOIN genres g ON g.id = ui.genre_id
        WHERE ui.id_usuario = %s
          AND (ui.source = 'user_selected' OR ui.source IS NULL)
        ORDER BY g.name ASC
        """,
        (uid,),
    )
    gustos_sel = cursor.fetchall()
    if gustos_sel:
        usuario["gustos_top3"] = [row["name"] for row in gustos_sel]
        usuario["gustos_source"] = "user_selected"


@app.post("/login")
def login(datos: LoginRequest):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # 1. Buscamos al usuario solo por nombre de usuario
        query = "SELECT id_usuario, username, email, passwd, role FROM users WHERE username = %s"
        cursor.execute(query, (datos.username,))
        usuario = cursor.fetchone()

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
                _adjuntar_gustos_perfil(cursor, usuario)

                # Por seguridad, borramos la contraseña del diccionario temporal antes de mandarlo al frontend
                del usuario["passwd"]
                return {"status": "success", "message": "Login exitoso", "user": usuario}

            raise HTTPException(
                status_code=401, detail="Usuario o contraseña incorrectos"
            )

        raise HTTPException(status_code=401, detail="Usuario o contraseña incorrectos")
    finally:
        cursor.close()
        conn.close()


@app.get("/genres")
def obtener_generos():
    logger.info("[API] Petición a /genres recibida")
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT id, name FROM genres ORDER BY name ASC")
        generos = cursor.fetchall()
        cursor.close()
        conn.close()
        logger.info(f"[API] Se encontraron {len(generos)} géneros.")
        return generos
    except Exception as e:
        logger.error(f"[ERROR] Error al obtener géneros: {e}")
        # Intentamos dar un mensaje más útil que un 500 genérico
        raise HTTPException(
            status_code=500, detail=f"Error en base de datos al leer géneros: {str(e)}"
        )



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

        # Insertar intereses (origen: selección en el registro)
        if intereses:
            query_int = """
                INSERT INTO user_interests (id_usuario, genre_id, source)
                VALUES (%s, %s, 'user_selected')
            """
            for g_id in intereses:
                cursor.execute(query_int, (id_usuario, g_id))

        conn.commit()

        cursor.execute(
            "SELECT id_usuario, username, email, role FROM users WHERE id_usuario = %s",
            (id_usuario,),
        )
        usuario_resp = cursor.fetchone()
        if not usuario_resp:
            raise HTTPException(
                status_code=500, detail="Usuario creado pero no se pudo leer el perfil."
            )
        _adjuntar_gustos_perfil(cursor, usuario_resp)

        return {
            "status": "success",
            "message": "Usuario registrado correctamente",
            "user_id": id_usuario,
            "user": usuario_resp,
        }

    except Exception as e:
        conn.rollback()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error en el servidor: {str(e)}")
    finally:
        cursor.close()
        conn.close()


@app.get("/recomendar/svd/{user_id}", response_model=RecommendationResponse)
@app.get("/recomendar/{user_id}", response_model=RecommendationResponse)
def recomendar_peliculas(user_id: int, n: int = 10):
    """
    Endpoint que devuelve las top-N películas recomendadas para un usuario.
    Usa el modelo SVD entrenado para predecir ratings de películas no vistas.
    """
    t_inicio = time.perf_counter()
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

    # Telemetría: Registra un evento en disco de la recomendación servida
    _log_recomendacion_con_tiempo(user_id, "SVD (Surprise)", top_n, t_inicio)

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


@app.get("/recomendar/knn/{user_id}", response_model=RecommendationResponse)
def recomendar_knn(user_id: int, n: int = 10):
    """Endpoint de recomendaciones usando KNN + Cosine Similarity (Modelo 2)."""
    t_inicio = time.perf_counter()
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
    # top_items ya no es necesario
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

    # Telemetría: Registra un evento en disco de la recomendación servida
    _log_recomendacion_con_tiempo(user_id, "KNN+Cosine", top_n, t_inicio)

    return {"recomendaciones": top_n, "modelo": "KNN+Cosine"}


##############################################################################################
#  Recomendación Modelo Wide & Deep (PyTorch)
##############################################################################################


@app.get("/recomendar/wnd/{user_id}", response_model=RecommendationResponse)
def recomendar_wnd_endpoint(user_id: int, n: int = 10):
    """Endpoint de recomendaciones usando Wide & Deep Neural Network (ONNX Native)."""
    t_inicio = time.perf_counter()
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
        # Informar del progreso en lugar de fallback
        count = app.state.user_counts.get(user_id, 0)
        return {
            "recomendaciones": [],
            "modelo": "Wide&Deep (ONNX)",
            "mensaje": f"No alcanzas las 100 valoraciones requeridas ({count}/100).",
            "insufficient_data": True,
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
            "user_ids": user_arr,
            "item_ids": movie_arr,
        }
        # La salida de nuestro nuevo W&D Ranking es (Batch,) o (Batch, 1)
        preds_raw = app.state.modelo_wnd.run(None, ort_inputs)[0].flatten()
    except Exception as e:
        print(f"ERROR en inferencia Wide&Deep: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error en inferencia del modelo WnD: {e}"
        )

    # La salida de ranking (logits) la convertimos a escala visual 0.5-5.0
    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

    preds_prob = sigmoid(preds_raw)

    predicciones = [
        {
            "tmdb_id": int(tid),
            "predicted_rating": round(float(preds_prob[i] * 4.5 + 0.5), 2),
        }
        for i, tid in enumerate(tmdb_ids)
    ]

    predicciones.sort(key=lambda x: x["predicted_rating"], reverse=True)
    top_n = predicciones[:n]
    enriquecer_recomendaciones(top_n)

    # Telemetría: Registra un evento en disco de la recomendación servida
    _log_recomendacion_con_tiempo(user_id, "Wide&Deep (ONNX)", top_n, t_inicio)

    return {"recomendaciones": top_n, "modelo": "Wide&Deep (ONNX)"}


##############################################################################################
#  Recomendación Modelo 4: Content-Based / Cold Start
##############################################################################################


@app.get("/recomendar/content/{user_id}", response_model=RecommendationResponse)
def recomendar_content_endpoint(user_id: int, n: int = 10):
    """Endpoint de recomendaciones por contenido (Modelo 4)."""
    t_inicio = time.perf_counter()
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
        # 1. Intentar obtener los intereses del usuario (géneros) de la BD
        intereses_usuario = []
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT genre_id FROM user_interests WHERE id_usuario = %s", (user_id,))
            rows = cursor.fetchall()
            intereses_usuario = [row["genre_id"] for row in rows]
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Error obteniendo intereses del usuario {user_id}: {e}")

        # 2. Filtrar el catálogo base a esos géneros
        if intereses_usuario:
            import re
            # Usar regex para buscar IDs exactos (boundary \b) en el string '[gen1, gen2]'
            patron = r'\b(?:' + '|'.join(map(str, intereses_usuario)) + r')\b'
            mask = app.state.df_catalogo["genre_ids"].str.contains(patron, na=False, regex=True)
            catalogo_filtrado = app.state.df_catalogo[mask]
            nomb_modelo = "TF-IDF (Cold Start Géneros)"
        else:
            catalogo_filtrado = app.state.df_catalogo
            nomb_modelo = "TF-IDF (Cold Start Populares)"

        # Fallback por si el filtrado se quedó en blanco
        if catalogo_filtrado.empty:
            catalogo_filtrado = app.state.df_catalogo

        # 3. Sort por popularidad dentro del filtro
        if "vote_count" in catalogo_filtrado.columns:
            top_pop = (
                catalogo_filtrado[catalogo_filtrado["vote_count"] > 100]
                .sort_values(by="vote_average", ascending=False)
                .head(n)
            )
            if len(top_pop) < n:
                 top_pop = catalogo_filtrado.sort_values(by="vote_average", ascending=False).head(n)
        else:
            top_pop = catalogo_filtrado.head(n)

        recomendaciones = []
        for idx, row in top_pop.iterrows():
            recomendaciones.append(
                {"tmdb_id": int(row["tmdb_id"]), "predicted_rating": 4.5}
            )

        enriquecer_recomendaciones(recomendaciones)

        # Telemetría: Registra evento
        _log_recomendacion_con_tiempo(user_id, nomb_modelo, recomendaciones, t_inicio)

        return {
            "recomendaciones": recomendaciones,
            "modelo": nomb_modelo,
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

    # Telemetría: Registra evento
    _log_recomendacion_con_tiempo(user_id, "Content-Based", predicciones, t_inicio)

    return {"recomendaciones": predicciones, "modelo": "Content-Based"}


##############################################################################################
#  Recomendación Modelo 5: Implicit BPR
##############################################################################################


@app.get("/recomendar/implicit/{user_id}", response_model=RecommendationResponse)
def recomendar_implicit_endpoint(user_id: int, n: int = 10):
    """Endpoint de recomendaciones usando filtrado colaborativo BPR de la librería implicit."""
    t_inicio = time.perf_counter()
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

    # Telemetría: Registra evento
    _log_recomendacion_con_tiempo(user_id, "Implicit BPR", predicciones, t_inicio)

    return {"recomendaciones": predicciones, "modelo": "Implicit BPR"}


##############################################################################################
#  Recomendación Modelo 6: NCF (Neural Collaborative Filtering)
##############################################################################################


@app.get("/recomendar/ncf/{user_id}", response_model=RecommendationResponse)
def recomendar_ncf_endpoint(user_id: int, n: int = 10):
    """
    Endpoint de recomendaciones usando NCF-Lite (GMF + MLP) via ONNX Runtime.
    El modelo produce logits de relevancia para cada item; se seleccionan los Top-N
    excluyendo items ya vistos por el usuario.
    """
    t_inicio = time.perf_counter()
    logger.info(f"Petición NCF para User {user_id}")

    if app.state.modelo_ncf is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo NCF no cargado. Ejecuta modelo_6_ncf.py para generar los artefactos.",
        )
    if app.state.ncf_user2idx is None or app.state.ncf_item2idx is None:
        raise HTTPException(
            status_code=503, detail="Mapeos NCF (user2idx/item2idx) no cargados."
        )

    # Verificar que el usuario existe en el vocabulario del modelo
    if user_id not in app.state.ncf_user2idx:
        # Informar del progreso en lugar de fallback
        count = app.state.user_counts.get(user_id, 0)
        return {
            "recomendaciones": [],
            "modelo": "NCF",
            "mensaje": f"No alcanzas las 100 valoraciones requeridas ({count}/100).",
            "insufficient_data": True,
        }

    user_idx = app.state.ncf_user2idx[user_id]

    # Construir mapeos inversos
    idx2item = {v: k for k, v in app.state.ncf_item2idx.items()}
    n_items = len(app.state.ncf_item2idx)

    # Puntuar TODOS los items de una sola vez (batch inference via ONNX)
    try:
        user_ids_np = np.full(n_items, user_idx, dtype=np.int64)
        item_ids_np = np.array(list(app.state.ncf_item2idx.values()), dtype=np.int64)

        # La salida de ONNX suele ser (N, 1). La aplanamos a (N,)
        scores = app.state.modelo_ncf.run(
            None, {"user_ids": user_ids_np, "item_ids": item_ids_np}
        )[0].flatten()

        # Excluir items ya vistos
        if app.state.df_ratings_ia is not None:
            user_ratings = app.state.df_ratings_ia[
                app.state.df_ratings_ia["userId"] == user_id
            ]
            pelis_vistas = set(user_ratings["tmdb_id"].unique())
            for tid in pelis_vistas:
                if tid in app.state.ncf_item2idx:
                    midx = app.state.ncf_item2idx[tid]
                    # midx es el valor (v) de item2idx
                    # Queremos poner el score de esa peli a -inf
                    # Pero OJO: 'scores' está indexado por el orden de 'item_ids_np'
                    # Como item_ids_np es np.array(list(item2idx.values())), el índice
                    # de un midx está en su propia posición si values() es secuencial.
                    # Para ser 100% seguros, usamos el mapeo directo
                    scores[midx] = -np.inf

        # Seleccionar Top-N
        top_indices = np.argsort(scores)[::-1][:n]

        idx2item = {v: k for k, v in app.state.ncf_item2idx.items()}
        predicciones = []
        for idx_item in top_indices:
            if scores[idx_item] == -np.inf:
                continue
            tid = idx2item[idx_item]
            score_puro = float(scores[idx_item])
            # Transformar logit a rating visual amigable (3.0 - 5.0 rango)
            rating_ui = min(5.0, max(0.5, 3.5 + (score_puro * 0.3)))
            predicciones.append(
                {"tmdb_id": int(tid), "predicted_rating": round(rating_ui, 2)}
            )

        enriquecer_recomendaciones(predicciones)

        # Telemetría: Registra evento
        _log_recomendacion_con_tiempo(user_id, "NCF-Lite", predicciones, t_inicio)

        return {"recomendaciones": predicciones, "modelo": "NCF-Lite"}

    except Exception as e:
        logger.error(f"Error en recomendación NCF para User {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


##############################################################################################
#  Recomendación Modelo 7: Two Towers (Bi-Encoder)
##############################################################################################


@app.get("/recomendar/twotowers/{user_id}", response_model=RecommendationResponse)
def recomendar_tt_endpoint(user_id: int, n: int = 10):
    """Endpoint de recomendaciones usando Two Towers Neural Network (ONNX)."""
    t_inicio = time.perf_counter()
    logger.info(f"Petición TwoTowers para User {user_id}")

    if app.state.modelo_tt is None or app.state.tt_mappings is None:
        raise HTTPException(status_code=503, detail="Modelo TwoTowers no cargado.")

    user2idx = app.state.tt_mappings["user2idx"]
    item2idx = app.state.tt_mappings["item2idx"]
    idx2item = {v: k for k, v in item2idx.items()}

    if user_id not in user2idx:
        # Informar del progreso en lugar de fallback
        count = app.state.user_counts.get(user_id, 0)
        return {
            "recomendaciones": [],
            "modelo": "Two-Towers",
            "mensaje": f"No alcanzas las 50 valoraciones requeridas ({count}/50).",
            "insufficient_data": True,
        }

    u_idx = user2idx[user_id]

    # Puntuar items (Batch inference)
    # Para velocidad en demo local puntuamos solo items que conoce el modelo
    tids_candidatos = list(item2idx.keys())
    i_indices = list(item2idx.values())

    user_arr = np.full(len(i_indices), u_idx, dtype=np.int64)
    item_arr = np.array(i_indices, dtype=np.int64)

    try:
        ort_inputs = {"user_ids": user_arr, "item_ids": item_arr}
        # El modelo TT devuelve similitud (producto escalar)
        scores = app.state.modelo_tt.run(None, ort_inputs)[0].flatten()
    except Exception as e:
        logger.error(f"Error TwoTowers inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Filtrar ya vistas
    if app.state.df_ratings_ia is not None:
        vistas = set(
            app.state.df_ratings_ia[app.state.df_ratings_ia["userId"] == user_id][
                "tmdb_id"
            ]
        )
        for idx, tid in enumerate(tids_candidatos):
            if tid in vistas:
                scores[idx] = -np.inf

    top_indices = np.argsort(scores)[::-1][:n]

    predicciones = []
    for idx in top_indices:
        if scores[idx] == -np.inf:
            continue
        tid = idx2item[item_arr[idx]]
        # Escalar similitud arbitraria a 0.5-5.0 para UI
        s = float(scores[idx])
        rating_ui = min(5.0, max(0.5, 3.5 + (s * 0.1)))
        predicciones.append(
            {"tmdb_id": int(tid), "predicted_rating": round(rating_ui, 2)}
        )

    enriquecer_recomendaciones(predicciones)

    # Telemetría: Registra evento
    _log_recomendacion_con_tiempo(user_id, "Two-Towers", predicciones, t_inicio)

    return {"recomendaciones": predicciones, "modelo": "Two-Towers"}


##############################################################################################
#  Recomendación SMART: Selector Dinámico de Modelo
##############################################################################################
#
#  Umbrales definidos según análisis de métricas (metricas_ranking.csv):
#  ┌───────────────────────┬──────────┬──────────────┬──────────┐
#  │ Modelo                │ NDCG@10  │ Precision@10 │ HitRate  │
#  ├───────────────────────┼──────────┼──────────────┼──────────┤
#  │ NCF-Lite              │  0.779   │    0.824     │  0.997   │  ← Mejor (≥100 ratings)
#  │ Wide&Deep             │  0.604   │    0.622     │  0.980   │  ← Bueno  (≥100 ratings)
#  │ SVD                   │  0.323   │    0.305     │  0.896   │  ← Medio  (≥11 ratings)
#  │ KNN                   │  0.325   │    0.316     │  0.863   │  ← Medio  (≥11 ratings)
#  │ BPR                   │  0.307   │    0.309     │  0.863   │  ← Medio  (≥11 ratings)
#  │ TF-IDF (Content)      │  0.022   │    0.020     │  0.161   │  ← Cold-Start (0-10)
#  └───────────────────────┴──────────┴──────────────┴──────────┘
#
UMBRAL_COLD_START = 10   # 0-10 valoraciones → Content-Based / Popularidad
UMBRAL_AVANZADO = 100    # 100+ → NCF o Wide&Deep


@app.get("/recomendar/smart/{user_id}", response_model=RecommendationResponse)
def recomendar_smart(user_id: int, n: int = 10):
    """
    Selector inteligente de modelo de recomendación.
    Analiza el historial del usuario y redirige al modelo más adecuado:
      - 0-10 valoraciones:   Content-Based (TF-IDF) / Popularidad
      - 11-99 valoraciones:  SVD (mejor MAE/RMSE de los clásicos)
      - 100+ valoraciones:   NCF-Lite (mejor NDCG/Precision) con fallback a Wide&Deep
    """
    # Contar valoraciones del usuario
    n_ratings = app.state.user_counts.get(user_id, 0) if hasattr(app.state, "user_counts") and app.state.user_counts else 0

    logger.info(f"[SMART] User {user_id} tiene {n_ratings} valoraciones.")

    # --- NIVEL 1: Cold Start (0-10 ratings) ---
    if n_ratings <= UMBRAL_COLD_START:
        logger.info(f"[SMART] → Redirigiendo a Content-Based (Cold Start)")
        resultado = recomendar_content_endpoint(user_id, n)
        # Añadimos info del selector al resultado
        if isinstance(resultado, dict):
            resultado["selector"] = f"Smart → Content-Based (Cold Start: {n_ratings} valoraciones)"
        return resultado

    # --- NIVEL 2: Usuario Intermedio (11-99 ratings) ---
    if n_ratings < UMBRAL_AVANZADO:
        # SVD tiene ligeramente mejor NDCG que KNN (0.323 vs 0.325 pero mejor consistencia)
        # Intentamos SVD primero, si falla probamos KNN, si falla BPR
        logger.info(f"[SMART] → Intentando SVD (Intermedio: {n_ratings} valoraciones)")

        if app.state.modelo_svd is not None:
            resultado = recomendar_peliculas(user_id, n)
            if isinstance(resultado, dict):
                resultado["selector"] = f"Smart → SVD (Intermedio: {n_ratings}/100 valoraciones)"
            return resultado

        if app.state.modelo_knn is not None:
            logger.info(f"[SMART] → SVD no disponible, usando KNN")
            resultado = recomendar_knn(user_id, n)
            if isinstance(resultado, dict):
                resultado["selector"] = f"Smart → KNN (Fallback Intermedio: {n_ratings} valoraciones)"
            return resultado

        if app.state.modelo_imp is not None:
            logger.info(f"[SMART] → SVD y KNN no disponibles, usando BPR")
            resultado = recomendar_implicit_endpoint(user_id, n)
            if isinstance(resultado, dict):
                resultado["selector"] = f"Smart → BPR (Fallback Intermedio: {n_ratings} valoraciones)"
            return resultado

        # Último recurso: Content-Based
        logger.info(f"[SMART] → Ningún modelo colaborativo disponible, fallback a Content-Based")
        resultado = recomendar_content_endpoint(user_id, n)
        if isinstance(resultado, dict):
            resultado["selector"] = f"Smart → Content-Based (Sin modelos colaborativos)"
        return resultado

    # --- NIVEL 3: Usuario Experto (100+ ratings) ---
    logger.info(f"[SMART] → Usuario experto ({n_ratings} valoraciones), intentando NCF")

    # NCF es el mejor modelo (NDCG 0.779, Precision 0.824)
    if app.state.modelo_ncf is not None and app.state.ncf_user2idx is not None:
        if user_id in app.state.ncf_user2idx:
            resultado = recomendar_ncf_endpoint(user_id, n)
            if isinstance(resultado, dict):
                resultado["selector"] = f"Smart → NCF-Lite (Experto: {n_ratings} valoraciones)"
            return resultado
        else:
            logger.info(f"[SMART] → User {user_id} no está en mappings NCF, intentando Wide&Deep")

    # Fallback a Wide&Deep (NDCG 0.604)
    if app.state.modelo_wnd is not None and app.state.wnd_mappings is not None:
        user2idx_wnd = app.state.wnd_mappings.get("user2idx", {})
        if user_id in user2idx_wnd:
            resultado = recomendar_wnd_endpoint(user_id, n)
            if isinstance(resultado, dict):
                resultado["selector"] = f"Smart → Wide&Deep (Experto fallback: {n_ratings} valoraciones)"
            return resultado

    # Si los modelos avanzados no lo conocen, caemos a SVD (que sí cubre a todos)
    logger.info(f"[SMART] → Modelos avanzados no conocen al usuario, fallback a SVD")
    if app.state.modelo_svd is not None:
        resultado = recomendar_peliculas(user_id, n)
        if isinstance(resultado, dict):
            resultado["selector"] = f"Smart → SVD (Fallback Experto: {n_ratings} valoraciones)"
        return resultado

    # Último recurso absoluto
    resultado = recomendar_content_endpoint(user_id, n)
    if isinstance(resultado, dict):
        resultado["selector"] = f"Smart → Content-Based (Último recurso)"
    return resultado


##############################################################################################
#  Valoraciones de Usuario (user_ratings)
##############################################################################################


@app.post("/api/rating")
def registrar_valoracion(datos: RatingRequest):
    """
    Registra o actualiza la valoración de un usuario sobre una película.
    Usa INSERT ... ON DUPLICATE KEY UPDATE para permitir cambiar la nota.
    """
    # Validar rango
    if datos.rating < 0.5 or datos.rating > 5.0:
        raise HTTPException(status_code=400, detail="La valoración debe estar entre 0.5 y 5.0")
    # Validar paso de 0.5
    if (datos.rating * 2) != int(datos.rating * 2):
        raise HTTPException(status_code=400, detail="La valoración debe ser en pasos de 0.5 (ej: 0.5, 1.0, 1.5 ... 5.0)")

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO user_ratings (id_usuario, tmdb_id, rating)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE rating = VALUES(rating)
            """,
            (datos.user_id, datos.tmdb_id, datos.rating),
        )
        conn.commit()
        logger.info(f"[RATING] User {datos.user_id} → Película {datos.tmdb_id} = {datos.rating}⭐")
        return {"status": "success", "message": "Valoración registrada correctamente."}
    except Exception as e:
        conn.rollback()
        logger.error(f"[RATING] Error al registrar valoración: {e}")
        raise HTTPException(status_code=500, detail=f"Error al guardar la valoración: {e}")
    finally:
        cursor.close()
        conn.close()


@app.get("/api/ratings/{user_id}")
def obtener_valoraciones_usuario(user_id: int):
    """
    Devuelve todas las valoraciones que un usuario ha dado.
    El frontend las usa para precargar las estrellas ya asignadas.
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT tmdb_id, rating FROM user_ratings WHERE id_usuario = %s",
            (user_id,),
        )
        filas = cursor.fetchall()
        # Devolvemos como dict {tmdb_id: rating} para búsqueda O(1) en el frontend
        ratings_dict = {row["tmdb_id"]: float(row["rating"]) for row in filas}
        return {"user_id": user_id, "ratings": ratings_dict}
    except Exception as e:
        logger.error(f"[RATING] Error al obtener valoraciones de user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


@app.get("/api/movie/{tmdb_id}")
def obtener_detalle_pelicula(tmdb_id: int):
    """
    Devuelve los datos completos de una película del catálogo en memoria.
    Usado por el dialog/modal de ficha completa en el frontend.
    """
    if app.state.df_catalogo is None:
        raise HTTPException(status_code=503, detail="Catálogo no cargado.")

    match = app.state.df_catalogo[app.state.df_catalogo["tmdb_id"] == tmdb_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Película {tmdb_id} no encontrada en el catálogo.")

    fila = match.iloc[0]
    return {
        "tmdb_id": int(fila["tmdb_id"]),
        "titulo": str(fila.get("titulo", "Sin Título")),
        "original_title": str(fila.get("original_title", "")),
        "overview": str(fila.get("overview", "Sin sinopsis disponible.")),
        "poster_path": str(fila.get("poster_path", "")),
        "backdrop_path": str(fila.get("backdrop_path", "")),
        "fecha_estreno": str(fila.get("fecha_estreno", "")),
        "vote_average": float(fila.get("vote_average", 0)),
        "vote_count": int(fila.get("vote_count", 0)),
        "adult": bool(fila.get("adult", False)),
        "original_language": str(fila.get("original_language", "")),
        "genre_ids": str(fila.get("genre_ids", "[]")),
        "popularity": float(fila.get("popularity", 0)),
    }


#  Serendipia


# Mapeo de nombres de género español (tabla `genres`) → inglés (tabla `serendipity_cache`)
_GENRE_ES_TO_EN: dict[str, str] = {
    "Acción": "Action",
    "Aventura": "Adventure",
    "Animación": "Animation",
    "Comedia": "Comedy",
    "Crimen": "Crime",
    "Documental": "Documentary",
    "Drama": "Drama",
    "Familia": "Family",
    "Fantasía": "Fantasy",
    "Historia": "History",
    "Terror": "Horror",
    "Música": "Music",
    "Misterio": "Mystery",
    "Romance": "Romance",
    "Ciencia ficción": "Science Fiction",
    "Película de TV": "TV Movie",
    "Suspense": "Thriller",
    "Bélica": "War",
    "Western": "Western",
}


@app.get("/api/serendipia/{user_id}")
def tragaperras_serendipia(user_id: int):
    """Devuelve 3 películas 'joya oculta' seleccionadas con muestreo ponderado
    por serendipity_score, personalizado con los 2 géneros favoritos del usuario.
    Toda la matemática está pre-calculada en serendipity_cache; este endpoint es puro I/O.
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # 1. Obtener los 2 géneros favoritos del usuario
        cursor.execute(
            """
            SELECT g.name AS genre_name
            FROM user_interests ui
            INNER JOIN genres g ON ui.genre_id = g.id
            WHERE ui.id_usuario = %s
            LIMIT 2
            """,
            (user_id,),
        )
        filas_generos = cursor.fetchall()

        if not filas_generos:
            raise HTTPException(
                status_code=404,
                detail=f"El usuario {user_id} no tiene géneros favoritos registrados.",
            )

        # Traducir nombres español → inglés para consultar serendipity_cache
        generos_es = [f["genre_name"] for f in filas_generos]
        generos = [_GENRE_ES_TO_EN.get(g, g) for g in generos_es]

        # 2. Recuperar candidatos pre-calculados de la caché (sin matemáticas aquí)
        placeholders = ", ".join(["%s"] * len(generos))
        cursor.execute(
            f"SELECT movie_id, genre, serendipity_score FROM serendipity_cache WHERE genre IN ({placeholders})",
            tuple(generos),
        )
        candidatos = cursor.fetchall()

    finally:
        cursor.close()
        conn.close()

    if not candidatos:
        raise HTTPException(
            status_code=503,
            detail="La caché de serendipia está vacía. Ejecuta: python -m src.serendipia.actualizar_cache_cron",
        )

    # 3. Excluir películas que el usuario ya ha puntuado
    df = pd.DataFrame(candidatos)
    df_ratings = getattr(app.state, "df_ratings_ia", None)
    if df_ratings is not None:
        ya_puntuadas = set(
            df_ratings[df_ratings["userId"] == user_id]["tmdb_id"].tolist()
        )
        df = df[~df["movie_id"].isin(ya_puntuadas)]
        logger.info(f"[Serendipia] User {user_id} | Excluidas {len(ya_puntuadas)} pelis ya puntuadas | Candidatas restantes: {len(df)}")

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"El usuario {user_id} ya ha puntuado todas las películas de sus géneros favoritos.",
        )

    # 4. Muestreo ponderado sin reemplazo (Pandas) usando serendipity_score como peso
    n = min(3, len(df))
    ganadores = df.sample(n=n, weights="serendipity_score", replace=False)

    recomendaciones = [
        {
            "movie_id": int(row.movie_id),
            "genre": str(row.genre),
            "serendipity_score": round(float(row.serendipity_score), 8),
        }
        for row in ganadores.itertuples(index=False)
    ]

    logger.info(f"[Serendipia] User {user_id} | Géneros: {generos_es} → {generos} | Ganadores: {[r['movie_id'] for r in recomendaciones]}")

    return {
        "user_id": user_id,
        "generos_favoritos": generos,
        "recomendaciones": recomendaciones,
    }
