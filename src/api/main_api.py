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
