from fastapi import FastAPI, HTTPException
from .database import get_db_connection
from .etl import ejecutar_importacion, limpiar_tablas_contenido
import csv
import pandas as pd
import numpy as np
from pydantic import BaseModel
import bcrypt
import pickle
import os

import joblib

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


# Clase para definir qué datos esperamos en el JSON del POST
class LoginRequest(BaseModel):
    username: str
    password: str


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


##############################################################################################
#  Recomendación Modelos
##############################################################################################

# Ruta al modelo SVD entrenado y al CSV de ratings, para saber qué pelis ya ha visto el usuario.
ruta_modelo_svd = "src/models/jj/modelo_1_SVD.joblib"
ruta_ratings = "src/data/ready/ratings_finales_ia.csv"
ruta_catalogo = "src/data/ready/dataset_final_movies.csv"

# Cargamos el modelo SVD una sola vez al arrancar el Backend, no en cada petición.
modelo_svd = None
df_ratings_ia = None
df_catalogo = None

# --- KNN + Cosine Similarity ---
ruta_modelo_knn = "src/models/jj/modelo_2_knn_cs.joblib"
modelo_knn = None

# --- Wide & Deep (ONNX) ---
ruta_modelo_wnd = "src/models/jj/modelo_3_wnd.onnx"
ruta_mapeos_wnd = "src/models/jj/wnd_mappings.pkl"
modelo_wnd = None
wnd_mappings = None

# --- Content-Based (TF-IDF) ---
ruta_tfidf_mat = "src/models/jj/modelo_4_matriz.joblib"
ruta_tfidf_idx = "src/models/jj/modelo_4_indices.joblib"
modelo_tfidf_mat = None
modelo_tfidf_idx = None

# --- Implicit BPR ---
ruta_imp_mod = "src/models/jj/modelo_5_implicit.pkl"
ruta_imp_dat = "src/models/jj/modelo_5_implicit_dataset.pkl"
modelo_imp = None
modelo_imp_dat = None


@app.on_event("startup")
def cargar_modelo_al_arrancar():
    """
    Se ejecuta automáticamente cuando FastAPI arranca.
    Carga los 3 modelos de IA y los DataFrames necesarios en memoria.
    """
    global modelo_svd, df_ratings_ia, df_catalogo
    global modelo_knn, modelo_wnd, wnd_mappings
    global modelo_tfidf_mat, modelo_tfidf_idx
    global modelo_imp, modelo_imp_dat

    # --- Modelo 1: SVD ---
    if os.path.exists(ruta_modelo_svd):
        modelo_svd = joblib.load(ruta_modelo_svd)
        print("Modelo SVD cargado correctamente (Joblib).")
    else:
        print(f"No se encontró el modelo SVD en {ruta_modelo_svd}")

    # --- Modelo 2: KNN + Cosine Similarity ---
    if os.path.exists(ruta_modelo_knn):
        modelo_knn = joblib.load(ruta_modelo_knn)
        print("Modelo KNN+Cosine cargado (Joblib).")
    else:
        print(f"No se encontró el modelo KNN en {ruta_modelo_knn}")

    # --- Modelo 3: Wide & Deep (ONNX) ---
    print(f"[WnD Startup] ONNX_DISPONIBLE={ONNX_DISPONIBLE}")
    if ONNX_DISPONIBLE:
        if os.path.exists(ruta_modelo_wnd) and os.path.exists(ruta_mapeos_wnd):
            try:
                with open(ruta_mapeos_wnd, "rb") as f:
                    wnd_mappings = pickle.load(f)
                num_users = len(wnd_mappings["user2idx"])
                num_movies = len(wnd_mappings["movie2idx"])
                print(
                    f"[ONNX Startup] Mappings WnD: {num_users} users, {num_movies} movies"
                )

                # Cargamos la sesión de inferencia de ONNX
                modelo_wnd = ort.InferenceSession(
                    ruta_modelo_wnd, providers=["CPUExecutionProvider"]
                )
                print("Modelo Wide&Deep ONNX cargado correctamente.")
            except Exception as e:
                print(f"ERROR al cargar el modelo: {e}")
                modelo_wnd = None
        else:
            print(f"Archivos no encontrados para WnD.")
    else:
        print("Saltando Wide&Deep porque onnxruntime no está instalado.")

    # --- Modelo 4: Content-Based ---
    if os.path.exists(ruta_tfidf_mat) and os.path.exists(ruta_tfidf_idx):
        try:
            modelo_tfidf_mat = joblib.load(ruta_tfidf_mat)
            modelo_tfidf_idx = joblib.load(ruta_tfidf_idx)
            print("Modelo TF-IDF cargado correctamente (Joblib).")
        except Exception as e:
            print(f"Error cargando TF-IDF: {e}")
    else:
        print(f"No se encontró el modelo TF-IDF.")

    # --- Modelo 5: Implicit BPR ---
    if os.path.exists(ruta_imp_mod) and os.path.exists(ruta_imp_dat):
        try:
            with open(ruta_imp_mod, "rb") as f:
                modelo_imp = pickle.load(f)
            with open(ruta_imp_dat, "rb") as f:
                modelo_imp_dat = pickle.load(f)
            print("Modelo Implicit cargado correctamente.")
        except Exception as e:
            print(f"Error cargando Implicit: {e}")
    else:
        print(f"No se encontró el modelo Implicit.")

    # --- Datos compartidos: Ratings y Catálogo ---
    if os.path.exists(ruta_ratings):
        df_ratings_ia = pd.read_csv(ruta_ratings)
        print(f"Ratings cargados: {len(df_ratings_ia):,} filas.")
    else:
        print(f"No se encontró {ruta_ratings}")

    if os.path.exists(ruta_catalogo):
        df_catalogo = pd.read_csv(ruta_catalogo, on_bad_lines="skip", engine="python")
        print(f"Catálogo cargado: {len(df_catalogo):,} películas.")
    else:
        print(f"No se encontró {ruta_catalogo}")


@app.get("/recomendar/{user_id}")
def recomendar_peliculas(user_id: int, n: int = 10):
    """
    Endpoint que devuelve las top-N películas recomendadas para un usuario.
    Usa el modelo SVD entrenado para predecir ratings de películas no vistas.
    """
    # Validamos que el modelo y los datos estén cargados
    if modelo_svd is None:
        raise HTTPException(
            status_code=503, detail="El modelo SVD no está cargado. Entrénalo primero."
        )
    if df_ratings_ia is None:
        raise HTTPException(
            status_code=503, detail="Los datos de ratings no están disponibles."
        )

    # 1. Películas que este usuario YA ha visto
    pelis_vistas = set(
        df_ratings_ia[df_ratings_ia["userId"] == user_id]["tmdb_id"].tolist()
    )

    # 2. Todas las películas disponibles en el sistema (aseguramos que parten del catálogo)
    if df_catalogo is not None:
        todas_las_pelis = set(df_catalogo["tmdb_id"].unique())
    else:
        todas_las_pelis = set(df_ratings_ia["tmdb_id"].unique())

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
        pred = modelo_svd.predict(user_id, tmdb_id)
        predicciones.append(
            {"tmdb_id": int(tmdb_id), "predicted_rating": round(pred.est, 2)}
        )

    # 5. Ordenamos de mayor a menor y nos quedamos con las top-N
    predicciones.sort(key=lambda x: x["predicted_rating"], reverse=True)
    top_n = predicciones[:n]

    # 6. Enriquecemos con datos del catálogo (título, poster, sinopsis, nota real)
    if df_catalogo is not None:
        for rec in top_n:
            match = df_catalogo[df_catalogo["tmdb_id"] == rec["tmdb_id"]]
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

    return {"recomendaciones": top_n}


##############################################################################################
#  Helper: Enriquecer recomendaciones con datos del catálogo
##############################################################################################


def enriquecer_recomendaciones(top_n):
    """Añade título, poster, sinopsis y nota real del catálogo a cada recomendación."""
    if df_catalogo is None:
        return
    for rec in top_n:
        match = df_catalogo[df_catalogo["tmdb_id"] == rec["tmdb_id"]]
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
    if modelo_knn is None:
        raise HTTPException(
            status_code=503, detail="El modelo KNN no está cargado. Entrénalo primero."
        )
    if df_ratings_ia is None:
        raise HTTPException(
            status_code=503, detail="Los datos de ratings no están disponibles."
        )

    pelis_vistas = set(
        df_ratings_ia[df_ratings_ia["userId"] == user_id]["tmdb_id"].tolist()
    )
    todas = (
        set(df_catalogo["tmdb_id"].unique())
        if df_catalogo is not None
        else set(df_ratings_ia["tmdb_id"].unique())
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
        pred = modelo_knn.predict(user_id, tmdb_id)
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
    if modelo_wnd is None or wnd_mappings is None:
        raise HTTPException(
            status_code=503, detail="El modelo Wide&Deep ONNX no está cargado."
        )
    if df_ratings_ia is None:
        raise HTTPException(
            status_code=503, detail="Los datos de ratings no están disponibles."
        )

    user2idx = wnd_mappings["user2idx"]
    movie2idx = wnd_mappings["movie2idx"]

    if user_id not in user2idx:
        return {
            "recomendaciones": [],
            "modelo": "Wide&Deep (ONNX)",
            "mensaje": f"El usuario {user_id} no cumplió el filtro de entrenamiento (>100 valoraciones).",
        }

    u_idx = user2idx[user_id]
    pelis_vistas = set(
        df_ratings_ia[df_ratings_ia["userId"] == user_id]["tmdb_id"].tolist()
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

    # Inferencia purísima en C++ (Zero Python Overhead) via ONNX Runtime
    inputs_onnx = {"user_id": user_arr, "movie_id": movie_arr}
    preds_onnx = modelo_wnd.run(["predicted_rating"], inputs_onnx)[0].flatten()

    # Restringir predicción a rango [0.5, 5.0]
    preds_onnx = np.clip(preds_onnx, 0.5, 5.0)

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
    if modelo_tfidf_mat is None or modelo_tfidf_idx is None:
        raise HTTPException(status_code=503, detail="Modelo TF-IDF no cargado.")
    if df_ratings_ia is None or df_catalogo is None:
        raise HTTPException(status_code=503, detail="Datos no cargados.")

    user_ratings = df_ratings_ia[df_ratings_ia["userId"] == user_id]

    # Cold Start (Usuario sin historial)
    if user_ratings.empty:
        # Recomendamos por popularidad general
        if "vote_count" in df_catalogo.columns:
            top_pop = (
                df_catalogo[df_catalogo["vote_count"] > 100]
                .sort_values(by="vote_average", ascending=False)
                .head(n)
            )
        else:
            top_pop = df_catalogo.head(n)

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
    fav = user_ratings.sort_values(by="rating", ascending=False).iloc[0]
    tid_fav = int(fav["tmdb_id"])

    if tid_fav not in modelo_tfidf_idx:
        return {
            "recomendaciones": [],
            "modelo": "Content-Based",
            "mensaje": "Película favorita no encontrada.",
        }

    idx_fav = modelo_tfidf_idx[tid_fav]
    vector_fav = modelo_tfidf_mat[idx_fav]

    from sklearn.metrics.pairwise import linear_kernel

    similitudes = linear_kernel(vector_fav, modelo_tfidf_mat).flatten()

    # Obtenemos los mas similares (ignorando el mismisimo 1.0)
    top_indices = similitudes.argsort()[::-1][1 : n + 1]

    predicciones = []
    for idx_sim in top_indices:
        peli = df_catalogo.iloc[idx_sim]
        score_sim = similitudes[idx_sim]
        # Creamos un rating visual combinando su rating original con la similitud
        pseudo_rating = min(5.0, fav["rating"] * (0.8 + 0.2 * score_sim))
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
    if modelo_imp is None or modelo_imp_dat is None:
        raise HTTPException(
            status_code=503, detail="Modelo Implicit BPR no está cargado."
        )
    if df_ratings_ia is None:
        raise HTTPException(
            status_code=503, detail="Los datos de ratings no están disponibles."
        )

    user2idx = modelo_imp_dat["user2idx"]
    item2idx = modelo_imp_dat["item2idx"]
    idx2item = {v: k for k, v in item2idx.items()}

    if user_id not in user2idx:
        return {
            "recomendaciones": [],
            "modelo": "Implicit BPR",
            "mensaje": f"El usuario {user_id} no tiene historial en el modelo BPR.",
        }

    u_idx = user2idx[user_id]

    # Extraemos arrays nativos mediante NumPy para evitar el bug Cython de implicit.recommend() en Windows!
    u_factors = np.asarray(modelo_imp.user_factors[u_idx])
    i_factors = np.asarray(modelo_imp.item_factors)

    # Producto escalar vectorizado (Calcula score para tooooodas las peliculas del modelo a la vez)
    scores = u_factors.dot(i_factors.T)

    # Localizamos las que ya ha visto en el dataset completo
    pelis_vistas = set(
        df_ratings_ia[df_ratings_ia["userId"] == user_id]["tmdb_id"].tolist()
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
