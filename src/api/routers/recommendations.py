"""
Router de Recomendaciones — Todos los endpoints de modelos de IA.
Incluye: SVD, KNN, Wide&Deep, Content-Based, Implicit BPR, NCF, TwoTowers, Smart.
"""

import re
import time
import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from sklearn.metrics.pairwise import cosine_similarity

from ..database import get_db_connection
from ...schemas.recommendation import RecommendationResponse
from ...tracking.logger import RecommendationLogger
from ...config.settings import UMBRAL_COLD_START, UMBRAL_AVANZADO

logger = logging.getLogger("streaming_api")
router = APIRouter()
telemetria = RecommendationLogger()


def _get_app_state():
    """Accede al estado de la app FastAPI a través del router."""
    from ..main_api import app
    return app.state


def _log_recomendacion_con_tiempo(
    user_id: int, modelo: str, recomendaciones_top_n: list, inicio: float
) -> None:
    """Registra telemetría de recomendación junto con su tiempo de respuesta."""
    tiempo_ms = round((time.perf_counter() - inicio) * 1000, 2)
    telemetria.log_recommendations(
        str(user_id), modelo, recomendaciones_top_n, tiempo_recomendacion_ms=tiempo_ms,
    )


def enriquecer_recomendaciones(recomendaciones):
    """Añade datos del catálogo (título, poster, etc.) a una lista de diccionarios con tmdb_id."""
    state = _get_app_state()
    if state.df_catalogo is not None:
        for rec in recomendaciones:
            match = state.df_catalogo[state.df_catalogo["tmdb_id"] == rec["tmdb_id"]]
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


# ── SVD ──────────────────────────────────────────────────────────────────────

@router.get("/recomendar/svd/{user_id}", response_model=RecommendationResponse)
@router.get("/recomendar/{user_id}", response_model=RecommendationResponse)
def recomendar_peliculas(user_id: int, n: int = 10):
    """Recomendaciones usando SVD (Surprise)."""
    t_inicio = time.perf_counter()
    state = _get_app_state()

    if state.modelo_svd is None:
        raise HTTPException(status_code=503, detail="Modelo SVD no cargado.")
    if state.df_ratings_ia is None:
        raise HTTPException(status_code=503, detail="CSV de ratings no cargado.")

    pelis_vistas = set(
        state.df_ratings_ia[state.df_ratings_ia["userId"] == user_id]["tmdb_id"].tolist()
    )
    todas_las_pelis = (
        set(state.df_catalogo["tmdb_id"].unique())
        if state.df_catalogo is not None
        else set(state.df_ratings_ia["tmdb_id"].unique())
    )
    pelis_no_vistas = todas_las_pelis - pelis_vistas

    if not pelis_no_vistas:
        return {
            "recomendaciones": [],
            "modelo": "SVD (Surprise)",
            "mensaje": "Ya lo has visto todo, ¡felicidades!!!",
        }

    predicciones = []
    for tmdb_id in pelis_no_vistas:
        pred = state.modelo_svd.predict(user_id, tmdb_id)
        predicciones.append({"tmdb_id": int(tmdb_id), "predicted_rating": round(pred.est, 2)})

    predicciones.sort(key=lambda x: x["predicted_rating"], reverse=True)
    top_n = predicciones[:n]
    enriquecer_recomendaciones(top_n)
    _log_recomendacion_con_tiempo(user_id, "SVD (Surprise)", top_n, t_inicio)
    return {"recomendaciones": top_n, "modelo": "SVD (Surprise)"}


# ── KNN ──────────────────────────────────────────────────────────────────────

@router.get("/recomendar/knn/{user_id}", response_model=RecommendationResponse)
def recomendar_knn(user_id: int, n: int = 10):
    """Recomendaciones usando KNN + Cosine Similarity."""
    t_inicio = time.perf_counter()
    state = _get_app_state()

    if state.modelo_knn is None:
        raise HTTPException(status_code=503, detail="Modelo KNN no cargado.")
    if state.df_ratings_ia is None:
        raise HTTPException(status_code=503, detail="CSV de ratings no cargado.")

    pelis_vistas = set(
        state.df_ratings_ia[state.df_ratings_ia["userId"] == user_id]["tmdb_id"].tolist()
    )
    todas = (
        set(state.df_catalogo["tmdb_id"].unique())
        if state.df_catalogo is not None
        else set(state.df_ratings_ia["tmdb_id"].unique())
    )
    pelis_no_vistas = todas - pelis_vistas

    if not pelis_no_vistas:
        return {
            "recomendaciones": [],
            "modelo": "KNN+Cosine",
            "mensaje": "Ya lo has visto todo, ¡felicidades!!!",
        }

    predicciones = []
    for tmdb_id in pelis_no_vistas:
        pred = state.modelo_knn.predict(user_id, tmdb_id)
        predicciones.append({"tmdb_id": int(tmdb_id), "predicted_rating": round(pred.est, 2)})

    predicciones.sort(key=lambda x: x["predicted_rating"], reverse=True)
    top_n = predicciones[:n]
    enriquecer_recomendaciones(top_n)
    _log_recomendacion_con_tiempo(user_id, "KNN+Cosine", top_n, t_inicio)
    return {"recomendaciones": top_n, "modelo": "KNN+Cosine"}


# ── Wide & Deep (ONNX) ──────────────────────────────────────────────────────

@router.get("/recomendar/wnd/{user_id}", response_model=RecommendationResponse)
def recomendar_wnd_endpoint(user_id: int, n: int = 10):
    """Recomendaciones usando Wide & Deep Neural Network (ONNX)."""
    t_inicio = time.perf_counter()
    state = _get_app_state()

    if state.modelo_wnd is None or state.wnd_mappings is None:
        raise HTTPException(status_code=503, detail="Modelo Wide&Deep no cargado.")
    if state.df_ratings_ia is None:
        raise HTTPException(status_code=503, detail="CSV de ratings no cargado.")

    user2idx = state.wnd_mappings["user2idx"]
    movie2idx = state.wnd_mappings["movie2idx"]

    if user_id not in user2idx:
        count = state.user_counts.get(user_id, 0)
        return {
            "recomendaciones": [],
            "modelo": "Wide&Deep (ONNX)",
            "mensaje": f"No alcanzas las 100 valoraciones requeridas ({count}/100).",
            "insufficient_data": True,
        }

    u_idx = user2idx[user_id]
    pelis_vistas = set(
        state.df_ratings_ia[state.df_ratings_ia["userId"] == user_id]["tmdb_id"].tolist()
    )
    candidatas = [(tid, midx) for tid, midx in movie2idx.items() if tid not in pelis_vistas]
    if not candidatas:
        return {
            "recomendaciones": [],
            "modelo": "Wide&Deep (ONNX)",
            "mensaje": "Ya lo has visto todo, ¡felicidades!!!",
        }

    tmdb_ids, movie_indices = zip(*candidatas)
    user_arr = np.array([u_idx] * len(movie_indices), dtype=np.int64)
    movie_arr = np.array(list(movie_indices), dtype=np.int64)

    try:
        ort_inputs = {"user_ids": user_arr, "item_ids": movie_arr}
        preds_raw = state.modelo_wnd.run(None, ort_inputs)[0].flatten()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia WnD: {e}")

    def sigmoid(x):
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

    preds_prob = sigmoid(preds_raw)
    predicciones = [
        {"tmdb_id": int(tid), "predicted_rating": round(float(preds_prob[i] * 4.5 + 0.5), 2)}
        for i, tid in enumerate(tmdb_ids)
    ]
    predicciones.sort(key=lambda x: x["predicted_rating"], reverse=True)
    top_n = predicciones[:n]
    enriquecer_recomendaciones(top_n)
    _log_recomendacion_con_tiempo(user_id, "Wide&Deep (ONNX)", top_n, t_inicio)
    return {"recomendaciones": top_n, "modelo": "Wide&Deep (ONNX)"}


# ── Content-Based / Cold Start ───────────────────────────────────────────────

@router.get("/recomendar/content/{user_id}", response_model=RecommendationResponse)
def recomendar_content_endpoint(user_id: int, n: int = 10):
    """Recomendaciones por contenido (TF-IDF) con Cold Start."""
    t_inicio = time.perf_counter()
    state = _get_app_state()

    if state.modelo_tfidf_mat is None or state.modelo_tfidf_idx is None:
        raise HTTPException(status_code=503, detail="Modelo TF-IDF no cargado.")
    if state.df_ratings_ia is None or state.df_catalogo is None:
        raise HTTPException(status_code=503, detail="Datos de ratings/catálogo no cargados.")

    user_ratings = state.df_ratings_ia[state.df_ratings_ia["userId"] == user_id]

    # Cold Start (Usuario sin historial)
    if user_ratings.empty:
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

        if intereses_usuario:
            patron = r'\b(?:' + '|'.join(map(str, intereses_usuario)) + r')\b'
            mask = state.df_catalogo["genre_ids"].str.contains(patron, na=False, regex=True)
            catalogo_filtrado = state.df_catalogo[mask]
            nomb_modelo = "TF-IDF (Cold Start Géneros)"
        else:
            catalogo_filtrado = state.df_catalogo
            nomb_modelo = "TF-IDF (Cold Start Populares)"

        if catalogo_filtrado.empty:
            catalogo_filtrado = state.df_catalogo

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

        recomendaciones = [
            {"tmdb_id": int(row["tmdb_id"]), "predicted_rating": 4.5}
            for _, row in top_pop.iterrows()
        ]
        enriquecer_recomendaciones(recomendaciones)
        _log_recomendacion_con_tiempo(user_id, nomb_modelo, recomendaciones, t_inicio)
        return {"recomendaciones": recomendaciones, "modelo": nomb_modelo}

    # Usuario con historial -> Content-Based por similitud
    viewed_ids = user_ratings["tmdb_id"].tolist()
    tfidf_indices = [
        state.modelo_tfidf_idx[mid] for mid in viewed_ids if mid in state.modelo_tfidf_idx
    ]
    if not tfidf_indices:
        return {
            "recomendaciones": [],
            "modelo": "Content-Based",
            "mensaje": "Película favorita no encontrada.",
        }

    user_profile = np.asarray(state.modelo_tfidf_mat[tfidf_indices].mean(axis=0))
    cos_sim = cosine_similarity(user_profile, state.modelo_tfidf_mat).flatten()
    top_indices = cos_sim.argsort()[::-1][1: n + 1]

    predicciones = []
    for idx_sim in top_indices:
        peli = state.df_catalogo.iloc[idx_sim]
        score_sim = cos_sim[idx_sim]
        pseudo_rating = min(5.0, 4.0 * (0.8 + 0.2 * score_sim))
        predicciones.append(
            {"tmdb_id": int(peli["tmdb_id"]), "predicted_rating": round(pseudo_rating, 2)}
        )
    enriquecer_recomendaciones(predicciones)
    _log_recomendacion_con_tiempo(user_id, "Content-Based", predicciones, t_inicio)
    return {"recomendaciones": predicciones, "modelo": "Content-Based"}


# ── Implicit BPR ─────────────────────────────────────────────────────────────

@router.get("/recomendar/implicit/{user_id}", response_model=RecommendationResponse)
def recomendar_implicit_endpoint(user_id: int, n: int = 10):
    """Recomendaciones usando filtrado colaborativo BPR (implicit)."""
    t_inicio = time.perf_counter()
    state = _get_app_state()

    if state.modelo_imp is None or state.modelo_imp_dat is None:
        raise HTTPException(status_code=503, detail="Modelo Implicit BPR no cargado.")
    if state.df_ratings_ia is None:
        raise HTTPException(status_code=503, detail="CSV de ratings no cargado.")

    user2idx = state.modelo_imp_dat["user2idx"]
    item2idx = state.modelo_imp_dat["item2idx"]
    idx2item = {v: k for k, v in item2idx.items()}

    if user_id not in user2idx:
        return {
            "recomendaciones": [],
            "modelo": "Implicit BPR",
            "mensaje": f"El usuario {user_id} no tiene historial en el modelo BPR.",
        }

    u_idx = user2idx[user_id]
    u_factors = np.asarray(state.modelo_imp.user_factors[u_idx])
    i_factors = np.asarray(state.modelo_imp.item_factors)
    scores = u_factors.dot(i_factors.T)

    pelis_vistas = set(
        state.df_ratings_ia[state.df_ratings_ia["userId"] == user_id]["tmdb_id"].tolist()
    )
    for tid in pelis_vistas:
        if tid in item2idx:
            scores[item2idx[tid]] = -np.inf

    top_indices = np.argsort(scores)[::-1][:n]
    predicciones = []
    for idx_sim in top_indices:
        tid = idx2item[idx_sim]
        score_puro = float(scores[idx_sim])
        rating_ui = min(5.0, max(0.5, 3.5 + (score_puro * 0.2)))
        predicciones.append({"tmdb_id": int(tid), "predicted_rating": round(rating_ui, 2)})

    enriquecer_recomendaciones(predicciones)
    _log_recomendacion_con_tiempo(user_id, "Implicit BPR", predicciones, t_inicio)
    return {"recomendaciones": predicciones, "modelo": "Implicit BPR"}


# ── NCF (Neural Collaborative Filtering) ────────────────────────────────────

@router.get("/recomendar/ncf/{user_id}", response_model=RecommendationResponse)
def recomendar_ncf_endpoint(user_id: int, n: int = 10):
    """Recomendaciones usando NCF-Lite (GMF + MLP) via ONNX Runtime."""
    t_inicio = time.perf_counter()
    state = _get_app_state()

    if state.modelo_ncf is None:
        raise HTTPException(status_code=503, detail="Modelo NCF no cargado.")
    if state.ncf_user2idx is None or state.ncf_item2idx is None:
        raise HTTPException(status_code=503, detail="Mapeos NCF no cargados.")

    if user_id not in state.ncf_user2idx:
        count = state.user_counts.get(user_id, 0)
        return {
            "recomendaciones": [],
            "modelo": "NCF",
            "mensaje": f"No alcanzas las 100 valoraciones requeridas ({count}/100).",
            "insufficient_data": True,
        }

    user_idx = state.ncf_user2idx[user_id]

    # BUG-6 FIX: Construir mapeo explícito de posición en el array → item_idx
    item_keys = list(state.ncf_item2idx.keys())     # tmdb_ids originales
    item_values = list(state.ncf_item2idx.values())  # índices internos del modelo
    n_items = len(item_keys)

    try:
        user_ids_np = np.full(n_items, user_idx, dtype=np.int64)
        item_ids_np = np.array(item_values, dtype=np.int64)

        scores = state.modelo_ncf.run(
            None, {"user_ids": user_ids_np, "item_ids": item_ids_np}
        )[0].flatten()

        # Excluir items ya vistos usando el índice de posición en el array (no el valor del mapping)
        if state.df_ratings_ia is not None:
            user_ratings = state.df_ratings_ia[state.df_ratings_ia["userId"] == user_id]
            pelis_vistas = set(user_ratings["tmdb_id"].unique())
            for pos, tid in enumerate(item_keys):
                if tid in pelis_vistas:
                    scores[pos] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n]

        predicciones = []
        for pos in top_indices:
            if scores[pos] == -np.inf:
                continue
            tid = item_keys[pos]
            score_puro = float(scores[pos])
            rating_ui = min(5.0, max(0.5, 3.5 + (score_puro * 0.3)))
            predicciones.append({"tmdb_id": int(tid), "predicted_rating": round(rating_ui, 2)})

        enriquecer_recomendaciones(predicciones)
        _log_recomendacion_con_tiempo(user_id, "NCF-Lite", predicciones, t_inicio)
        return {"recomendaciones": predicciones, "modelo": "NCF-Lite"}

    except Exception as e:
        logger.error(f"Error en recomendación NCF para User {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Two Towers ───────────────────────────────────────────────────────────────

@router.get("/recomendar/twotowers/{user_id}", response_model=RecommendationResponse)
def recomendar_tt_endpoint(user_id: int, n: int = 10):
    """Recomendaciones usando Two Towers Neural Network (ONNX)."""
    t_inicio = time.perf_counter()
    state = _get_app_state()

    if state.modelo_tt is None or state.tt_mappings is None:
        raise HTTPException(status_code=503, detail="Modelo TwoTowers no cargado.")

    user2idx = state.tt_mappings["user2idx"]
    item2idx = state.tt_mappings["item2idx"]
    idx2item = {v: k for k, v in item2idx.items()}

    if user_id not in user2idx:
        count = state.user_counts.get(user_id, 0)
        return {
            "recomendaciones": [],
            "modelo": "Two-Towers",
            "mensaje": f"No alcanzas las 50 valoraciones requeridas ({count}/50).",
            "insufficient_data": True,
        }

    u_idx = user2idx[user_id]
    tids_candidatos = list(item2idx.keys())
    i_indices = list(item2idx.values())
    user_arr = np.full(len(i_indices), u_idx, dtype=np.int64)
    item_arr = np.array(i_indices, dtype=np.int64)

    try:
        ort_inputs = {"user_ids": user_arr, "item_ids": item_arr}
        scores = state.modelo_tt.run(None, ort_inputs)[0].flatten()
    except Exception as e:
        logger.error(f"Error TwoTowers inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if state.df_ratings_ia is not None:
        vistas = set(
            state.df_ratings_ia[state.df_ratings_ia["userId"] == user_id]["tmdb_id"]
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
        s = float(scores[idx])
        rating_ui = min(5.0, max(0.5, 3.5 + (s * 0.1)))
        predicciones.append({"tmdb_id": int(tid), "predicted_rating": round(rating_ui, 2)})

    enriquecer_recomendaciones(predicciones)
    _log_recomendacion_con_tiempo(user_id, "Two-Towers", predicciones, t_inicio)
    return {"recomendaciones": predicciones, "modelo": "Two-Towers"}


# ── SMART: Selector Dinámico ─────────────────────────────────────────────────

@router.get("/recomendar/smart/{user_id}", response_model=RecommendationResponse)
def recomendar_smart(user_id: int, n: int = 10):
    """Selector inteligente de modelo de recomendación."""
    state = _get_app_state()
    n_ratings = (
        state.user_counts.get(user_id, 0)
        if hasattr(state, "user_counts") and state.user_counts
        else 0
    )
    logger.info(f"[SMART] User {user_id} tiene {n_ratings} valoraciones.")

    # NIVEL 1: Cold Start (0-10 ratings)
    if n_ratings <= UMBRAL_COLD_START:
        resultado = recomendar_content_endpoint(user_id, n)
        if isinstance(resultado, dict):
            resultado["selector"] = f"Smart → Content-Based (Cold Start: {n_ratings} valoraciones)"
        return resultado

    # NIVEL 2: Usuario Intermedio (11-99 ratings)
    if n_ratings < UMBRAL_AVANZADO:
        if state.modelo_svd is not None:
            resultado = recomendar_peliculas(user_id, n)
            if isinstance(resultado, dict):
                resultado["selector"] = f"Smart → SVD (Intermedio: {n_ratings}/100 valoraciones)"
            return resultado
        if state.modelo_knn is not None:
            resultado = recomendar_knn(user_id, n)
            if isinstance(resultado, dict):
                resultado["selector"] = f"Smart → KNN (Fallback Intermedio: {n_ratings} valoraciones)"
            return resultado
        if state.modelo_imp is not None:
            resultado = recomendar_implicit_endpoint(user_id, n)
            if isinstance(resultado, dict):
                resultado["selector"] = f"Smart → BPR (Fallback Intermedio: {n_ratings} valoraciones)"
            return resultado
        resultado = recomendar_content_endpoint(user_id, n)
        if isinstance(resultado, dict):
            resultado["selector"] = "Smart → Content-Based (Sin modelos colaborativos)"
        return resultado

    # NIVEL 3: Usuario Experto (100+ ratings)
    if state.modelo_ncf is not None and state.ncf_user2idx is not None:
        if user_id in state.ncf_user2idx:
            resultado = recomendar_ncf_endpoint(user_id, n)
            if isinstance(resultado, dict):
                resultado["selector"] = f"Smart → NCF-Lite (Experto: {n_ratings} valoraciones)"
            return resultado

    if state.modelo_wnd is not None and state.wnd_mappings is not None:
        user2idx_wnd = state.wnd_mappings.get("user2idx", {})
        if user_id in user2idx_wnd:
            resultado = recomendar_wnd_endpoint(user_id, n)
            if isinstance(resultado, dict):
                resultado["selector"] = f"Smart → Wide&Deep (Experto fallback: {n_ratings} valoraciones)"
            return resultado

    if state.modelo_svd is not None:
        resultado = recomendar_peliculas(user_id, n)
        if isinstance(resultado, dict):
            resultado["selector"] = f"Smart → SVD (Fallback Experto: {n_ratings} valoraciones)"
        return resultado

    resultado = recomendar_content_endpoint(user_id, n)
    if isinstance(resultado, dict):
        resultado["selector"] = "Smart → Content-Based (Último recurso)"
    return resultado
