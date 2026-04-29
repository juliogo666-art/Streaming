"""
Router de Valoraciones — CRUD de ratings de usuario y detalle de películas.
"""

import logging

from fastapi import APIRouter, HTTPException

from ..database import get_db_connection
from ...schemas.schemas import RatingRequest

logger = logging.getLogger("streaming_api")
router = APIRouter()


def _get_app_state():
    from ..main_api import app
    return app.state


@router.post("/api/rating")
def registrar_valoracion(datos: RatingRequest):
    """Registra o actualiza la valoración de un usuario sobre una película."""
    if datos.rating < 0.5 or datos.rating > 5.0:
        raise HTTPException(status_code=400, detail="La valoración debe estar entre 0.5 y 5.0")
    if (datos.rating * 2) != int(datos.rating * 2):
        raise HTTPException(status_code=400, detail="La valoración debe ser en pasos de 0.5")

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


@router.get("/api/ratings/{user_id}")
def obtener_valoraciones_usuario(user_id: int):
    """Devuelve todas las valoraciones que un usuario ha dado."""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            "SELECT tmdb_id, rating FROM user_ratings WHERE id_usuario = %s",
            (user_id,),
        )
        filas = cursor.fetchall()
        ratings_dict = {row["tmdb_id"]: float(row["rating"]) for row in filas}
        return {"user_id": user_id, "ratings": ratings_dict}
    except Exception as e:
        logger.error(f"[RATING] Error al obtener valoraciones de user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


@router.get("/api/movie/{tmdb_id}")
def obtener_detalle_pelicula(tmdb_id: int):
    """Devuelve los datos completos de una película del catálogo en memoria."""
    state = _get_app_state()
    if state.df_catalogo is None:
        raise HTTPException(status_code=503, detail="Catálogo no cargado.")

    match = state.df_catalogo[state.df_catalogo["tmdb_id"] == tmdb_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Película {tmdb_id} no encontrada.")

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
