"""
Router de Serendipia — Tragaperras cinéfila con joyas ocultas.
"""

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from ..database import get_db_connection

logger = logging.getLogger("streaming_api")
router = APIRouter()


def _get_app_state():
    from ..main_api import app
    return app.state


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


@router.get("/api/serendipia/{user_id}")
def tragaperras_serendipia(user_id: int):
    """Devuelve 3 películas 'joya oculta' seleccionadas con muestreo ponderado
    por serendipity_score, personalizado con los 2 géneros favoritos del usuario."""
    state = _get_app_state()
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

        # 2. Recuperar candidatos pre-calculados de la caché
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
    df_ratings = getattr(state, "df_ratings_ia", None)
    if df_ratings is not None:
        ya_puntuadas = set(df_ratings[df_ratings["userId"] == user_id]["tmdb_id"].tolist())
        df = df[~df["movie_id"].isin(ya_puntuadas)]

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"El usuario {user_id} ya ha puntuado todas las películas de sus géneros favoritos.",
        )

    # 4. Muestreo ponderado sin reemplazo
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

    logger.info(f"[Serendipia] User {user_id} | Géneros: {generos_es} → {generos}")

    return {
        "user_id": user_id,
        "generos_favoritos": generos,
        "recomendaciones": recomendaciones,
    }
