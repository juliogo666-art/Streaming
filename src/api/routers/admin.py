"""
Router de Administración — Importación de datos, listado de usuarios y géneros.
"""

import logging
import re

import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException

from ..database import get_db_connection
from ..etl import ejecutar_importacion, limpiar_tablas_contenido

logger = logging.getLogger("streaming_api")

router = APIRouter()


def importar_series(conn):
    print("inicio importacion series")
    # Importamos Series
    ejecutar_importacion(conn, "src/data/clean/tmdb_shows_limpio.csv", "tv")
    print("fin importacion series")


def importar_peliculas(conn):
    print("inicio importacion peliculas, se viene lo chungo")
    # Importamos Películas
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


@router.get("/usuarios")
def obtener_usuarios():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM users")
        usuarios = cursor.fetchall()
        return usuarios
    except Exception as e:
        logger.error(f"[ERROR] Error al obtener usuarios: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        cursor.close()
        conn.close()


@router.get("/genres")
def obtener_generos():
    logger.info("[API] Petición a /genres recibida")
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, name FROM genres ORDER BY name ASC")
        generos = cursor.fetchall()
        logger.info(f"[API] Se encontraron {len(generos)} géneros.")
        return generos
    except Exception as e:
        logger.error(f"[ERROR] Error al obtener géneros: {e}")
        # Intentamos dar un mensaje más útil que un 500 genérico
        raise HTTPException(
            status_code=500, detail=f"Error en base de datos al leer géneros: {str(e)}"
        )
    finally:
        cursor.close()
        conn.close()


@router.post("/importar_datos")
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
