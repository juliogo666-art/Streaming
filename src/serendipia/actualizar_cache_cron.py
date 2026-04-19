"""
actualizar_cache_cron.py
========================
Job semanal que recalcula los scores de serendipia y vuelca el resultado
en la tabla ``serendipity_cache`` de la base de datos.

Flujo
-----
1. Lee ``dataset_final_movies.csv`` desde ``src/data/ready/``.
2. Normaliza el DataFrame (explode géneros, mapea IDs → nombres).
3. Calcula Media Bayesiana y Serendipity Score via :mod:`calculadora_serendipia`.
4. Extrae el Top-1000 por género.
5. TRUNCATE + INSERT en la tabla ``serendipity_cache``.

Uso (cron — todos los domingos a las 03:00):
    0 3 * * 0  cd /ruta/proyecto && python -m src.serendipia.actualizar_cache_cron

Uso manual:
    python -m src.serendipia.actualizar_cache_cron
"""

import ast
import logging
import sys
from pathlib import Path

import pandas as pd

from src.serendipia.calculadora_serendipia import (
    CSV_PATH,
    TMDB_GENRES,
    TOP_N,
    calcular_serendipity,
    top_por_genero,
)
try:
    from src.api.database import get_db_connection
except ModuleNotFoundError:
    # Fallback cuando el script se ejecuta directamente (no como módulo).
    # Añadimos la raíz del proyecto al Python path para permitir imports absolutos "src.*".
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    from src.api.database import get_db_connection

# ---------------------------------------------------------------------------
# Logging — salida a consola + fichero de log
# ---------------------------------------------------------------------------

_log_dir = Path("logs")
_log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_log_dir / "cron_serendipia.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("serendipia.cron")

# ---------------------------------------------------------------------------
# Constantes internas
# ---------------------------------------------------------------------------

INSERT_BATCH_SIZE: int = 500  # Filas por lote en executemany


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------


def _cargar_y_preparar_csv(csv_path: Path) -> pd.DataFrame:
    """Carga el CSV real y lo normaliza al esquema esperado por la calculadora.

    Args:
        csv_path: Ruta absoluta al archivo ``dataset_final_movies.csv``.

    Returns:
        DataFrame con columnas ``[movie_id, genre, rating_mean, vote_count]``.
    """
    logger.info("Leyendo CSV: %s", csv_path)
    raw = pd.read_csv(csv_path)

    # Parsear genre_ids de string "[27, 878]" → list[int] y explotar
    raw["genre_ids"] = raw["genre_ids"].apply(ast.literal_eval)
    exploded = raw.explode("genre_ids").copy()
    exploded["genre_ids"] = exploded["genre_ids"].astype(int)

    # Mapear ID numérico → nombre de género TMDB
    exploded["genre"] = exploded["genre_ids"].map(TMDB_GENRES)
    exploded = exploded.dropna(subset=["genre"])

    df = (
        exploded
        .rename(columns={"tmdb_id": "movie_id", "vote_average": "rating_mean"})
        [["movie_id", "genre", "rating_mean", "vote_count"]]
        .dropna()
        .copy()
    )

    # Filtrado por género con umbral adaptativo:
    #   - Se aplica MIN_VOTES=10 como umbral global de votos mínimos.
    #   - Si un género no alcanza GENRE_MIN_MOVIES películas tras el filtro,
    #     se relaja el umbral para ese género hasta incluir las
    #     GENRE_MIN_MOVIES películas con más votos disponibles.
    # Esto garantiza cobertura de todos los géneros TMDB sin sacrificar
    # la calidad estadística en géneros grandes.
    MIN_VOTES: int = 10
    GENRE_MIN_MOVIES: int = 10

    # Excluir siempre películas sin rating real
    df = df[df["rating_mean"] > 0].copy()

    partes: list[pd.DataFrame] = []
    for genre, grupo in df.groupby("genre"):
        candidatos = grupo[grupo["vote_count"] >= MIN_VOTES]
        if len(candidatos) < GENRE_MIN_MOVIES:
            # Relajar umbral: tomar las GENRE_MIN_MOVIES con más votos
            candidatos = grupo.nlargest(GENRE_MIN_MOVIES, "vote_count")
            logger.warning(
                "Género '%s': solo %d películas con vote_count>=%d. "
                "Usando top-%d por votos (mín votos real: %d).",
                genre,
                len(grupo[grupo["vote_count"] >= MIN_VOTES]),
                MIN_VOTES,
                GENRE_MIN_MOVIES,
                int(candidatos["vote_count"].min()),
            )
        partes.append(candidatos)

    df = pd.concat(partes, ignore_index=True)
    logger.info("CSV preparado: %d filas (película × género).", len(df))
    return df


def _vaciar_tabla(conexion) -> None:
    """Vacía la tabla ``serendipity_cache`` con TRUNCATE para el volcado fresco.

    Args:
        conexion: Conexión activa a MySQL obtenida con :func:`get_db_connection`.
    """
    cursor = conexion.cursor()
    cursor.execute("TRUNCATE TABLE serendipity_cache;")
    conexion.commit()
    cursor.close()
    logger.info("Tabla serendipity_cache vaciada (TRUNCATE).")


def _insertar_en_lotes(conexion, df: pd.DataFrame) -> int:
    """Inserta el DataFrame en ``serendipity_cache`` en lotes para mejor rendimiento.

    Args:
        conexion: Conexión activa a MySQL.
        df:       DataFrame con columnas
                  ``[movie_id, genre, rating_mean, vote_count,
                  weighted_rating, serendipity_score]``.

    Returns:
        Número total de registros insertados.
    """
    cursor = conexion.cursor()
    sql = """
        INSERT INTO serendipity_cache
            (movie_id, genre, rating_mean, vote_count, weighted_rating, serendipity_score)
        VALUES (%s, %s, %s, %s, %s, %s)
    """

    registros = [
        (
            int(row.movie_id),
            str(row.genre),
            float(row.rating_mean),
            int(row.vote_count),
            float(row.weighted_rating),
            float(row.serendipity_score),
        )
        for row in df.itertuples(index=False)
    ]

    total_insertados = 0
    for i in range(0, len(registros), INSERT_BATCH_SIZE):
        lote = registros[i : i + INSERT_BATCH_SIZE]
        cursor.executemany(sql, lote)
        conexion.commit()
        total_insertados += len(lote)
        logger.debug("Lote insertado: %d/%d registros.", total_insertados, len(registros))

    cursor.close()
    return total_insertados


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------


def ejecutar_volcado() -> None:
    """Orquesta el proceso completo de actualización de la caché de serendipia.

    Secuencia:
        1. Carga y normaliza el CSV de películas.
        2. Calcula los scores de serendipia.
        3. Extrae el Top-1000 por género.
        4. TRUNCATE + INSERT masivo en ``serendipity_cache``.
    """
    logger.info("=" * 60)
    logger.info("INICIO DE VOLCADO — Actualización caché serendipia")
    logger.info("=" * 60)

    # --- 1. Cargar y preparar datos ---
    df_raw = _cargar_y_preparar_csv(CSV_PATH)

    # --- 2. Calcular métricas ---
    logger.info("Calculando métricas de serendipia...")
    df_scored = calcular_serendipity(df_raw)
    df_top = top_por_genero(df_scored, n=TOP_N)
    logger.info(
        "Métricas calculadas: %d registros para %d géneros.",
        len(df_top),
        df_top["genre"].nunique(),
    )

    # --- 3. Conectar a la BD y volcar ---
    logger.info("Conectando a la base de datos...")
    conexion = get_db_connection()
    try:
        _vaciar_tabla(conexion)
        total_insertados = _insertar_en_lotes(conexion, df_top)
    finally:
        conexion.close()
        logger.info("Conexión a la BD cerrada.")

    logger.info("=" * 60)
    logger.info("VOLCADO COMPLETADO con %d registros en serendipity_cache.", total_insertados)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ejecutar_volcado()
