"""
calculadora_serendipia.py
=========================
Lógica core del cálculo del Score de Serendipia para la "Tragaperras".

Fórmulas aplicadas
------------------
WR (Weighted Rating — Media Bayesiana):
    WR = (v * R + m * C) / (v + m)
    donde:
        v  = vote_count  (número de votos de la película)
        R  = rating_mean (nota media de la película)
        m  = 50          (umbral mínimo de votos, constante)
        C  = media global de rating_mean de TODO el catálogo

Serendipity Score:
    serendipity_score = WR / log10(v + 10)
    Castiga a las superproducciones con muchos votos para aflorar "joyas ocultas".

Uso standalone (modo desarrollo):
    python -m src.serendipia.calculadora_serendipia
"""

import ast
import logging
import math
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

M: int = 50        # Umbral mínimo de votos para la Media Bayesiana
TOP_N: int = 1000  # Número de candidatos a conservar por género

# Catálogo oficial de géneros TMDB (ID → Nombre)
TMDB_GENRES: dict[int, str] = {
    28:    "Action",
    12:    "Adventure",
    16:    "Animation",
    35:    "Comedy",
    80:    "Crime",
    99:    "Documentary",
    18:    "Drama",
    10751: "Family",
    14:    "Fantasy",
    36:    "History",
    27:    "Horror",
    10402: "Music",
    9648:  "Mystery",
    10749: "Romance",
    878:   "Science Fiction",
    10770: "TV Movie",
    53:    "Thriller",
    10752: "War",
    37:    "Western",
}

# Ruta al CSV real de películas (relativa a la raíz del proyecto)
CSV_PATH: Path = (
    Path(__file__).resolve().parents[2] / "src" / "data" / "ready" / "dataset_final_movies.csv"
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("serendipia.calculadora")


# ---------------------------------------------------------------------------
# Funciones públicas
# ---------------------------------------------------------------------------


def calcular_serendipity(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las fórmulas de Media Bayesiana y Serendipity Score al DataFrame.

    Columnas requeridas en df: ``movie_id``, ``genre``, ``rating_mean``, ``vote_count``.
    Columnas añadidas al retorno: ``weighted_rating``, ``serendipity_score``.

    Args:
        df: DataFrame con los datos de películas ya normalizados.

    Returns:
        Copia del DataFrame enriquecida con ``weighted_rating`` y
        ``serendipity_score``.

    Raises:
        ValueError: Si faltan columnas requeridas en el DataFrame.
    """
    required = {"movie_id", "genre", "rating_mean", "vote_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en el DataFrame: {missing}")

    df = df.copy()
    df = df.dropna(subset=["rating_mean", "vote_count"])
    df["vote_count"] = df["vote_count"].astype(int)
    df["rating_mean"] = df["rating_mean"].astype(float)

    # C = Media global de rating_mean de todo el catálogo
    C: float = df["rating_mean"].mean()
    logger.info("Media global del catálogo (C) = %.4f  |  Películas únicas: %d", C, df["movie_id"].nunique())

    v = df["vote_count"]
    R = df["rating_mean"]

    # WR = (v * R + m * C) / (v + m)
    df["weighted_rating"] = (v * R + M * C) / (v + M)

    # Serendipity Score = WR / log10(v + 10)
    df["serendipity_score"] = df["weighted_rating"] / (v + 10).apply(math.log10)

    return df


def top_por_genero(df: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """Filtra el DataFrame devolviendo el top-N por género según ``serendipity_score``.

    Args:
        df: DataFrame ya enriquecido con ``serendipity_score`` (salida de
            :func:`calcular_serendipity`).
        n:  Número máximo de registros a conservar por género.

    Returns:
        DataFrame con el top-N de cada género, ordenado por
        ``serendipity_score`` descendente.

    Raises:
        ValueError: Si el DataFrame no contiene la columna ``serendipity_score``.
    """
    if "serendipity_score" not in df.columns:
        raise ValueError(
            "El DataFrame no contiene 'serendipity_score'. "
            "Ejecuta calcular_serendipity() primero."
        )

    resultado = (
        df.sort_values("serendipity_score", ascending=False)
        .groupby("genre", group_keys=False)
        .head(n)
        .reset_index(drop=True)
    )
    logger.info(
        "Top-%d extraído → %d géneros | %d filas totales.",
        n,
        resultado["genre"].nunique(),
        len(resultado),
    )
    return resultado


# ---------------------------------------------------------------------------
# Helper interno: carga y normalización del CSV real
# ---------------------------------------------------------------------------


def _cargar_csv_real(csv_path: Path) -> pd.DataFrame:
    """Carga ``dataset_final_movies.csv`` y lo normaliza al esquema interno.

    El CSV almacena ``genre_ids`` como strings de lista (e.g. ``"[27, 28, 878]"``).
    Esta función los parsea, explota en filas individuales y mapea los IDs al
    nombre de género usando :data:`TMDB_GENRES`.

    Args:
        csv_path: Ruta absoluta al archivo CSV.

    Returns:
        DataFrame con columnas ``[movie_id, genre, rating_mean, vote_count]``.
    """
    logger.info("Leyendo CSV real: %s", csv_path)
    raw = pd.read_csv(csv_path)

    # Parsear genre_ids de string "[27, 28, 35]" → list[int]
    raw["genre_ids"] = raw["genre_ids"].apply(ast.literal_eval)

    # Explotar: una fila por (película, género)
    exploded = raw.explode("genre_ids").copy()
    exploded["genre_ids"] = exploded["genre_ids"].astype(int)

    # Mapear ID numérico → nombre TMDB
    exploded["genre"] = exploded["genre_ids"].map(TMDB_GENRES)
    exploded = exploded.dropna(subset=["genre"])  # ignorar IDs no catalogados

    df = (
        exploded
        .rename(columns={"tmdb_id": "movie_id", "vote_average": "rating_mean"})
        [["movie_id", "genre", "rating_mean", "vote_count"]]
        .dropna()
        .copy()
    )
    logger.info("CSV cargado y normalizado: %d filas (película × género).", len(df))
    return df


# ---------------------------------------------------------------------------
# MODO DESARROLLO — verificación rápida de la matemática
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df_raw = _cargar_csv_real(CSV_PATH)
    df_scored = calcular_serendipity(df_raw)
    df_top = top_por_genero(df_scored, n=TOP_N)

    GENEROS_MUESTRA = ["Drama", "Horror"]

    for genero in GENEROS_MUESTRA:
        subset = df_top[df_top["genre"] == genero].head(3)
        if subset.empty:
            logger.warning("No se encontraron películas para el género '%s'.", genero)
            continue

        print(f"\n{'=' * 60}")
        print(f"  TOP 3 — Género: {genero}")
        print(f"{'=' * 60}")
        print(
            subset[
                ["movie_id", "genre", "rating_mean", "vote_count", "weighted_rating", "serendipity_score"]
            ].to_string(index=False)
        )

    print("\n[OK] Matemática verificada correctamente.")
