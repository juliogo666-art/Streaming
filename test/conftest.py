"""
conftest.py — Fixtures compartidas para todos los tests del proyecto SPIRE.
"""

import os
import sys
import pytest
import pandas as pd

# Asegurar que el root del proyecto esté en sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


# ── Fixtures de datos sintéticos ──────────────────────────────────────────────


@pytest.fixture
def df_ratings_sintetico():
    """DataFrame con ratings ficticios para tests unitarios."""
    return pd.DataFrame({
        "userId": [1, 1, 1, 2, 2, 3],
        "tmdb_id": [100, 200, 300, 100, 400, 200],
        "rating": [4.5, 3.0, 5.0, 2.0, 4.0, 1.0],
    })


@pytest.fixture
def df_catalogo_sintetico():
    """DataFrame de catálogo ficticio con películas de ejemplo."""
    return pd.DataFrame({
        "tmdb_id": [100, 200, 300, 400, 500],
        "titulo": ["Película A", "Película B", "Película C", "Película D", "Película E"],
        "genre_ids": ["[28, 12]", "[35]", "[18, 80]", "[28]", "[35, 18]"],
        "vote_average": [7.5, 6.0, 8.2, 5.5, 7.0],
        "vote_count": [1500, 200, 3000, 50, 800],
        "overview": ["Sinopsis A", "Sinopsis B", "Sinopsis C", "Sinopsis D", "Sinopsis E"],
        "poster_path": ["/a.jpg", "/b.jpg", "/c.jpg", "/d.jpg", "/e.jpg"],
        "fecha_estreno": ["2020-01-15", "2019-06-20", "2021-11-03", "2018-03-10", "2022-07-01"],
    })


@pytest.fixture
def recomendaciones_sinteticas():
    """Dict de recomendaciones {user_id: [tmdb_id, ...]} para tests de métricas."""
    return {
        1: [100, 200, 500, 300, 400],  # User 1: recomienda 5 películas
        2: [300, 100, 500, 200, 400],  # User 2: recomienda 5 películas
        3: [200, 400, 100, 500, 300],  # User 3: recomienda 5 películas
    }


@pytest.fixture
def ground_truth_sintetico(df_ratings_sintetico):
    """Ground truth basado en ratings > 3.0 (indica que le gustó)."""
    return df_ratings_sintetico[df_ratings_sintetico["rating"] >= 3.0].copy()
