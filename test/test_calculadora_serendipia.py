"""
Tests unitarios de la calculadora de serendipia.
Usa datos sintéticos sin necesidad de BD.
"""

import pytest
import pandas as pd

from src.serendipia.calculadora_serendipia import (
    calcular_serendipity,
    top_por_genero,
    M,  # Umbral mínimo de votos (constante)
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def df_peliculas():
    """DataFrame simulando películas con columnas requeridas por calcular_serendipity."""
    return pd.DataFrame(
        {
            "movie_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "vote_count": [5000, 50, 10, 3000, 8, 200, 1, 100],
            "rating_mean": [8.5, 9.0, 7.0, 6.0, 10.0, 7.5, 5.0, 8.0],
            "genre": [
                "Action",
                "Action",
                "Action",
                "Drama",
                "Drama",
                "Drama",
                "Comedy",
                "Comedy",
            ],
        }
    )


# ── Tests Serendipity Score ───────────────────────────────────────────────────


class TestCalcularSerendipity:
    def test_devuelve_dataframe(self, df_peliculas):
        resultado = calcular_serendipity(df_peliculas)
        assert isinstance(resultado, pd.DataFrame)

    def test_columna_serendipity_presente(self, df_peliculas):
        resultado = calcular_serendipity(df_peliculas)
        assert "serendipity_score" in resultado.columns
        assert "weighted_rating" in resultado.columns

    def test_scores_positivos(self, df_peliculas):
        resultado = calcular_serendipity(df_peliculas)
        assert (resultado["serendipity_score"] >= 0).all()

    def test_favorece_joyas_ocultas(self, df_peliculas):
        """Las películas poco votadas pero bien valoradas deben tener mayor score."""
        resultado = calcular_serendipity(df_peliculas)
        # movie_id=2: pocos votos (50), rating alta (9.0)
        # movie_id=1: muchos votos (5000), rating alta (8.5)
        score_joya = resultado[resultado["movie_id"] == 2]["serendipity_score"].values[
            0
        ]
        score_popular = resultado[resultado["movie_id"] == 1][
            "serendipity_score"
        ].values[0]
        assert score_joya > score_popular

    def test_no_modifica_original(self, df_peliculas):
        """Verificar que no muta el DataFrame original."""
        original_cols = set(df_peliculas.columns)
        calcular_serendipity(df_peliculas)
        assert set(df_peliculas.columns) == original_cols

    def test_error_sin_columnas_requeridas(self):
        """Si faltan columnas requeridas, debe lanzar ValueError."""
        df_malo = pd.DataFrame({"movie_id": [1], "titulo": ["Test"]})
        with pytest.raises(ValueError, match="Columnas faltantes"):
            calcular_serendipity(df_malo)

    def test_weighted_rating_formula(self, df_peliculas):
        """Verifica la fórmula WR = (v*R + m*C) / (v+m)."""
        resultado = calcular_serendipity(df_peliculas)
        C = df_peliculas["rating_mean"].mean()
        # Para movie_id=1: v=5000, R=8.5
        fila = resultado[resultado["movie_id"] == 1].iloc[0]
        v, R = 5000, 8.5
        wr_esperado = (v * R + M * C) / (v + M)
        assert abs(fila["weighted_rating"] - wr_esperado) < 0.01


# ── Tests Top por Género ─────────────────────────────────────────────────────


class TestTopPorGenero:
    def test_respeta_limite(self, df_peliculas):
        resultado = calcular_serendipity(df_peliculas)
        top = top_por_genero(resultado, n=2)
        for _, grupo in top.groupby("genre"):
            assert len(grupo) <= 2

    def test_devuelve_todos_los_generos(self, df_peliculas):
        resultado = calcular_serendipity(df_peliculas)
        top = top_por_genero(resultado, n=10)
        generos_entrada = set(df_peliculas["genre"].unique())
        generos_salida = set(top["genre"].unique())
        assert generos_entrada == generos_salida

    def test_ordenado_por_score(self, df_peliculas):
        resultado = calcular_serendipity(df_peliculas)
        top = top_por_genero(resultado, n=5)
        for _, grupo in top.groupby("genre"):
            scores = grupo["serendipity_score"].tolist()
            assert scores == sorted(scores, reverse=True)

    def test_error_sin_serendipity_score(self, df_peliculas):
        """top_por_genero requiere que se haya calculado el score primero."""
        with pytest.raises(ValueError, match="serendipity_score"):
            top_por_genero(df_peliculas, n=5)
