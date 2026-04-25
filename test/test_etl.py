"""
Tests unitarios de las funciones de transformación del ETL.
"""

import pytest
import pandas as pd
import numpy as np

from src.api.etl import normalizar_datos


class TestNormalizarDatos:
    def test_normaliza_movie_devuelve_dict(self):
        """Normalizar una fila de película devuelve un dict con los campos esperados."""
        fila = {
            "tmdb_id": "550",
            "titulo": "Fight Club",
            "original_title": "Fight Club",
            "overview": "Un insomne y un vendedor de jabón...",
            "poster_path": "/pB8BM7pdSp6B6Ih7QI4S2t0POoD.jpg",
            "backdrop_path": "/hZkgoQYus5dXo3H8T7Uef6DNknx.jpg",
            "fecha_estreno": "1999-10-15",
            "vote_average": "8.43",
            "vote_count": "26000",
            "genre_ids": "[18, 53]",
            "original_language": "en",
            "popularity": "73.5",
            "adult": "False",
        }
        resultado = normalizar_datos(fila, "movie")
        assert isinstance(resultado, dict)
        assert resultado["tmdb_id"] == 550
        assert resultado["title"] == "Fight Club"
        assert resultado["content_type"] == "movie"

    def test_normaliza_serie(self):
        """Normalizar una fila de serie con campos distintos."""
        fila = {
            "tmdb_id": "1399",
            "titulo": "Game of Thrones",
            "original_name": "Game of Thrones",
            "overview": "Siete reinos nobles...",
            "poster_path": "/u3bZgnGQ9T01sWNhyveQz0wH0Hl.jpg",
            "backdrop_path": "/suopoADq0k8YZr4dQXcU6pToj6s.jpg",
            "fecha_estreno": "2011-04-17",
            "vote_average": "8.44",
            "vote_count": "20000",
            "genre_ids": "[10765, 18, 10759]",
            "original_language": "en",
            "popularity": "369.5",
        }
        resultado = normalizar_datos(fila, "tv")
        assert isinstance(resultado, dict)
        assert resultado["tmdb_id"] == 1399
        assert resultado["content_type"] == "tv"

    def test_generos_parseados_correctamente(self):
        """genre_ids se parsea de string a lista de enteros."""
        fila = {
            "tmdb_id": "100",
            "titulo": "Test",
            "overview": "",
            "fecha_estreno": "2020-01-01",
            "vote_average": "7.0",
            "vote_count": "500",
            "genre_ids": "[28, 12, 35]",
            "original_language": "en",
            "popularity": "10.0",
        }
        resultado = normalizar_datos(fila, "movie")
        assert resultado["generos"] == [28, 12, 35]

    def test_generos_invalidos_devuelve_lista_vacia(self):
        """Si genre_ids es basura, devuelve lista vacía sin error."""
        fila = {
            "tmdb_id": "100",
            "titulo": "Test",
            "overview": "",
            "fecha_estreno": "2020-01-01",
            "vote_average": "7.0",
            "vote_count": "500",
            "genre_ids": "BASURA_NO_PARSEABLE",
            "original_language": "en",
            "popularity": "10.0",
        }
        resultado = normalizar_datos(fila, "movie")
        assert resultado["generos"] == []

    def test_fecha_desconocida_es_none(self):
        """Fecha 'Desconocida' se convierte en None."""
        fila = {
            "tmdb_id": "100",
            "titulo": "Test",
            "overview": "",
            "fecha_estreno": "Desconocida",
            "vote_average": "7.0",
            "vote_count": "500",
            "genre_ids": "[]",
            "original_language": "en",
            "popularity": "10.0",
        }
        resultado = normalizar_datos(fila, "movie")
        assert resultado["release_date"] is None

    def test_fecha_vacia_es_none(self):
        """Fecha vacía se convierte en None."""
        fila = {
            "tmdb_id": "100",
            "titulo": "Test",
            "overview": "",
            "fecha_estreno": "",
            "vote_average": "7.0",
            "vote_count": "500",
            "genre_ids": "[]",
            "original_language": "en",
            "popularity": "10.0",
        }
        resultado = normalizar_datos(fila, "movie")
        assert resultado["release_date"] is None

    def test_adult_true_a_entero(self):
        """adult='True' se convierte a 1."""
        fila = {
            "tmdb_id": "100",
            "titulo": "Test",
            "overview": "",
            "fecha_estreno": "2020-01-01",
            "vote_average": "7.0",
            "vote_count": "500",
            "genre_ids": "[]",
            "original_language": "en",
            "popularity": "10.0",
            "adult": "True",
        }
        resultado = normalizar_datos(fila, "movie")
        assert resultado["adult"] == 1

    def test_video_por_defecto_false(self):
        """Si no hay campo 'video', vale 0."""
        fila = {
            "tmdb_id": "100",
            "titulo": "Test",
            "overview": "",
            "fecha_estreno": "2020-01-01",
            "vote_average": "7.0",
            "vote_count": "500",
            "genre_ids": "[]",
            "original_language": "en",
            "popularity": "10.0",
        }
        resultado = normalizar_datos(fila, "movie")
        assert resultado["video"] == 0
