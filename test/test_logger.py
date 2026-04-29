"""
Tests unitarios del sistema de tracking de recomendaciones.
"""

import json
import os
import pytest

from src.tracking.logger import RecommendationLogger


class TestRecommendationLogger:
    def test_crea_directorio(self, tmp_path):
        """El logger crea el directorio de logs si no existe."""
        log_file = str(tmp_path / "test_logs" / "recs.jsonl")
        logger = RecommendationLogger(log_path=log_file)
        assert os.path.isdir(str(tmp_path / "test_logs"))

    def test_log_crea_archivo(self, tmp_path):
        """Cada log genera un archivo JSONL."""
        log_file = str(tmp_path / "test_logs" / "recs.jsonl")
        logger = RecommendationLogger(log_path=log_file)
        logger.log_recommendations(
            user_id="42",
            modelo="SVD",
            recomendaciones_top_n=[{"tmdb_id": 100, "predicted_rating": 4.5}],
        )
        assert os.path.exists(log_file)

    def test_log_formato_json_valido(self, tmp_path):
        """Cada línea del JSONL es un JSON válido."""
        log_file = str(tmp_path / "test_logs" / "recs.jsonl")
        logger = RecommendationLogger(log_path=log_file)
        logger.log_recommendations(
            user_id="1",
            modelo="KNN",
            recomendaciones_top_n=[
                {"tmdb_id": 100, "predicted_rating": 3.0},
                {"tmdb_id": 200, "predicted_rating": 4.5},
            ],
        )
        with open(log_file, "r", encoding="utf-8") as f:
            for linea in f:
                registro = json.loads(linea.strip())
                assert "user_id" in registro
                assert "modelo" in registro
                assert "items_recommended" in registro

    def test_log_contiene_campos_esperados(self, tmp_path):
        """Verifica que los campos obligatorios estén presentes."""
        log_file = str(tmp_path / "test_logs" / "recs.jsonl")
        logger = RecommendationLogger(log_path=log_file)
        logger.log_recommendations(
            user_id="99",
            modelo="Wide&Deep",
            recomendaciones_top_n=[{"tmdb_id": 550}],
            tiempo_recomendacion_ms=42.5,
        )
        with open(log_file, "r", encoding="utf-8") as f:
            registro = json.loads(f.readline().strip())
        assert registro["user_id"] == "99"
        assert registro["modelo"] == "Wide&Deep"
        assert "timestamp" in registro
        assert registro["tiempo_recomendacion_ms"] == 42.5

    def test_multiples_logs_acumulan(self, tmp_path):
        """Varios logs se acumulan en el mismo archivo."""
        log_file = str(tmp_path / "test_logs" / "recs.jsonl")
        logger = RecommendationLogger(log_path=log_file)
        for i in range(5):
            logger.log_recommendations(
                user_id=str(i),
                modelo="Test",
                recomendaciones_top_n=[],
            )
        with open(log_file, "r", encoding="utf-8") as f:
            lineas = f.readlines()
        assert len(lineas) == 5

    def test_evento_recommendation_served(self, tmp_path):
        """Cada log tiene el evento 'RECOMMENDATION_SERVED'."""
        log_file = str(tmp_path / "test_logs" / "recs.jsonl")
        logger = RecommendationLogger(log_path=log_file)
        logger.log_recommendations(
            user_id="1",
            modelo="SVD",
            recomendaciones_top_n=[],
        )
        with open(log_file, "r", encoding="utf-8") as f:
            registro = json.loads(f.readline().strip())
        assert registro["event"] == "RECOMMENDATION_SERVED"

    def test_log_interaction(self, tmp_path):
        """log_interaction registra un evento USER_INTERACTION."""
        log_file = str(tmp_path / "test_logs" / "recs.jsonl")
        logger = RecommendationLogger(log_path=log_file)
        logger.log_interaction(user_id="5", item_id="550", interaction="CLICK")
        with open(log_file, "r", encoding="utf-8") as f:
            registro = json.loads(f.readline().strip())
        assert registro["event"] == "USER_INTERACTION"
        assert registro["user_id"] == "5"
        assert registro["item_id"] == "550"
        assert registro["interaction"] == "CLICK"

    def test_serializa_pydantic_model(self, tmp_path):
        """La serialización maneja objetos con model_dump (Pydantic)."""
        from pydantic import BaseModel

        class ItemTest(BaseModel):
            tmdb_id: int
            score: float

        log_file = str(tmp_path / "test_logs" / "recs.jsonl")
        logger = RecommendationLogger(log_path=log_file)
        item = ItemTest(tmdb_id=100, score=4.5)
        logger.log_recommendations(
            user_id="1",
            modelo="Test",
            recomendaciones_top_n=[item],
        )
        with open(log_file, "r", encoding="utf-8") as f:
            registro = json.loads(f.readline().strip())
        assert registro["items_recommended"][0]["tmdb_id"] == 100

    def test_serializa_enum(self, tmp_path):
        """La serialización maneja objetos con .value (enums)."""
        from enum import Enum

        class Modelo(Enum):
            SVD = "svd_model"

        log_file = str(tmp_path / "test_logs" / "recs.jsonl")
        logger = RecommendationLogger(log_path=log_file)
        logger.log_recommendations(
            user_id="1",
            modelo="Test",
            recomendaciones_top_n=[{"modelo_tipo": Modelo.SVD}],
        )
        with open(log_file, "r", encoding="utf-8") as f:
            registro = json.loads(f.readline().strip())
        assert registro["items_recommended"][0]["modelo_tipo"] == "svd_model"
