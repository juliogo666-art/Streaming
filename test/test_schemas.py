"""
Tests unitarios de los schemas Pydantic.
"""

import pytest
from pydantic import ValidationError
from src.schemas.schemas import LoginRequest, RegisterRequest, RatingRequest
from src.schemas.recommendation import RecommendationResponse


class TestLoginRequest:
    def test_login_valido(self):
        datos = LoginRequest(username="user1", password="pass123")
        assert datos.username == "user1"
        assert datos.password == "pass123"

    def test_login_sin_username(self):
        with pytest.raises(ValidationError):
            LoginRequest(password="pass123")

    def test_login_sin_password(self):
        with pytest.raises(ValidationError):
            LoginRequest(username="user1")


class TestRegisterRequest:
    def test_registro_completo(self):
        datos = RegisterRequest(
            username="nuevo_user",
            email="test@example.com",
            password="segura123",
            fecha_nacimiento="1990-01-15",
            sexo="M",
        )
        assert datos.username == "nuevo_user"
        assert datos.email == "test@example.com"

    def test_registro_sin_email(self):
        with pytest.raises(ValidationError):
            RegisterRequest(
                username="user",
                password="pass",
                fecha_nacimiento="2000-01-01",
                sexo="F",
            )


class TestRatingRequest:
    def test_rating_valido(self):
        datos = RatingRequest(user_id=1, tmdb_id=550, rating=4.5)
        assert datos.user_id == 1
        assert datos.tmdb_id == 550
        assert datos.rating == 4.5

    def test_rating_sin_user_id(self):
        with pytest.raises(ValidationError):
            RatingRequest(tmdb_id=550, rating=4.5)

    def test_rating_sin_tmdb_id(self):
        with pytest.raises(ValidationError):
            RatingRequest(user_id=1, rating=4.5)


class TestRecommendationResponse:
    def test_respuesta_valida(self):
        resp = RecommendationResponse(
            recomendaciones=[{"tmdb_id": 100, "predicted_rating": 4.2}],
            modelo="SVD (Surprise)",
        )
        assert resp.modelo == "SVD (Surprise)"
        assert len(resp.recomendaciones) == 1

    def test_respuesta_vacia(self):
        resp = RecommendationResponse(
            recomendaciones=[],
            modelo="SVD (Surprise)",
        )
        assert resp.recomendaciones == []

    def test_respuesta_con_mensaje(self):
        resp = RecommendationResponse(
            recomendaciones=[],
            modelo="Content-Based",
            mensaje="Sin datos suficientes.",
        )
        assert resp.mensaje == "Sin datos suficientes."

    def test_respuesta_sin_modelo_falla(self):
        with pytest.raises(ValidationError):
            RecommendationResponse(recomendaciones=[])
