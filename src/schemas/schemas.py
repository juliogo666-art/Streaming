"""
Esquemas Pydantic para validación de datos en la API.
Definen la estructura de las peticiones HTTP que recibe el Backend.
"""

from pydantic import BaseModel


class LoginRequest(BaseModel):
    """Esquema para la petición de inicio de sesión."""
    username: str
    password: str


class RegisterRequest(BaseModel):
    """Esquema para la petición de registro de nuevo usuario."""
    username: str
    email: str
    password: str
    fecha_nacimiento: str = None
    sexo: str = None
    intereses: list[int] = []


class RatingRequest(BaseModel):
    """Esquema para la petición de valoración de una película por un usuario."""
    user_id: int
    tmdb_id: int
    rating: float  # 0.5 a 5.0 en pasos de 0.5

