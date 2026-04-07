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
