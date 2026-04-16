"""
=============================================================================
ARCHIVO: recommendation.py (Schemas de Pydantic)
=============================================================================
¿PARA QUÉ ES?: 
Este archivo define la estructura exacta (tipado fuerte) que deben tener 
los datos que la API envía al Frontend cuando el usuario pide películas.

¿QUÉ HACE?:
Usa Pydantic para validar que cada película recomendada siempre tenga un 
'tmdb_id' entero, un 'predicted_rating' flotante, y opcionalmente el título, 
póster, etc. Si el Backend intenta devolver un formato erróneo, FastAPI 
lanzará un error antes de que llegue al cliente, evitando bugs visuales.

¿DÓNDE SE CONECTA?:
Se importa en 'src/api/main_api.py'. Cada endpoint de recomendación 
(@app.get("/recomendar/...")) tiene el parámetro 'response_model=RecommendationResponse'.
Así FastAPI usa este archivo como su "guardia de seguridad" en la puerta de salida.
=============================================================================
"""

from pydantic import BaseModel
from typing import List, Optional

class RecommendationItem(BaseModel):
    """Esquema para una película individual recomendada."""
    tmdb_id: int
    predicted_rating: float
    titulo: Optional[str] = None
    poster_path: Optional[str] = None
    overview: Optional[str] = None
    vote_average: Optional[float] = None

class RecommendationResponse(BaseModel):
    """Esquema de la respuesta completa del endpoint de recomendaciones."""
    recomendaciones: List[RecommendationItem]
    modelo: str
    mensaje: Optional[str] = None
    insufficient_data: Optional[bool] = None
    selector: Optional[str] = None  # Info del selector dinámico Smart
