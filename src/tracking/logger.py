"""
=============================================================================
ARCHIVO: logger.py (Telemetría / Tracking)
=============================================================================
¿PARA QUÉ ES?: 
Guardar un registro histórico (log) de qué recomendaciones se han hecho 
a qué usuarios y con qué nivel de confianza. Esto es vital para MLOps.

¿QUÉ HACE?:
Cada vez que la API responde con películas, este script captura esos datos 
y los escribe de forma estructurada en un archivo JSON Lines (logs/recommendations.jsonl).
A futuro, estos datos sirven para analizar si el modelo es efectivo, si 
el usuario hace click en lo que recomendamos, o para reentrenar modelos.

¿DÓNDE SE CONECTA?:
Se importa y se instancia en 'src/api/main_api.py'. En cada endpoint de 
los modelos (SVD, KNN, etc.), justo antes de devolver el JSON final al 
frontend, llamamos a `logger_telemetria.log_recommendations(...)` y pasamos
la lista top_n para que la guarde discretamente en disco.
=============================================================================
"""

import json
import logging
from datetime import datetime
from pathlib import Path

# Usamos el logger nativo de Python para la consola
logger = logging.getLogger("streaming_tracking")
logger.setLevel(logging.INFO)

class RecommendationLogger:
    """
    Class to extract the feedback from clients on different recommendations.

    Parameters
    ----------
    path : str
        The path where this information will be hosted.
    """

    def __init__(self, log_path: str = "logs/recommendations.jsonl") -> None:
        """Initializes the recommendation Logger."""
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_recommendations(
        self,
        user_id: str,
        modelo: str,
        recomendaciones_top_n: list,
    ) -> None:
        """
        Guarda la información de la recomendación en un archivo JSONL.

        Parameters
        ----------
        user_id : str
            El ID del usuario que pidió la recomendación.
        modelo : str
            Qué modelo sirvió estas peliculas (e.g. 'SVD', 'KNN+Cosine').
        recomendaciones_top_n: list
            Lista de películas recomendadas devueltas por el modelo.
        """
        record = {
            "event": "RECOMMENDATION_SERVED",
            "modelo": modelo,
            "user_id": user_id,
            "items_recommended": recomendaciones_top_n,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self._write(record)

    def log_interaction(
        self, user_id: str, item_id: str, interaction: str
    ) -> None:
        """
        Guarda un log cuando el usuario interactua (ej: click, visualización).

        Parameters
        ----------
        user_id : str
            El ID del usuario.
        item_id : str
            La ID de la peli interactuada.
        interaction : str
            Tipo de interacción ("CLICK", "PLAY", "LIKE").
        """
        record = {
            "event": "USER_INTERACTION",
            "user_id": user_id,
            "item_id": str(item_id),
            "interaction": interaction,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self._write(record)

    def _write(self, record: dict) -> None:
        """
        Writing the logs into the desired function to ensure there is a
        tracking and tracibility to compute the desired values.

        Parameters
        ----------
        record : dict[str, any]
            The records to be writing in the function.
        """
        try:
            serialized = self._serialize(record)

            with open(self.log_path, "a") as f:
                f.write(json.dumps(serialized) + "\n")

        except Exception as e:
            logger.exception(f"Error writing log: {e}")

    def _serialize(self, obj):
        """Recursively serialize objects for JSON dumping."""
        if hasattr(obj, "model_dump"):
            return self._serialize(obj.model_dump())

        if isinstance(obj, dict):
            return {k: self._serialize(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._serialize(v) for v in obj]

        if hasattr(obj, "value"):
            return obj.value

        return obj
