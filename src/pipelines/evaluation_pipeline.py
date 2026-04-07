"""
=============================================================================
ARCHIVO: evaluation_pipeline.py
=============================================================================
Esta clase orquesta todas las métricas definidas en src/metrics/ para
evaluar varios modelos de IA sin duplicar cálculos o código.

Se le inyecta una lista de métricas (MetricProtocol) al inicializarse.
Luego, para cada modelo, se le llama a `evaluate_model(...)`.
=============================================================================
"""

import pandas as pd
from typing import Any, List
from src.metrics.protocols import MetricProtocol


class EvaluationPipeline:
    def __init__(self, metrics: List[MetricProtocol]):
        """
        Inicializa el pipeline con una lista de métricas que cumplen el protocolo.
        """
        self.metrics = metrics
        self.results = {}

    def evaluate_model(
        self,
        model_name: str,
        recommendations: dict[Any, list[Any]],
        ground_truth: pd.DataFrame,
        k: int = 10,
    ) -> dict[str, float]:
        """
        Evalúa un modelo con las métricas cargadas.

        Parámetros
        ----------
        model_name : str
            El nombre del modelo (ej: 'SVD', 'KNN')
        recommendations : dict
            Diccionario {userId: [lista de tmdb_id_recomendadas]}
        ground_truth : pd.DataFrame
            DataFrame con el subconjunto oculto (el examen de los usuarios).
            Debe tener columnas de id de usuario, id de peli y, opcional, rating.
        k : int
            Límite de corte (Top K).

        Devuelve
        -------
        Un diccionario con la puntuación de cada métrica evaludada.
        """
        model_results = {}

        # Ejecutamos la regla del protocolo (OOP puro)
        for metric in self.metrics:
            name = metric.__class__.__name__
            try:
                score = metric.compute(recommendations, ground_truth, k)
                model_results[name] = score
            except Exception as e:
                print(f"[ALERTA] Fallo al computar {name} en {model_name}: {e}")
                model_results[name] = 0.0

        self.results[model_name] = model_results
        return model_results

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Exporta todos los resultados de los modelos evaluados en un DataFrame limpio.
        """
        records = []
        for model_name, metrics_dict in self.results.items():
            record = {"Modelo": model_name}
            record.update(metrics_dict)
            records.append(record)

        return pd.DataFrame(records)
