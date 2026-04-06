from typing import Any
import pandas as pd
from src.metrics.protocols import MetricProtocol

class CoverageAtK(MetricProtocol):
    """Mide qué porcentaje del catálogo total es capaz de recomendar el modelo (Diversidad)."""

    def __init__(self, catalog_size: int):
        self.catalog_size = catalog_size

    def compute(
        self,
        recommendations: dict[Any, list[Any]],
        ground_truth: pd.DataFrame,
        k: int,
    ) -> float:
        if self.catalog_size <= 0:
            return 0.0
            
        unicas_recomendadas = set()
        for _, recs in recommendations.items():
            unicas_recomendadas.update(recs[:k])
            
        return len(unicas_recomendadas) / self.catalog_size
