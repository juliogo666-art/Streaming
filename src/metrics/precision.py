from typing import Any
import pandas as pd
from src.metrics.protocols import MetricProtocol

class PrecisionAtK(MetricProtocol):
    """Mide qué porcentaje de las K sugeridas fueron realmente acertadas."""

    def __init__(self, user_col="userId", item_col="tmdb_id"):
        self.user_col = user_col
        self.item_col = item_col

    def compute(
        self,
        recommendations: dict[Any, list[Any]],
        ground_truth: pd.DataFrame,
        k: int,
    ) -> float:
        precisions: list[float] = []
        gt = ground_truth.groupby(self.user_col)[self.item_col].apply(set).to_dict()

        for user_id, recs in recommendations.items():
            if user_id not in gt:
                continue
            hits = len(set(recs[:k]) & gt[user_id])
            precisions.append(hits / k)

        return sum(precisions) / len(precisions) if precisions else 0.0
