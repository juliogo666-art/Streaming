from typing import Any
import pandas as pd
from src.metrics.protocols import MetricProtocol

class RecallAtK(MetricProtocol):
    """Mide cuántas de las pelis que le gustaron fuimos capaces de encontrar."""

    def __init__(self, user_col="userId", item_col="tmdb_id"):
        self.user_col = user_col
        self.item_col = item_col

    def compute(
        self,
        recommendations: dict[Any, list[Any]],
        ground_truth: pd.DataFrame,
        k: int,
    ) -> float:
        recalls: list[float] = []
        gt = ground_truth.groupby(self.user_col)[self.item_col].apply(set).to_dict()

        for user_id, recs in recommendations.items():
            if user_id not in gt:
                continue
            relevantes = gt[user_id]
            if not relevantes:
                continue
            hits = len(set(recs[:k]) & relevantes)
            recalls.append(hits / len(relevantes))

        return sum(recalls) / len(recalls) if recalls else 0.0
