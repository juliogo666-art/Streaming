from typing import Any
import pandas as pd
from src.metrics.protocols import MetricProtocol

class HitRateAtK(MetricProtocol):
    """Métrica binaria: ¿Hubo al menos 1 acierto en el Top K? (1 Sí / 0 No)."""

    def __init__(self, user_col="userId", item_col="tmdb_id"):
        self.user_col = user_col
        self.item_col = item_col

    def compute(
        self,
        recommendations: dict[Any, list[Any]],
        ground_truth: pd.DataFrame,
        k: int,
    ) -> float:
        hits_totales = 0
        usuarios_evaluados = 0
        gt = ground_truth.groupby(self.user_col)[self.item_col].apply(set).to_dict()

        for user_id, recs in recommendations.items():
            if user_id not in gt:
                continue
            usuarios_evaluados += 1
            if len(set(recs[:k]) & gt[user_id]) > 0:
                hits_totales += 1

        return hits_totales / usuarios_evaluados if usuarios_evaluados > 0 else 0.0
