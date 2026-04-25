from typing import Any
import pandas as pd
import math
from src.metrics.protocols import MetricProtocol

class NDCGAtK(MetricProtocol):
    """
    NDCG (Normalized Discounted Cumulative Gain):
    Penaliza a los modelos si ponen las pelis top del usuario en posiciones bajas.
    """

    def __init__(self, user_col="userId", item_col="tmdb_id", rating_col="rating"):
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col

    def compute(
        self,
        recommendations: dict[Any, list[Any]],
        ground_truth: pd.DataFrame,
        k: int,
    ) -> float:
        ndcgs: list[float] = []

        # Convertimos ground truth en diccionarios anidados: {user_id: {tmdb_id: rating}}
        gt_dict = {}
        for _, row in ground_truth.iterrows():
            u = row[self.user_col]
            i = row[self.item_col]
            r = row[self.rating_col]
            if u not in gt_dict:
                gt_dict[u] = {}
            gt_dict[u][i] = r

        for user_id, recs in recommendations.items():
            if user_id not in gt_dict:
                continue

            relevantes_escala = gt_dict[user_id]
            recs_k = recs[:k]
            
            # DCG
            dcg = 0.0
            for i, item in enumerate(recs_k):
                if item in relevantes_escala:
                    relevancia = relevantes_escala[item]
                    dcg += relevancia / math.log2(i + 2)

            # IDCG
            idcg = 0.0
            ideal_relevancias = sorted(list(relevantes_escala.values()), reverse=True)
            for i, rel in enumerate(ideal_relevancias[: len(recs_k)]):
                idcg += rel / math.log2(i + 2)

            if idcg > 0:
                ndcgs.append(dcg / idcg)
            else:
                ndcgs.append(0.0)

        return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
