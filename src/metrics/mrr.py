"""
=============================================================================
ARCHIVO: mrr.py (Mean Reciprocal Rank @ K)
=============================================================================
¿QUÉ MIDE?:
MRR responde a la pregunta: "¿En qué posición aparece el PRIMER acierto?"
Si la IA acierta una película que al usuario le gustó en la posición 1,
el MRR para ese usuario es 1.0 (perfecto). Si la acierta en la posición 3,
su MRR es 1/3 = 0.333. Si no acierta ninguna en las Top K, su MRR es 0.

Fórmula:
    MRR@K = promedio( 1 / posicion_del_primer_acierto )
             para todos los usuarios evaluados.

¿POR QUÉ ES ÚTIL?:
Complementa a Precision@K y NDCG@K porque penaliza fuertemente los modelos
que aciertan pero "esconden" el acierto en posiciones bajas del ranking.
Un modelo que siempre acierta en posición 1 es mejor que uno que acierta
en posición 10, aunque ambos tengan la misma Precision@10.
=============================================================================
"""

from typing import Any
import pandas as pd
from src.metrics.protocols import MetricProtocol


class MRRAtK(MetricProtocol):
    """Mean Reciprocal Rank @ K: mide lo rápido que aparece el primer acierto."""

    def __init__(self, user_col: str = "userId", item_col: str = "tmdb_id"):
        self.user_col = user_col
        self.item_col = item_col

    def compute(
        self,
        recommendations: dict[Any, list[Any]],
        ground_truth: pd.DataFrame,
        k: int,
    ) -> float:
        """
        Calcula el MRR@K promediando el recíproco de la posición del primer
        acierto entre las Top K recomendaciones de cada usuario.

        Returns
        -------
        float
            MRR promedio entre 0.0 (ningún acierto) y 1.0 (acierto siempre en posición 1).
        """
        reciprocal_ranks: list[float] = []

        # Agrupamos las películas reales que le gustaron a cada usuario
        items_reales_por_usuario = (
            ground_truth.groupby(self.user_col)[self.item_col].apply(set).to_dict()
        )

        for id_usuario, lista_recomendadas in recommendations.items():
            if id_usuario not in items_reales_por_usuario:
                continue

            items_que_le_gustaron = items_reales_por_usuario[id_usuario]
            # Buscamos el primer acierto dentro del Top K
            recíproco = 0.0
            for posicion, item_recomendado in enumerate(lista_recomendadas[:k], start=1):
                if item_recomendado in items_que_le_gustaron:
                    recíproco = 1.0 / posicion
                    break  # Solo nos interesa el PRIMER acierto

            reciprocal_ranks.append(recíproco)

        if not reciprocal_ranks:
            return 0.0

        return sum(reciprocal_ranks) / len(reciprocal_ranks)
