"""
SVD (espacio latente de usuario) + kNN usuario-usuario sobre ratings de vecinos,
sin rerank por géneros ni vote_average.

Comparte caché y preprocesado con `model_SVD_KNN_RERANK_con_generos`; solo cambia
el orden final del ranking para alinearlo mejor con evaluaciones tipo SVD/BPR.
"""

from typing import Dict, List, Optional

import numpy as np

from src.models.tx.model_SVD_KNN_RERANK_con_generos import (
    KNNRecommenderTopN,
    build_user_item_matrix,
    get_processed_data,
    get_user_factors,
    get_user_knn,
    load_catalog_features,
)


class KNNRecommenderSinRerank(KNNRecommenderTopN):
    def recommend(
        self,
        raw_user_id,
        top_n: int = 10,
        n_neighbors: int = 50,
        n_candidates: int = 200,
        rerank_alpha: float = 0.7,
        popularity_weight: float = 0.3,
        genre_weight: float = 0.7,
        seen_tmdb_ids_override: Optional[set] = None,
    ) -> List[Dict[str, object]]:
        """
        Top-N ordenado únicamente por el score kNN agregado (vecinos × ratings).
        Los parámetros rerank_alpha, popularity_weight y genre_weight se ignoran
        (se mantienen por compatibilidad de firma con el recommender con rerank).
        """
        try:
            user_idx = int(self.user_encoder.transform([raw_user_id])[0])
        except Exception:
            return self._cold_start(raw_user_id, top_n=top_n)

        user_row = self.user_item_matrix.getrow(user_idx)
        if seen_tmdb_ids_override is None:
            seen_item_indices = set(user_row.indices.tolist())
        else:
            seen_item_indices = set()
            for tmdb_id in seen_tmdb_ids_override:
                try:
                    seen_item_indices.add(
                        int(self.item_encoder.transform([int(tmdb_id)])[0])
                    )
                except Exception:
                    continue
        if user_row.nnz == 0:
            return self._cold_start(raw_user_id, top_n=top_n)

        max_neighbors_fit = getattr(self.knn_model, "n_neighbors", None)
        distances, neighbor_indices = self.knn_model.kneighbors(
            self.user_factors[user_idx].reshape(1, -1),
            n_neighbors=min(
                n_neighbors + 1,
                self.user_factors.shape[0],
                max_neighbors_fit,
            )
            if max_neighbors_fit is not None
            else min(n_neighbors + 1, self.user_factors.shape[0]),
            return_distance=True,
        )
        distances = distances[0]
        neighbor_indices = neighbor_indices[0]
        similarities = 1.0 - distances

        candidates_neighbors = [
            (int(v), float(sim))
            for v, sim in zip(neighbor_indices, similarities)
            if int(v) != user_idx and sim > -1e9
        ]
        if not candidates_neighbors:
            return self._cold_start(raw_user_id, top_n=top_n)

        cand_score_sum: Dict[int, float] = {}
        cand_sim_sum: Dict[int, float] = {}
        for neigh_user_idx, sim in candidates_neighbors:
            neigh_row = self.user_item_matrix.getrow(neigh_user_idx)
            if neigh_row.nnz == 0:
                continue
            for item_i, rating_v in zip(neigh_row.indices, neigh_row.data):
                item_i = int(item_i)
                if item_i in seen_item_indices:
                    continue
                cand_score_sum[item_i] = cand_score_sum.get(item_i, 0.0) + sim * float(
                    rating_v
                )
                cand_sim_sum[item_i] = cand_sim_sum.get(item_i, 0.0) + abs(sim)

        if not cand_score_sum:
            return self._cold_start(raw_user_id, top_n=top_n)

        cand_items = np.array(list(cand_score_sum.keys()), dtype=np.int32)
        cand_knn_scores = np.array(
            [cand_score_sum[i] / (cand_sim_sum[i] + 1e-12) for i in cand_items],
            dtype=np.float32,
        )
        top_idx = np.argsort(cand_knn_scores)[::-1][
            : min(n_candidates, len(cand_items))
        ]
        cand_items = cand_items[top_idx]
        cand_knn_scores = cand_knn_scores[top_idx]

        order = np.argsort(cand_knn_scores)[::-1][: min(top_n, len(cand_items))]
        final_items = cand_items[order]
        final_scores = cand_knn_scores[order]

        tmdb_ids = self.item_encoder.inverse_transform(final_items)

        results: List[Dict[str, object]] = []
        for tmdb_id, item_i, score in zip(tmdb_ids, final_items, final_scores):
            item_i_int = int(item_i)
            title = (
                self.catalog_features.item_title[item_i_int]
                if item_i_int < len(self.catalog_features.item_title)
                else ""
            )
            results.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "titulo": title,
                    "score": float(score),
                }
            )
        return results


def build_recommender(
    force_reprocess: bool = False,
    force_svd_train: bool = False,
    force_knn_train: bool = False,
    force_catalog_features_rebuild: bool = False,
    min_ratings: int = 5,
    latent_dim: int = 50,
    knn_neighbors_fit: int = 80,
) -> KNNRecommenderSinRerank:
    df_prep, user_encoder, item_encoder = get_processed_data(
        force_reprocess=force_reprocess, min_ratings=min_ratings
    )
    user_item_matrix = build_user_item_matrix(df_prep)

    _, user_factors = get_user_factors(
        user_item_matrix, latent_dim=latent_dim, force_train=force_svd_train
    )
    knn_model = get_user_knn(
        user_factors, n_neighbors=knn_neighbors_fit, force_train=force_knn_train
    )
    catalog_features = load_catalog_features(
        item_encoder, force_rebuild=force_catalog_features_rebuild
    )

    return KNNRecommenderSinRerank(
        df_prep=df_prep,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        user_item_matrix=user_item_matrix,
        user_factors=user_factors,
        knn_model=knn_model,
        catalog_features=catalog_features,
    )
