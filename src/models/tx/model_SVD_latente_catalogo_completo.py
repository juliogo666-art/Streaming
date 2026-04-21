"""
Ranking por factorización de matriz explícita (TruncatedSVD): score(u,i) = u_f · v_i
sobre **todos** los ítems del encoder (sin kNN de vecinos, sin n_candidates ni rerank).

Comparte caché de datos y de SVD con el resto de modelos tx/ (mismos parquet/npy).
"""

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

from src.models.tx.model_SVD_KNN_RERANK_con_generos import (
    CatalogFeatures,
    PATH_ITEM_FACTORS,
    build_user_item_matrix,
    get_processed_data,
    get_user_factors,
    load_catalog_features,
)


class SVDFullCatalogLatentRecommender:
    """
    Top-N por producto escalar usuario × ítem en el espacio latente del SVD,
    excluyendo ítems ya vistos (o el conjunto pasado en seen_tmdb_ids_override).
    """

    def __init__(
        self,
        df_prep: pd.DataFrame,
        user_encoder: LabelEncoder,
        item_encoder: LabelEncoder,
        user_item_matrix: csr_matrix,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        catalog_features: CatalogFeatures,
    ):
        self.df_prep = df_prep
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.user_item_matrix = user_item_matrix
        self.user_factors = np.asarray(user_factors, dtype=np.float32)
        self.item_factors = np.asarray(item_factors, dtype=np.float32)
        self.catalog_features = catalog_features
        self.item_rating_counts = np.bincount(
            df_prep["item_idx"].values, minlength=user_item_matrix.shape[1]
        ).astype(np.float32)

    def recommend(
        self,
        raw_user_id,
        top_n: int = 10,
        n_neighbors: int = 50,
        n_candidates: int = 200,
        rerank_alpha: float = 0.7,
        popularity_weight: float = 0.3,
        genre_weight: float = 0.7,
        seen_tmdb_ids_override: Optional[Set] = None,
    ) -> List[Dict[str, object]]:
        """Parámetros kNN/rerank se ignoran; se mantienen por compatibilidad de API."""
        del n_neighbors, n_candidates, rerank_alpha, popularity_weight, genre_weight

        try:
            user_idx = int(self.user_encoder.transform([raw_user_id])[0])
        except Exception:
            return self._cold_start(top_n=top_n)

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
            return self._cold_start(top_n=top_n)

        u = self.user_factors[user_idx]
        # (n_items,) = (k,) @ (k, n_items)
        scores = u @ self.item_factors.T

        mask = np.ones(scores.shape[0], dtype=bool)
        for idx in seen_item_indices:
            if 0 <= idx < mask.size:
                mask[idx] = False
        scores_eff = scores.copy()
        scores_eff[~mask] = -np.inf

        order_all = np.argsort(scores_eff)[::-1]
        finite_ok = np.isfinite(scores_eff[order_all])
        order = order_all[finite_ok][:top_n]
        if order.size == 0:
            return self._cold_start(top_n=top_n)

        tmdb_ids = self.item_encoder.inverse_transform(order)
        results: List[Dict[str, object]] = []
        for tmdb_id, item_i in zip(tmdb_ids, order):
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
                    "score": float(scores[item_i_int]),
                }
            )
        return results

    def _cold_start(self, top_n: int) -> List[Dict[str, object]]:
        order = np.argsort(self.item_rating_counts)[::-1][:top_n]
        tmdb_ids = self.item_encoder.inverse_transform(order)
        results: List[Dict[str, object]] = []
        for tmdb_id, item_i in zip(tmdb_ids, order):
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
                    "score": float(self.item_rating_counts[item_i_int]),
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
) -> SVDFullCatalogLatentRecommender:
    _ = force_knn_train, knn_neighbors_fit

    df_prep, user_encoder, item_encoder = get_processed_data(
        force_reprocess=force_reprocess, min_ratings=min_ratings
    )
    user_item_matrix = build_user_item_matrix(df_prep)
    n_items = user_item_matrix.shape[1]

    _, user_factors = get_user_factors(
        user_item_matrix, latent_dim=latent_dim, force_train=force_svd_train
    )
    item_factors = np.load(PATH_ITEM_FACTORS)

    if (
        item_factors.shape[0] != n_items
        or item_factors.shape[1] != user_factors.shape[1]
    ):
        _, user_factors = get_user_factors(
            user_item_matrix, latent_dim=latent_dim, force_train=True
        )
        item_factors = np.load(PATH_ITEM_FACTORS)

    catalog_features = load_catalog_features(
        item_encoder, force_rebuild=force_catalog_features_rebuild
    )

    return SVDFullCatalogLatentRecommender(
        df_prep=df_prep,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        user_item_matrix=user_item_matrix,
        user_factors=user_factors,
        item_factors=item_factors,
        catalog_features=catalog_features,
    )
