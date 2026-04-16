import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

# Tracking centralizado de métricas (historial_metricas.csv)
from src.utils.registrar_metricas import registrar_metricas


# Rutas de datos
DATA_READY_RATINGS = "src/data/ready/ratings_finales_ia.csv"
DATA_READY_CATALOG_MOVIES = "src/data/ready/dataset_final_movies.csv"

# Caché compartida
CACHE_DIR = "src/data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

PATH_DF_PREP = os.path.join(CACHE_DIR, "df_prep.parquet")
PATH_USER_ENC = os.path.join(CACHE_DIR, "user_encoder.joblib")
PATH_ITEM_ENC = os.path.join(CACHE_DIR, "item_encoder.joblib")

PATH_SVD_MODEL = os.path.join(CACHE_DIR, "model3_svd.joblib")
PATH_USER_FACTORS = os.path.join(CACHE_DIR, "model3_user_factors.npy")
PATH_KNN_MODEL = os.path.join(CACHE_DIR, "model3_knn.joblib")
PATH_ITEM_VOTE = os.path.join(CACHE_DIR, "model3_item_vote.npy")
PATH_ITEM_TITLE = os.path.join(CACHE_DIR, "model3_item_title.joblib")


def get_processed_data(force_reprocess: bool = False, min_ratings: int = 30) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Carga ratings limpios y codifica user/item.
    Usa caché para acelerar.
    """
    if (
        not force_reprocess
        and os.path.exists(PATH_DF_PREP)
        and os.path.exists(PATH_USER_ENC)
        and os.path.exists(PATH_ITEM_ENC)
    ):
        return pd.read_parquet(PATH_DF_PREP), joblib.load(PATH_USER_ENC), joblib.load(PATH_ITEM_ENC)

    df = pd.read_csv(DATA_READY_RATINGS)
    df = df.drop_duplicates(["userId", "tmdb_id"])

    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df["user_idx"] = user_enc.fit_transform(df["userId"])
    df["item_idx"] = item_enc.fit_transform(df["tmdb_id"])

    counts = df["item_idx"].value_counts()
    keep_idx = counts[counts > min_ratings].index
    df = df[df["item_idx"].isin(keep_idx)].copy()

    df["rating"] = df["rating"].astype(np.float32)
    df["user_idx"] = df["user_idx"].astype(np.int32)
    df["item_idx"] = df["item_idx"].astype(np.int32)

    df.to_parquet(PATH_DF_PREP)
    joblib.dump(user_enc, PATH_USER_ENC)
    joblib.dump(item_enc, PATH_ITEM_ENC)
    return df, user_enc, item_enc


def build_user_item_matrix(df: pd.DataFrame) -> csr_matrix:
    return csr_matrix(
        (df["rating"], (df["user_idx"], df["item_idx"])),
        shape=(df["user_idx"].max() + 1, df["item_idx"].max() + 1),
    )


def get_user_factors(user_item_matrix: csr_matrix, latent_dim: int = 50, force_train: bool = False) -> np.ndarray:
    if not force_train and os.path.exists(PATH_SVD_MODEL) and os.path.exists(PATH_USER_FACTORS):
        svd = joblib.load(PATH_SVD_MODEL)
        user_factors = np.load(PATH_USER_FACTORS)
        if getattr(svd, "n_components", None) == latent_dim and user_factors.shape[1] == latent_dim:
            return user_factors

    svd = TruncatedSVD(n_components=latent_dim, random_state=42)
    user_factors = svd.fit_transform(user_item_matrix)
    joblib.dump(svd, PATH_SVD_MODEL)
    np.save(PATH_USER_FACTORS, user_factors)
    return user_factors


def get_knn_model(user_factors: np.ndarray, n_neighbors: int = 80, force_train: bool = False) -> NearestNeighbors:
    if not force_train and os.path.exists(PATH_KNN_MODEL):
        try:
            return joblib.load(PATH_KNN_MODEL)
        except Exception:
            pass

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    knn.fit(user_factors)
    joblib.dump(knn, PATH_KNN_MODEL)
    return knn


def get_item_vote_feature(item_encoder: LabelEncoder, force_rebuild: bool = False) -> np.ndarray:
    """
    Feature sencilla para re-rank: vote_average por item_idx.
    """
    if not force_rebuild and os.path.exists(PATH_ITEM_VOTE):
        votes = np.load(PATH_ITEM_VOTE)
        if votes.shape[0] == len(item_encoder.classes_):
            return votes

    catalog = pd.read_csv(DATA_READY_CATALOG_MOVIES)
    if "tmdb_id" not in catalog.columns:
        raise ValueError(f"Falta 'tmdb_id' en {DATA_READY_CATALOG_MOVIES}")

    catalog["tmdb_id"] = pd.to_numeric(catalog["tmdb_id"], errors="coerce")
    catalog = catalog.dropna(subset=["tmdb_id"])
    catalog["tmdb_id"] = catalog["tmdb_id"].astype(np.int64)
    catalog = catalog.set_index("tmdb_id", drop=False)

    use_vote = "vote_average" if "vote_average" in catalog.columns else None
    votes = np.zeros(len(item_encoder.classes_), dtype=np.float32)

    if use_vote is not None:
        for item_idx, tmdb_id in enumerate(item_encoder.classes_):
            try:
                tmdb_id = int(tmdb_id)
            except Exception:
                continue
            if tmdb_id in catalog.index:
                try:
                    votes[item_idx] = float(catalog.loc[tmdb_id][use_vote])
                except Exception:
                    votes[item_idx] = 0.0

    np.save(PATH_ITEM_VOTE, votes)
    return votes


def get_item_title_feature(item_encoder: LabelEncoder, force_rebuild: bool = False) -> List[str]:
    """
    Mapa sencillo item_idx -> titulo para mostrar resultados.
    """
    if not force_rebuild and os.path.exists(PATH_ITEM_TITLE):
        titles = joblib.load(PATH_ITEM_TITLE)
        if len(titles) == len(item_encoder.classes_):
            return titles

    catalog = pd.read_csv(DATA_READY_CATALOG_MOVIES)
    if "tmdb_id" not in catalog.columns:
        raise ValueError(f"Falta 'tmdb_id' en {DATA_READY_CATALOG_MOVIES}")

    catalog["tmdb_id"] = pd.to_numeric(catalog["tmdb_id"], errors="coerce")
    catalog = catalog.dropna(subset=["tmdb_id"])
    catalog["tmdb_id"] = catalog["tmdb_id"].astype(np.int64)
    catalog = catalog.set_index("tmdb_id", drop=False)

    use_title = "titulo" if "titulo" in catalog.columns else None
    titles = ["" for _ in range(len(item_encoder.classes_))]

    if use_title is not None:
        for item_idx, tmdb_id in enumerate(item_encoder.classes_):
            try:
                tmdb_id = int(tmdb_id)
            except Exception:
                continue
            if tmdb_id in catalog.index:
                try:
                    value = catalog.loc[tmdb_id][use_title]
                    titles[item_idx] = "" if pd.isna(value) else str(value)
                except Exception:
                    titles[item_idx] = ""

    joblib.dump(titles, PATH_ITEM_TITLE)
    return titles


class SimpleRobustRecommender:
    def __init__(
        self,
        df: pd.DataFrame,
        user_encoder: LabelEncoder,
        item_encoder: LabelEncoder,
        user_item_matrix: csr_matrix,
        user_factors: np.ndarray,
        knn_model: NearestNeighbors,
        item_vote_avg: np.ndarray,
        item_titles: List[str],
    ):
        self.df = df
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.user_item_matrix = user_item_matrix
        self.user_factors = user_factors
        self.knn_model = knn_model
        self.item_vote_avg = item_vote_avg
        self.item_titles = item_titles
        self.item_popularity = np.bincount(df["item_idx"].values, minlength=user_item_matrix.shape[1]).astype(np.float32)

    def recommend_top10(
        self,
        raw_user_id: int,
        n_neighbors: int = 50,
        n_candidates: int = 500,
        alpha_cf: float = 0.8,
    ) -> List[Dict[str, object]]:
        """
        Devuelve exactamente top10 con:
        - tmdb_id
        - score
        """
        alpha_cf = float(np.clip(alpha_cf, 0.0, 1.0))

        try:
            user_idx = int(self.user_encoder.transform([raw_user_id])[0])
        except Exception:
            return self._cold_start_top10()

        user_row = self.user_item_matrix.getrow(user_idx)
        seen = set(user_row.indices.tolist())
        if user_row.nnz == 0:
            return self._cold_start_top10()

        max_fit_neighbors = getattr(self.knn_model, "n_neighbors", None)
        n_query = min(n_neighbors + 1, self.user_factors.shape[0], max_fit_neighbors) if max_fit_neighbors is not None else min(n_neighbors + 1, self.user_factors.shape[0])
        distances, neigh_idx = self.knn_model.kneighbors(self.user_factors[user_idx].reshape(1, -1), n_neighbors=n_query)
        sims = 1.0 - distances[0]
        neighbors = [
            (int(nu), float(sim))
            for nu, sim in zip(neigh_idx[0], sims)
            if int(nu) != user_idx
        ]
        if not neighbors:
            return self._cold_start_top10()

        # Score CF robusto: promedio ponderado por similitud absoluta.
        score_sum: Dict[int, float] = {}
        sim_sum: Dict[int, float] = {}
        for neigh_user_idx, sim in neighbors:
            row = self.user_item_matrix.getrow(neigh_user_idx)
            for item_i, rating_v in zip(row.indices, row.data):
                item_i = int(item_i)
                if item_i in seen:
                    continue
                score_sum[item_i] = score_sum.get(item_i, 0.0) + sim * float(rating_v)
                sim_sum[item_i] = sim_sum.get(item_i, 0.0) + abs(sim)

        if not score_sum:
            return self._cold_start_top10()

        cand_items = np.array(list(score_sum.keys()), dtype=np.int32)
        cf_scores = np.array([score_sum[i] / (sim_sum[i] + 1e-12) for i in cand_items], dtype=np.float32)
        top_idx = np.argsort(cf_scores)[::-1][: min(n_candidates, len(cand_items))]
        cand_items = cand_items[top_idx]
        cf_scores = cf_scores[top_idx]

        # Normalización 0..1 para combinar de forma estable.
        cf_norm = (cf_scores - cf_scores.min()) / ((cf_scores.max() - cf_scores.min()) + 1e-12)
        vote_vals = self.item_vote_avg[cand_items]
        vote_norm = (vote_vals - vote_vals.min()) / ((vote_vals.max() - vote_vals.min()) + 1e-12)

        final_scores = alpha_cf * cf_norm + (1.0 - alpha_cf) * vote_norm
        order = np.argsort(final_scores)[::-1][:10]

        final_items = cand_items[order]
        final_scores = final_scores[order]
        tmdb_ids = self.item_encoder.inverse_transform(final_items)

        return [
            {
                "tmdb_id": int(tmdb),
                "titulo": self.item_titles[int(item_i)] if int(item_i) < len(self.item_titles) else "",
                "score": float(score),
            }
            for tmdb, item_i, score in zip(tmdb_ids, final_items, final_scores)
        ]

    def _cold_start_top10(self) -> List[Dict[str, object]]:
        order = np.argsort(self.item_popularity)[::-1][:10]
        tmdb_ids = self.item_encoder.inverse_transform(order)
        # Escala de score simple y consistente para fallback.
        pop = self.item_popularity[order]
        pop_norm = (pop - pop.min()) / ((pop.max() - pop.min()) + 1e-12)
        return [
            {
                "tmdb_id": int(tmdb),
                "titulo": self.item_titles[int(item_i)] if int(item_i) < len(self.item_titles) else "",
                "score": float(score),
            }
            for tmdb, item_i, score in zip(tmdb_ids, order, pop_norm)
        ]


def _dcg_at_k(relevances: np.ndarray, k: int) -> float:
    rel = relevances[:k].astype(np.float32)
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2, dtype=np.float32))
    return float(np.sum(rel * discounts))


def ndcg_at_k(relevances: List[int], k: int) -> float:
    rel = np.asarray(relevances, dtype=np.float32)
    dcg = _dcg_at_k(rel, k)
    ideal = np.sort(rel)[::-1]
    idcg = _dcg_at_k(ideal, k)
    return 0.0 if idcg <= 0 else float(dcg / idcg)


def precision_at_k(relevances: List[int], k: int) -> float:
    rel = np.asarray(relevances, dtype=np.float32)[:k]
    if rel.size == 0:
        return 0.0
    return float(np.sum(rel > 0) / k)


def _predict_scores_for_pairs(
    recommender: SimpleRobustRecommender,
    user_ids: np.ndarray,
    item_ids_tmdb: np.ndarray,
    top_for_lookup: int = 10,
) -> np.ndarray:
    """
    Predice score para pares (userId, tmdb_id) con recommend_top10.
    Si no aparece en top_for_lookup, score=0.
    Luego escala 0..1 a rating aproximado 1..5 para RMSE/MAE.
    """
    preds = np.zeros(len(user_ids), dtype=np.float32)
    cache: Dict[int, Dict[int, float]] = {}

    for i, (u, tmdb) in enumerate(zip(user_ids, item_ids_tmdb)):
        u_int = int(u)
        tmdb_int = int(tmdb)
        if u_int not in cache:
            recs = recommender.recommend_top10(u_int, n_candidates=max(500, top_for_lookup))
            cache[u_int] = {int(r["tmdb_id"]): float(r["score"]) for r in recs}
        score = float(cache[u_int].get(tmdb_int, 0.0))
        preds[i] = float(1.0 + 4.0 * np.clip(score, 0.0, 1.0))
    return preds


def evaluate_model_3(
    recommender: SimpleRobustRecommender,
    *,
    k: int = 10,
    test_size: float = 0.2,
    relevance_threshold: float = 4.0,
    max_users: int = 500,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Métricas equivalentes a model_metricas.py:
    - RMSE / MAE (aproximado)
    - Precision@K / NDCG@K
    """
    df = recommender.df[["userId", "tmdb_id", "rating", "user_idx", "item_idx"]].copy()

    rng = np.random.default_rng(random_state)
    users = df["user_idx"].unique()
    if max_users is not None and len(users) > max_users:
        sample = rng.choice(users, size=max_users, replace=False)
        df = df[df["user_idx"].isin(sample)].copy()
    if verbose:
        print(f"[eval] usuarios en muestra: {df['user_idx'].nunique()} | interacciones: {len(df)}")

    train_parts = []
    test_parts = []
    for _, g in df.groupby("user_idx", sort=False):
        if len(g) < 2:
            train_parts.append(g)
            continue
        g_train, g_test = train_test_split(g, test_size=test_size, random_state=random_state)
        train_parts.append(g_train)
        test_parts.append(g_test)

    df_train = pd.concat(train_parts, ignore_index=True)
    df_test = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)
    if verbose:
        print(f"[eval] train: {len(df_train)} interacciones | test: {len(df_test)} interacciones")

    # Reentrenado fair en memoria para evitar leakage y no pisar caché de inferencia.
    matrix_train = build_user_item_matrix(df_train)
    latent_dim = recommender.user_factors.shape[1]
    svd_eval = TruncatedSVD(n_components=latent_dim, random_state=42)
    user_factors_train = svd_eval.fit_transform(matrix_train)
    knn_train = NearestNeighbors(
        n_neighbors=getattr(recommender.knn_model, "n_neighbors", 80),
        metric="cosine",
        algorithm="brute",
    )
    knn_train.fit(user_factors_train)

    reco_eval = SimpleRobustRecommender(
        df=df_train,
        user_encoder=recommender.user_encoder,
        item_encoder=recommender.item_encoder,
        user_item_matrix=matrix_train,
        user_factors=user_factors_train,
        knn_model=knn_train,
        item_vote_avg=recommender.item_vote_avg,
        item_titles=recommender.item_titles,
    )

    # RMSE / MAE
    if len(df_test) > 0:
        y_true = df_test["rating"].astype(np.float32).values
        if verbose:
            print("[eval] calculando RMSE/MAE (aprox)...")
        y_pred = _predict_scores_for_pairs(
            reco_eval,
            df_test["userId"].values,
            df_test["tmdb_id"].values,
            top_for_lookup=max(k, 10),
        )
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    else:
        mae = 0.0
        rmse = 0.0

    # Ranking
    precisions = []
    ndcgs = []
    test_pos = (
        df_test[df_test["rating"] >= relevance_threshold]
        .groupby("userId")["tmdb_id"]
        .apply(lambda s: set(map(int, s.values)))
    )
    if verbose:
        print(f"[eval] usuarios con positivos (rating>={relevance_threshold}): {len(test_pos)}")

    for user_id, pos_items in test_pos.items():
        recs = reco_eval.recommend_top10(int(user_id))
        rec_tmdb = [int(r["tmdb_id"]) for r in recs[:k]]
        rel = [1 if itm in pos_items else 0 for itm in rec_tmdb]
        precisions.append(precision_at_k(rel, k))
        ndcgs.append(ndcg_at_k(rel, k))

    precision_k = float(np.mean(precisions)) if precisions else 0.0
    ndcg_k = float(np.mean(ndcgs)) if ndcgs else 0.0

    return {
        "RMSE": rmse,
        "MAE": mae,
        f"Precision@{k}": precision_k,
        f"NDCG@{k}": ndcg_k,
        "users_evaluated_ranking": float(len(precisions)),
        "test_interactions": float(len(df_test)),
    }


def build_recommender(
    force_reprocess: bool = False,
    force_train: bool = False,
    min_ratings: int = 30,
    latent_dim: int = 50,
    knn_neighbors_fit: int = 80,
) -> SimpleRobustRecommender:
    df, user_enc, item_enc = get_processed_data(force_reprocess=force_reprocess, min_ratings=min_ratings)
    matrix = build_user_item_matrix(df)
    user_factors = get_user_factors(matrix, latent_dim=latent_dim, force_train=force_train)
    knn = get_knn_model(user_factors, n_neighbors=knn_neighbors_fit, force_train=force_train)
    item_vote = get_item_vote_feature(item_enc)
    item_titles = get_item_title_feature(item_enc)
    return SimpleRobustRecommender(df, user_enc, item_enc, matrix, user_factors, knn, item_vote, item_titles)


if __name__ == "__main__":
    recommender = build_recommender(
        force_reprocess=False,
        force_train=False,
        min_ratings=30,
        latent_dim=50,
        knn_neighbors_fit=80,
    )

    sample_user_id = int(recommender.df["userId"].iloc[0])
    top10 = recommender.recommend_top10(sample_user_id)
    print(f"Top 10 para userId={sample_user_id}")
    for i, item in enumerate(top10, start=1):
        print(f"{i:02d}. tmdb_id={item['tmdb_id']} | titulo={item['titulo']} | score={item['score']:.6f}")

    metrics = evaluate_model_3(
        recommender,
        k=10,
        test_size=0.2,
        relevance_threshold=4.0,
        max_users=500,
        random_state=42,
        verbose=True,
    )
    print("Métricas (muestra):", metrics)

    # Registro centralizado en historial_metricas.csv
    registrar_metricas(
        modelo="SVD+KNN+Rerank Simple (TX)",
        hiperparams={
            "min_ratings": 30,
            "latent_dim": 50,
            "knn_neighbors_fit": 80,
        },
        metricas={
            "RMSE": metrics.get("RMSE", 0.0),
            "MAE": metrics.get("MAE", 0.0),
            "Precision_10": metrics.get("Precision@10", 0.0),
            "NDCG_10": metrics.get("NDCG@10", 0.0),
        },
        dataset_size=len(recommender.df),
        notas="Evaluación muestreo 500 usuarios, test_size=0.2",
    )

