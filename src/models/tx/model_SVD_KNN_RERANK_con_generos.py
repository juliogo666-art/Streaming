import ast
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Tracking centralizado de métricas (historial_metricas.csv)
from src.utils.registrar_metricas import registrar_metricas


DATA_READY_RATINGS = "src/data/ready/ratings_finales_ia.csv"
DATA_READY_CATALOG_MOVIES = "src/data/ready/dataset_final_movies.csv"


CACHE_DIR = "src/data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

PATH_DF_PREP = os.path.join(CACHE_DIR, "df_prep.parquet")
PATH_USER_ENC = os.path.join(CACHE_DIR, "user_encoder.joblib")
PATH_ITEM_ENC = os.path.join(CACHE_DIR, "item_encoder.joblib")

PATH_SVD_MODEL = os.path.join(CACHE_DIR, "model_svd.joblib")
PATH_USER_FACTORS = os.path.join(CACHE_DIR, "user_factors.npy")
PATH_ITEM_FACTORS = os.path.join(CACHE_DIR, "item_factors.npy")

PATH_KNN_USER = os.path.join(CACHE_DIR, "model_knn_user.joblib")

PATH_CATALOG_FEATURES = os.path.join(CACHE_DIR, "catalog_movies_item_features.joblib")


def get_processed_data(force_reprocess: bool = False, min_ratings: int = 5) -> Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    """
    Preprocesado de ratings con caché compartida.
    Devuelve df_prep con columnas: userId, tmdb_id, rating, user_idx, item_idx.
    """
    if not force_reprocess and os.path.exists(PATH_DF_PREP) and os.path.exists(PATH_USER_ENC) and os.path.exists(PATH_ITEM_ENC):
        df_prep = pd.read_parquet(PATH_DF_PREP)
        user_enc = joblib.load(PATH_USER_ENC)
        item_enc = joblib.load(PATH_ITEM_ENC)
        return df_prep, user_enc, item_enc

    df_ratings = pd.read_csv(DATA_READY_RATINGS)
    df_ratings = df_ratings.drop_duplicates(["userId", "tmdb_id"])

    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df_ratings["user_idx"] = user_enc.fit_transform(df_ratings["userId"])
    df_ratings["item_idx"] = item_enc.fit_transform(df_ratings["tmdb_id"])

    # Filtrado por relevancia (mínimo min_ratings ratings por ítem)
    counts = df_ratings["item_idx"].value_counts()
    keep_item_idx = counts[counts > min_ratings].index
    df_prep = df_ratings[df_ratings["item_idx"].isin(keep_item_idx)].copy()

    # Tipos ligeros para RAM
    df_prep["rating"] = df_prep["rating"].astype(np.float32)
    df_prep["user_idx"] = df_prep["user_idx"].astype(np.int32)
    df_prep["item_idx"] = df_prep["item_idx"].astype(np.int32)

    df_prep.to_parquet(PATH_DF_PREP)
    joblib.dump(user_enc, PATH_USER_ENC)
    joblib.dump(item_enc, PATH_ITEM_ENC)
    return df_prep, user_enc, item_enc


def build_user_item_matrix(df_prep: pd.DataFrame) -> csr_matrix:
    """Construye una matriz user-item dispersa con ratings explícitos."""
    return csr_matrix(
        (df_prep["rating"], (df_prep["user_idx"], df_prep["item_idx"])),
        shape=(df_prep["user_idx"].max() + 1, df_prep["item_idx"].max() + 1),
    )


def get_user_factors(
    user_item_matrix: csr_matrix, latent_dim: int = 50, force_train: bool = False
) -> Tuple[TruncatedSVD, np.ndarray]:
    """
    Obtiene factores latentes de usuarios con SVD (se usan como "espacio" para kNN).
    """
    if (
        not force_train
        and os.path.exists(PATH_SVD_MODEL)
        and os.path.exists(PATH_USER_FACTORS)
        and os.path.exists(PATH_ITEM_FACTORS)
    ):
        svd = joblib.load(PATH_SVD_MODEL)
        user_factors = np.load(PATH_USER_FACTORS)
        # Si el caché no coincide con latent_dim, reentrenamos.
        if getattr(svd, "n_components", None) == latent_dim and user_factors.shape[1] == latent_dim:
            return svd, user_factors

    svd = TruncatedSVD(n_components=latent_dim, random_state=42)
    user_factors = svd.fit_transform(user_item_matrix)
    item_factors = svd.components_.T

    joblib.dump(svd, PATH_SVD_MODEL)
    np.save(PATH_USER_FACTORS, user_factors)
    np.save(PATH_ITEM_FACTORS, item_factors)
    return svd, user_factors


def get_user_knn(user_factors: np.ndarray, n_neighbors: int = 50, force_train: bool = False) -> NearestNeighbors:
    """
    kNN en el espacio latente (para encontrar usuarios similares).
    """
    if not force_train and os.path.exists(PATH_KNN_USER):
        try:
            return joblib.load(PATH_KNN_USER)
        except Exception:
            pass

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    knn.fit(user_factors)
    joblib.dump(knn, PATH_KNN_USER)
    return knn


def _parse_genre_ids(value) -> Set[int]:
    """
    Convierte el campo genre_ids (string tipo "[27, 28]") a set[int].
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    if isinstance(value, str):
        s = value.strip()
        if not s or s.lower() == "nan":
            return set()
        # genre_ids suele venir como lista en formato JSON-like
        try:
            parsed = ast.literal_eval(s)
        except Exception:
            return set()
    elif isinstance(value, (list, tuple, set)):
        parsed = value
    else:
        return set()

    if not isinstance(parsed, (list, tuple, set)):
        return set()
    out: Set[int] = set()
    for g in parsed:
        try:
            out.add(int(g))
        except Exception:
            continue
    return out


@dataclass
class CatalogFeatures:
    item_genres: List[Set[int]]  # índice item_idx -> set(genres)
    item_vote_avg: np.ndarray  # índice item_idx -> float
    item_title: List[str]  # índice item_idx -> titulo


def load_catalog_features(item_encoder: LabelEncoder, force_rebuild: bool = False) -> CatalogFeatures:
    """
    Prepara las features de re-rankeado usando el catálogo:
    - genre_ids (para afinidad del usuario)
    - vote_average y titulo (para robustez de scoring)
    """
    if not force_rebuild and os.path.exists(PATH_CATALOG_FEATURES):
        try:
            features = joblib.load(PATH_CATALOG_FEATURES)
            if (
                len(features.item_genres) == len(item_encoder.classes_)
                and features.item_vote_avg.shape[0] == len(item_encoder.classes_)
            ):
                return features
        except Exception:
            # Caché incompatible/corrupta (p.ej. pickles serializados desde otro contexto de módulo).
            # Forzamos reconstrucción limpia.
            force_rebuild = True

    catalog_df = pd.read_csv(DATA_READY_CATALOG_MOVIES)
    if "tmdb_id" not in catalog_df.columns:
        raise ValueError(f"El catálogo no tiene columna 'tmdb_id': {DATA_READY_CATALOG_MOVIES}")

    # Garantiza tipos numéricos si es posible
    catalog_df["tmdb_id"] = pd.to_numeric(catalog_df["tmdb_id"], errors="coerce")
    catalog_df = catalog_df.dropna(subset=["tmdb_id"])
    catalog_df["tmdb_id"] = catalog_df["tmdb_id"].astype(np.int64)

    # Preparar índices para lookup rápido por tmdb_id
    catalog_df = catalog_df.set_index("tmdb_id", drop=False)

    genres_col = "genre_ids" if "genre_ids" in catalog_df.columns else None
    vote_avg_col = "vote_average" if "vote_average" in catalog_df.columns else None
    title_col = "titulo" if "titulo" in catalog_df.columns else None

    # Construimos arreglos alineados con item_idx (que corresponde a item_encoder.transform(tmdb_id))
    n_items = len(item_encoder.classes_)
    item_genres: List[Set[int]] = [set() for _ in range(n_items)]
    item_vote_avg = np.zeros(n_items, dtype=np.float32)
    item_title: List[str] = ["" for _ in range(n_items)]

    for item_idx, tmdb_id in enumerate(item_encoder.classes_):
        tmdb_id_int: Optional[int]
        try:
            tmdb_id_int = int(tmdb_id)
        except Exception:
            tmdb_id_int = None
        if tmdb_id_int is None or tmdb_id_int not in catalog_df.index:
            continue

        row = catalog_df.loc[tmdb_id_int]

        if genres_col is not None and genres_col in row:
            item_genres[item_idx] = _parse_genre_ids(row[genres_col])

        if vote_avg_col is not None and vote_avg_col in row:
            try:
                item_vote_avg[item_idx] = float(row[vote_avg_col])
            except Exception:
                item_vote_avg[item_idx] = 0.0

        if title_col is not None and title_col in row:
            item_title[item_idx] = str(row[title_col]) if not pd.isna(row[title_col]) else ""

    features = CatalogFeatures(
        item_genres=item_genres,
        item_vote_avg=item_vote_avg,
        item_title=item_title,
    )
    joblib.dump(features, PATH_CATALOG_FEATURES)
    return features


def _user_genre_profile(
    user_item_matrix: csr_matrix,
    user_idx: int,
    item_genres: Sequence[Set[int]],
    rated_item_indices: Optional[np.ndarray] = None,
    rated_item_ratings: Optional[np.ndarray] = None,
) -> Dict[int, float]:
    """
    Construye un perfil de géneros del usuario como frecuencias ponderadas por rating.
    """
    if rated_item_indices is None or rated_item_ratings is None:
        row = user_item_matrix.getrow(user_idx)
        rated_item_indices = row.indices
        rated_item_ratings = row.data

    genre_freq: Dict[int, float] = {}
    for item_i, rating in zip(rated_item_indices, rated_item_ratings):
        genres = item_genres[item_i] if item_i < len(item_genres) else set()
        if not genres:
            continue
        w = float(rating)
        for g in genres:
            genre_freq[g] = genre_freq.get(g, 0.0) + w
    return genre_freq


class KNNRecommenderTopN:
    def __init__(
        self,
        df_prep: pd.DataFrame,
        user_encoder: LabelEncoder,
        item_encoder: LabelEncoder,
        user_item_matrix: csr_matrix,
        user_factors: np.ndarray,
        knn_model: NearestNeighbors,
        catalog_features: CatalogFeatures,
    ):
        self.df_prep = df_prep
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.user_item_matrix = user_item_matrix
        self.user_factors = user_factors
        self.knn_model = knn_model
        self.catalog_features = catalog_features

        # Popularidad: conteos por item (para cold-start)
        self.item_rating_counts = np.bincount(df_prep["item_idx"].values, minlength=user_item_matrix.shape[1]).astype(np.float32)

    def recommend(
        self,
        raw_user_id,
        top_n: int = 10,
        n_neighbors: int = 50,
        n_candidates: int = 200,
        rerank_alpha: float = 0.7,
        popularity_weight: float = 0.3,
        genre_weight: float = 0.7,
    ) -> List[Dict[str, object]]:
        """
        Devuelve un ranking top-N para un userId.
        - kNN genera candidatos con scoring basado en vecinos
        - re-rankeado mezcla scoring kNN y afinidad de géneros + vote_average del catálogo
        """
        # Validación de alpha
        rerank_alpha = float(np.clip(rerank_alpha, 0.0, 1.0))
        genre_weight = float(np.clip(genre_weight, 0.0, 1.0))
        popularity_weight = 1.0 - genre_weight if popularity_weight is None else float(np.clip(popularity_weight, 0.0, 1.0))

        # Obtener índices del usuario
        try:
            user_idx = int(self.user_encoder.transform([raw_user_id])[0])
        except Exception:
            return self._cold_start(raw_user_id, top_n=top_n)

        # Ítems ya vistos
        user_row = self.user_item_matrix.getrow(user_idx)
        seen_item_indices = set(user_row.indices.tolist())
        if user_row.nnz == 0:
            return self._cold_start(raw_user_id, top_n=top_n)

        # Vecinos: en el espacio latente
        # Nota: knn_model se entrena con n_neighbors fijo; por eso aseguramos pedir al menos n_neighbors+1.
        max_neighbors_fit = getattr(self.knn_model, "n_neighbors", None)
        distances, neighbor_indices = self.knn_model.kneighbors(
            self.user_factors[user_idx].reshape(1, -1),
            n_neighbors=min(n_neighbors + 1, self.user_factors.shape[0], max_neighbors_fit) if max_neighbors_fit is not None else min(n_neighbors + 1, self.user_factors.shape[0]),
            return_distance=True,
        )
        distances = distances[0]
        neighbor_indices = neighbor_indices[0]
        similarities = 1.0 - distances  # cosine: sim = 1 - distance

        # Quitar el propio usuario si aparece
        candidates_neighbors = [
            (int(v), float(sim))
            for v, sim in zip(neighbor_indices, similarities)
            if int(v) != user_idx and sim > -1e9
        ]
        if not candidates_neighbors:
            return self._cold_start(raw_user_id, top_n=top_n)

        # Generar candidatos con scoring kNN basado en ratings de vecinos + normalización por sum(similitud)
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
                cand_score_sum[item_i] = cand_score_sum.get(item_i, 0.0) + sim * float(rating_v)
                cand_sim_sum[item_i] = cand_sim_sum.get(item_i, 0.0) + abs(sim)

        if not cand_score_sum:
            return self._cold_start(raw_user_id, top_n=top_n)

        # Top candidatos por scoring kNN normalizado
        cand_items = np.array(list(cand_score_sum.keys()), dtype=np.int32)
        cand_knn_scores = np.array(
            [cand_score_sum[i] / (cand_sim_sum[i] + 1e-12) for i in cand_items],
            dtype=np.float32,
        )
        top_idx = np.argsort(cand_knn_scores)[::-1][: min(n_candidates, len(cand_items))]
        cand_items = cand_items[top_idx]
        cand_knn_scores = cand_knn_scores[top_idx]

        # Normalizar scoring kNN
        knn_min = float(np.min(cand_knn_scores))
        knn_max = float(np.max(cand_knn_scores))
        denom = (knn_max - knn_min) + 1e-12
        cand_knn_norm = (cand_knn_scores - knn_min) / denom

        # Perfil de géneros del usuario (para re-rankeado)
        rated_item_indices = user_row.indices
        rated_item_ratings = user_row.data
        genre_freq = _user_genre_profile(
            self.user_item_matrix,
            user_idx,
            self.catalog_features.item_genres,
            rated_item_indices=rated_item_indices,
            rated_item_ratings=rated_item_ratings,
        )
        total_genre = float(sum(genre_freq.values())) + 1e-12

        # Normalizar vote_average dentro del set candidato (evita que la escala cambie)
        cand_vote_avg = self.catalog_features.item_vote_avg[cand_items]
        vmin = float(np.min(cand_vote_avg))
        vmax = float(np.max(cand_vote_avg))
        vden = (vmax - vmin) + 1e-12
        cand_vote_norm = (cand_vote_avg - vmin) / vden

        # Re-rankeado
        reranked_scores = []
        for idx, (item_i, knn_norm) in enumerate(zip(cand_items, cand_knn_norm)):
            genres = self.catalog_features.item_genres[int(item_i)]
            genre_score = 0.0
            if genres:
                genre_score = sum(genre_freq.get(g, 0.0) for g in genres) / total_genre

            content_score = genre_weight * genre_score + popularity_weight * float(cand_vote_norm[idx])
            final_score = rerank_alpha * float(knn_norm) + (1.0 - rerank_alpha) * float(content_score)
            reranked_scores.append(final_score)

        reranked_scores = np.array(reranked_scores, dtype=np.float32)
        order = np.argsort(reranked_scores)[::-1][: min(top_n, len(cand_items))]

        final_items = cand_items[order]
        final_scores = reranked_scores[order]

        # Decodificar a tmdb_id
        tmdb_ids = self.item_encoder.inverse_transform(final_items)

        results: List[Dict[str, object]] = []
        for tmdb_id, item_i, score in zip(tmdb_ids, final_items, final_scores):
            item_i_int = int(item_i)
            title = self.catalog_features.item_title[item_i_int] if item_i_int < len(self.catalog_features.item_title) else ""
            results.append(
                {
                    "tmdb_id": int(tmdb_id),
                    "titulo": title,
                    "score": float(score),
                }
            )
        return results

    def _cold_start(self, raw_user_id, top_n: int = 10) -> List[Dict[str, object]]:
        """
        Si el usuario es desconocido, devolvemos popularidad (fallback).
        """
        # items ordenados por numero de ratings (y desempate por vote_average si se desea)
        order = np.argsort(self.item_rating_counts)[::-1][:top_n]
        tmdb_ids = self.item_encoder.inverse_transform(order)
        results: List[Dict[str, object]] = []
        for tmdb_id, item_i in zip(tmdb_ids, order):
            item_i_int = int(item_i)
            title = self.catalog_features.item_title[item_i_int] if item_i_int < len(self.catalog_features.item_title) else ""
            results.append({"tmdb_id": int(tmdb_id), "titulo": title, "score": float(self.item_rating_counts[item_i_int])})
        return results


def _dcg_at_k(relevances: np.ndarray, k: int) -> float:
    rel = relevances[:k].astype(np.float32)
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2, dtype=np.float32))
    return float(np.sum(rel * discounts))


def ndcg_at_k(relevances: Sequence[int], k: int) -> float:
    rel = np.asarray(relevances, dtype=np.float32)
    dcg = _dcg_at_k(rel, k)
    ideal = np.sort(rel)[::-1]
    idcg = _dcg_at_k(ideal, k)
    return 0.0 if idcg <= 0 else float(dcg / idcg)


def precision_at_k(relevances: Sequence[int], k: int) -> float:
    rel = np.asarray(relevances, dtype=np.float32)[:k]
    if rel.size == 0:
        return 0.0
    return float(np.sum(rel > 0) / k)


def _predict_scores_for_pairs(
    recommender: KNNRecommenderTopN,
    user_ids: np.ndarray,
    item_ids_tmdb: np.ndarray,
    top_for_lookup: int = 500,
) -> np.ndarray:
    """
    Predice un score para pares (userId, tmdb_id) usando el ranking top-N del recommender.
    Si un ítem no aparece en el top_for_lookup, score=0.0 (aproximación).
    """
    preds = np.zeros(len(user_ids), dtype=np.float32)
    cache: Dict[int, Dict[int, float]] = {}

    for i, (u, tmdb) in enumerate(zip(user_ids, item_ids_tmdb)):
        u_int = int(u)
        tmdb_int = int(tmdb)
        if u_int not in cache:
            recs = recommender.recommend(u_int, top_n=top_for_lookup)
            cache[u_int] = {int(r["tmdb_id"]): float(r["score"]) for r in recs}
        score = float(cache[u_int].get(tmdb_int, 0.0))
        # Normalizar score 0..1 a rango aproximado de rating 1..5
        preds[i] = float(1.0 + 4.0 * np.clip(score, 0.0, 1.0))
    return preds


def evaluate_model_2(
    df_all: pd.DataFrame,
    user_encoder: LabelEncoder,
    item_encoder: LabelEncoder,
    catalog_features: CatalogFeatures,
    *,
    k: int = 10,
    test_size: float = 0.2,
    relevance_threshold: float = 4.0,
    max_users: int = 500,
    random_state: int = 42,
    latent_dim: int = 50,
    knn_neighbors_fit: int = 80,
    n_neighbors_reco: int = 50,
    n_candidates: int = 500,
    rerank_alpha: float = 0.7,
    genre_weight: float = 0.7,
    force_train_models: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evalúa:
    - RMSE/MAE (sobre ratings explícitos del test; predicción aproximada via score del ranking)
    - Precision@K y NDCG@K (sobre positivos del test con rating >= relevance_threshold)

    NOTA: con datasets muy grandes, usa max_users para muestrear usuarios y hacerlo viable.
    """
    df = df_all[["userId", "tmdb_id", "rating", "user_idx", "item_idx"]].copy()

    # Muestreo de usuarios para velocidad
    rng = np.random.default_rng(random_state)
    unique_users = df["user_idx"].unique()
    if max_users is not None and len(unique_users) > max_users:
        sampled_users = rng.choice(unique_users, size=max_users, replace=False)
        df = df[df["user_idx"].isin(sampled_users)].copy()
    if verbose:
        print(f"[eval] usuarios en muestra: {df['user_idx'].nunique()} | interacciones: {len(df)}")

    # Split por usuario (cada usuario aporta test_size de sus interacciones)
    train_parts = []
    test_parts = []
    for u, g in df.groupby("user_idx", sort=False):
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

    # Entrena recommender SOLO con train.
    # Por defecto forzamos reentrenado durante evaluación para evitar leakage desde cachés globales.
    user_item_matrix_train = build_user_item_matrix(df_train)
    if verbose:
        print("[eval] preparando SVD/kNN...")
        if not force_train_models:
            print("[eval][warning] force_train_models=False puede reutilizar caché global y sesgar métricas.")
    _, user_factors = get_user_factors(user_item_matrix_train, latent_dim=latent_dim, force_train=force_train_models)
    knn_model = get_user_knn(user_factors, n_neighbors=knn_neighbors_fit, force_train=force_train_models)

    reco = KNNRecommenderTopN(
        df_prep=df_train,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        user_item_matrix=user_item_matrix_train,
        user_factors=user_factors,
        knn_model=knn_model,
        catalog_features=catalog_features,
    )

    # ---------- RMSE / MAE ----------
    # Predicción aproximada: score del ranking para el ítem del test (si no aparece en top_for_lookup -> 0)
    if len(df_test) > 0:
        y_true = df_test["rating"].astype(np.float32).values
        if verbose:
            print("[eval] calculando RMSE/MAE (aprox)...")
        y_pred = _predict_scores_for_pairs(
            reco,
            df_test["userId"].values,
            df_test["tmdb_id"].values,
            top_for_lookup=max(k, 500),
        )
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    else:
        mae = 0.0
        rmse = 0.0

    # ---------- Ranking: Precision@K y NDCG@K ----------
    precisions = []
    ndcgs = []

    # Positivos del test por usuario (en tmdb_id)
    test_pos = (
        df_test[df_test["rating"] >= relevance_threshold]
        .groupby("userId")["tmdb_id"]
        .apply(lambda s: set(map(int, s.values)))
    )
    if verbose:
        print(f"[eval] usuarios con positivos (rating>={relevance_threshold}): {len(test_pos)}")

    for user_id, pos_items in test_pos.items():
        if not pos_items:
            continue
        recs = reco.recommend(
            int(user_id),
            top_n=k,
            n_neighbors=n_neighbors_reco,
            n_candidates=n_candidates,
            rerank_alpha=rerank_alpha,
            genre_weight=genre_weight,
        )
        rec_tmdb = [int(r["tmdb_id"]) for r in recs]
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
    force_svd_train: bool = False,
    force_knn_train: bool = False,
    force_catalog_features_rebuild: bool = False,
    min_ratings: int = 5,
    latent_dim: int = 50,
    knn_neighbors_fit: int = 80,
) -> KNNRecommenderTopN:
    df_prep, user_encoder, item_encoder = get_processed_data(force_reprocess=force_reprocess, min_ratings=min_ratings)
    user_item_matrix = build_user_item_matrix(df_prep)

    _, user_factors = get_user_factors(user_item_matrix, latent_dim=latent_dim, force_train=force_svd_train)
    knn_model = get_user_knn(user_factors, n_neighbors=knn_neighbors_fit, force_train=force_knn_train)
    catalog_features = load_catalog_features(item_encoder, force_rebuild=force_catalog_features_rebuild)

    return KNNRecommenderTopN(
        df_prep=df_prep,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        user_item_matrix=user_item_matrix,
        user_factors=user_factors,
        knn_model=knn_model,
        catalog_features=catalog_features,
    )


if __name__ == "__main__":
    # Entrenamiento/carga con caché y ejemplo de uso.
    recommender = build_recommender(
        force_reprocess=False,
        force_svd_train=False,
        force_knn_train=False,
        force_catalog_features_rebuild=False,
        min_ratings=30,
        latent_dim=50,
        knn_neighbors_fit=80,
    )

    # Ejemplo: si el userId existe en ratings_finales_ia.csv
    example_user_id = int(recommender.df_prep["userId"].iloc[0])
    recs = recommender.recommend(example_user_id, top_n=10)
    print(f"Top-N para userId={example_user_id}:")
    for r in recs:
        print(r)

    # Evaluación rápida (muestreando usuarios para que sea viable)
    metrics = evaluate_model_2(
        recommender.df_prep,
        recommender.user_encoder,
        recommender.item_encoder,
        recommender.catalog_features,
        k=10,
        test_size=0.2,
        relevance_threshold=4.0,
        max_users=500,
        random_state=42,
        latent_dim=50,
        knn_neighbors_fit=80,
        n_neighbors_reco=50,
        n_candidates=500,
        rerank_alpha=0.7,
        genre_weight=0.7,
        force_train_models=True,
        verbose=True,
    )
    print("Métricas (muestra):", metrics)

    # Registro centralizado en historial_metricas.csv
    registrar_metricas(
        modelo="SVD+KNN+Rerank Géneros (TX)",
        hiperparams={
            "min_ratings": 30,
            "latent_dim": 50,
            "knn_neighbors_fit": 80,
            "rerank_alpha": 0.7,
            "genre_weight": 0.7,
        },
        metricas={
            "RMSE": metrics.get("RMSE", 0.0),
            "MAE": metrics.get("MAE", 0.0),
            "Precision_10": metrics.get("Precision@10", 0.0),
            "NDCG_10": metrics.get("NDCG@10", 0.0),
        },
        dataset_size=len(recommender.df_prep),
        notas="Evaluación muestreo 500 usuarios, test_size=0.2",
    )

