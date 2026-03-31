"""
evaluar_modelo_final.py
=======================
Script de evaluacion oficial del modelo NCF-Lite para la competicion de modelos.

Calcula las 4 metricas acordadas para la decision final:
  - RMSE         (error de regresion, logit -> escala 0-5)
  - MAE          (error de regresion, logit -> escala 0-5)
  - NDCG@10      (calidad de ranking con descuento logaritmico)
  - Precision@10 (fraccion de aciertos en el Top-10 recomendado)

Flujo:
  1. Carga y filtrado K-Core (K=500) del dataset.
  2. Split Leave-One-Out por usuario.
  3. Entrenamiento NCF-Lite (5 epocas, BCE pairwise).
  4. Evaluacion por usuario sobre el conjunto de test.
  5. Impresion de tabla resumen por consola. Sin guardar archivos.

Uso
---
    python src/models/evaluar_modelo_final.py
    python src/models/evaluar_modelo_final.py --k-core 300 --epochs 10 --topk 10
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Logger  — nivel WARNING para silenciar INFO durante la evaluacion limpia
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend de computo: DirectML (AMD/Intel GPU en Windows) o CPU como fallback
# ---------------------------------------------------------------------------
try:
    import torch_directml  # type: ignore
    DEVICE = torch_directml.device()
    _BACKEND: str = f"DirectML (GPU) — {torch_directml.device_name(0)}"
except ImportError:
    DEVICE = torch.device("cpu")
    _BACKEND = "CPU (torch-directml no disponible)"

# ---------------------------------------------------------------------------
# Constantes por defecto
# ---------------------------------------------------------------------------
DATA_PATH: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "ready", "ratings_finales_ia.csv"
)
MIN_K_CORE: int = 500     # K-Core agresivo para reducir datos y acelerar la prueba
EMB_DIM: int = 32
BATCH_SIZE: int = 2048
EPOCHS: int = 5
NEG_SAMPLES: int = 4
LEARNING_RATE: float = 1e-3
TOP_K: int = 10
MAX_EVAL_USERS: int = 1000  # Cap de usuarios evaluados (velocidad en PC local)


# ===========================================================================
# BLOQUE 1 — CARGA Y FILTRADO K-CORE
# ===========================================================================

def cargar_y_filtrar(
    ruta: str,
    k_core: int = MIN_K_CORE,
) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    """
    Lee el CSV de ratings y aplica filtrado K-Core iterativo hasta convergencia.
    Todos los usuarios e items que permanezcan tendran al menos `k_core`
    interacciones, garantizando un subgrafo denso y bien estimado.

    Parameters
    ----------
    ruta : str
        Ruta al CSV con columnas (userId, tmdb_id, rating).
    k_core : int
        Numero minimo de valoraciones requeridas tanto para usuarios como items.

    Returns
    -------
    df : pd.DataFrame
        Dataset filtrado con IDs reindexados a enteros contiguos [0, N).
    user2idx : Dict[int, int]
        Mapeo userId_original -> indice de embedding.
    item2idx : Dict[int, int]
        Mapeo tmdb_id_original -> indice de embedding.
    """
    logger.info("Cargando datos: %s", ruta)
    df = pd.read_csv(ruta, dtype={"userId": "int32", "tmdb_id": "int32", "rating": "float32"})
    logger.info("  Filas originales : %d", len(df))

    prev_len = -1
    itr = 0
    while len(df) != prev_len:
        prev_len = len(df)
        itr += 1
        uc = df["userId"].value_counts()
        df = df[df["userId"].isin(uc[uc >= k_core].index)]
        ic = df["tmdb_id"].value_counts()
        df = df[df["tmdb_id"].isin(ic[ic >= k_core].index)]

    logger.info(
        "  K-Core (%d): %d filas | %d usuarios | %d items (%d iters)",
        k_core, len(df), df["userId"].nunique(), df["tmdb_id"].nunique(), itr,
    )

    user2idx: Dict[int, int] = {int(u): i for i, u in enumerate(sorted(df["userId"].unique()))}
    item2idx: Dict[int, int] = {int(it): i for i, it in enumerate(sorted(df["tmdb_id"].unique()))}
    df["userId"] = df["userId"].map(user2idx).astype("int32")
    df["tmdb_id"] = df["tmdb_id"].map(item2idx).astype("int32")
    return df.reset_index(drop=True), user2idx, item2idx


# ===========================================================================
# BLOQUE 2 — SPLIT LEAVE-ONE-OUT
# ===========================================================================

def split_leave_one_out(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Para cada usuario reserva su interaccion con el rating mas alto como
    muestra de test (desempate por el primero en orden). El resto va a train.

    Este esquema es el estandar en la literatura de ranking (NCF, LightGCN).

    Returns
    -------
    train_df, test_df : Tuple[pd.DataFrame, pd.DataFrame]
    """
    df_sorted = df.sort_values(["userId", "rating"], ascending=[True, False])
    test_idx = df_sorted.groupby("userId").head(1).index
    test_df = df.loc[test_idx].reset_index(drop=True)
    train_df = df.drop(index=test_idx).reset_index(drop=True)
    logger.info(
        "Split LOO: train=%d | test=%d (1 item/usuario)", len(train_df), len(test_df)
    )
    return train_df, test_df


# ===========================================================================
# BLOQUE 3 — DATASET PAIRWISE
# ===========================================================================

class PairwiseDataset(Dataset):
    """
    Por cada interaccion positiva (u, i+) muestrea `neg_samples` items
    negativos que el usuario no haya valorado. No se usa ninguna matriz densa.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        n_items: int,
        neg_samples: int = NEG_SAMPLES,
    ) -> None:
        self.n_items = n_items
        self.neg_samples = neg_samples
        self.user_positives: Dict[int, Set[int]] = (
            df.groupby("userId")["tmdb_id"].apply(set).to_dict()
        )
        self.interactions: np.ndarray = df[["userId", "tmdb_id"]].values.astype(np.int64)

    def __len__(self) -> int:
        return len(self.interactions) * self.neg_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_idx = idx // self.neg_samples
        u, i_pos = self.interactions[pos_idx]
        positives = self.user_positives[u]
        i_neg = np.random.randint(self.n_items)
        while i_neg in positives:
            i_neg = np.random.randint(self.n_items)
        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i_pos, dtype=torch.long),
            torch.tensor(i_neg, dtype=torch.long),
        )


# ===========================================================================
# BLOQUE 4 — MODELO NCF-LITE (GMF + MLP)
# ===========================================================================

class NCFLite(nn.Module):
    """
    Neural Collaborative Filtering con dos ramas fusionadas:
      - GMF: producto elemento a elemento de embeddings (interacciones lineales).
      - MLP: concatenacion + capas densas ReLU (interacciones no lineales).

    Entrenado con BCE sobre pares (positivo, negativo).
    Referencia: He et al., 2017 — "Neural Collaborative Filtering."
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = EMB_DIM,
        mlp_layers: List[int] | None = None,
    ) -> None:
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [emb_dim * 2, emb_dim, emb_dim // 2]
        self._mlp_out_dim = mlp_layers[-1]

        # Embeddings independientes por rama
        self.gmf_user = nn.Embedding(n_users, emb_dim)
        self.gmf_item = nn.Embedding(n_items, emb_dim)
        self.mlp_user = nn.Embedding(n_users, emb_dim)
        self.mlp_item = nn.Embedding(n_items, emb_dim)

        # Bloque MLP dinamico
        mlp_modules: List[nn.Module] = []
        in_size = emb_dim * 2
        for out_size in mlp_layers:
            mlp_modules += [nn.Linear(in_size, out_size), nn.ReLU()]
            in_size = out_size
        self.mlp = nn.Sequential(*mlp_modules)

        # Capa de fusion y proyeccion a logit escalar
        self.output_layer = nn.Linear(emb_dim + self._mlp_out_dim, 1)

        for emb in (self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item):
            nn.init.normal_(emb.weight, std=0.01)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Logit de relevancia para pares (u, i). Shape: [B]."""
        gmf_out = self.gmf_user(users) * self.gmf_item(items)
        mlp_in = torch.cat([self.mlp_user(users), self.mlp_item(items)], dim=1)
        mlp_out = self.mlp(mlp_in)
        fused = torch.cat([gmf_out, mlp_out], dim=1)
        return self.output_layer(fused).squeeze(1)

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """BCE Loss sobre tripletas (u, i+, i-)."""
        logits = torch.cat([self.score(users, pos_items), self.score(users, neg_items)])
        labels = torch.cat([torch.ones(len(users)), torch.zeros(len(users))])
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)

    @torch.no_grad()
    def score_all_items(self, user_idx: int, n_items: int) -> np.ndarray:
        """
        Calcula el logit del usuario `user_idx` frente a todos los items.
        Retorna un array numpy de shape [n_items].
        Solo materializa un vector, nunca la matriz N x M completa.
        """
        user_t = torch.tensor(user_idx, dtype=torch.long, device=DEVICE)
        all_items = torch.arange(n_items, dtype=torch.long, device=DEVICE)
        users_exp = user_t.expand(n_items)
        return self.score(users_exp, all_items).cpu().numpy()


# ===========================================================================
# BLOQUE 5 — ENTRENAMIENTO
# ===========================================================================

def entrenar(
    model: NCFLite,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    epochs: int,
) -> None:
    """Bucle de entrenamiento con gradient clipping."""
    model.train()
    logger.info("Entrenando NCF-Lite durante %d epocas …", epochs)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        t0 = time.perf_counter()
        for batch in train_loader:
            users, pos_items, neg_items = [t.to(DEVICE) for t in batch]
            optimizer.zero_grad()
            loss = model(users, pos_items, neg_items)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        logger.info(
            "  Epoca %d/%d | Loss: %.5f | %.1fs",
            epoch, epochs, total_loss / len(train_loader), time.perf_counter() - t0,
        )
    logger.info("Entrenamiento completado.")


# ===========================================================================
# BLOQUE 6 — EVALUACION
# ===========================================================================

@dataclass
class ResultadosEvaluacion:
    """Contenedor de las 4 metricas de la competicion."""
    rmse: float
    mae: float
    ndcg_at_k: float
    precision_at_k: float
    n_usuarios: int
    k: int


def evaluar(
    model: NCFLite,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    n_items: int,
    k: int = TOP_K,
    max_users: int = MAX_EVAL_USERS,
) -> ResultadosEvaluacion:
    """
    Evalua el modelo sobre el conjunto de test con las 4 metricas acordadas.

    Para cada usuario:
      1. Puntua todos los items con `score_all_items`.
      2. Excluye los items vistos en train (pone score=-inf).
      3. Toma el Top-K por score descendente.

    Metricas calculadas
    -------------------
    RMSE / MAE:
        Convierte el logit del item de test a escala [0, 5] mediante
        sigmoid(logit) * 5.0 y lo compara con el rating real.
        Nota: el modelo esta optimizado para ranking, no para regresion;
        estas metricas son indicativas de calibracion.

    Precision@K:
        (# items relevantes en Top-K) / K.
        Con Leave-One-Out: 0.0 si el item de test no aparece, 1/K si aparece.

    NDCG@K:
        Con un unico item relevante por usuario:
          DCG  = 1 / log2(pos + 2)  si el item esta en Top-K (pos = rango 0-indexado)
          IDCG = 1  (el item relevante en posicion 0 seria el ideal)
          NDCG = DCG / IDCG

    Parameters
    ----------
    max_users : int
        Limite de usuarios evaluados. Garantiza un tiempo acotado en PC local.
    """
    model.eval()

    # Items vistos en train por usuario (para exclusion en el ranking)
    train_seen: Dict[int, Set[int]] = (
        train_df.groupby("userId")["tmdb_id"].apply(set).to_dict()
    )

    test_users = test_df["userId"].unique()
    if len(test_users) > max_users:
        rng = np.random.default_rng(seed=42)
        test_users = rng.choice(test_users, size=max_users, replace=False)

    sq_errors: List[float] = []
    abs_errors: List[float] = []
    ndcg_scores: List[float] = []
    precision_scores: List[float] = []

    logger.info("Evaluando %d usuarios (Top-%d) …", len(test_users), k)
    t0 = time.perf_counter()

    for uid in test_users:
        row = test_df[test_df["userId"] == uid].iloc[0]
        test_item: int = int(row["tmdb_id"])
        true_rating: float = float(row["rating"])

        # --- Score de todos los items (vector 1D, sin matrix densa) ---
        scores = model.score_all_items(uid, n_items)

        # Excluir items vistos en train
        for seen_item in train_seen.get(uid, set()):
            scores[seen_item] = -np.inf

        # --- RMSE / MAE: convertir logit del item de test a escala 0-5 ---
        raw_logit = float(scores[test_item]) if scores[test_item] != -np.inf else 0.0
        pred_rating = float(1.0 / (1.0 + math.exp(-raw_logit))) * 5.0  # sigmoid * 5
        sq_errors.append((pred_rating - true_rating) ** 2)
        abs_errors.append(abs(pred_rating - true_rating))

        # --- Top-K por score descendente ---
        top_k_idx = int(min(k, n_items))
        # argpartition para eficiencia: O(n) en vez de O(n log n)
        candidates = np.argpartition(scores, -top_k_idx)[-top_k_idx:]
        top_k: List[int] = candidates[np.argsort(scores[candidates])[::-1]].tolist()

        # --- Precision@K ---
        hit = 1 if test_item in top_k else 0
        precision_scores.append(hit / k)

        # --- NDCG@K (un unico item relevante por usuario) ---
        if hit:
            pos = top_k.index(test_item)          # rango 0-indexado
            ndcg = 1.0 / math.log2(pos + 2)       # IDCG = 1 con 1 item relevante
        else:
            ndcg = 0.0
        ndcg_scores.append(ndcg)

    elapsed = time.perf_counter() - t0
    logger.info("Evaluacion completada en %.1f s", elapsed)

    return ResultadosEvaluacion(
        rmse=float(np.sqrt(np.mean(sq_errors))),
        mae=float(np.mean(abs_errors)),
        ndcg_at_k=float(np.mean(ndcg_scores)),
        precision_at_k=float(np.mean(precision_scores)),
        n_usuarios=len(test_users),
        k=k,
    )


# ===========================================================================
# BLOQUE 7 — PRESENTACION DE RESULTADOS
# ===========================================================================

def imprimir_tabla(resultados: ResultadosEvaluacion) -> None:
    """
    Imprime una tabla ASCII profesional con las 4 metricas de la competicion.
    Usa solo caracteres ASCII para maxima compatibilidad en terminales Windows.
    """
    k = resultados.k
    filas = [
        ("RMSE",           f"{resultados.rmse:.6f}"),
        ("MAE",            f"{resultados.mae:.6f}"),
        (f"NDCG@{k}",      f"{resultados.ndcg_at_k:.6f}"),
        (f"Precision@{k}", f"{resultados.precision_at_k:.6f}"),
    ]

    titulo  = f"  NCF-Lite - Resultados de Evaluacion Final  "
    col_w   = max(len(f[0]) for f in filas) + 2   # ancho columna metrica
    val_w   = max(len(f[1]) for f in filas) + 2   # ancho columna valor
    total_w = col_w + val_w + 3                    # 3 = separadores '|'+'|'

    # Ajustar titulo si es mas ancho que la tabla
    total_w = max(total_w, len(titulo) + 2)
    col_w   = total_w - val_w - 3

    sep      = "+" + "-" * col_w + "+" + "-" * val_w + "+"
    sep_wide = "+" + "-" * (total_w) + "+"

    print()
    print(sep_wide)
    print("|" + titulo.center(total_w) + "|")
    print(sep_wide)
    print("|" + " Metrica".ljust(col_w) + "|" + " Valor".ljust(val_w) + "|")
    print(sep)
    for metrica, valor in filas:
        print("|" + f" {metrica}".ljust(col_w) + "|" + f" {valor}".ljust(val_w) + "|")
    print(sep)
    print("|" + f" Usuarios evaluados : {resultados.n_usuarios}".ljust(total_w) + "|")
    print(sep_wide)
    print()


# ===========================================================================
# BLOQUE 8 — PUNTO DE ENTRADA
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evalua NCF-Lite con RMSE, MAE, NDCG@K y Precision@K."
    )
    parser.add_argument("--data-path",  type=str,   default=DATA_PATH)
    parser.add_argument("--k-core",     type=int,   default=MIN_K_CORE,       help="Umbral K-Core (default: 500)")
    parser.add_argument("--emb-dim",    type=int,   default=EMB_DIM,          help="Dimension embedding (default: 32)")
    parser.add_argument("--epochs",     type=int,   default=EPOCHS,           help="Epocas de entrenamiento (default: 5)")
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE,       help="Batch size (default: 2048)")
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE,    help="Learning rate (default: 1e-3)")
    parser.add_argument("--neg-samples",type=int,   default=NEG_SAMPLES,      help="Negativos por positivo (default: 4)")
    parser.add_argument("--topk",       type=int,   default=TOP_K,            help="K para Precision y NDCG (default: 10)")
    parser.add_argument("--max-users",  type=int,   default=MAX_EVAL_USERS,   help="Max usuarios a evaluar (default: 1000)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Carga y filtrado K-Core
    df, user2idx, item2idx = cargar_y_filtrar(args.data_path, k_core=args.k_core)
    n_users = len(user2idx)
    n_items = len(item2idx)

    # 2. Split Leave-One-Out
    train_df, test_df = split_leave_one_out(df)

    # 3. DataLoader de entrenamiento
    dataset = PairwiseDataset(train_df, n_items=n_items, neg_samples=args.neg_samples)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,                          # num_workers=0: compatibilidad Windows
        pin_memory=False,  # pin_memory solo aplica a CUDA; DirectML y CPU no lo usan
    )

    # 4. Modelo y optimizador
    model = NCFLite(n_users=n_users, n_items=n_items, emb_dim=args.emb_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger.info("Backend de computo : %s", _BACKEND)
    logger.info(
        "NCF-Lite | usuarios: %d | items: %d | parametros: %d",
        n_users, n_items, sum(p.numel() for p in model.parameters()),
    )

    # 5. Entrenamiento
    entrenar(model, train_loader, optimizer, epochs=args.epochs)

    # 6. Evaluacion
    resultados = evaluar(
        model, test_df, train_df,
        n_items=n_items,
        k=args.topk,
        max_users=args.max_users,
    )

    # 7. Tabla de resultados — sin guardar archivos
    imprimir_tabla(resultados)


if __name__ == "__main__":
    main()
