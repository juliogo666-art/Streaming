"""
train_modelo_ncf.py
===================
Pipeline oficial de entrenamiento y exportacion a produccion del modelo NCF-Lite.

Flujo completo:
  1. Carga del dataset con filtrado K-Core (proteccion de RAM).
  2. Reindexado de IDs y construccion del PairwiseDataset.
  3. Entrenamiento del modelo NCFLite (GMF + MLP) con BCE pairwise.
  4. Exportacion de artefactos de produccion:
       - ncf_model.onnx   -> grafo de inferencia portable (sin dependencia de PyTorch)
       - user2idx.json    -> mapeo userId_original -> indice embedding
       - item2idx.json    -> mapeo tmdb_id_original -> indice embedding

Restricciones de diseno
-----------------------
  * NO se usar .pkl ni ningún formato de serializacion Python-dependiente.
  * NO se construyen matrices densas N x M.
  * TODO opera por lotes (DataLoader) con embeddings dispersos.

Uso
---
    python src/models/train_modelo_ncf.py
    python src/models/train_modelo_ncf.py --epochs 10 --emb-dim 64 --batch-size 4096
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

# Tracking centralizado de métricas (historial_metricas.csv)
from src.utils.registrar_metricas import registrar_metricas
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Logger
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
    print("[INFO] torch-directml detectado. Usando GPU AMD/Intel via DirectML.")
    _BACKEND: str = f"DirectML (GPU) — {torch_directml.device_name(0)}"
except ImportError:
    DEVICE = torch.device("cpu")
    print("[INFO] torch-directml no disponible. Usando CPU.")
    _BACKEND = "CPU (torch-directml no disponible)"

# ---------------------------------------------------------------------------
# Rutas y constantes por defecto
# ---------------------------------------------------------------------------
_MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH: str = os.path.join(_MODELS_DIR, "..", "data", "ready", "ratings_finales_ia.csv")
# Los artefactos entrenados se guardan centralizados en artifacts/
ONNX_PATH: str = "artifacts/exports/nil_ncf_model.onnx"
USER2IDX_PATH: str = "artifacts/mappings/nil_ncf_user2idx.json"
ITEM2IDX_PATH: str = "artifacts/mappings/nil_ncf_item2idx.json"

# Hiperparametros por defecto
MIN_USER_RATINGS: int = 200   # Umbral K-Core para usuarios
MIN_ITEM_RATINGS: int = 100   # Umbral K-Core para items
EMB_DIM: int = 32             # Dimension de embedding
BATCH_SIZE: int = 2048
EPOCHS: int = 5
NEG_SAMPLES: int = 4          # Negativos por positivo en el Dataset
LEARNING_RATE: float = 1e-3


# ===========================================================================
# SECCION 1 - CARGA Y FILTRADO K-CORE
# ===========================================================================

def cargar_y_submuestrar(
    ruta: str,
    min_user_ratings: int = MIN_USER_RATINGS,
    min_item_ratings: int = MIN_ITEM_RATINGS,
) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    """
    Lee el CSV de ratings y aplica filtrado K-Core iterativo hasta convergencia.
    Retorna el DataFrame filtrado y reindexado junto con los mapeos de IDs
    originales a indices de embedding.

    El filtrado K-Core garantiza que solo entren en produccion usuarios e items
    con suficiente historial, reduciendo ruido y consumo de RAM.

    Parameters
    ----------
    ruta : str
        Ruta al CSV con columnas (userId, tmdb_id, rating).
    min_user_ratings : int
        Numero minimo de valoraciones para conservar un usuario.
    min_item_ratings : int
        Numero minimo de valoraciones para conservar un item.

    Returns
    -------
    df : pd.DataFrame
        Dataset filtrado con IDs reindexados a enteros contiguos [0, N).
    user2idx : Dict[int, int]
        Mapeo userId_original -> indice de embedding.
    item2idx : Dict[int, int]
        Mapeo tmdb_id_original -> indice de embedding.
    """
    logger.info("Cargando dataset desde: %s", ruta)
    dtype_map: Dict[str, str] = {
        "userId": "int32",
        "tmdb_id": "int32",
        "rating": "float32",
    }
    df = pd.read_csv(ruta, dtype=dtype_map)
    logger.info("  Filas originales: %d", len(df))

    # Filtrado iterativo K-Core hasta que el tamano del DataFrame converja
    prev_len = -1
    iteration = 0
    while len(df) != prev_len:
        prev_len = len(df)
        iteration += 1

        user_counts = df["userId"].value_counts()
        df = df[df["userId"].isin(user_counts[user_counts >= min_user_ratings].index)]

        item_counts = df["tmdb_id"].value_counts()
        df = df[df["tmdb_id"].isin(item_counts[item_counts >= min_item_ratings].index)]

    logger.info(
        "  Sub-muestra K-Core: %d filas | %d usuarios | %d items (%d iteraciones)",
        len(df), df["userId"].nunique(), df["tmdb_id"].nunique(), iteration,
    )

    # Construir mapeos: ID_original -> indice contiguo [0, N)
    # Se ordenan para garantizar determinismo entre ejecuciones
    user2idx: Dict[int, int] = {
        int(u): i for i, u in enumerate(sorted(df["userId"].unique()))
    }
    item2idx: Dict[int, int] = {
        int(it): i for i, it in enumerate(sorted(df["tmdb_id"].unique()))
    }

    # Aplicar reindexado al DataFrame
    df["userId"] = df["userId"].map(user2idx).astype("int32")
    df["tmdb_id"] = df["tmdb_id"].map(item2idx).astype("int32")

    return df.reset_index(drop=True), user2idx, item2idx


# ===========================================================================
# SECCION 2 - DATASET PAIRWISE (BCE sobre pares positivo / negativo)
# ===========================================================================

class PairwiseDataset(Dataset):
    """
    Por cada interaccion positiva (u, i+) muestrea `neg_samples` items
    negativos aleatorios que el usuario no haya valorado.

    No se materializa ninguna matriz densa; solo se guarda la lista de
    interacciones y el conjunto de positivos por usuario para el rechazo.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        n_items: int,
        neg_samples: int = NEG_SAMPLES,
    ) -> None:
        self.n_items = n_items
        self.neg_samples = neg_samples

        # Conjunto de items positivos por usuario (para muestreo de negativos)
        self.user_positives: Dict[int, set] = (
            df.groupby("userId")["tmdb_id"].apply(set).to_dict()
        )
        # Array de interacciones (u, i_pos) como numpy para acceso rapido
        self.interactions: np.ndarray = df[["userId", "tmdb_id"]].values.astype(
            np.int64
        )

    def __len__(self) -> int:
        return len(self.interactions) * self.neg_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_idx = idx // self.neg_samples
        u, i_pos = self.interactions[pos_idx]
        positives = self.user_positives[u]

        # Muestreo negativo por rechazo simple
        i_neg = np.random.randint(self.n_items)
        while i_neg in positives:
            i_neg = np.random.randint(self.n_items)

        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(i_pos, dtype=torch.long),
            torch.tensor(i_neg, dtype=torch.long),
        )


# ===========================================================================
# SECCION 3 - MODELO NCF-LITE (GMF + MLP)
# ===========================================================================

class NCFLite(nn.Module):
    """
    Neural Collaborative Filtering con dos ramas fusionadas:

      - GMF (Generalized Matrix Factorization): producto elemento a elemento
        de los embeddings de usuario e item. Captura interacciones lineales.

      - MLP: concatena embeddings y los pasa por capas densas con ReLU.
        Captura interacciones no lineales de orden superior.

    La capa de salida combina ambas ramas en un logit interpretado como
    probabilidad de interaccion positiva (entrenado con BCE).

    Referencia: He et al., 2017 - "Neural Collaborative Filtering."
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
            # Arquitectura piramidal por defecto:
            # entrada: 2*emb_dim -> compresion a emb_dim -> emb_dim//2
            mlp_layers = [emb_dim * 2, emb_dim, emb_dim // 2]

        # Embeddings independientes para cada rama (mejora capacidad del modelo)
        self.gmf_user = nn.Embedding(n_users, emb_dim)
        self.gmf_item = nn.Embedding(n_items, emb_dim)
        self.mlp_user = nn.Embedding(n_users, emb_dim)
        self.mlp_item = nn.Embedding(n_items, emb_dim)

        # Construccion dinamica del bloque MLP
        mlp_modules: List[nn.Module] = []
        in_size = emb_dim * 2  # concatenacion de user_emb + item_emb
        for out_size in mlp_layers:
            mlp_modules += [nn.Linear(in_size, out_size), nn.ReLU()]
            in_size = out_size
        self.mlp = nn.Sequential(*mlp_modules)

        # Capa de fusion: [gmf_out || mlp_out] -> logit escalar
        self.output_layer = nn.Linear(emb_dim + mlp_layers[-1], 1)

        # Inicializacion de pesos pequeños para estabilidad del entrenamiento
        for emb in (self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item):
            nn.init.normal_(emb.weight, std=0.01)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Calcula el logit de relevancia para pares (usuario, item).

        Parameters
        ----------
        users : torch.Tensor of shape [B]
        items : torch.Tensor of shape [B]

        Returns
        -------
        logits : torch.Tensor of shape [B]
        """
        # Rama GMF: producto elemento a elemento
        gmf_out = self.gmf_user(users) * self.gmf_item(items)   # [B, D]

        # Rama MLP: concatenacion + capas densas
        mlp_in = torch.cat([self.mlp_user(users), self.mlp_item(items)], dim=1)  # [B, 2D]
        mlp_out = self.mlp(mlp_in)                               # [B, H]

        # Fusion y proyeccion final
        fused = torch.cat([gmf_out, mlp_out], dim=1)             # [B, D+H]
        return self.output_layer(fused).squeeze(1)               # [B]

    def forward(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BCE Loss sobre un batch de tripletas (u, i+, i-).

        Etiqueta 1 para el item positivo, 0 para el negativo.
        """
        pos_logits = self.score(users, pos_items)
        neg_logits = self.score(users, neg_items)

        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([
            torch.ones_like(pos_logits),
            torch.zeros_like(neg_logits),
        ])
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)

    def predict(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """
        Interfaz de inferencia para ONNX: acepta tensores de userId e itemId
        y devuelve los logits de relevancia.

        Esta firma es la que se traza para exportar el grafo ONNX.
        """
        return self.score(users, items)


# ===========================================================================
# SECCION 4 - BUCLE DE ENTRENAMIENTO
# ===========================================================================

def entrenar(
    model: NCFLite,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    epochs: int,
) -> None:
    """
    Ejecuta el bucle de entrenamiento estandar con gradient clipping.

    Parameters
    ----------
    model : NCFLite
    train_loader : DataLoader
        Loader que emite batches de tripletas (users, pos_items, neg_items).
    optimizer : optim.Optimizer
    epochs : int
    """
    model.train()
    logger.info("Iniciando entrenamiento por %d epocas …", epochs)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        t0 = time.perf_counter()

        for batch in train_loader:
            users, pos_items, neg_items = [t.to(DEVICE) for t in batch]
            optimizer.zero_grad()
            loss = model(users, pos_items, neg_items)
            loss.backward()
            # Gradient clipping para evitar explosiones del gradiente
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        elapsed = time.perf_counter() - t0
        logger.info(
            "  Epoca %d/%d | Loss: %.5f | Tiempo: %.1fs",
            epoch, epochs, epoch_loss / len(train_loader), elapsed,
        )

    logger.info("Entrenamiento completado.")


# ===========================================================================
# SECCION 5 - EXPORTACION A PRODUCCION
# ===========================================================================

def exportar_onnx(
    model: NCFLite,
    onnx_path: str,
) -> None:
    """
    Exporta el modelo a formato ONNX utilizando torch.onnx.export.

    Problema que resuelve este wrapper:
      model.forward() acepta (users, pos_items, neg_items) y devuelve una loss
      (interfaz de entrenamiento), pero para ONNX necesitamos trazar
      model.predict(users, items) que devuelve scores de inferencia.
      torch.onnx.export llama a model.__call__ con los dummy args, por lo que
      es OBLIGATORIO envolver predict() en un modulo con forward() de 2 args.

    Ademas, torch.onnx.export requiere tensores en CPU; si el modelo esta en
    DirectML se copian los pesos a CPU, se exporta, y se devuelve al dispositivo
    original.

    El grafo exportado tiene:
      - Entradas:  user_ids (int64, [batch_size])
                   item_ids (int64, [batch_size])
      - Salida:    scores   (float32, [batch_size])

    Parameters
    ----------
    model : NCFLite
        Modelo entrenado listo para exportar.
    onnx_path : str
        Ruta destino del archivo .onnx.
    """

    # --- Wrapper de inferencia: expone predict() como forward() ---
    # Necesario porque torch.onnx.export traza model.forward(), no model.predict()
    class _InferenceWrapper(nn.Module):
        def __init__(self, m: NCFLite) -> None:
            super().__init__()
            self.m = m

        def forward(
            self, users: torch.Tensor, items: torch.Tensor
        ) -> torch.Tensor:
            return self.m.predict(users, items)

    # --- ONNX export requiere CPU; mover si el modelo esta en otro dispositivo ---
    original_device = next(model.parameters()).device
    cpu_model = model.to(torch.device("cpu"))
    cpu_model.eval()

    wrapper = _InferenceWrapper(cpu_model)
    wrapper.eval()

    # Tensores dummy en CPU (batch_size=1) para trazar el grafo
    dummy_users = torch.zeros(1, dtype=torch.long)
    dummy_items = torch.zeros(1, dtype=torch.long)

    logger.info("Exportando modelo a ONNX: %s", onnx_path)
    torch.onnx.export(
        wrapper,
        args=(dummy_users, dummy_items),
        f=onnx_path,
        input_names=["user_ids", "item_ids"],
        output_names=["scores"],
        dynamic_axes={
            # Permite batch variable en inferencia sin re-exportar
            "user_ids": {0: "batch_size"},
            "item_ids": {0: "batch_size"},
            "scores":   {0: "batch_size"},
        },
        opset_version=17,
        export_params=True,        # Pesos incluidos en el archivo ONNX
        do_constant_folding=True,  # Optimizacion de constantes en el grafo
    )

    # Devolver el modelo al dispositivo original para cualquier uso posterior
    model.to(original_device)
    logger.info("  Modelo ONNX guardado correctamente.")


def exportar_mappings(
    user2idx: Dict[int, int],
    item2idx: Dict[int, int],
    user2idx_path: str,
    item2idx_path: str,
) -> None:
    """
    Guarda los diccionarios de mapeo de IDs en formato JSON puro.

    JSON es el formato de interoperabilidad por defecto: legible por humanos,
    parseado de forma nativa por cualquier lenguaje de programacion y sin
    riesgo de ejecucion de codigo arbitrario (a diferencia de .pkl).

    Parameters
    ----------
    user2idx : Dict[int, int]
        Mapeo userId_original -> indice de embedding.
    item2idx : Dict[int, int]
        Mapeo tmdb_id_original -> indice de embedding.
    user2idx_path : str
        Ruta destino del archivo user2idx.json.
    item2idx_path : str
        Ruta destino del archivo item2idx.json.
    """
    logger.info("Guardando mapeos de IDs …")

    with open(user2idx_path, "w", encoding="utf-8") as f:
        # Las claves de JSON deben ser strings; convertimos los ints
        json.dump({str(k): v for k, v in user2idx.items()}, f)
    logger.info("  user2idx.json guardado: %d entradas", len(user2idx))

    with open(item2idx_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in item2idx.items()}, f)
    logger.info("  item2idx.json guardado: %d entradas", len(item2idx))


# ===========================================================================
# SECCION 6 - PUNTO DE ENTRADA
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena NCF-Lite y exporta artefactos de produccion (ONNX + JSON)."
    )
    parser.add_argument(
        "--data-path", type=str, default=DATA_PATH,
        help="Ruta al CSV de ratings (userId, tmdb_id, rating).",
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Numero de epocas de entrenamiento (por defecto: {EPOCHS}).",
    )
    parser.add_argument(
        "--emb-dim", type=int, default=EMB_DIM,
        help=f"Dimension de los embeddings (por defecto: {EMB_DIM}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"Tamano del batch (por defecto: {BATCH_SIZE}).",
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help=f"Learning rate del optimizador Adam (por defecto: {LEARNING_RATE}).",
    )
    parser.add_argument(
        "--neg-samples", type=int, default=NEG_SAMPLES,
        help=f"Negativos por positivo en el Dataset (por defecto: {NEG_SAMPLES}).",
    )
    parser.add_argument(
        "--min-user-ratings", type=int, default=MIN_USER_RATINGS,
        help=f"Umbral K-Core usuarios (por defecto: {MIN_USER_RATINGS}).",
    )
    parser.add_argument(
        "--min-item-ratings", type=int, default=MIN_ITEM_RATINGS,
        help=f"Umbral K-Core items (por defecto: {MIN_ITEM_RATINGS}).",
    )
    parser.add_argument(
        "--onnx-path", type=str, default=ONNX_PATH,
        help="Ruta de salida del archivo .onnx.",
    )
    parser.add_argument(
        "--user2idx-path", type=str, default=USER2IDX_PATH,
        help="Ruta de salida del archivo user2idx.json.",
    )
    parser.add_argument(
        "--item2idx-path", type=str, default=ITEM2IDX_PATH,
        help="Ruta de salida del archivo item2idx.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Carga y filtrado K-Core
    # ------------------------------------------------------------------
    df, user2idx, item2idx = cargar_y_submuestrar(
        ruta=args.data_path,
        min_user_ratings=args.min_user_ratings,
        min_item_ratings=args.min_item_ratings,
    )
    n_users: int = len(user2idx)
    n_items: int = len(item2idx)
    logger.info("Vocabulario: %d usuarios | %d items", n_users, n_items)

    # ------------------------------------------------------------------
    # 2. Dataset y DataLoader
    # ------------------------------------------------------------------
    dataset = PairwiseDataset(df, n_items=n_items, neg_samples=args.neg_samples)
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers=0 para maxima compatibilidad en Windows
        num_workers=0,
        pin_memory=False,  # pin_memory solo aplica a CUDA; DirectML y CPU no lo usan
    )
    logger.info(
        "Dataset: %d interacciones x %d negativos = %d muestras de entrenamiento",
        len(df), args.neg_samples, len(dataset),
    )

    # ------------------------------------------------------------------
    # 3. Modelo y optimizador
    # ------------------------------------------------------------------
    model = NCFLite(
        n_users=n_users,
        n_items=n_items,
        emb_dim=args.emb_dim,
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Backend de computo : %s", _BACKEND)
    logger.info("NCF-Lite instanciado: %d parametros entrenables", total_params)

    # ------------------------------------------------------------------
    # 4. Entrenamiento
    # ------------------------------------------------------------------
    t_start = time.perf_counter()
    entrenar(model, train_loader, optimizer, epochs=args.epochs)
    t_total = time.perf_counter() - t_start
    logger.info("Tiempo total de entrenamiento: %.1f s", t_total)

    # ------------------------------------------------------------------
    # 5. Exportacion de artefactos de produccion
    # ------------------------------------------------------------------
    # 5a. Modelo en formato ONNX (inferencia portable, sin PyTorch en prod)
    exportar_onnx(model, onnx_path=args.onnx_path)

    # 5b. Mapeos de IDs en JSON puro (seguro, interoperable, sin .pkl)
    exportar_mappings(
        user2idx=user2idx,
        item2idx=item2idx,
        user2idx_path=args.user2idx_path,
        item2idx_path=args.item2idx_path,
    )

    logger.info(
        "Pipeline completado. Artefactos guardados en:\n"
        "  ONNX  : %s\n"
        "  users : %s\n"
        "  items : %s",
        args.onnx_path,
        args.user2idx_path,
        args.item2idx_path,
    )

    # ------------------------------------------------------------------
    # 6. Registro centralizado de métricas en historial_metricas.csv
    # ------------------------------------------------------------------
    registrar_metricas(
        modelo="NCF-Lite (nil)",
        hiperparams={
            "emb_dim": args.emb_dim,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "min_user_ratings": args.min_user_ratings,
            "min_item_ratings": args.min_item_ratings,
        },
        metricas={},  # Las métricas de ranking se calculan aparte en evaluación
        dataset_size=len(train_loader.dataset),
        train_time_s=round(t_total, 1),
        notas="Entrenamiento NCF-Lite (pipeline nil)",
    )


if __name__ == "__main__":
    main()
