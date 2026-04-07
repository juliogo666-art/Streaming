"""
##############################################################################################
#
#  MODELO 6: NCF (Neural Collaborative Filtering)
#  ===============================================
#  Arquitectura de Deep Learning que fusiona dos ramas complementarias:
#
#  1. GMF (Generalized Matrix Factorization):
#     - Producto elemento a elemento de embeddings de usuario e item.
#     - Captura interacciones lineales directas (similar a SVD pero más flexible).
#
#  2. MLP (Multi-Layer Perceptron):
#     - Concatena embeddings y los pasa por capas densas con ReLU.
#     - Captura interacciones NO lineales de orden superior.
#
#  La capa de salida combina ambas ramas en un logit que indica la probabilidad
#  de que al usuario le "guste" el item (entrenado con BCE sobre pares +/-).
#
#  Referencia: He et al., 2017 – "Neural Collaborative Filtering"
#
#  Exportación: el modelo se exporta a ONNX para inferencia sin dependencia de PyTorch.
#
##############################################################################################
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import json
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Añadir el directorio raíz al PATH para importar utilidades
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
from src.utils.registrar_metricas import registrar_metricas

##############################################################################################
#  CONFIGURACIÓN GLOBAL
##############################################################################################

# Rutas de archivos
RUTA_RATINGS = "src/data/ready/ratings_finales_ia.csv"

# Rutas de salida (artefactos de producción)
RUTA_MODELO_ONNX = "src/models/jj/modelo_6_ncf.onnx"
RUTA_USER2IDX = "src/models/jj/ncf_user2idx.json"
RUTA_ITEM2IDX = "src/models/jj/ncf_item2idx.json"

# Hiperparámetros
EMB_DIM = 32  # Dimensión de los embeddings
BATCH_SIZE = 2048  # Tamaño de lote
EPOCHS = 5  # Épocas de entrenamiento
NEG_SAMPLES = 4  # Negativos por cada positivo (pairwise learning)
LEARNING_RATE = 1e-3  # Learning rate de Adam
MIN_RATINGS_USER = 200  # Umbral K-Core para usuarios
MIN_RATINGS_ITEM = 100  # Umbral K-Core para items

# Dispositivo de cómputo
try:
    import torch_directml  # type: ignore

    DEVICE = torch_directml.device()
    print("[NCF] torch-directml detectado. Usando GPU AMD/Intel via DirectML.")
    _BACKEND = f"DirectML (GPU) — {torch_directml.device_name(0)}"
except ImportError:
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        _BACKEND = f"CUDA (GPU) — {torch.cuda.get_device_name(0)}"
    else:
        DEVICE = torch.device("cpu")
        _BACKEND = "CPU"

print(f"[NCF] Backend de cómputo: {_BACKEND}")


##############################################################################################
#  PASO 1: CARGA Y FILTRADO K-CORE
##############################################################################################


def cargar_y_filtrar(ruta: str) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    """
    Lee el CSV de ratings y aplica filtrado K-Core iterativo hasta convergencia.
    Retorna el DataFrame filtrado + reindexado junto con los mapeos de IDs.
    """
    print("=" * 70)
    print("  MODELO 6: NCF — Carga y Filtrado K-Core")
    print("=" * 70)

    print(f"\n  Leyendo {ruta}...")
    df = pd.read_csv(
        ruta, dtype={"userId": "int32", "tmdb_id": "int32", "rating": "float32"}
    )
    print(f"  -> Filas originales: {len(df):,}")

    # Filtrado iterativo K-Core hasta convergencia
    prev_len = -1
    iteration = 0
    while len(df) != prev_len:
        prev_len = len(df)
        iteration += 1
        user_counts = df["userId"].value_counts()
        df = df[df["userId"].isin(user_counts[user_counts >= MIN_RATINGS_USER].index)]
        item_counts = df["tmdb_id"].value_counts()
        df = df[df["tmdb_id"].isin(item_counts[item_counts >= MIN_RATINGS_ITEM].index)]

    print(
        f"  -> Sub-muestra K-Core ({MIN_RATINGS_USER}/{MIN_RATINGS_ITEM}): "
        f"{len(df):,} filas | {df['userId'].nunique():,} usuarios | "
        f"{df['tmdb_id'].nunique():,} items ({iteration} iteraciones)"
    )

    # Construir mapeos: ID_original -> índice contiguo [0, N)
    user2idx = {int(u): i for i, u in enumerate(sorted(df["userId"].unique()))}
    item2idx = {int(it): i for i, it in enumerate(sorted(df["tmdb_id"].unique()))}

    # Aplicar reindexado
    df["userId"] = df["userId"].map(user2idx).astype("int32")
    df["tmdb_id"] = df["tmdb_id"].map(item2idx).astype("int32")

    return df.reset_index(drop=True), user2idx, item2idx


##############################################################################################
#  PASO 2: DATASET PAIRWISE
##############################################################################################


class PairwiseDataset(Dataset):
    """
    Por cada interacción positiva (u, i+) muestrea negativos aleatorios.
    No construye matrices densas — solo guarda listas de positivos por usuario.
    """

    def __init__(self, df: pd.DataFrame, n_items: int, neg_samples: int = NEG_SAMPLES):
        self.n_items = n_items
        self.neg_samples = neg_samples
        self.user_positives = df.groupby("userId")["tmdb_id"].apply(set).to_dict()
        self.interactions = df[["userId", "tmdb_id"]].values.astype(np.int64)

    def __len__(self) -> int:
        return len(self.interactions) * self.neg_samples

    def __getitem__(self, idx: int):
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


##############################################################################################
#  PASO 3: ARQUITECTURA NCF (GMF + MLP)
##############################################################################################


class NCF(nn.Module):
    """
    Neural Collaborative Filtering con dos ramas fusionadas:
      - GMF: producto elemento a elemento de embeddings → captura interacciones lineales.
      - MLP: concatena embeddings + capas densas → captura interacciones no lineales.

    La capa de fusión combina ambas en un logit de relevancia.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = EMB_DIM,
        mlp_layers: List[int] | None = None,
    ):
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [emb_dim * 2, emb_dim, emb_dim // 2]

        # Embeddings independientes para cada rama
        self.gmf_user = nn.Embedding(n_users, emb_dim)
        self.gmf_item = nn.Embedding(n_items, emb_dim)
        self.mlp_user = nn.Embedding(n_users, emb_dim)
        self.mlp_item = nn.Embedding(n_items, emb_dim)

        # Bloque MLP dinámico
        mlp_modules: List[nn.Module] = []
        in_size = emb_dim * 2
        for out_size in mlp_layers:
            mlp_modules += [nn.Linear(in_size, out_size), nn.ReLU(), nn.Dropout(0.1)]
            in_size = out_size
        self.mlp = nn.Sequential(*mlp_modules)

        # Capa de fusión: [gmf_out || mlp_out] -> logit escalar
        self.output_layer = nn.Linear(emb_dim + mlp_layers[-1], 1)

        # Inicialización de pesos pequeños
        for emb in (self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item):
            nn.init.normal_(emb.weight, std=0.01)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Calcula el logit de relevancia para pares (usuario, item)."""
        gmf_out = self.gmf_user(users) * self.gmf_item(items)
        mlp_in = torch.cat([self.mlp_user(users), self.mlp_item(items)], dim=1)
        mlp_out = self.mlp(mlp_in)
        fused = torch.cat([gmf_out, mlp_out], dim=1)
        return self.output_layer(fused).squeeze(1)

    def forward(self, users, pos_items, neg_items):
        """BCE Loss sobre pares positivo/negativo."""
        pos_logits = self.score(users, pos_items)
        neg_logits = self.score(users, neg_items)
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)

    def predict(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Interfaz de inferencia para ONNX (2 args -> scores)."""
        return self.score(users, items)


##############################################################################################
#  PASO 4: ENTRENAMIENTO
##############################################################################################


def entrenar(model, train_loader, optimizer, epochs):
    """Bucle de entrenamiento con gradient clipping."""
    model.train()
    print(f"\n  Iniciando entrenamiento por {epochs} épocas...")

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        t0 = time.perf_counter()

        for batch in train_loader:
            users, pos_items, neg_items = [t.to(DEVICE) for t in batch]
            optimizer.zero_grad()
            loss = model(users, pos_items, neg_items)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        elapsed = time.perf_counter() - t0
        avg_loss = epoch_loss / len(train_loader)
        print(
            f"  Época {epoch}/{epochs} | Loss: {avg_loss:.5f} | Tiempo: {elapsed:.1f}s"
        )

    print("  Entrenamiento completado.\n")


##############################################################################################
#  PASO 5: EXPORTACIÓN A PRODUCCIÓN
##############################################################################################


def exportar_onnx(model):
    """
    Exporta el modelo NCF a formato ONNX.
    Usa un wrapper porque model.forward() acepta 3 args (entrenamiento)
    pero para inferencia necesitamos model.predict() con 2 args.
    """

    class _InferenceWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, users, items):
            return self.m.predict(users, items)

    # ONNX export requiere CPU
    original_device = next(model.parameters()).device
    cpu_model = model.to(torch.device("cpu"))
    cpu_model.eval()

    wrapper = _InferenceWrapper(cpu_model)
    wrapper.eval()

    dummy_users = torch.zeros(1, dtype=torch.long)
    dummy_items = torch.zeros(1, dtype=torch.long)

    print(f"  Exportando modelo a ONNX: {RUTA_MODELO_ONNX}")
    torch.onnx.export(
        wrapper,
        args=(dummy_users, dummy_items),
        f=RUTA_MODELO_ONNX,
        input_names=["user_ids", "item_ids"],
        output_names=["scores"],
        dynamic_axes={
            "user_ids": {0: "batch_size"},
            "item_ids": {0: "batch_size"},
            "scores": {0: "batch_size"},
        },
        opset_version=17,
        export_params=True,
        do_constant_folding=True,
    )

    model.to(original_device)
    tamano_mb = os.path.getsize(RUTA_MODELO_ONNX) / (1024 * 1024)
    print(f"  -> Modelo ONNX guardado: {RUTA_MODELO_ONNX} ({tamano_mb:.1f} MB)")


def exportar_mappings(user2idx, item2idx):
    """Guarda los diccionarios de mapeo de IDs en formato JSON."""
    with open(RUTA_USER2IDX, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in user2idx.items()}, f)
    print(f"  -> user2idx.json guardado: {len(user2idx)} usuarios")

    with open(RUTA_ITEM2IDX, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in item2idx.items()}, f)
    print(f"  -> item2idx.json guardado: {len(item2idx)} items")


##############################################################################################
#  EJECUCIÓN PRINCIPAL
##############################################################################################

if __name__ == "__main__":
    # 1. Carga y filtrado K-Core
    df, user2idx, item2idx = cargar_y_filtrar(RUTA_RATINGS)
    n_users = len(user2idx)
    n_items = len(item2idx)
    print(f"  Vocabulario: {n_users} usuarios | {n_items} items")

    # 2. Dataset y DataLoader
    dataset = PairwiseDataset(df, n_items=n_items, neg_samples=NEG_SAMPLES)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 0 para compatibilidad Windows
        pin_memory=False,
    )
    print(
        f"  Dataset: {len(df):,} interacciones x {NEG_SAMPLES} negativos = {len(dataset):,} muestras"
    )

    # 3. Modelo y optimizador
    model = NCF(n_users=n_users, n_items=n_items, emb_dim=EMB_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  NCF instanciado: {total_params:,} parámetros entrenables")
    print(f"  Backend: {_BACKEND}")

    # 4. Entrenamiento
    t_start = time.perf_counter()
    entrenar(model, train_loader, optimizer, epochs=EPOCHS)
    t_total = time.perf_counter() - t_start
    print(f"  Tiempo total: {t_total:.1f}s")

    # 5. Exportación
    print("\n" + "=" * 70)
    print("  MODELO 6: NCF — Exportación a Producción")
    print("=" * 70)
    exportar_onnx(model)
    exportar_mappings(user2idx, item2idx)

    # 6. Registrar métricas
    registrar_metricas(
        modelo="NCF",
        hiperparams={
            "embedding_dim": EMB_DIM,
            "n_epocas": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "neg_samples": NEG_SAMPLES,
            "min_ratings_user": MIN_RATINGS_USER,
            "min_ratings_item": MIN_RATINGS_ITEM,
        },
        metricas={},
        dataset_size=len(df),
        train_time_s=t_total,
        notas="Modelo de ranking (BCE pairwise). MAE/RMSE no aplican directamente.",
    )

    print("\n  ¡Modelo NCF exportado a ONNX con éxito!")
    print("=" * 70)
