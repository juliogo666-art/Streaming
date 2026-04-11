"""
========================================================================================
 SCRIPT MAESTRO: ENTRENAMIENTO WIDE & DEEP
========================================================================================
 Este archivo importa nuestra arquitectura de 'rn.py' y carga los datos para entrenarla.

 ¿QUÉ ES EL MODELO WIDE & DEEP?
 Es una arquitectura híbrida de Google que combina dos formas de "pensar":

 1. LA PARTE "WIDE" (Ancha / Memoria Directa):
    - Literalmente memoriza qué usuario ha visto qué película mediante Embeddings directos.
    - Se encarga de las "excepciones" y correlaciones específicas ("A Juan le gusta Matrix").
    - Aporta precisión cruda basada en el historial exacto.

 2. LA PARTE "DEEP" (Profunda / Generalización):
    - Pasa los datos por varias capas ocultas (por ej. 64 y 32 neuronas).
    - Permite a la IA descubrir patrones ocultos y generalizar a cosas nuevas.
    - ("Si a Juan le gusta Matrix, a lo mejor le gusta Blade Runner porque comparten rasgos abstractos").

 La red suma los instintos de ambas partes para dar la predicción final.

 Importante:
 - Si se entrena con CPU se requerira de 30 a 40 horas.
 - Si se usa una GPU se reducira significativamente el tiempo de entrenamiento.
 - Nuestro caso usaremos toda la potencia de la GPU (mi caso una RTX 5060) si PyTorch+CUDA está instalado.
 - Se recomienda usar un entorno virtual con las dependencias instaladas.
========================================================================================
"""

import pandas as pd
import numpy as np
import time
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Importamos nuestra Red Neuronal local

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
try:
    from src.networks.dl.rn_mlp import WideAndDeepModel
except ImportError:
    from networks.dl.rn_mlp import WideAndDeepModel
from src.utils.registrar_metricas import registrar_metricas

# -----------------------------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL
# -----------------------------------------------------------------------------------------
ruta_ratings = "src/data/ready/ratings_finales_ia.csv"
ruta_modelo = "artifacts/checkpoints/modelo_3_wnd.pth"  # Checkpoint PyTorch
ruta_mapeos = "artifacts/mappings/wnd_mappings.pkl"  # Mapeo de IDs reales <-> internos

# Parámetros del Deep Learning (Hiperparámetros)
BATCH_SIZE = (
    4096  # Cuántas valoraciones procesamos de golpe. Las GPU aman los lotes grandes.
)
EPOCHS = 10
LEARNING_RATE = 0.001

# Hacemos recomendaciones al 100% de tus usuarios.
# Hacemos recomendaciones al 100% de tus usuarios.
MIN_RATINGS_USUARIO = 1000
# Con 1000 valoraciones por peli sincronizamos con el evaluador
MIN_RATINGS_PELICULA = 1000

# Relación de muestreo negativo (cuántos "no-vistos" por cada "visto")
NEG_SAMPLES_PER_POS = 4


# -----------------------------------------------------------------------------------------
# PASO 1: DATASET PERSONALIZADO DE PYTORCH CON NEGATIVE SAMPLING (OPTIMIZADO)
# -----------------------------------------------------------------------------------------
class RankingDataset(Dataset):
    """
    Dataset para Entrenamiento de Ranking (0/1).
    Usa pre-asignación de arrays de Numpy para manejar millones de filas sin latencia de .append().
    """

    def __init__(self, pos_users, pos_movies, n_items, neg_sample_ratio):
        n_pos = len(pos_users)
        n_total = n_pos * (1 + neg_sample_ratio)

        print(f"  Preparando {n_total:,} ejemplos para entrenamiento...")

        # Pre-asignamos memoria en Numpy (Mucho más rápido que .append() en listas de Python)
        self.users_np = np.zeros(n_total, dtype=np.int64)
        self.items_np = np.zeros(n_total, dtype=np.int64)
        self.labels_np = np.zeros(n_total, dtype=np.float32)

        # Set de pelis por usuario para muestreo rápido
        from collections import defaultdict

        user_watched = defaultdict(set)
        for u, m in zip(pos_users, pos_movies):
            user_watched[u].add(m)

        print(f"  Generando {neg_sample_ratio} negativos por cada positivo...")

        # 1. Rellenar Positivos
        self.users_np[:n_pos] = pos_users
        self.items_np[:n_pos] = pos_movies
        self.labels_np[:n_pos] = 1.0

        # 2. Rellenar Negativos
        import random

        curr_idx = n_pos
        num_items_total = len(n_items) if isinstance(n_items, list) else n_items
        for u in pos_users:
            if len(user_watched[u]) >= num_items_total * 0.9:
                continue
            for _ in range(neg_sample_ratio):
                neg_m = random.randrange(num_items_total)
                while neg_m in user_watched[u]:
                    neg_m = random.randrange(num_items_total)
                self.users_np[curr_idx] = u
                self.items_np[curr_idx] = neg_m
                self.labels_np[curr_idx] = 0.0
                curr_idx += 1

        self.users = torch.from_numpy(self.users_np)
        self.movies = torch.from_numpy(self.items_np)
        self.labels = torch.from_numpy(self.labels_np)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.labels[idx]


def cargar_y_preparar_datos():
    print("=" * 70)
    print("  MODELO 3: WIDE & DEEP (PyTorch) — Preparando Datos de Ranking")
    print("=" * 70)

    print(f"\n  Leyendo {ruta_ratings}...")
    df = pd.read_csv(ruta_ratings, on_bad_lines="skip")
    print(f"  -> Filas en bruto: {len(df):,}")

    # --- FILTRADO más estricto para calidad y manejo de memoria ---
    print(
        f"\n  Filtrando (usuario>={MIN_RATINGS_USUARIO}, pelicula>={MIN_RATINGS_PELICULA})..."
    )
    conteo_u = df.groupby("userId").size()
    df = df[df["userId"].isin(conteo_u[conteo_u >= MIN_RATINGS_USUARIO].index)]
    conteo_m = df.groupby("tmdb_id").size()
    df = df[df["tmdb_id"].isin(conteo_m[conteo_m >= MIN_RATINGS_PELICULA].index)]
    conteo_u = df.groupby("userId").size()
    df = df[df["userId"].isin(conteo_u[conteo_u >= MIN_RATINGS_USUARIO].index)]
    conteo_m = df.groupby("tmdb_id").size()
    df = df[df["tmdb_id"].isin(conteo_m[conteo_m >= MIN_RATINGS_PELICULA].index)]
    print(f"  -> Filas tras filtro: {len(df):,}")

    user_ids = df["userId"].unique()
    movie_ids = df["tmdb_id"].unique()

    user2idx = {o: i for i, o in enumerate(user_ids)}
    movie2idx = {o: i for i, o in enumerate(movie_ids)}

    df["user_idx"] = df["userId"].map(user2idx)
    df["movie_idx"] = df["tmdb_id"].map(movie2idx)

    num_users = len(user_ids)
    num_movies = len(movie_ids)

    print(f"  -> Usuarios unicos: {num_users:,}")
    print(f"  -> Peliculas unicas: {num_movies:,}")

    # Dividir en Entrenamiento (80%) y Test (20%)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    # Guardar mapeos
    with open(ruta_mapeos, "wb") as f:
        pickle.dump({"user2idx": user2idx, "movie2idx": movie2idx}, f)

    return df_train, df_test, num_users, num_movies, list(range(num_movies))


# -----------------------------------------------------------------------------------------
# PASO 2: ENTRENAMIENTO EN LA GRÁFICA (GPU)
# -----------------------------------------------------------------------------------------
def entrenar_modelo(df_train, df_test, num_users, num_movies, all_movie_indices):
    print("=" * 70)
    print("  MODELO 3: ENTRENAMIENTO W&D — MODALIDAD RANKING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  -> Usando: {device}")

    model = WideAndDeepModel(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=64,  # Aumentamos capacidad
        hidden_layers=[128, 64, 32],
    ).to(device)

    # Creamos Datasets con Negativos
    print("\n  Preparando Dataset de Entrenamiento (Positivos y Negativos)...")
    train_dataset = RankingDataset(
        df_train["user_idx"].values,
        df_train["movie_idx"].values,
        all_movie_indices,
        NEG_SAMPLES_PER_POS,
    )
    test_dataset = RankingDataset(
        df_test["user_idx"].values,
        df_test["movie_idx"].values,
        all_movie_indices,
        1,  # Menos negativos en test para velocidad
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Usamos Binary Cross Entropy with Logits (ideal para Ranking 0/1)
    criterio = nn.BCEWithLogitsLoss()
    optimizador = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        inicio_epoca = time.time()

        for i, (users, movies, labels) in enumerate(train_loader):
            users = users.to(device)
            movies = movies.to(device)
            labels = labels.to(device)

            # Forward
            logits = model(users, movies)
            loss = criterio(logits, labels)

            # Backward
            optimizador.zero_grad()
            loss.backward()
            optimizador.step()

            total_loss += loss.item() * len(labels)

        avg_train_loss = total_loss / len(train_dataset)
        print(
            f"  Época {epoch + 1:02d}/{EPOCHS} | Loss: {avg_train_loss:.4f} | {time.time() - inicio_epoca:.1f}s"
        )

    # Evaluación simple de Accuracy en Ranking
    model.eval()
    hits = 0
    with torch.no_grad():
        for users, movies, labels in test_loader:
            users, movies, labels = (
                users.to(device),
                movies.to(device),
                labels.to(device),
            )
            logits = model(users, movies)
            preds = (torch.sigmoid(logits) > 0.5).float()
            hits += (preds == labels).sum().item()

    acc = hits / len(test_dataset)
    print(f"\n  Ranking Accuracy en Test: {acc * 100:.2f}%")

    # Guardado .pth
    torch.save(model.state_dict(), ruta_modelo)

    # EXPORTACIÓN A ONNX (Crucial para la API)
    ruta_onnx = ruta_modelo.replace(".pth", ".onnx")
    print(f"\n  Exportando a ONNX en {ruta_onnx}...")
    model.cpu()
    dummy_u = torch.zeros(1, dtype=torch.long)
    dummy_m = torch.zeros(1, dtype=torch.long)
    torch.onnx.export(
        model,
        (dummy_u, dummy_m),
        ruta_onnx,
        input_names=["user_ids", "item_ids"],
        output_names=["output"],
        dynamic_axes={"user_ids": {0: "batch_size"}, "item_ids": {0: "batch_size"}},
    )

    registrar_metricas(
        modelo="Wide&Deep-Ranking",
        hiperparams={"epochs": EPOCHS, "neg_ratio": NEG_SAMPLES_PER_POS, "emb_dim": 64},
        metricas={"Accuracy": acc},
        dataset_size=len(train_dataset),
    )


if __name__ == "__main__":
    df_train, df_test, num_u, num_m, all_m = cargar_y_preparar_datos()
    entrenar_modelo(df_train, df_test, num_u, num_m, all_m)
