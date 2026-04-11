import pandas as pd
import numpy as np
import time
import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.networks.dl.two_towers_net import TwoTowersModel
from src.utils.registrar_metricas import registrar_metricas

# -----------------------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------------------
ruta_ratings = "src/data/ready/ratings_finales_ia.csv"
ruta_modelo = "artifacts/checkpoints/modelo_7_twotowers.pth"
ruta_mapeos = "artifacts/mappings/twotowers_mappings.pkl"

BATCH_SIZE = 2048
EPOCHS = 5
LEARNING_RATE = 0.001
EMBEDDING_DIM = 64

# Umbrales de filtrado. Originalmente 1000/1000, rebajados para aumentar cobertura.
# Si quieres reentrenar con otros valores, cambia aquí. El sufijo se añade automáticamente.
MIN_RATINGS_USUARIO = 100    # Antes: 1000 → Solo 2.5% de users
MIN_RATINGS_PELICULA = 50    # Antes: 1000 → Pocas pelis


# -----------------------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------------------
class InteractionDataset(Dataset):
    def __init__(self, users, items):
        self.users = torch.tensor(users, dtype=torch.long)
        self.items = torch.tensor(items, dtype=torch.long)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx]


def cargar_datos():
    print("=" * 70)
    print("  MODELO 7: TWO-TOWERS (Bi-Encoder) — Preparación")
    print("=" * 70)

    # on_bad_lines='skip' evita el fallo si el CSV tiene alguna fila corrupta
    df = pd.read_csv(ruta_ratings, on_bad_lines='skip')
    
    # Filtrado estricto para calidad y manejo de memoria
    print(f"  Filtrando (U>={MIN_RATINGS_USUARIO}, P>={MIN_RATINGS_PELICULA})...")
    conteo_u = df.groupby("userId").size()
    df = df[df["userId"].isin(conteo_u[conteo_u >= MIN_RATINGS_USUARIO].index)]
    conteo_m = df.groupby("tmdb_id").size()
    df = df[df["tmdb_id"].isin(conteo_m[conteo_m >= MIN_RATINGS_PELICULA].index)]
    
    user_ids = df["userId"].unique()
    item_ids = df["tmdb_id"].unique()
    
    user2idx = {o: i for i, o in enumerate(user_ids)}
    item2idx = {o: i for i, o in enumerate(item_ids)}
    
    df["u_idx"] = df["userId"].map(user2idx)
    df["i_idx"] = df["tmdb_id"].map(item2idx)
    
    with open(ruta_mapeos, "wb") as f:
        pickle.dump({"user2idx": user2idx, "item2idx": item2idx}, f)
        
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
    
    return df_train, df_test, len(user_ids), len(item_ids)


# -----------------------------------------------------------------------------------------
# ENTRENAMIENTO CON IN-BATCH NEGATIVES
# -----------------------------------------------------------------------------------------
def train():
    df_train, df_test, n_users, n_items = cargar_datos()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  -> Usuarios: {n_users}, Items: {n_items} | Device: {device}")
    
    model = TwoTowersModel(n_users, n_items, EMBEDDING_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # DataLoader
    train_ds = InteractionDataset(df_train["u_idx"].values, df_train["i_idx"].values)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    print("\n  Iniciando entrenamiento (In-Batch Negatives)...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for u, i in train_loader:
            u, i = u.to(device), i.to(device)
            
            # Forward: Obtenemos los vectores de salida de las torres
            user_vectors = model.user_tower(u) # [B, D]
            item_vectors = model.item_tower(i) # [B, D]
            
            # Matriz de afinidad: (B, D) @ (D, B) -> (B, B)
            # scores[j, k] es la afinidad del usuario J con el item K del batch
            scores = torch.matmul(user_vectors, item_vectors.t())
            
            # Queremos que la diagonal (u_j con su propio i_j) sea máxima
            # La CrossEntropy contra un vector [0, 1, 2, ..., B-1] hace esto.
            target = torch.arange(u.size(0)).to(device)
            
            loss = nn.CrossEntropyLoss()(scores, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"  Época {epoch+1:02d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")
        
    print("\n  Entrenamiento completado.")

    # Guardado de artefactos con sufijo del umbral para no pisar modelos anteriores.
    # Ejemplo: si MIN_RATINGS_USUARIO=100, genera modelo_7_twotowers_r100.pth
    sufijo = f"_r{MIN_RATINGS_USUARIO}" if MIN_RATINGS_USUARIO != 1000 else ""
    ruta_modelo_final = ruta_modelo.replace(".pth", f"{sufijo}.pth")
    ruta_mapeos_final = ruta_mapeos.replace(".pkl", f"{sufijo}.pkl")

    torch.save(model.state_dict(), ruta_modelo_final)
    print(f"  Modelo guardado en: {ruta_modelo_final}")

    # Guardar mappings (con sufijo)
    with open(ruta_mapeos_final, "wb") as f:
        pickle.dump({"user2idx": user2idx, "item2idx": item2idx}, f)

    # Exportar a ONNX
    ruta_onnx = ruta_modelo_final.replace(".pth", ".onnx")
    ruta_onnx = ruta_onnx.replace("checkpoints", "exports")
    print(f"  Exportando a ONNX: {ruta_onnx}")
    model.cpu().eval()

    dummy_u = torch.zeros(1, dtype=torch.long)
    dummy_i = torch.zeros(1, dtype=torch.long)

    torch.onnx.export(
        model,
        (dummy_u, dummy_i),
        ruta_onnx,
        input_names=["user_ids", "item_ids"],
        output_names=["similarity"],
        dynamic_axes={"user_ids": {0: "batch_size"}, "item_ids": {0: "batch_size"}}
    )

    registrar_metricas(
        modelo=f"TwoTowers{sufijo}",
        hiperparams={
            "batch": BATCH_SIZE,
            "emb": EMBEDDING_DIM,
            "min_ratings_user": MIN_RATINGS_USUARIO,
            "min_ratings_item": MIN_RATINGS_PELICULA,
        },
        metricas={"Loss": total_loss/len(train_loader)},
        dataset_size=len(df_train)
    )

if __name__ == "__main__":
    train()
