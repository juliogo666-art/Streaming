##############################################################################################
#
#  MODELO SVD + COSINE SCORE
#  =========================
#  Basado en modelo_1_SVD.py, pero el score de recomendación se calcula con
#  similitud del coseno en el espacio latente aprendido por SVD.
#
##############################################################################################

import os
import pickle
import sys
import time
from typing import List, Dict

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

# Añadir raíz para utilidades compartidas
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
from src.utils.registrar_metricas import registrar_metricas


# --------------------------------------------------------------------------------------------
# Configuración
# --------------------------------------------------------------------------------------------
ruta_ratings = "src/data/ready/ratings_finales_ia.csv"
ruta_modelo = "artifacts/weights/modelo_SVD_CS.pkl"
min_ratings_por_usuario = 20
min_ratings_por_item = 20

n_factores = 100
n_epocas = 30
learning_rate = 0.005
regularizacion = 0.02


def cargar_datos() -> pd.DataFrame:
    """Lee ratings y filtra usuarios con poca actividad."""
    print("=" * 70)
    print("  MODELO SVD+CS — Carga de Datos")
    print("=" * 70)

    df = pd.read_csv(ruta_ratings)

    # 1) Filtro por ítem: quitamos películas con muy pocas valoraciones
    conteo_por_item = df.groupby("tmdb_id").size()
    items_validos = conteo_por_item[
        conteo_por_item >= min_ratings_por_item
    ].index
    df = df[df["tmdb_id"].isin(items_validos)].copy()

    # 2) Filtro por usuario
    conteo_por_usuario = df.groupby("userId").size()
    usuarios_validos = conteo_por_usuario[
        conteo_por_usuario >= min_ratings_por_usuario
    ].index
    df_filtrado = df[df["userId"].isin(usuarios_validos)].copy()

    print(f"  Filas bruto: {len(df):,}")
    print(f"  Filtro item >= {min_ratings_por_item} ratings aplicado")
    print(f"  Filas filtrado: {len(df_filtrado):,}")
    print(f"  Usuarios filtrado: {df_filtrado['userId'].nunique():,}")
    print(f"  Peliculas filtrado: {df_filtrado['tmdb_id'].nunique():,}\n")
    return df_filtrado


def entrenar_modelo(df: pd.DataFrame):
    """Entrena SVD para aprender embeddings latentes de usuario e ítem."""
    print("=" * 70)
    print("  MODELO SVD+CS — Entrenamiento")
    print("=" * 70)

    reader = Reader(rating_scale=(0.5, 5.0))
    datos = Dataset.load_from_df(df[["userId", "tmdb_id", "rating"]], reader)
    trainset, testset = train_test_split(datos, test_size=0.2, random_state=42)

    modelo = SVD(
        n_factors=n_factores,
        n_epochs=n_epocas,
        lr_all=learning_rate,
        reg_all=regularizacion,
        verbose=True,
    )

    inicio = time.time()
    modelo.fit(trainset)
    duracion = time.time() - inicio
    print(f"\n  Entrenamiento completado en {duracion:.1f} segundos")

    predicciones = modelo.test(testset)
    rmse = accuracy.rmse(predicciones, verbose=False)
    mae = accuracy.mae(predicciones, verbose=False)
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE : {mae:.4f}")

    return modelo, trainset, rmse, mae


def guardar_modelo(payload: dict) -> None:
    """
    Guarda un payload con:
      - modelo: objeto SVD entrenado
      - trainset: para mapear raw ids <-> inner ids
    """
    os.makedirs(os.path.dirname(ruta_modelo), exist_ok=True)
    with open(ruta_modelo, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  Modelo guardado en: {ruta_modelo}")


def cargar_modelo_guardado():
    if not os.path.exists(ruta_modelo):
        print(f"  No existe {ruta_modelo}")
        return None
    with open(ruta_modelo, "rb") as f:
        payload = pickle.load(f)
    print(f"  Modelo cargado desde {ruta_modelo}")
    return payload


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Coseno robusto para dos vectores 1D."""
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / den)


def recomendar_con_coseno(
    modelo: SVD,
    trainset,
    user_id: int,
    df_ratings: pd.DataFrame,
    n: int = 10,
) -> List[Dict]:
    """
    Recomendación top-N con score de coseno en espacio latente:
      score = cosine(embedding_usuario, embedding_item)
    """
    # Si usuario no estuvo en entrenamiento, no hay embedding
    try:
        u_inner = trainset.to_inner_uid(user_id)
    except ValueError:
        return []

    user_vec = modelo.pu[u_inner]
    pelis_vistas = set(df_ratings[df_ratings["userId"] == user_id]["tmdb_id"].tolist())
    todas_las_pelis = set(df_ratings["tmdb_id"].unique())
    candidatas = todas_las_pelis - pelis_vistas

    recs = []
    for tmdb_id in candidatas:
        try:
            i_inner = trainset.to_inner_iid(tmdb_id)
        except ValueError:
            # Ítem no visto en trainset
            continue
        item_vec = modelo.qi[i_inner]
        score_cs = _cosine_similarity(user_vec, item_vec)
        recs.append({"tmdb_id": int(tmdb_id), "score": round(score_cs, 6)})

    recs.sort(key=lambda x: x["score"], reverse=True)
    return recs[:n]


if __name__ == "__main__":
    df = cargar_datos()
    modelo, trainset, rmse, mae = entrenar_modelo(df)

    guardar_modelo({"modelo": modelo, "trainset": trainset})

    registrar_metricas(
        modelo="SVD+CS",
        hiperparams={
            "n_factores": n_factores,
            "n_epocas": n_epocas,
            "learning_rate": learning_rate,
            "regularizacion": regularizacion,
            "min_ratings_user": min_ratings_por_usuario,
        },
        metricas={"MAE": mae, "RMSE": rmse},
        dataset_size=len(df),
    )

    print("\n" + "=" * 70)
    print("  DEMO: Recomendaciones con score coseno (Usuario 1)")
    print("=" * 70)
    recs = recomendar_con_coseno(modelo, trainset, user_id=1, df_ratings=df, n=10)
    for i, r in enumerate(recs, 1):
        print(f"  {i:02d}. tmdb_id={r['tmdb_id']} | score={r['score']}")

