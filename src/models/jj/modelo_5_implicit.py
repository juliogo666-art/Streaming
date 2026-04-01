##############################################################################################
#
#  MODELO 5: IMPLICIT BPR (Bayesian Personalized Ranking)
#  ==============================================================================
#  *importante: se descarto LightFM por un bug de segmentación nativo en Windows (en MSVC),
#  pasamos a la librería `implicit`.
#
#  ¿Qué pretendemos con Implicit BPR?
#  ---------------------
#  Al ser un modelo de filtrado colaborativo diseñado puramente para datos implícitos,
#  solo sabemos qué vio o le gustó al usuario. No predice una nota como el SVD,
#  sino que optimiza directamente el ranking relativo.
#
#  Dicho de otra forma, mientras SVD predice una nota, BPR predice una probabilidad
#  de que al usuario le guste una película.
#
##############################################################################################

import pandas as pd
import numpy as np
import pickle
import time
import json
from scipy.sparse import csr_matrix
from implicit.bpr import BayesianPersonalizedRanking
from implicit.evaluation import precision_at_k, AUC_at_k

# Rutas de datos
RUTA_RATINGS = "src/data/ready/ratings_finales_ia.csv"
RUTA_CATALOGO = "src/data/ready/dataset_final_movies.csv"

# Rutas de salida del modelo
RUTA_MODELO = "src/models/jj/modelo_5_implicit.pkl"
RUTA_DATASET = "src/models/jj/modelo_5_implicit_dataset.pkl"
RUTA_METRICAS = "src/models/jj/modelo_5_metricas.json"

# Hiperparámetros
UMBRAL_POSITIVO = 3.5
NUM_COMPONENTES = 64
NUM_EPOCAS = 150  # BPR es super rápido, podemos entrenar 150 épocas en menos tiempo que LightFM 10
LEARNING_RATE = 0.05
REGULARIZATION = 0.01


def cargar_y_preparar_datos():
    print("=" * 70)
    print("  MODELO 5: Implicit BPR — Carga y Preparación")
    print("=" * 70)

    # 1. Leer ratings
    print(f"\n  Leyendo {RUTA_RATINGS}...")
    df_ratings = pd.read_csv(RUTA_RATINGS)

    # Solo interacciones positivas (BPR necesita solo 'likes')
    df_positivos = df_ratings[df_ratings["rating"] >= UMBRAL_POSITIVO].copy()

    # 2. Mapeos a IDs continuos
    # Para la matriz dispersa necesitamos que los IDs vayan de 0 a N continuamente
    user_ids = df_positivos["userId"].unique()
    item_ids = df_positivos["tmdb_id"].unique()

    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {i: c for c, i in enumerate(item_ids)}

    df_positivos["user_idx"] = df_positivos["userId"].map(user2idx)
    df_positivos["item_idx"] = df_positivos["tmdb_id"].map(item2idx)

    # 3. Crear matriz CSR (Usuarios x Items)
    print("\n  Construyendo Matriz Dispersa de Interacciones...")
    # Es muy importante castear a int32 porque implicit usa extensiones Cython (C++)
    # y el int64 de Windows provoca crashes extraños de "types/dtypes/memoryviews"
    datos = np.ones(len(df_positivos), dtype=np.float32)
    filas = df_positivos["user_idx"].values.astype(np.int32)
    columnas = df_positivos["item_idx"].values.astype(np.int32)

    matriz_interacciones = csr_matrix(
        (datos, (filas, columnas)), shape=(len(user_ids), len(item_ids))
    )

    print(f"  -> Usuarios: {len(user_ids):,}")
    print(f"  -> Películas: {len(item_ids):,}")
    print(f"  -> Interacciones (no-ceros): {matriz_interacciones.nnz:,}")

    return df_ratings, matriz_interacciones, user2idx, item2idx, df_positivos


def entrenar_modelo(matriz_interacciones):
    print("\n" + "=" * 70)
    print("  MODELO 5: Implicit — Entrenamiento BPR")
    print("=" * 70)

    modelo = BayesianPersonalizedRanking(
        factors=NUM_COMPONENTES,
        learning_rate=LEARNING_RATE,
        regularization=REGULARIZATION,
        iterations=NUM_EPOCAS,
        verify_negative_samples=True,
        random_state=42,
    )

    print(f"  Entrenando {NUM_EPOCAS} épocas con {NUM_COMPONENTES} factores...")
    inicio = time.time()

    # Implicit espera una matriz CRS de usuarios x items.
    modelo.fit(matriz_interacciones)

    tiempo = time.time() - inicio
    print(f"  -> Entrenamiento completado en {tiempo:.1f} segundos.")

    return modelo


def evaluar_modelo(modelo, matriz_interacciones):
    print(
        "\n  Saltando métricas internas de implicit para evitar bug de Cython en Windows."
    )
    print("  La evaluación real se hará mediante evaluacion_ranking.py.")
    return {"model": "BPR"}


def recomendar(
    modelo,
    user_id,
    user2idx,
    item2idx,
    matriz_interacciones,
    df_ratings_completos,
    n=10,
):
    """Función de test para probar predicciones manualmente"""
    if user_id not in user2idx:
        return []

    u_idx = user2idx[user_id]

    # Implicit recommend devuelve (item_idxs, scores)
    # filter_already_liked_items puede fallar en Windows por el casting de memoryviews,
    # así que pedimos extra y filtramos nosotros
    ids, scores = modelo.recommend(
        u_idx, matriz_interacciones[u_idx], N=n + 50, filter_already_liked_items=False
    )

    # Obtener peliculas ya vistas por el usuario
    vistas = set(
        df_ratings_completos[df_ratings_completos["userId"] == user_id][
            "tmdb_id"
        ].tolist()
    )

    # Invertir el mapeo y filtrar
    idx2item = {v: k for k, v in item2idx.items()}
    recomendadas = []

    for idx, sc in zip(ids, scores):
        tmdb_id = idx2item[idx]
        if tmdb_id not in vistas:
            recomendadas.append({"tmdb_id": tmdb_id, "score": float(sc)})
            if len(recomendadas) == n:
                break

    return recomendadas


if __name__ == "__main__":
    df_ratings, matriz, user2idx, item2idx, df_pos = cargar_y_preparar_datos()

    modelo = entrenar_modelo(matriz)
    metricas = evaluar_modelo(modelo, matriz)

    # Guardar el modelo y los mapeos
    print("\n  Guardando modelo y datasets en disco...")

    # Guardamos el modelo
    with open(RUTA_MODELO, "wb") as f:
        pickle.dump(modelo, f)

    dataset = {
        "user2idx": user2idx,
        "item2idx": item2idx,
        # Guardamos también la matriz CSR para que la inferencia sepa qué vio el usuario ya
        "matriz_csr": matriz,
    }
    with open(RUTA_DATASET, "wb") as f:
        pickle.dump(dataset, f)

    with open(RUTA_METRICAS, "w") as f:
        json.dump(metricas, f, indent=4)

    print(f"  Modelo salvado en: {RUTA_MODELO}")

    # Demo Usuario 1
    print("\n" + "=" * 70)
    print("  DEMO: Recomendaciones para Usuario 1")
    print("=" * 70)
    recs = recomendar(modelo, 1, user2idx, item2idx, matriz, df_ratings)
    df_cat = pd.read_csv(RUTA_CATALOGO, on_bad_lines="skip", engine="python")

    for i, r in enumerate(recs, 1):
        info = df_cat[df_cat["tmdb_id"] == r["tmdb_id"]]
        titulo = info["titulo"].values[0] if not info.empty else f"ID: {r['tmdb_id']}"
        print(f"  {i}. {titulo[:45]:<45} | Score: {r['score']:.4f}")

    print("\n  ¡Modelo Implicit listoo!")
