"""
#######################################################################################
# SCRIPT DE EVALUACIÓN DE RANKING (NDCG K, Precision K, Hit Rate)
# =======================================================================================
# Compara los 4 modelos (SVD, KNN, W&D, Content-Based) sobre un conjunto de usuarios.
# Oculta películas que el usuario ha valorado positivamente (relevantes) y comprueba
# si el modelo logra sugerirlas en su Top 10.
#######################################################################################
"""

import pandas as pd
import numpy as np
import pickle
import os
import torch
import math

############################################################################################

# ---- Rutas ----
RUTA_CATALOGO = "src/data/ready/dataset_final_movies.csv"
RUTA_RATINGS = "src/data/ready/ratings_finales_ia.csv"

# Modelos
RUTA_SVD = "src/models/jj/modelo_1_SVD.pkl"
RUTA_KNN = "src/models/jj/modelo_2_knn_cs.pkl"
RUTA_WND_PTH = "src/models/jj/modelo_3_wnd.pth"
RUTA_WND_MAP = "src/models/jj/wnd_mappings.pkl"
RUTA_TFIDF_MOD = "src/models/jj/modelo_4_tfidf.pkl"
RUTA_TFIDF_MAT = "src/models/jj/modelo_4_matriz.pkl"
RUTA_TFIDF_IDX = "src/models/jj/modelo_4_indices.pkl"

# Guardar Resultados
RUTA_RESULTADOS = "src/utils/metricas_ranking.csv"

# Configuración
K = 10  # Queremos evaluar el Top-10
NUM_USUARIOS = 300  # Aumentado a 300 usuarios para dar rigor estadístico al paper
UMBRAL_RELEVANTE = (
    4.0  # Una película es 'Relevante' si el usuario le dio >= 4 estrellas
)

############################################################################################

# ---- FUNCIONES DE MÉTRICAS MATEMÁTICAS ----


def precision_at_k(recomendadas, relevantes):
    """Porcentaje de recomendaciones (Top K) que son realmente películas relevantes."""
    if not recomendadas:
        return 0.0
    aciertos = len(set(recomendadas) & set(relevantes))
    return aciertos / len(recomendadas)


############################################################################################


def hit_rate(recomendadas, relevantes):
    """1 = si al menos recomendó 1 película relevante, 0 = si no acertó ni una."""
    return 1 if len(set(recomendadas) & set(relevantes)) > 0 else 0


############################################################################################


def ndcg_at_k(recomendadas, relevantes_escala):
    """
    Normalized Discounted Cumulative Gain.
    Recompensamos mas si el acierto ocurre en el Top 1 que en el Top 10.
    relevantes_escala es un dict: {tmdb_id: nota_real}
    """
    dcg = 0.0
    idcg = 0.0

    # Calcular DCG
    for i, peli in enumerate(recomendadas):
        if peli in relevantes_escala:
            relevancia = relevantes_escala[peli]  # Ej. 4.5 o 5.0
            dcg += relevancia / math.log2(i + 2)  # Penalización de posición

    # Calcular IDCG (El DCG Ideal si lo hubiese ordenado perfectamente)
    ideal_relevancias = sorted(list(relevantes_escala.values()), reverse=True)
    for i, rel in enumerate(ideal_relevancias[: len(recomendadas)]):
        idcg += rel / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


############################################################################################
# ---- LÓGICA PRINCIPAL ----


def cargar_modelos():
    print("=" * 60)
    print("  Cargando modelos de IA...")
    print("=" * 60)

    modelos = {}
    # SVD
    if os.path.exists(RUTA_SVD):
        with open(RUTA_SVD, "rb") as f:
            modelos["SVD"] = pickle.load(f)

    # KNN
    if os.path.exists(RUTA_KNN):
        with open(RUTA_KNN, "rb") as f:
            modelos["KNN"] = pickle.load(f)

    # Content-Based (TF-IDF)
    if os.path.exists(RUTA_TFIDF_MAT):
        with open(RUTA_TFIDF_MAT, "rb") as f:
            modelos["TFIDF_MAT"] = pickle.load(f)
        with open(RUTA_TFIDF_IDX, "rb") as f:
            modelos["TFIDF_IDX"] = pickle.load(f)

    # Wide&Deep
    if os.path.exists(RUTA_WND_PTH):
        try:
            import sys

            sys.path.insert(
                0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            )
            from src.networks.dl.rn_mlp import WideAndDeepModel

            with open(RUTA_WND_MAP, "rb") as f:
                wnd_maps = pickle.load(f)
            wnd = WideAndDeepModel(
                len(wnd_maps["user2idx"]), len(wnd_maps["movie2idx"]), 32, [64, 32]
            )
            wnd.load_state_dict(
                torch.load(RUTA_WND_PTH, map_location="cpu", weights_only=True)
            )
            wnd.eval()
            modelos["WND"] = wnd
            modelos["WND_MAPS"] = wnd_maps
        except Exception as e:
            print(f"  Error cargando W&D: {e}")

    print(f"  Modelos listos: {list(modelos.keys())}")
    return modelos


############################################################################################


def predecir_svd_knn(modelo, user_id, candidatas):
    preds = []
    for tmdb_id in candidatas:
        pred = modelo.predict(user_id, tmdb_id)
        preds.append((tmdb_id, pred.est))
    preds.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in preds[:K]]


############################################################################################


def predecir_wnd(modelo, mapas, user_id, candidatas):
    u2idx = mapas["user2idx"]
    m2idx = mapas["movie2idx"]

    if user_id not in u2idx:
        return []
    u_idx = u2idx[user_id]

    candidatas_validas = [(tid, m2idx[tid]) for tid in candidatas if tid in m2idx]
    if not candidatas_validas:
        return []

    tids, m_idxs = zip(*candidatas_validas)
    t_users = torch.tensor([u_idx] * len(m_idxs), dtype=torch.long)
    t_items = torch.tensor(list(m_idxs), dtype=torch.long)

    with torch.no_grad():
        # NO usamos torch.clamp(..., 0.5, 5.0) aquí porque destruye el orden relativo
        # de las películas favoritas si muchas de ellas se saturan al tope de la escala.
        # Para ranking, nos da igual que prediga 5.3 o 6.8, solo nos importa quién es mayor.
        preds = modelo(t_users, t_items)

    pares = [(tids[i], preds[i].item()) for i in range(len(tids))]
    pares.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in pares[:K]]


############################################################################################


def predecir_content(matriz, indices, df_movies, user_ratings_vistas, candidatas):
    from sklearn.metrics.pairwise import linear_kernel

    # Película con nota más alta del historial de "vistas"
    if user_ratings_vistas.empty:
        return []
    fav = user_ratings_vistas.sort_values(by="rating", ascending=False).iloc[0]
    tid_fav = int(fav["tmdb_id"])

    if tid_fav not in indices:
        return []
    idx_fav = indices[tid_fav]

    # Simulamos Similitud en bloque
    vector_fav = matriz[idx_fav]
    similitudes = linear_kernel(vector_fav, matriz).flatten()

    # Filtrar solo candidatos
    preds_sims = []
    for tid in candidatas:
        if tid in indices:
            preds_sims.append((tid, similitudes[indices[tid]]))

    preds_sims.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in preds_sims[:K]]


############################################################################################


def evaluar():
    print("\n" + "=" * 60)
    print(f"  Evaluacion del ranking (Top {K})")
    print("=" * 60)

    modelos = cargar_modelos()

    df = pd.read_csv(RUTA_RATINGS)
    df_movies = pd.read_csv(RUTA_CATALOGO)

    # Método Netflix:
    # Filtramos el catálogo candidato a películas con más de 50 votos globales.
    # ¿Por qué? Las Redes Neuronales (W&D) inicializan sus pesos (Embeddings) al azar.
    # Si una película tiene solo 1 o 2 votos, la red no la entrena, y su peso aleatorio puede
    # quedar inflado, provocando que recomiende "basura desconocida" en el Top 10 y saque 0% de acierto.
    if "vote_count" in df_movies.columns:
        pelis_conocidas = df_movies[df_movies["vote_count"] > 50]
        todas_pelis = set(pelis_conocidas["tmdb_id"].unique())
    else:
        todas_pelis = set(df_movies["tmdb_id"].unique())

    # 1. Muestrear usuarios
    conteos = df.groupby("userId").size()
    u_abundantes = conteos[conteos > 30].index.tolist()
    u_evaluar = np.random.choice(u_abundantes, NUM_USUARIOS, replace=False)

    resultados = {
        "SVD": {"p": 0, "n": 0, "h": 0},
        "KNN": {"p": 0, "n": 0, "h": 0},
        "WND": {"p": 0, "n": 0, "h": 0},
        "TFIDF": {"p": 0, "n": 0, "h": 0},
    }

    print(f"\n  Iniciando evaluación sobre {NUM_USUARIOS} usuarios...")

    for u in u_evaluar:
        user_data = df[df["userId"] == u]

        # Ocultamos artificialmente el 20% de sus películas altas (Relevantes)
        relevantes = user_data[user_data["rating"] >= UMBRAL_RELEVANTE]
        if len(relevantes) < 5:
            continue  # Muy pocas favoritas para testear

        test_oculto = relevantes.sample(frac=0.2, random_state=42)
        vistas = user_data.drop(test_oculto.index)

        # El conjunto "Verdadero (Ground Truth)" para el test
        dict_relevantes = {
            row["tmdb_id"]: row["rating"] for _, row in test_oculto.iterrows()
        }
        pelis_relevantes = list(dict_relevantes.keys())

        # Conjunto Candidato: Todas las pelis menos las "Vistas"
        # Incluimos las ocultas para ver si las pilla
        candidatas = todas_pelis - set(vistas["tmdb_id"].unique())

        # Evaluate SVD
        if "SVD" in modelos:
            top_svd = predecir_svd_knn(modelos["SVD"], u, candidatas)
            resultados["SVD"]["p"] += precision_at_k(top_svd, pelis_relevantes)
            resultados["SVD"]["n"] += ndcg_at_k(top_svd, dict_relevantes)
            resultados["SVD"]["h"] += hit_rate(top_svd, pelis_relevantes)

        # Evaluate KNN
        if "KNN" in modelos:
            top_knn = predecir_svd_knn(modelos["KNN"], u, candidatas)
            resultados["KNN"]["p"] += precision_at_k(top_knn, pelis_relevantes)
            resultados["KNN"]["n"] += ndcg_at_k(top_knn, dict_relevantes)
            resultados["KNN"]["h"] += hit_rate(top_knn, pelis_relevantes)

        # Evaluate W&D
        if "WND" in modelos:
            top_wnd = predecir_wnd(modelos["WND"], modelos["WND_MAPS"], u, candidatas)
            if top_wnd:
                resultados["WND"]["p"] += precision_at_k(top_wnd, pelis_relevantes)
                resultados["WND"]["n"] += ndcg_at_k(top_wnd, dict_relevantes)
                resultados["WND"]["h"] += hit_rate(top_wnd, pelis_relevantes)

        # Evaluate Content-Based
        if "TFIDF_MAT" in modelos:
            top_tf = predecir_content(
                modelos["TFIDF_MAT"],
                modelos["TFIDF_IDX"],
                df_movies,
                vistas,
                candidatas,
            )
            resultados["TFIDF"]["p"] += precision_at_k(top_tf, pelis_relevantes)
            resultados["TFIDF"]["n"] += ndcg_at_k(top_tf, dict_relevantes)
            resultados["TFIDF"]["h"] += hit_rate(top_tf, pelis_relevantes)

    # Convertir sumas a medias --> Dividimos entre N usuarios testados válidos
    print("\n" + "=" * 60)
    print(f"  RESULTADOS GLOBALES (Promedios para Top-{K}):")
    print("=" * 60)

    records = []
    for mod, vals in resultados.items():
        if vals["n"] == 0 and mod != "WND":
            continue  # Saltamos los no evaluados

        prec = vals["p"] / NUM_USUARIOS
        ndcg = vals["n"] / NUM_USUARIOS
        hr = vals["h"] / NUM_USUARIOS

        records.append(
            {
                "Modelo": mod,
                f"Precision_{K}": prec,
                f"NDCG_{K}": ndcg,
                f"Hit_Rate_{K}": hr,
            }
        )
        print(
            f"  -> {mod:<8} | Precision: {prec * 100:04.1f}% | NDCG: {ndcg:.4f} | HR: {hr * 100:04.1f}%"
        )

    # Guardar a CSV
    df_res = pd.DataFrame(records)
    df_res.to_csv(RUTA_RESULTADOS, index=False)
    print("\n  Reporte de métricas guardado en:", RUTA_RESULTADOS)
    print("=" * 60)


############################################################################################

if __name__ == "__main__":
    evaluar()
