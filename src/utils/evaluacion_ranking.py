"""
#######################################################################################
# SCRIPT DE EVALUACIÓN DE RANKING (NDCG K, Precision K, Hit Rate)
# =======================================================================================
# Compara los modelos (SVD, KNN, W&D, Content-Based, LightFM) sobre un conjunto de usuarios.
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
RUTA_IMP_MOD = "src/models/jj/modelo_5_implicit.pkl"
RUTA_IMP_DAT = "src/models/jj/modelo_5_implicit_dataset.pkl"

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


def recall_at_k(recomendadas, relevantes):
    """De todas las películas relevantes, ¿qué % logró capturar el Top K?"""
    if not relevantes:
        return 0.0
    aciertos = len(set(recomendadas) & set(relevantes))
    return aciertos / len(relevantes)


def coverage(recs_totales, n_catalogo):
    """Porcentaje del catálogo total que nuestro sistema es capaz de recomendar."""
    if n_catalogo == 0:
        return 0.0
    unicas_recomendadas = set()
    for lista in recs_totales:
        unicas_recomendadas.update(lista)
    return len(unicas_recomendadas) / n_catalogo


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

    # Implicit BPR
    if os.path.exists(RUTA_IMP_MOD) and os.path.exists(RUTA_IMP_DAT):
        try:
            with open(RUTA_IMP_MOD, "rb") as f:
                modelos["IMP"] = pickle.load(f)
            with open(RUTA_IMP_DAT, "rb") as f:
                modelos["IMP_DAT"] = pickle.load(f)
        except Exception as e:
            print(f"  Error cargando Implicit: {e}")

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


def predecir_implicit(modelo, dataset, user_id, test_vistas_ratings, candidatas):
    """
    Predicción usando el modelo de la librería `implicit`.
    """
    user2idx = dataset["user2idx"]
    item2idx = dataset["item2idx"]
    
    if user_id not in user2idx:
        return []
        
    u_idx = user2idx[user_id]
    
    # Preparamos las candidatas soportadas (que están en el catálogo entrenado)
    cands_validas = [(tid, item2idx[tid]) for tid in candidatas if tid in item2idx]
    if not cands_validas:
        return []
        
    # Implicit tiene el parámetro `item_idxs` en `recommend` en sus versiones recientes,
    # pero para mayor seguridad, vamos a usar `recommend` limitando o extrayendo a mano los scores
    # O, lo más estándar: pedir muchas y filtrar. Pero `predict` es más fácil:
    # `scores = model.user_factors[u] @ model.item_factors.T`
    
    tids, m_idxs = zip(*cands_validas)
    
    # En implicit, predecir el score es producto escalar del factor de usuario con el factor de item
    u_factors = np.asarray(modelo.user_factors)[u_idx]
    
    # Si las dimensiones son compatibles:
    if hasattr(modelo, "item_factors"):
        i_factors = np.asarray(modelo.item_factors)[list(m_idxs)]
        # Producto escalar para obtener los scores brutos
        scores = np.dot(i_factors, u_factors)
        
        pares = list(zip(tids, scores))
        pares.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in pares[:K]]
    else:
        return []


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

    # Inicializamos contadores por cada modelo
    # 'p' = precision, 'r' = recall, 'n' = ndcg, 'h' = hit_rate, 'recs' = lista todas recs (para coverage)
    metricas_base = {"p": 0, "r": 0, "n": 0, "h": 0, "recs": []}
    resultados = {
        "SVD": metricas_base.copy() if "SVD" in modelos else None,
        "KNN": metricas_base.copy() if "KNN" in modelos else None,
        "WND": metricas_base.copy() if "WND" in modelos else None,
        "TFIDF": metricas_base.copy() if "TFIDF_MAT" in modelos else None,
        "IMP": metricas_base.copy() if "IMP" in modelos else None,
    }
    # Reset lists inside dicts (importante por .copy() superficial)
    for k in resultados:
        if resultados[k] is not None:
            resultados[k] = {"p": 0, "r": 0, "n": 0, "h": 0, "recs": []}

    n_usuarios_final = 0

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
        top_svd = []
        if resultados["SVD"] is not None:
            top_svd = predecir_svd_knn(modelos["SVD"], u, candidatas)
            resultados["SVD"]["p"] += precision_at_k(top_svd, pelis_relevantes)
            resultados["SVD"]["r"] += recall_at_k(top_svd, pelis_relevantes)
            resultados["SVD"]["n"] += ndcg_at_k(top_svd, dict_relevantes)
            resultados["SVD"]["h"] += hit_rate(top_svd, pelis_relevantes)
            resultados["SVD"]["recs"].append(top_svd)

        # Evaluate KNN
        top_knn = []
        if resultados["KNN"] is not None:
            top_knn = predecir_svd_knn(modelos["KNN"], u, candidatas)
            resultados["KNN"]["p"] += precision_at_k(top_knn, pelis_relevantes)
            resultados["KNN"]["r"] += recall_at_k(top_knn, pelis_relevantes)
            resultados["KNN"]["n"] += ndcg_at_k(top_knn, dict_relevantes)
            resultados["KNN"]["h"] += hit_rate(top_knn, pelis_relevantes)
            resultados["KNN"]["recs"].append(top_knn)

        # Evaluate W&D
        top_wnd = []
        if resultados["WND"] is not None:
            top_wnd = predecir_wnd(modelos["WND"], modelos["WND_MAPS"], u, candidatas)
            if top_wnd:
                resultados["WND"]["p"] += precision_at_k(top_wnd, pelis_relevantes)
                resultados["WND"]["r"] += recall_at_k(top_wnd, pelis_relevantes)
                resultados["WND"]["n"] += ndcg_at_k(top_wnd, dict_relevantes)
                resultados["WND"]["h"] += hit_rate(top_wnd, pelis_relevantes)
                resultados["WND"]["recs"].append(top_wnd)

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
            resultados["TFIDF"]["r"] += recall_at_k(top_tf, pelis_relevantes)
            resultados["TFIDF"]["n"] += ndcg_at_k(top_tf, dict_relevantes)
            resultados["TFIDF"]["h"] += hit_rate(top_tf, pelis_relevantes)
            resultados["TFIDF"]["recs"].append(top_tf)

        # Evaluate Implicit BPR
        top_imp = []
        if resultados["IMP"] is not None:
            top_imp = predecir_implicit(
                modelos["IMP"], modelos["IMP_DAT"], u, vistas, candidatas
            )
            if top_imp:
                resultados["IMP"]["p"] += precision_at_k(top_imp, pelis_relevantes)
                resultados["IMP"]["r"] += recall_at_k(top_imp, pelis_relevantes)
                resultados["IMP"]["n"] += ndcg_at_k(top_imp, dict_relevantes)
                resultados["IMP"]["h"] += hit_rate(top_imp, pelis_relevantes)
                resultados["IMP"]["recs"].append(top_imp)

        # Solo si procesamos un usuario válido, aumentamos el contador
        if (
            top_svd
            or top_knn
            or top_wnd
            or top_tf
            or (resultados.get("IMP") and top_imp)
        ):
            n_usuarios_final += 1

    # Convertir sumas a medias --> Dividimos entre N usuarios testados válidos
    n_catalogo = len(todas_pelis)
    print("\n" + "=" * 80)
    print(
        f"  RESULTADOS GLOBALES (Promedios para Top-{K} sobre {n_usuarios_final} usuarios):"
    )
    print("=" * 80)

    records = []
    for mod, vals in resultados.items():
        if vals is None or not vals["recs"]:
            continue

        prec = vals["p"] / n_usuarios_final
        rec = vals["r"] / n_usuarios_final
        ndcg = vals["n"] / n_usuarios_final
        hr = vals["h"] / n_usuarios_final
        cov = coverage(vals["recs"], n_catalogo)

        records.append(
            {
                "Modelo": mod,
                f"Precision_{K}": prec,
                f"Recall_{K}": rec,
                f"NDCG_{K}": ndcg,
                f"Hit_Rate_{K}": hr,
                f"Coverage_{K}": cov,
            }
        )
        print(
            f"  -> {mod:<8} | Precision: {prec * 100:4.1f}% | Recall: {rec * 100:4.1f}% | "
            f"NDCG: {ndcg:.4f} | HR: {hr * 100:4.1f}% | Cov: {cov * 100:4.1f}%"
        )

    # Guardar a CSV
    df_res = pd.DataFrame(records)
    df_res.to_csv(RUTA_RESULTADOS, index=False)
    print("\n  Reporte de métricas guardado en:", RUTA_RESULTADOS)
    print("=" * 60)


############################################################################################

if __name__ == "__main__":
    evaluar()
