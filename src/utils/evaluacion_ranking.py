import pandas as pd
import numpy as np
import pickle
import os
import math
import onnxruntime as ort

# ======================================================================================
# CONFIGURACIÓN DE RUTAS Y CONSTANTES
# ======================================================================================
RUTA_CATALOGO = "src/data/ready/dataset_final_movies.csv"
RUTA_RATINGS = "src/data/ready/ratings_finales_ia.csv"

# Modelos
RUTA_SVD = "src/models/jj/modelo_1_SVD.pkl"
RUTA_KNN = "src/models/jj/modelo_2_knn_cs.pkl"
RUTA_WND_ONNX = "src/models/jj/modelo_3_wnd.onnx"
RUTA_WND_MAP = "src/models/jj/wnd_mappings.pkl"
RUTA_TFIDF_MOD = "src/models/jj/modelo_4_tfidf.pkl"
RUTA_TFIDF_MAT = "src/models/jj/modelo_4_matriz.pkl"
RUTA_TFIDF_IDX = "src/models/jj/modelo_4_indices.pkl"
RUTA_IMP_MOD = "src/models/jj/modelo_5_implicit.pkl"
RUTA_IMP_DAT = "src/models/jj/modelo_5_implicit_dataset.pkl"
RUTA_NCF_ONNX = "src/models/jj/modelo_6_ncf.onnx"
RUTA_NCF_USER2IDX = "src/models/jj/ncf_user2idx.json"
RUTA_NCF_ITEM2IDX = "src/models/jj/ncf_item2idx.json"
RUTA_TT_ONNX = "src/models/jj/modelo_7_twotowers.onnx"
RUTA_TT_MAP = "src/models/jj/twotowers_mappings.pkl"

# Guardar Resultados
RUTA_RESULTADOS = "src/utils/metricas_ranking.csv"

# Parámetros de la Evaluación
K = 10  # Número de recomendaciones a generar por modelo
NUM_USUARIOS = 300  # Cantidad de usuarios de prueba para el benchmark
UMBRAL_RELEVANTE = (
    4.0  # Nota mínima para considerar que a un usuario "le gustó" la peli
)

# ======================================================================================
# Definición de metricas de ranking
# ======================================================================================


def precision_at_k(recomendadas, relevantes):
    """Mide qué porcentaje de las K sugeridas fueron realmente acertadas."""
    if not recomendadas:
        return 0.0
    aciertos = len(set(recomendadas) & set(relevantes))
    return aciertos / len(recomendadas)


def recall_at_k(recomendadas, relevantes):
    """Mide cuántas de las pelis que le gustaron fuimos capaces de encontrar."""
    if not relevantes:
        return 0.0
    aciertos = len(set(recomendadas) & set(relevantes))
    return aciertos / len(relevantes)


def coverage(recs_totales, n_catalogo):
    """Mide qué porcentaje del catálogo total es capaz de recomendar el modelo (Diversidad)."""
    if n_catalogo == 0:
        return 0.0
    unicas_recomendadas = set()
    for lista in recs_totales:
        unicas_recomendadas.update(lista)
    return len(unicas_recomendadas) / n_catalogo


def hit_rate(recomendadas, relevantes):
    """Métrica binaria: ¿Hubo al menos 1 acierto en el Top K? (1 Sí / 0 No)."""
    return 1 if len(set(recomendadas) & set(relevantes)) > 0 else 0


def ndcg_at_k(recomendadas, relevantes_escala):
    """
    NDCG (Normalized Discounted Cumulative Gain):
    Es la métrica más importante. No solo mide si acertamos, sino si pusimos las
    mejores películas en las primeras posiciones (penaliza si la favorita sale la #10).
    """
    dcg = 0.0
    idcg = 0.0
    for i, peli in enumerate(recomendadas):
        if peli in relevantes_escala:
            relevancia = relevantes_escala[peli]
            # La importancia decrece logarítmicamente con la posición
            dcg += relevancia / math.log2(i + 2)

    # IDCG es el "escenario perfecto" para normalizar el resultado entre 0 y 1
    ideal_relevancias = sorted(list(relevantes_escala.values()), reverse=True)
    for i, rel in enumerate(ideal_relevancias[: len(recomendadas)]):
        idcg += rel / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0


# ======================================================================================
# Carga de modelos (Joblib, Pickle y ONNX Runtime)
# ======================================================================================


def cargar_modelos():
    """Busca y carga todos los modelos entrenados disponibles en disco."""
    print("  Cargando modelos de IA...")
    modelos = {}

    # Modelos clásicos (Surprise / Sklearn)
    if os.path.exists(RUTA_SVD):
        with open(RUTA_SVD, "rb") as f:
            modelos["SVD"] = pickle.load(f)
    if os.path.exists(RUTA_KNN):
        with open(RUTA_KNN, "rb") as f:
            modelos["KNN"] = pickle.load(f)

    # Content-Based (TF-IDF)
    if os.path.exists(RUTA_TFIDF_MAT):
        with open(RUTA_TFIDF_MAT, "rb") as f:
            modelos["TFIDF_MAT"] = pickle.load(f)
        with open(RUTA_TFIDF_IDX, "rb") as f:
            modelos["TFIDF_IDX"] = pickle.load(f)

    # Deep Learning (ONNX Runtime para alta velocidad)
    if os.path.exists(RUTA_WND_ONNX):
        modelos["WND_ONNX"] = ort.InferenceSession(RUTA_WND_ONNX)
        with open(RUTA_WND_MAP, "rb") as f:
            modelos["WND_MAPS"] = pickle.load(f)

    if os.path.exists(RUTA_IMP_MOD):
        with open(RUTA_IMP_MOD, "rb") as f:
            modelos["IMP"] = pickle.load(f)
        with open(RUTA_IMP_DAT, "rb") as f:
            modelos["IMP_DAT"] = pickle.load(f)

    if os.path.exists(RUTA_NCF_ONNX):
        modelos["NCF_ONNX"] = ort.InferenceSession(RUTA_NCF_ONNX)
        import json

        with open(RUTA_NCF_USER2IDX, "r") as f:
            modelos["NCF_U"] = {int(k): v for k, v in json.load(f).items()}
        with open(RUTA_NCF_ITEM2IDX, "r") as f:
            modelos["NCF_I"] = {int(k): v for k, v in json.load(f).items()}

    if os.path.exists(RUTA_TT_ONNX):
        modelos["TT_ONNX"] = ort.InferenceSession(RUTA_TT_ONNX)
        with open(RUTA_TT_MAP, "rb") as f:
            modelos["TT_MAPS"] = pickle.load(f)

    return modelos


# ======================================================================================
# Predicciones
# ======================================================================================


def predecir_svd_knn(modelo, user_id, candidatas):
    """Predicción clásica basada en puntuación estimada."""
    preds = []
    for tid in candidatas:
        preds.append((tid, modelo.predict(user_id, tid).est))
    preds.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in preds[:K]]


def predecir_wnd(sess, maps, user_id, candidatas):
    """Inferencia Wide & Deep usando ONNX Runtime."""
    u2idx = maps["user2idx"]
    m2idx = maps["movie2idx"]
    if user_id not in u2idx:
        return []
    u_idx = u2idx[user_id]
    # Filtramos candidatos que el modelo conoce
    cv = [(tid, m2idx[tid]) for tid in candidatas if tid in m2idx]
    if not cv:
        return []
    tids, idxs = zip(*cv)
    # Batch predict: Evaluamos miles de pelis de golpe
    u_arr = np.full(len(idxs), u_idx, dtype=np.int64)
    i_arr = np.array(idxs, dtype=np.int64)
    scores = sess.run(None, {"user_ids": u_arr, "item_ids": i_arr})[0].flatten()
    pares = sorted(zip(tids, scores), key=lambda x: x[1], reverse=True)
    return [p[0] for p in pares[:K]]


def predecir_tt(sess, maps, user_id, candidatas):
    """Inferencia de la arquitectura 'Two Towers' bi-encoder."""
    u2idx = maps["user2idx"]
    i2idx = maps["item2idx"]
    if user_id not in u2idx:
        return []
    u_idx = u2idx[user_id]
    cv = [(tid, i2idx[tid]) for tid in candidatas if tid in i2idx]
    if not cv:
        return []
    tids, idxs = zip(*cv)
    u_arr = np.full(len(idxs), u_idx, dtype=np.int64)
    i_arr = np.array(idxs, dtype=np.int64)
    # El producto escalar se resuelve dentro del grafo ONNX
    scores = sess.run(None, {"user_ids": u_arr, "item_ids": i_arr})[0].flatten()
    pares = sorted(zip(tids, scores), key=lambda x: x[1], reverse=True)
    return [p[0] for p in pares[:K]]


def predecir_ncf(sess, u2idx, i2idx, user_id, candidatas):
    """Neural Collaborative Filtering (GMF + MLP)."""
    if user_id not in u2idx:
        return []
    u_idx = u2idx[user_id]
    cv = [(tid, i2idx[tid]) for tid in candidatas if tid in i2idx]
    if not cv:
        return []
    tids, idxs = zip(*cv)
    u_arr = np.full(len(idxs), u_idx, dtype=np.int64)
    i_arr = np.array(list(idxs), dtype=np.int64)
    scores = sess.run(None, {"user_ids": u_arr, "item_ids": i_arr})[0].flatten()
    pares = sorted(zip(tids, scores), key=lambda x: x[1], reverse=True)
    return [p[0] for p in pares[:K]]


def predecir_content(mat, idxs, user_vistas, candidatas):
    """Recomendador basado en similitud de coseno sobre TF-IDF."""
    from sklearn.metrics.pairwise import cosine_similarity

    if user_vistas.empty:
        return []
    # Usamos su película favorita como ancla
    fav = user_vistas.sort_values(by="rating", ascending=False).iloc[0]
    tid_fav = int(fav["tmdb_id"])
    if tid_fav not in idxs:
        return []
    idx_fav = idxs[tid_fav]
    sims = cosine_similarity(mat[idx_fav], mat).flatten()
    preds = []
    for tid in candidatas:
        if tid in idxs:
            preds.append((tid, sims[idxs[tid]]))
    preds.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in preds[:K]]


def predecir_implicit(mod, dat, user_id, candidatas):
    """Bayesian Personalized Ranking (BPR) sobre factores latentes."""
    u2idx = dat["user2idx"]
    i2idx = dat["item2idx"]
    idx2i = {v: k for k, v in i2idx.items()}

    if user_id not in u2idx:
        return []

    u_idx = u2idx[user_id]
    uf = np.asarray(mod.user_factors[u_idx])
    iff = np.asarray(mod.item_factors)

    # Cálculo manual del producto escalar para evitar fallos de memoria en Windows
    scores = uf @ iff.T
    c_set = set(candidatas)
    preds = []
    for midx, s in enumerate(scores):
        tid = idx2i[midx]
        if tid in c_set:
            preds.append((tid, s))

    # Ordenar por el score calculado
    preds.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in preds[:K]]


# ---- EVALUACIÓN ----


def evaluar():
    """Ejecuta el protocolo de validación: Ocultar datos reales y medir si la IA los predice."""
    print("\n INICIANDO EVALUACIÓN DEFINITIVA")
    modelos = cargar_modelos()
    df = pd.read_csv(RUTA_RATINGS, on_bad_lines="skip")
    df_cat = pd.read_csv(RUTA_CATALOGO)

    # Catálogo de pelis "válidas" (con suficientes votos) para no recomendar basura
    todas_pelis = (
        set(df_cat[df_cat["vote_count"] > 100]["tmdb_id"].unique())
        if "vote_count" in df_cat.columns
        else set(df_cat["tmdb_id"].unique())
    )

    # Elegimos usuarios con historial rico para un benchmark exigente
    counts = df.groupby("userId").size()
    core_users = counts[counts >= 1000].index.tolist()
    u_eval = np.random.choice(
        core_users, min(len(core_users), NUM_USUARIOS), replace=False
    )

    # Optimizacion O(1): Indexar el dataset por userId para búsquedas ultra rápidas
    df_idx = df.set_index("userId")

    # Contenedores para métricas acumuladas
    res = {
        m: {"p": 0, "r": 0, "n": 0, "h": 0, "recs": []}
        for m in ["SVD", "KNN", "WND", "TFIDF", "IMP", "NCF", "TT"]
    }
    n_final = 0

    print(f"  Analizando {len(u_eval)} usuarios expertos...")
    for u in u_eval:
        try:
            # 1. Obtener historial del usuario
            ud = df_idx.loc[[u]].reset_index()
            rel = ud[ud["rating"] >= UMBRAL_RELEVANTE]
            if len(rel) < 5:
                continue

            # 2. GROUND TRUTH: Ocultamos el 20% de lo que le gustó para usarlo como 'examen'
            oculto = rel.sample(frac=0.2, random_state=42)
            vistas = ud.drop(oculto.index)

            # El diccionario GT tiene los IDs que la IA deberia adivinar
            gt_dict = {row["tmdb_id"]: row["rating"] for _, row in oculto.iterrows()}
            gt_list = list(gt_dict.keys())

            # Candidatas: El resto del catálogo que el usuario no ha visto aún
            cands = todas_pelis - set(vistas["tmdb_id"].unique())

            # 3. Lanzar predicciones y acumular metricas
            # Se repite para cada modelo cargado
            if "SVD" in modelos:
                t = predecir_svd_knn(modelos["SVD"], u, cands)
                res["SVD"]["p"] += precision_at_k(t, gt_list)
                res["SVD"]["r"] += recall_at_k(t, gt_list)
                res["SVD"]["n"] += ndcg_at_k(t, gt_dict)
                res["SVD"]["h"] += hit_rate(t, gt_list)
                res["SVD"]["recs"].append(t)

            if "KNN" in modelos:
                t = predecir_svd_knn(modelos["KNN"], u, cands)
                res["KNN"]["p"] += precision_at_k(t, gt_list)
                res["KNN"]["r"] += recall_at_k(t, gt_list)
                res["KNN"]["n"] += ndcg_at_k(t, gt_dict)
                res["KNN"]["h"] += hit_rate(t, gt_list)
                res["KNN"]["recs"].append(t)

            if "WND_ONNX" in modelos:
                t = predecir_wnd(modelos["WND_ONNX"], modelos["WND_MAPS"], u, cands)
                res["WND"]["p"] += precision_at_k(t, gt_list)
                res["WND"]["r"] += recall_at_k(t, gt_list)
                res["WND"]["n"] += ndcg_at_k(t, gt_dict)
                res["WND"]["h"] += hit_rate(t, gt_list)
                res["WND"]["recs"].append(t)

            if "TFIDF_MAT" in modelos:
                t = predecir_content(
                    modelos["TFIDF_MAT"], modelos["TFIDF_IDX"], vistas, cands
                )
                res["TFIDF"]["p"] += precision_at_k(t, gt_list)
                res["TFIDF"]["r"] += recall_at_k(t, gt_list)
                res["TFIDF"]["n"] += ndcg_at_k(t, gt_dict)
                res["TFIDF"]["h"] += hit_rate(t, gt_list)
                res["TFIDF"]["recs"].append(t)

            if "IMP" in modelos:
                # Corregimos BPR que generaba error de variable
                u2idx_imp = modelos["IMP_DAT"]["user2idx"]
                i2idx_imp = modelos["IMP_DAT"]["item2idx"]
                idx2i_imp = {v: k for k, v in i2idx_imp.items()}
                u_idx_imp = u2idx_imp[u]
                uf = np.asarray(modelos["IMP"].user_factors[u_idx_imp])
                iff = np.asarray(modelos["IMP"].item_factors)
                scores_imp = uf @ iff.T
                pares_imp = sorted(
                    [
                        (idx2i_imp[mi], sc)
                        for mi, sc in enumerate(scores_imp)
                        if idx2i_imp[mi] in cands
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )
                t = [p[0] for p in pares_imp[:K]]
                res["IMP"]["p"] += precision_at_k(t, gt_list)
                res["IMP"]["r"] += recall_at_k(t, gt_list)
                res["IMP"]["n"] += ndcg_at_k(t, gt_dict)
                res["IMP"]["h"] += hit_rate(t, gt_list)
                res["IMP"]["recs"].append(t)

            if "NCF_ONNX" in modelos:
                t = predecir_ncf(
                    modelos["NCF_ONNX"], modelos["NCF_U"], modelos["NCF_I"], u, cands
                )
                res["NCF"]["p"] += precision_at_k(t, gt_list)
                res["NCF"]["r"] += recall_at_k(t, gt_list)
                res["NCF"]["n"] += ndcg_at_k(t, gt_dict)
                res["NCF"]["h"] += hit_rate(t, gt_list)
                res["NCF"]["recs"].append(t)

            if "TT_ONNX" in modelos:
                t = predecir_tt(modelos["TT_ONNX"], modelos["TT_MAPS"], u, cands)
                res["TT"]["p"] += precision_at_k(t, gt_list)
                res["TT"]["r"] += recall_at_k(t, gt_list)
                res["TT"]["n"] += ndcg_at_k(t, gt_dict)
                res["TT"]["h"] += hit_rate(t, gt_list)
                res["TT"]["recs"].append(t)

            n_final += 1
        except:
            continue

    # ======================================================================================
    # Reporte final y exportación
    # ======================================================================================
    print("\n RESULTADOS FINALES (Promedio):")
    records = []
    cat_n = len(todas_pelis)
    for m, v in res.items():
        if not v["recs"]:
            continue
        # Promediar métricas por el número de usuarios evaluados
        p, r, n, h, c = (
            v["p"] / n_final,
            v["r"] / n_final,
            v["n"] / n_final,
            v["h"] / n_final,
            coverage(v["recs"], cat_n),
        )
        print(
            f"  {m:<6} | Prec: {p * 100:4.1f}% | NDCG: {n:.3f} | Hit: {h * 100:4.1f}%"
        )
        records.append(
            {
                "Modelo": m,
                "Precision_10": p,
                "Recall_10": r,
                "NDCG_10": n,
                "Hit_Rate_10": h,
                "Coverage_10": c,
            }
        )

    # Guardamos para que el Admin Panel lo muestre en la web
    pd.DataFrame(records).to_csv(RUTA_RESULTADOS, index=False)
    print(f"\n Informe guardado en: {RUTA_RESULTADOS}")


if __name__ == "__main__":
    evaluar()
