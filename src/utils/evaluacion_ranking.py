import pandas as pd
import numpy as np
import pickle
import os
import math
import onnxruntime as ort

# ---- Rutas ----
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

# Configuración
K = 10
NUM_USUARIOS = 300
UMBRAL_RELEVANTE = 4.0

# ---- FUNCIONES DE MÉTRICAS ----

def precision_at_k(recomendadas, relevantes):
    if not recomendadas: return 0.0
    aciertos = len(set(recomendadas) & set(relevantes))
    return aciertos / len(recomendadas)

def recall_at_k(recomendadas, relevantes):
    if not relevantes: return 0.0
    aciertos = len(set(recomendadas) & set(relevantes))
    return aciertos / len(relevantes)

def coverage(recs_totales, n_catalogo):
    if n_catalogo == 0: return 0.0
    unicas_recomendadas = set()
    for lista in recs_totales:
        unicas_recomendadas.update(lista)
    return len(unicas_recomendadas) / n_catalogo

def hit_rate(recomendadas, relevantes):
    return 1 if len(set(recomendadas) & set(relevantes)) > 0 else 0

def ndcg_at_k(recomendadas, relevantes_escala):
    dcg = 0.0
    idcg = 0.0
    for i, peli in enumerate(recomendadas):
        if peli in relevantes_escala:
            relevancia = relevantes_escala[peli]
            dcg += relevancia / math.log2(i + 2)
    ideal_relevancias = sorted(list(relevantes_escala.values()), reverse=True)
    for i, rel in enumerate(ideal_relevancias[: len(recomendadas)]):
        idcg += rel / math.log2(i + 2)
    return dcg / idcg if idcg > 0 else 0.0

# ---- LÓGICA DE CARGA ----

def cargar_modelos():
    print("  Cargando modelos de IA...")
    modelos = {}
    
    if os.path.exists(RUTA_SVD):
        with open(RUTA_SVD, "rb") as f: modelos["SVD"] = pickle.load(f)
    if os.path.exists(RUTA_KNN):
        with open(RUTA_KNN, "rb") as f: modelos["KNN"] = pickle.load(f)
    if os.path.exists(RUTA_TFIDF_MAT):
        with open(RUTA_TFIDF_MAT, "rb") as f: modelos["TFIDF_MAT"] = pickle.load(f)
        with open(RUTA_TFIDF_IDX, "rb") as f: modelos["TFIDF_IDX"] = pickle.load(f)
    if os.path.exists(RUTA_WND_ONNX):
        modelos["WND_ONNX"] = ort.InferenceSession(RUTA_WND_ONNX)
        with open(RUTA_WND_MAP, "rb") as f: modelos["WND_MAPS"] = pickle.load(f)
    if os.path.exists(RUTA_IMP_MOD):
        with open(RUTA_IMP_MOD, "rb") as f: modelos["IMP"] = pickle.load(f)
        with open(RUTA_IMP_DAT, "rb") as f: modelos["IMP_DAT"] = pickle.load(f)
    if os.path.exists(RUTA_NCF_ONNX):
        modelos["NCF_ONNX"] = ort.InferenceSession(RUTA_NCF_ONNX)
        import json
        with open(RUTA_NCF_USER2IDX, "r") as f: modelos["NCF_U"] = {int(k): v for k, v in json.load(f).items()}
        with open(RUTA_NCF_ITEM2IDX, "r") as f: modelos["NCF_I"] = {int(k): v for k, v in json.load(f).items()}
    if os.path.exists(RUTA_TT_ONNX):
        modelos["TT_ONNX"] = ort.InferenceSession(RUTA_TT_ONNX)
        with open(RUTA_TT_MAP, "rb") as f: modelos["TT_MAPS"] = pickle.load(f)
        
    return modelos

# ---- FUNCIONES DE PREDICCIÓN ----

def predecir_svd_knn(modelo, user_id, candidatas):
    preds = []
    for tid in candidatas:
        preds.append((tid, modelo.predict(user_id, tid).est))
    preds.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in preds[:K]]

def predecir_wnd(sess, maps, user_id, candidatas):
    u2idx = maps["user2idx"]; m2idx = maps["movie2idx"]
    if user_id not in u2idx: return []
    u_idx = u2idx[user_id]
    cv = [(tid, m2idx[tid]) for tid in candidatas if tid in m2idx]
    if not cv: return []
    tids, idxs = zip(*cv)
    u_arr = np.full(len(idxs), u_idx, dtype=np.int64); i_arr = np.array(idxs, dtype=np.int64)
    scores = sess.run(None, {"user_ids": u_arr, "item_ids": i_arr})[0].flatten()
    pares = sorted(zip(tids, scores), key=lambda x: x[1], reverse=True)
    return [p[0] for p in pares[:K]]

def predecir_tt(sess, maps, user_id, candidatas):
    u2idx = maps["user2idx"]; i2idx = maps["item2idx"]
    if user_id not in u2idx: return []
    u_idx = u2idx[user_id]
    cv = [(tid, i2idx[tid]) for tid in candidatas if tid in i2idx]
    if not cv: return []
    tids, idxs = zip(*cv)
    u_arr = np.full(len(idxs), u_idx, dtype=np.int64); i_arr = np.array(idxs, dtype=np.int64)
    scores = sess.run(None, {"user_ids": u_arr, "item_ids": i_arr})[0].flatten()
    pares = sorted(zip(tids, scores), key=lambda x: x[1], reverse=True)
    return [p[0] for p in pares[:K]]

def predecir_ncf(sess, u2idx, i2idx, user_id, candidatas):
    if user_id not in u2idx: return []
    u_idx = u2idx[user_id]
    cv = [(tid, i2idx[tid]) for tid in candidatas if tid in i2idx]
    if not cv: return []
    tids, idxs = zip(*cv)
    u_arr = np.full(len(idxs), u_idx, dtype=np.int64); i_arr = np.array(list(idxs), dtype=np.int64)
    scores = sess.run(None, {"user_ids": u_arr, "item_ids": i_arr})[0].flatten()
    pares = sorted(zip(tids, scores), key=lambda x: x[1], reverse=True)
    return [p[0] for p in pares[:K]]

def predecir_content(mat, idxs, user_vistas, candidatas):
    from sklearn.metrics.pairwise import cosine_similarity
    if user_vistas.empty: return []
    fav = user_vistas.sort_values(by="rating", ascending=False).iloc[0]
    tid_fav = int(fav["tmdb_id"])
    if tid_fav not in idxs: return []
    idx_fav = idxs[tid_fav]
    sims = cosine_similarity(mat[idx_fav], mat).flatten()
    preds = []
    for tid in candidatas:
        if tid in idxs: preds.append((tid, sims[idxs[tid]]))
    preds.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in preds[:K]]

def predecir_implicit(mod, dat, user_id, candidatas):
    u2idx = dat["user2idx"]; i2idx = dat["item2idx"]; idx2i = {v:k for k,v in i2idx.items()}
    if user_id not in u2idx: return []
    u_idx = u2idx[user_id]
    uf = np.asarray(mod.user_factors[u_idx]); iff = np.asarray(mod.item_factors)
    scores = uf @ iff.T
    c_set = set(candidatas)
    preds = []
    for midx, s in enumerate(scores):
        tid = idx2i[midx]
        if tid in c_set: preds.append((tid, s))
    preds.sort(key=lambda x: x[1], reverse=True)
    return [p[0] for p in preds[:K]]

# ---- EVALUACIÓN ----

def evaluar():
    print("\n  INICIANDO EVALUACIÓN DEFINITIVA 2026")
    modelos = cargar_modelos()
    df = pd.read_csv(RUTA_RATINGS, on_bad_lines="skip")
    df_cat = pd.read_csv(RUTA_CATALOGO)
    todas_pelis = set(df_cat[df_cat["vote_count"] > 100]["tmdb_id"].unique()) if "vote_count" in df_cat.columns else set(df_cat["tmdb_id"].unique())

    # Sampler: Usuarios CORE (historia rica)
    counts = df.groupby("userId").size()
    core_users = counts[counts >= 1000].index.tolist()
    u_eval = np.random.choice(core_users, min(len(core_users), NUM_USUARIOS), replace=False)
    
    # Optimizacion O(1)
    df_idx = df.set_index("userId")
    
    res = {m: {"p":0, "r":0, "n":0, "h":0, "recs":[]} for m in ["SVD","KNN","WND","TFIDF","IMP","NCF","TT"]}
    n_final = 0

    print(f"  Test loops: {len(u_eval)} users...")
    for u in u_eval:
        try:
            ud = df_idx.loc[[u]].reset_index()
            rel = ud[ud["rating"] >= UMBRAL_RELEVANTE]
            if len(rel) < 5: continue
            oculto = rel.sample(frac=0.2, random_state=42)
            vistas = ud.drop(oculto.index)
            gt_dict = {row["tmdb_id"]: row["rating"] for _, row in oculto.iterrows()}
            gt_list = list(gt_dict.keys())
            cands = todas_pelis - set(vistas["tmdb_id"].unique())
            
            # SVD
            if "SVD" in modelos:
                t = predecir_svd_knn(modelos["SVD"], u, cands)
                res["SVD"]["p"]+=precision_at_k(t, gt_list); res["SVD"]["r"]+=recall_at_k(t, gt_list); res["SVD"]["n"]+=ndcg_at_k(t, gt_dict); res["SVD"]["h"]+=hit_rate(t, gt_list); res["SVD"]["recs"].append(t)
            # KNN
            if "KNN" in modelos:
                t = predecir_svd_knn(modelos["KNN"], u, cands)
                res["KNN"]["p"]+=precision_at_k(t, gt_list); res["KNN"]["r"]+=recall_at_k(t, gt_list); res["KNN"]["n"]+=ndcg_at_k(t, gt_dict); res["KNN"]["h"]+=hit_rate(t, gt_list); res["KNN"]["recs"].append(t)
            # WND
            if "WND_ONNX" in modelos:
                t = predecir_wnd(modelos["WND_ONNX"], modelos["WND_MAPS"], u, cands)
                res["WND"]["p"]+=precision_at_k(t, gt_list); res["WND"]["r"]+=recall_at_k(t, gt_list); res["WND"]["n"]+=ndcg_at_k(t, gt_dict); res["WND"]["h"]+=hit_rate(t, gt_list); res["WND"]["recs"].append(t)
            # TFIDF
            if "TFIDF_MAT" in modelos:
                t = predecir_content(modelos["TFIDF_MAT"], modelos["TFIDF_IDX"], vistas, cands)
                res["TFIDF"]["p"]+=precision_at_k(t, gt_list); res["TFIDF"]["r"]+=recall_at_k(t, gt_list); res["TFIDF"]["n"]+=ndcg_at_k(t, gt_dict); res["TFIDF"]["h"]+=hit_rate(t, gt_list); res["TFIDF"]["recs"].append(t)
            # IMP
            if "IMP" in modelos:
                t = predecir_implicit(modelos["IMP"], modelos["IMP_DAT"], u, cands)
                res["IMP"]["p"]+=precision_at_k(t, gt_list); res["IMP"]["r"]+=recall_at_k(t, gt_list); res["IMP"]["n"]+=ndcg_at_k(t, gt_dict); res["IMP"]["h"]+=hit_rate(t, gt_list); res["IMP"]["recs"].append(t)
            # NCF
            if "NCF_ONNX" in modelos:
                t = predecir_ncf(modelos["NCF_ONNX"], modelos["NCF_U"], modelos["NCF_I"], u, cands)
                res["NCF"]["p"]+=precision_at_k(t, gt_list); res["NCF"]["r"]+=recall_at_k(t, gt_list); res["NCF"]["n"]+=ndcg_at_k(t, gt_dict); res["NCF"]["h"]+=hit_rate(t, gt_list); res["NCF"]["recs"].append(t)
            # TT
            if "TT_ONNX" in modelos:
                t = predecir_tt(modelos["TT_ONNX"], modelos["TT_MAPS"], u, cands)
                res["TT"]["p"]+=precision_at_k(t, gt_list); res["TT"]["r"]+=recall_at_k(t, gt_list); res["TT"]["n"]+=ndcg_at_k(t, gt_dict); res["TT"]["h"]+=hit_rate(t, gt_list); res["TT"]["recs"].append(t)
            
            n_final += 1
        except: continue

    print("\n  RESULTADOS FINALES:")
    records = []
    cat_n = len(todas_pelis)
    for m, v in res.items():
        if not v["recs"]: continue
        p, r, n, h, c = v["p"]/n_final, v["r"]/n_final, v["n"]/n_final, v["h"]/n_final, coverage(v["recs"], cat_n)
        print(f"  {m:<6} | Prec: {p*100:4.1f}% | NDCG: {n:.3f}")
        records.append({"Modelo":m, "Precision_10":p, "Recall_10":r, "NDCG_10":n, "Hit_Rate_10":h, "Coverage_10":c})
    
    pd.DataFrame(records).to_csv(RUTA_RESULTADOS, index=False)

if __name__ == "__main__":
    evaluar()
