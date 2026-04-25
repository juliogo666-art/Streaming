"""
settings.py — Configuración centralizada del proyecto SPIRE Streaming.

Todas las constantes, rutas y URLs que se usan en múltiples módulos
deben declararse aquí para evitar hardcodear valores dispersos.
"""

import os

# --- API ---
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

# --- Rutas de Datos (CSV) ---
RUTA_RATINGS = "src/data/ready/ratings_finales_ia.csv"
RUTA_CATALOGO = "src/data/ready/dataset_final_movies.csv"

# --- Rutas de Modelos: Weights (clásicos) ---
RUTA_MODELO_SVD = "artifacts/weights/modelo_1_SVD.joblib"
RUTA_MODELO_KNN = "artifacts/weights/modelo_2.5_knn_msd.joblib"
RUTA_TFIDF_MAT = "artifacts/weights/modelo_4_matriz.joblib"
RUTA_TFIDF_IDX = "artifacts/weights/modelo_4_indices.joblib"
RUTA_IMP = "artifacts/weights/modelo_5_implicit.pkl"
RUTA_IMP_DAT = "artifacts/weights/modelo_5_implicit_dataset.pkl"

# --- Rutas de Modelos: Exports (ONNX) ---
RUTA_MODELO_WND = "artifacts/exports/modelo_3_wnd.onnx"
RUTA_MODELO_NCF = "artifacts/exports/modelo_6_ncf.onnx"
RUTA_MODELO_TT = "artifacts/exports/modelo_7_twotowers.onnx"

# --- Rutas de Modelos: Mapeos ---
RUTA_WND_MAP = "artifacts/mappings/wnd_mappings.pkl"
RUTA_NCF_USER2IDX = "artifacts/mappings/ncf_user2idx.json"
RUTA_NCF_ITEM2IDX = "artifacts/mappings/ncf_item2idx.json"
RUTA_TT_MAP = "artifacts/mappings/twotowers_mappings.pkl"

# --- Smart Selector: Umbrales ---
UMBRAL_COLD_START = 10   # 0-10 valoraciones → Content-Based / Popularidad
UMBRAL_AVANZADO = 100    # 100+ → NCF o Wide&Deep
