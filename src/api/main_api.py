"""
main_api.py — Punto de entrada de la API FastAPI (SPIRE Streaming).

Responsabilidades:
  - Ciclo de vida (lifespan): carga de modelos y datos al arrancar.
  - Registro de routers (auth, admin, recommendations, ratings, serendipia).
  - Endpoints auxiliares (status, heartbeat watchdog).
"""

from contextlib import asynccontextmanager
import logging
import time
import os
import json
import pickle

import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config.settings import (
    RUTA_RATINGS, RUTA_CATALOGO,
    RUTA_MODELO_SVD, RUTA_MODELO_KNN,
    RUTA_TFIDF_MAT, RUTA_TFIDF_IDX,
    RUTA_IMP, RUTA_IMP_DAT,
    RUTA_MODELO_WND, RUTA_WND_MAP,
    RUTA_MODELO_NCF, RUTA_NCF_USER2IDX, RUTA_NCF_ITEM2IDX,
    RUTA_MODELO_TT, RUTA_TT_MAP,
)

try:
    import onnxruntime as ort
    ONNX_DISPONIBLE = True
    print(f"[WnD] ONNX Runtime {ort.__version__} disponible.")
except ImportError as e:
    print(f"[WnD] onnxruntime no instalado: {e}")
    ONNX_DISPONIBLE = False

# --- Configuración de Logging Estructurado ---
os.makedirs("logs", exist_ok=True)
log_handler = logging.FileHandler("logs/backend.log", encoding="utf-8")
log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger = logging.getLogger("streaming_api")
logger.addHandler(log_handler)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


@asynccontextmanager
async def lifespan(the_app: FastAPI):
    """Gestiona el ciclo de vida de la aplicación: carga modelos al arrancar, limpia al cerrar."""
    t_startup_total = time.perf_counter()
    logger.info("[STARTUP] Iniciando carga de modelos de IA en memoria...")

    _bench: dict[str, float] = {}

    # --- Inicialización de estado ---
    the_app.state.modelo_svd = None
    the_app.state.df_ratings_ia = None
    the_app.state.df_catalogo = None
    the_app.state.modelo_knn = None
    the_app.state.modelo_wnd = None
    the_app.state.wnd_mappings = None
    the_app.state.modelo_tfidf_mat = None
    the_app.state.modelo_tfidf_idx = None
    the_app.state.modelo_imp = None
    the_app.state.modelo_imp_dat = None
    the_app.state.modelo_ncf = None
    the_app.state.ncf_user2idx = None
    the_app.state.ncf_item2idx = None
    the_app.state.modelo_tt = None
    the_app.state.tt_mappings = None

    # --- Modelo 1: SVD ---
    _t0 = time.perf_counter()
    if os.path.exists(RUTA_MODELO_SVD):
        try:
            import joblib
            the_app.state.modelo_svd = joblib.load(RUTA_MODELO_SVD)
            logger.info("Modelo SVD cargado correctamente (Joblib).")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo SVD: {e}.")
    _bench["Modelo 1 · SVD (Joblib)"] = time.perf_counter() - _t0

    # --- Modelo 2: KNN ---
    _t0 = time.perf_counter()
    if os.path.exists(RUTA_MODELO_KNN):
        try:
            import joblib
            the_app.state.modelo_knn = joblib.load(RUTA_MODELO_KNN)
            logger.info("Modelo KNN+Cosine cargado (Joblib).")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo KNN: {e}")
    _bench["Modelo 2 · KNN+Cosine (Joblib)"] = time.perf_counter() - _t0

    # --- Modelo 3: Wide & Deep (ONNX Runtime) ---
    _t0 = time.perf_counter()
    if os.path.exists(RUTA_MODELO_WND) and os.path.exists(RUTA_WND_MAP):
        try:
            import onnxruntime as ort
            the_app.state.modelo_wnd = ort.InferenceSession(RUTA_MODELO_WND)
            with open(RUTA_WND_MAP, "rb") as f:
                the_app.state.wnd_mappings = pickle.load(f)
            logger.info("Modelo Wide&Deep ONNX cargado correctamente.")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo Wide&Deep: {e}")
    _bench["Modelo 3 · Wide&Deep (ONNX)"] = time.perf_counter() - _t0

    # --- Modelo 4: Content-Based (TF-IDF) ---
    _t0 = time.perf_counter()
    if os.path.exists(RUTA_TFIDF_MAT) and os.path.exists(RUTA_TFIDF_IDX):
        try:
            import joblib
            the_app.state.modelo_tfidf_mat = joblib.load(RUTA_TFIDF_MAT)
            the_app.state.modelo_tfidf_idx = joblib.load(RUTA_TFIDF_IDX)
            logger.info("Modelo TF-IDF cargado correctamente (Joblib).")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo TF-IDF: {e}")
    _bench["Modelo 4 · TF-IDF Content-Based (Joblib)"] = time.perf_counter() - _t0

    # --- Modelo 5: Implicit BPR ---
    _t0 = time.perf_counter()
    if os.path.exists(RUTA_IMP) and os.path.exists(RUTA_IMP_DAT):
        try:
            with open(RUTA_IMP, "rb") as f:
                the_app.state.modelo_imp = pickle.load(f)
            with open(RUTA_IMP_DAT, "rb") as f:
                the_app.state.modelo_imp_dat = pickle.load(f)
            logger.info("Modelo Implicit cargado correctamente.")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo Implicit: {e}")
    _bench["Modelo 5 · Implicit BPR (Pickle)"] = time.perf_counter() - _t0

    # --- Modelo 6: NCF (ONNX) ---
    _t0 = time.perf_counter()
    if os.path.exists(RUTA_MODELO_NCF):
        try:
            import onnxruntime as ort
            the_app.state.modelo_ncf = ort.InferenceSession(RUTA_MODELO_NCF)
            if os.path.exists(RUTA_NCF_USER2IDX):
                with open(RUTA_NCF_USER2IDX, "r") as f:
                    the_app.state.ncf_user2idx = {int(k): v for k, v in json.load(f).items()}
            if os.path.exists(RUTA_NCF_ITEM2IDX):
                with open(RUTA_NCF_ITEM2IDX, "r") as f:
                    the_app.state.ncf_item2idx = {int(k): v for k, v in json.load(f).items()}
            logger.info("Modelo NCF ONNX cargado correctamente.")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo NCF: {e}")
    _bench["Modelo 6 · NCF (ONNX)"] = time.perf_counter() - _t0

    # --- Modelo 7: Two Towers (ONNX) ---
    _t0 = time.perf_counter()
    if os.path.exists(RUTA_MODELO_TT) and os.path.exists(RUTA_TT_MAP):
        try:
            import onnxruntime as ort
            the_app.state.modelo_tt = ort.InferenceSession(RUTA_MODELO_TT)
            with open(RUTA_TT_MAP, "rb") as f:
                the_app.state.tt_mappings = pickle.load(f)
            logger.info("Modelo TwoTowers ONNX cargado correctamente.")
        except Exception as e:
            logger.error(f"No se pudo cargar el modelo TwoTowers: {e}")
    _bench["Modelo 7 · TwoTowers (ONNX)"] = time.perf_counter() - _t0

    # --- CARGA DE DATOS (CSV) ---
    _t0 = time.perf_counter()
    try:
        if os.path.exists(RUTA_RATINGS):
            the_app.state.df_ratings_ia = pd.read_csv(RUTA_RATINGS)
            logger.info(f"Ratings cargados: {len(the_app.state.df_ratings_ia):,} filas.")
            the_app.state.user_counts = (
                the_app.state.df_ratings_ia.groupby("userId").size().to_dict()
            )
    except Exception as e:
        logger.error(f"Error al cargar CSV de ratings: {e}")
    _bench["CSV · ratings_finales_ia (~434MB)"] = time.perf_counter() - _t0

    _t0 = time.perf_counter()
    try:
        if os.path.exists(RUTA_CATALOGO):
            the_app.state.df_catalogo = pd.read_csv(RUTA_CATALOGO)
            logger.info(f"Catálogo cargado: {len(the_app.state.df_catalogo):,} películas.")
    except Exception as e:
        logger.error(f"Error al cargar CSV de catálogo: {e}")
    _bench["CSV · dataset_final_movies"] = time.perf_counter() - _t0

    # ── BENCHMARK RESUMEN ─────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_startup_total
    sep = "─" * 55
    logger.info(f"[BENCHMARK] {sep}")
    for nombre, segundos in _bench.items():
        alerta = "LENTO" if segundos > 5.0 else ""
        logger.info(f"[BENCHMARK]   {nombre:<42} {segundos:>6.2f}s{alerta}")
    logger.info(f"[BENCHMARK] {sep}")
    logger.info(f"[BENCHMARK]   {'TOTAL ARRANQUE':<42} {t_total:>6.2f}s")
    logger.info(f"[BENCHMARK] {sep}")

    # Registrar tiempo de arranque en el colector de rendimiento
    from ..tracking.performance import colector as perf_colector
    perf_colector.tiempo_arranque_s = t_total

    logger.info("[STARTUP] Sistema listo para servir peticiones.")
    yield  # La app está corriendo
    logger.info("[SHUTDOWN] Cerrando aplicación...")


# ── Creación de la App y registro de Routers ──────────────────────────────────

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware de monitorización de rendimiento
from ..tracking.performance import PerformanceMiddleware
app.add_middleware(PerformanceMiddleware)

# Importar y registrar routers
from .routers.auth import router as auth_router
from .routers.admin import router as admin_router
from .routers.recommendations import router as recommendations_router
from .routers.ratings import router as ratings_router
from .routers.serendipia import router as serendipia_router

app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(recommendations_router)
app.include_router(ratings_router)
app.include_router(serendipia_router)


# ── Endpoints auxiliares ──────────────────────────────────────────────────────

@app.get("/status")
def check_status():
    return {"status": "Backend funcionando correctamente"}


# --- Heartbeat Watchdog ---
_heartbeat_iniciado = False
_tiempo_ultimo_heartbeat = time.time()


@app.post("/api/heartbeat")
def heartbeat():
    global _heartbeat_iniciado, _tiempo_ultimo_heartbeat
    _heartbeat_iniciado = True
    _tiempo_ultimo_heartbeat = time.time()
    return {"status": "ok"}


@app.get("/api/heartbeat_status")
def heartbeat_status():
    if not _heartbeat_iniciado:
        return {"seconds_since_last": 0}
    return {"seconds_since_last": time.time() - _tiempo_ultimo_heartbeat}


# --- Endpoint de Rendimiento ---
@app.get("/api/performance")
def get_performance():
    """Devuelve métricas de rendimiento en vivo (latencia, RAM, CPU, arranque)."""
    from ..tracking.performance import colector as perf_colector
    return perf_colector.exportar_todo()
