"""
download_models.py — Descarga modelos entrenados desde HuggingFace Hub.

Comprueba qué archivos faltan en artifacts/ y los descarga del repositorio
público JulioJ777/Streaming. No requiere token (repo público).

Puede usarse:
  1. Como módulo importable: from src.utils.download_models import verificar_y_descargar
  2. Como script directo:    python -m src.utils.download_models

La función `verificar_y_descargar()` devuelve un dict con el estado de cada archivo
para que el frontend pueda mostrar progreso.
"""

import os
import logging
from typing import Callable

logger = logging.getLogger("streaming_api")

# --- CONFIGURACIÓN ---
HF_REPO = "JulioJ777/Streaming"

# Directorio raíz del proyecto (2 niveles arriba de src/utils/)
PROYECTO_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ARTIFACTS_DIR = os.path.join(PROYECTO_RAIZ, "artifacts")

# Mapa de archivos esperados: ruta_en_repo -> ruta_local_relativa_a_artifacts
# Esto define exactamente qué archivos necesitamos para que la API funcione.
ARCHIVOS_REQUERIDOS = {
    # --- Weights (modelos clásicos) ---
    "weights/modelo_1_SVD.joblib": "weights/modelo_1_SVD.joblib",
    "weights/modelo_2.5_knn_msd.joblib": "weights/modelo_2.5_knn_msd.joblib",
    "weights/modelo_4_indices.joblib": "weights/modelo_4_indices.joblib",
    "weights/modelo_4_matriz.joblib": "weights/modelo_4_matriz.joblib",
    "weights/modelo_4_tfidf.joblib": "weights/modelo_4_tfidf.joblib",
    "weights/modelo_5_implicit.pkl": "weights/modelo_5_implicit.pkl",
    "weights/modelo_5_implicit_dataset.pkl": "weights/modelo_5_implicit_dataset.pkl",
    # --- Exports (ONNX) ---
    "exports/modelo_3_wnd.onnx": "exports/modelo_3_wnd.onnx",
    "exports/modelo_3_wnd.onnx.data": "exports/modelo_3_wnd.onnx.data",
    "exports/modelo_6_ncf.onnx": "exports/modelo_6_ncf.onnx",
    "exports/modelo_6_ncf.onnx.data": "exports/modelo_6_ncf.onnx.data",
    "exports/modelo_7_twotowers.onnx": "exports/modelo_7_twotowers.onnx",
    "exports/modelo_7_twotowers.onnx.data": "exports/modelo_7_twotowers.onnx.data",
    # --- Mappings ---
    "mappings/wnd_mappings.pkl": "mappings/wnd_mappings.pkl",
    "mappings/ncf_user2idx.json": "mappings/ncf_user2idx.json",
    "mappings/ncf_item2idx.json": "mappings/ncf_item2idx.json",
    "mappings/twotowers_mappings.pkl": "mappings/twotowers_mappings.pkl",
    # --- Checkpoints (PyTorch) ---
    "checkpoints/modelo_3_wnd.pth": "checkpoints/modelo_3_wnd.pth",
    "checkpoints/modelo_7_twotowers.pth": "checkpoints/modelo_7_twotowers.pth",
}


def obtener_archivos_faltantes() -> list[str]:
    """Devuelve la lista de claves (rutas en repo) de archivos que faltan localmente."""
    faltantes = []
    for ruta_repo, ruta_local_rel in ARCHIVOS_REQUERIDOS.items():
        ruta_absoluta = os.path.join(ARTIFACTS_DIR, ruta_local_rel)
        if not os.path.exists(ruta_absoluta):
            faltantes.append(ruta_repo)
    return faltantes


def verificar_y_descargar(
    callback_progreso: Callable[[str, int, int], None] | None = None,
) -> dict:
    """
    Comprueba qué modelos faltan y los descarga de HuggingFace.

    Args:
        callback_progreso: Función opcional (nombre_archivo, actual, total) que se
                           llama tras cada descarga para informar al frontend.

    Returns:
        dict con claves:
            - "faltantes": int, número de archivos que faltaban
            - "descargados": int, número descargados con éxito
            - "errores": list[str], descripciones de errores si los hubo
            - "completo": bool, True si todo está OK
    """
    faltantes = obtener_archivos_faltantes()

    resultado = {
        "faltantes": len(faltantes),
        "descargados": 0,
        "errores": [],
        "completo": len(faltantes) == 0,
    }

    if not faltantes:
        logger.info("[MODELS] Todos los modelos están presentes localmente. ✓")
        return resultado

    logger.info(f"[MODELS] Faltan {len(faltantes)} archivos. Iniciando descarga desde HuggingFace...")

    # Importamos aquí para no forzar la dependencia si los modelos ya existen
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        msg = "huggingface_hub no está instalado. Ejecuta: pip install huggingface_hub"
        logger.error(f"[MODELS] {msg}")
        resultado["errores"].append(msg)
        return resultado

    for idx, ruta_repo in enumerate(faltantes, 1):
        ruta_local_rel = ARCHIVOS_REQUERIDOS[ruta_repo]
        ruta_absoluta = os.path.join(ARTIFACTS_DIR, ruta_local_rel)

        # Asegurar que la subcarpeta existe
        os.makedirs(os.path.dirname(ruta_absoluta), exist_ok=True)

        nombre_corto = os.path.basename(ruta_repo)
        logger.info(f"[MODELS]   [{idx}/{len(faltantes)}] Descargando {ruta_repo}...")

        if callback_progreso:
            callback_progreso(nombre_corto, idx, len(faltantes))

        try:
            # hf_hub_download descarga a cache y devuelve la ruta del fichero cacheado
            ruta_cache = hf_hub_download(
                repo_id=HF_REPO,
                filename=ruta_repo,
                repo_type="model",
                local_dir=ARTIFACTS_DIR,        # Descarga directamente a artifacts/
                local_dir_use_symlinks=False,    # Copia real, no symlinks (Windows)
            )
            resultado["descargados"] += 1
            logger.info(f"[MODELS]           ✓ {nombre_corto} descargado.")

        except Exception as e:
            msg = f"Error descargando {ruta_repo}: {e}"
            logger.error(f"[MODELS]           ✗ {msg}")
            resultado["errores"].append(msg)

    resultado["completo"] = len(resultado["errores"]) == 0
    logger.info(
        f"[MODELS] Descarga finalizada: {resultado['descargados']}/{len(faltantes)} "
        f"exitosos, {len(resultado['errores'])} errores."
    )
    return resultado


# --- Ejecución directa como script ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 60)
    print("  Verificador y Descargador de Modelos")
    print(f"  Repositorio: {HF_REPO}")
    print(f"  Destino:     {ARTIFACTS_DIR}")
    print("=" * 60)

    faltantes = obtener_archivos_faltantes()
    if not faltantes:
        print("\n✓ Todos los modelos están presentes. No hay nada que descargar.")
    else:
        print(f"\n⚠ Faltan {len(faltantes)} archivos:")
        for f in faltantes:
            print(f"  - {f}")
        print()

        respuesta = input("¿Descargar ahora? (s/n): ").strip().lower()
        if respuesta == "s":
            result = verificar_y_descargar(
                callback_progreso=lambda nombre, actual, total: print(
                    f"  Progreso: {actual}/{total} - {nombre}"
                )
            )
            print(f"\nResultado: {result}")
        else:
            print("Descarga cancelada.")
