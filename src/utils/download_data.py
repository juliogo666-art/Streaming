"""
Descarga datos desde HuggingFace Hub.

Comprueba qué archivos faltan en data/ y los descárgalos del repositorio
público JulioJ777/Streaming. No requiere token para lectura.

Puede usarse:
  1. Como módulo importable: from src.utils.download_data import verificar_y_descargar_datos
  2. Como script directo:    python -m src.utils.download_data

"""

import os
import logging
from typing import Callable

logger = logging.getLogger("streaming_api")

# --- CONFIGURACIÓN ---
HF_REPO = "JulioJ777/Streaming"

# Directorio raíz del proyecto (2 niveles arriba de src/utils/)
PROYECTO_RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# El directorio base local va a ser src/ para alinear "data/..." de HF con "src/data/..." local.
SRC_DIR = os.path.join(PROYECTO_RAIZ, "src")

# Mapa de archivos esperados: ruta_en_repo -> ruta_local_relativa_a_src
# En el repositorio HF, los archivos están subidos bajo la carpeta "data/...".
ARCHIVOS_REQUERIDOS = {
    # --- Ready ---
    "data/ready/dataset_final_movies.csv": "data/ready/dataset_final_movies.csv",
    "data/ready/dataset_final_shows.csv": "data/ready/dataset_final_shows.csv",
    "data/ready/ratings_finales_ia.csv": "data/ready/ratings_finales_ia.csv",
    
    # --- Clean ---
    "data/clean/links_limpio.csv": "data/clean/links_limpio.csv",
    "data/clean/ratings_limpio.csv": "data/clean/ratings_limpio.csv",
    "data/clean/tmdb_movies_limpio.csv": "data/clean/tmdb_movies_limpio.csv",
    "data/clean/tmdb_shows_limpio.csv": "data/clean/tmdb_shows_limpio.csv",
    "data/clean/trakt_movies_limpio.csv": "data/clean/trakt_movies_limpio.csv",
    "data/clean/trakt_shows_limpio.csv": "data/clean/trakt_shows_limpio.csv",
}


def obtener_archivos_faltantes() -> list[str]:
    """Devuelve la lista de claves (rutas en repo) de archivos que faltan localmente."""
    faltantes = []
    for ruta_repo, ruta_local_rel in ARCHIVOS_REQUERIDOS.items():
        ruta_absoluta = os.path.join(SRC_DIR, ruta_local_rel)
        if not os.path.exists(ruta_absoluta):
            faltantes.append(ruta_repo)
    return faltantes


def verificar_y_descargar_datos(
    callback_progreso: Callable[[str, int, int], None] | None = None,
) -> dict:
    """
    Comprueba qué archivos de datos faltan y los descarga de HuggingFace.

    Returns: dict con el estado de las descargas ("faltantes", "descargados", "errores", "completo").
    """
    faltantes = obtener_archivos_faltantes()

    resultado = {
        "faltantes": len(faltantes),
        "descargados": 0,
        "errores": [],
        "completo": len(faltantes) == 0,
    }

    if not faltantes:
        logger.info("[DATA] Todos los archivos de datos están presentes localmente. ✓")
        return resultado

    logger.info(f"[DATA] Faltan {len(faltantes)} archivos. Iniciando descarga desde HuggingFace...")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        msg = "huggingface_hub no está instalado. Ejecuta: pip install huggingface_hub"
        logger.error(f"[DATA] {msg}")
        resultado["errores"].append(msg)
        return resultado

    for idx, ruta_repo in enumerate(faltantes, 1):
        ruta_local_rel = ARCHIVOS_REQUERIDOS[ruta_repo]
        ruta_absoluta = os.path.join(SRC_DIR, ruta_local_rel)

        # Asegurar que la subcarpeta existe
        os.makedirs(os.path.dirname(ruta_absoluta), exist_ok=True)

        nombre_corto = os.path.basename(ruta_repo)
        logger.info(f"[DATA]   [{idx}/{len(faltantes)}] Descargando {ruta_repo}...")

        if callback_progreso:
            callback_progreso(nombre_corto, idx, len(faltantes))

        try:
            # hf_hub_download descarga a cache y devuelve la ruta al fichero cacheado
            ruta_cache = hf_hub_download(
                repo_id=HF_REPO,
                filename=ruta_repo,
                repo_type="model",              # Asumimos que también están en el workspace de modelo (default de la librería)
                local_dir=SRC_DIR,              # Descargando a src/ para que 'data/...' se sitúe correctamente
                local_dir_use_symlinks=False,   # Copia real, no symlinks (necesario en Windows)
            )
            resultado["descargados"] += 1
            logger.info(f"[DATA]           ✓ {nombre_corto} descargado.")

        except Exception as e:
            msg = f"Error descargando {ruta_repo}: {e}"
            logger.error(f"[DATA]           ✗ {msg}")
            resultado["errores"].append(msg)

    resultado["completo"] = len(resultado["errores"]) == 0
    logger.info(
        f"[DATA] Descarga finalizada: {resultado['descargados']}/{len(faltantes)} "
        f"exitosos, {len(resultado['errores'])} errores."
    )
    return resultado


# --- Ejecución directa como script ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 60)
    print("  Verificador y Descargador de Datos")
    print(f"  Repositorio: {HF_REPO}")
    print(f"  Destino:     {SRC_DIR}/data")
    print("=" * 60)

    faltantes = obtener_archivos_faltantes()
    if not faltantes:
        print("\n✓ Todos los archivos de datos están presentes. No hay nada que descargar.")
    else:
        print(f"\n⚠ Faltan {len(faltantes)} archivos:")
        for f in faltantes:
            print(f"  - {f}")
        print()

        respuesta = input("¿Descargar ahora (esto puede tardar debido al peso de los archivos)? (s/n): ").strip().lower()
        if respuesta == "s":
            result = verificar_y_descargar_datos(
                callback_progreso=lambda nombre, actual, total: print(
                    f"  Progreso: {actual}/{total} - {nombre}"
                )
            )
            print(f"\nResultado: {result}")
        else:
            print("Descarga cancelada.")
