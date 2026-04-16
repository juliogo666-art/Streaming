"""
upload_models.py — Sube TODOS los artefactos de modelos entrenados a HuggingFace Hub.

Escanea automáticamente las subcarpetas de artifacts/ (weights, exports, checkpoints, mappings)
y sube cada fichero manteniendo la estructura de carpetas en el repositorio remoto.

Uso:
    python -m src.scripts.upload_models
"""

from huggingface_hub import HfApi, login
from dotenv import load_dotenv
import os
import sys

# --- 1. CONFIGURACIÓN ---
load_dotenv()  # Carga variables desde .env

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO = "JulioJ777/Streaming"

# Subcarpetas de artifacts/ que queremos subir
SUBCARPETAS = ["weights", "exports", "checkpoints", "mappings"]
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")

# Extensiones de archivos que queremos subir (excluimos __init__.py y similar)
EXTENSIONES_MODELO = {".pkl", ".joblib", ".pth", ".onnx", ".json", ".data"}


def main():
    # --- 2. VALIDACIONES ---
    if not HF_TOKEN:
        print("ERROR: No se encontró HF_TOKEN en el archivo .env")
        print("Añade la línea: HF_TOKEN = hf_tu_token_aqui")
        sys.exit(1)

    artifacts_abs = os.path.abspath(ARTIFACTS_DIR)
    if not os.path.isdir(artifacts_abs):
        print(f"ERROR: No se encuentra la carpeta artifacts en: {artifacts_abs}")
        sys.exit(1)

    # --- 3. INICIAR SESIÓN ---
    print(f"Autenticando con HuggingFace Hub...")
    login(token=HF_TOKEN)
    api = HfApi()

    # --- 4. RECOPILAR ARCHIVOS A SUBIR ---
    archivos_a_subir = []  # Lista de tuplas (ruta_local, ruta_en_repo)

    for subcarpeta in SUBCARPETAS:
        dir_local = os.path.join(artifacts_abs, subcarpeta)
        if not os.path.isdir(dir_local):
            print(f"  Carpeta '{subcarpeta}/' no encontrada, saltando...")
            continue

        for nombre_archivo in os.listdir(dir_local):
            ruta_local = os.path.join(dir_local, nombre_archivo)

            # Saltar directorios y archivos no relevantes
            if os.path.isdir(ruta_local):
                continue
            if nombre_archivo.startswith("__"):
                continue

            # Comprobar extensión
            _, ext = os.path.splitext(nombre_archivo)
            if ext.lower() not in EXTENSIONES_MODELO:
                print(f"  Saltando {nombre_archivo} (extensión {ext} no incluida)")
                continue

            # La ruta en el repo mantiene la estructura: weights/modelo_1.joblib
            ruta_en_repo = f"{subcarpeta}/{nombre_archivo}"
            archivos_a_subir.append((ruta_local, ruta_en_repo))

    if not archivos_a_subir:
        print("No se encontraron archivos para subir.")
        sys.exit(0)

    # --- 5. RESUMEN PREVIO ---
    print(f"\n{'═' * 60}")
    print(f"  Repositorio destino: {HF_REPO}")
    print(f"  Archivos a subir:    {len(archivos_a_subir)}")
    total_bytes = sum(os.path.getsize(r) for r, _ in archivos_a_subir)
    total_mb = total_bytes / (1024 * 1024)
    print(f"  Tamaño total:        {total_mb:,.1f} MB")
    print(f"{'═' * 60}\n")

    # --- 6. PROCESO DE SUBIDA ---
    subidos = 0
    errores = 0

    for idx, (ruta_local, ruta_en_repo) in enumerate(archivos_a_subir, 1):
        size_mb = os.path.getsize(ruta_local) / (1024 * 1024)
        print(
            f"  [{idx}/{len(archivos_a_subir)}] Subiendo {ruta_en_repo} ({size_mb:,.1f} MB)..."
        )

        try:
            api.upload_file(
                path_or_fileobj=ruta_local,
                path_in_repo=ruta_en_repo,
                repo_id=HF_REPO,
                repo_type="model",
            )
            print(f"Subido correctamente.")
            subidos += 1
        except Exception as e:
            print(f"ERROR: {e}")
            errores += 1

    # --- 7. RESUMEN FINAL ---
    print(f"\n{'═' * 60}")
    print(f"  RESULTADO FINAL")
    print(f"  Subidos correctamente: {subidos}/{len(archivos_a_subir)}")
    if errores:
        print(f"  Con errores:           {errores}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
