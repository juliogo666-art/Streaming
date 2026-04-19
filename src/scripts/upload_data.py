"""
Sube toda la carpeta data a HuggingFace Hub.

Escanea automáticamente las subcarpetas de data y sube cada fichero manteniendo la estructura de carpetas en el repositorio remoto.

Uso:
    python -m src.scripts.upload_data
"""

from huggingface_hub import HfApi, login
from dotenv import load_dotenv
import os
import sys

# --- 1. CONFIGURACIÓN ---
load_dotenv()  # Carga variables desde .env

HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO = "JulioJ777/Streaming"

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

def main():
    # --- 2. VALIDACIONES ---
    if not HF_TOKEN:
        print("ERROR: No se encontró HF_TOKEN en el archivo .env")
        print("Añade la línea: HF_TOKEN = hf_tu_token_aqui")
        sys.exit(1)

    data_abs = os.path.abspath(DATA_DIR)
    if not os.path.isdir(data_abs):
        print(f"ERROR: No se encuentra la carpeta data en: {data_abs}")
        sys.exit(1)

    # --- 3. INICIAR SESIÓN ---
    print(f"Autenticando con HuggingFace Hub...")
    login(token=HF_TOKEN)
    api = HfApi()

    # --- 4. RECOPILAR ARCHIVOS A SUBIR ---
    archivos_a_subir = []

    # Recorremos todas las subcarpetas dentro de src/data
    for root, dirs, files in os.walk(data_abs):
        # Evitar subir carpetas de caché u otras auto-generadas
        if "__pycache__" in root or ".pytest_cache" in root:
            continue
            
        for nombre_archivo in files:
            # Saltar archivos ocultos o no relevantes
            if nombre_archivo.startswith(".") or nombre_archivo.startswith("__"):
                continue
                
            ruta_local = os.path.join(root, nombre_archivo)
            
            # Calculamos la ruta relativa para mantener la estructura en el repo
            # Ejemplo de "data_abs/clean/archivo.csv" -> "clean/archivo.csv"
            ruta_relativa = os.path.relpath(ruta_local, data_abs)
            
            # Normalizamos la barra para prevenir problemas en Windows
            ruta_relativa = ruta_relativa.replace("\\", "/")
            
            # Guardaremos todos los archivos bajo el prefijo "data/" en HF
            # para diferenciar de la subida de modelos.
            ruta_en_repo = f"data/{ruta_relativa}"
            
            archivos_a_subir.append((ruta_local, ruta_en_repo))

    if not archivos_a_subir:
        print("No se encontraron archivos para subir en la carpeta data.")
        sys.exit(0)

    # --- 5. RESUMEN PREVIO ---
    print(f"\n{'═' * 60}")
    print(f"  Repositorio destino: {HF_REPO}")
    print(f"  Carpeta origen:      {data_abs}")
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
        print(f"  [{idx}/{len(archivos_a_subir)}] Subiendo {ruta_en_repo} ({size_mb:,.1f} MB)...")

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
