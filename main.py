import subprocess
import time
import os
import sys


def verificar_modelos():
    """
    Comprueba si los modelos entrenados están presentes en artifacts/.
    Si faltan, los descarga automáticamente desde HuggingFace Hub.
    """
    from src.utils.download_models import (
        obtener_archivos_faltantes,
        verificar_y_descargar,
    )

    faltantes = obtener_archivos_faltantes()
    if not faltantes:
        print("Todos los modelos están presentes localmente.")
        return True

    print(f"\n{'═' * 60}")
    print(f"Faltan {len(faltantes)} archivos de modelos.")
    print(f"Descargando desde HuggingFace Hub...")
    print(f"{'═' * 60}\n")

    resultado = verificar_y_descargar(
        callback_progreso=lambda nombre, actual, total: print(
            f"[{actual}/{total}] Descargando {nombre}..."
        )
    )

    if resultado["completo"]:
        print(f"\nTodos los modelos descargados correctamente.")
        return True
    else:
        print(f"\nAlgunos modelos no se pudieron descargar:")
        for err in resultado["errores"]:
            print(f"    - {err}")
        print("La API arrancará con los modelos disponibles.")
        return False


def main():
    # Obtener el root
    ruta_raiz = os.getcwd()
    env_config = os.environ.copy()
    env_config["PYTHONPATH"] = ruta_raiz

    # 0. Verificar y descargar modelos si faltan
    verificar_modelos()

    # 1. Lanzar el Backend (FastAPI)
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main_api:app", "--reload"],
        env=env_config,
    )

    # 2. Esperar un par de segundos a que el backend suba
    time.sleep(2)

    # 3. Lanzar el Frontend (Streamlit)
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/frontend/app_ui.py"],
        env=env_config,
    )

    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        backend.terminate()
        frontend.terminate()


def main_debug():
    # Obtenemos la ruta raíz del proyecto
    ruta_raiz = os.getcwd()

    # Preparar el entorno para que los subprocesos vean el proyecto completo
    env_config = os.environ.copy()
    env_config["PYTHONPATH"] = ruta_raiz

    # 0. Verificar y descargar modelos si faltan
    verificar_modelos()

    print("Iniciando Debug de Backend...")
    # Usar el root como base para que los imports relativos de 'src' funcionen
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main_api:app", "--reload"],
        env=env_config,
    )

    time.sleep(2)

    print("Iniciando Debug de Frontend...")
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/frontend/app_ui.py"],
        env=env_config,
    )

    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        backend.terminate()
        frontend.terminate()


if __name__ == "__main__":
    # main()
    main_debug()
