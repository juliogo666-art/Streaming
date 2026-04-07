import subprocess
import time
import os
import sys


def main():
    # Obtener el root
    ruta_raiz = os.getcwd()
    env_config = os.environ.copy()
    env_config["PYTHONPATH"] = ruta_raiz

    # 1. Lanzar el Backend (FastAPI)
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main_api:app", "--reload"],
        env=env_config
    )

    # 2. Esperar un par de segundos a que el backend suba
    time.sleep(2)

    # 3. Lanzar el Frontend (Streamlit)
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/frontend/app_ui.py"],
        env=env_config
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
