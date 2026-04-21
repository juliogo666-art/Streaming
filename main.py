import subprocess
import time
import os
import sys

from src.utils.download_models import obtener_archivos_faltantes, verificar_y_descargar
from src.utils.download_data import (
    obtener_archivos_faltantes as obtener_datos_faltantes,
    verificar_y_descargar_datos,
)


########################################################################################
# Verificación y descarga de modelos
########################################################################################


def verificar_modelos():
    """
    Comprueba si los modelos entrenados están presentes en artifacts/.
    Si faltan, los descarga automáticamente desde HuggingFace Hub.
    """

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


########################################################################################
# Verificación y descarga de datos
########################################################################################


def verificar_datos():
    """
    Comprueba si los datos procesados están presentes en src/data/.
    Si faltan, los descarga automáticamente desde HuggingFace Hub.
    """

    faltantes = obtener_datos_faltantes()
    if not faltantes:
        print("Todos los archivos de datos están presentes localmente.")
        return True

    print(f"\n{'═' * 60}")
    print(f"Faltan {len(faltantes)} archivos de datos (CSVs).")
    print(f"Descargando desde HuggingFace Hub (esto puede tardar según tu conexión)...")
    print(f"{'═' * 60}\n")

    resultado = verificar_y_descargar_datos(
        callback_progreso=lambda nombre, actual, total: print(
            f"[{actual}/{total}] Descargando {nombre}..."
        )
    )

    if resultado["completo"]:
        print(f"\nTodos los datos descargados correctamente.")
        return True
    else:
        print(f"\nAlgunos datos no se pudieron descargar:")
        for err in resultado["errores"]:
            print(f"    - {err}")
        print(
            "La API arrancará con los datos disponibles (puede fallar si faltan archivos críticos)."
        )
        return False


########################################################################################
# Main
########################################################################################


def main():
    # Obtener el root
    ruta_raiz = os.getcwd()
    env_config = os.environ.copy()
    env_config["PYTHONPATH"] = ruta_raiz

    # 0. Verificar y descargar modelos si faltan
    verificar_modelos()

    # 0.5 Verificar y descargar datos si faltan
    verificar_datos()

    # 1. Lanzar el Backend (FastAPI)
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main_api:app"],
        env=env_config,
    )

    # 2. Esperar un par de segundos a que el backend suba
    time.sleep(3)
    if backend.poll() is not None:
        print("El backend terminó al arrancar. Revisa el error mostrado en consola.")
        return

    # 3. Lanzar el Frontend (Streamlit)
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/frontend/app_ui.py"],
        env=env_config,
    )

    try:
        import requests
        while True:
            if backend.poll() is not None or frontend.poll() is not None:
                break
            
            try:
                r = requests.get("http://127.0.0.1:8000/api/heartbeat_status", timeout=2)
                if r.status_code == 200 and r.json().get("seconds_since_last", 0) > 15:
                    print("\n[Watchdog] El navegador se ha cerrado (Timeout). Terminando procesos...")
                    break
            except Exception:
                pass
            
            time.sleep(3)
    except KeyboardInterrupt:
        pass
    finally:
        print("Cerrando procesos...")
        backend.terminate()
        frontend.terminate()


########################################################################################
# Main Debug
########################################################################################


def main_debug():
    # Obtenemos la ruta raíz del proyecto
    ruta_raiz = os.getcwd()

    # Preparar el entorno para que los subprocesos vean el proyecto completo
    env_config = os.environ.copy()
    env_config["PYTHONPATH"] = ruta_raiz

    # 0. Verificar y descargar modelos si faltan
    verificar_modelos()

    # 0.5 Verificar y descargar datos si faltan
    verificar_datos()

    print("Iniciando Debug de Backend...")
    # Usar el root como base para que los imports relativos de 'src' funcionen
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main_api:app"],
        env=env_config,
    )

    time.sleep(3)
    if backend.poll() is not None:
        print("El backend terminó al arrancar. Revisa el error mostrado en consola.")
        return

    print("Iniciando Debug de Frontend...")
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "src/frontend/app_ui.py"],
        env=env_config,
    )

    try:
        import requests
        while True:
            if backend.poll() is not None or frontend.poll() is not None:
                break
            
            try:
                r = requests.get("http://127.0.0.1:8000/api/heartbeat_status", timeout=2)
                if r.status_code == 200 and r.json().get("seconds_since_last", 0) > 15:
                    print("\n[Watchdog] El navegador se ha cerrado (Timeout). Terminando procesos...")
                    break
            except Exception:
                pass
            
            time.sleep(3)
    except KeyboardInterrupt:
        pass
    finally:
        print("Cerrando procesos...")
        backend.terminate()
        frontend.terminate()


########################################################################################
# Verificación y descarga de modelos
########################################################################################

if __name__ == "__main__":
    # main()
    main_debug()
