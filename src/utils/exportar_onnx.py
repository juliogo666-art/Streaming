"""
#######################################################################################
# SCRIPT DE EXPORTACIÓN Y OPTIMIZACIÓN DE MODELOS
# =====================================================================================
#  1. Convierte el modelo PyTorch (Wide&Deep) de .pth a .onnx para inferencia nativa.
#  2. Migra los modelos SVD, KNN y TF-IDF de Pickle (.pkl) a Joblib (.joblib)
#     para aumentar drásticamente la velocidad de carga y reducir picos de memoria.
#######################################################################################
"""

import os
import pickle
import joblib
import torch

# --- Rutas de origen (checkpoints PyTorch y mapeos) ---
RUTA_WND_PTH = "artifacts/checkpoints/modelo_3_wnd.pth"
RUTA_WND_MAP = "artifacts/mappings/wnd_mappings.pkl"

# Archivos .pkl originales que se migran a .joblib (ahora en artifacts/weights/)
RUTAS_PICKLE = [
    "artifacts/weights/modelo_1_SVD.pkl",
    "artifacts/weights/modelo_2.5_knn_msd.pkl",
    "artifacts/weights/modelo_4_tfidf.pkl",
    "artifacts/weights/modelo_4_matriz.pkl",
    "artifacts/weights/modelo_4_indices.pkl",
]

# --- Ruta destino para la exportación ONNX ---
RUTA_WND_ONNX = "artifacts/exports/modelo_3_wnd.onnx"

print("=" * 60)
print("  Iniciando Exportación y Optimización de Modelos")
print("=" * 60)


# =========================================================================
# 1. EXPORTACIÓN WIDE & DEEP -> ONNX
# =========================================================================
print("\n[1] Exportando Wide&Deep a formato ONNX...")

if os.path.exists(RUTA_WND_PTH) and os.path.exists(RUTA_WND_MAP):
    try:
        # Importamos la arquitectura de la red neuronal
        import sys

        sys.path.insert(
            0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        )
        from src.networks.dl.rn_mlp import WideAndDeepModel

        # Cargamos los mapeos para saber los tamaños num_users y num_movies
        with open(RUTA_WND_MAP, "rb") as f:
            wnd_maps = pickle.load(f)

        num_users = len(wnd_maps["user2idx"])
        num_movies = len(wnd_maps["movie2idx"])

        print(
            f"  -> Recreando red neuronal ({num_users} users, {num_movies} movies)..."
        )
        wnd = WideAndDeepModel(num_users, num_movies, 32, [64, 32])
        wnd.load_state_dict(
            torch.load(RUTA_WND_PTH, map_location="cpu", weights_only=True)
        )
        wnd.eval()  # Importante: Poner el modelo en modo Inferencia

        # ONNX necesita entradas "ficticias" (dummy inputs) para simular una pasada por la red
        # y "trazar" matemáticamente todas las operaciones internas.
        # Creamos un batch falso con 1 usuario y 1 película:
        dummy_user = torch.tensor([0], dtype=torch.long)
        dummy_item = torch.tensor([0], dtype=torch.long)

        print("  -> Ejecutando torch.onnx.export()...")
        torch.onnx.export(
            wnd,
            (dummy_user, dummy_item),  # Entradas
            RUTA_WND_ONNX,  # Salida
            export_params=True,  # Exportar con los pesos aprendidos
            opset_version=14,  # Versión estándar comprobada
            do_constant_folding=True,  # Optimizar operaciones estáticas
            dynamo=False,  # Desactivar dynamo explícitamente para evitar onnxscript bug en v2.6+
            input_names=["user_id", "movie_id"],  # Nombres amigables para la API
            output_names=["predicted_rating"],
            # Habilitamos soporte para arrays de diferente longitud
            dynamic_axes={
                "user_id": {0: "batch_size"},
                "movie_id": {0: "batch_size"},
                "predicted_rating": {0: "batch_size"},
            },
        )

        tam_onnx = os.path.getsize(RUTA_WND_ONNX) / (1024 * 1024)
        print(
            f"  Completado. Modelo ONNX generado: {RUTA_WND_ONNX} ({tam_onnx:.2f} MB)"
        )

    except Exception as e:
        print(f"  Error exportando ONNX: {e}")
else:
    print(f"  No se encontraron los archivos Pytorch en: {RUTA_WND_PTH}")


# =========================================================================
# 2. CONVERSIÓN PICKLE -> JOBLIB
# =========================================================================
print("\n[2] Migrando Pickle (.pkl) a Joblib (.joblib)...")

for archivo_pkl in RUTAS_PICKLE:
    if os.path.exists(archivo_pkl):
        # La nueva ruta será la misma pero acabada en .joblib
        archivo_joblib = archivo_pkl.replace(".pkl", ".joblib")

        if os.path.exists(archivo_joblib):
            print(
                f"  〰️ Saltando {os.path.basename(archivo_pkl)} (Ya convertido a Joblib previamente)"
            )
            continue

        print(f"  -> Procesando {os.path.basename(archivo_pkl)}...")
        try:
            # Ampliamos el límite de recursividad de Python para objetos gigantes anidados de Surprise.
            import sys

            sys.setrecursionlimit(50000)

            # 1. Leemos con pickle normal
            with open(archivo_pkl, "rb") as f:
                objeto = pickle.load(f)

            # 2. Volcamos con joblib, con compresión 3 (equilibrada)
            joblib.dump(objeto, archivo_joblib, compress=3)

            tam_pkl = os.path.getsize(archivo_pkl) / (1024 * 1024)
            tam_jb = os.path.getsize(archivo_joblib) / (1024 * 1024)
            print(
                f"     Convertido! Joblib: {tam_jb:.1f} MB (Pickle original: {tam_pkl:.1f} MB)"
            )

        except Exception as e:
            print(f"     Error procesando {os.path.basename(archivo_pkl)}: {e}")
    else:
        print(f"  〰️ Saltando {os.path.basename(archivo_pkl)} (No existe en disco)")

print("\n" + "=" * 60)
print("  Proceso completado. Si el backend utiliza .joblib, recuerda")
print("  actualizar 'import pickle' por 'import joblib' en main_api.py")
print("=" * 60)
