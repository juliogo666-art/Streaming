"""
registrar_metricas.py
=====================
Sistema de registro histórico de métricas de los modelos de IA.

Registra cada ejecución de entrenamiento o evaluación en un CSV acumulativo,
permitiendo comparar el rendimiento de diferentes configuraciones de
hiperparámetros a lo largo del tiempo.

Uso:
    from src.utils.registrar_metricas import registrar_metricas

    registrar_metricas(
        modelo="SVD",
        hiperparams={"n_factores": 100, "n_epocas": 20, "learning_rate": 0.005},
        metricas={"MAE": 0.68, "RMSE": 0.87, "NDCG_10": 0.0953},
        dataset_size=25769933,
        train_time_s=245.3,
    )
"""

import csv
import os
from datetime import datetime

# Ruta del CSV acumulativo
RUTA_HISTORIAL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "historial_metricas.csv"
)

# Columnas completas del registro
COLUMNAS = [
    "timestamp",
    "modelo",
    # --- Hiperparámetros ---
    "n_factores",
    "n_epocas",
    "learning_rate",
    "regularizacion",
    "embedding_dim",
    "batch_size",
    "k_vecinos",
    "neg_samples",
    "min_ratings_user",
    "min_ratings_item",
    "hidden_layers",
    # --- Métricas de Error ---
    "MAE",
    "RMSE",
    # --- Métricas de Ranking ---
    "NDCG_10",
    "Precision_10",
    "Recall_10",
    "Hit_Rate_10",
    "Coverage_10",
    "MRR_10",
    # --- Rendimiento ---
    "train_time_s",
    "dataset_size",
    "notas",
]


def registrar_metricas(
    modelo: str,
    hiperparams: dict = None,
    metricas: dict = None,
    dataset_size: int = None,
    train_time_s: float = None,
    notas: str = "",
    ruta_csv: str = None,
) -> str:
    """
    Añade una fila al registro acumulativo de métricas.

    Parameters
    ----------
    modelo : str
        Nombre del modelo (ej: 'SVD', 'KNN', 'Wide&Deep', 'NCF', 'Implicit').
    hiperparams : dict
        Diccionario con los hiperparámetros usados en el entrenamiento.
        Las claves deben coincidir con las columnas definidas en COLUMNAS.
        Ejemplo: {"n_factores": 100, "n_epocas": 20, "learning_rate": 0.005}
    metricas : dict
        Diccionario con las métricas de evaluación.
        Ejemplo: {"MAE": 0.68, "RMSE": 0.87, "NDCG_10": 0.0953}
    dataset_size : int
        Número de filas del dataset usado para entrenamiento.
    train_time_s : float
        Tiempo total de entrenamiento en segundos.
    notas : str
        Notas libres sobre esta ejecución.
    ruta_csv : str
        Ruta al CSV de historial. Si es None, usa la ruta por defecto.

    Returns
    -------
    str
        Ruta absoluta del CSV donde se guardó el registro.
    """
    if ruta_csv is None:
        ruta_csv = RUTA_HISTORIAL
    if hiperparams is None:
        hiperparams = {}
    if metricas is None:
        metricas = {}

    # Construir la fila completa con "NA" para valores no proporcionados
    fila = {}
    for col in COLUMNAS:
        if col == "timestamp":
            fila[col] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elif col == "modelo":
            fila[col] = modelo
        elif col == "dataset_size":
            fila[col] = dataset_size if dataset_size is not None else "NA"
        elif col == "train_time_s":
            fila[col] = round(train_time_s, 2) if train_time_s is not None else "NA"
        elif col == "notas":
            fila[col] = notas
        elif col in hiperparams:
            fila[col] = hiperparams[col]
        elif col in metricas:
            val = metricas[col]
            fila[col] = round(val, 6) if isinstance(val, float) else val
        else:
            fila[col] = "NA"

    # Escribir al CSV (crear cabecera si no existe)
    existe = os.path.exists(ruta_csv)
    with open(ruta_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS)
        if not existe:
            writer.writeheader()
        writer.writerow(fila)

    print(f"  [Registro] Métricas de '{modelo}' guardadas en {ruta_csv}")
    return ruta_csv


def leer_historial(ruta_csv: str = None):
    """
    Lee el historial de métricas como DataFrame de pandas.

    Returns
    -------
    pd.DataFrame o None si el archivo no existe.
    """
    if ruta_csv is None:
        ruta_csv = RUTA_HISTORIAL

    if not os.path.exists(ruta_csv):
        print(f"  [Registro] No existe historial en {ruta_csv}")
        return None

    import pandas as pd
    df = pd.read_csv(ruta_csv)
    print(f"  [Registro] {len(df)} registros cargados desde {ruta_csv}")
    return df
