"""
=============================================================================
ARCHIVO: evaluation_pipeline.py
=============================================================================
Esta clase orquesta todas las métricas definidas en src/metrics/ para
evaluar varios modelos de IA sin duplicar cálculos o código.

Se le inyecta una lista de métricas (MetricProtocol) al inicializarse.
Luego, para cada modelo, se le llama a `evaluate_model(...)`.

Desde la v2 (Abril 2026), incluye integración automática con el CSV
de historial para que cada evaluación quede registrada centralizadamente.
=============================================================================
"""

import pandas as pd
from typing import Any, Dict, List, Optional
from src.metrics.protocols import MetricProtocol
from src.utils.registrar_metricas import registrar_metricas


# Mapeo de nombres de clases de métricas -> columnas del CSV de historial.
# Esto conecta el mundo OOP (clases) con el sistema de tracking (CSV plano).
NOMBRE_METRICA_A_COLUMNA_CSV: Dict[str, str] = {
    "PrecisionAtK": "Precision_10",
    "RecallAtK": "Recall_10",
    "HitRateAtK": "Hit_Rate_10",
    "NDCGAtK": "NDCG_10",
    "CoverageAtK": "Coverage_10",
    "MRRAtK": "MRR_10",
}


class EvaluationPipeline:
    def __init__(self, metrics: List[MetricProtocol]):
        """
        Inicializa el pipeline con una lista de métricas que cumplen el protocolo.
        """
        self.metrics = metrics
        self.results: Dict[str, Dict[str, float]] = {}

    def evaluate_model(
        self,
        model_name: str,
        recommendations: dict[Any, list[Any]],
        ground_truth: pd.DataFrame,
        k: int = 10,
    ) -> dict[str, float]:
        """
        Evalúa un modelo con las métricas cargadas.

        Parámetros
        ----------
        model_name : str
            El nombre del modelo (ej: 'SVD', 'KNN')
        recommendations : dict
            Diccionario {userId: [lista de tmdb_id_recomendadas]}
        ground_truth : pd.DataFrame
            DataFrame con el subconjunto oculto (el examen de los usuarios).
            Debe tener columnas de id de usuario, id de peli y, opcional, rating.
        k : int
            Límite de corte (Top K).

        Devuelve
        -------
        Un diccionario con la puntuación de cada métrica evaludada.
        """
        resultados_del_modelo: Dict[str, float] = {}

        # Ejecutamos la regla del protocolo (OOP puro)
        for metric in self.metrics:
            nombre_metrica = metric.__class__.__name__
            try:
                puntuacion = metric.compute(recommendations, ground_truth, k)
                resultados_del_modelo[nombre_metrica] = puntuacion
            except Exception as e:
                print(f"[ALERTA] Fallo al computar {nombre_metrica} en {model_name}: {e}")
                resultados_del_modelo[nombre_metrica] = 0.0

        self.results[model_name] = resultados_del_modelo
        return resultados_del_modelo

    def registrar_resultados_en_csv(
        self,
        nombre_modelo: str,
        hiperparametros: Optional[dict] = None,
        tamano_dataset: Optional[int] = None,
        tiempo_entrenamiento_s: Optional[float] = None,
        notas: str = "",
    ) -> None:
        """
        Registra los resultados de la última evaluación de un modelo
        en el archivo CSV centralizado (historial_metricas.csv).

        Este método conecta el pipeline de evaluación OOP con el sistema de
        tracking histórico, para que no haya métricas que se calculen pero
        no se guarden.

        Parámetros
        ----------
        nombre_modelo : str
            Nombre del modelo tal como aparece en self.results.
        hiperparametros : dict, optional
            Diccionario con los hiperparámetros usados (ej: {"n_factores": 100}).
        tamano_dataset : int, optional
            Número de filas del dataset de entrenamiento.
        tiempo_entrenamiento_s : float, optional
            Duración del entrenamiento en segundos.
        notas : str
            Notas adicionales sobre esta ejecución.
        """
        if nombre_modelo not in self.results:
            print(f"[ALERTA] No hay resultados para '{nombre_modelo}'. Evalúa primero.")
            return

        # Convertimos los nombres de clases OOP a columnas del CSV
        metricas_para_csv: Dict[str, float] = {}
        for nombre_clase, valor in self.results[nombre_modelo].items():
            columna_csv = NOMBRE_METRICA_A_COLUMNA_CSV.get(nombre_clase, nombre_clase)
            metricas_para_csv[columna_csv] = valor

        # Delegamos al sistema de registro centralizado
        registrar_metricas(
            modelo=nombre_modelo,
            hiperparams=hiperparametros or {},
            metricas=metricas_para_csv,
            dataset_size=tamano_dataset,
            train_time_s=tiempo_entrenamiento_s,
            notas=notas,
        )

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Exporta todos los resultados de los modelos evaluados en un DataFrame limpio.
        """
        registros = []
        for nombre_modelo, metricas_dict in self.results.items():
            registro = {"Modelo": nombre_modelo}
            registro.update(metricas_dict)
            registros.append(registro)

        return pd.DataFrame(registros)
