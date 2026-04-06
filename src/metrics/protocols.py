"""
=============================================================================
ARCHIVO: protocols.py (Interfaces / Contratos de OOP)
=============================================================================
¿QUÉ HACE ESTE ARCHIVO?:
Define un "Contrato" (Protocol) para las métricas.
No guarda hiperparámetros ni datos experimentales. Su única misión es obligar a
cualquier clase que quiera considerarse una "Métrica" (como Precision, NDCG, etc.)
a implementar exactamente el método `compute()` con los parámetros definidos
abajo (recomendaciones, ground_truth y k).

¿POR QUÉ ES ÚTIL?:
Al usar este molde, logramos que el archivo de evaluación pipeline pueda iterar
sobre una lista de métricas distintas de forma genérica:
    for metrica in mis_metricas:
        resultado = metrica.compute(recs, datos_reales, 10)

Sin este contrato, cada métrica tendría nombres de funciones distintos y
no se podría automatizar su cálculo ("duck typing" seguro).
=============================================================================
"""

from typing import Any, Protocol
import pandas as pd


class MetricProtocol(Protocol):
    """Interfaz maestra/Plantilla para todas las métricas de evaluación."""

    def compute(
        self,
        recommendations: dict[Any, list[Any]],
        ground_truth: pd.DataFrame,
        k: int,
    ) -> float:
        """
        Interfaz de computación obligatoria. Todas las métricas que hereden
        de esta clase deben implementar este método con exactitud.

        Parámetros
        ----------
        recommendations : dict[Any, list[Any]]
            Diccionario con las recomendaciones para cada usuario {userId: [lista_tmdb_ids]}.
        ground_truth : pd.DataFrame
            Dataset con los datos reales que sabemos que al usuario le gustaron (respuestas del test).
        k : int
            El número de recomendaciones (Top K) que vamos a evaluar.

        Devuelve
        -------
        float
            El resultado final calculado de la métrica (de 0.0 a 1.0 por regla general).
        """
        ...
