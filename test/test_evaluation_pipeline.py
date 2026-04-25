"""
Tests unitarios del pipeline de evaluación.
"""

import pytest
import pandas as pd

from src.pipelines.evaluation_pipeline import EvaluationPipeline
from src.metrics.precision import PrecisionAtK
from src.metrics.recall import RecallAtK
from src.metrics.hitrate import HitRateAtK
from src.metrics.mrr import MRRAtK


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def pipeline_basico():
    """Pipeline con métricas básicas."""
    return EvaluationPipeline(metrics=[PrecisionAtK(), RecallAtK()])


@pytest.fixture
def pipeline_completo():
    """Pipeline con todas las métricas típicas."""
    return EvaluationPipeline(
        metrics=[PrecisionAtK(), RecallAtK(), HitRateAtK(), MRRAtK()]
    )


@pytest.fixture
def datos_evaluacion():
    """Datos mínimos para evaluar."""
    recs = {
        1: [100, 200, 300, 400, 500],
        2: [200, 300, 400, 500, 600],
    }
    gt = pd.DataFrame(
        {
            "userId": [1, 1, 2, 2],
            "tmdb_id": [100, 300, 200, 600],
            "rating": [5.0, 4.0, 3.5, 4.5],
        }
    )
    return recs, gt


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestEvaluationPipeline:
    def test_pipeline_devuelve_dict(self, pipeline_basico, datos_evaluacion):
        recs, gt = datos_evaluacion
        resultado = pipeline_basico.evaluate_model("TestModel", recs, gt, k=5)
        assert isinstance(resultado, dict)

    def test_pipeline_todas_las_metricas_presentes(
        self, pipeline_completo, datos_evaluacion
    ):
        recs, gt = datos_evaluacion
        resultado = pipeline_completo.evaluate_model("TestModel", recs, gt, k=5)
        assert len(resultado) == 4  # 4 métricas

    def test_pipeline_valores_en_rango(self, pipeline_completo, datos_evaluacion):
        recs, gt = datos_evaluacion
        resultado = pipeline_completo.evaluate_model("TestModel", recs, gt, k=5)
        for nombre, valor in resultado.items():
            assert 0.0 <= valor <= 1.0, f"{nombre} fuera de rango: {valor}"

    def test_pipeline_sin_metricas(self, datos_evaluacion):
        pipeline = EvaluationPipeline(metrics=[])
        recs, gt = datos_evaluacion
        resultado = pipeline.evaluate_model("TestModel", recs, gt, k=5)
        assert resultado == {}

    def test_pipeline_sin_recomendaciones(self, pipeline_basico):
        gt = pd.DataFrame(
            {
                "userId": [1, 2],
                "tmdb_id": [100, 200],
                "rating": [5.0, 4.0],
            }
        )
        resultado = pipeline_basico.evaluate_model("TestModel", {}, gt, k=5)
        # Todas las métricas deben ser 0 sin recomendaciones
        for valor in resultado.values():
            assert valor == 0.0

    def test_resultados_se_acumulan(self, pipeline_basico, datos_evaluacion):
        """Evaluar múltiples modelos acumula resultados."""
        recs, gt = datos_evaluacion
        pipeline_basico.evaluate_model("ModeloA", recs, gt, k=5)
        pipeline_basico.evaluate_model("ModeloB", recs, gt, k=5)
        assert "ModeloA" in pipeline_basico.results
        assert "ModeloB" in pipeline_basico.results

    def test_get_summary_dataframe(self, pipeline_completo, datos_evaluacion):
        """get_summary_dataframe devuelve un DataFrame con todas las métricas."""
        recs, gt = datos_evaluacion
        pipeline_completo.evaluate_model("ModeloA", recs, gt, k=5)
        pipeline_completo.evaluate_model("ModeloB", recs, gt, k=5)
        df = pipeline_completo.get_summary_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "Modelo" in df.columns

    def test_get_summary_vacio(self):
        """get_summary_dataframe sin evaluaciones previas → DataFrame vacío."""
        pipeline = EvaluationPipeline(metrics=[])
        df = pipeline.get_summary_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_registrar_resultados_en_csv(self, pipeline_basico, datos_evaluacion, tmp_path):
        """Verificar que registrar_resultados_en_csv llama sin error."""
        recs, gt = datos_evaluacion
        pipeline_basico.evaluate_model("TestCSV", recs, gt, k=5)
        # No debería lanzar excepción
        pipeline_basico.registrar_resultados_en_csv(
            "TestCSV",
            hiperparametros={"k": 5},
            tamano_dataset=100,
            tiempo_entrenamiento_s=1.5,
            notas="Test unitario",
        )

    def test_registrar_modelo_no_evaluado(self, pipeline_basico):
        """registrar_resultados_en_csv con modelo no evaluado no lanza excepción."""
        # Debería imprimir alerta pero no crashear
        pipeline_basico.registrar_resultados_en_csv("ModeloInexistente")

    def test_metrica_con_excepcion(self, datos_evaluacion):
        """Si una métrica lanza excepción, el pipeline la captura y pone 0.0."""

        class MetricaRota:
            def compute(self, recs, gt, k):
                raise RuntimeError("Métrica rota intencionalmente")

        pipeline = EvaluationPipeline(metrics=[MetricaRota()])
        recs, gt = datos_evaluacion
        resultado = pipeline.evaluate_model("TestRota", recs, gt, k=5)
        assert resultado["MetricaRota"] == 0.0
