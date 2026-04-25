"""
Tests unitarios de todas las métricas de evaluación de recomendaciones.
Usa datos sintéticos (sin BD ni modelos).
"""

import pytest
import pandas as pd

from src.metrics.precision import PrecisionAtK
from src.metrics.recall import RecallAtK
from src.metrics.ndcg import NDCGAtK
from src.metrics.hitrate import HitRateAtK
from src.metrics.coverage import CoverageAtK
from src.metrics.mrr import MRRAtK


# ── Fixtures locales ──────────────────────────────────────────────────────────


@pytest.fixture
def ground_truth():
    """Películas que realmente le gustaron a cada usuario."""
    return pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 3, 3],
            "tmdb_id": [100, 200, 300, 200, 400, 100, 500],
            "rating": [5.0, 4.0, 3.5, 4.5, 3.0, 5.0, 4.0],
        }
    )


@pytest.fixture
def recs_perfectas():
    """Recomendaciones que aciertan todas las relevantes."""
    return {
        1: [100, 200, 300, 999, 998],  # 3 aciertos para user 1
        2: [200, 400, 888, 777, 666],  # 2 aciertos para user 2
        3: [100, 500, 555, 444, 333],  # 2 aciertos para user 3
    }


@pytest.fixture
def recs_malas():
    """Recomendaciones que no aciertan nada."""
    return {
        1: [999, 998, 997, 996, 995],
        2: [888, 887, 886, 885, 884],
        3: [777, 776, 775, 774, 773],
    }


@pytest.fixture
def recs_parciales():
    """Recomendaciones con aciertos solo en posiciones tardías."""
    return {
        1: [999, 998, 100, 200, 997],  # Primer acierto en posición 3
        2: [888, 200, 887, 886, 885],  # Primer acierto en posición 2
        3: [777, 776, 775, 100, 774],  # Primer acierto en posición 4
    }


# ── Tests Precision@K ────────────────────────────────────────────────────────


class TestPrecisionAtK:
    def test_precision_perfecta(self, recs_perfectas, ground_truth):
        metric = PrecisionAtK()
        # k=3: user1=3/3, user2=2/3, user3=2/3
        resultado = metric.compute(recs_perfectas, ground_truth, k=3)
        assert resultado > 0.0
        assert resultado <= 1.0

    def test_precision_cero(self, recs_malas, ground_truth):
        metric = PrecisionAtK()
        resultado = metric.compute(recs_malas, ground_truth, k=5)
        assert resultado == 0.0

    def test_precision_vacia(self, ground_truth):
        metric = PrecisionAtK()
        resultado = metric.compute({}, ground_truth, k=5)
        assert resultado == 0.0

    def test_precision_k_mayor(self, recs_perfectas, ground_truth):
        metric = PrecisionAtK()
        # k=1: solo mira el primer item
        resultado_k1 = metric.compute(recs_perfectas, ground_truth, k=1)
        resultado_k5 = metric.compute(recs_perfectas, ground_truth, k=5)
        # Precision@1 >= Precision@5 (menos denominador)
        assert resultado_k1 >= resultado_k5


# ── Tests Recall@K ────────────────────────────────────────────────────────────


class TestRecallAtK:
    def test_recall_perfecta(self, recs_perfectas, ground_truth):
        metric = RecallAtK()
        resultado = metric.compute(recs_perfectas, ground_truth, k=5)
        assert resultado > 0.0

    def test_recall_cero(self, recs_malas, ground_truth):
        metric = RecallAtK()
        resultado = metric.compute(recs_malas, ground_truth, k=5)
        assert resultado == 0.0

    def test_recall_vacia(self, ground_truth):
        metric = RecallAtK()
        resultado = metric.compute({}, ground_truth, k=5)
        assert resultado == 0.0


# ── Tests NDCG@K ──────────────────────────────────────────────────────────────


class TestNDCGAtK:
    def test_ndcg_positiva(self, recs_perfectas, ground_truth):
        metric = NDCGAtK()
        resultado = metric.compute(recs_perfectas, ground_truth, k=5)
        assert resultado > 0.0
        assert resultado <= 1.0

    def test_ndcg_cero(self, recs_malas, ground_truth):
        metric = NDCGAtK()
        resultado = metric.compute(recs_malas, ground_truth, k=5)
        assert resultado == 0.0

    def test_ndcg_penaliza_posicion(self, ground_truth):
        """El acierto en posición 1 debe dar NDCG mayor que en posición 5."""
        metric = NDCGAtK()
        # Acierto en posición 1
        recs_top = {1: [100, 999, 998, 997, 996]}
        # Acierto en posición 5
        recs_bajo = {1: [999, 998, 997, 996, 100]}
        ndcg_top = metric.compute(recs_top, ground_truth, k=5)
        ndcg_bajo = metric.compute(recs_bajo, ground_truth, k=5)
        assert ndcg_top > ndcg_bajo


# ── Tests HitRate@K ───────────────────────────────────────────────────────────


class TestHitRateAtK:
    def test_hitrate_total(self, recs_perfectas, ground_truth):
        metric = HitRateAtK()
        resultado = metric.compute(recs_perfectas, ground_truth, k=5)
        assert resultado == 1.0  # Todos los usuarios tienen al menos 1 acierto

    def test_hitrate_cero(self, recs_malas, ground_truth):
        metric = HitRateAtK()
        resultado = metric.compute(recs_malas, ground_truth, k=5)
        assert resultado == 0.0

    def test_hitrate_rango(self, recs_parciales, ground_truth):
        metric = HitRateAtK()
        resultado = metric.compute(recs_parciales, ground_truth, k=5)
        assert 0.0 <= resultado <= 1.0


# ── Tests Coverage@K ─────────────────────────────────────────────────────────


class TestCoverageAtK:
    def test_coverage_basica(self, recs_perfectas):
        gt_dummy = pd.DataFrame()  # Coverage no usa ground_truth
        metric = CoverageAtK(catalog_size=1000)
        resultado = metric.compute(recs_perfectas, gt_dummy, k=5)
        assert resultado > 0.0
        assert resultado <= 1.0

    def test_coverage_catalogo_cero(self, recs_perfectas):
        gt_dummy = pd.DataFrame()
        metric = CoverageAtK(catalog_size=0)
        resultado = metric.compute(recs_perfectas, gt_dummy, k=5)
        assert resultado == 0.0

    def test_coverage_sin_recomendaciones(self):
        gt_dummy = pd.DataFrame()
        metric = CoverageAtK(catalog_size=100)
        resultado = metric.compute({}, gt_dummy, k=5)
        assert resultado == 0.0


# ── Tests MRR@K ───────────────────────────────────────────────────────────────


class TestMRRAtK:
    def test_mrr_primer_acierto(self, ground_truth):
        """Acierto en posición 1 → MRR=1.0."""
        metric = MRRAtK()
        recs = {1: [100, 999, 998, 997, 996]}  # Acierto en posición 1
        resultado = metric.compute(recs, ground_truth, k=5)
        assert resultado == 1.0

    def test_mrr_tercer_acierto(self, ground_truth):
        """Acierto en posición 3 → MRR=1/3."""
        metric = MRRAtK()
        recs = {1: [999, 998, 100, 997, 996]}  # Acierto en posición 3
        resultado = metric.compute(recs, ground_truth, k=5)
        assert abs(resultado - 1 / 3) < 0.001

    def test_mrr_sin_acierto(self, recs_malas, ground_truth):
        metric = MRRAtK()
        resultado = metric.compute(recs_malas, ground_truth, k=5)
        assert resultado == 0.0

    def test_mrr_penaliza_tardio(self, ground_truth):
        """MRR con acierto temprano > MRR con acierto tardío."""
        metric = MRRAtK()
        recs_rapido = {1: [100, 999, 998]}
        recs_lento = {1: [999, 998, 100]}
        mrr_rapido = metric.compute(recs_rapido, ground_truth, k=3)
        mrr_lento = metric.compute(recs_lento, ground_truth, k=3)
        assert mrr_rapido > mrr_lento
