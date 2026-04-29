"""
Tests unitarios del sistema de registro de métricas.
"""

import csv
import os
import pytest

from src.utils.registrar_metricas import registrar_metricas, leer_historial, COLUMNAS


class TestRegistrarMetricas:
    def test_crea_csv(self, tmp_path):
        """El registro crea el CSV si no existe."""
        ruta = str(tmp_path / "test_hist.csv")
        resultado = registrar_metricas("SVD", ruta_csv=ruta)
        assert os.path.exists(resultado)

    def test_cabecera_correcta(self, tmp_path):
        """La cabecera del CSV debe coincidir con COLUMNAS."""
        ruta = str(tmp_path / "test_hist.csv")
        registrar_metricas("SVD", ruta_csv=ruta)
        with open(ruta, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            cabecera = next(reader)
        assert cabecera == COLUMNAS

    def test_registro_con_hiperparams(self, tmp_path):
        """Hiperparámetros se guardan correctamente."""
        ruta = str(tmp_path / "test_hist.csv")
        registrar_metricas(
            "KNN",
            hiperparams={"k_vecinos": 40, "n_epocas": 10},
            ruta_csv=ruta,
        )
        with open(ruta, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fila = next(reader)
        assert fila["modelo"] == "KNN"
        assert fila["k_vecinos"] == "40"
        assert fila["n_epocas"] == "10"

    def test_registro_con_metricas(self, tmp_path):
        """Métricas de evaluación se guardan redondeadas."""
        ruta = str(tmp_path / "test_hist.csv")
        registrar_metricas(
            "NCF",
            metricas={"MAE": 0.6789, "RMSE": 0.8765432},
            ruta_csv=ruta,
        )
        with open(ruta, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fila = next(reader)
        assert fila["MAE"] == "0.6789"
        assert fila["RMSE"] == "0.876543"

    def test_acumula_registros(self, tmp_path):
        """Múltiples llamadas acumulan filas."""
        ruta = str(tmp_path / "test_hist.csv")
        registrar_metricas("SVD", ruta_csv=ruta)
        registrar_metricas("KNN", ruta_csv=ruta)
        registrar_metricas("NCF", ruta_csv=ruta)
        with open(ruta, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            filas = list(reader)
        assert len(filas) == 3

    def test_campos_vacios_son_na(self, tmp_path):
        """Campos no proporcionados se rellenan con 'NA'."""
        ruta = str(tmp_path / "test_hist.csv")
        registrar_metricas("SVD", ruta_csv=ruta)
        with open(ruta, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fila = next(reader)
        assert fila["embedding_dim"] == "NA"

    def test_timestamp_presente(self, tmp_path):
        """El timestamp se genera automáticamente."""
        ruta = str(tmp_path / "test_hist.csv")
        registrar_metricas("SVD", ruta_csv=ruta)
        with open(ruta, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fila = next(reader)
        assert fila["timestamp"]  # No vacío
        assert "-" in fila["timestamp"]  # Formato fecha


class TestLeerHistorial:
    def test_lee_csv_existente(self, tmp_path):
        ruta = str(tmp_path / "test_hist.csv")
        registrar_metricas("SVD", ruta_csv=ruta)
        registrar_metricas("KNN", ruta_csv=ruta)
        df = leer_historial(ruta_csv=ruta)
        assert df is not None
        assert len(df) == 2

    def test_devuelve_none_si_no_existe(self, tmp_path):
        ruta = str(tmp_path / "no_existe.csv")
        result = leer_historial(ruta_csv=ruta)
        assert result is None
