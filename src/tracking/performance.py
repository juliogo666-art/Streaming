"""
=============================================================================
Monitorización de Rendimiento
=============================================================================
Middleware FastAPI + colector de métricas del sistema.

Registra automáticamente:
  - Latencia (ms) de cada petición HTTP agrupada por endpoint
  - Snapshots de CPU y RAM del proceso actual
  - Tiempo de arranque del backend (desde el lifespan)

Se conecta con main_api.py para exponer datos via /api/performance.
=============================================================================
"""

import os
import time
import threading
from collections import defaultdict
from typing import Dict, List, Optional

import psutil
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class PerformanceCollector:
    """Singleton que acumula métricas de rendimiento."""

    def __init__(self):
        self.latencias: Dict[str, List[float]] = defaultdict(list)
        self.tiempo_arranque_s: Optional[float] = None
        self._proceso = psutil.Process(os.getpid())
        self._lock = threading.Lock()

    def registrar_latencia(self, endpoint: str, latencia_ms: float) -> None:
        with self._lock:
            self.latencias[endpoint].append(latencia_ms)

    def obtener_metricas_sistema(self) -> dict:
        """Devuelve snapshot actual de CPU/RAM."""
        mem = psutil.virtual_memory()
        proc_mem = self._proceso.memory_info()

        return {
            "ram_total_gb": round(mem.total / (1024**3), 2),
            "ram_usada_sistema_gb": round(mem.used / (1024**3), 2),
            "ram_porcentaje_sistema": mem.percent,
            "ram_proceso_gb": round(proc_mem.rss / (1024**3), 3),
            "ram_proceso_mb": round(proc_mem.rss / (1024**2), 1),
            "cpu_porcentaje_sistema": psutil.cpu_percent(interval=0.5),
            "cpu_cores": psutil.cpu_count(logical=True),
        }

    def obtener_resumen_latencias(self) -> dict:
        """Devuelve estadísticas de latencia por endpoint."""
        resumen = {}
        with self._lock:
            for endpoint, tiempos in self.latencias.items():
                if not tiempos:
                    continue
                tiempos_ord = sorted(tiempos)
                n = len(tiempos_ord)
                p95_idx = min(int(n * 0.95), n - 1)
                resumen[endpoint] = {
                    "peticiones": n,
                    "media_ms": round(sum(tiempos_ord) / n, 2),
                    "min_ms": round(tiempos_ord[0], 2),
                    "max_ms": round(tiempos_ord[-1], 2),
                    "p95_ms": round(tiempos_ord[p95_idx], 2),
                }
        return resumen

    def exportar_todo(self) -> dict:
        """Exporta todas las métricas para el informe."""
        return {
            "sistema": self.obtener_metricas_sistema(),
            "latencia_por_endpoint": self.obtener_resumen_latencias(),
            "tiempo_arranque_s": self.tiempo_arranque_s,
            "latencias_crudas": dict(self.latencias),
        }


# Instancia global del colector
colector = PerformanceCollector()


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware que mide la latencia de cada petición HTTP."""

    async def dispatch(self, request: Request, call_next) -> Response:
        inicio = time.perf_counter()
        response = await call_next(request)
        latencia_ms = (time.perf_counter() - inicio) * 1000

        # Solo rastrear endpoints de la API (no archivos estáticos)
        path = request.url.path
        if path.startswith("/api/") or path.startswith("/recomendar"):
            # Normalizar paths con IDs: /recomendar/svd/198 → /recomendar/svd/{id}
            partes = path.strip("/").split("/")
            partes_norm = []
            for p in partes:
                if p.isdigit():
                    partes_norm.append("{id}")
                else:
                    partes_norm.append(p)
            endpoint_norm = "/" + "/".join(partes_norm)
            colector.registrar_latencia(endpoint_norm, latencia_ms)

        return response
