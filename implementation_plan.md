# Plan de Refactorización: Arquitectura MLOps y API

Este documento detalla el plan de acción para integrar la nueva estructura de carpetas (`metrics`, `pipelines`, `tracking`, `schemas`, `test`) en el proyecto base, aprovechando las 3 semanas de margen. Se define qué se va a adaptar y qué se descartará por no ser aplicable a un sistema de recomendación de películas.

## Consideraciones Generales y Qué ELIMINAR 🗑️

> [!WARNING]
> **Eliminar `src/schemas/inference.py`**: Este archivo, copiado del repositorio del máster, contiene variables como `stress`, `burnout`, `motivation`, etc. Corresponden a un proyecto de Recursos Humanos o Bienestar, no a nuestro recomendador de Streaming. Lo eliminaremos (o ignoraremos) para crear esquemas verdaderamente útiles para nuestro negocio.

> [!NOTE]
> **No refactorizar el entrenamiento todavía**: Los modelos ya están entrenados y exportados a `.ONNX` y `.pkl`. Por ahora, aplicaremos *Pipelines* principalmente a la **evaluación** (`evaluacion_ranking`) y dejaremos los scripts de entrenamiento tal como están, para priorizar la estabilidad de la API.

---

## Cambios Propuestos (Implementación) 🚀

### 1. Schemas (`src/schemas/`)
**Objetivo**: Tipado fuerte para la API de FastAPI. Evitar errores de formato entre Frontend y Backend.
*   **Crear `src/schemas/recommendation.py`**:
    *   Definiremos `RecommendationResponse`, `RecommendationItem`, etc.
*   **Modificar `src/api/main_api.py`**:
    *   Actualizar todos los endpoints (`/recomendar/svd/...`, `/recomendar/wnd/...`) para que retornen datos validados por Pydantic (los schemas creados) en lugar de diccionarios creados al vuelo.

### 2. Tracking / Telemetría (`src/tracking/`)
**Objetivo**: Guardar un historial estructurado de qué recomendamos y a quién, vital para un sistema MLOps en producción.
*   **Modificar `src/api/main_api.py`**:
    *   Importar `RecommendationLogger` de `src/tracking/logger.py`.
    *   Instanciar el logger al iniciar la App.
    *   Inyectar el logging al final de cada endpoint de recomendación para que guarde en `logs/recommendations.jsonl` un evento cada vez que un usuario pide recomendaciones.

### 3. Métricas de Evaluación (`src/metrics/`)
**Objetivo**: Desacoplar las fórmulas matemáticas del script principal.
*   **Modificar `src/utils/evaluacion_ranking.py`**:
    *   Actualmente, contiene funciones como `ndcg_at_k`, `precision_at_k`, `recall_at_k`, `hit_rate` en la cabecera.
    *   Extraeremos estas funciones conformando clases que sigan el patrón definido en `src/metrics/protocols.py` y las guardaremos en `src/metrics/ranking.py`.

### 4. Pipelines (`src/pipelines/`)
**Objetivo**: Estandarizar la evaluación multimodelo.
*   **Modificar `src/utils/evaluacion_ranking.py`**:
    *   El bucle gigante de evaluación (la función `evaluar()`) se refactorizará para hacer uso de una clase similar a `EvaluationPipeline` (basada en el archivo que copiaste).
    *   El objetivo es que evaluar un modelo nuevo sea tan simple como registrarlo en el pipeline y darle al *Run*, en vez de añadir múltiples bloques `if "NUEVO_MODELO" in modelos: ...`.

### 5. Configuración de Tests (`test/`)
**Objetivo**: Asegurar que la API no se rompe con cambios futuros.
*   **Implementar `test/test_api.py`**:
    *   Crear unos tests básicos usando `TestClient` de FastAPI para verificar que los endpoints de recomendación responden HTTP 200 y devuelven el esquema correcto.

## Open Questions

> [!IMPORTANT]
> 1. **¿Confirmas que podemos ignorar/borrar `src/schemas/inference.py` (el que tiene temas de estrés/burnout)?**
> 2. **¿Os gustaría que implementemos todo el plan directamente o prefieres que lo hagamos paso a paso, empezando solo por Schemas y Tracking para la API?**

## Verification Plan

### Testeo manual:
1. Levantaremos el servidor de desarrollo (`fastapi dev src/api/main_api.py`).
2. Comprobaremos mediante peticiones que ahora se generan archivos enriquecidos en `logs/recommendations.jsonl`.
3. Revisaremos que los endpoints devuelvan las recomendaciones exactamente igual que antes, pero validadas.

### Testeo Automático:
1. Se correrá `pytest test/` para confirmar que los nuevos esquemas de respuesta de la API no se desvían de lo planificado.
