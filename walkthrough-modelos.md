# Walkthrough - Optimización Neuronal 2026

Hemos completado la optimización de los modelos neuronales y la integración de la arquitectura bi-encoder **Two Towers** para el proyecto de recomendación.

## Cambios Clave

### 1. Modelo Wide & Deep (Ranking)
- **Negative Sampling Eficiente**: Se ha optimizado `RankingDataset` mediante pre-asignación de arrays de `numpy`, logrando procesar 100M+ muestras en segundos.
- **Cambio de Loss**: Se ha migrado de `MSELoss` (regresión) a `BCEWithLogitsLoss` (ranking binario), permitiendo al modelo aprender quién es mejor en lugar de una nota exacta.
- **Filtro Core**: Los modelos neuronales se han centrado en usuarios con `>=1000` valoraciones para maximizar la calidad de los patrones aprendidos.

### 2. Arquitectura Two Towers (Retrieval)
- **Bi-Encoder**: Se han implementado dos torres separadas (Usuario e Item) en `two_towers_net.py`. Esta arquitectura es el estándar de oro para **Recuperación (Retrieval)** porque permite calcular embeddings de items por adelantado y realizar búsquedas de milisegundos sobre millones de candidatos.
- **In-Batch Negatives**: El modelo entrena maximizando la probabilidad de los items vistos contra el resto del lote como ejemplos negativos, lo que acelera dramáticamente la convergencia.

### 3. Integración en el Backend (API)
- **FastAPI / ONNX**: Todos los modelos profundos (NCF, W&D, Two Towers) se sirven ahora mediante **ONNX Runtime**, lo que garantiza una inferencia de alto rendimiento y bajo consumo de CPU.
- **Nuevos Endpoints**:
  - `/recomendar/wnd/{user_id}`: Ranking neuronal preciso.
  - `/recomendar/twotowers/{user_id}`: Recuperación bi-encoder de alta velocidad.

## Resultados de Evaluación - Definitive Leaderboard 2026

Utilizamos el evaluador optimizado `src/utils/evaluacion_ranking.py` sobre 300 usuarios con historial denso (CORE):

| Modelo | Precision@10 | NDCG@10 | Hit Rate | Descripción |
| :--- | :--- | :--- | :--- | :--- |
| **NCF** | **81.6%** | **0.769** | **99.6%** | El campeón absoluto para usuarios recurrentes. |
| **KNN** | 49.9% | 0.508 | 90.3% | Sorprendente rendimiento de vecindarios directos. |
| **Implicit BPR** | 30.2% | 0.295 | 85.0% | Recuperación balanceada. |
| **Two Towers** | 29.6% | 0.277 | 85.6% | Arquitectura de Recuperación escalable. |
| **SVD** | 27.6% | 0.295 | 85.6% | Filtrado colaborativo clásico. |
| **Wide & Deep** | 3.2% | 0.024 | 23.0% | Mejorado, pero requiere más épocas para el set de 25M. |
| **TF-IDF** | 1.3% | 0.016 | 10.3% | El "Cold Start" fallback. |

> [!TIP]
> **Recomendación Estratégica**: Utilizar el **Modelo Híbrido** donde el **Two Towers** genera los 500 candidatos iniciales y el **NCF** los ordena finalmente para el usuario en el Top 10.

## Referencias Técnicas
- **ONNX Models**: Localizados en `src/models/jj/`.
- **API**: `src/api/main_api.py`.
- **Evaluador**: `src/utils/evaluacion_ranking.py`.
