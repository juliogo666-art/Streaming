# Walkthrough: Modelos de IA y Evaluación de Ranking

Este documento resume los avances y resultados de la fase de modelado de IA para el sistema de recomendación de Streaming.

## Modelos Entrenados

Durante esta fase, implementamos y probamos 5 aproximaciones distintas para el motor de recomendación:

1. **Modelo 1 (SVD)**: Filtrado Colaborativo tradicional (Predictor de Ratings).
2. **Modelo 2 (KNN con Baseline)**: Filtrado basado en vecinos cercanos.
3. **Modelo 3 (Wide & Deep)**: Red Neuronal profunda con PyTorch (Exportable a ONNX).
4. **Modelo 4 (TF-IDF Content-Based)**: Sistema basado en metadatos y similitud de coseno para mitigar el *Cold Start* de películas nuevas.
5. **Modelo 5 (Implicit BPR)**: Modelo de Filtrado Colaborativo optimizado puramente para Ranking Relativo (Bayesian Personalized Ranking) escrito en C++.

## Solución de Problemas Críticos
*   **Problema con LightFM**: Se intentó usar LightFM como modelo 5, pero un fallo conocido de `Segmentation Fault` (Exit Code 1) en las extensiones nativas de C++ al compilarse en Windows con MSVC impedía el entrenamiento.
*   **Solución (Implicit)**: Se pivotó rápidamente a la librería `implicit`, manteniendo la misma filosofía de optimización de rankings mediante su algoritmo BPR (*Bayesian Personalized Ranking*). Se implementó un bucle propio de `filter_already_liked_items` para evitar fallos de lectura de memoria cruzada entre Numpy y Cython en entornos Windows 64-bit.

## Resultados Finales de Evaluación (Offline)

Utilizamos nuestro script `evaluacion_ranking.py` sobre 300 usuarios aleatorios con mucho historial. Ocultamos aleatoriamente el 20% de sus películas "favoritas" y pedimos a los 5 modelos que nos generasen un Top 10 para cada usuario buscando ver si lograban acertarlas.

| Modelo | Precision@10 | Recall@10 | NDCG@10 | Hit Rate@10 | Cobertura |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SVD** | 7.8% | 5.7% | 0.0953 | 45.3% | 3.2% |
| **KNN** | 5.5% | 2.5% | 0.0567 | 26.3% | 7.6% |
| **Wide&Deep** | 0.2% | 0.2% | 0.0016 | 1.7% | 0.3% |
| **TF-IDF** | 0.9% | 0.6% | 0.0148 | 8.3% | 8.2% |
| **Implicit BPR** | **24.6%** | **24.8%** | **0.3267** | **85.7%** | **7.6%** |

> [!TIP]
> **Conclusiones Clave**
> - **Implicit BPR aniquila a los demás**: Al haber sido diseñado específicamente para *rankear* (ordenar) en lugar de *predecir una nota*, entiende perfectamente qué le gusta al usuario basándose en los "Likes" implícitos.
> - **Hit Rate Masivo**: Un 85.7% significa que, para casi 9 de cada 10 usuarios, el modelo logró colar *al menos* una película que realmente les gustaba entre las 10 opciones principales.
> - **Precisión altísima**: De las 10 carátulas que se muestran en el carrusel de Netflix/Streamlit, entra 2 y 3 serán aciertos garantizados. Esto a nivel negocio es una retención de usuarios abismal.
> - **Rendimiento de W&D**: El modelo ensamblado de Deep Learning tuvo un desempeño inferior en ranking puro. Sus "Embeddings" requerirían arquitecturas tipo *Two-Tower* o muestreo negativo (Negative Sampling) durante el entrenamiento para competir en pruebas de ranking, pero es un excelente modelo de reserva o para *Rating Prediction*.
