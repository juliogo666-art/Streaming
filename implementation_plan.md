# Plan de Finalización: Proyecto 4 - Streaming

Tras revisar detalladamente el **Enunciado del Proyecto** y el documento **Review Proyect 4 Streaming**, he diseñado el plan definitivo para alcanzar el *Nivel 5 (Máxima Calificación)* en la rúbrica de Algoritmos y Técnicas Big Data.

Actualmente tenemos 3 modelos puramente Colaborativos (dependen de los historiales de los usuarios). Si hacemos la Evaluación de Ranking ahora, nos penalizará el problema del "Cold Start" (usuarios nuevos).

## Propuesta de Próximos Pasos

A continuación se detalla el orden lógico para cerrar el apartado de Inteligencia Artificial del proyecto:

### 1. Modelo 4: Content-Based Filtering (TF-IDF)
**Por qué es necesario:** Soluciona el problema del *Cold Start*. Si un usuario nuevo entra sin valoraciones previas, SVD o Wide&Deep fallan. Un modelo basado en contenido recomienda películas basándose en la similitud de su **sinopsis (overview) y géneros**.
**Acción:** 
- Crear `src/models/jj/modelo_4_content_based.py`.
- Generar una matriz TF-IDF usando descripciones y similitud del coseno.
- Es ultrarrápido de entrenar.

### 2. Script Maestro de Evaluación (Ranking Metrics)
**Por qué es necesario:** El archivo `info/20260313_Evaluation_Metrics_Recommendation_Systems.pptx.pdf` deja clarísimo que RMSE y MAE **no** son suficientes para un sistema real. Necesitamos métricas de Ranking.
**Acción:** 
- Crear `src/models/jj/evaluacion_ranking.py`.
- Este script tomará a 1.000 usuarios de prueba y enfrentará a los 4 modelos (SVD, KNN, W&D, Content-Based).
- Calculará **NDCG@10**, **Precision@10**, y **Hit Rate**, y exportará una tabla comparativa preciosa.

### 3. Fusión en el Backend (Ensemble & Fallbacks)
**Acción:**
- Actualizaremos los endpoints de FastAPI en `main_api.py`.
- Si un usuario tiene historial -> Usa SVD o Wide&Deep.
- Si un usuario es **NUEVO** -> Actuará el *Fallback* usando el Modelo 4 (Content-Based) o un top tendecias.

### 4. Dashboard de Análisis en Streamlit
**Acción:**
- Añadir a la interfaz del Administrador la prometida pestaña de "Comparación de Modelos".
- Mostrar los resultados del script de evaluación mediante gráficos de barras claros (NDCG, RMSE, Tiempo de entrenamiento) para impresionar en el vídeo demostrativo.

---

## Preguntas Abiertas para el Usuario

> [!IMPORTANT]
> **Decisión Requerida:** ¿Damos luz verde a la creación del **Modelo 4 (Content-Based usando TF-IDF)** ahora mismo, y una vez terminado preparamos la super-evaluación de Ranking con todos los modelos juntos? ¿O prefieres saltarte el Modelo 4 e ir directo a evaluar los 3 que ya tenemos?
