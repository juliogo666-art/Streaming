##############################################################################################
#
#  MODELO 5: LightFM — Modelo Híbrido (Collaborative Filtering + Content-Based)
#  ==============================================================================
#  Sistema de Recomendación Híbrido que combina las ventajas de dos mundos:
#
#  ¿Qué es LightFM?
#  -----------------
#  Es un modelo de factorización matricial que, a diferencia del SVD clásico,
#  puede incorporar "features" adicionales tanto de los usuarios como de los ítems.
#
#  Mientras que SVD solo trabaja con la matriz user × movie (IDs puros),
#  LightFM además puede saber que "Matrix" tiene géneros [Action, Sci-Fi]
#  y que está en idioma "en". Esto le da una ventaja ENORME:
#
#  → Puede recomendar películas NUEVAS (que nadie ha visto) basándose en sus géneros.
#  → Puede recomendar A USUARIOS NUEVOS si les preguntamos sus géneros favoritos.
#  → Esto resuelve el problema de "Cold Start" de forma nativa.
#
#  ¿Qué es WARP Loss?
#  -------------------
#  WARP = Weighted Approximate-Rank Pairwise.
#  A diferencia de SVD (que intenta predecir la nota exacta: 4.2, 3.7...),
#  WARP optimiza directamente el RANKING: le importa que la película correcta
#  esté en el Top-10, no la nota exacta. Esto produce rankings mucho mejores
#  en la práctica, que es exactamente lo que un sistema de streaming necesita.
#
#  Diferencia clave con los otros modelos:
#  ----------------------------------------
#  | Modelo        | Tipo          | Optimiza      | Cold Start |
#  |---------------|---------------|---------------|------------|
#  | SVD           | CF puro       | Rating (RMSE) | ❌ No      |
#  | KNN           | CF puro       | Rating (RMSE) | ❌ No      |
#  | Wide&Deep     | DL con IDs    | Rating (MSE)  | ❌ No      |
#  | TF-IDF        | Content puro  | Similitud     | ⚠️ Parcial |
#  | LightFM       | HÍBRIDO       | Ranking (WARP)| ✅ SÍ      |
#
#  Paper: "Metadata Embeddings for User and Item Cold-start Recommendations"
#  (Maciej Kula, 2015). Librería: lightfm
#
##############################################################################################

import pandas as pd
import numpy as np
import pickle
import os
import time
import json

from lightfm import LightFM
from lightfm.data import Dataset as LFMDataset
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score



##############################################################################################
#  CONFIGURACIÓN GLOBAL
##############################################################################################

# Rutas de datos
RUTA_RATINGS = "src/data/ready/ratings_finales_ia.csv"
RUTA_CATALOGO = "src/data/ready/dataset_final_movies.csv"

# Rutas de salida del modelo
RUTA_MODELO = "src/models/jj/modelo_5_lightfm.pkl"
RUTA_DATASET = "src/models/jj/modelo_5_lfm_dataset.pkl"
RUTA_METRICAS = "src/models/jj/modelo_5_metricas.json"

# --- Hiperparámetros de LightFM ---

# Umbral: Solo consideramos como "interacción positiva" los ratings >= 3.5.
# ¿Por qué? WARP loss trabaja con datos implícitos (visto/no visto, le gustó/no le gustó).
# Un rating de 1.0 o 2.0 NO es una señal positiva, así que los excluimos.
UMBRAL_POSITIVO = 3.5

# Dimensiones del embedding (factores latentes).
# Cada usuario y película se representan como un vector de N números.
# 64 es un buen equilibrio entre capacidad y velocidad para datasets grandes.
NUM_COMPONENTES = 64

# Número de épocas de entrenamiento.
# WARP necesita más épocas que BPR o logistic loss para converger bien.
NUM_EPOCAS = 30

# Tasa de aprendizaje: Cuánto se ajustan los pesos en cada paso.
# Más bajo = más estable pero más lento. 0.05 es agresivo pero eficiente con WARP.
LEARNING_RATE = 0.05

# Penalización L2 para item features (regularización).
# Previene que el modelo se sobreajuste a las features más comunes.
ITEM_ALPHA = 1e-6

# Función de pérdida: WARP optimiza el ranking directamente.
# Alternativas: 'bpr' (Bayesian Personalized Ranking), 'logistic' (cross-entropy)
LOSS = "warp"

# Número de hilos para el entrenamiento en paralelo.
# En Windows, LightFM no soporta OpenMP nativo, así que usamos 1.
NUM_THREADS = 1


##############################################################################################
#  PASO 1: CARGA Y PREPARACIÓN DE DATOS
##############################################################################################


def cargar_datos():
    """
    Carga el CSV de ratings y el catálogo de películas.
    Filtra los ratings para quedarnos solo con las interacciones positivas (>= 3.5).
    Esto es porque WARP necesita datos implícitos: "le gustó" o "no le gustó",
    no la nota exacta.
    """
    print("=" * 70)
    print("  MODELO 5: LightFM HÍBRIDO — Carga de Datos")
    print("=" * 70)

    # 1. Leer el CSV de ratings completo (33.8M filas)
    print(f"\n  Leyendo {RUTA_RATINGS}...")
    df_ratings = pd.read_csv(RUTA_RATINGS)
    print(f"  -> Filas totales en bruto: {len(df_ratings):,}")
    print(f"  -> Usuarios unicos: {df_ratings['userId'].nunique():,}")
    print(f"  -> Peliculas unicas: {df_ratings['tmdb_id'].nunique():,}")

    # 2. Filtrar: solo nos quedamos con ratings >= 3.5 (interacciones positivas)
    # WARP aprende de la AUSENCIA de interacción como negativo implícito,
    # por lo que solo necesitamos las interacciones positivas explícitas.
    df_positivos = df_ratings[df_ratings["rating"] >= UMBRAL_POSITIVO].copy()

    print(f"\n  Filtro: rating >= {UMBRAL_POSITIVO} (interacciones positivas)")
    print(f"  -> Interacciones positivas: {len(df_positivos):,}")
    print(f"  -> Usuarios con al menos 1 positiva: {df_positivos['userId'].nunique():,}")
    print(f"  -> Peliculas con al menos 1 positiva: {df_positivos['tmdb_id'].nunique():,}")

    # 3. Cargar catálogo de películas para extraer features (géneros, idioma, año)
    print(f"\n  Leyendo catálogo {RUTA_CATALOGO}...")
    df_catalogo = pd.read_csv(RUTA_CATALOGO, on_bad_lines="skip", engine="python")
    print(f"  -> Peliculas en catálogo: {len(df_catalogo):,}")

    return df_ratings, df_positivos, df_catalogo


##############################################################################################
#  PASO 2: PREPARAR ITEM FEATURES (Géneros, Idioma, Década)
##############################################################################################


def preparar_item_features(df_catalogo):
    """
    Convierte las columnas del catálogo en "features" que LightFM puede entender.
    
    LightFM espera features como etiquetas de texto: 'genre:Action', 'lang:en', 'decade:2010'.
    Internamente, cada etiqueta se convierte en un vector one-hot que se suma al embedding
    base de la película, enriqueciendo su representación.
    
    ¿Por qué esto es poderoso?
    Si "Matrix" tiene features [Action, SciFi, en, 1990s] y una película nueva
    tiene features [Action, SciFi, en, 2020s], LightFM ya "sabe" algo sobre ella
    aunque nadie la haya visto, porque comparte features con películas conocidas.
    """
    print("\n  Preparando Item Features...")

    item_features_lista = {}  # {tmdb_id: [lista de features]}
    todas_features = set()    # Acumulador global de features únicas

    for _, row in df_catalogo.iterrows():
        tmdb_id = row["tmdb_id"]
        features = []

        # --- Feature 1: Géneros ---
        # genre_ids viene como string tipo "[28, 12, 878]" o "Action, Adventure"
        generos_raw = str(row.get("genre_ids", ""))
        if generos_raw and generos_raw != "nan":
            # Limpiamos caracteres de lista y separamos
            generos_raw = generos_raw.replace("[", "").replace("]", "").replace("'", "").replace('"', '')
            for g in generos_raw.split(","):
                g = g.strip()
                if g:
                    feature = f"genre:{g}"
                    features.append(feature)
                    todas_features.add(feature)

        # --- Feature 2: Idioma original ---
        idioma = str(row.get("original_language", ""))
        if idioma and idioma != "nan":
            feature = f"lang:{idioma}"
            features.append(feature)
            todas_features.add(feature)

        # --- Feature 3: Década de estreno ---
        # Agrupamos por década para no tener 100+ años individuales
        ano = row.get("ano", None)
        if pd.notna(ano):
            try:
                decada = int(float(ano)) // 10 * 10  # 2015 -> 2010, 1999 -> 1990
                feature = f"decade:{decada}s"
                features.append(feature)
                todas_features.add(feature)
            except (ValueError, TypeError):
                pass

        # Guardamos las features de esta película
        if features:
            item_features_lista[tmdb_id] = features

    print(f"  -> Features únicas encontradas: {len(todas_features)}")
    print(f"  -> Películas con features: {len(item_features_lista):,}")

    # Mostramos algunas features de ejemplo
    ejemplo_id = list(item_features_lista.keys())[0]
    print(f"  -> Ejemplo (tmdb_id={ejemplo_id}): {item_features_lista[ejemplo_id][:5]}")

    return item_features_lista, list(todas_features)


##############################################################################################
#  PASO 3: CONSTRUIR EL DATASET DE LIGHTFM
##############################################################################################


def construir_dataset(df_positivos, item_features_lista, todas_features):
    """
    LightFM necesita su propio formato de Dataset interno.
    
    El proceso es:
    1. Registrar todos los user_ids y item_ids que existen
    2. Registrar todas las features posibles
    3. Construir la matriz de interacciones (sparse: usuario x película = 1 si le gustó)
    4. Construir la matriz de item features (sparse: película x feature = 1 si la tiene)
    
    Todo se guarda en matrices dispersas (Sparse/CSR) que ocupan poquísima RAM
    incluso con millones de entradas, porque la gran mayoría de la matriz son ceros.
    """
    print("\n" + "=" * 70)
    print("  MODELO 5: LightFM — Construyendo Dataset")
    print("=" * 70)

    # Crear el Dataset de LightFM y registrar los IDs y features
    dataset = LFMDataset()

    # fit() registra todos los IDs de usuarios e ítems que vamos a usar
    # y las features adicionales que queremos asociar a los ítems
    print("\n  Registrando usuarios, películas y features en el dataset...")
    dataset.fit(
        users=df_positivos["userId"].unique(),
        items=df_positivos["tmdb_id"].unique(),
        item_features=todas_features,  # Las etiquetas como 'genre:Action', 'lang:en'
    )

    num_users, num_items = dataset.interactions_shape()
    print(f"  -> Matriz de interacciones: {num_users:,} usuarios x {num_items:,} películas")

    # Construir la matriz de interacciones (lo que vio cada usuario)
    # build_interactions espera tuplas: (userId, tmdb_id) o (userId, tmdb_id, weight)
    # Usamos el rating como peso para dar más importancia a las películas con 5.0
    print("\n  Construyendo matriz de interacciones (esto puede tardar 1-2 min)...")
    inicio = time.time()

    # Construimos las tuplas de forma vectorizada (zip sobre columnas NumPy)
    # Esto es ~100x más rápido que iterrows() para 33M de filas
    interacciones_tuplas = list(zip(
        df_positivos["userId"].values,
        df_positivos["tmdb_id"].values,
        df_positivos["rating"].values,
    ))

    # build_interactions devuelve 2 matrices:
    # - interactions: la matriz user x item con los pesos (rating)
    # - weights: la misma pero en formato CSR para LightFM
    interacciones, pesos = dataset.build_interactions(interacciones_tuplas)

    duracion = time.time() - inicio
    print(f"  -> Interacciones construidas en {duracion:.1f}s")
    print(f"  -> Forma: {interacciones.shape}, NNZ (no-ceros): {interacciones.nnz:,}")

    # Construir la matriz de Item Features
    # build_item_features espera tuplas: (tmdb_id, [lista_de_features])
    print("\n  Construyendo matriz de item features...")
    item_features_tuplas = [
        (tmdb_id, feats) for tmdb_id, feats in item_features_lista.items()
        if tmdb_id in set(df_positivos["tmdb_id"].unique())  # Solo pelis con interacciones
    ]

    item_features_matrix = dataset.build_item_features(item_features_tuplas)
    print(f"  -> Item features: {item_features_matrix.shape}")

    return dataset, interacciones, pesos, item_features_matrix


##############################################################################################
#  PASO 4: ENTRENAMIENTO DEL MODELO
##############################################################################################


def entrenar_modelo(interacciones, item_features_matrix):
    """
    Entrena el modelo LightFM con WARP loss.
    
    WARP funciona así en cada paso de entrenamiento:
    1. Toma un usuario y una película que SÍ le gustó (positivo)
    2. Muestrea aleatoriamente una película que NO ha visto (negativo)
    3. Si el modelo ya rankea el positivo por encima del negativo, no hace nada
    4. Si el negativo aparece más arriba que el positivo → Penaliza y ajusta pesos
    
    Esto es MUCHO más eficiente que entrenar con todos los negativos explícitamente.
    El negative sampling es AUTOMÁTICO con WARP, no necesitamos generarlo manualmente.
    """
    print("\n" + "=" * 70)
    print("  MODELO 5: LightFM — Entrenamiento WARP")
    print("=" * 70)

    # Crear el modelo con los hiperparámetros configurados
    modelo = LightFM(
        no_components=NUM_COMPONENTES,  # Dimensión de los embeddings
        learning_rate=LEARNING_RATE,    # Tasa de aprendizaje
        loss=LOSS,                       # WARP: optimiza ranking directamente
        item_alpha=ITEM_ALPHA,           # Regularización L2 de item features
        random_state=42,                 # Semilla para reproducibilidad
    )

    print("\n  Hiperparámetros:")
    print(f"    - Componentes (embedding dim): {NUM_COMPONENTES}")
    print(f"    - Loss: {LOSS} (Weighted Approximate-Rank Pairwise)")
    print(f"    - Learning Rate: {LEARNING_RATE}")
    print(f"    - Regularización items: {ITEM_ALPHA}")
    print(f"    - Épocas: {NUM_EPOCAS}")
    print(f"    - Threads: {NUM_THREADS}")

    # Entrenamos época por época para monitorizar el progreso
    print(f"\n  Entrenando {NUM_EPOCAS} épocas...")
    inicio_total = time.time()

    for epoca in range(NUM_EPOCAS):
        inicio_epoca = time.time()

        # fit_partial entrena UNA época más sobre el modelo existente
        # Sin necesidad de empezar de cero cada vez
        modelo.fit_partial(
            interactions=interacciones,
            item_features=item_features_matrix,
            num_threads=NUM_THREADS,
            epochs=1,
        )

        tiempo_epoca = time.time() - inicio_epoca

        # Cada 5 épocas, mostramos el progreso con una métrica rápida
        if (epoca + 1) % 5 == 0 or epoca == 0:
            # Precision@10 sobre el train set (indicador rápido de convergencia, no de calidad final)
            train_precision = precision_at_k(
                modelo, interacciones, item_features=item_features_matrix, k=10, num_threads=NUM_THREADS
            ).mean()

            print(
                f"    Época {epoca + 1:02d}/{NUM_EPOCAS} | "
                f"Precision@10 (train): {train_precision:.4f} | "
                f"Tiempo: {tiempo_epoca:.1f}s"
            )

    duracion_total = time.time() - inicio_total
    print(f"\n  Entrenamiento completado en {duracion_total:.1f} segundos ({duracion_total/60:.1f} min)")

    return modelo


##############################################################################################
#  PASO 5: EVALUACIÓN DEL MODELO
##############################################################################################


def evaluar_modelo(modelo, interacciones, item_features_matrix):
    """
    Evalúa el modelo calculando métricas de ranking sobre el conjunto de entrenamiento.
    
    Nota: Para una evaluación rigurosa, deberíamos hacer un train/test split temporal.
    Pero como ya tenemos evaluacion_ranking.py que compara los 5 modelos de forma justa,
    aquí solo calculamos métricas rápidas para asegurar que el modelo ha convergido.
    """
    print("\n" + "=" * 70)
    print("  MODELO 5: LightFM — Evaluación")
    print("=" * 70)

    # Precision@10: ¿Cuántas de las top-10 recomendaciones son relevantes?
    precision = precision_at_k(
        modelo, interacciones, item_features=item_features_matrix, k=10, num_threads=NUM_THREADS
    ).mean()

    # Recall@10: De todas las películas relevantes, ¿qué % captura el top-10?
    recall = recall_at_k(
        modelo, interacciones, item_features=item_features_matrix, k=10, num_threads=NUM_THREADS
    ).mean()

    # AUC: Probabilidad de que un positivo tenga mayor score que un negativo aleatorio
    # 0.5 = modelo aleatorio, 1.0 = modelo perfecto
    auc = auc_score(
        modelo, interacciones, item_features=item_features_matrix, num_threads=NUM_THREADS
    ).mean()

    print("\n  ╔════════════════════════════════════════════╗")
    print("  ║  RESULTADOS DE LightFM (WARP)              ║")
    print("  ╠════════════════════════════════════════════╣")
    print(f"  ║  Precision@10: {precision:.4f}                    ║")
    print(f"  ║  Recall@10:    {recall:.4f}                    ║")
    print(f"  ║  AUC Score:    {auc:.4f}                    ║")
    print("  ╚════════════════════════════════════════════╝")

    metricas = {
        "precision_at_10": round(float(precision), 4),
        "recall_at_10": round(float(recall), 4),
        "auc_score": round(float(auc), 4),
    }

    return metricas


##############################################################################################
#  PASO 6: GUARDAR EL MODELO
##############################################################################################


def guardar_modelo(modelo, dataset, metricas):
    """
    Guarda el modelo LightFM y el dataset (que contiene los mappings de IDs)
    en archivos .pkl para que el Backend los cargue sin re-entrenar.
    
    También guardamos las métricas en JSON para documentación y comparación.
    """
    print("\n  Guardando modelo y dataset...")

    # Guardar el modelo entrenado (pesos, embeddings, configuración)
    with open(RUTA_MODELO, "wb") as f:
        pickle.dump(modelo, f)
    tamano_modelo = os.path.getsize(RUTA_MODELO) / (1024 * 1024)

    # Guardar el Dataset de LightFM (contiene los mappings user_id <-> idx, item_id <-> idx)
    # Sin estos mappings, no podemos traducir IDs reales a los índices internos del modelo
    with open(RUTA_DATASET, "wb") as f:
        pickle.dump(dataset, f)
    tamano_dataset = os.path.getsize(RUTA_DATASET) / (1024 * 1024)

    # Guardar métricas en JSON legible
    with open(RUTA_METRICAS, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    print(f"  -> Modelo guardado en: {RUTA_MODELO} ({tamano_modelo:.1f} MB)")
    print(f"  -> Dataset guardado en: {RUTA_DATASET} ({tamano_dataset:.1f} MB)")
    print(f"  -> Métricas guardadas en: {RUTA_METRICAS}")


##############################################################################################
#  PASO 7: FUNCIÓN DE RECOMENDACIÓN (para uso desde el Backend)
##############################################################################################


def recomendar(modelo, dataset, item_features_matrix, user_id, df_ratings, n=10):
    """
    Genera las top-N recomendaciones para un usuario.
    
    A diferencia de SVD (que predice un rating exacto),
    LightFM devuelve un SCORE de ranking: cuanto más alto, más probable
    que al usuario le guste esa película. El score no tiene un rango fijo.
    
    Parámetros:
    ----------
    modelo              : Modelo LightFM entrenado
    dataset             : Dataset de LightFM con los mappings de IDs
    item_features_matrix: Matriz de features de ítems
    user_id             : ID real del usuario (de la tabla users)
    df_ratings           : DataFrame con los ratings (para saber qué ya vio)
    n                   : Número de recomendaciones a devolver
    
    Returns:
    -------
    Lista de dicts: [{tmdb_id, score}, ...]
    """
    # 1. Obtener los mappings internos de LightFM
    # user_id_map: {user_id_real: idx_interno}
    # item_id_map: {tmdb_id_real: idx_interno}
    user_id_map, _, item_id_map, _ = dataset.mapping()

    # 2. Verificar que el usuario existe en el modelo
    if user_id not in user_id_map:
        return []  # Usuario no visto en entrenamiento (Cold Start de usuario)

    user_idx = user_id_map[user_id]

    # 3. Obtener películas ya vistas por el usuario
    pelis_vistas = set(df_ratings[df_ratings["userId"] == user_id]["tmdb_id"].tolist())

    # 4. Generar scores para TODAS las películas del modelo de golpe
    # Esto es MUCHO más rápido que iterar una por una como hacemos en SVD (0.1s vs 10s+)
    n_items = len(item_id_map)
    scores = modelo.predict(
        user_ids=user_idx,                 # Un solo usuario
        item_ids=np.arange(n_items),       # Todas las películas a la vez
        item_features=item_features_matrix,
    )

    # 5. Crear un mapeo inverso (idx_interno -> tmdb_id_real) para traducir resultados
    idx_to_tmdb = {idx: tmdb_id for tmdb_id, idx in item_id_map.items()}

    # 6. Emparejar cada película con su score y filtrar las ya vistas
    predicciones = []
    for item_idx, score in enumerate(scores):
        tmdb_id = idx_to_tmdb.get(item_idx)
        if tmdb_id is not None and tmdb_id not in pelis_vistas:
            predicciones.append({
                "tmdb_id": int(tmdb_id),
                "predicted_rating": round(float(score), 4),
            })

    # 7. Ordenar por score descendente y devolver top-N
    predicciones.sort(key=lambda x: x["predicted_rating"], reverse=True)
    return predicciones[:n]


##############################################################################################
#  PASO 8: CARGAR MODELO GUARDADO (para uso desde el Backend)
##############################################################################################


def cargar_modelo_guardado():
    """
    Carga el modelo LightFM y su dataset desde disco.
    El Backend (FastAPI) usará esta función al arrancar.
    """
    if not os.path.exists(RUTA_MODELO) or not os.path.exists(RUTA_DATASET):
        print(f"  Error: No se encontró el modelo LightFM en {RUTA_MODELO}")
        print(f"  Ejecuta primero: uv run src/models/jj/modelo_5_lightfm.py")
        return None, None

    with open(RUTA_MODELO, "rb") as f:
        modelo = pickle.load(f)
    with open(RUTA_DATASET, "rb") as f:
        dataset = pickle.load(f)

    print(f"  Modelo LightFM cargado desde {RUTA_MODELO}")
    return modelo, dataset


##############################################################################################
#  PRINCIPAL: Se ejecuta directamente
#
#   Comando: uv run src/models/jj/modelo_5_lightfm.py
##############################################################################################

if __name__ == "__main__":
    # Paso 1: Cargar y filtrar los datos
    df_ratings, df_positivos, df_catalogo = cargar_datos()

    # Paso 2: Extraer features del catálogo (géneros, idioma, década)
    item_features_lista, todas_features = preparar_item_features(df_catalogo)

    # Paso 3: Construir el Dataset de LightFM con interacciones + features
    dataset, interacciones, pesos, item_features_matrix = construir_dataset(
        df_positivos, item_features_lista, todas_features
    )

    # Paso 4: Entrenar el modelo con WARP loss
    modelo = entrenar_modelo(interacciones, item_features_matrix)

    # Paso 5: Evaluar métricas de ranking
    metricas = evaluar_modelo(modelo, interacciones, item_features_matrix)

    # Paso 6: Guardar modelo y dataset para el Backend
    guardar_modelo(modelo, dataset, metricas)

    # Paso 7: Demo rápida — Probamos recomendaciones para un usuario de ejemplo
    print("\n" + "=" * 70)
    print("  DEMO: Recomendaciones LightFM para el Usuario 1")
    print("=" * 70)

    recomendaciones = recomendar(
        modelo, dataset, item_features_matrix,
        user_id=1, df_ratings=df_ratings, n=10,
    )

    # Intentamos mostrar títulos bonitos cruzando con el catálogo
    try:
        print("\n  {:<5} {:<45} {:<15}".format('Pos', 'Título', 'Score'))
        print("  " + "-" * 65)
        for i, rec in enumerate(recomendaciones, 1):
            match = df_catalogo[df_catalogo["tmdb_id"] == rec["tmdb_id"]]
            titulo = (
                match["titulo"].values[0]
                if not match.empty
                else f"ID: {rec['tmdb_id']}"
            )
            # Truncamos títulos largos para que queden bonitos
            if len(str(titulo)) > 42:
                titulo = str(titulo)[:39] + "..."
            print(f"  {i:<5} {titulo:<45} Score: {rec['predicted_rating']}")
    except Exception:
        # Si no tenemos el catálogo, mostramos solo los IDs
        for i, rec in enumerate(recomendaciones, 1):
            print(f"  {i}. tmdb_id={rec['tmdb_id']} | Score: {rec['predicted_rating']}")

    print("\n  ¡Modelo LightFM WARP Híbrido listo para producción!")
    print("=" * 70)
