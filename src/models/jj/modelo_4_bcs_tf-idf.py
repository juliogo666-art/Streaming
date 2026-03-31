"""
#######################################################################################
#
# MODELO 4: CONTENT-BASED FILTERING (TF-IDF)
#
# =======================================================================================
# TF-IDF: Frecuencia de Término - Frecuencia Inversa de Documento
# Técnica matemática que da más peso a las palabras raras o únicas de una sinopsis
# ej. "Tatooine", "Skywalker", y menos a las palabras comunes ej. "el", "una", "historia".
# Así logra capturar el verdadero tema de la película.
#
# =======================================================================================
# Entonces analizamos directamente el contenido de las películas: Sinopsis + Géneros + Título.
#
# Objetivo Principal: Solucionar el "Cold Start".
# Si entra un usuario nuevo sin valoraciones, SVD y Wide&Deep no sirven. Este modelo
# puede recomendarle películas basadas en la popularidad general o buscando películas
# similares a su primera visualización usando similitud de coseno sobre textos.
#
#######################################################################################
"""

import pandas as pd
import pickle
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

############################################################################################

# Archivos
RUTA_CATALOGO = "src/data/ready/dataset_final_movies.csv"
RUTA_RATINGS = "src/data/ready/ratings_finales_ia.csv"

# Destinos guardados
RUTA_MODELO_TFIDF = "src/models/jj/modelo_4_tfidf.pkl"
RUTA_MATRIZ_TFIDF = "src/models/jj/modelo_4_matriz.pkl"
RUTA_MAPPING_IDX = "src/models/jj/modelo_4_indices.pkl"

############################################################################################


def preparar_datos():
    """
    Carga el catálogo de películas y crea una 'sopa de texto' combinando el título,
    géneros y sinopsis.
    """
    print("=" * 70)
    print("  MODELO 4: CONTENT-BASED — Preparando Datos")
    print("=" * 70)

    df_movies = pd.read_csv(RUTA_CATALOGO, on_bad_lines="skip", engine="python")
    print(f"  -> Películas cargadas: {len(df_movies):,}")

    # Rellenar valores nulos
    df_movies["overview"] = df_movies["overview"].fillna("")
    df_movies["genre_ids"] = df_movies["genre_ids"].fillna("")
    df_movies["titulo"] = df_movies["titulo"].fillna("")

    # Crear el "perfil_textual" de palabras clave para que TF-IDF lo analice.
    # Le damos más peso al título repitiéndolo 2 veces, sumado a la sinopsis y géneros.
    df_movies["perfil_textual"] = (
        df_movies["titulo"]
        + " "
        + df_movies["titulo"]
        + " "
        + df_movies["genre_ids"]
        + " "
        + df_movies["overview"]
    )

    # Reseteamos índice para que coincida exactamente con las filas de la matriz TF-IDF
    df_movies = df_movies.reset_index(drop=True)

    return df_movies


############################################################################################


def entrenar_tfidf(df_movies):
    """
    Convierte todo el texto de las películas en una matriz matemática gigante
    donde el conteo de palabras da peso a las temáticas.
    """
    print(
        "\n  Iniciando Vectorización TF-IDF (Extracción de conceptos de las sinopsis)..."
    )
    inicio = time.time()

    # Usamos TF-IDF para palabras en inglés y español (stop words) para quitar preposiciones inútiles
    tfidf = TfidfVectorizer(stop_words="english", max_features=10000)

    # Ajustar y transformar el perfil textual a una matriz matemática (muy eficiente en memoria Sparse)
    tfidf_matrix = tfidf.fit_transform(df_movies["perfil_textual"])

    duracion = time.time() - inicio
    print(f"  -> Vectorización completada en {duracion:.1f} segundos.")
    print(
        f"  -> Forma de la Matriz Textual: {tfidf_matrix.shape} (Películas x Conceptos)"
    )

    # Mapeo rápido de tmdb_id a fila de la matriz para luego buscar instantáneamente
    indices = pd.Series(df_movies.index, index=df_movies["tmdb_id"]).to_dict()

    return tfidf, tfidf_matrix, indices


############################################################################################


def guardar_modelo(tfidf, matrix, indices):
    """Guarda el modelo para consumo en Backend."""
    print("\n  Guardando modelo Content-Based en disco...")

    with open(RUTA_MODELO_TFIDF, "wb") as f:
        pickle.dump(tfidf, f)
    with open(RUTA_MATRIZ_TFIDF, "wb") as f:
        pickle.dump(matrix, f)
    with open(RUTA_MAPPING_IDX, "wb") as f:
        pickle.dump(indices, f)

    mb1 = os.path.getsize(RUTA_MODELO_TFIDF) / (1024 * 1024)
    mb2 = os.path.getsize(RUTA_MATRIZ_TFIDF) / (1024 * 1024)
    print(f"  -> Extractor palabras guardado ({mb1:.1f} MB)")
    print(f"  -> Matriz Textual Dispersa guardada ({mb2:.1f} MB)")
    print("=" * 70)


############################################################################################


def recomendar_por_contenido_usuario(
    user_id, df_ratings, df_movies, tfidf_matrix, indices, n=10
):
    """
    La lógica en el arranque en frío:
    1. Si el user_id no existe en ratings (nuevo) -> Recomienda las películas más populares del catálogo.
    2. Si el usuario existe -> Busca su película mejor valorada, y a través de TF-IDF devuelve películas
       con una sinopsis/género muy similar.
    """

    # Consultamos si el usuario ha votado alguna vez
    user_ratings = df_ratings[df_ratings["userId"] == user_id]

    # Caso 1: COLD START - Usuario sin historial
    if user_ratings.empty:
        print(
            f"  [Cold Start] Usuario {user_id} es nuevo. Recomendando por popularidad global..."
        )
        # Ordenamos el catálogo por popularidad y nota
        if "popularity" in df_movies.columns and "vote_average" in df_movies.columns:
            top_populares = df_movies.sort_values(
                by=["popularity", "vote_average"], ascending=[False, False]
            )
        else:
            top_populares = df_movies.head(50)

        recomendaciones = []
        for idx, row in top_populares.head(n).iterrows():
            recomendaciones.append(
                {
                    "tmdb_id": int(row["tmdb_id"]),
                    "predicted_rating": 4.5,  # Placeholder simbólico "Te gustará"
                    "razon": "Tendencia / Popular",
                }
            )
        return recomendaciones

    # Caso 2: CONTENT-BASED ITEM-TO-ITEM - Usuario con historial y encontramos a su favorita
    # Buscamos su película favorita histórica (la de ID más alto entre las que votó 5 o 4 estrellas)
    favorita = user_ratings.sort_values(by="rating", ascending=False).iloc[0]
    tmdb_id_fav = int(favorita["tmdb_id"])

    if tmdb_id_fav not in indices:
        return []  # Película favorita no encontrada en catálogo limpio

    idx_peli_fav = indices[tmdb_id_fav]
    print(
        f"  [Content] Basando recomendaciones en su película favorita: tmdb_id={tmdb_id_fav} (Nota: {favorita['rating']})"
    )

    # Calculamos la similitud de coseno solo entre la película favorita y las otras 55.000.
    # No cruzamos las 55k x 55k porque nos daría error de memoria.
    # linear_kernel ultrarrápido (1x55000).
    vector_favorita = tfidf_matrix[idx_peli_fav]
    similitudes = linear_kernel(vector_favorita, tfidf_matrix).flatten()

    # Obtenemos los índices ordenados de mayor a menor similitud
    # [::-1] invierte el array para tener los más altos primero y ponemos
    # [1:n+1] para saltar la peli en sí misma (similitud 1.0)
    top_indices = similitudes.argsort()[::-1][1 : n + 1]

    recomendaciones = []
    for idx_similar in top_indices:
        peli_sim = df_movies.iloc[idx_similar]
        score_similitud = similitudes[idx_similar]

        # Simulamos un "predicted rating" combinando la nota real que él dio a la base
        # y la puntuación de similitud textual.
        nota_base = favorita["rating"]
        pseudo_rating = min(5.0, nota_base * (0.8 + 0.2 * score_similitud))

        recomendaciones.append(
            {
                "tmdb_id": int(peli_sim["tmdb_id"]),
                "predicted_rating": round(pseudo_rating, 2),
                "razon": f"Similar a tu favorita ({score_similitud * 100:.1f}% coincidencia de texto)",
            }
        )

    return recomendaciones


############################################################################################

if __name__ == "__main__":
    df = preparar_datos()
    modelo_tfidf, matriz_textual, dist_indices = entrenar_tfidf(df)
    guardar_modelo(modelo_tfidf, matriz_textual, dist_indices)

    print("\n  DEMO: Recomendaciones para usuario NUEVO (9999999)")
    df_ratings_demo = pd.read_csv(
        RUTA_RATINGS, nrows=1000
    )  # solo es demo, leemos poquito
    rec_nuevo = recomendar_por_contenido_usuario(
        9999999, df_ratings_demo, df, matriz_textual, dist_indices
    )
    for i, r in enumerate(rec_nuevo, 1):
        print(f"  {i}. tmdb_id={r['tmdb_id']} -> {r['razon']}")

    print("\n  DEMO: Recomendaciones para usuario Antiguo (ID 1)")
    rec_viejo = recomendar_por_contenido_usuario(
        1, df_ratings_demo, df, matriz_textual, dist_indices
    )
    for i, r in enumerate(rec_viejo, 1):
        print(
            f"  {i}. tmdb_id={r['tmdb_id']} -> {r['razon']} (Nota Predicha: {r['predicted_rating']})"
        )

    print("\n  ¡Modelo TF-IDF de Contenido compilado con éxito!")
