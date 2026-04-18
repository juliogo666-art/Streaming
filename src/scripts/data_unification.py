import pandas as pd
import os

def unificar_datos(tipo="movies"):
    print(f"Unificacion para {tipo.upper()}...")

    ruta_tmdb = f"src/data/clean/tmdb_{tipo}_limpio.csv"

    os.makedirs("src/data/ready", exist_ok=True)
    ruta_salida = f"src/data/ready/dataset_final_{tipo}.csv"

    try:
        df_base = pd.read_csv(ruta_tmdb)
    except FileNotFoundError:
        print(f"Falta el archivo limpio de TMDB: {ruta_tmdb}")
        return

    df_final = df_base.copy()

    # Derivar columnas equivalentes a las que antes venían de Trakt, usando TMDB
    # nota_media y total_votos ya existen en TMDB como vote_average / vote_count
    df_final['nota_media'] = df_final['vote_average']
    df_final['total_votos'] = df_final['vote_count']

    # Extraer año de la fecha de estreno
    fecha_col = 'fecha_estreno' if 'fecha_estreno' in df_final.columns else 'first_air_date'
    if fecha_col in df_final.columns:
        df_final['ano'] = pd.to_datetime(df_final[fecha_col], errors='coerce').dt.year.astype('Int64')

    # Derivar flags de tendencia/popularidad desde el score de popularidad de TMDB
    if 'popularity' in df_final.columns:
        df_final['es_tendencia'] = df_final['popularity'] > 20
        df_final['es_popular'] = df_final['popularity'] > 10
    else:
        df_final['es_tendencia'] = False
        df_final['es_popular'] = False

    # Métricas de streaming sin equivalente en TMDB → valores neutros
    df_final['certificacion'] = 'NR'
    df_final['espectadores_live'] = 0
    df_final['reproducciones_totales'] = 0
    df_final['es_historico_vistas'] = False

    if tipo == "movies":
        try:
            df_ratings = pd.read_csv("src/data/ready/ratings_finales_ia.csv")
            peliculas_con_rating = df_ratings['tmdb_id'].unique()
            df_final = df_final[df_final['tmdb_id'].isin(peliculas_con_rating)]
            print(f"Filtro IA aplicado: Quedan {len(df_final)} peliculas validas con historial de ratings.")
        except FileNotFoundError:
            print("Aviso: No se encontro ratings_finales_ia.csv, no se pudo filtrar las peliculas.")
        

    df_final.to_csv(ruta_salida, index=False)
    print(f"Dataset unificado Guardado en: {ruta_salida}\n")

def preparar_matriz_ia():

    print("Unificando Matriz IA de MovieLens...")

    ruta_links_clean = "src/data/clean/links_limpio.csv"
    ruta_ratings_clean = "src/data/clean/ratings_limpio.csv"
    ruta_salida = "src/data/ready/ratings_finales_ia.csv"

    try:
        df_links = pd.read_csv(ruta_links_clean)
        df_ratings = pd.read_csv(ruta_ratings_clean)
        # INNER JOIN: Cruzamos valoraciones con los links para conseguir el tmdbId
        df_ia = pd.merge(df_ratings, df_links[['movieId', 'tmdbId']], on='movieId', how='inner')
    # Renombramos para estandarizar con el resto de la base de datos
        df_ia = df_ia.rename(columns={'tmdbId': 'tmdb_id'})
        # Reordenamos columnas (userId, tmdb_id, rating)
        df_ia = df_ia[['userId', 'tmdb_id', 'rating']]
        

        df_ia.to_csv(ruta_salida, index=False)
        print(f"Matriz IA lista para entrenar. Guardado en: {ruta_salida}\n")

    except FileNotFoundError:
        print(
            "Error: Faltan archivos limpios de MovieLens. Ejecuta data_cleaning.py primero."
        )


if __name__ == "__main__":
    preparar_matriz_ia()
    unificar_datos("movies")
    unificar_datos("shows")
    
    print("Sincronizando base de datos de calificaciones (IA) con el catálogo final de películas...")
    try:
        df_movies = pd.read_csv("src/data/ready/dataset_final_movies.csv")
        df_ia = pd.read_csv("src/data/ready/ratings_finales_ia.csv")
        
        ids_validos = df_movies['tmdb_id'].unique()
        filas_antes = len(df_ia)
        df_ia = df_ia[df_ia['tmdb_id'].isin(ids_validos)]
        filas_despues = len(df_ia)
        
        df_ia.to_csv("src/data/ready/ratings_finales_ia.csv", index=False)
        print(f"Matriz IA sincronizada: {filas_antes} -> {filas_despues} valoraciones restantes (eliminadas {filas_antes - filas_despues} valoraciones de películas fantasma).\n")
    except FileNotFoundError:
        print("Error: No se encontraron los archivos finales para la sincronización.")