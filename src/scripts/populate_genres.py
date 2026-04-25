import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

db_config = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "root"),
    "database": os.getenv("DB_NAME", "streaming_db"),
    "port": 3306
}

# Géneros estándar de TMDB (Películas y Series)
generos = [
    (28, 'Acción'), (12, 'Aventura'), (16, 'Animación'), (35, 'Comedia'), 
    (80, 'Crimen'), (99, 'Documental'), (18, 'Drama'), (10751, 'Familia'), 
    (14, 'Fantasía'), (36, 'Historia'), (27, 'Terror'), (10402, 'Música'), 
    (9648, 'Misterio'), (10749, 'Romance'), (878, 'Ciencia ficción'), 
    (10770, 'Película de TV'), (53, 'Suspense'), (10752, 'Bélica'), 
    (37, 'Western'), (10759, 'Action & Adventure'), (10762, 'Kids'), 
    (10763, 'News'), (10764, 'Reality'), (10765, 'Sci-Fi & Fantasy'), 
    (10766, 'Soap'), (10767, 'Talk'), (10768, 'War & Politics')
]

def populate():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        sql = "INSERT IGNORE INTO genres (id, name) VALUES (%s, %s)"
        cursor.executemany(sql, generos)
        
        conn.commit()
        print(f"Se han insertado/actualizado {cursor.rowcount} géneros correctamente.")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    populate()
