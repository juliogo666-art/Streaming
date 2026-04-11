import pandas as pd
import numpy as np
import pickle
import os
import sys
import math
import onnxruntime as ort

# Añadimos el directorio raíz al PATH para que Python encuentre la carpeta "src"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# ======================================================================================
# CONFIGURACIÓN DE RUTAS Y CONSTANTES
# ======================================================================================
RUTA_CATALOGO = "src/data/ready/dataset_final_movies.csv"
RUTA_RATINGS = "src/data/ready/ratings_finales_ia.csv"

# Pesos de modelos clásicos (guardados en artifacts/weights/)
RUTA_SVD = "artifacts/weights/modelo_1_SVD.pkl"
RUTA_KNN = "artifacts/weights/modelo_2.5_knn_msd.pkl"
RUTA_TFIDF_MOD = "artifacts/weights/modelo_4_tfidf.pkl"
RUTA_TFIDF_MAT = "artifacts/weights/modelo_4_matriz.pkl"
RUTA_TFIDF_IDX = "artifacts/weights/modelo_4_indices.pkl"
RUTA_IMP_MOD = "artifacts/weights/modelo_5_implicit.pkl"
RUTA_IMP_DAT = "artifacts/weights/modelo_5_implicit_dataset.pkl"
# Modelos exportados a ONNX (guardados en artifacts/exports/)
RUTA_WND_ONNX = "artifacts/exports/modelo_3_wnd.onnx"
RUTA_NCF_ONNX = "artifacts/exports/modelo_6_ncf.onnx"
RUTA_TT_ONNX = "artifacts/exports/modelo_7_twotowers.onnx"
# Mapeos de IDs internos <-> reales (guardados en artifacts/mappings/)
RUTA_WND_MAP = "artifacts/mappings/wnd_mappings.pkl"
RUTA_NCF_USER2IDX = "artifacts/mappings/ncf_user2idx.json"
RUTA_NCF_ITEM2IDX = "artifacts/mappings/ncf_item2idx.json"
RUTA_TT_MAP = "artifacts/mappings/twotowers_mappings.pkl"

# Guardar Resultados
RUTA_RESULTADOS = "src/utils/metricas_ranking.csv"

# Parámetros de la Evaluación
K = 10  # Número de recomendaciones a generar por modelo
NUM_USUARIOS = 300  # Cantidad de usuarios de prueba para el benchmark
UMBRAL_RELEVANTE = (
    4.0  # Nota mínima para considerar que a un usuario "le gustó" la peli
)

# ======================================================================================
# PIPELINES Y MÉTRICAS OOP
# ======================================================================================
from src.pipelines.evaluation_pipeline import EvaluationPipeline
from src.metrics.precision import PrecisionAtK
from src.metrics.recall import RecallAtK
from src.metrics.hitrate import HitRateAtK
from src.metrics.ndcg import NDCGAtK
from src.metrics.coverage import CoverageAtK
from src.metrics.mrr import MRRAtK


# ======================================================================================
# Carga de modelos (Joblib, Pickle y ONNX Runtime)
# ======================================================================================


def cargar_modelos():
    """Busca y carga todos los modelos entrenados disponibles en disco."""
    print("  Cargando modelos de IA...")
    modelos = {}

    # Modelos clásicos (Surprise / Sklearn)
    if os.path.exists(RUTA_SVD):
        with open(RUTA_SVD, "rb") as f:
            modelos["SVD"] = pickle.load(f)
    if os.path.exists(RUTA_KNN):
        with open(RUTA_KNN, "rb") as f:
            modelos["KNN"] = pickle.load(f)

    # Content-Based (TF-IDF)
    if os.path.exists(RUTA_TFIDF_MAT):
        with open(RUTA_TFIDF_MAT, "rb") as f:
            modelos["TFIDF_MAT"] = pickle.load(f)
        with open(RUTA_TFIDF_IDX, "rb") as f:
            modelos["TFIDF_IDX"] = pickle.load(f)

    # Deep Learning (ONNX Runtime para alta velocidad)
    if os.path.exists(RUTA_WND_ONNX):
        modelos["WND_ONNX"] = ort.InferenceSession(RUTA_WND_ONNX)
        with open(RUTA_WND_MAP, "rb") as f:
            modelos["WND_MAPS"] = pickle.load(f)

    if os.path.exists(RUTA_IMP_MOD):
        with open(RUTA_IMP_MOD, "rb") as f:
            modelos["IMP"] = pickle.load(f)
        with open(RUTA_IMP_DAT, "rb") as f:
            modelos["IMP_DAT"] = pickle.load(f)

    if os.path.exists(RUTA_NCF_ONNX):
        modelos["NCF_ONNX"] = ort.InferenceSession(RUTA_NCF_ONNX)
        import json

        with open(RUTA_NCF_USER2IDX, "r") as f:
            modelos["NCF_U"] = {int(k): v for k, v in json.load(f).items()}
        with open(RUTA_NCF_ITEM2IDX, "r") as f:
            modelos["NCF_I"] = {int(k): v for k, v in json.load(f).items()}

    if os.path.exists(RUTA_TT_ONNX):
        modelos["TT_ONNX"] = ort.InferenceSession(RUTA_TT_ONNX)
        with open(RUTA_TT_MAP, "rb") as f:
            modelos["TT_MAPS"] = pickle.load(f)

    return modelos


# ======================================================================================
# Predicciones
# ======================================================================================


def predecir_svd_knn(modelo, user_id, candidatas):
    """Predicción clásica basada en puntuación estimada."""
    predicciones = []

    # Para cada película candidata, le pedimos al modelo que adivine la nota
    for id_peli in candidatas:
        nota_adivinada = modelo.predict(user_id, id_peli).est
        predicciones.append((id_peli, nota_adivinada))

    # Ordenamos de nota más alta a más baja
    predicciones.sort(key=lambda x: x[1], reverse=True)

    # Devolvemos solo las IDs de las 10 mejores
    return [pelicula[0] for pelicula in predicciones[:K]]


def predecir_wnd(sesion_modelo, diccionario_mapas, user_id, candidatas):
    """Inferencia Wide & Deep usando ONNX Runtime."""
    mapa_usuario = diccionario_mapas["user2idx"]
    mapa_peli = diccionario_mapas["movie2idx"]

    # Si el modelo nunca vio a este usuario en el entrenamiento, no puede recomendarle nada
    if user_id not in mapa_usuario:
        return []

    id_interno_usuario = mapa_usuario[user_id]

    # Filtramos las películas candidatas, quedándonos solo con las que el modelo conoce
    candidatas_validas = [
        (id_tmdb, mapa_peli[id_tmdb]) for id_tmdb in candidatas if id_tmdb in mapa_peli
    ]
    if not candidatas_validas:
        return []

    lista_ids_tmdb, lista_ids_internos = zip(*candidatas_validas)

    # Para evaluar rápido con ONNX, creamos dos arrays paralelos.
    # Ejemplo: Si evaluamos 3 pelis, sería Usuario=[5, 5, 5] Pelis=[12, 45, 87]
    array_usuarios = np.full(
        len(lista_ids_internos), id_interno_usuario, dtype=np.int64
    )
    array_peliculas = np.array(lista_ids_internos, dtype=np.int64)

    # Le pasamos los arrays al modelo y nos devuelve las puntuaciones de todas de golpe
    puntuaciones = sesion_modelo.run(
        None, {"user_ids": array_usuarios, "item_ids": array_peliculas}
    )[0].flatten()

    # Emparejamos cada película con su puntuación y ordenamos de mejor a peor
    pares_ordenados = sorted(
        zip(lista_ids_tmdb, puntuaciones), key=lambda x: x[1], reverse=True
    )

    return [pelicula[0] for pelicula in pares_ordenados[:K]]


def predecir_tt(sesion_modelo, diccionario_mapas, user_id, candidatas):
    """Inferencia de la arquitectura 'Two Towers' bi-encoder."""
    mapa_usuario = diccionario_mapas["user2idx"]
    mapa_peli = diccionario_mapas["item2idx"]

    if user_id not in mapa_usuario:
        return []

    id_interno_usuario = mapa_usuario[user_id]

    candidatas_validas = [
        (id_tmdb, mapa_peli[id_tmdb]) for id_tmdb in candidatas if id_tmdb in mapa_peli
    ]
    if not candidatas_validas:
        return []

    lista_ids_tmdb, lista_ids_internos = zip(*candidatas_validas)

    array_usuarios = np.full(
        len(lista_ids_internos), id_interno_usuario, dtype=np.int64
    )
    array_peliculas = np.array(lista_ids_internos, dtype=np.int64)

    # La red neuronal Two Towers calcula la similitud entre el usuario y la peli internamente
    puntuaciones = sesion_modelo.run(
        None, {"user_ids": array_usuarios, "item_ids": array_peliculas}
    )[0].flatten()
    pares_ordenados = sorted(
        zip(lista_ids_tmdb, puntuaciones), key=lambda x: x[1], reverse=True
    )

    return [pelicula[0] for pelicula in pares_ordenados[:K]]


def predecir_ncf(sesion_modelo, mapa_usuario, mapa_peli, user_id, candidatas):
    """Neural Collaborative Filtering (GMF + MLP)."""
    if user_id not in mapa_usuario:
        return []

    id_interno_usuario = mapa_usuario[user_id]

    candidatas_validas = [
        (id_tmdb, mapa_peli[id_tmdb]) for id_tmdb in candidatas if id_tmdb in mapa_peli
    ]
    if not candidatas_validas:
        return []

    lista_ids_tmdb, lista_ids_internos = zip(*candidatas_validas)

    array_usuarios = np.full(
        len(lista_ids_internos), id_interno_usuario, dtype=np.int64
    )
    array_peliculas = np.array(list(lista_ids_internos), dtype=np.int64)

    puntuaciones = sesion_modelo.run(
        None, {"user_ids": array_usuarios, "item_ids": array_peliculas}
    )[0].flatten()
    pares_ordenados = sorted(
        zip(lista_ids_tmdb, puntuaciones), key=lambda x: x[1], reverse=True
    )

    return [pelicula[0] for pelicula in pares_ordenados[:K]]


def predecir_content(
    matriz_distancias, diccionario_indices, historial_vistas, candidatas
):
    """Recomendador basado en similitud de coseno sobre TF-IDF (Busca pelis que se parezcan de tramas/género)."""
    from sklearn.metrics.pairwise import cosine_similarity

    if historial_vistas.empty:
        return []

    # Cogemos la película que mejor nota le ha puesto este usuario (su favorita)
    peli_favorita = historial_vistas.sort_values(by="rating", ascending=False).iloc[0]
    id_favorita = int(peli_favorita["tmdb_id"])

    if id_favorita not in diccionario_indices:
        return []

    # Sacamos el vector matemático, la huella dactilar, de su película favorita
    indice_favorita = diccionario_indices[id_favorita]

    # Comparamos la huella de la favorita con las huellas de TODAS las demás del catálogo
    similitudes = cosine_similarity(
        matriz_distancias[indice_favorita], matriz_distancias
    ).flatten()

    predicciones = []
    # Nos quedamos solo con las distancias de las películas que no ha visto (candidatas)
    for id_peli in candidatas:
        if id_peli in diccionario_indices:
            parecido = similitudes[diccionario_indices[id_peli]]
            predicciones.append((id_peli, parecido))

    # Las que más se parezcan (mayor porcentaje de similitud) van primero
    predicciones.sort(key=lambda x: x[1], reverse=True)
    return [pelicula[0] for pelicula in predicciones[:K]]


def predecir_implicit(modelo, datos, user_id, candidatas):
    """Bayesian Personalized Ranking (BPR) sobre grupos invisibles matemáticos."""
    mapa_usuario = datos["user2idx"]
    mapa_peli = datos["item2idx"]
    # Creamos un mapa inverso: de indice interno a ID real de TMDB
    mapa_inverso_peli = {indice: id_tmdb for id_tmdb, indice in mapa_peli.items()}

    if user_id not in mapa_usuario:
        return []

    id_interno_usuario = mapa_usuario[user_id]

    # Extraemos las "preferencias invisibles" del usuario
    gustos_usuario = np.asarray(modelo.user_factors[id_interno_usuario])
    # Extraemos los "rasgos invisibles" de todas las películas
    rasgos_peliculas = np.asarray(modelo.item_factors)

    # Multiplicamos matemáticamente sus gustos por los rasgos de las pelis para sacar la "afinidad"
    afinidades = gustos_usuario @ rasgos_peliculas.T

    set_candidatas = set(candidatas)
    predicciones = []

    for indice_interno, afinidad in enumerate(afinidades):
        id_tmdb = mapa_inverso_peli[indice_interno]
        # Si la pelicula no la ha visto, la apuntamos
        if id_tmdb in set_candidatas:
            predicciones.append((id_tmdb, afinidad))

    # Ordenamos por las de mayor afinidad
    predicciones.sort(key=lambda x: x[1], reverse=True)
    return [pelicula[0] for pelicula in predicciones[:K]]


# ---- EVALUACIÓN ----


def evaluar():
    """
    Ejecuta el examen final de los modelos.
    ¿Cómo lo hace?
    Si el usuario vio 10 películas que le gustaron, ocultamos 2 de ellas.
    Dejamos que la IA nos recomiende basándose en las 8 restantes,
    y si adivina alguna de esas 2 que escondimos, ¡bingo, ha acertado!
    """
    print("\n INICIANDO EVALUACIÓN DEFINITIVA")
    modelos = cargar_modelos()

    # 1. Cargamos el Big Data original
    df_interacciones = pd.read_csv(RUTA_RATINGS, on_bad_lines="skip")
    df_catalogo = pd.read_csv(RUTA_CATALOGO)

    # 2. Por limpieza, solo dejamos que el modelo recomiende películas conocidas (> 100 votos)
    # Así no recomienda basura indie que a nadie le importa.
    pelis_populares_permitidas = (
        set(df_catalogo[df_catalogo["vote_count"] > 100]["tmdb_id"].unique())
        if "vote_count" in df_catalogo.columns
        else set(df_catalogo["tmdb_id"].unique())
    )
    tamano_del_catalogo = len(pelis_populares_permitidas)

    # 3. Preparando la máquina examinadora --> El Pipeline de evaluación
    # Le cargamos la lista de reglas para puntuar
    mis_metricas = [
        PrecisionAtK(user_col="userId", item_col="tmdb_id"),  # Cuántas ha acertado
        RecallAtK(
            user_col="userId", item_col="tmdb_id"
        ),  # Cuánto del total fue capaz de capturar
        HitRateAtK(
            user_col="userId", item_col="tmdb_id"
        ),  # ¿Acertó alguna al menos? (1 = Si, 0 = No)
        NDCGAtK(
            user_col="userId", item_col="tmdb_id", rating_col="rating"
        ),  # ¿Las que acertó las puso de primeras?
        CoverageAtK(
            catalog_size=tamano_del_catalogo
        ),  # ¿Se arriesga con distintas opciones o siempre recomienda Marvel?
        MRRAtK(
            user_col="userId", item_col="tmdb_id"
        ),  # ¿En qué posición aparece el primer acierto?
    ]
    pipeline_juez = EvaluationPipeline(metrics=mis_metricas)

    # 4. Buscamos a los "Cinéfilos" (Usuarios expertos que han visto más de 1000 películas)
    # Hacerle un examen a un usuario que solo ha visto 2 películas no tiene sentido lógico.
    conteo_vistas = df_interacciones.groupby("userId").size()
    usuarios_expertos = conteo_vistas[conteo_vistas >= 1000].index.tolist()

    # Elegimos usuarios al azar de entre los expertos, para no freir el PC en tiempo
    usuarios_test = np.random.choice(
        usuarios_expertos, min(len(usuarios_expertos), NUM_USUARIOS), replace=False
    )

    # Indexamos por usuario para buscar rápido (Optimización de velocidad)
    df_busqueda_rapida = df_interacciones.set_index("userId")

    # Aquí guardaremos el "Examen" oficial (los datos ocultos) de todos los usuarios juntos
    lista_respuestas_examen = []

    # Aquí guardaremos lo que responde la IA en el examen. Un diccionario por cada inteligencia.
    respuestas_de_la_ia = {
        m: {}
        for m in ["SVD", "KNN", "WND_ONNX", "TFIDF_MAT", "IMP", "NCF_ONNX", "TT_ONNX"]
    }

    print(
        f"  Empezando el test de Inteligencia Artificial para {len(usuarios_test)} usuarios cinéfilos..."
    )

    # Empezamos el bucle, usuario a usuario
    for id_usuario in usuarios_test:
        try:
            # A) Buscamos su libreta de películas vistas
            historial = df_busqueda_rapida.loc[[id_usuario]].reset_index()

            # Filtramos solo las que le encantaron (nota >= 4.0). De descartamos lo irrelevante.
            pelis_que_le_fascinan = historial[historial["rating"] >= UMBRAL_RELEVANTE]
            if len(pelis_que_le_fascinan) < 5:
                continue

            # B) Cogemos el 20% al azar y las "tapamos"
            pelis_escondidas = pelis_que_le_fascinan.sample(frac=0.2, random_state=42)
            lista_respuestas_examen.append(pelis_escondidas)

            # Lo que le damos al modelo son las que "No hemos escondido"
            historial_visible = historial.drop(pelis_escondidas.index)

            # Y las "posibles respuestas" (candidatas) son TODAS las pelis del mundo, restando las que ya hemos visto claramente.
            peliculas_candidatas_para_recomendar = pelis_populares_permitidas - set(
                historial_visible["tmdb_id"].unique()
            )

            # C) HORA DEL EXAMEN: Llamamos a todos los modelos para que hagan sus apuestas de Recomendación
            if "SVD" in modelos:
                respuestas_de_la_ia["SVD"][id_usuario] = predecir_svd_knn(
                    modelos["SVD"], id_usuario, peliculas_candidatas_para_recomendar
                )

            if "KNN" in modelos:
                respuestas_de_la_ia["KNN"][id_usuario] = predecir_svd_knn(
                    modelos["KNN"], id_usuario, peliculas_candidatas_para_recomendar
                )

            if "WND_ONNX" in modelos:
                respuestas_de_la_ia["WND_ONNX"][id_usuario] = predecir_wnd(
                    modelos["WND_ONNX"],
                    modelos["WND_MAPS"],
                    id_usuario,
                    peliculas_candidatas_para_recomendar,
                )

            if "TFIDF_MAT" in modelos:
                respuestas_de_la_ia["TFIDF_MAT"][id_usuario] = predecir_content(
                    modelos["TFIDF_MAT"],
                    modelos["TFIDF_IDX"],
                    historial_visible,
                    peliculas_candidatas_para_recomendar,
                )

            if "IMP" in modelos:
                # Código extraído porque el BPR implícito funciona con tensores matemáticos diferentes
                mapa_usuarios_imp = modelos["IMP_DAT"]["user2idx"]
                mapa_pelis_imp = modelos["IMP_DAT"]["item2idx"]
                mapa_inverso_imp = {v: k for k, v in mapa_pelis_imp.items()}

                if id_usuario in mapa_usuarios_imp:
                    idx_interno_usuario = mapa_usuarios_imp[id_usuario]
                    gustos = np.asarray(
                        modelos["IMP"].user_factors[idx_interno_usuario]
                    )
                    peli_rasgos = np.asarray(modelos["IMP"].item_factors)
                    puntuaciones = gustos @ peli_rasgos.T

                    # Filtramos quedándonos solo las candidatas factibles
                    posibles = sorted(
                        [
                            (mapa_inverso_imp[idx], punto)
                            for idx, punto in enumerate(puntuaciones)
                            if mapa_inverso_imp[idx]
                            in peliculas_candidatas_para_recomendar
                        ],
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    respuestas_de_la_ia["IMP"][id_usuario] = [
                        p[0] for p in posibles[:K]
                    ]

            if "NCF_ONNX" in modelos:
                respuestas_de_la_ia["NCF_ONNX"][id_usuario] = predecir_ncf(
                    modelos["NCF_ONNX"],
                    modelos["NCF_U"],
                    modelos["NCF_I"],
                    id_usuario,
                    peliculas_candidatas_para_recomendar,
                )

            if "TT_ONNX" in modelos:
                respuestas_de_la_ia["TT_ONNX"][id_usuario] = predecir_tt(
                    modelos["TT_ONNX"],
                    modelos["TT_MAPS"],
                    id_usuario,
                    peliculas_candidatas_para_recomendar,
                )

        except Exception as error_escondido:
            # Si falla un usuario, pasamos silenciosamente al siguiente
            continue

    # ======================================================================================
    # Computación de Métricas en Batch con el Pipeline
    # ======================================================================================
    # Juntamos los "Exámenes oficiales" sueltos de cada usuario en un mega-examen (un solo DataFrame)
    mega_examen_oficial = pd.concat(lista_respuestas_examen, ignore_index=True)

    print("\n Pasándole los exámenes al Tribunal (Pipeline OOP)...")

    # Para que en la web los nombres queden bonitos, mapeamos código
    nombres_comerciales = {
        "SVD": "SVD",
        "KNN": "KNN Clásico",
        "WND_ONNX": "Wide&Deep (Red Neuronal)",
        "TFIDF_MAT": "Recomendación Basada en la Trama (TF-IDF)",
        "IMP": "Filtrado Implícito (BPR)",
        "NCF_ONNX": "Red Neuronal NCF-Lite",
        "TT_ONNX": "Two-Towers Bi-Encoder",
    }

    # Le damos a revisar al Tribunal modelo por modelo
    for id_del_modelo, predicciones_modelo in respuestas_de_la_ia.items():
        if not predicciones_modelo:
            continue

        nombre_bonito = nombres_comerciales[id_del_modelo]

        # Le decimos al sistema central: "Evalúa este modelo dadas sus respuestas y las correctas"
        pipeline_juez.evaluate_model(
            nombre_bonito, predicciones_modelo, mega_examen_oficial, k=K
        )

        # Registramos automáticamente estos resultados en el CSV centralizado
        # para que no se pierdan y queden en el historial de experimentos
        pipeline_juez.registrar_resultados_en_csv(
            nombre_modelo=nombre_bonito,
            tamano_dataset=len(df_interacciones),
            notas=f"Evaluación ranking K={K}, {len(usuarios_test)} usuarios test",
        )

    # El tribunal escupe el tablero de puntuaciones limpio
    dataframe_resultados_finales = pipeline_juez.get_summary_dataframe()

    print("\n RESULTADOS FINALES:")
    print(dataframe_resultados_finales.to_markdown(index=False))

    # Lo exportamos a Excel/CSV para que la APP lo enseñe en Streamlit
    dataframe_resultados_finales.to_csv(RUTA_RESULTADOS, index=False)
    print(f"\n Informe final generado y guardado en: {RUTA_RESULTADOS}")


if __name__ == "__main__":
    evaluar()
