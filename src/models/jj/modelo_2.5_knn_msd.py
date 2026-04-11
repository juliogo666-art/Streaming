##############################################################################################
#
#  EL MODELO DE CINE BASADO EN LOS GUSTOS DE LAS MASAS (KNN + Diferencia Cuadrática Media / MSD)
#  ================================================================================
#  Imagina que estamos en el clase y quieres saber si una peli te va a gustar.
#  En vez de preguntarle a los profesores de cine (que usan algoritmos complejos como SVD),
#  este modelo le pregunta a "los grupos" (tus vecinos).
#
#  ¿Cómo funciona en palabras sencillas?
#  -------------------------------------
#  Este modelo usa una regla llamada "Basado en Objetos" (Item-Based).
#  1. Si a ti te encantó 'Matrix' (5 estrellas).
#  2. El modelo busca en su base de datos y dice: "A ver, la gente que le dio 5 a Matrix...
#     ¿a qué otra peli le dieron 5 estrellas masivamente?".
#  3. Descubre que todos esos frikis le dieron un 5 a 'Inception' (Origen).
#  4. Entonces te dice: "¡Eh! Deberías ver Inception".
#
#  Usamos Inteligencia Artificial (KNNWithMeans) pero en el fondo es solo estadística masiva:
#  Busca 'K' (el número) de Vecinos (personas o películas similares).
#
##############################################################################################

import pandas as pd
import pickle
import os
import sys
import time

# Añadir el directorio raíz al PATH para importar utilidades
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
from src.utils.registrar_metricas import registrar_metricas

from surprise import KNNWithMeans, Dataset, Reader
from surprise.model_selection import train_test_split, GridSearchCV
from surprise import accuracy

##############################################################################################
#  CONFIGURACIÓN GLOBAL (Reglas del juego)
##############################################################################################

# ¿Ruta al archivo CSV con las valoraciones?
UBICACION_DE_VALORACIONES = "src/data/ready/ratings_finales_ia.csv"

# ¿Dónde guardaremos el cerebro de la IA una vez entrenado?
UBICACION_PARA_GUARDAR_MODELO = "artifacts/weights/modelo_2.5_knn_msd.pkl"

# Filtros para no sobrecargar el ordenador (porque calcular millones de vecinos peta la Memoria RAM)
MINIMO_PELIS_VISTAS_POR_USUARIO = 50
MINIMO_VOTOS_POR_PELICULA = 150

# --- Ajustes del algoritmo de los "Vecinos" ---
MAXIMO_DE_VECINOS_CERCANOS = (
    30  # Cuántas películas "amigas" queremos coger para tomar una decisión
)
MINIMO_DE_VECINOS_PARA_FIARSE = 5  # Si no hay al menos 5 de similitud, la descartamos


##############################################################################################
#  PASO 1: CARGAR Y LIMPIAR LOS DATOS (Quitar la morralla)
##############################################################################################


def cargar_datos():
    """Lee las valoraciones de los espectadores y tira a la basura la info irrelevante."""
    print("=" * 70)
    print("  Modelo 2: Carga de Datos")
    print("=" * 70)

    print(f"\n  Leyendo: {UBICACION_DE_VALORACIONES}...")
    tabla_votos = pd.read_csv(UBICACION_DE_VALORACIONES)

    # 1. Filtro de peliculas --> Tiramos películas rarísimas que nadie ha visto (menos de 150 votos)
    conteo_por_pelicula = tabla_votos.groupby("tmdb_id").size()
    peliculas_populares = conteo_por_pelicula[
        conteo_por_pelicula >= MINIMO_VOTOS_POR_PELICULA
    ].index
    tabla_limpia = tabla_votos[tabla_votos["tmdb_id"].isin(peliculas_populares)]

    # 2. Filtro de usuarios --> Tiramos a usuarios vagos que no valoran lo suficiente (menos de 50 votos)
    conteo_por_usuario = tabla_limpia.groupby("userId").size()
    usuarios_activos = conteo_por_usuario[
        conteo_por_usuario >= MINIMO_PELIS_VISTAS_POR_USUARIO
    ].index
    tabla_limpia = tabla_limpia[tabla_limpia["userId"].isin(usuarios_activos)]

    print(
        f"  -> Películas que pasan el corte (> {MINIMO_VOTOS_POR_PELICULA} votos): {tabla_limpia['tmdb_id'].nunique():,}"
    )
    print(
        f"  -> Usuarios cinéfilos (> {MINIMO_PELIS_VISTAS_POR_USUARIO} votos): {tabla_limpia['userId'].nunique():,}"
    )
    print(f"  -> Total de votos válidos para estudiar: {len(tabla_limpia):,}\n")

    return tabla_limpia


##############################################################################################
#  PASO 2: ENTRENAMIENTO DEL MODELO KNN
##############################################################################################


def entrenar_modelo(tabla_limpia):
    """Encontramos la configuración perfecta, estudiamos y hacemos un examen de prueba."""
    print("=" * 70)
    print("  Modelo 2: Entrenamiento KNNWithMeans")
    print("=" * 70)

    # Le decimos a la IA que las notas en el cine van de 0.5 a 5 estrellas
    lector_de_notas = Reader(rating_scale=(0.5, 5.0))
    # Transformamos nuestra tabla Excel en el formato raro que entiende el algoritmo matemático
    datos_formateados_para_ia = Dataset.load_from_df(
        tabla_limpia[["userId", "tmdb_id", "rating"]], lector_de_notas
    )

    print(
        "\n  Ejecutando el 'Buscador Automático' para probar cuál es la mejor configuración..."
    )
    # Le damos a probar varios números de vecinos (30 o 40) a ver cuál atina más.
    opciones_de_prueba = {
        "k": [30, 40],
        "sim_options": {
            "name": ["msd"],  # La formula 'msd' es súper rápida
            "user_based": [False],
        },
    }

    # GridSearchCV es nuestro buscador.
    # cv=3 significa que dividirá los datos en 3 partes y hará 3 entrenamientos
    # Le ordenamos probar todo y trabajar con 2 núcleos del PC (n_jobs=2)
    # No usar todos los nucleos con n_jobs=-1 porque puede dar problemas con la memoria RAM
    # y desbordamiento de memoria a la ssd
    buscador_automatico = GridSearchCV(
        KNNWithMeans, opciones_de_prueba, measures=["rmse", "mae"], cv=3, n_jobs=2
    )

    cronometro = time.time()
    buscador_automatico.fit(datos_formateados_para_ia)

    print(
        f"  ¡Ha terminado de pensar las combinaciones en {(time.time() - cronometro):.1f} segundos!"
    )
    print(
        f"\n  El mejor margen de error que pudo lograr es: {buscador_automatico.best_score['rmse']:.4f}"
    )

    print("\n  Dividiendo: 80% para aprender / 20% guardarlo para el examen")
    # Partimos los datos. Estudia sobre el 80% y luego le preguntamos sobre el 20% que no había visto.
    datos_para_estudiar, datos_para_examinarse = train_test_split(
        datos_formateados_para_ia, test_size=0.2, random_state=42
    )

    # Rescatamos la configuración ganadora del buscador
    numero_ideal_de_vecinos = buscador_automatico.best_params["rmse"]["k"]
    mejores_reglas_similitud = buscador_automatico.best_params["rmse"]["sim_options"]

    # Creamos el cerebro de IA oficial con la configuración ganadora
    inteligencia_artificial_final = KNNWithMeans(
        k=numero_ideal_de_vecinos,
        min_k=MINIMO_DE_VECINOS_PARA_FIARSE,
        sim_options=mejores_reglas_similitud,
        verbose=True,
    )

    print(f"\n  Calculando parentescos entre películas...")
    cronometro = time.time()
    inteligencia_artificial_final.fit(datos_para_estudiar)  # ¡Aquí es donde aprende!
    print(
        f"\n  Red de parentesco construida en {time.time() - cronometro:.1f} segundos"
    )

    print("\n  Haciéndole el examen sobre el 20% de datos ocultos...")
    respuestas_examen = inteligencia_artificial_final.test(datos_para_examinarse)

    # Revisamos cuántas falló el alumno
    fallo_cuadratico = accuracy.rmse(respuestas_examen, verbose=False)
    fallo_absoluto = accuracy.mae(respuestas_examen, verbose=False)

    print(f"\n  ╔════════════════════════════════════════╗")
    print(f"  ║  NOTA FINAL EN LA CARTILLA DE LA IA  ║")
    print(f"  ╠════════════════════════════════════════╣")
    print(f"  ║  Error RMSE (El grave): {fallo_cuadratico:.4f}         ║")
    print(f"  ║  Error MAE (El normal): {fallo_absoluto:.4f}         ║")
    print(f"  ╚════════════════════════════════════════╝")

    return (
        inteligencia_artificial_final,
        fallo_cuadratico,
        fallo_absoluto,
        buscador_automatico.best_params["rmse"],
    )


##############################################################################################
#  PASO 3: GUARDAR EL MODELO ENTRENADO
##############################################################################################


def guardar_modelo(inteligencia_artificial_final):
    """
    Coge toda la inmensa red de conocimiento adquirida y la empaqueta en un archivo .pkl
    (Esto luego la web FastAPI se encarga de abrirlo para hacer recomendaciones en tiempo real).
    """
    with open(UBICACION_PARA_GUARDAR_MODELO, "wb") as archivo_final:
        pickle.dump(inteligencia_artificial_final, archivo_final)

    tamano_megas = os.path.getsize(UBICACION_PARA_GUARDAR_MODELO) / (1024 * 1024)
    print(
        f"\n  Cerebro metido en el frasco: {UBICACION_PARA_GUARDAR_MODELO} (Ocupa: {tamano_megas:.1f} MB en tu disco)"
    )


##############################################################################################
#  PASO 4: PRUEBA RÁPIDA DE FUEGO
##############################################################################################


def recomendar(inteligencia_artificial, id_del_usuario, historial_entero, limite=10):
    """
    Si el sistema funciona, pasándole un usuario nos escupirá sus 10 películas soñadas.
    """
    # 1. Mirar qué pelis ya ha visto para NO recomendárselas de nuevo
    pelis_que_ya_vio = set(
        historial_entero[historial_entero["userId"] == id_del_usuario][
            "tmdb_id"
        ].tolist()
    )

    # 2. Rebuscar entre TODAS las pelis existentes
    todas_las_pelis_existentes = set(historial_entero["tmdb_id"].unique())

    # 3. Quitarle las que ya vio para dejarle solo las "inéditas"
    pelis_listas_para_adivinar = todas_las_pelis_existentes - pelis_que_ya_vio

    apuestas_de_la_ia = []
    # Usamos la IA para que ponga una nota imaginaria a cada posible película
    for id_peli in pelis_listas_para_adivinar:
        prediccion_del_robot = inteligencia_artificial.predict(id_del_usuario, id_peli)
        apuestas_de_la_ia.append(
            {
                "tmdb_id": int(id_peli),
                "predicted_rating": round(prediccion_del_robot.est, 2),
            }
        )

    # Las ordenamos poniendo las más altas que adivinó arriba de todo
    apuestas_de_la_ia.sort(key=lambda equis: equis["predicted_rating"], reverse=True)
    return apuestas_de_la_ia[:limite]


##############################################################################################
#  ARRANQUE DE LA MÁQUINA
##############################################################################################

if __name__ == "__main__":
    # 1. Cargamos el Excel sucio y lo lavamos
    tabla = cargar_datos()

    # 2. Entrenamos a la IA usando el buscador automático de combinaciones
    ia_terminada, error_cuadratico, error_absoluto, secretos_ganadores = (
        entrenar_modelo(tabla)
    )

    # 3. Guardamos su cerebro para la posteridad
    guardar_modelo(ia_terminada)

    # 4. Apuntamos en la libreta sus "Notas finales" para que el equipo lo vea
    registrar_metricas(
        modelo="KNNWithMeans_Buscador",
        hiperparams={
            "vecinos": secretos_ganadores["k"],
            "formula_similitud": secretos_ganadores["sim_options"]["name"],
            "votos_usuario_minimo": MINIMO_PELIS_VISTAS_POR_USUARIO,
            "votos_peli_minimo": MINIMO_VOTOS_POR_PELICULA,
        },
        metricas={
            "Fallo_Normal_MAE": error_absoluto,
            "Fallo_Grave_RMSE": error_cuadratico,
        },
        dataset_size=len(tabla),
    )

    print("\n" + "=" * 70)
    print("  PRUEBA EN VIVO: Recomendándole películas al Usuario Número 1")
    print("=" * 70)

    lista_magica = recomendar(ia_terminada, id_del_usuario=1, historial_entero=tabla)

    try:
        # Intenta cargar los nombres de las películas para que veamos texto bonito en vez de números feos
        catalogo_nombres = pd.read_csv(
            "src/data/ready/dataset_final_movies.csv",
            on_bad_lines="skip",
            engine="python",
        )
        print(f"\n  {'Pos':<5} {'Título':<45} {'Nota Simulada':<15}")
        print("  " + "-" * 65)
        for num, peli_recomendada in enumerate(lista_magica, 1):
            coincidencia = catalogo_nombres[
                catalogo_nombres["tmdb_id"] == peli_recomendada["tmdb_id"]
            ]
            texto_titulo = (
                coincidencia["titulo"].values[0]
                if not coincidencia.empty
                else f"Código ID: {peli_recomendada['tmdb_id']}"
            )
            if len(texto_titulo) > 42:
                texto_titulo = texto_titulo[:39] + "..."
            print(
                f"  {num:<5} {texto_titulo:<45} estrellas predichas: {peli_recomendada['predicted_rating']}"
            )
    except Exception as problema_tonto:
        # Si no encuentra el archivo de los nombres, escupimos números secos
        for num, peli_recomendada in enumerate(lista_magica, 1):
            print(
                f"  Posición {num}. Número de Peli = {peli_recomendada['tmdb_id']} | Le gustaría muchísimo con un: {peli_recomendada['predicted_rating']}"
            )

    print(
        "\n  ¡Listo! La Inteligencia Artificial basada en vecinos ha terminado sus deberes."
    )
    print("=" * 70)
