"""
Vista de Usuario — Módulo principal de la interfaz para usuarios estándar.
Este módulo se encarga de mostrar la vista principal después de que un usuario
ha iniciado sesión. No incluye opciones de administración, centrándose
exclusivamente en presentar recomendaciones personalizadas y el catálogo general.
"""

import os
import time
import requests
import pandas as pd
import streamlit as st

# URL base para la API del recomendador. Idealmente esto debería venir de una variable de entorno.
API_BASE_URL = "http://127.0.0.1:8000"


def render():
    """
    Función principal para dibujar la interfaz del usuario.
    Es llamada desde app_ui.py una vez que se ha comprobado que el usuario tiene permisos.
    """
    # 1. Obtener la información del usuario actual desde la sesión
    usuario = st.session_state.get("usuario_actual", {})
    nombre_mostrar = usuario.get(
        "username", f"Perfil #{usuario.get('id_usuario', '?')}"
    )
    id_usuario = usuario.get("id_usuario", "?")

    # 2. Construcción de la cabecera: Título, Información del usuario y Botón de cierre de sesión
    col_titulo, col_info, col_logout = st.columns([6, 3, 1])

    with col_titulo:
        st.title(f"Bienvenido a SPIRE, {nombre_mostrar}!")

    with col_info:
        # Etiqueta visual para identificar rápidamente quién está usando la aplicación
        gustos_top3 = usuario.get("gustos_top3", [])
        gustos_source = usuario.get("gustos_source")
        gustos_texto = ""
        if gustos_source == "ml_inferred" and gustos_top3:
            gustos_texto = " · ".join(gustos_top3[:4])
            # st.caption(f"Tus gustos detectados (ML): {gustos_texto}")
        html_identidad = f"""
            <div style="
                background: rgba(0,31,63,0.6);
                border: 1px solid #1a3a5f;
                border-radius: 8px;
                padding: 8px 16px;
                margin-top: 18px;
                text-align: center;
            ">
                <span style="color:#B8860B; font-weight:1200; font-size:1.8rem;">{nombre_mostrar}</span>
                <span style="color:#aaa; font-size:1.2rem;"> · ID: {id_usuario}</span></br>
                <span style="color:#aaa; font-size:1.2rem;"> · Tus gustos detectados (ML): {gustos_texto}</span>
            </div>
        """
        st.markdown(html_identidad, unsafe_allow_html=True)

    with col_logout:
        st.write("")  # Espaciado para alinear verticalmente el botón
        if st.button("Cerrar Sesión", key="btn_logout_usuario"):
            # Limpiar el estado de sesión y recargar la página
            st.session_state["autenticado"] = False
            st.session_state["usuario_actual"] = None
            st.session_state["role"] = None
            st.rerun()

    # Routing: si el usuario activó la tragaperras, redirigir a esa vista
    if st.session_state.get("vista_serendipia", False):
        render_tragaperras(id_usuario)
        return

    st.markdown(
        "Descubre Películas y Series gracias a nuestro Recomendador de Inteligencia Artificial."
    )

    # Botón de acceso a la tragaperras
    col_btn_slot, _ = st.columns([2, 8])
    with col_btn_slot:
        st.markdown(
            """
            <style>
            div[data-testid="stButton"] button[kind="primary"] {
                background: linear-gradient(135deg, #8B0000 0%, #B8860B 100%);
                border: 2px solid #FFD700;
                border-radius: 12px;
                font-size: 1.15rem;
                font-weight: 700;
                letter-spacing: 0.05em;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "🎰  Joya Oculta",
            key="btn_open_tragaperras",
            type="primary",
            use_container_width=True,
        ):
            st.session_state["vista_serendipia"] = True
            st.session_state["serendipia_resultado"] = None
            st.rerun()

    # 3. Barra de búsqueda general
    texto_busqueda = st.text_input("Busca por título...", key="input_busqueda_usuario")

    # 4. Carga de los catálogos de películas y series
    datos_peliculas, datos_series = cargar_datos_catalogo()

    # 5. Estructura principal en pestañas para separar películas de series
    pestana_peliculas, pestana_series = st.tabs(["Películas", "Series"])

    with pestana_peliculas:
        dibujar_contenido_pestana(
            dataframe=datos_peliculas,
            termino_busqueda=texto_busqueda,
            es_pelicula=True,
            identificador_usuario=id_usuario,
            prefijo_clave_streamlit="usr_mov",
        )

    with pestana_series:
        dibujar_contenido_pestana(
            dataframe=datos_series,
            termino_busqueda=texto_busqueda,
            es_pelicula=False,
            identificador_usuario=id_usuario,
            prefijo_clave_streamlit="usr_tv",
        )


# ##############################################################################
#  Funciones auxiliares (Uso interno de este módulo)
# ##############################################################################


@st.cache_data
def cargar_datos_catalogo():
    """
    Carga los datasets finales en memoria.
    Usa caché (@st.cache_data) para evitar leer los CSV cada vez que el usuario interactúa con la UI.
    Devuelve dos DataFrames: uno para películas y otro para series.
    """
    ruta_peliculas = "src/data/ready/dataset_final_movies.csv"
    ruta_series = "src/data/ready/dataset_final_shows.csv"

    # Inicializar DataFrames vacíos por si fallan las cargas
    df_peliculas = pd.DataFrame()
    df_series = pd.DataFrame()

    if os.path.exists(ruta_peliculas):
        try:
            df_peliculas = pd.read_csv(
                ruta_peliculas, on_bad_lines="skip", engine="python"
            )
        except Exception as e:
            # En un entorno productivo, aquí registraríamos el error en un log (logger.error)
            print(f"Error cargando películas: {e}")

    if os.path.exists(ruta_series):
        try:
            df_series = pd.read_csv(ruta_series, on_bad_lines="skip", engine="python")
        except Exception as e:
            print(f"Error cargando series: {e}")

    return df_peliculas, df_series


def dibujar_tarjetas_contenido(
    dataframe, limite_mostrar=10, prefijo_clave="tarjeta", columna_fecha="fecha_estreno"
):
    """
    Dibuja una cuadrícula de pósteres de películas/series a partir de un DataFrame.
    """
    if dataframe.empty:
        st.info("No hay datos disponibles para mostrar en esta sección.")
        return

    # Usamos 5 columnas para que quepan bien los pósteres
    NUMERO_COLUMNAS = 5
    columnas_diseno = st.columns(NUMERO_COLUMNAS)

    # Iteramos solo sobre las primeras filas según el límite establecido
    for indice, (_, fila_datos) in enumerate(dataframe.head(limite_mostrar).iterrows()):
        columna_actual = columnas_diseno[indice % NUMERO_COLUMNAS]

        with columna_actual:
            # 1. Determinar URL del póster (usando placeholder si no existe)
            url_poster = "https://via.placeholder.com/300x450.png?text=Sin+Poster"
            ruta_poster_bd = fila_datos.get("poster_path")

            if pd.notna(ruta_poster_bd) and str(ruta_poster_bd).strip() != "":
                url_poster = f"https://image.tmdb.org/t/p/w500{ruta_poster_bd}"

            st.image(url_poster, use_container_width=True)

            # 2. Formatear y mostrar el Título (truncamos si es muy largo para no romper el diseño)
            titulo_obra = str(fila_datos.get("titulo", "Sin Título"))
            if len(titulo_obra) > 30:
                titulo_obra = titulo_obra[:27] + "..."

            # 3. Extraer el Año de estreno
            valor_fecha = str(fila_datos.get(columna_fecha, ""))
            año_estreno = valor_fecha[
                :4
            ]  # Tomamos solo los primeros 4 caracteres (YYYY)

            if año_estreno and año_estreno.isdigit():
                st.markdown(f"**{titulo_obra}** ({año_estreno})")
            else:
                st.markdown(f"**{titulo_obra}**")

            # 4. Mostrar Puntuación Media
            puntuacion = fila_datos.get("vote_average", 0)
            if pd.notna(puntuacion) and puntuacion > 0:
                st.caption(f"{puntuacion} / 10")

            # 5. Botón interactivo para ver más detalles (sinopsis)
            clave_boton_unica = f"{prefijo_clave}_btn_{indice}"
            if st.button("Ver sinopsis", key=clave_boton_unica):
                texto_sinopsis = str(
                    fila_datos.get(
                        "overview", "Sin sinopsis disponible para este título."
                    )
                )
                st.toast(texto_sinopsis)


def solicitar_y_dibujar_recomendaciones(
    identificador_usuario, prefijo_clave="ia_smart"
):
    """
    Solicita recomendaciones al backend FastAPI y las muestra visualmente.
    Usa el Smart selector para ofrecer el mejor modelo según el historial del usuario.
    """
    if not identificador_usuario or identificador_usuario == "?":
        st.info(
            "Tu perfil no tiene un ID válido asociado para generar recomendaciones personalizadas."
        )
        return

    try:
        url_endpoint = f"{API_BASE_URL}/recomendar/smart/{identificador_usuario}"
        respuesta = requests.get(
            url_endpoint, params={"n": 10}, timeout=10
        )  # 10s de timeout

        if respuesta.status_code == 200:
            datos_json = respuesta.json()
            lista_recomendaciones = datos_json.get("recomendaciones", [])
            mensaje_backend = datos_json.get("mensaje", "")

            # Si hay recomendaciones devueltas, las pintamos
            if lista_recomendaciones:
                columnas_ui = st.columns(5)

                for indice, recomendacion in enumerate(lista_recomendaciones):
                    columna_actual = columnas_ui[indice % 5]

                    with columna_actual:
                        # Gestión del póster
                        poster = recomendacion.get("poster_path", "")
                        if poster and str(poster) != "nan":
                            st.image(
                                f"https://image.tmdb.org/t/p/w500{poster}",
                                use_container_width=True,
                            )
                        else:
                            st.image(
                                "https://via.placeholder.com/300x450.png?text=Sin+Poster",
                                use_container_width=True,
                            )

                        # Título de la recomendación
                        titulo_rec = recomendacion.get("titulo", "Sin Título")
                        if len(titulo_rec) > 30:
                            titulo_rec = titulo_rec[:27] + "..."

                        st.markdown(f"**{titulo_rec}**")
                        st.caption(
                            f"Predicción IA: {recomendacion.get('predicted_rating', 'N/A')} / 5.0"
                        )

                        # Botón sinopsis
                        if st.button(
                            "Ver sinopsis", key=f"{prefijo_clave}_btn_{indice}"
                        ):
                            st.toast(
                                recomendacion.get(
                                    "overview", "Sin sinopsis disponible."
                                )
                            )

                # Mostramos qué modelo se utilizó por debajo (ej. LightGCN, SVD...)
                info_modelo_usado = datos_json.get("selector", "")
                if info_modelo_usado:
                    st.caption(f"Modelo utilizado: {info_modelo_usado}")

            # Si el modelo contestó pero indica falta de datos (cold start)
            elif "insufficient_data" in datos_json or "No alcanzas" in mensaje_backend:
                st.info(f"**Nota del motor de recomendaciones**: {mensaje_backend}")
            else:
                st.info(
                    "No se encontraron recomendaciones para tu perfil en este momento."
                )

        # Manejo de error 503 (Servicio no disponible - Modelo cargando)
        elif respuesta.status_code == 503:
            detalle_error = respuesta.json().get(
                "detail", "Error desconocido del servicio."
            )
            st.warning(f"El sistema de IA reporta: {detalle_error}")
        else:
            st.warning("No se pudieron obtener recomendaciones en este momento.")

    except requests.exceptions.RequestException as error_conexion:
        st.warning(
            "No se pudo establecer conexión con el motor de recomendaciones. Asegúrate de que el backend esté en ejecución."
        )


def dibujar_contenido_pestana(
    dataframe,
    termino_busqueda,
    es_pelicula=True,
    identificador_usuario=None,
    prefijo_clave_streamlit="usr",
):
    """
    Estructura principal de visualización dentro de cada pestaña (Películas o Series).
    Decide en qué orden pintar las secciones: Resultados de búsqueda, Recomendaciones, Top y Populares.
    """
    # Determinamos prefijos y nombres de columnas base según si es película o serie
    prefijo_tipo = f"{prefijo_clave_streamlit}_{'mov' if es_pelicula else 'tv'}"
    columna_fecha = "fecha_estreno" if es_pelicula else "first_air_date"

    # --- FLUJO DE BÚSQUEDA ---
    # Si el usuario escribió algo, solo mostramos resultados que coincidan
    if termino_busqueda:
        st.subheader(f"Resultados de búsqueda para: '{termino_busqueda}'")

        if not dataframe.empty and "titulo" in dataframe.columns:
            # Filtramos de forma insensible a mayúsculas/minúsculas usando regex
            mascara = dataframe["titulo"].str.contains(
                termino_busqueda, case=False, na=False
            )
            resultados_filtrados = dataframe[mascara]

            if not resultados_filtrados.empty:
                dibujar_tarjetas_contenido(
                    dataframe=resultados_filtrados,
                    limite_mostrar=12,
                    prefijo_clave=f"{prefijo_tipo}_search",
                    columna_fecha=columna_fecha,
                )
            else:
                st.info("No se encontraron coincidencias para tu búsqueda.")
        else:
            st.info("El catálogo no está disponible para realizar la búsqueda.")
        return  # Termina temprano: si hay búsqueda no mostramos las secciones por defecto

    # --- FLUJO NORMAL (Sin búsqueda activa) ---

    # Sección 1: Recomendaciones Personalizadas (IA)
    st.subheader("Recomendado para ti")
    if es_pelicula:
        solicitar_y_dibujar_recomendaciones(
            identificador_usuario, prefijo_clave=f"{prefijo_tipo}_ia"
        )
    else:
        # Por regla del negocio actual, las recomendaciones de IA son para películas
        st.info(
            "Las recomendaciones personalizadas para series están en desarrollo. ¡Disfruta del catálogo mientras tanto!"
        )

    st.divider()

    # Sección 2: Top Mejor Puntuadas
    st.subheader("Mejor puntuadas por la comunidad")
    # Aseguramos que existan las columnas para calcular el Top
    if not dataframe.empty and all(
        col in dataframe.columns for col in ["vote_average", "vote_count"]
    ):
        # Filtramos por mínimo de votos para evitar que algo con 1 voto de '10/10' ensucie el ranking
        datos_ranking = dataframe[dataframe["vote_count"] > 500].sort_values(
            by="vote_average", ascending=False
        )
        dibujar_tarjetas_contenido(
            dataframe=datos_ranking,
            limite_mostrar=10,
            prefijo_clave=f"{prefijo_tipo}_top",
            columna_fecha=columna_fecha,
        )
    else:
        st.info("No hay datos suficientes para generar el ranking de la comunidad.")

    st.divider()

    # Sección 3: Lo Más Visto (Populares)
    st.subheader("Lo más popular")
    if not dataframe.empty and "vote_count" in dataframe.columns:
        # Aquí usamos 'vote_count' como un proxy de "popularidad / vistas"
        datos_mas_vistos = dataframe.sort_values(by="vote_count", ascending=False)
        dibujar_tarjetas_contenido(
            dataframe=datos_mas_vistos,
            limite_mostrar=10,
            prefijo_clave=f"{prefijo_tipo}_pop",
            columna_fecha=columna_fecha,
        )
    else:
        st.info("No hay datos suficientes para determinar lo más popular.")


# ##############################################################################
#  Vista Tragaperras — Joya Oculta (Serendipia)
# ##############################################################################


_CSS_TRAGAPERRAS = """
<style>
.slot-header {
    text-align: center;
    padding: 24px 0 8px 0;
}
.slot-title {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #FFD700, #FF8C00, #FFD700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 0.08em;
    text-shadow: none;
}
.slot-subtitle {
    color: #aaa;
    font-size: 1.1rem;
    margin-top: 4px;
}
.slot-card {
    background: rgba(0, 20, 45, 0.85);
    border: 2px solid #B8860B;
    border-radius: 16px;
    padding: 18px 14px 14px 14px;
    text-align: center;
    height: 100%;
}
.slot-card img { border-radius: 8px; }
.slot-movie-title {
    color: #FFD700;
    font-weight: 700;
    font-size: 1rem;
    margin-top: 10px;
    min-height: 2.4em;
}
.slot-genre-badge {
    display: inline-block;
    background: rgba(184, 134, 11, 0.25);
    border: 1px solid #B8860B;
    color: #FFD700;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.8rem;
    margin-top: 6px;
}
.slot-score-bar-wrap {
    background: rgba(255,255,255,0.08);
    border-radius: 8px;
    height: 8px;
    margin-top: 10px;
    overflow: hidden;
}
.slot-score-bar-fill {
    height: 8px;
    border-radius: 8px;
    background: linear-gradient(90deg, #B8860B, #FFD700);
}
.slot-score-label {
    color: #aaa;
    font-size: 0.75rem;
    margin-top: 4px;
}
</style>
"""


def _tarjeta_serendipia_html(titulo: str, poster_url: str, genero: str, score: float, overview: str, año: str) -> str:
    """Genera el HTML de una tarjeta de película para la tragaperras."""
    pct = int(score * 100)
    titulo_safe = titulo[:35] + "..." if len(titulo) > 35 else titulo
    overview_safe = overview[:120] + "..." if overview and len(overview) > 120 else (overview or "")
    return f"""
    <div class="slot-card">
        <img src="{poster_url}" width="100%" style="border-radius:8px;">
        <div class="slot-movie-title">{titulo_safe} <span style='color:#aaa;font-weight:400;font-size:0.85rem;'>({año})</span></div>
        <div><span class="slot-genre-badge">{genero}</span></div>
        <div class="slot-score-bar-wrap">
            <div class="slot-score-bar-fill" style="width:{pct}%"></div>
        </div>
        <div class="slot-score-label">Serendipity: {score:.2f}</div>
        <div style="color:#ccc;font-size:0.78rem;margin-top:8px;text-align:left;">{overview_safe}</div>
    </div>
    """


def render_tragaperras(id_usuario):
    """Vista inmersiva de la tragaperras Joya Oculta."""
    st.markdown(_CSS_TRAGAPERRAS, unsafe_allow_html=True)

    # Cabecera
    st.markdown(
        """
        <div class="slot-header">
            <div class="slot-title">🎰 JOYA OCULTA 🎰</div>
            <div class="slot-subtitle">Descubre películas sorprendentes que quizás no conocías</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Botones de control
    col_volver, col_tirar, col_empty = st.columns([1, 2, 1])
    with col_volver:
        if st.button("← Volver", key="btn_volver_tragaperras", use_container_width=True):
            st.session_state["vista_serendipia"] = False
            st.session_state["serendipia_resultado"] = None
            st.rerun()

    with col_tirar:
        tirar = st.button(
            "🎰  TIRAR",
            key="btn_tirar_tragaperras",
            type="primary",
            use_container_width=True,
        )

    # Cargar catálogo para enriquecer los resultados con poster/título/overview
    df_peliculas, _ = cargar_datos_catalogo()

    # Si se pulsó TIRAR, llamar a la API
    if tirar:
        st.session_state["serendipia_resultado"] = None  # reset para animar
        with st.spinner("Buscando joyas ocultas..."):
            try:
                resp = requests.get(
                    f"{API_BASE_URL}/api/serendipia/{id_usuario}", timeout=10
                )
                if resp.status_code == 200:
                    st.session_state["serendipia_resultado"] = resp.json()
                elif resp.status_code == 404:
                    st.session_state["serendipia_resultado"] = {
                        "error": resp.json().get("detail", "No hay géneros favoritos registrados.")
                    }
                else:
                    st.session_state["serendipia_resultado"] = {
                        "error": f"Error del servidor ({resp.status_code}): {resp.json().get('detail', '')}"
                    }
            except requests.exceptions.RequestException:
                st.session_state["serendipia_resultado"] = {
                    "error": "No se pudo conectar con el backend. ¿Está en ejecución?"
                }

    # Mostrar resultado si existe
    resultado = st.session_state.get("serendipia_resultado")
    if resultado is None:
        st.markdown(
            "<div style='text-align:center;color:#555;margin-top:40px;font-size:1.3rem;'>"
            "Pulsa <b style='color:#FFD700;'>TIRAR</b> para descubrir tu próxima joya oculta"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    if "error" in resultado:
        st.error(resultado["error"])
        return

    recomendaciones = resultado.get("recomendaciones", [])
    generos_favoritos = resultado.get("generos_favoritos", [])

    if generos_favoritos:
        st.markdown(
            f"<div style='text-align:center;color:#aaa;margin-bottom:16px;'>"
            f"Basado en tus géneros favoritos: "
            f"<b style='color:#FFD700;'>{' · '.join(generos_favoritos)}</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    cols = st.columns(3)
    for idx, rec in enumerate(recomendaciones):
        movie_id = rec.get("movie_id")
        genero = rec.get("genre", "")
        score = rec.get("serendipity_score", 0.0)

        # Buscar datos enriquecidos en el catálogo local
        titulo = f"Película {movie_id}"
        poster_url = "https://via.placeholder.com/300x450.png?text=Sin+Poster"
        overview = ""
        año = ""

        if not df_peliculas.empty and "tmdb_id" in df_peliculas.columns:
            fila = df_peliculas[df_peliculas["tmdb_id"] == movie_id]
            if not fila.empty:
                f = fila.iloc[0]
                titulo = str(f.get("titulo", titulo))
                overview = str(f.get("overview", ""))
                if overview == "nan":
                    overview = ""
                valor_fecha = str(f.get("fecha_estreno", ""))
                año = valor_fecha[:4] if valor_fecha and valor_fecha[:4].isdigit() else ""
                ruta_poster = f.get("poster_path", "")
                if pd.notna(ruta_poster) and str(ruta_poster).strip():
                    poster_url = f"https://image.tmdb.org/t/p/w500{ruta_poster}"

        with cols[idx]:
            st.markdown(
                _tarjeta_serendipia_html(titulo, poster_url, genero, score, overview, año),
                unsafe_allow_html=True,
            )
