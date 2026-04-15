"""
Vista de Usuario — Módulo importable.
Solo muestra la recomendación Smart Auto personalizada + catálogo.
Sin sidebar, sin selector de modelo, sin selector de ID.
"""

import streamlit as st
import requests
import pandas as pd
import os


def render():
    """Punto de entrada que app_ui.py llama tras verificar rol == 'user'."""
    usuario = st.session_state["usuario_actual"]
    nombre_mostrar = usuario.get("username", f"Usuario #{usuario.get('id_usuario')}")
    id_usuario = usuario.get("id_usuario", "?")

    # ── Barra superior: título + etiqueta identidad + botón logout ──
    col_titulo, col_info, col_logout = st.columns([6, 3, 1])
    with col_titulo:
        st.title(f"Bienvenido a SPIRE, {nombre_mostrar}!")
    with col_info:
        st.markdown(
            f"""
            <div style="
                background: rgba(0,31,63,0.6);
                border: 1px solid #1a3a5f;
                border-radius: 8px;
                padding: 8px 16px;
                margin-top: 18px;
                text-align: center;
            ">
                <span style="color:#B8860B; font-weight:600; font-size:2.1rem;">{nombre_mostrar}</span>
                <span style="color:#aaa; font-size:1.1rem;"> · ID: {id_usuario}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_logout:
        st.write("")  # Espaciado
        if st.button("Cerrar Sesión", key="logout_usuario"):
            st.session_state["autenticado"] = False
            st.session_state["usuario_actual"] = None
            st.session_state["role"] = None
            st.rerun()

    st.markdown(
        "Descubre Películas y Series gracias a nuestro Recomendador de Inteligencia Artificial."
    )

    # --- BUSCADOR GLOBAL ---
    search_query = st.text_input(
        "Busca por título o palabras clave...", key="search_user"
    )

    # --- CARGA DE DATOS ---
    df_movies, df_shows = _load_catalog_data()

    # --- PESTAÑAS PRINCIPALES ---
    tab_movies, tab_shows = st.tabs(["Películas", "Series"])

    with tab_movies:
        _render_tab_content(
            df_movies,
            search_query,
            is_movie=True,
            user_id=id_usuario,
            key_prefix="usr_mov",
        )

    with tab_shows:
        _render_tab_content(
            df_shows,
            search_query,
            is_movie=False,
            user_id=id_usuario,
            key_prefix="usr_tv",
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Funciones auxiliares (privadas al módulo)
# ══════════════════════════════════════════════════════════════════════════════


@st.cache_data
def _load_catalog_data():
    movies_path = "src/data/ready/dataset_final_movies.csv"
    shows_path = "src/data/ready/dataset_final_shows.csv"

    df_movies = pd.DataFrame()
    df_shows = pd.DataFrame()

    if os.path.exists(movies_path):
        try:
            df_movies = pd.read_csv(movies_path, on_bad_lines="skip", engine="python")
        except Exception:
            pass

    if os.path.exists(shows_path):
        try:
            df_shows = pd.read_csv(shows_path, on_bad_lines="skip", engine="python")
        except Exception:
            pass

    return df_movies, df_shows


def _render_cards(df, limit=8, key_prefix="card", date_col="fecha_estreno"):
    """Dibuja un grid de postales con poster, título y botón de sinopsis."""
    if df.empty:
        st.info("No hay datos disponibles para mostrar.")
        return

    cols = st.columns(4)
    for idx, (_, row) in enumerate(df.head(limit).iterrows()):
        with cols[idx % 4]:
            # Poster
            poster_url = "https://via.placeholder.com/300x450.png?text=Sin+Poster"
            if pd.notna(row.get("poster_path")) and str(row.get("poster_path")) != "":
                poster_url = f"https://image.tmdb.org/t/p/w500{row['poster_path']}"
            st.image(poster_url, use_container_width=True)

            # Título truncado
            titulo = str(row.get("titulo", "Sin Título"))
            if len(titulo) > 30:
                titulo = titulo[:27] + "..."

            # Año
            year = str(row.get(date_col, ""))[:4]
            if year and year != "nan":
                st.markdown(f"**{titulo}** ({year})")
            else:
                st.markdown(f"**{titulo}**")

            # Nota
            nota = row.get("vote_average", 0)
            if nota:
                st.caption(f"{nota}")

            # Botón de sinopsis
            if st.button("Ver sinopsis", key=f"{key_prefix}_{idx}"):
                st.toast(str(row.get("overview", "Sin sinopsis disponible.")))


def _render_recomendaciones_smart(user_id, key_prefix="smart"):
    """Llama al endpoint Smart y pinta las recomendaciones automáticas."""
    if not user_id:
        st.info("Tu perfil no tiene un ID asociado para generar recomendaciones.")
        return

    try:
        resp = requests.get(
            f"http://127.0.0.1:8000/recomendar/smart/{user_id}", params={"n": 8}
        )
        if resp.status_code == 200:
            datos = resp.json()
            recomendaciones = datos.get("recomendaciones", [])
            mensaje = datos.get("mensaje", "")

            if recomendaciones:
                cols_ia = st.columns(4)
                for idx, rec in enumerate(recomendaciones):
                    with cols_ia[idx % 4]:
                        poster = rec.get("poster_path", "")
                        if poster and poster != "" and poster != "nan":
                            st.image(
                                f"https://image.tmdb.org/t/p/w500{poster}",
                                use_container_width=True,
                            )
                        else:
                            st.image(
                                "https://via.placeholder.com/300x450.png?text=Sin+Poster",
                                use_container_width=True,
                            )
                        titulo_rec = rec.get("titulo", "Sin Título")
                        if len(titulo_rec) > 30:
                            titulo_rec = titulo_rec[:27] + "..."
                        st.markdown(f"**{titulo_rec}**")
                        st.caption(f"Predicción IA: {rec['predicted_rating']} / 5.0")
                        if st.button("Ver sinopsis", key=f"{key_prefix}_{idx}"):
                            st.toast(rec.get("overview", "Sin sinopsis disponible."))

                # Info del selector Smart
                selector_info = datos.get("selector", "")
                if selector_info:
                    st.caption(f"{selector_info}")
            elif "insufficient_data" in datos or "No alcanzas" in mensaje:
                st.info(f"**Requisito del Modelo**: {mensaje}")
            else:
                st.info("No se encontraron recomendaciones para tu perfil.")
        elif resp.status_code == 503:
            error_det = resp.json().get("detail", "Error desconocido")
            st.warning(f"El Backend reporta: {error_det}")
        else:
            st.warning("No se pudieron obtener recomendaciones.")
    except requests.exceptions.ConnectionError:
        st.warning("No se pudo conectar con el Backend para recomendaciones.")


def _render_tab_content(df, search, is_movie=True, user_id=None, key_prefix="usr"):
    """Dibuja las secciones dentro de una pestaña: IA Smart, Top Rated, Más Visto."""
    prefix = f"{key_prefix}_{'m' if is_movie else 't'}"
    date_col = "fecha_estreno" if is_movie else "first_air_date"

    # Si hay búsqueda activa, mostramos los resultados filtrados
    if search:
        st.subheader("Resultados de búsqueda")
        if not df.empty and "titulo" in df.columns:
            mask = df["titulo"].str.contains(search, case=False, na=False)
            resultados = df[mask]
            if not resultados.empty:
                _render_cards(
                    resultados,
                    limit=12,
                    key_prefix=f"{prefix}_search",
                    date_col=date_col,
                )
            else:
                st.info("No se encontraron resultados para tu búsqueda.")
        return

    # --- Sección 1: Recomendaciones IA Smart ---
    st.subheader("Recomendado para ti")
    if is_movie:
        _render_recomendaciones_smart(user_id, key_prefix=f"{prefix}_ia")
    else:
        st.info(
            "Las recomendaciones de series están en desarrollo. De momento disfruta del catálogo."
        )

    st.divider()

    # --- Sección 2: Top mejor puntuadas ---
    st.subheader("Mejor puntuadas por la comunidad")
    if not df.empty and "vote_average" in df.columns and "vote_count" in df.columns:
        df_top = df[df["vote_count"] > 500].sort_values(
            by="vote_average", ascending=False
        )
        _render_cards(df_top, limit=8, key_prefix=f"{prefix}_top", date_col=date_col)
    else:
        st.info("No hay datos suficientes para generar el ranking.")

    st.divider()

    # --- Sección 3: Lo más visto ---
    st.subheader("Lo más visto")
    if not df.empty and "vote_count" in df.columns:
        df_popular = df.sort_values(by="vote_count", ascending=False)
        _render_cards(
            df_popular, limit=8, key_prefix=f"{prefix}_pop", date_col=date_col
        )
    else:
        st.info("No hay datos suficientes para generar lo más visto.")
