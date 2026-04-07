import streamlit as st
import requests
import datetime


# Validar si el usuario está logueado en Streamlit Session State
if "usuario_autenticado" not in st.session_state:
    st.session_state["usuario_autenticado"] = False
    st.session_state["usuario_actual"] = None

# ---- ESTILOS PERSONALIZADOS (Azul Marino y Oro) ----
st.markdown(
    """
    <style>
    /* Fondo y colores principales */
    .stApp {
        background-color: #001220;
        color: #f0f0f0;
    }
    
    /* Títulos en Dorado Premium (Más profundo) */
    h1, h2, h3, .stSubheader {
        color: #B8860B !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    /* Texto normal */
    p, span, label {
        color: #e0e0e0 !important;
    }
    
    /* Estilo para los Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: rgba(0, 31, 63, 0.5);
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: transparent;
        border-radius: 5px;
        color: #f0f0f0;
        border: 1px solid transparent;
        transition: all 0.2s;
    }
    /* Tab seleccionado: Oro oscuro con texto azul muy oscuro para contraste */
    .stTabs [aria-selected="true"] {
        background-color: #B8860B !important;
        color: #001220 !important;
        border-color: #B8860B !important;
        font-weight: bold;
    }
    
    /* Botones Premium */
    div.stButton > button {
        background-color: #B8860B !important;
        color: #001220 !important;
        font-weight: bold !important;
        border: 1px solid #B8860B !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.3s ease !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    div.stButton > button:hover {
        background-color: #001f3f !important;
        color: #B8860B !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 20px rgba(184, 134, 11, 0.4) !important;
    }
    
    /* Inputs y widgets estilizados */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stMultiSelect>div>div>div, .stDateInput>div>div>input {
        background-color: #001f3f !important;
        color: white !important;
        border: 1px solid #1a3a5f !important;
        border-radius: 8px !important;
    }
    
    /* Checkboxes personalizados (Grid de gustos) */
    .stCheckbox > label > div[data-testid="stMarkdownContainer"] > p {
        color: #f0f0f0 !important;
        font-size: 0.9rem;
    }
    /* Estilo del cuadro del checkbox */
    span[data-baseweb="checkbox"] > div {
        border-color: #B8860B !important;
    }
    /* Checkbox marcado: fondo oro oscuro */
    div[data-checked="true"] {
        background-color: #B8860B !important;
    }
    
    /* Mensajes de información y éxito */
    .stAlert {
        border-radius: 10px !important;
        border: 1px solid #B8860B !important;
        background-color: rgba(0, 31, 63, 0.8) !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if not st.session_state["usuario_autenticado"]:
    st.title("BIENVENIDO A SPIRE STREAMING")
    st.markdown(
        "Accede a la experiencia definitiva de cine o crea tu cuenta exclusiva."
    )

    tab_login, tab_registro = st.tabs(["Iniciar Sesión", "Registrarse"])

    # --- PESTAÑA LOGIN ---
    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            submit_button = st.form_submit_button("Entrar")

            if submit_button:
                payload = {"username": username, "password": password}
                try:
                    response = requests.post(
                        "http://localhost:8000/login", json=payload
                    )

                    if response.status_code == 200:
                        datos_usuario = response.json()
                        st.session_state["usuario_autenticado"] = True
                        st.session_state["usuario_actual"] = datos_usuario["user"]
                        st.success("¡Login exitoso!")
                        st.rerun()  # Recarga la página para mostrar el catálogo
                    elif response.status_code == 401:
                        st.error(
                            "Credenciales incorrectas. Verifica tu usuario o regístrate en la otra pestaña."
                        )
                    else:
                        st.error(f"Error en el servidor: {response.status_code}")
                except Exception as e:
                    st.error(f"Error de conexión: {e}")

    # --- PESTAÑA REGISTRO ---
    with tab_registro:
        st.subheader("Crea tu cuenta")

        # Intentar cargar géneros para el selector de gustos
        opciones_generos = {}
        try:
            resp_gen = requests.get("http://localhost:8000/genres")
            if resp_gen.status_code == 200:
                for g in resp_gen.json():
                    opciones_generos[g["name"]] = g["id"]
        except Exception as e:
            st.error(f"Error cargando categorías: {e}")

        with st.form("register_form"):
            col1, col2 = st.columns(2)
            with col1:
                new_username = st.text_input(
                    "Usuario", placeholder="Tu nombre de usuario"
                )
                new_email = st.text_input("Email", placeholder="ejemplo@correo.com")
                new_password = st.text_input("Contraseña", type="password")

            with col2:
                # Ajustamos el calendario para cubrir desde 1900
                new_fecha_nac = st.date_input(
                    "Fecha de Nacimiento",
                    value=None,
                    min_value=datetime.date(1900, 1, 1),
                    max_value=datetime.date.today(),
                )
                new_sexo = st.selectbox(
                    "Sexo",
                    ["Hombre", "Mujer", "Otro"],
                    index=None,
                    placeholder="Selecciona...",
                )
                new_password_confirm = st.text_input(
                    "Confirmar Contraseña", type="password"
                )

            st.write("---")
            st.markdown("**¿Qué generos te gusta ver?**")

            # --- REFACTOR: Grid de Checkboxes para Categorías ---
            if not opciones_generos:
                st.caption("Cargando categorías o servidor no disponible...")

            # Mostramos las categorías en un grid de 4 columnas para que quepa bien
            if opciones_generos:
                generos_lista = list(opciones_generos.keys())
                num_cols = 4
                rows = [
                    generos_lista[i : i + num_cols]
                    for i in range(0, len(generos_lista), num_cols)
                ]

                for row_genres in rows:
                    cols = st.columns(num_cols)
                    for i, genre_name in enumerate(row_genres):
                        # Usamos key para persistir el valor en session_state y capturarlo al enviar
                        cols[i].checkbox(genre_name, key=f"genre_{genre_name}")

            btn_register = st.form_submit_button("REGISTRARSE AHORA")

            if btn_register:
                # Recopilamos los gustos marcados desde el session_state
                gustos_actuales = [
                    opciones_generos[name]
                    for name in opciones_generos
                    if st.session_state.get(f"genre_{name}")
                ]

                if not new_username or not new_password or not new_email:
                    st.warning(
                        "Por favor, rellena los campos obligatorios (Usuario, Email y Contraseña)."
                    )
                elif new_password != new_password_confirm:
                    st.error("Las contraseñas no coinciden.")
                else:
                    # Preparar payload
                    payload = {
                        "username": new_username,
                        "email": new_email,
                        "password": new_password,
                        "fecha_nacimiento": str(new_fecha_nac)
                        if new_fecha_nac
                        else None,
                        "sexo": new_sexo,
                        "intereses": gustos_actuales,
                    }

                    try:
                        response = requests.post(
                            "http://127.0.0.1:8000/register", json=payload
                        )
                        if response.status_code == 200:
                            st.success(
                                f"¡Bienvenido, {new_username}! Tu cuenta ha sido creada."
                            )
                            st.info(
                                "Ya puedes ir a la pestaña 'Iniciar Sesión' para entrar."
                            )
                        else:
                            error_detail = response.json().get(
                                "detail", "Error desconocido"
                            )
                            st.error(f"Error al registrar: {error_detail}")
                    except Exception as e:
                        st.error(f"No se pudo conectar con el servidor: {e}")
#################################################################################################
# ---- PANTALLA PRINCIPAL DEL USUARIO (Logueado) ----

else:
    import pandas as pd
    import os

    usuario = st.session_state["usuario_actual"]
    nombre_mostrar = usuario.get("username", f"Usuario #{usuario.get('id_usuario')}")

    # Barra superior con Logout
    col_titulo, col_logout = st.columns([8, 2])
    with col_titulo:
        st.title(f"Bienvenido a SPIRE, {nombre_mostrar}!")
    with col_logout:
        st.write("")  # Espacio
        if st.button("Cerrar Sesión"):
            st.session_state["usuario_autenticado"] = False
            st.session_state["usuario_actual"] = None
            st.rerun()

    st.markdown(
        "Descubre Películas y Series gracias a nuestro Recomendador de Inteligencia Artificial."
    )

    # --- BUSCADOR GLOBAL ---
    search_query = st.text_input("Busca por título o palabras clave...")

    # --- SELECTOR DE MODELO DE IA ---
    modelo_ia = st.sidebar.selectbox(
        "Motor de Recomendación",
        [
            "SVD (Rápido)",
            "KNN + Cosine (Explicable)",
            "Wide & Deep (Profundo)",
            "Content-Based (Cold-Start)",
            "Implicit BPR (Ranking Top)",
            "NCF (Deep Learning)",
            "Two Towers (Retrieval)",
        ],
        index=4,  # Ponemos el BPR por defecto porque es el mejor
    )
    mapa_endpoints = {
        "SVD (Rápido)": "recomendar",
        "KNN + Cosine (Explicable)": "recomendar/knn",
        "Wide & Deep (Profundo)": "recomendar/wnd",
        "Content-Based (Cold-Start)": "recomendar/content",
        "Implicit BPR (Ranking Top)": "recomendar/implicit",
        "NCF (Deep Learning)": "recomendar/ncf",
        "Two Towers (Retrieval)": "recomendar/twotowers",
    }
    endpoint_ia = mapa_endpoints[modelo_ia]

    # --- DEV TOOL: Simular otro usuario ---
    id_simulado = st.sidebar.number_input(
        "Datos ratings - ID Usuario ",
        value=usuario.get("id_usuario", 1),
        step=1,
        help="Permite simular predicciones para IDs de súper-usuarios (ej. 9) que existen en el set de datos pero no en tu base de datos local.",
    )

    # --- CARGA DE DATOS ---
    @st.cache_data
    def load_catalog_data():
        movies_path = "src/data/ready/dataset_final_movies.csv"
        shows_path = "src/data/ready/dataset_final_shows.csv"

        df_movies = pd.DataFrame()
        df_shows = pd.DataFrame()

        if os.path.exists(movies_path):
            try:
                df_movies = pd.read_csv(
                    movies_path, on_bad_lines="skip", engine="python"
                )
            except Exception:
                pass

        if os.path.exists(shows_path):
            try:
                df_shows = pd.read_csv(shows_path, on_bad_lines="skip", engine="python")
            except Exception:
                pass

        return df_movies, df_shows

    df_movies, df_shows = load_catalog_data()

    # =====================================================================================
    # Dibuja un grid de postales con poster, título y sinopsis
    # =====================================================================================
    def render_cards(df, limit=8, key_prefix="card", date_col="fecha_estreno"):
        """Dibuja una fila de postales con poster, título y botón de sinopsis."""
        if df.empty:
            st.info("No hay datos disponibles para mostrar.")
            return

        cols = st.columns(4)
        for idx, (_, row) in enumerate(df.head(limit).iterrows()):
            with cols[idx % 4]:
                # Poster
                poster_url = "https://via.placeholder.com/300x450.png?text=Sin+Poster"
                if (
                    pd.notna(row.get("poster_path"))
                    and str(row.get("poster_path")) != ""
                ):
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

    # =====================================================================================
    # Renderiza las recomendaciones IA dentro de una pestaña
    # =====================================================================================
    def render_recomendaciones_ia(key_prefix="ia", endpoint="recomendar"):
        """Llama al Backend y pinta las recomendaciones del modelo SVD."""
        # Usamos el id_simulado del sidebar para pruebas con IA
        user_id_ia = id_simulado
        if not user_id_ia:
            st.info("Tu perfil no tiene un ID asociado para generar recomendaciones.")
            return

        try:
            resp_ia = requests.get(
                f"http://127.0.0.1:8000/{endpoint}/{user_id_ia}", params={"n": 8}
            )
            if resp_ia.status_code == 200:
                datos = resp_ia.json()
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
                            st.caption(
                                f" Predicción IA: {rec['predicted_rating']} / 5.0"
                            )
                            if st.button("Ver sinopsis", key=f"{key_prefix}_{idx}"):
                                st.toast(
                                    rec.get("overview", "Sin sinopsis disponible.")
                                )
                elif "insufficient_data" in datos or "No alcanzas" in mensaje:
                    st.info(f"**Requisito del Modelo**: {mensaje}")
                    st.caption(
                        "Esta red neuronal requiere que el usuario tenga un historial denso (1000+ valoraciones) para capturar patrones profundos de comportamiento."
                    )
                else:
                    st.info("No se encontraron recomendaciones para tu perfil.")
            elif resp_ia.status_code == 503:
                error_det = resp_ia.json().get("detail", "Error desconocido")
                st.warning(f"El Backend reporta: {error_det}")
            else:
                st.warning("No se pudieron obtener recomendaciones.")
        except requests.exceptions.ConnectionError:
            st.warning("No se pudo conectar con el Backend para recomendaciones.")

    # =====================================================================================
    # Contenido de una pestaña (recomendaciones + top rated + más vistos)
    # =====================================================================================
    def render_tab_content(df, search, is_movie=True, endpoint_ia="recomendar"):
        """Dibuja las 3 secciones dentro de una pestaña: IA, Top Rated, Más Visto."""
        prefix = "mov" if is_movie else "tv"
        date_col = "fecha_estreno" if is_movie else "first_air_date"

        # Si hay búsqueda activa, mostramos los resultados filtrados
        if search:
            st.subheader("Resultados de búsqueda")
            if not df.empty and "titulo" in df.columns:
                mask = df["titulo"].str.contains(search, case=False, na=False)
                resultados = df[mask]
                if not resultados.empty:
                    render_cards(
                        resultados,
                        limit=12,
                        key_prefix=f"{prefix}_search",
                        date_col=date_col,
                    )
                else:
                    st.info("No se encontraron resultados para tu búsqueda.")
            return

        # --- Sección 1: Recomendaciones IA ---
        st.subheader("Recomendado para ti")
        if is_movie:
            render_recomendaciones_ia(key_prefix=f"{prefix}_ia", endpoint=endpoint_ia)
        else:
            st.info(
                "Las recomendaciones de series están en desarrollo. De momento disfruta del catálogo."
            )

        st.divider()

        # --- Sección 2: Top mejor puntuadas ---
        st.subheader("Mejor puntuadas por la comunidad")
        if not df.empty and "vote_average" in df.columns and "vote_count" in df.columns:
            # Filtro mínimo de votos para que no salgan pelis con 1 voto y nota 10
            df_top = df[df["vote_count"] > 500].sort_values(
                by="vote_average", ascending=False
            )
            render_cards(df_top, limit=8, key_prefix=f"{prefix}_top", date_col=date_col)
        else:
            st.info("No hay datos suficientes para generar el ranking.")

        st.divider()

        # --- Sección 3: Lo más visto ---
        st.subheader("Lo más visto")
        if not df.empty and "vote_count" in df.columns:
            df_popular = df.sort_values(by="vote_count", ascending=False)
            render_cards(
                df_popular, limit=8, key_prefix=f"{prefix}_pop", date_col=date_col
            )
        else:
            st.info("No hay datos suficientes para generar lo más visto.")

    ###########################################################################################
    # --- PESTAÑAS PRINCIPALES ---
    ###########################################################################################

    tab_movies, tab_shows = st.tabs(["Películas", "Series"])

    with tab_movies:
        render_tab_content(
            df_movies, search_query, is_movie=True, endpoint_ia=endpoint_ia
        )

    with tab_shows:
        render_tab_content(
            df_shows, search_query, is_movie=False, endpoint_ia=endpoint_ia
        )
