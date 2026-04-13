import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import time
import extra_streamlit_components as stx  # Librería extra para manejar Cookies en el navegador

# Inicializamos el gestor de cookies para poder recordar al administrador
cookie_manager = stx.CookieManager()

# Configuramos el nombre de la pestaña del navegador y que ocupe el 100% del ancho de banda
st.set_page_config(page_title="Admin Panel - SPIRE", layout="wide")

# ---- ESTILOS PERSONALIZADOS (Azul Marino y Oro — Consistente con 2_Usuario) ----
st.markdown(
    """
    <style>
    /* Fondo y colores principales */
    .stApp {
        background-color: #001220;
        color: #f0f0f0;
    }

    /* Títulos en Dorado Premium */
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

    /* Inputs estilizados */
    .stTextInput>div>div>input, .stSelectbox>div>div>div {
        background-color: #001f3f !important;
        color: white !important;
        border: 1px solid #1a3a5f !important;
        border-radius: 8px !important;
    }

    /* DataFrames y tablas */
    .stDataFrame {
        border: 1px solid #1a3a5f;
        border-radius: 8px;
    }

    /* Métricas */
    [data-testid="stMetric"] {
        background-color: rgba(0, 31, 63, 0.5);
        border: 1px solid #1a3a5f;
        border-radius: 10px;
        padding: 15px;
    }
    [data-testid="stMetricLabel"] {
        color: #B8860B !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: rgba(0, 31, 63, 0.5);
        border-radius: 8px;
        color: #B8860B !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Intentamos leer todas las cookies instaladas actualmente en el navegador del usuario
cookies = cookie_manager.get_all()

# A veces las cookies tardan un pelín en cargar.
# Si no detectamos nada, pausamos 0.5 segundos y reintentamos.
if not cookies:
    time.sleep(0.5)
    cookies = cookie_manager.get_all()

# Buscamos específicamente si dentro de sus cookies existe la nuestra, llamada "spire_admin_session"
try:
    saved_cookie = cookies.get(cookie="spire_admin_session")
except Exception:
    saved_cookie = None

##########################################################################################
st.session_state["admin_autenticado"] = True  # añadido para pruebas, eliminar luego
##########################################################################################

# 'st.session_state' es una especie de memoria temporal RAM que Streamlit mantiene viva
# mientras interactúas con la página. Si recargas (F5), se borra.
# Aquí comprobamos si el admin ya tiene la variable de "autenticado" guardada en esa memoria.
if "admin_autenticado" not in st.session_state:
    # Si no la tenía, comprobamos si sí que tiene la cookie guardada en el disco de su navegador
    if saved_cookie:
        st.session_state["admin_autenticado"] = True
    else:
        st.session_state["admin_autenticado"] = False

# --- ZONA DE LOGIN ---
# Si en la memoria temporal dice que NO está autenticado, le mostramos el formulario
if not st.session_state["admin_autenticado"]:
    st.title("Acceso de Administrador")
    st.markdown(
        "Por favor, identifícate como parte del equipo de desarrollo para acceder al panel."
    )

    # Declaramos un bloque de tipo formulario. Streamlit agrupa todo lo que haya dentro
    with st.form("admin_login_form"):
        admin_user = st.text_input("Usuario")
        admin_pass = st.text_input(
            "Contraseña", type="password"
        )  # type="password" oculta el texto con asteriscos
        submit_admin = st.form_submit_button("Acceder")

        # Este bloque 'if' solo se ejecuta en el instante en que el usuario hace clic en el botón "Acceder"
        if submit_admin:
            # Preparamos un diccionario con los datos, para mandarlo en formato JSON
            payload = {"username": admin_user, "password": admin_pass}
            try:
                # Enviamos una petición de tipo POST a la ruta segura /login de nuestro Backend FastAPI
                response = requests.post("http://localhost:8000/login", json=payload)

                # Si FastAPI nos devuelve un código HTTP 200 (¡Todo correcto!)
                if response.status_code == 200:
                    datos_usuario = (
                        response.json()
                    )  # Traducimos la respuesta a un diccionario Python

                    # Guardamos en la memoria temporal que el login ha sido un éxito
                    st.session_state["admin_autenticado"] = True
                    st.session_state["usuario_info"] = datos_usuario["user"]

                    # IMPORTANTE: Grabamos una Cookie en el navegador del usuario para que la próxima
                    # vez que entre mañana, no tenga que volver a poner la contraseña.
                    cookie_manager.set(
                        "disney_admin_session",
                        val=admin_user,
                        key="login_cookie",
                        path="/",
                    )

                    st.success(f"Bienvenido, {admin_user}.")
                    # Al hacer st.rerun(), obligamos a que el script de python se vuelva a leer desde
                    # la línea 1. Como ahora st.session_state ya es True, entrará directo por el ELSE principal (Línea 74).
                    st.rerun()

                elif response.status_code == 401:
                    st.error("Credenciales de administrador incorrectas.")
                else:
                    st.error(f"Error en el servidor: {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("No se pudo conectar con el servidor backend.")

# --- PANEL DE ADMINISTRADOR (SI ESTÁ AUTENTICADO) ---
# Si llega a este 'else', es porque st.session_state["admin_autenticado"] es igual a True
else:
    st.title("Panel de Control de Administrador")
    st.markdown("Gestion de los datos y visualización de métricas clave.")

    # Botón de cerrar sesión: Borramos todas las pruebas de acceso
    if st.button("Cerrar Sesión de Administrador"):
        # 1. Borrar la cookie permanente del navegador
        cookie_manager.delete("disney_admin_session", key="logout_cookie")
        # 2. Limpiar la memoria temporal session_state
        st.session_state["admin_autenticado"] = False
        # Recargamos la app desde cero. Al llegar al 'if' de arriba, verá False y nos mandará al cajón de login.
        st.rerun()

    # st.sidebar es una barra lateral izquierda pegada al borde. Ponemos botones ahí.
    st.sidebar.header("Acciones de Datos")

    # Si se pulsa este botón le pedimos al Backend que averigue todos los usuarios de la base de datos MySQL
    if st.sidebar.button("Sincronizar con MySQL", use_container_width=True):
        response = requests.get("http://localhost:8000/usuarios")

        if response.status_code == 200:
            # pd.DataFrame transforma la enorme lista JSON en una tabla bidimensional
            df = pd.DataFrame(response.json())
            # Guardamos esa tabla en la memoria temporal para pintarla luego en pantalla.
            st.session_state["datos_usuarios"] = df
            st.sidebar.success("Usuarios cargados desde MySQL")
        else:
            st.sidebar.error("Error al conectar con el Backend")

    # Si pulsa importar, lanzamos el ETL
    # Ojo: Este botón congela la app mientras se procesan miles de filas
    if st.sidebar.button("Importar Datos a BD", use_container_width=True):
        response = requests.post("http://localhost:8000/importar_datos")
        if response.status_code == 200:
            st.sidebar.success("Catálogos importados correctamente!")
        else:
            st.sidebar.error("Error al conectar con el Backend")

    tab1, tab2, tab3 = st.tabs(
        ["Gestión de Usuarios", "Análisis Exploratorio (EDA)", "Rendimiento Modelos IA"]
    )

    with tab1:
        st.subheader("Listado de Usuarios (Sincronizados)")
        # Revisamos si hemos cargado los usuarios dándole al botón Sincronizar de la barra lateral
        if "datos_usuarios" in st.session_state:
            # st.dataframe se encarga de pintar la tabla en pantalla (habilitando filtros, poder hacer clic, etc)
            st.dataframe(st.session_state["datos_usuarios"], use_container_width=True)
        else:
            st.info(
                "Haz clic en 'Sincronizar con MySQL' en la barra lateral para ver los usuarios."
            )

    with tab2:
        st.subheader("Análisis Exploratorio de Datos del Catálogo")

        EDA_DIR = "static/eda"
        img_top10 = os.path.join(EDA_DIR, "top10_peliculas.png")
        img_usuarios = os.path.join(EDA_DIR, "distribucion_valoraciones_usuario.png")
        img_puntuaciones = os.path.join(EDA_DIR, "distribucion_puntuaciones.png")
        json_metricas = os.path.join(EDA_DIR, "metricas.json")

        # Comprobamos si el administrador ya ha generado las imágenes estáticas
        charts_generados = all(
            os.path.exists(p) for p in [img_top10, img_usuarios, img_puntuaciones]
        )

        if not charts_generados:
            st.warning(
                "**Las imágenes del EDA no se han generado aún.**\n\n"
                "Ejecuta el script offline desde la raíz del proyecto con:\n\n"
                "```\npython -m src.scripts.generar_eda_charts\n```\n\n"
                "Esto generará las gráficas en `static/eda/` y la próxima vez cargarán al instante."
            )
        else:
            st.markdown(
                "### Top 10 Películas con Mejores Valoraciones (Mín. 500 votos)"
            )
            st.image(img_top10, use_container_width=True)

            st.divider()

            st.markdown("### Distribución de Valoraciones por Usuario")
            st.image(img_usuarios, use_container_width=True)

            # Métricas desde JSON (calculadas una sola vez offline)
            if os.path.exists(json_metricas):
                with open(json_metricas, "r", encoding="utf-8") as f:
                    m = json.load(f)
                cols_info = st.columns(3)
                cols_info[0].metric(
                    "Media valoraciones / usuario",
                    f"{m.get('media_valoraciones_usuario', '—')}",
                )
                cols_info[1].metric(
                    "Usuarios con < 20 valoraciones",
                    f"{m.get('usuarios_menos_20_valoraciones', '—'):,}",
                )
                cols_info[2].metric(
                    "Total de usuarios", f"{m.get('total_usuarios', '—'):,}"
                )

            st.divider()

            st.markdown("### Distribución General de Puntuaciones (Estrellas)")
            st.image(img_puntuaciones, use_container_width=True)

            # Información sobre cuándo se generaron las imágenes
            mtime = os.path.getmtime(img_top10)
            import datetime

            fecha_gen = datetime.datetime.fromtimestamp(mtime).strftime(
                "%d/%m/%Y %H:%M"
            )
            st.caption(
                f"Charts generados el {fecha_gen} · Para actualizar ejecuta `python -m src.scripts.generar_eda_charts`"
            )

    with tab3:
        st.subheader("Evaluación Comparativa de Modelos de Recomendación")
        metricas_path = "src/utils/metricas_ranking.csv"

        if os.path.exists(metricas_path):
            df_metricas = pd.read_csv(metricas_path)

            # Formatear porcentajes para la tabla
            st.markdown("### Tabla de Resultados (Métricas de Ranking Offline)")
            st.dataframe(df_metricas, use_container_width=True)

            st.divider()

            # Extraer K dinámicamente de los nombres de columna (ej. NDCG_10)
            ndcg_col = [c for c in df_metricas.columns if "NDCG" in c][0]
            prec_col = [c for c in df_metricas.columns if "Precision" in c][0]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Comparativa de {ndcg_col}** (Calidad de ordenación)")
                fig_ndcg, ax_ndcg = plt.subplots(figsize=(8, 5))
                sns.barplot(
                    x="Modelo",
                    y=ndcg_col,
                    data=df_metricas,
                    palette="Blues_d",
                    ax=ax_ndcg,
                )
                st.pyplot(fig_ndcg)

            with col2:
                st.markdown(f"**Comparativa de {prec_col}** (Acierto en Top-K)")
                fig_prec, ax_prec = plt.subplots(figsize=(8, 5))
                sns.barplot(
                    x="Modelo",
                    y=prec_col,
                    data=df_metricas,
                    palette="Greens_d",
                    ax=ax_prec,
                )
                st.pyplot(fig_prec)

            st.info(
                "**NDCG** mide la calidad matemática del orden exacto (Si tu favorita sale #1 suma más que si sale #10)"
            )
            st.info(
                "**Precision** mide el porcentaje bruto de películas relevantes sugeridas."
            )
        else:
            st.info(
                "No hay datos de evaluación disponibles. El administrador de la IA debe ejecutar el script `evaluacion_ranking.py` primero."
            )
