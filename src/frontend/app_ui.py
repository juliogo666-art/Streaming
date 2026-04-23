"""
app_ui.py — Punto de entrada del frontend Streamlit.

Gestiona el login centralizado y redirige a la vista correcta según el rol:
  - role == 'user'  → vista_usuario.render()
  - role == 'admin' → vista_admin.render()

Sin sidebar de navegación: toda la lógica de routing es por session_state.
"""

import streamlit as st
import requests
import datetime
import base64
import os
import time


# ── Configuración de página (debe ser la primera llamada de Streamlit) ──
st.set_page_config(
    page_title="SPIRE Streaming",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ── Función para fondo base64 ──
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_bg_hack(main_bg):
    bin_str = get_base64_of_bin_file(main_bg)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    /* Añadir una capa de contraste para que el contenido sea legible */
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 18, 32, 0.7); /* Azul Marino con opacidad */
        z-index: -1;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


# Aplicar fondo si existe
fondo_path = "static/spire/fondo_1.png"
if os.path.exists(fondo_path):
    set_bg_hack(fondo_path)

# ── Estilos premium (Azul Marino + Oro) — compartidos por login, usuario y admin ──
st.markdown(
    """
    <style>
    /* Ocultar sidebar y menú de hamburguesa */
    [data-testid="stSidebar"] { display: none; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }

    /* Fondo y colores principales */
    .stApp {
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

    /* Inputs y widgets estilizados */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stMultiSelect>div>div>div, .stDateInput>div>div>input {
        background-color: #001f3f !important;
        color: white !important;
        border: 1px solid #1a3a5f !important;
        border-radius: 8px !important;
    }

    /* Checkboxes personalizados */
    .stCheckbox > label > div[data-testid="stMarkdownContainer"] > p {
        color: #f0f0f0 !important;
        font-size: 0.9rem;
    }
    span[data-baseweb="checkbox"] > div {
        border-color: #B8860B !important;
    }
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

# ── Inicialización del session_state ──
if "autenticado" not in st.session_state:
    st.session_state["autenticado"] = False
    st.session_state["usuario_actual"] = None
    st.session_state["role"] = None
if "backend_listo" not in st.session_state:
    st.session_state["backend_listo"] = False


# ══════════════════════════════════════════════════════════════════════════════
#  VERIFICACIÓN DEL BACKEND (bloquea hasta que esté listo)
# ══════════════════════════════════════════════════════════════════════════════


def _comprobar_backend():
    """Intenta conectar con el backend. Devuelve True si responde."""
    try:
        r = requests.get("http://127.0.0.1:8000/status", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# Si aún no sabemos si el backend está listo, comprobamos
if not st.session_state["backend_listo"]:
    if _comprobar_backend():
        st.session_state["backend_listo"] = True
    else:
        # ── PANTALLA DE CARGA ──
        st.markdown(
            """
        <style>
        @keyframes pulse {
            0%, 100% { opacity: 0.4; transform: scale(0.95); }
            50% { opacity: 1; transform: scale(1.05); }
        }
        .loading-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 60vh;
            text-align: center;
        }
        .loading-logo {
            animation: pulse 2s ease-in-out infinite;
            max-width: 250px;
            margin-bottom: 30px;
        }
        .loading-text {
            color: #B8860B !important;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 10px;
        }
        .loading-sub {
            color: #8899aa !important;
            font-size: 1rem;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        # st.markdown('<div class="loading-container">', unsafe_allow_html=True)
        col_img, col_text = st.columns([1, 1])
        # Mostrar logo centrado con animación
        with col_img:
            if os.path.exists("static/spire/spire.png"):
                st.image("static/spire/spire.png", use_container_width=True)
        with col_text:
            st.markdown(
                '<h2 class="loading-text">Preparando SPIRE Streaming...</h2>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<h3 class="loading-sub">Cargando modelos de inteligencia artificial y catálogo de películas.</h3>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<h3 class="loading-sub">Esto puede tardar un par de minutos en el primer arranque.</h3>',
                unsafe_allow_html=True,
            )
            with st.spinner("Cargando..."):
                time.sleep(110)
        st.markdown("</div>", unsafe_allow_html=True)

        # Esperar y reintentar automáticamente
        time.sleep(5)
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PANTALLA DE LOGIN (si no está autenticado y backend listo)
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state["autenticado"] and st.session_state["backend_listo"]:
    col_logo, col_main = st.columns([1, 1], gap="large")

    with col_logo:
        st.image("static/spire/spire.png", use_container_width=True)

    with col_main:
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
                            "http://127.0.0.1:8000/login", json=payload, timeout=5
                        )

                        if response.status_code == 200:
                            datos_usuario = response.json()
                            user_info = datos_usuario["user"]

                            st.session_state["autenticado"] = True
                            st.session_state["usuario_actual"] = user_info
                            st.session_state["role"] = user_info.get("role", "user")

                            st.success("¡Login exitoso!")
                            st.rerun()
                        elif response.status_code == 401:
                            st.error(
                                "Credenciales incorrectas. Verifica tu usuario o regístrate en la otra pestaña."
                            )
                        else:
                            st.error(f"Error en el servidor: {response.status_code}")
                    except requests.exceptions.ConnectionError:
                        st.error(
                            "No se pudo conectar con el servidor backend. ¿Está arrancado?"
                        )
                    except Exception as e:
                        st.error(f"Error de conexión: {e}")

        # --- PESTAÑA REGISTRO ---
        with tab_registro:
            st.subheader("Crea tu cuenta")

            # Cargar géneros para el selector de gustos
            opciones_generos = {}
            try:
                resp_gen = requests.get("http://127.0.0.1:8000/genres", timeout=5)
                if resp_gen.status_code == 200:
                    for g in resp_gen.json():
                        opciones_generos[g["name"]] = g["id"]
            except Exception:
                pass  # El backend puede tardar ~2min en arrancar; se reintenta al refrescar

            with st.form("register_form"):
                col1, col2 = st.columns(2)
                with col1:
                    new_username = st.text_input(
                        "Usuario", placeholder="Tu nombre de usuario"
                    )
                    new_email = st.text_input("Email", placeholder="ejemplo@correo.com")
                    new_password = st.text_input("Contraseña", type="password")

                with col2:
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
                st.markdown(
                    "**¿Qué géneros te gusta ver?** "
                    "<span style='color:#B8860B;font-size:0.88rem;'>(elige exactamente 3)</span>",
                    unsafe_allow_html=True,
                )

                if not opciones_generos:
                    st.caption(
                        "El servidor aún está cargando. Refresca la página (F5) en unos segundos."
                    )

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
                            cols[i].checkbox(genre_name, key=f"genre_{genre_name}")

                col_reg, col_omitir = st.columns([3, 2])
                with col_reg:
                    btn_register = st.form_submit_button(
                        "REGISTRARSE AHORA", use_container_width=True
                    )
                with col_omitir:
                    btn_omitir = st.form_submit_button(
                        "Omitir géneros →",
                        use_container_width=True,
                        help="Asigna automáticamente los 3 géneros más populares: Drama, Acción y Comedia.",
                    )

                # IDs de los 3 géneros más populares (Drama=18, Acción=28, Comedia=35)
                _GENEROS_POPULARES_IDS = [18, 28, 35]

                if btn_register or btn_omitir:
                    if btn_omitir:
                        # Usar los 3 géneros populares; si la API devolvió IDs distintos, adaptarse
                        ids_disponibles = set(opciones_generos.values())
                        gustos_actuales = [
                            gid for gid in _GENEROS_POPULARES_IDS if gid in ids_disponibles
                        ]
                        # Fallback: tomar los primeros 3 de la lista si no coinciden
                        if len(gustos_actuales) < 3:
                            gustos_actuales = list(opciones_generos.values())[:3]
                    else:
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
                    elif btn_register and len(gustos_actuales) != 3:
                        st.error(
                            f"Debes seleccionar exactamente **3 géneros** "
                            f"(tienes {len(gustos_actuales)} seleccionados). "
                            "Si no sabes cuáles elegir, usa el botón «Omitir géneros»."
                        )
                    else:
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
                                datos = response.json()
                                user_info = datos.get("user")
                                if user_info:
                                    st.session_state["autenticado"] = True
                                    st.session_state["usuario_actual"] = user_info
                                    st.session_state["role"] = user_info.get(
                                        "role", "user"
                                    )
                                    st.success(
                                        f"¡Bienvenido, {new_username}! Entrando a SPIRE…"
                                    )
                                    st.rerun()
                                else:
                                    st.success(
                                        f"¡Bienvenido, {new_username}! Tu cuenta ha sido creada."
                                    )
                                    st.info(
                                        "Inicia sesión en la pestaña «Iniciar Sesión»."
                                    )
                            else:
                                error_detail = response.json().get(
                                    "detail", "Error desconocido"
                                )
                                st.error(f"Error al registrar: {error_detail}")
                        except Exception as e:
                            st.error(f"No se pudo conectar con el servidor: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTING POR ROL (si está autenticado)
# ══════════════════════════════════════════════════════════════════════════════

elif st.session_state["autenticado"]:
    role = st.session_state.get("role", "user")

    if role == "admin":
        from src.frontend.vista_admin import render as render_admin

        render_admin()
    else:
        from src.frontend.vista_usuario import render as render_usuario

        render_usuario()
