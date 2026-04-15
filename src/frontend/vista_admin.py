"""
Vista de Administrador — Módulo importable.
Panel completo con 4 tabs: Gestión de Usuarios, EDA, Rendimiento Modelos, Recomendaciones IA.
Incluye selector de modelo IA y selector de usuario por ID en el tab de Recomendaciones.
"""

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json


def render():
    """Punto de entrada que app_ui.py llama tras verificar rol == 'admin'."""
    usuario = st.session_state["usuario_actual"]
    nombre_mostrar = usuario.get("username", f"Admin #{usuario.get('id_usuario')}")
    id_usuario = usuario.get("id_usuario", "?")

    # ── Barra superior: título + etiqueta identidad + botón logout ──
    col_titulo, col_info, col_logout = st.columns([6, 3, 1])
    with col_titulo:
        st.title("Panel de Control de Administrador")
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
                <span style="color:#B8860B; font-weight:600;">{nombre_mostrar}</span>
                <span style="color:#aaa; font-size:1.1rem;"> · ID: {id_usuario}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_logout:
        st.write("")  # Espaciado
        if st.button("Cerrar Sesión", key="logout_admin"):
            st.session_state["autenticado"] = False
            st.session_state["usuario_actual"] = None
            st.session_state["role"] = None
            st.rerun()

    st.markdown(
        "Gestión de los datos, visualización de métricas clave y motor de recomendación."
    )

    # ── TABS PRINCIPALES ──
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Gestión de Usuarios",
            "Análisis Exploratorio (EDA)",
            "Rendimiento Modelos IA",
            "Recomendaciones IA",
        ]
    )

    with tab1:
        _render_tab_gestion()

    with tab2:
        _render_tab_eda()

    with tab3:
        _render_tab_rendimiento()

    with tab4:
        _render_tab_recomendaciones()


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 1: Gestión de Usuarios
# ══════════════════════════════════════════════════════════════════════════════


def _render_tab_gestion():
    st.subheader("Listado de Usuarios")

    col_sync, col_import, col_spacer = st.columns([2, 2, 6])
    with col_sync:
        if st.button("Sincronizar con MySQL", use_container_width=True, key="btn_sync"):
            try:
                response = requests.get("http://localhost:8000/usuarios")
                if response.status_code == 200:
                    df = pd.DataFrame(response.json())
                    st.session_state["datos_usuarios"] = df
                    st.success("Usuarios cargados desde MySQL")
                else:
                    st.error("Error al conectar con el Backend")
            except requests.exceptions.ConnectionError:
                st.error("No se pudo conectar con el servidor backend.")

    with col_import:
        if st.button("Importar Datos a BD", use_container_width=True, key="btn_import"):
            try:
                response = requests.post("http://localhost:8000/importar_datos")
                if response.status_code == 200:
                    st.success("Catálogos importados correctamente!")
                else:
                    st.error("Error al conectar con el Backend")
            except requests.exceptions.ConnectionError:
                st.error("No se pudo conectar con el servidor backend.")

    if "datos_usuarios" in st.session_state:
        st.dataframe(st.session_state["datos_usuarios"], use_container_width=True)
    else:
        st.info("Haz clic en 'Sincronizar con MySQL' para ver los usuarios.")


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 2: Análisis Exploratorio (EDA)
# ══════════════════════════════════════════════════════════════════════════════


def _render_tab_eda():
    st.subheader("Análisis Exploratorio de Datos del Catálogo")

    EDA_DIR = "static/eda"
    img_top10 = os.path.join(EDA_DIR, "top10_peliculas.png")
    img_usuarios = os.path.join(EDA_DIR, "distribucion_valoraciones_usuario.png")
    img_puntuaciones = os.path.join(EDA_DIR, "distribucion_puntuaciones.png")
    json_metricas = os.path.join(EDA_DIR, "metricas.json")

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
        st.markdown("### Top 10 Películas con Mejores Valoraciones (Mín. 500 votos)")
        st.image(img_top10, use_container_width=True)

        st.divider()

        st.markdown("### Distribución de Valoraciones por Usuario")
        st.image(img_usuarios, use_container_width=True)

        # Métricas desde JSON
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

        # Fecha de generación
        import datetime

        mtime = os.path.getmtime(img_top10)
        fecha_gen = datetime.datetime.fromtimestamp(mtime).strftime("%d/%m/%Y %H:%M")
        st.caption(
            f"Charts generados el {fecha_gen} · Para actualizar ejecuta `python -m src.scripts.generar_eda_charts`"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 3: Rendimiento de Modelos IA
# ══════════════════════════════════════════════════════════════════════════════


def _render_tab_rendimiento():
    st.subheader("Evaluación Comparativa de Modelos de Recomendación")
    metricas_path = "src/utils/metricas_ranking.csv"

    if os.path.exists(metricas_path):
        df_metricas = pd.read_csv(metricas_path)

        st.markdown("### Tabla de Resultados (Métricas de Ranking Offline)")
        st.dataframe(df_metricas, use_container_width=True)

        st.divider()

        # Extraer K dinámicamente de los nombres de columna
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
            "No hay datos de evaluación disponibles. El administrador debe ejecutar el script `evaluacion_ranking.py` primero."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Tab 4: Recomendaciones IA (NUEVO — funcionalidad del usuario, con controles)
# ══════════════════════════════════════════════════════════════════════════════


def _render_tab_recomendaciones():
    st.subheader("Motor de Recomendación — Vista Admin")
    st.markdown(
        "Prueba todos los modelos de IA disponibles con cualquier ID de usuario del dataset."
    )

    # --- CONTROLES ---
    col_modelo, col_id = st.columns([3, 2])

    with col_modelo:
        modelo_ia = st.selectbox(
            "Motor de Recomendación",
            [
                "Smart (Auto)",
                "SVD (Rápido)",
                "KNN + Cosine (Explicable)",
                "Wide & Deep (Profundo)",
                "Content-Based (Cold-Start)",
                "Implicit BPR (Ranking Top)",
                "NCF (Deep Learning)",
                "Two Towers (Retrieval)",
            ],
            index=0,
            key="admin_modelo_ia",
        )

    with col_id:
        id_simulado = st.number_input(
            "ID de Usuario (del dataset de ratings)",
            value=st.session_state["usuario_actual"].get("id_usuario", 1),
            step=1,
            key="admin_id_simulado",
            help="Permite simular predicciones para IDs de súper-usuarios (ej. 9) que existen en el set de datos.",
        )

    mapa_endpoints = {
        "Smart (Auto)": "recomendar/smart",
        "SVD (Rápido)": "recomendar",
        "KNN + Cosine (Explicable)": "recomendar/knn",
        "Wide & Deep (Profundo)": "recomendar/wnd",
        "Content-Based (Cold-Start)": "recomendar/content",
        "Implicit BPR (Ranking Top)": "recomendar/implicit",
        "NCF (Deep Learning)": "recomendar/ncf",
        "Two Towers (Retrieval)": "recomendar/twotowers",
    }
    endpoint_ia = mapa_endpoints[modelo_ia]

    st.divider()

    # --- RECOMENDACIONES ---
    _render_recomendaciones_ia(
        user_id=id_simulado,
        endpoint=endpoint_ia,
        key_prefix="admin_rec",
    )


def _render_recomendaciones_ia(user_id, endpoint="recomendar", key_prefix="admin_ia"):
    """Llama al Backend y pinta las recomendaciones del modelo seleccionado."""
    if not user_id:
        st.info("Introduce un ID de usuario para generar recomendaciones.")
        return

    try:
        resp = requests.get(
            f"http://127.0.0.1:8000/{endpoint}/{user_id}", params={"n": 8}
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
                        st.caption(f"⭐ Predicción IA: {rec['predicted_rating']} / 5.0")
                        if st.button("Ver sinopsis", key=f"{key_prefix}_{idx}"):
                            st.toast(rec.get("overview", "Sin sinopsis disponible."))

                # Info del selector Smart
                selector_info = datos.get("selector", "")
                if selector_info:
                    st.caption(f"{selector_info}")
            elif "insufficient_data" in datos or "No alcanzas" in mensaje:
                st.info(f"**Requisito del Modelo**: {mensaje}")
                st.caption(
                    "Esta red neuronal requiere que el usuario tenga un historial denso de valoraciones."
                )
            else:
                st.info("No se encontraron recomendaciones para este usuario.")
        elif resp.status_code == 503:
            error_det = resp.json().get("detail", "Error desconocido")
            st.warning(f"El Backend reporta: {error_det}")
        else:
            st.warning("No se pudieron obtener recomendaciones.")
    except requests.exceptions.ConnectionError:
        st.warning("No se pudo conectar con el Backend para recomendaciones.")
