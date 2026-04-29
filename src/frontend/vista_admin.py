"""
Vista de Administrador — Módulo importable para la interfaz de Streamlit.
Panel completo de control con 4 apartados (tabs):
Gestión de Usuarios, Exploración Cuantitativa (EDA), Rendimiento de Modelos de IA, y Sandbox de Recomendaciones.
"""

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json


def render():
    """
    Punto de entrada principal. El script de enrutamiento frontend llama a esta función
    tras verificar que el usuario actual posee credenciales (rol) de 'admin'.
    """
    usuario = st.session_state["usuario_actual"]
    # Intentamos obtener el username, en su defecto mostramos 'Admin #ID'
    nombre_mostrar = usuario.get("username", f"Admin #{usuario.get('id_usuario')}")
    id_usuario = usuario.get("id_usuario", "Desconocido")

    # Barra superior visual:
    # Título a la izquierda, etiqueta de usuario al centro y iconografía a la derecha
    col_titulo, col_info, col_logout = st.columns([6, 3, 1])
    with col_titulo:
        st.title("Panel de Control de Administrador")

    with col_info:
        # Etiqueta HTML estizada con los datos de ingreso (decoración minimalista)
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
        st.write(
            ""
        )  # Truco simple para bajar verticalmente el botón y alinear con los otros elementos
        # Botón para purgar el contexto de sesión y forzar recarga frontal.
        if st.button("Cerrar Sesión", key="logout_admin"):
            st.session_state["autenticado"] = False
            st.session_state["usuario_actual"] = None
            st.session_state["role"] = None
            st.rerun()  # Dispara una reactualización forzosa de la pantalla hacia la página de login

    st.markdown(
        "Panel técnico de administración: Gestión de bases de datos relacionales subyacentes, "
        "visualización de telemetría clave e inicializador artificial de recomendación."
    )

    # ── CREACIÓN DE LAS 4 PESTAÑAS (TABS) PRINCIPALES DE LA PANTALLA ──
    tab_gestion, tab_eda, tab_metricas, tab_ia = st.tabs(
        [
            "Gestión de Usuarios",
            "Análisis Exploratorio (EDA)",
            "Rendimiento Modelos IA",
            "Recomendaciones IA",
        ]
    )

    # Bloque 1
    with tab_gestion:
        _render_tab_gestion()
    # Bloque 2
    with tab_eda:
        _render_tab_eda()
    # Bloque 3
    with tab_metricas:
        _render_tab_rendimiento()
    # Bloque 4
    with tab_ia:
        _render_tab_recomendaciones()


# ##############################################################################
#  Tab 1: Listado y Administración de Usuarios MySQL/Backend
# ##############################################################################


def _render_tab_gestion():
    st.subheader("Listado y Obtención de la Carga de Usuarios")

    # Columnas para organizar los disparadores o botones visuales
    col_sincronizacion, col_importar, col_espacio_blanco = st.columns([2, 2, 6])

    with col_sincronizacion:
        # Petición asincrónica a FastAPI para capturar los usuarios y popular un dataframe
        if st.button("Sincronizar con MySQL", use_container_width=True, key="btn_sync"):
            try:
                # Utilizamos verbo GET al tratarse de recolección inofensiva de información
                respuesta_api = requests.get("http://localhost:8000/usuarios")
                if respuesta_api.status_code == 200:
                    # Mapear el array JSON retransmitido a una estructura tabular 2D manipulable de Pandas
                    dataframe_usuarios_mysql = pd.DataFrame(respuesta_api.json())
                    # Estacionamos este objeto en Memoria persistente para evitar perder la vista si ocurre refresh
                    st.session_state["datos_usuarios_cargados"] = (
                        dataframe_usuarios_mysql
                    )
                    st.success("Toda la matriz de usuarios fue cargada desde MySQL!")
                else:
                    st.error(
                        f"Se denegó la Petición. Estatus HTTP Recibido: {respuesta_api.status_code}"
                    )
            except requests.exceptions.ConnectionError:
                st.error(
                    "Error Grave: No se detecta ruteo con Uvicorn (Asegúrese que la API este corriendo)"
                )

    with col_importar:
        # Disparo intensivo al backend para importar CSV crudos hacia los esquemas de MySQL
        if st.button("Importar Datos a BD", use_container_width=True, key="btn_import"):
            try:
                # Empleo de POST debido que modificamos o ingresamos fuertemente volumen al entorno destino (database)
                respuesta_importacion = requests.post(
                    "http://localhost:8000/importar_datos"
                )
                if respuesta_importacion.status_code == 200:
                    st.success(
                        "Se activó el puente exitosamente: Catálogos insertados a la DB relacional."
                    )
                else:
                    st.error(
                        f"API notificó un fallo de inserción de tablas. (HTTP {respuesta_importacion.status_code})"
                    )
            except requests.exceptions.ConnectionError:
                st.error(
                    "Error Operacional: El Endpoint Backend figura como desconectado y/o puerto 8000 en inactividad."
                )

    # Control de vista condicional: Extraemos la tabla si poseemos el caché disponible desde backend
    if "datos_usuarios_cargados" in st.session_state:
        st.dataframe(
            st.session_state["datos_usuarios_cargados"], use_container_width=True
        )
    else:
        st.info("Haz clic en 'Sincronizar con MySQL' para ver los usuarios.")


# ##############################################################################
#  Tab 2: Renderización Grafica y Análisis Exploratorio de Datos (EDA)
# ##############################################################################


def _render_tab_eda():
    st.subheader("Análisis Exploratorio y Estadístico del Catálogo de Audiencias")

    # Carpeta base del compilado analítico general
    CARPETA_SOPORTE_EDA = "static/eda"

    # Agrupamos las rutas de gráficos de soporte generados por scripts background localmente.
    archivo_grafico_top10 = os.path.join(CARPETA_SOPORTE_EDA, "top10_peliculas.png")
    archivo_grafico_usuarios = os.path.join(
        CARPETA_SOPORTE_EDA, "distribucion_valoraciones_usuario.png"
    )
    archivo_grafico_distribucion_stars = os.path.join(
        CARPETA_SOPORTE_EDA, "distribucion_puntuaciones.png"
    )

    # Este manifiesto adjunta contadores importantes exportados por Pandas al generar las gráficas
    archivo_json_metricas_estaticas = os.path.join(CARPETA_SOPORTE_EDA, "metricas.json")

    # Iterador 'all' confirma validez de existencia en disco de la totalidad de archivos referenciados
    graficos_pre_renderizados = all(
        os.path.exists(ruta_actual)
        for ruta_actual in [
            archivo_grafico_top10,
            archivo_grafico_usuarios,
            archivo_grafico_distribucion_stars,
        ]
    )

    # Salvaguarda: Evita errores de imagen "rota" bloqueando si no existe el pack computado
    if not graficos_pre_renderizados:
        st.warning(
            "**Los reportes cartográficos del Set de Análisis no han sido localizados en el disco.**\n\n"
            "Dado sus consumos pesados, estos deben ser generados previamente escribiendo en su consola:\n\n"
            "```\npython -m src.scripts.generar_eda_charts\n```\n\n"
            "Esto efectuará escrituras de gráficas exportadas sobre `static/eda/` rellenando automáticamente aquí."
        )
    else:
        # Seccion 1 - Ranking General Historico
        st.markdown("### Top 10 Producciones Mejor Listadas (Umbral >= 500 reviews)")
        st.image(archivo_grafico_top10, use_container_width=True)

        st.divider()

        # Seccion 2 - Histogramas de actividad de los espectadores (Interacción social)
        st.markdown("### Histograma: Densidad de Valoraciones por Usuario Único")
        st.image(archivo_grafico_usuarios, use_container_width=True)

        # Si encontramos métricas textuales anexadas:
        if os.path.exists(archivo_json_metricas_estaticas):
            with open(
                archivo_json_metricas_estaticas, "r", encoding="utf-8"
            ) as archivo_kpi_abierto:
                diccionario_metricas = json.load(archivo_kpi_abierto)

            columnas_indicadores_ui = st.columns(3)
            # Poblado dinámico usando la función .get() de los diccionarios por resiliencia y prolijo.
            columnas_indicadores_ui[0].metric(
                "Promedio histórico valoraciones/usuario",
                f"{diccionario_metricas.get('media_valoraciones_usuario', 'No Datos')}",
            )
            columnas_indicadores_ui[1].metric(
                "Clúster Usuarios Fantasmas (<20 Calificaciones)",
                f"{diccionario_metricas.get('usuarios_menos_20_valoraciones', 'No Datos'):,}",
            )
            columnas_indicadores_ui[2].metric(
                "Base total de Población Muestral",
                f"{diccionario_metricas.get('total_usuarios', 'No Datos'):,}",
            )

        st.divider()

        # Seccion 3 - Campana de Gauss / Análisis del Sentimiento de Estrellas Calificatorias en masa global
        st.markdown(
            "### Comportamiento Subconsciente en Uso General de Estrellas (1.0 vs 5.0)"
        )
        st.image(archivo_grafico_distribucion_stars, use_container_width=True)

        # Para asegurar transparencia se indica de cuando data este Snapshot de Big Data
        import datetime

        fecha_archivo_mtime = os.path.getmtime(archivo_grafico_top10)
        formato_fecha_legible = datetime.datetime.fromtimestamp(
            fecha_archivo_mtime
        ).strftime("%d/%m/%Y a las %H:%M:%S")

        st.caption(
            f"Conjunto algoritmico congelado el: {formato_fecha_legible} · Para generar modelo actualizado ejecute `generar_eda_charts`."
        )


# ##############################################################################
#  Tab 3: Metrías Offline Algorítmicas — Rendimiento de Modelos IA
# ##############################################################################


def _render_tab_rendimiento():
    st.subheader("Auditoría Científica: Tabla Combinada de Rendimientos IA (Benchmark)")

    # Destino en donde el sistema exporta las métricas de Test Offline al entrenar AI Models.
    ruta_dataset_metricas_csv = "src/utils/metricas_ranking.csv"

    if os.path.exists(ruta_dataset_metricas_csv):
        # Levantamiento e instanciación a tabla virtual Pandas Dataframe
        dataframe_rankings_ai = pd.read_csv(ruta_dataset_metricas_csv)

        st.markdown("### Matriz de Precisión y Evaluación Categórica (Desempeño)")
        st.dataframe(dataframe_rankings_ai, use_container_width=True)

        st.divider()

        # [RECURSO TÉCNICO]
        # Buscamos en las llaves (Header de columna) que incluyan la keyword explícita "NDCG" , "Precision ...."
        # Es ideal extraer a listas porque es probable que existan parámetros dinámicos ej. `@10` ó `@50` o similares en cabecera

        encabezado_métrica_precision = [
            indice_str
            for indice_str in dataframe_rankings_ai.columns
            if "Precision" in indice_str
        ][0]

        encabezado_métrica_recall = [
            indice_str
            for indice_str in dataframe_rankings_ai.columns
            if "Recall" in indice_str
        ][0]

        encabezado_metrica_hitrate = [
            indice_str
            for indice_str in dataframe_rankings_ai.columns
            if "HitRate" in indice_str
        ][0]

        encabezado_métrica_ndcg = [
            indice_str
            for indice_str in dataframe_rankings_ai.columns
            if "NDCG" in indice_str
        ][0]

        encabezado_metrica_coverage = [
            indice_str
            for indice_str in dataframe_rankings_ai.columns
            if "Coverage" in indice_str
        ][0]

        encabezado_métrica_mrr = [
            indice_str
            for indice_str in dataframe_rankings_ai.columns
            if "MRR" in indice_str
        ][0]

        st.markdown("### Graficas comparativas de los modelos")
        # Presentación dual gráfica: Dos gráficos simultáneos distribuidos en espacio parejo (2 columns)

        contenedor_grafico_precision, contenedor_grafico_recall = st.columns(2)
        contenedor_grafico_hitrate, contenedor_grafico_ndcg = st.columns(2)
        contenedor_grafico_coverage, contenedor_grafico_mrr = st.columns(2)

        with contenedor_grafico_precision:
            st.markdown(f"**Competitividad en {encabezado_métrica_precision}**")
            figura_seaborn_2, plano_ejes_2 = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=encabezado_métrica_precision,
                y="Modelo",
                data=dataframe_rankings_ai,
                palette="Greens_d",
                ax=plano_ejes_2,
            )

            st.pyplot(figura_seaborn_2)

        with contenedor_grafico_recall:
            st.markdown(f"**Competitividad en {encabezado_métrica_recall}**")
            figura_seaborn_3, plano_ejes_3 = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=encabezado_métrica_recall,
                y="Modelo",
                data=dataframe_rankings_ai,
                palette="Reds_d",
                ax=plano_ejes_3,
            )
            st.pyplot(figura_seaborn_3)

        with contenedor_grafico_hitrate:
            st.markdown(f"**Competitividad en {encabezado_metrica_hitrate}**")
            figura_seaborn_4, plano_ejes_4 = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=encabezado_metrica_hitrate,
                y="Modelo",
                data=dataframe_rankings_ai,
                palette="Greens_d",
                ax=plano_ejes_4,
            )
            st.pyplot(figura_seaborn_4)

        with contenedor_grafico_ndcg:
            st.markdown(f"**Competitividad en {encabezado_métrica_ndcg}**")
            figura_seaborn_1, plano_ejes_1 = plt.subplots(figsize=(10, 6))
            # Despliegue Seaborn barplot (Barcharts)
            sns.barplot(
                x=encabezado_métrica_ndcg,
                y="Modelo",
                data=dataframe_rankings_ai,
                palette="Blues_d",
                ax=plano_ejes_1,
            )
            st.pyplot(figura_seaborn_1)

        with contenedor_grafico_coverage:
            st.markdown(f"**Competitividad en {encabezado_metrica_coverage}**")
            figura_seaborn_5, plano_ejes_5 = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=encabezado_metrica_coverage,
                y="Modelo",
                data=dataframe_rankings_ai,
                palette="Greens_d",
                ax=plano_ejes_5,
            )
            st.pyplot(figura_seaborn_5)

        with contenedor_grafico_mrr:
            st.markdown(f"**Competitividad en {encabezado_métrica_mrr}**")
            figura_seaborn_6, plano_ejes_6 = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=encabezado_métrica_mrr,
                y="Modelo",
                data=dataframe_rankings_ai,
                palette="Reds_d",
                ax=plano_ejes_6,
            )
            st.pyplot(figura_seaborn_6)

        st.markdown("### Glosario de términos")
        # Diccionario de terminología para fácil repaso (Glossary)

        with st.expander("Términos"):
            st.info(
                "**Precision Top-K**: Simplemente indica sobre el volúmen propuesto K qué porcentaje son relevantes, independientemente del ordenamiento riguroso."
            )
            st.info(
                "**Recall Top-K**: Mide qué porcentaje de los elementos relevantes fueron realmente sugeridos (Mide cuántas de las pelis que le gustaron fuimos capaces de encontrar."
            )
            st.info(
                "**Hit Rate Top-K**: Mide qué porcentaje de los usuarios tuvieron al menos una recomendación relevante.(Hubo al menos 1 acierto en el Top K)"
            )
            st.info(
                "**NDCG (Normalized Discounted Cumulative Gain)**: Observa si el Motor sabe ubicar "
                "los elementos de más encaje natural en primeros rangos. (Premia y bonifica estar en topes #1, #2 y penaliza retrasos)."
            )
            st.info(
                "**Coverage (Cobertura)**: Indica qué porcentaje de la base de ítems fue utilizada por el sistema para generar recomendaciones.(Mide qué porcentaje del catálogo total es capaz de recomendar el modelo (Diversidad))."
            )
            st.info(
                "**MRR (Mean Reciprocal Rank)**: Mide la posición promedio del primer acierto.(Si el primer acierto está en la posición 1, el valor es 1. Si está en la posición 2, el valor es 0.5, etc.)"
            )

    else:
        st.info(
            "No existen trazos operacionales logeados de evaluación. "
            "Es necesario que se ejecuten y computen los ciclos del script de Benchmark: `evaluacion_ranking.py`."
        )


# ##############################################################################
#  Tab 4: Laboratorio de Diagnóstico: Receptor de Recomendaciones IA
# ##############################################################################


def _render_tab_recomendaciones():
    st.subheader(
        "Entorno de Sandbox: Test y Depuración Analítica sobre Modelos en Actividad"
    )
    st.markdown(
        "Utilidad interna de evaluación de redes. Permuta de modelos instantáneamente y envía tokens/ids numéricos a "
        "comprobar los resultados dictaminados por la IA en real-time."
    )

    # --- CONTROLES Y FILTROS INTERACTIVOS ---
    # Organizador posicional entre selector modelo (giga) y barra de identificación
    columna_selector_modelo_ia, columna_caja_id = st.columns([3, 2])

    with columna_selector_modelo_ia:
        modelo_inteligencia_actual = st.selectbox(
            "Modelos de Recomendación - Target Endpoint a Enlazar::",
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

    with columna_caja_id:
        id_falso_pruebas = st.number_input(
            "ID Numérico de usuario (Debe existir histórico en data)",
            value=st.session_state["usuario_actual"].get("id_usuario", 1),
            step=1,
            key="admin_id_simulado",
            help="Sobreescribe e inyecta un ID manual hacia la red neuronal (Usalo para auditar predicciones variopintas).",
        )

    # Interconector lógico con Router FastAPI para direccionar la request (Endpoint Map)
    mapa_direccion_endpoints = {
        "Smart (Auto)": "recomendar/smart",
        "SVD (Rápido)": "recomendar",
        "KNN + Cosine (Explicable)": "recomendar/knn",
        "Wide & Deep (Profundo)": "recomendar/wnd",
        "Content-Based (Cold-Start)": "recomendar/content",
        "Implicit BPR (Ranking Top)": "recomendar/implicit",
        "NCF (Deep Learning)": "recomendar/ncf",
        "Two Towers (Retrieval)": "recomendar/twotowers",
    }

    # Derivamos dinámicamente nuestra selección al respectivo target URL endpoint.
    endpoint_backend_destinado = mapa_direccion_endpoints[modelo_inteligencia_actual]

    st.divider()

    # --- COMPILADOR Y VISTA PREVIA RESULTANTE ----
    _render_recomendaciones_ia(
        id_perfil_usuario=id_falso_pruebas,
        endpoint_elegido=endpoint_backend_destinado,
        identificador_keys_botones="admin_rec",
    )


def _render_recomendaciones_ia(
    id_perfil_usuario,
    endpoint_elegido="recomendar",
    identificador_keys_botones="admin_ia",
):
    """
    Función helper: Ejecuta la petición REST y se encarga integramente del layout en columnas iteradas, asignando
    rutas de imagen TMDB API o fallbacks visuales si el recurso no es servido.
    """
    if not id_perfil_usuario:
        st.info(
            "Formulario Abortado: Proporciona un ID identificativo para inicializar predicciones."
        )
        return

    try:
        # Cadena de compilación URL completa adjuntando params URL-Query. Solicitaremos únicamente 8 propuestas.
        url_servidor_compilado = (
            f"http://127.0.0.1:8000/{endpoint_elegido}/{id_perfil_usuario}"
        )
        respuesta_red_api = requests.get(url_servidor_compilado, params={"n": 10})

        # Validamos código HTTP exitoso (Todo Correcto = 200)
        if respuesta_red_api.status_code == 200:
            paquete_datos_respuesta = respuesta_red_api.json()
            catalogo_recomendado = paquete_datos_respuesta.get("recomendaciones", [])
            mensaje_backend_notificaciones = paquete_datos_respuesta.get("mensaje", "")

            if catalogo_recomendado:
                # Estructuramos bloque cartesiano de 5 espacios (UI Rejilla Grid)
                bloques_columnar_ui = st.columns(5)

                # Descomprimimos individualmente toda iteración sobre recomendación dictada guardando índices
                for posicion_indice, pelicula_sugerida in enumerate(
                    catalogo_recomendado
                ):
                    # Logica del modulo de 5, garantiza iterar cíclicamente por las columnas creadas previamente (1, 2, 3, 4 y se retorna)
                    with bloques_columnar_ui[posicion_indice % 5]:
                        ruta_directorio_imagen = pelicula_sugerida.get(
                            "poster_path", ""
                        )

                        # Manejo asertivo de la falta de Caratular de la Database (Impedir fallos y bugs visuales)
                        if (
                            ruta_directorio_imagen
                            and ruta_directorio_imagen != ""
                            and ruta_directorio_imagen != "nan"
                        ):
                            st.image(
                                f"https://image.tmdb.org/t/p/w500{ruta_directorio_imagen}",
                                use_container_width=True,
                            )
                        else:
                            # Imágen Place-holder en ausencia de imagen por defecto.
                            st.image(
                                "https://via.placeholder.com/300x450.png?text=Ausencia+de+Imagen",
                                use_container_width=True,
                            )

                        # Resumir el titular para que no deteriore responsividad HTML (Desbordes)
                        titulo_obra = pelicula_sugerida.get(
                            "titulo", "Nombres Inexistentes"
                        )
                        if len(titulo_obra) > 30:
                            titulo_obra = titulo_obra[:27] + "..."

                        # Tipografía general final de la Tarjeta Rejilla
                        st.markdown(f"**{titulo_obra}**")
                        st.caption(
                            f"Predicción / Afinamiento IA: {pelicula_sugerida['predicted_rating']} de 5.0"
                        )

                        # Botón que invoca resumen emergente mediante sistema interno nativo Toast Message.
                        if st.button(
                            "Ver sinopsis",
                            key=f"{identificador_keys_botones}_{posicion_indice}",
                        ):
                            texto_sinopsis = pelicula_sugerida.get(
                                "overview",
                                "El objeto listado carece de descripción/sinopsis en nuestros metadatos.",
                            )
                            st.toast(texto_sinopsis)

                # Detección Condicional: Captar y notificar información de enrutamiento especial
                # si el administrador eligió Smart (Ej: Te derive implicitamente a 'SVD' o 'ColdStart' según reglas)
                etiqueta_selector = paquete_datos_respuesta.get("selector", "")
                if etiqueta_selector:
                    st.caption(
                        f"**Módulo Decisor Transitorio / Algorítmo Evaluador:** {etiqueta_selector}"
                    )

            # Si el array recomendado llegó en limpio por fallas algorítmicas o filtros duros del Backend (Cold-Start Limit)
            elif (
                "insufficient_data" in paquete_datos_respuesta
                or "No alcanzas" in mensaje_backend_notificaciones
            ):
                st.info(
                    f"**Incompatibilidad Operativa Interrumpida**: {mensaje_backend_notificaciones}"
                )
                st.caption(
                    "Esta red neuronal requiere que el usuario tenga un historial denso de valoraciones."
                )
            else:
                st.info("No se encontraron recomendaciones para este usuario.")

        # API Exige 503 HTTP Retorno -> Sucede típicamente al arrancar recintemente uvicorn (Cold booting pesado de Tensores o PKLs)
        elif respuesta_red_api.status_code == 503:
            explicativo_error = respuesta_red_api.json().get(
                "detail", "Error Irresoluto a priori."
            )
            st.warning(
                f"Gateway Backend bloqueado / indisponible. Motivo explícito de Uvicorn: {explicativo_error}"
            )
        else:
            # Mostrar el código real y el cuerpo de respuesta para diagnosticar
            try:
                detalle_error = respuesta_red_api.json().get("detail", respuesta_red_api.text[:200])
            except Exception:
                detalle_error = respuesta_red_api.text[:200]
            st.warning(
                f"Error HTTP {respuesta_red_api.status_code}: {detalle_error}"
            )

    # Filtro contra apagon repentino de server backend FastAPI (Imposibilidad de Networking hacia localhost loopback)
    except requests.exceptions.ConnectionError:
        st.warning(
            "Enlace Roto con Motor API: Chequea Uvicorn o su Terminal para verificar que está operando y libre."
        )
