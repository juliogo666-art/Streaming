######################################################################################

# Proyecto 4 — Plataforma de Streaming con IA

#######################################################################################

Un magnate de Dubai quiere hacer su propia plataforma de películas y series, quiere todo lo que hay en el mundo quitando lo inmoral.

## Arquitectura del Proyecto

```
Proyecto 4 - Streaming
├── main.py                           # Lanzador unificado (Backend + Frontend)
├── pyproject.toml                    # Dependencias del proyecto (uv / pip)
├── .env / .env.sample                # Variables de entorno (credenciales BD, APIs)
│
├── src/
│   ├── api/                          # ── BACKEND (FastAPI) ──
│   │   ├── main_api.py               #   API REST: 7 endpoints de recomendación IA
│   │   ├── database.py               #   Pool de conexiones MySQL
│   │   └── etl.py                    #   Pipeline ETL: CSV → MySQL
│   │
│   ├── frontend/                     # ── FRONTEND (Streamlit) ──
│   │   ├── app_ui.py                 #   Landing page / Bienvenida
│   │   └── pages/
│   │       ├── 1_Administrador.py    #   Panel Admin: ETL + EDA + Métricas IA
│   │       └── 2_Usuario.py          #   Login/Registro + Catálogo + Recomendaciones
│   │
│   ├── models/                       # ── MODELOS DE IA ──
│   │   └── jj/
│   │       ├── modelo_1_SVD.py       #   Filtrado Colaborativo (Surprise SVD)
│   │       ├── modelo_2_knn+cs.py    #   KNN + Similitud del Coseno
│   │       ├── modelo_3_wide&deep.py #   Wide & Deep Neural Network (PyTorch→ONNX)
│   │       ├── modelo_4_bcs_tf-idf.py#   Content-Based TF-IDF (Cold Start)
│   │       ├── modelo_5_implicit.py  #   Implicit BPR (Ranking puro)
│   │       └── modelo_6_ncf.py       #   Neural Collaborative Filtering (GMF+MLP)
│   │
│   ├── networks/dl/                  # ── ARQUITECTURAS DE REDES NEURONALES ──
│   │   └── rn_mlp.py                #   Wide & Deep Model (nn.Module)
│   │
│   ├── schemas/                      # ── ESQUEMAS PYDANTIC ──
│   │   └── schemas.py                #   LoginRequest, RegisterRequest, etc.
│   │
│   ├── config/                       # ── CONFIGURACIÓN ──
│   │   └── rules_cleaning.yaml       #   Reglas ETL de limpieza de datos
│   │
│   ├── utils/                        # ── UTILIDADES ──
│   │   ├── evaluacion_ranking.py     #   Evaluador comparativo (NDCG, Precision, etc.)
│   │   ├── exportar_onnx.py          #   Conversor PKL→Joblib + PTH→ONNX
│   │   └── registrar_metricas.py     #   Registro histórico CSV de métricas por modelo
│   │
│   ├── data/                         # ── DATOS ──
│   │   ├── raw/                      #   Datos brutos de APIs (TMDB, Trakt)
│   │   ├── clean/                    #   Datos post-limpieza
│   │   ├── ready/                    #   Datos finales para modelos IA
│   │   └── scripts_sql/             #   Scripts DDL de MySQL
│   │
│   ├── scripts/                      # ── SCRIPTS DE DATOS ──
│   │   ├── data_cleaning.py          #   Limpieza de CSVs
│   │   ├── data_unification.py       #   Unificación de datasets
│   │   ├── tmdb_api_down.py          #   Descarga de datos TMDB
│   │   └── ...
│   │
│   └── image/                        # Recursos gráficos
│
├── logs/                             # Logs del servidor
└── info/                             # Documentación y PDFs del enunciado
```


## Que tenemos

1. Datos disponibles
   1. Datos de películas (~55.000 películas con sinopsis, géneros, posters)
   2. Datos de series (~28.000 series)
   3. ~25.7M valoraciones de usuarios (MovieLens)

2. Interfaz de usuario
   1. Catálogo predeterminado (Top puntuadas, Más vistas)
   2. Buscador por título
   3. Selector de motor de IA (6 modelos)
   4. Sistema de login/registro con selección de gustos

3. Panel de administrador
   1. Sincronización con MySQL
   2. Importación ETL de datos
   3. EDA interactivo (gráficos de distribución)
   4. Comparativa de métricas de los modelos de IA

## Cuál es el flujo

1. **INPUTS**: El usuario accede con su ID → Se consulta su historial
2. **BASE DE DATOS**: Consulta catálogo de películas/series desde MySQL
3. **MODELO IA**: El motor seleccionado predice ratings o scores de ranking
4. **RESULTADOS**: Se muestran las Top-N recomendaciones con poster y sinopsis

## Preparación del Entorno

### 1. Base de Datos MySQL

Instalar MySQL Server y Workbench:
https://dev.mysql.com/downloads/installer/

configurar mysql server
ejemplo:
usuario: txema
contraseña: root
contraseña_root: root

añadir mysql al path:
- editar variables de entorno
- editar path
- añadir al final:
    C:\Program Files\MySQL\MySQL Server 8.0\bin

desde consola ejecutar y escribir contraseña de root definida en la instalación:
mysql -u root -p

creamos la base de datos:
CREATE DATABASE streaming_db;
```

Ejecutar el script de creación de tablas:
```
src\data\scripts_sql\create_database.sql

CONECTAR A BBDD LOCAL

- instalar conector:
    pip install mysql-connector-python


### 2. Backend (FastAPI)
```bash
pip install fastapi uvicorn requests

Usamos fastapi para montar un backend
    pip install fastapi uvicorn requests
```

### 3. Frontend (Streamlit)
```bash
pip install streamlit
pip install streamlit extra-streamlit-components
```

### 4. Configurar Variables de Entorno
Copiar `.env.sample` a `.env` y rellenar las credenciales:
```bash
cp .env.sample .env
```

### 5. Ejecutar el proyecto
```bash
python main.py
```

## Tecnologías Utilizadas

| Componente | Tecnología |
|---|---|
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Base de Datos | MySQL 8.0 |
| Modelos IA | Surprise, PyTorch, Implicit, scikit-learn |
| Inferencia | ONNX Runtime |
| Datos | TMDB API, Trakt API, MovieLens 25M |
