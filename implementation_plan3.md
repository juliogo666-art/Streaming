# Plan de Implementación: Fase 3 (Refactorización, Rendimiento y Front-end)

A continuación se detalla la hoja de ruta establecida para la siguiente fase del proyecto, enfocada en la experiencia de usuario, optimización de tiempos de carga y despliegue robusto de los modelos.

## Cambios Propuestos

### 1. Sistema de Recomendación Dinámico (Clustering de Usuarios)
- **Definición de Segmentos**: Clasificar usuarios según su volumen de valoraciones/visualizaciones (ej. Nuevos, Ocasionales, Expertos).
- **Asignación de Modelos**: 
  - *Cold-Start (Nuevos)*: TF-IDF (Contenido) o Popularidad.
  - *Medios/Expertos*: Redirigir hacia modelos colaborativos como SVD, NCF o Wide&Deep dependiendo de los umbrales de seguridad de datos.

### 2. Límite y Visualización de Recomendaciones
- Ajustar y estandarizar el parámetro `K` (límite superior) mostrado en la interfaz.
- Garantizar que las recomendaciones se adapten visualmente sin romper el grid de Streamlit/FastAPI.

### 3. Optimización del EDA (Generación de Imágenes Estáticas)
- **Problema actual**: El frontend renderiza las gráficas en vivo, lo cual toma demasiado tiempo y atasca el hilo de ejecución principal.
- **Solución**: Refactorizar la sección EDA para que consuma imágenes (PNG/WebP) pre-renderizadas generadas en un paso offline o asíncrono, acelerando la navegación masivamente.

### 4. Unificación del Frontend y Sistema de Accesos
- Fusión de páginas para evitar navegar desde un "Home" redundante.
- Implementación de un portal de acceso (Login) común:
  - **Usuario Estándar**: Acceso directo al recomendador.
  - **Administrador**: Panel de visualización con acceso a EDA, métricas y reentrenamiento.

### 5. Repositorio Remoto de Modelos (Kaggle/HuggingFace)
- Diseñar una lógica de fallback (respaldo) durante el arranque de la API en `main.py`.
- Si los archivos `.pkl` o `.onnx` no existen localmente, el sistema los descargará automáticamente y de manera silenciosa desde un bucket o repositorio de HuggingFace.
- Esto aligerará el repositorio Git original, evitando commits pesados de modelos binarios.

### 6. Benchmark de Arranque (`main.py`)
- Añadir trazas de tiempo (`time.time()`) detalladas durante la fase de carga de modelos.
- Analizar qué formato (`.pkl`, `.onnx`, o `.joblib`) bloquea la CPU en el inicio del servidor, sustituyendo las cargas ineficientes si se detectan cuellos de botella severos.

## Preguntas Abiertas

> [!IMPORTANT]
> **Decisiones de Diseño a confirmar mañana:**
> 1. ¿Usaremos una base de datos pequeña (SQLite/JSON) para la tabla del Login o quieres usuarios/contraseñas fijos quemados en un `.env`?
> 2. ¿Para alojar los modelos, prefieres crear una cuenta dedicada en 🤗 HuggingFace o prefieres un Release de GitHub / Google Drive por facilidad?

## Plan de Verificación

### Pruebas Manuales
- **Login**: Iniciar sesión consecutivamente como Admin y Usuario comprobando las restricciones de interfaz (ej. que el usuario normal no vea el EDA).
- **Arranque en Frío**: Borrar deliberadamente los archivos de `artifacts/checkpoints` y comprobar que la API logra bajarlos solita antes de iniciar.
- **Rendimiento Visual**: Cargar el endpoint del EDA y medir el tiempo que tarda con las imágenes estáticas vs el render en vivo.
