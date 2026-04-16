# Plan de Implementación: Fase 4 y 5 (Hub de Modelos, Unificación UI y Lógica Dinámica)

Esta fase se centra en hacer el proyecto robusto para despliegue: externalizar los binarios de modelos a HuggingFace, unificar la interfaz de usuario con un sistema de control de acceso, e implementar el motor de recomendación dinámico.

## Requiere Revisión del Usuario

> [!IMPORTANT]
> **Integración con HuggingFace:** Asumo que el token está guardado en el archivo `.env` como `HF_TOKEN`. Asegúrate de que tiene permisos de "Escritura" en tu repositorio `JulioJ777/Streaming`.

> [!NOTE]
> **Cambios en Base de Datos:** Planeo añadir una columna `role` (ENUM('admin', 'user')) a la tabla `users` para gestionar correctamente el control de acceso.

## Cambios Propuestos

---

### 1. Repositorio Remoto de Modelos (HuggingFace)

#### [MODIFICAR] [upload_models.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/scripts/upload_models.py)
- Refactorizar para que escanee automáticamente las subcarpetas no vacías de `artifacts/` (`weights`, `exports`, `checkpoints`, `mappings`).
- Mantener la estructura de carpetas en el repositorio de HuggingFace (ej: `weights/modelo_1_SVD.joblib`).
- Usar `python-dotenv` para cargar secretos desde `.env`.
- Añadir barra de progreso y resumen al final.

#### [NUEVO] [download_models.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/utils/download_models.py)
- Implementar utilidad usando `huggingface_hub.snapshot_download` o `hf_hub_download`.
- Lógica para descargar solo si los archivos faltan localmente.
- Función que reporte progreso para que la UI pueda mostrar una barra de carga.
- Función `verificar_y_descargar()` que se llame al arrancar la API.

#### [MODIFICAR] [main.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/main.py)
- Antes de lanzar el backend, llamar a `verificar_y_descargar()` para asegurar que los modelos existen.

---

### 2. Unificación del Frontend y Sistema de Accesos

#### [MODIFICAR] [app_ui.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/frontend/app_ui.py)
- Implementar un formulario de Login como pantalla principal.
- Gestionar `st.session_state["user"]` y `st.session_state["role"]`.
- Antes del login, si faltan modelos, mostrar la barra de progreso "Descargando Modelos...".
- Restringir la visibilidad de las páginas en el sidebar según el rol del usuario.

#### [MODIFICAR] [1_Administrador.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/frontend/pages/1_Administrador.py)
#### [MODIFICAR] [2_Usuario.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/frontend/pages/2_Usuario.py)
- Añadir comprobaciones de autorización al inicio de cada página para evitar el acceso directo por URL.

---

### 3. Sistema de Recomendación Dinámico

#### [MODIFICAR] [main_api.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/api/main_api.py)
- Refactorizar la lógica de recomendación en un "Selector Inteligente":
  - **Usuario Nuevo (0 valoraciones)**: Usar `Popularidad / Content-Based (TF-IDF)`.
  - **Usuario Ocasional (1-50 valoraciones)**: Usar `SVD` o `KNN`.
  - **Usuario Experto (>50 valoraciones)**: Usar `Wide&Deep` o `TwoTowers` si los mappings existen.
- Centralizar esta lógica en un único endpoint `/recomendar/smart/{user_id}` o en una función auxiliar.
- Los endpoints individuales (`/recomendar/svd/`, `/recomendar/knn/`, etc.) seguirán existiendo para pruebas directas.

---

### 4. Mejoras Generales y Auditoría

- **Normalización de Rutas**: Asegurar que todas las rutas usen `os.path.join` o rutas relativas desde la raíz para evitar problemas de compatibilidad Windows/Linux.
- **Variables de Entorno**: Mover toda la información sensible hardcodeada al `.env`.
- **Carga Progresiva**: En `main_api.py`, asegurar que el arranque no se caiga si un modelo falla al descargar; simplemente marcar ese modelo como no disponible.

## Preguntas Abiertas

> [!IMPORTANT]
> 1. **Lógica de Roles**: La tabla `users` no tiene columna `role`. ¿Añado la columna a la BBDD o prefieres una lista fija de nombres de usuario "Admin" (ej: `['root', 'admin']`)?
> 2. **Repositorio HuggingFace**: Has mencionado `JulioJ777/Streaming`. ¿Es Público o Privado? Si es privado, necesitamos el `HF_TOKEN` en `.env`.
> 3. **Contenido del Hub**: ¿Subimos **todo** lo de `artifacts/` (incluidos los SVD pesados de >700MB) o solo los exportados ONNX que son más ligeros?
> 4. **Umbrales Dinámicos**: Para la recomendación dinámica, ¿te parecen bien 0 valoraciones = Cold-Start, 1-50 = SVD/KNN, y >50 = Wide&Deep/TwoTowers?

## Plan de Verificación

### Pruebas Automatizadas
- `pytest` para la utilidad `download_models` (mockeando la API de HuggingFace).
- Test de endpoints de la API (`/login`, `/recomendar/smart`).

### Verificación Manual
1. **Recuperación de Modelos**: Borrar una carpeta de `artifacts/`, iniciar `main.py`, y verificar que se redescarga automáticamente.
2. **Control de Acceso**: Iniciar sesión como 'Usuario' y verificar que la página 'Administrador' no es accesible/visible.
3. **Lógica Dinámica**: Comprobar en los logs del terminal qué modelo se está usando para distintos IDs de usuario.
