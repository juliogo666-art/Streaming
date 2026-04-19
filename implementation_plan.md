# Sistema de Valoraciones y Tarjetas Interactivas

## Descripción

Implementar dos mejoras en la vista de usuario:
1. **Sistema de valoraciones**: Los usuarios pueden puntuar películas (1-5 estrellas) desde la interfaz, almacenándose en una tabla nueva `user_ratings` en MySQL.
2. **Tarjetas interactivas**: Reemplazar el botón "Ver sinopsis" actual por tarjetas con hover (muestra resumen) y click (abre modal/dialog con info completa).

---

## User Review Required

> [!IMPORTANT]
> **Punto 1 — Valoraciones y su impacto en los modelos de IA:**
> Las valoraciones del usuario se guardarán en una **tabla nueva** (`user_ratings`) en MySQL, **separada** del CSV `ratings_finales_ia.csv` (~434MB) que alimenta los modelos de IA. Esto significa:
> - Las valoraciones nuevas **SÍ se registran** de forma persistente e inmediata.
> - Los modelos de IA **NO se ven afectados** instantáneamente (siguen usando el CSV precargado en memoria al arrancar).
> - En un futuro se podría crear un pipeline de reentrenamiento que incorpore estas valoraciones, pero eso queda fuera de este scope.
> 
> Este enfoque **no rompe nada** de la funcionalidad actual.

> [!WARNING]
> **Punto 2 — Limitaciones de Streamlit con interactividad:**
> Streamlit no tiene soporte nativo para tooltips en hover sobre imágenes ni modals/popups con click. La solución es inyectar HTML/CSS/JS personalizado:
> - **Hover**: overlay CSS puro (no necesita JS) que muestra la sinopsis al pasar el ratón sobre el póster.
> - **Click**: Usaremos `st.dialog` (decorador nativo de Streamlit ≥1.35) para abrir un popup con la información completa. Esto es más limpio que el sidebar actual y compatible con tu versión de Streamlit.

---

## Proposed Changes

### Base de Datos — Nueva tabla `user_ratings`

#### [NEW] [create_user_ratings.sql](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/data/scripts_sql/create_user_ratings.sql)
- Tabla `user_ratings` con columnas: `id_usuario`, `tmdb_id`, `rating` (DECIMAL 2,1 para 0.5-5.0), `created_at`.
- Primary key compuesta `(id_usuario, tmdb_id)` con `ON DUPLICATE KEY UPDATE` para permitir cambiar la valoración.
- FK hacia `users(id_usuario)`.

#### [MODIFY] [setup_completo.sql](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/data/scripts_sql/setup_completo.sql)
- Añadir el bloque de `user_ratings` al script de instalación completa.

---

### Backend API — Endpoints de valoración

#### [MODIFY] [schemas.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/schemas/schemas.py)
- Añadir `RatingRequest(BaseModel)` con campos `user_id: int`, `tmdb_id: int`, `rating: float`.

#### [MODIFY] [main_api.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/api/main_api.py)
- **`POST /api/rating`**: Registra o actualiza la valoración del usuario (`INSERT ... ON DUPLICATE KEY UPDATE`).
- **`GET /api/ratings/{user_id}`**: Devuelve todas las valoraciones del usuario (para precargar estrellas ya dadas).
- **`GET /api/movie/{tmdb_id}`**: Devuelve los datos completos de una película (adult, idioma, descripción, fecha, géneros) para el modal de detalle.

---

### Frontend — Vista de Usuario

#### [MODIFY] [vista_usuario.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/frontend/vista_usuario.py)

**Cambios en `dibujar_tarjetas_contenido()`** (catálogo general):
- Eliminar el botón "Ver sinopsis" y el `st.toast()`.
- Reemplazar `st.image()` por un bloque HTML con overlay CSS que muestra la sinopsis al hacer hover sobre el póster.
- Añadir un botón discreto "ℹ️ Info" debajo de cada tarjeta que abre un `@st.dialog` con la ficha completa (adultos, idioma, descripción, fecha, géneros).
- Añadir un widget de estrellas (⭐) interactivo debajo de cada tarjeta usando `st.feedback("stars")` (nativo Streamlit) ó un selector `st.select_slider` estilizado con estrellas, que al cambiar hace `POST /api/rating`.

**Cambios en `solicitar_y_dibujar_recomendaciones()`** (sección IA):
- Mismos cambios: overlay hover + botón info + estrellas de valoración.
- Eliminar el botón "Ver sinopsis" aquí también.

**Cambios en sección Tragaperras** (serendipia):
- Las tarjetas ya son HTML con sinopsis visible. Añadir únicamente el widget de estrellas al pie de cada tarjeta serendipia.

---

## Diseño Visual de las Tarjetas

```
┌─────────────────────┐
│                     │
│    [POSTER IMG]     │  ← Hover: overlay semitransparente
│                     │     con sinopsis (CSS puro)
│    ╔═══════════╗    │
│    ║ Sinopsis  ║    │  ← Solo visible al hover
│    ║ text ...  ║    │
│    ╚═══════════╝    │
│                     │
├─────────────────────┤
│ **Título** (2024)   │
│ 7.5 / 10            │
│ ⭐⭐⭐⭐☆  Tu nota │  ← Interactivo (Streamlit widget)
│ [ℹ️ Info]            │  ← Abre @st.dialog
└─────────────────────┘
```

**Dialog al hacer click en "ℹ️ Info":**
```
┌──────────────────────────────────┐
│          TÍTULO (2024)      [✕]  │
│──────────────────────────────────│
│ [POSTER]  │ Idioma: en           │
│           │ Adultos: No          │
│           │ Fecha: 2024-01-15    │
│           │ Géneros: Acción, Sci │
│           │                      │
│           │ Descripción completa │
│           │ del contenido...     │
└──────────────────────────────────┘
```

---

## Open Questions

> [!IMPORTANT]
> **¿Escala de valoraciones?** Actualmente los modelos usan escala 0.5-5.0 (con incrementos de 0.5). ¿Quieres la misma escala para las valoraciones del usuario o prefieres estrellas enteras 1-5?
> 
> Streamlit tiene `st.feedback("stars")` que da 5 estrellas enteras. Si necesitas medias estrellas sería un slider customizado.

---

## Verification Plan

### Automated Tests
- Ejecutar el SQL `create_user_ratings.sql` y verificar que la tabla se crea.
- Arrancar la aplicación con `uv run main.py` y probar:
  - `POST /api/rating` con curl/httpx.
  - `GET /api/ratings/{user_id}` para verificar persistencia.
  - `GET /api/movie/{tmdb_id}` para verificar datos completos.

### Manual Verification (Browser)
- Abrir la vista de usuario en el navegador.
- Verificar que al hacer hover sobre un póster aparece la sinopsis.
- Verificar que al hacer click en "ℹ️ Info" se abre el dialog con la ficha completa.
- Verificar que al dar estrellas se guarda la valoración y persiste al recargar.
