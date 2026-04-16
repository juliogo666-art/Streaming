# Unificación del Frontend: Login Centralizado y Roles

## Contexto

Actualmente el frontend tiene 3 archivos Streamlit separados con sidebars que permiten navegar entre páginas. El usuario quiere:
1. **Eliminar el sidebar de navegación** entre páginas
2. **`app_ui.py` = Pantalla de Login** única (con el diseño premium azul marino + oro de `2_Usuario.py`)
3. **Redirección por rol**: usuario → vista simplificada / admin → panel completo
4. **Vista de Usuario simplificada**: sin sidebar, solo la recomendación Smart más adecuada
5. **Nuevo tab en Admin**: "Recomendaciones IA" con selector de modelo, selector de usuario por ID, y las recomendaciones (lo que actualmente tiene la pantalla de usuario)
6. **Etiqueta de identidad**: nombre de usuario + ID junto al botón "Cerrar Sesión"

## Cambios Propuestos

### 1. Reestructurar la Navegación

> [!IMPORTANT]
> Streamlit con `pages/` crea automáticamente el sidebar de navegación. Para eliminarlo necesitamos que toda la lógica viva dentro de `app_ui.py` y los archivos de `pages/` se eliminen. Esto centraliza todo en un solo archivo que gestiona el estado con `st.session_state`.

**Problema**: Streamlit auto-descubre archivos en la carpeta `pages/` y los muestra en el sidebar.  
**Solución**: Eliminar los archivos de `pages/` y mover toda la lógica a `app_ui.py`, gestionando qué "pantalla" se muestra según `st.session_state["role"]`.

---

### 2. Archivo Principal

#### [MODIFICAR] [app_ui.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/frontend/app_ui.py) — REESCRITURA COMPLETA

El archivo se reestructura en 3 secciones según el estado de sesión:

**Estado 1 — No autenticado (`st.session_state["autenticado"] == False`)**:
- Pantalla de login idéntica visualmente a la actual de `2_Usuario.py` (título "BIENVENIDO A SPIRE STREAMING", tabs "Iniciar Sesión" / "Registrarse")
- Al hacer login, guarda en `session_state`: `autenticado=True`, `usuario_actual={...}`, `role="user"|"admin"`
- Según el `role` devuelto por `/login`, redirige a la vista correspondiente

**Estado 2 — Autenticado como `user`**:
- **Sin sidebar** (se oculta con CSS `display: none`)
- Barra superior: `"Bienvenido a SPIRE, {nombre}!"` + etiqueta `"👤 {username} (ID: {id})"` + botón `"Cerrar Sesión"`
- Solo muestra la recomendación Smart Auto (endpoint `recomendar/smart/{user_id}`) directamente
- Mantiene las tabs "Películas" / "Series" con el catálogo (Top rated, Más vistos)
- **No hay selector de modelo ni selector de ID** — el usuario solo ve su recomendación personalizada

**Estado 3 — Autenticado como `admin`**:
- Barra superior: `"Panel de Control de Administrador"` + etiqueta `"🔑 {username} (ID: {id})"` + botón `"Cerrar Sesión"`
- 4 tabs:
  1. **📊 Gestión de Usuarios**: Botón "Sincronizar con MySQL", botón "Importar Datos", tabla de usuarios
  2. **📈 Análisis Exploratorio (EDA)**: Gráficas estáticas (igual que ahora)
  3. **🤖 Rendimiento Modelos IA**: Tabla de métricas + gráficas comparativas (igual que ahora)
  4. **🎬 Recomendaciones IA** *(NUEVO)*: Selector de modelo, input de user ID, y las recomendaciones renderizadas — la funcionalidad actual de la vista de usuario pero con todos los controles

---

### 3. Archivos a Eliminar

#### [DELETE] [1_Administrador.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/frontend/pages/1_Administrador.py)
#### [DELETE] [2_Usuario.py](file:///c:/Users/User/Desktop/Programacion/Proyecto%204%20-%20Streaming/Proyecto%204%20-%20Streaming%20-%20main/src/frontend/pages/2_Usuario.py)

Se eliminan para que Streamlit no genere el sidebar de navegación automática. **Todo su código se integra en `app_ui.py`**.

> [!WARNING]
> Al eliminar la carpeta `pages/`, el sidebar automático de Streamlit desaparece por completo. Los botones "Sincronizar con MySQL" y "Importar Datos" que estaban en el sidebar del admin se moverán a la primera tab "Gestión de Usuarios".

---

## Resumen Visual

```
┌─────────────────────────────────────────────┐
│              app_ui.py                      │
│                                             │
│  ┌─ No autenticado ──────────────────────┐  │
│  │  Login / Registro (estilo premium)    │  │
│  └───────────────────────────────────────┘  │
│                                             │
│  ┌─ Rol: user ───────────────────────────┐  │
│  │  Sin sidebar                          │  │
│  │  👤 nombre (ID: X) | [Cerrar Sesión]  │  │
│  │  Recomendación Smart Auto directa     │  │
│  │  Tabs: Películas | Series             │  │
│  └───────────────────────────────────────┘  │
│                                             │
│  ┌─ Rol: admin ──────────────────────────┐  │
│  │  🔑 nombre (ID: X) | [Cerrar Sesión]  │  │
│  │  Tabs: Gestión | EDA | Modelos IA     │  │
│  │        | 🎬 Recomendaciones (NUEVO)   │  │
│  └───────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

## Verificación

### Pruebas Manuales
1. **Login como usuario normal**: Verificar que NO aparece sidebar, solo se ve la recomendación Smart y la etiqueta con nombre+ID
2. **Login como admin (root)**: Verificar que aparecen los 4 tabs, que el tab "Recomendaciones IA" tiene selector de modelo y selector de usuario por ID
3. **Cerrar sesión**: Verificar que vuelve a la pantalla de login
4. **Intentar acceder sin login**: Verificar que siempre se muestra el login primero
