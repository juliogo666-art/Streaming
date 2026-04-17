# Importación de usuarios MovieLens (`import_ml_users_from_ratings.py`)

Documentación del script: **`src/scripts/import_ml_users_from_ratings.py`**.

Este script **no sustituye** al registro por API (`POST /register` en Streamlit). Sirve para **poblar la base de datos** con usuarios derivados del fichero **`src/data/ready/ratings_finales_ia.csv`**, que solo contiene interacciones `userId`, `tmdb_id`, `rating`. Así puedes tener en MySQL usuarios con IDs alineados al dataset de entrenamiento y una **estimación de gustos por género** en la tabla `user_interests`, útil para comparar con recomendaciones o enriquecer el perfil en el futuro.

## Requisitos previos

1. **MySQL** con las tablas del proyecto (`users`, `user_interests`, `genres`, `content_genres`, etc.) creadas según `src/data/scripts_sql/`.
2. **Catálogo y relaciones género–contenido** ya importadas, de modo que exista el cruce `tmdb_id` → `genre_id` en **`content_genres`** (y los géneros en **`genres`**). Sin esto no se pueden inferir intereses. Suele hacerse con el ETL (`POST /importar_datos` en FastAPI o `uv run python src/api/etl.py` desde la raíz del repo).
3. **`.env`** con `DB_USER` y `DB_PASSWORD` válidos (misma configuración que usa `src/api/database.py`).
4. El CSV **`ratings_finales_ia.csv`** es muy grande (~25,7M filas); la importación hace **dos lecturas completas** en bloques (chunks). Reserva tiempo y espacio en disco razonable.

## Cómo ejecutarlo

Desde la **raíz del repositorio** (recomendado):

```bash
uv run python src/scripts/import_ml_users_from_ratings.py
```

Opciones útiles:

| Opción | Descripción |
|--------|-------------|
| `--dry-run` | Ejecuta el análisis del CSV y los conteos, pero **no escribe** en la base de datos (sí consulta BD para cargar `genres` y `content_genres`). |
| `--ratings RUTA` | Ruta alternativa al CSV de valoraciones (por defecto: `src/data/ready/ratings_finales_ia.csv`). |
| `--chunk-size N` | Filas por bloque al leer el CSV (por defecto `500000`). |
| `--min-ratings N` | Umbral mínimo de valoraciones por usuario para ser importado (por defecto `1000`). |

En Windows, si la salida se retrasa al redirigir logs o ejecutar en segundo plano, conviene forzar salida sin buffer:

```powershell
$env:PYTHONUNBUFFERED="1"
uv run python -u src/scripts/import_ml_users_from_ratings.py
```

## Qué usuarios se importan

- Se recorre todo el CSV y se **cuenta cuántas valoraciones** tiene cada `userId`.
- Solo entran los usuarios con **al menos 1000 valoraciones** (configurable con `--min-ratings`).
- En una ejecución real de referencia, el número de usuarios que cumplen el criterio fue del orden de **~2000**; depende del CSV y del umbral.

## Cómo se eligen los géneros (lo más importante)

El objetivo es aproximar **“qué géneros le gustan”** a partir del historial de votos, de forma parecida a la idea de perfilar gustos para el login o el front (la UI concreta puede venir después).

1. **Solo votos “positivos”**  
   Se consideran filas con **`rating >= 4.0`**. El resto no suma al perfil de género (se asume que no indican preferencia clara).

2. **Cruce con la base de datos**  
   Para cada par `(userId, tmdb_id)` elegible, se obtienen los **`genre_id`** asociados a ese `tmdb_id` mediante la tabla **`content_genres`** (el `content_id` coincide con el `tmdb_id` del catálogo).  
   Los nombres de género están en **`genres`** (`id`, `name`); el script los carga para tener el catálogo coherente con la BD (la selección del top usa **IDs**).

3. **Puntuación por género**  
   Para cada género aparecido en una película/ítem valorada con ≥4, se suma el **valor del rating** como peso (no solo contar “1” por película). Así, un 5 estrellas pesa más que un 4 en el mismo género.

4. **Top 3**  
   Por usuario se ordenan los géneros por esa suma y se guardan los **tres primeros** en `user_interests` (`id_usuario`, `genre_id`).

5. **Casos extremos**  
   Si un usuario no tiene ningún voto ≥4 con `tmdb_id` presente en `content_genres`, puede quedar **sin filas** en `user_interests` (es raro si el catálogo está bien poblado).

## Datos sintéticos del usuario (`users`)

El CSV no trae nombre, email ni contraseña. Para cada `userId` importado se rellena:

| Campo | Valor |
|--------|--------|
| `id_usuario` | Igual al **`userId`** del CSV (inserción explícita, no autoincrement aleatorio). |
| `username` | `usuario_<id>` |
| `email` | `usuario_<id>@gmail.com` |
| `passwd` | Hash **bcrypt** de la contraseña fija **`recomiendame`** (misma idea que en el registro por API). |
| `fecha_nacimiento` | **Pseudoaleatoria** pero **reproducible**: se usa `random.Random(user_id)` entre **1990-01-01** y **2005-01-01**. |
| `sexo` | Aleatorio entre `Hombre`, `Mujer`, `Otro` con la misma semilla por `user_id`. |
| `role` | Siempre **`user`**. |
| `fecha_registro` | **`NOW()`** en el momento del insert/update. |

## Tabla `user_interests` y convivencia con datos existentes

- **No** se hace `TRUNCATE` de toda la tabla.
- Para cada usuario que el script **actualiza o crea** en esta ingesta, se hace **`DELETE FROM user_interests WHERE id_usuario = ?`** **solo para ese** `id_usuario` y después se insertan hasta **3** filas con los géneros inferidos. El resto de usuarios no se toca.

## Conflictos con usuarios “reales”

Si ya existe un `id_usuario` en `users` pero el **`username` no es** `usuario_<id>` (por ejemplo un registro manual distinto), el script **no modifica** ese usuario ni sus intereses (`skip_conflict`), para no pisar cuentas reales.

## Rendimiento esperado

- Una pasada para **conteos por usuario** y otra para **agregar scores de género** sobre ~25M filas dominan el tiempo (del orden de **varios minutos** en máquina típica, según disco y CPU).
- La fase de **escritura en BD** es un bucle por usuario candidato (miles de commits pequeños); es mucho más ligera que las dos pasadas al CSV.

## Resumen del flujo por pasos (salida del script)

1. Carga desde BD: `genres` + `content_genres`.
2. Cuenta valoraciones por `userId`; filtra ≥ umbral.
3. Segunda pasada del CSV: acumula scores de género (rating ≥ 4).
4. Inserta/actualiza `users` y reescribe `user_interests` por usuario según las reglas anteriores.
