-- ============================================================
-- setup_completo.sql
-- Instalación completa de streaming_db en un solo script.
-- Equivale a ejecutar en orden:
--   1. create_database.sql
--   2. modificaciones.sql
--   3. migration_user_v2.sql
--   4. migration_user_v3.sql
--   5. create_serendipity_cache.sql
--   6. create_user_ratings.sql
-- ============================================================

CREATE DATABASE IF NOT EXISTS streaming_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE streaming_db;

-- ============================================================
-- BLOQUE 1: create_database.sql — Tablas base
-- ============================================================

-- 1. Tabla Maestra de Contenido (Películas y Series)
CREATE TABLE IF NOT EXISTS contents (
    tmdb_id INT PRIMARY KEY,
    content_type ENUM('movie', 'tv') NOT NULL,
    title VARCHAR(255) NOT NULL, -- Mapea 'title' (movie) y 'name' (tv)
    video BOOLEAN DEFAULT FALSE,
    original_title VARCHAR(255),
    overview TEXT,
    release_date DATE,           -- Mapea 'release_date' y 'first_air_date'
    original_language VARCHAR(10),
    popularity DECIMAL(10,3),
    poster_path VARCHAR(255),
    backdrop_path VARCHAR(255),
    vote_average DECIMAL(4,2),
    vote_count INT,
    adult BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Tabla de Géneros (Catálogo estático de TMDB)
CREATE TABLE IF NOT EXISTS genres (
    id INT PRIMARY KEY,
    name VARCHAR(50) NOT NULL
);

-- 3. Relación Muchos a Muchos (Un contenido puede tener varios géneros)
CREATE TABLE IF NOT EXISTS content_genres (
    content_id INT,
    genre_id INT,
    PRIMARY KEY (content_id, genre_id),
    FOREIGN KEY (content_id) REFERENCES contents(tmdb_id) ON DELETE CASCADE,
    FOREIGN KEY (genre_id) REFERENCES genres(id) ON DELETE CASCADE
);

-- 4. Tabla Unificada de Estadísticas
CREATE TABLE IF NOT EXISTS content_stats (
    content_id INT PRIMARY KEY,
    certificacion VARCHAR(10),
    espectadores_live INT DEFAULT 0,
    reproducciones_totales INT DEFAULT 0,
    es_tendencia BOOLEAN DEFAULT FALSE,
    es_popular BOOLEAN DEFAULT FALSE,
    es_historico_vistas BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (content_id) REFERENCES contents(tmdb_id) ON DELETE CASCADE
);

-- 5. Tabla de Usuarios
CREATE TABLE IF NOT EXISTS users (
    id_usuario INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    passwd VARCHAR(255) NOT NULL, -- Recuerda guardar hashes, no texto plano
    fecha_nacimiento DATE,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================
-- BLOQUE 2: modificaciones.sql — Ajustes sobre las tablas base
-- ============================================================

-- Permitir puntuaciones 10.0 (más decimales en vote_average)
ALTER TABLE contents MODIFY COLUMN vote_average DECIMAL(4,2);

-- Subir el tamaño máximo de paquete para descripciones largas / imágenes base64
SET GLOBAL max_allowed_packet=67108864; -- 64MB

-- Añadir columna video si no existe (añadida posteriormente al DDL original)
ALTER TABLE contents ADD COLUMN IF NOT EXISTS video BOOLEAN DEFAULT FALSE AFTER title;

-- Hacer que id_usuario comience en 500000 (margen para el dataset de ratings)
-- Se revisó el dataset de ratings: max id_usuario deja margen para +150000 ids
ALTER TABLE users AUTO_INCREMENT = 500000;

-- Si ya existen usuarios, reasignar id_usuario para que empiecen en 500000
SET @nuevo_id := 499999;
UPDATE users u
JOIN (
    SELECT id_usuario, (@nuevo_id := @nuevo_id + 1) AS id_reasignado
    FROM users
    ORDER BY id_usuario
) r ON u.id_usuario = r.id_usuario
SET u.id_usuario = r.id_reasignado
WHERE u.id_usuario > 0;

-- ============================================================
-- BLOQUE 3: migration_user_v2.sql — Sexo e Intereses de Usuario
-- ============================================================

-- Añadir columna sexo si no existe
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS sexo ENUM('Hombre', 'Mujer', 'Otro') AFTER fecha_nacimiento;

-- Tabla de intereses del usuario (géneros favoritos)
CREATE TABLE IF NOT EXISTS user_interests (
    id_usuario INT,
    genre_id INT,
    PRIMARY KEY (id_usuario, genre_id),
    FOREIGN KEY (id_usuario) REFERENCES users(id_usuario) ON DELETE CASCADE,
    FOREIGN KEY (genre_id) REFERENCES genres(id) ON DELETE CASCADE
);

-- ============================================================
-- BLOQUE 4: migration_user_v3.sql — Rol para control de acceso
-- ============================================================

-- Añadir columna sexo si no existe
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS sexo ENUM('Hombre', 'Mujer', 'Otro') AFTER fecha_nacimiento;

-- Añadir columna role si no existe
-- Valores posibles: 'admin' para administradores, 'user' para usuarios estándar
ALTER TABLE users
    ADD COLUMN IF NOT EXISTS role ENUM('admin', 'user') NOT NULL DEFAULT 'user' AFTER sexo;

-- Asignar rol de administrador al usuario 'root' (si existe)
UPDATE users SET role = 'admin' WHERE username = 'root';

-- ============================================================
-- BLOQUE 5: modificaciones.sql (cont.) — Origen de intereses
-- ============================================================

-- Añadir columna source para distinguir selección manual vs inferencia ML
ALTER TABLE user_interests
    ADD COLUMN IF NOT EXISTS source VARCHAR(20) NOT NULL DEFAULT 'user_selected' AFTER genre_id;

-- Backfill del origen de intereses:
--   · usuarios importados desde MovieLens (id < 500000) => ml_inferred
--   · usuarios de registro propio (id >= 500000)        => user_selected
UPDATE user_interests ui
JOIN users u ON u.id_usuario = ui.id_usuario
SET ui.source = CASE
    WHEN u.id_usuario < 500000 THEN 'ml_inferred'
    ELSE 'user_selected'
END;

-- ============================================================
-- BLOQUE 6: create_serendipity_cache.sql — Tragaperras de Serendipia
-- ============================================================

-- Tabla de Caché de Serendipia (Tragaperras)
-- Vaciar (TRUNCATE) y rellenar semanalmente con src/serendipia/actualizar_cache_cron.py
-- Sin FK hacia contents para permitir TRUNCATE rápido sin bloqueos de integridad
CREATE TABLE IF NOT EXISTS serendipity_cache (
    id INT AUTO_INCREMENT PRIMARY KEY,
    movie_id INT NOT NULL,              -- tmdb_id de la película (ref. contents.tmdb_id)
    genre VARCHAR(50) NOT NULL,         -- Nombre del género TMDB (ej: Horror, Drama)
    rating_mean DECIMAL(5,3) NOT NULL,  -- Nota media original (vote_average de TMDB)
    vote_count INT NOT NULL,            -- Número total de votos de la película
    weighted_rating DECIMAL(8,6) NOT NULL,   -- Media Bayesiana: WR = (v*R + m*C) / (v+m), m=50
    serendipity_score DECIMAL(10,8) NOT NULL, -- Score final: WR / log10(v+10)
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_genre (genre),
    INDEX idx_genre_score (genre, serendipity_score DESC)
);

-- ============================================================
-- BLOQUE 7: create_user_ratings.sql — Valoraciones de usuarios
-- ============================================================

-- Tabla para almacenar las valoraciones que los usuarios
-- realizan desde la interfaz de SPIRE Streaming.
-- Separada del CSV ratings_finales_ia.csv que alimenta los modelos.
CREATE TABLE IF NOT EXISTS user_ratings (
    id_usuario INT NOT NULL,
    tmdb_id    INT NOT NULL,
    rating     DECIMAL(2,1) NOT NULL,          -- 0.5 a 5.0 (en pasos de 0.5)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (id_usuario, tmdb_id),
    FOREIGN KEY (id_usuario) REFERENCES users(id_usuario) ON DELETE CASCADE,

    -- Restricción para garantizar rango válido
    CONSTRAINT chk_rating_range CHECK (rating >= 0.5 AND rating <= 5.0)
);
