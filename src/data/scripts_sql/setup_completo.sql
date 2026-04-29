-- ============================================================
-- setup_completo.sql
-- Instalación completa de streaming_db (DDL en CREATE TABLE + DML mínimo).
-- Los scripts sueltos del mismo directorio siguen sirviendo para migraciones
-- o referencia, pero el esquema definitivo de instalación en frío es este.
-- ============================================================

CREATE DATABASE IF NOT EXISTS streaming_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE streaming_db;

-- ============================================================
-- BLOQUE 1: solo CREATE TABLE (orden por dependencias de FK)
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

-- 4. Usuarios (id manual desde 500000: margen frente a ids del dataset de ratings)
CREATE TABLE IF NOT EXISTS users (
    id_usuario INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    passwd VARCHAR(255) NOT NULL, -- Recuerda guardar hashes, no texto plano
    fecha_nacimiento DATE,
    sexo ENUM('Hombre', 'Mujer', 'Otro'),
    role ENUM('admin', 'user') NOT NULL DEFAULT 'user', -- 'admin' | 'user'
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) AUTO_INCREMENT=500000;

-- 5. Intereses del usuario (géneros favoritos + origen del dato)
CREATE TABLE IF NOT EXISTS user_interests (
    id_usuario INT,
    genre_id INT,
    source VARCHAR(20) NOT NULL DEFAULT 'user_selected',
    PRIMARY KEY (id_usuario, genre_id),
    FOREIGN KEY (id_usuario) REFERENCES users(id_usuario) ON DELETE CASCADE,
    FOREIGN KEY (genre_id) REFERENCES genres(id) ON DELETE CASCADE
);

-- 6. Caché de Serendipia (Tragaperras)
-- Vaciar (TRUNCATE) y rellenar semanalmente con src/etl/actualizar_cache_cron.py
-- Sin FK hacia contents para TRUNCATE rápido
CREATE TABLE IF NOT EXISTS serendipity_cache (
    id INT AUTO_INCREMENT PRIMARY KEY,
    movie_id INT NOT NULL,              -- tmdb_id (ref. contents.tmdb_id)
    genre VARCHAR(50) NOT NULL,         -- Nombre TMDB (ej. Horror, Drama)
    rating_mean DECIMAL(5,3) NOT NULL,
    vote_count INT NOT NULL,
    weighted_rating DECIMAL(8,6) NOT NULL,
    serendipity_score DECIMAL(10,8) NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_genre (genre),
    INDEX idx_genre_score (genre, serendipity_score DESC)
);

-- 7. Valoraciones desde la app (distinto de ratings_finales_ia.csv)
CREATE TABLE IF NOT EXISTS user_ratings (
    id_usuario INT NOT NULL,
    tmdb_id    INT NOT NULL,
    rating     DECIMAL(2,1) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

    PRIMARY KEY (id_usuario, tmdb_id),
    FOREIGN KEY (id_usuario) REFERENCES users(id_usuario) ON DELETE CASCADE,

    CONSTRAINT chk_rating_range CHECK (rating >= 0.5 AND rating <= 5.0)
);

-- ============================================================
-- BLOQUE 2: parámetros de servidor y DML
-- (La reasignación de ids solo es segura con user_interests vacía)
-- ============================================================

SET GLOBAL max_allowed_packet=67108864; -- 64MB (descripciones / imágenes base64)

-- Reasignar ids de usuario a partir de 500000 (BD vacía o sin filas en user_interests)
SET @nuevo_id := 499999;
UPDATE users u
JOIN (
    SELECT id_usuario, (@nuevo_id := @nuevo_id + 1) AS id_reasignado
    FROM users
    ORDER BY id_usuario
) r ON u.id_usuario = r.id_usuario
SET u.id_usuario = r.id_reasignado
WHERE u.id_usuario > 0;

UPDATE users SET role = 'admin' WHERE username = 'root';

-- Origen de intereses: import MovieLens (id < 500000) vs registro app (id >= 500000)
UPDATE user_interests ui
JOIN users u ON u.id_usuario = ui.id_usuario
SET ui.source = CASE
    WHEN u.id_usuario < 500000 THEN 'ml_inferred'
    ELSE 'user_selected'
END;
