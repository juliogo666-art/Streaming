-- ============================================================
-- create_database.sql
-- Esquema base consolidado (sin migraciones posteriores)
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

-- 5. Tabla de Usuarios (incluye sexo y role desde origen)
CREATE TABLE IF NOT EXISTS users (
    id_usuario INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    passwd VARCHAR(255) NOT NULL, -- Recuerda guardar hashes, no texto plano
    fecha_nacimiento DATE,
    sexo ENUM('Hombre', 'Mujer', 'Otro'),
    role ENUM('admin', 'user') NOT NULL DEFAULT 'user',
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) AUTO_INCREMENT = 500000;

-- 6. Tabla de intereses del usuario (géneros favoritos)
CREATE TABLE IF NOT EXISTS user_interests (
    id_usuario INT,
    genre_id INT,
    source VARCHAR(20) NOT NULL DEFAULT 'user_selected', -- user_selected | ml_inferred
    PRIMARY KEY (id_usuario, genre_id),
    FOREIGN KEY (id_usuario) REFERENCES users(id_usuario) ON DELETE CASCADE,
    FOREIGN KEY (genre_id) REFERENCES genres(id) ON DELETE CASCADE
);

-- 7. Tabla para valoraciones de usuarios en la app
CREATE TABLE IF NOT EXISTS user_ratings (
    id_usuario INT NOT NULL,
    tmdb_id INT NOT NULL,
    rating DECIMAL(2,1) NOT NULL, -- 0.5 a 5.0 (en pasos de 0.5)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id_usuario, tmdb_id),
    FOREIGN KEY (id_usuario) REFERENCES users(id_usuario) ON DELETE CASCADE,
    CONSTRAINT chk_rating_range CHECK (rating >= 0.5 AND rating <= 5.0)
);

-- 8. Tabla de Caché de Serendipia (Tragaperras)
-- Sin FK hacia contents para permitir TRUNCATE rápido sin bloqueos de integridad
CREATE TABLE IF NOT EXISTS serendipity_cache (
    id INT AUTO_INCREMENT PRIMARY KEY,
    movie_id INT NOT NULL, -- tmdb_id de la película (ref. contents.tmdb_id)
    genre VARCHAR(50) NOT NULL, -- Nombre del género TMDB (ej: Horror, Drama)
    rating_mean DECIMAL(5,3) NOT NULL, -- Nota media original (vote_average de TMDB)
    vote_count INT NOT NULL, -- Número total de votos de la película
    weighted_rating DECIMAL(8,6) NOT NULL, -- Media Bayesiana: WR = (v*R + m*C) / (v+m), m=50
    serendipity_score DECIMAL(10,8) NOT NULL, -- Score final: WR / log10(v+10)
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_genre (genre),
    INDEX idx_genre_score (genre, serendipity_score DESC)
);