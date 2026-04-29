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

-- 4. Tabla de Usuarios (misma definición que en setup_completo.sql; ids desde 500000)
CREATE TABLE users (
    id_usuario INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    passwd VARCHAR(255) NOT NULL, -- Recuerda guardar hashes, no texto plano
    fecha_nacimiento DATE,
    sexo ENUM('Hombre', 'Mujer', 'Otro'),
    role ENUM('admin', 'user') NOT NULL DEFAULT 'user',
    sexo ENUM('Hombre', 'Mujer', 'Otro'),
    role ENUM('admin', 'user') NOT NULL DEFAULT 'user',
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) AUTO_INCREMENT=500000;