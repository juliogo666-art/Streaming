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
