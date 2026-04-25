-- ============================================================
-- create_user_ratings.sql
-- Tabla para almacenar las valoraciones que los usuarios
-- realizan desde la interfaz de SPIRE Streaming.
-- Separada del CSV ratings_finales_ia.csv que alimenta los modelos.
-- ============================================================

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
