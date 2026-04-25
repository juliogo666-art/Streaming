-- Migración v2: Añadir Sexo e Intereses de Usuario
ALTER TABLE users 
ADD COLUMN sexo ENUM('Hombre', 'Mujer', 'Otro') AFTER fecha_nacimiento;

CREATE TABLE IF NOT EXISTS user_interests (
    id_usuario INT,
    genre_id INT,
    PRIMARY KEY (id_usuario, genre_id),
    FOREIGN KEY (id_usuario) REFERENCES users(id_usuario) ON DELETE CASCADE,
    FOREIGN KEY (genre_id) REFERENCES genres(id) ON DELETE CASCADE
);
