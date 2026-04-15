-- Migración v3: Añadir columna de Rol para control de acceso
-- Valores posibles: 'admin' para administradores, 'user' para usuarios estándar
-- Por defecto todos los usuarios nuevos serán 'user'

ALTER TABLE users
ADD COLUMN role ENUM('admin', 'user') NOT NULL DEFAULT 'user' AFTER sexo;

-- Asignar rol de administrador al usuario 'root' (si existe)
UPDATE users SET role = 'admin' WHERE username = 'root';
