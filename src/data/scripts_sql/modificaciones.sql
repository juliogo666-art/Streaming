#cambiamos el num de decimales para permitir las puntuaciones 10.0
ALTER TABLE contents MODIFY COLUMN vote_average DECIMAL(4,2);

#subir el tamaño máximo de paquete para permitir la inserción de datos más grandes, como descripciones largas o imágenes codificadas en base64
SET GLOBAL max_allowed_packet=67108864; -- Esto lo sube a 64MB

#añadir columna video que faltaba en el DDL original
ALTER TABLE contents ADD COLUMN video BOOLEAN DEFAULT FALSE AFTER title;

#hacer que el id_usuario comience en 500000
#se ha revisado el dataset de ratings para comprobar el max id_usuario, dejamos margen para mas de 150000 ids que podamos añadir a este dataset
ALTER TABLE users AUTO_INCREMENT = 500000;

#si ya existen usuarios, reasignar id_usuario para que empiecen en 500000
SET @nuevo_id := 499999;
UPDATE users u
JOIN (
    SELECT id_usuario, (@nuevo_id := @nuevo_id + 1) AS id_reasignado
    FROM users
    ORDER BY id_usuario
) r ON u.id_usuario = r.id_usuario
SET u.id_usuario = r.id_reasignado
where u.id_usuario > 0;

#añadir origen de interés para distinguir selección manual vs inferencia ML
ALTER TABLE user_interests
ADD COLUMN source VARCHAR(20) NOT NULL DEFAULT 'user_selected' AFTER genre_id;

#backfill de intereses ya existentes:
#- usuarios importados desde MovieLens (id < 500000) => ml_inferred
#- usuarios de registro propio (id >= 500000)       => user_selected
UPDATE user_interests ui
JOIN users u ON u.id_usuario = ui.id_usuario
SET ui.source = CASE
    WHEN u.id_usuario < 500000 THEN 'ml_inferred'
    ELSE 'user_selected'
END;