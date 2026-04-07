import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

db_config = {
    "host": "localhost",
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_NAME", "streaming_db"),
    "port": 3306
}

migration_sql = """
ALTER TABLE users 
ADD COLUMN sexo ENUM('Hombre', 'Mujer', 'Otro') AFTER fecha_nacimiento;

CREATE TABLE IF NOT EXISTS user_interests (
    id_usuario INT,
    genre_id INT,
    PRIMARY KEY (id_usuario, genre_id),
    FOREIGN KEY (id_usuario) REFERENCES users(id_usuario) ON DELETE CASCADE,
    FOREIGN KEY (genre_id) REFERENCES genres(id) ON DELETE CASCADE
);
"""

try:
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    
    # Split the migration into separate statements because execute() can't run multiple
    for statement in migration_sql.split(";"):
        if statement.strip():
            cursor.execute(statement)
            print(f"Executed: {statement.strip()[:50]}...")
    
    conn.commit()
    print("Migration successful!")
    cursor.close()
    conn.close()
except Exception as e:
    print(f"Error: {e}")
