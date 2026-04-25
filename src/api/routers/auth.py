"""
Router de Autenticación — Login y Registro de usuarios.
"""

import logging

from fastapi import APIRouter, HTTPException
import bcrypt

from ..database import get_db_connection
from ...schemas.schemas import LoginRequest, RegisterRequest

logger = logging.getLogger("streaming_api")

router = APIRouter()


def _adjuntar_gustos_perfil(cursor, usuario: dict) -> None:
    """Rellena gustos_top3 y gustos_source (prioridad ML; si no hay, user_selected)."""
    uid = usuario["id_usuario"]
    cursor.execute(
        """
        SELECT g.id, g.name
        FROM user_interests ui
        JOIN genres g ON g.id = ui.genre_id
        WHERE ui.id_usuario = %s
          AND ui.source = 'ml_inferred'
        ORDER BY g.name ASC
        LIMIT 3
        """,
        (uid,),
    )
    gustos_ml = cursor.fetchall()
    if gustos_ml:
        usuario["gustos_top3"] = [row["name"] for row in gustos_ml]
        usuario["gustos_source"] = "ml_inferred"
        return
    cursor.execute(
        """
        SELECT g.id, g.name
        FROM user_interests ui
        JOIN genres g ON g.id = ui.genre_id
        WHERE ui.id_usuario = %s
          AND (ui.source = 'user_selected' OR ui.source IS NULL)
        ORDER BY g.name ASC
        """,
        (uid,),
    )
    gustos_sel = cursor.fetchall()
    if gustos_sel:
        usuario["gustos_top3"] = [row["name"] for row in gustos_sel]
        usuario["gustos_source"] = "user_selected"


@router.post("/login")
def login(datos: LoginRequest):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # 1. Buscamos al usuario solo por nombre de usuario
        query = "SELECT id_usuario, username, email, passwd, role FROM users WHERE username = %s"
        cursor.execute(query, (datos.username,))
        usuario = cursor.fetchone()

        if usuario:
            # Obtenemos el hash que estaba guardado en BD
            hash_guardado = usuario["passwd"]

            # 2. Comprobamos si la contraseña coincide (Soportando tanto Hash nuevo como Texto Plano antiguo)
            es_valido = False
            if hash_guardado.startswith("$2b$") or hash_guardado.startswith("$2a$"):
                # Si es un hash de bcrypt válido
                es_valido = bcrypt.checkpw(
                    datos.password.encode("utf-8"), hash_guardado.encode("utf-8")
                )
            else:
                # Si es una contraseña antigua en texto plano (ej: 'root')
                es_valido = datos.password == hash_guardado

            if es_valido:
                _adjuntar_gustos_perfil(cursor, usuario)

                # Por seguridad, borramos la contraseña del diccionario temporal antes de mandarlo al frontend
                del usuario["passwd"]
                return {"status": "success", "message": "Login exitoso", "user": usuario}

            raise HTTPException(
                status_code=401, detail="Usuario o contraseña incorrectos"
            )

        raise HTTPException(status_code=401, detail="Usuario o contraseña incorrectos")
    finally:
        cursor.close()
        conn.close()


@router.post("/register")
def register(datos: RegisterRequest):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        # 1. Verificar si el usuario o email ya existen
        cursor.execute(
            "SELECT id_usuario FROM users WHERE username = %s OR email = %s",
            (datos.username, datos.email),
        )
        if cursor.fetchone():
            raise HTTPException(
                status_code=400, detail="El nombre de usuario o email ya están en uso."
            )

        # 2. Hashear la contraseña
        password_hash = bcrypt.hashpw(
            datos.password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

        # 3. Insertar usuario
        query_user = """
            INSERT INTO users (username, email, passwd, fecha_nacimiento, sexo)
            VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(
            query_user,
            (
                datos.username,
                datos.email,
                password_hash,
                datos.fecha_nacimiento,
                datos.sexo,
            ),
        )
        id_usuario = cursor.lastrowid

        # 4. Manejar intereses
        intereses = datos.intereses
        if not intereses:
            # Seleccionar los 3 géneros más populares (basado en reproducciones totales)
            query_pop = """
                SELECT cg.genre_id
                FROM content_genres cg
                JOIN content_stats cs ON cg.content_id = cs.content_id
                GROUP BY cg.genre_id
                ORDER BY SUM(cs.reproducciones_totales) DESC
                LIMIT 3
            """
            cursor.execute(query_pop)
            res_pop = cursor.fetchall()
            intereses = [row["genre_id"] for row in res_pop]

        # Insertar intereses (origen: selección en el registro)
        if intereses:
            query_int = """
                INSERT INTO user_interests (id_usuario, genre_id, source)
                VALUES (%s, %s, 'user_selected')
            """
            for g_id in intereses:
                cursor.execute(query_int, (id_usuario, g_id))

        conn.commit()

        cursor.execute(
            "SELECT id_usuario, username, email, role FROM users WHERE id_usuario = %s",
            (id_usuario,),
        )
        usuario_resp = cursor.fetchone()
        if not usuario_resp:
            raise HTTPException(
                status_code=500, detail="Usuario creado pero no se pudo leer el perfil."
            )
        _adjuntar_gustos_perfil(cursor, usuario_resp)

        return {
            "status": "success",
            "message": "Usuario registrado correctamente",
            "user_id": id_usuario,
            "user": usuario_resp,
        }

    except Exception as e:
        conn.rollback()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error en el servidor: {str(e)}")
    finally:
        cursor.close()
        conn.close()
