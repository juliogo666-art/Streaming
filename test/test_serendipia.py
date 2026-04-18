"""
test_serendipia.py
==================
Tests de integración para el endpoint GET /api/serendipia/{user_id}.

Flujo de cada test:
  1. Inserta géneros favoritos de prueba en `user_interests` para un usuario real.
  2. Llama al endpoint vía TestClient (sin levantar servidor).
  3. Valida esquema y coherencia de la respuesta.
  4. Limpia los datos de prueba (fixture con yield).

Uso:
    uv run pytest test/test_serendipia.py -v
"""

import pytest
from fastapi.testclient import TestClient

from src.api.database import get_db_connection
from src.api.main_api import app, _GENRE_ES_TO_EN

client = TestClient(app)

# ---------------------------------------------------------------------------
# Constantes de prueba
# ---------------------------------------------------------------------------

# Usamos el primer usuario real de la BD.
# Los géneros elegidos tienen traducción directa español → inglés confirmada.
# genre_id=18 → "Drama"      → cache: "Drama"
# genre_id=10749 → "Romance"  → cache: "Romance"
TEST_GENRE_IDS = [18, 10749]   # Drama, Romance
_TEST_USER_ID: int | None = None  # se rellena en la fixture


def _get_first_user_id() -> int:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id_usuario FROM users ORDER BY id_usuario LIMIT 1")
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        pytest.skip("No hay usuarios en la base de datos.")
    return row[0]


# ---------------------------------------------------------------------------
# Fixture: inserta/elimina user_interests de prueba
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def usuario_con_generos():
    """Inserta intereses de prueba y los elimina al terminar el módulo."""
    user_id = _get_first_user_id()

    conn = get_db_connection()
    cur = conn.cursor()
    # Insertar géneros favoritos de prueba (IGNORE por si ya existen)
    for gid in TEST_GENRE_IDS:
        cur.execute(
            "INSERT IGNORE INTO user_interests (id_usuario, genre_id, source) VALUES (%s, %s, 'test')",
            (user_id, gid),
        )
    conn.commit()
    cur.close()
    conn.close()

    yield user_id

    # --- Teardown: limpiar solo las filas con source='test' ---
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM user_interests WHERE id_usuario = %s AND source = 'test'",
        (user_id,),
    )
    conn.commit()
    cur.close()
    conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSerendipiaEndpoint:

    def test_respuesta_200(self, usuario_con_generos):
        """El endpoint debe devolver HTTP 200."""
        user_id = usuario_con_generos
        response = client.get(f"/api/serendipia/{user_id}")
        assert response.status_code == 200, response.text

    def test_esquema_respuesta(self, usuario_con_generos):
        """La respuesta debe contener user_id, generos_favoritos y recomendaciones."""
        user_id = usuario_con_generos
        data = client.get(f"/api/serendipia/{user_id}").json()

        assert "user_id" in data
        assert "generos_favoritos" in data
        assert "recomendaciones" in data

    def test_user_id_correcto(self, usuario_con_generos):
        """El user_id de la respuesta debe coincidir con el solicitado."""
        user_id = usuario_con_generos
        data = client.get(f"/api/serendipia/{user_id}").json()
        assert data["user_id"] == user_id

    def test_devuelve_exactamente_3_peliculas(self, usuario_con_generos):
        """Debe devolver exactamente 3 recomendaciones."""
        user_id = usuario_con_generos
        data = client.get(f"/api/serendipia/{user_id}").json()
        assert len(data["recomendaciones"]) == 3

    def test_estructura_de_cada_recomendacion(self, usuario_con_generos):
        """Cada recomendación debe tener movie_id, genre y serendipity_score."""
        user_id = usuario_con_generos
        data = client.get(f"/api/serendipia/{user_id}").json()
        for rec in data["recomendaciones"]:
            assert "movie_id" in rec, f"Falta movie_id en {rec}"
            assert "genre" in rec, f"Falta genre en {rec}"
            assert "serendipity_score" in rec, f"Falta serendipity_score en {rec}"

    def test_scores_entre_0_y_1(self, usuario_con_generos):
        """Los serendipity_score deben estar en el rango (0, 1]."""
        user_id = usuario_con_generos
        data = client.get(f"/api/serendipia/{user_id}").json()
        for rec in data["recomendaciones"]:
            score = rec["serendipity_score"]
            assert 0 < score <= 1.0, f"Score fuera de rango: {score}"

    def test_generos_son_los_correctos(self, usuario_con_generos):
        """Los géneros devueltos deben corresponder a los intereses del usuario."""
        user_id = usuario_con_generos
        data = client.get(f"/api/serendipia/{user_id}").json()

        # Los géneros en la respuesta son los nombres en español de la tabla genres
        generos_respuesta = set(data["generos_favoritos"])
        # Traducir TEST_GENRE_IDS a nombres español esperados
        conn = get_db_connection()
        cur = conn.cursor()
        placeholders = ",".join(["%s"] * len(TEST_GENRE_IDS))
        cur.execute(f"SELECT name FROM genres WHERE id IN ({placeholders})", TEST_GENRE_IDS)
        nombres_esperados = {row[0] for row in cur.fetchall()}
        cur.close()
        conn.close()

        assert generos_respuesta == nombres_esperados, (
            f"Géneros respuesta: {generos_respuesta} | Esperados: {nombres_esperados}"
        )

    def test_peliculas_son_distintas(self, usuario_con_generos):
        """Las 3 películas recomendadas deben tener movie_id diferente."""
        user_id = usuario_con_generos
        data = client.get(f"/api/serendipia/{user_id}").json()
        ids = [r["movie_id"] for r in data["recomendaciones"]]
        assert len(ids) == len(set(ids)), f"movie_ids duplicados: {ids}"

    def test_resultado_varía_entre_llamadas(self, usuario_con_generos):
        """El muestreo ponderado debe producir resultados distintos en sucesivas llamadas
        (probabilidad de colisión con 1000 candidatos ≈ 0 en 5 intentos)."""
        user_id = usuario_con_generos
        resultados = set()
        for _ in range(5):
            data = client.get(f"/api/serendipia/{user_id}").json()
            ids = tuple(sorted(r["movie_id"] for r in data["recomendaciones"]))
            resultados.add(ids)
        assert len(resultados) > 1, "El endpoint devuelve siempre el mismo resultado (no aleatorio)"

    def test_usuario_sin_generos_devuelve_404(self):
        """Un user_id sin intereses registrados debe devolver 404."""
        response = client.get("/api/serendipia/999999999")
        assert response.status_code == 404


class TestGeneroMapeo:
    """Verifica que el mapeo español→inglés sea completo y coherente."""

    def test_todos_los_generos_espanoles_tienen_traduccion(self):
        """Todos los géneros de la tabla `genres` deben tener traducción en _GENRE_ES_TO_EN."""
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT name FROM genres")
        nombres_db = {row[0] for row in cur.fetchall()}
        cur.close()
        conn.close()

        # Géneros TV de TMDB (Action & Adventure, Kids, etc.) no están en serendipity_cache
        # y pueden ignorarse en el mapeo; sólo validamos los géneros de película.
        generos_pelicula_es = {
            "Acción", "Aventura", "Animación", "Comedia", "Crimen", "Documental",
            "Drama", "Familia", "Fantasía", "Historia", "Terror", "Música",
            "Misterio", "Romance", "Ciencia ficción", "Película de TV", "Suspense",
            "Bélica", "Western",
        }
        sin_traduccion = generos_pelicula_es - set(_GENRE_ES_TO_EN.keys())
        assert not sin_traduccion, f"Géneros sin traducción en _GENRE_ES_TO_EN: {sin_traduccion}"

    def test_traducciones_apuntan_a_generos_en_cache(self):
        """Los valores del mapeo deben existir como géneros en serendipity_cache."""
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT genre FROM serendipity_cache")
        generos_cache = {row[0] for row in cur.fetchall()}
        cur.close()
        conn.close()

        sin_cache = set(_GENRE_ES_TO_EN.values()) - generos_cache
        assert not sin_cache, f"Géneros del mapeo no encontrados en serendipity_cache: {sin_cache}"


# ---------------------------------------------------------------------------
# Ejecución manual: python -m test.test_serendipia  (o uv run python -m test.test_serendipia)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    user_id = _get_first_user_id()

    # Insertar géneros de prueba temporalmente
    conn = get_db_connection()
    cur = conn.cursor()
    for gid in TEST_GENRE_IDS:
        cur.execute(
            "INSERT IGNORE INTO user_interests (id_usuario, genre_id, source) VALUES (%s, %s, 'test')",
            (user_id, gid),
        )
    conn.commit()
    cur.close()
    conn.close()

    print(f"\n{'='*60}")
    print(f"  DEMO SERENDIPIA — usuario {user_id}")
    print(f"{'='*60}")

    try:
        # Llamar al endpoint 3 veces para mostrar que los resultados varían
        for ronda in range(1, 4):
            response = client.get(f"/api/serendipia/{user_id}")
            if response.status_code != 200:
                print(f"ERROR {response.status_code}: {response.json()}")
                break

            data = response.json()
            generos = data["generos_favoritos"]
            recs = data["recomendaciones"]

            print(f"\n--- Ronda {ronda} ---")
            print(f"Géneros favoritos : {' | '.join(generos)}")
            print(f"{'movie_id':<12} {'género':<20} {'score':>10}")
            print(f"{'-'*12} {'-'*20} {'-'*10}")
            for r in recs:
                print(f"{r['movie_id']:<12} {r['genre']:<20} {r['serendipity_score']:>10.6f}")

    finally:
        # Limpiar datos de prueba
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "DELETE FROM user_interests WHERE id_usuario = %s AND source = 'test'",
            (user_id,),
        )
        conn.commit()
        cur.close()
        conn.close()
        print(f"\n{'='*60}")
        print("  Intereses de prueba eliminados.")
