"""
Ingesta de usuarios "MovieLens" desde ratings_finales_ia.csv hacia MySQL.

Criterios (por defecto):
- Usuarios con al menos 1000 valoraciones en el CSV.
- Top 3 géneros por usuario usando solo ratings >= 4 (peso = suma de esos ratings).
- Cruce tmdb_id -> genre_id vía tabla content_genres en BD.

Uso (desde la raíz del repo):
  uv run python src/scripts/import_ml_users_from_ratings.py
  uv run python src/scripts/import_ml_users_from_ratings.py --dry-run

Documentación detallada: src/scripts/readme_import_ml_users_from_ratings.md
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import Dict, Iterable, List, Set, Tuple

import bcrypt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.api.database import get_db_connection  # noqa: E402

DEFAULT_RATINGS = os.path.join(ROOT, "src", "data", "ready", "ratings_finales_ia.csv")
MIN_RATINGS = 1000
RATING_MIN_FOR_GENRES = 4.0
TOP_GENRES = 3
SEED_PASSWORD = "recomiendame"


def _expected_username(user_id: int) -> str:
    return f"usuario_{user_id}"


def _expected_email(user_id: int) -> str:
    return f"usuario_{user_id}@gmail.com"


def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _pseudo_birth_and_sexo(user_id: int) -> Tuple[date, str]:
    rng = random.Random(user_id)
    start = date(1990, 1, 1)
    end = date(2005, 1, 1)
    days = (end - start).days
    birth = start + timedelta(days=rng.randint(0, max(days, 0)))
    sexo = rng.choice(["Hombre", "Mujer", "Otro"])
    return birth, sexo


def load_genres_and_content_genres() -> Tuple[Dict[int, str], Dict[int, List[int]]]:
    conn = get_db_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT id, name FROM genres")
        genre_rows = cur.fetchall()
        genre_names = {int(r["id"]): str(r["name"]) for r in genre_rows}

        cur.execute("SELECT content_id, genre_id FROM content_genres")
        cg_rows = cur.fetchall()
        tmdb_to_genres: Dict[int, List[int]] = defaultdict(list)
        for r in cg_rows:
            cid = int(r["content_id"])
            gid = int(r["genre_id"])
            tmdb_to_genres[cid].append(gid)
        cur.close()
        return dict(genre_names), {k: v for k, v in tmdb_to_genres.items()}
    finally:
        conn.close()


def count_ratings_per_user(
    ratings_path: str, chunk_size: int
) -> Counter:
    counts: Counter = Counter()
    for chunk in pd.read_csv(
        ratings_path,
        usecols=["userId"],
        chunksize=chunk_size,
        on_bad_lines="skip",
        engine="python",
    ):
        counts.update(chunk["userId"].value_counts().to_dict())
    return counts


def aggregate_genre_scores_for_users(
    ratings_path: str,
    user_ids: Set[int],
    tmdb_to_genres: Dict[int, List[int]],
    chunk_size: int,
) -> Dict[int, Counter]:
    scores: Dict[int, Counter] = defaultdict(Counter)
    for chunk in pd.read_csv(
        ratings_path,
        usecols=["userId", "tmdb_id", "rating"],
        chunksize=chunk_size,
        on_bad_lines="skip",
        engine="python",
    ):
        chunk = chunk.dropna(subset=["userId", "tmdb_id", "rating"])
        chunk["userId"] = chunk["userId"].astype(int)
        chunk["tmdb_id"] = chunk["tmdb_id"].astype(int)
        chunk["rating"] = chunk["rating"].astype(float)
        sub = chunk[
            chunk["userId"].isin(user_ids) & (chunk["rating"] >= RATING_MIN_FOR_GENRES)
        ]
        if sub.empty:
            continue
        u = sub["userId"].to_numpy(dtype=np.int64, copy=False)
        t = sub["tmdb_id"].to_numpy(dtype=np.int64, copy=False)
        r = sub["rating"].to_numpy(dtype=np.float64, copy=False)
        for i in range(len(sub)):
            uid_i = int(u[i])
            tid_i = int(t[i])
            rf = float(r[i])
            for gid in tmdb_to_genres.get(tid_i, ()):
                scores[uid_i][int(gid)] += rf
    return scores


def top_genre_ids(user_scores: Counter, k: int) -> List[int]:
    if not user_scores:
        return []
    pairs = user_scores.most_common(k * 2)
    out: List[int] = []
    seen: Set[int] = set()
    for gid, _ in pairs:
        if gid in seen:
            continue
        seen.add(gid)
        out.append(gid)
        if len(out) >= k:
            break
    return out


def upsert_user_and_interests(
    user_id: int,
    passwd_hash: str,
    birth: date,
    sexo: str,
    genre_ids: List[int],
) -> str:
    """
    Inserta usuario sintético o actualiza si ya existe con el mismo patrón usuario_<id>.
    Intereses: solo borra filas de ESE usuario y reinserta (no toca otros usuarios).
    """
    uname = _expected_username(user_id)
    email = _expected_email(user_id)

    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(
            "SELECT id_usuario, username FROM users WHERE id_usuario = %s",
            (user_id,),
        )
        row = cur.fetchone()

        if row is not None:
            if str(row["username"]) != uname:
                return "skip_conflict"

        if row is None:
            cur.execute(
                """
                INSERT INTO users
                    (id_usuario, username, email, passwd, fecha_nacimiento, sexo, role, fecha_registro)
                VALUES (%s, %s, %s, %s, %s, %s, 'user', NOW())
                """,
                (user_id, uname, email, passwd_hash, birth, sexo),
            )
        else:
            cur.execute(
                """
                UPDATE users
                SET username = %s,
                    email = %s,
                    passwd = %s,
                    fecha_nacimiento = %s,
                    sexo = %s,
                    role = 'user'
                WHERE id_usuario = %s
                """,
                (uname, email, passwd_hash, birth, sexo, user_id),
            )

        cur.execute("DELETE FROM user_interests WHERE id_usuario = %s", (user_id,))
        for gid in genre_ids:
            cur.execute(
                "INSERT INTO user_interests (id_usuario, genre_id, source) VALUES (%s, %s, %s)",
                (user_id, int(gid), "ml_inferred"),
            )

        conn.commit()
        return "ok"
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Importa usuarios MovieLens (>=1000 ratings) e intereses top-3 géneros."
    )
    parser.add_argument(
        "--ratings",
        default=DEFAULT_RATINGS,
        help="Ruta al CSV ratings_finales_ia.csv",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Filas por chunk al leer el CSV",
    )
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=MIN_RATINGS,
        help="Mínimo de valoraciones para incluir al usuario",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="No escribe en BD; solo muestra conteos",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    ratings_path = os.path.abspath(args.ratings)
    if not os.path.isfile(ratings_path):
        raise SystemExit(f"No existe el fichero: {ratings_path}")

    print("[1/4] Cargando géneros y content_genres desde BD...")
    genre_names, tmdb_to_genres = load_genres_and_content_genres()
    print(f"      genres: {len(genre_names)} | pares content_genres (tmdb distintos): {len(tmdb_to_genres)}")

    print("[2/4] Contando valoraciones por usuario (puede tardar)...")
    user_counts = count_ratings_per_user(ratings_path, args.chunk_size)
    important = {int(u) for u, c in user_counts.items() if c >= args.min_ratings}
    print(f"      Usuarios con >= {args.min_ratings} ratings: {len(important):,}")

    print("[3/4] Agregando scores de género (rating >= 4)...")
    genre_scores = aggregate_genre_scores_for_users(
        ratings_path, important, tmdb_to_genres, args.chunk_size
    )

    passwd_hash = _hash_password(SEED_PASSWORD)
    print("[4/4] Insertando / actualizando usuarios e intereses...")
    ok = skip_conflict = no_genres = 0
    for uid in sorted(important):
        birth, sexo = _pseudo_birth_and_sexo(uid)
        gids = top_genre_ids(genre_scores.get(uid, Counter()), TOP_GENRES)
        if not gids:
            no_genres += 1
        if args.dry_run:
            continue
        status = upsert_user_and_interests(uid, passwd_hash, birth, sexo, gids)
        if status == "ok":
            ok += 1
        elif status == "skip_conflict":
            skip_conflict += 1

    print("--- Resumen ---")
    print(f"Usuarios candidatos: {len(important):,}")
    if args.dry_run:
        print("(dry-run) no se escribió en BD.")
    else:
        print(f"Insertados/actualizados OK: {ok:,}")
        print(f"Omitidos (id_usuario ya ocupado por otro username): {skip_conflict:,}")
    print(f"Sin géneros deducibles (0 intereses tras import): {no_genres:,}")


if __name__ == "__main__":
    main()
