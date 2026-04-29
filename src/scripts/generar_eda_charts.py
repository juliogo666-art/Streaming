"""
generar_eda_charts.py
=====================
Script OFFLINE que genera los gráficos del EDA como imágenes estáticas PNG.
Se ejecuta una vez (o cuando cambien los datos) para poblar static/eda/.
El frontend de Administrador los carga directamente con st.image(), eliminando
la carga de pandas + seaborn en tiempo real durante la navegación.

Uso (desde la raíz del proyecto):
    python -m src.scripts.generar_eda_charts
    -- o bien --
    uv run python -m src.scripts.generar_eda_charts
"""

import ast
import json
import os
import sys
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Forzamos backend sin ventana (compatible con servidores sin pantalla)
matplotlib.use("Agg")

# --- Rutas ---
MOVIES_PATH = "src/data/ready/dataset_final_movies.csv"
RATINGS_PATH = "src/data/ready/ratings_finales_ia.csv"
OUT_DIR = "static/eda"

# Umbrales del Smart Selector (sincronizados con src/config/settings.py)
UMBRAL_COLD_START = 10
UMBRAL_AVANZADO = 100

# Estilo visual: oscuro para coherencia con el panel de administrador
sns.set_theme(style="darkgrid")
plt.rcParams.update(
    {
        "figure.facecolor": "#001220",
        "axes.facecolor": "#001f3f",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "text.color": "#e0e0e0",
        "grid.color": "#1a3a5f",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
    }
)
TITLE_COLOR = "#B8860B"  # Dorado premium, igual que el panel admin


def _asegurar_directorio(ruta: str) -> None:
    """Crea el directorio de salida si no existe."""
    os.makedirs(ruta, exist_ok=True)
    print(f"[EDA] Directorio de salida: {os.path.abspath(ruta)}")


def _cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga y valida los CSVs necesarios."""
    for path in (MOVIES_PATH, RATINGS_PATH):
        if not os.path.exists(path):
            print(f"[ERROR] No se encontró: {path}")
            print("  Ejecuta el script desde la raíz del proyecto.")
            sys.exit(1)

    print("[EDA] Cargando dataset de películas...")
    df_movies = pd.read_csv(MOVIES_PATH, on_bad_lines="skip", engine="python")
    print(f"       {len(df_movies):,} películas cargadas.")

    print("[EDA] Cargando ratings (puede tardar ~30-40s)...")
    df_ratings = pd.read_csv(RATINGS_PATH, on_bad_lines="skip", engine="python")
    print(f"       {len(df_ratings):,} ratings cargados.")

    return df_movies, df_ratings


def generar_top10_peliculas(df_movies: pd.DataFrame) -> None:
    """Gráfico 1 — Top 10 películas mejores valoradas (mín. 500 votos)."""
    print("[EDA] Generando gráfico 1: Top 10 películas...")

    top_movies = (
        df_movies[df_movies["vote_count"] > 500]
        .sort_values(by="vote_average", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    bar = sns.barplot(
        x="vote_average",
        y="titulo",
        hue="titulo",
        data=top_movies,
        palette="YlOrBr",
        legend=False,
        ax=ax,
    )

    # Etiquetas con la puntuación
    for container in bar.containers:
        ax.bar_label(container, fmt="%.2f", padding=4, color="#B8860B", fontsize=10)

    ax.set_title(
        "Top 10 Películas · Mejor Valoración (Mín. 500 votos)",
        color=TITLE_COLOR,
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Nota Media (Promedio de votos)", fontsize=11)
    ax.set_ylabel("Título", fontsize=11)
    ax.set_xlim(0, 10)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "top10_peliculas.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"       Guardado -> {out_path}")


def generar_distribucion_usuarios(df_ratings: pd.DataFrame) -> dict:
    """
    Gráfico 2 — Distribución de valoraciones por usuario.
    Devuelve las métricas calculadas para guardarlas en JSON.
    """
    print("[EDA] Generando gráfico 2: Distribución de valoraciones por usuario...")

    user_counts = df_ratings["userId"].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(user_counts, bins=100, kde=True, color="#4a9eff", ax=ax)
    ax.set_title(
        "Distribución de Valoraciones por Usuario",
        color=TITLE_COLOR,
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Nº de películas valoradas por el usuario", fontsize=11)
    ax.set_ylabel("Cantidad de Usuarios", fontsize=11)
    ax.set_xlim(0, 500)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "distribucion_valoraciones_usuario.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"       Guardado -> {out_path}")

    # Métricas calculadas de una vez (no volverán a computarse en el frontend)
    metricas = {
        "media_valoraciones_usuario": round(float(user_counts.mean()), 2),
        "usuarios_menos_20_valoraciones": int((user_counts < 20).sum()),
        "total_usuarios": int(len(user_counts)),
    }
    return metricas


def generar_distribucion_puntuaciones(df_ratings: pd.DataFrame) -> None:
    """Gráfico 3 — Distribución general de ratings (estrellas)."""
    print("[EDA] Generando gráfico 3: Distribución de puntuaciones...")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(
        x="rating",
        data=df_ratings,
        hue="rating",
        palette="coolwarm",
        legend=False,
        ax=ax,
    )
    ax.set_title(
        "Distribución General de Puntuaciones (Estrellas)",
        color=TITLE_COLOR,
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Puntuación (Rating)", fontsize=11)
    ax.set_ylabel("Cantidad de Votos", fontsize=11)

    # Etiquetas encima de cada barra
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height()):,}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#e0e0e0",
        )

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, "distribucion_puntuaciones.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"       Guardado -> {out_path}")


def generar_smart_selector_grupos(df_ratings: pd.DataFrame) -> dict:
    """
    Gráfico 4 — Distribución de usuarios por grupo del Smart Selector.
    Muestra cuántos usuarios caen en cada nivel de triaje (Cold Start / Intermedio / Experto).
    Devuelve las métricas para incluirlas en el JSON.
    """
    print("[EDA] Generando gráfico 4: Smart Selector — Grupos de usuarios...")

    user_counts = df_ratings["userId"].value_counts()

    cold_start = int((user_counts <= UMBRAL_COLD_START).sum())
    intermedio = int(
        ((user_counts > UMBRAL_COLD_START) & (user_counts < UMBRAL_AVANZADO)).sum()
    )
    experto = int((user_counts >= UMBRAL_AVANZADO).sum())
    total = cold_start + intermedio + experto

    grupos = [
        f"Cold Start\n(0-{UMBRAL_COLD_START})",
        f"Intermedio\n({UMBRAL_COLD_START + 1}-{UMBRAL_AVANZADO - 1})",
        f"Experto\n({UMBRAL_AVANZADO}+)",
    ]
    valores = [cold_start, intermedio, experto]
    colores = ["#4a9eff", "#B8860B", "#ff6b35"]
    iconos = ["Content-Based\nTF-IDF", "SVD / KNN\nBPR", "NCF / Wide&Deep\nTwo-Towers"]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(
        grupos, valores, color=colores, edgecolor="#0a1929", linewidth=2, width=0.6
    )

    for bar_item, val, icono in zip(bars, valores, iconos):
        pct = val / total * 100
        ax.text(
            bar_item.get_x() + bar_item.get_width() / 2,
            bar_item.get_height() + total * 0.015,
            f"{val:,} usuarios\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#e0e0e0",
        )
        # Modelo asignado dentro de la barra
        ax.text(
            bar_item.get_x() + bar_item.get_width() / 2,
            bar_item.get_height() / 2,
            icono,
            ha="center",
            va="center",
            fontsize=9,
            color="#001220",
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.15,
                edgecolor="none",
            ),
        )

    ax.set_title(
        "Smart Selector — Distribución de Usuarios por Grupo de Triaje",
        color=TITLE_COLOR,
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylabel("Número de Usuarios", fontsize=11)
    ax.set_xlabel("Grupo del Smart Selector", fontsize=11)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "smart_selector_grupos.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"       Guardado -> {out_path}")

    return {
        "smart_cold_start": cold_start,
        "smart_intermedio": intermedio,
        "smart_experto": experto,
    }


def generar_sparsity_heatmap(df_ratings: pd.DataFrame) -> dict:
    """
    Gráfico 5 — Mapa de calor de la matriz usuario x película (muestra).
    Visualiza la sparsity (~99%+ celdas vacías) que los modelos deben predecir.
    """
    print("[EDA] Generando gráfico 5: Mapa de Sparsity...")

    # Muestra de 80 usuarios más activos × 80 películas más valoradas
    top_users = df_ratings["userId"].value_counts().head(80).index
    top_movies = df_ratings["tmdb_id"].value_counts().head(80).index
    sample = df_ratings[
        df_ratings["userId"].isin(top_users) & df_ratings["tmdb_id"].isin(top_movies)
    ]
    matrix = sample.pivot_table(index="userId", columns="tmdb_id", values="rating")

    # Calcular sparsity global real
    n_users = df_ratings["userId"].nunique()
    n_items = df_ratings["tmdb_id"].nunique()
    n_ratings = len(df_ratings)
    posibles = n_users * n_items
    sparsity = (1 - n_ratings / posibles) * 100 if posibles > 0 else 0

    fig, ax = plt.subplots(figsize=(14, 9))
    sns.heatmap(
        matrix.notna().astype(int),
        cmap=["#001f3f", "#B8860B"],
        cbar_kws={"label": "Valorado (1) / Vacío (0)", "shrink": 0.6},
        linewidths=0.05,
        linecolor="#0a1929",
        ax=ax,
    )
    ax.set_title(
        f"Sparsity de la Matriz Usuario × Película — {sparsity:.2f}% vacío\n"
        f"(muestra: {len(matrix)} usuarios × {len(matrix.columns)} películas)",
        color=TITLE_COLOR,
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Películas (tmdb_id)", fontsize=10)
    ax.set_ylabel("Usuarios (userId)", fontsize=10)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "sparsity_heatmap.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"       Guardado -> {out_path}")

    return {
        "sparsity_pct": round(sparsity, 2),
        "total_usuarios_unicos": n_users,
        "total_peliculas_unicas": n_items,
        "total_interacciones": n_ratings,
    }


def generar_top_generos(df_movies: pd.DataFrame) -> None:
    """Gráfico 6 — Top 15 géneros más representados en el catálogo."""
    print("[EDA] Generando gráfico 6: Top géneros del catálogo...")

    all_genres: list[int] = []
    for ids in df_movies["genre_ids"].dropna():
        try:
            parsed = ast.literal_eval(str(ids))
            if isinstance(parsed, list):
                all_genres.extend(int(g) for g in parsed if str(g).isdigit())
        except (ValueError, SyntaxError):
            pass

    if not all_genres:
        print("       [WARN] No se pudieron parsear genre_ids, saltando gráfico.")
        return

    # Mapeo TMDB de genre_id → nombre legible
    TMDB_GENRES = {
        28: "Acción",
        12: "Aventura",
        16: "Animación",
        35: "Comedia",
        80: "Crimen",
        99: "Documental",
        18: "Drama",
        10751: "Familiar",
        14: "Fantasía",
        36: "Historia",
        27: "Terror",
        10402: "Música",
        9648: "Misterio",
        10749: "Romance",
        878: "Ciencia Ficción",
        10770: "TV Movie",
        53: "Suspense",
        10752: "Bélica",
        37: "Western",
    }

    genre_counts = Counter(all_genres)
    top15 = genre_counts.most_common(15)
    nombres = [TMDB_GENRES.get(gid, f"ID {gid}") for gid, _ in top15]
    cantidades = [c for _, c in top15]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(
        nombres[::-1], cantidades[::-1], color="#B8860B", edgecolor="#0a1929"
    )

    for bar_item in bars:
        ax.text(
            bar_item.get_width() + max(cantidades) * 0.01,
            bar_item.get_y() + bar_item.get_height() / 2,
            f"{int(bar_item.get_width()):,}",
            ha="left",
            va="center",
            fontsize=9,
            color="#e0e0e0",
        )

    ax.set_title(
        "Top 15 Géneros más Representados en el Catálogo",
        color=TITLE_COLOR,
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Número de Películas", fontsize=11)
    fig.tight_layout()

    out_path = os.path.join(OUT_DIR, "top_generos_catalogo.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"       Guardado -> {out_path}")


def guardar_metricas(metricas: dict) -> None:
    """Guarda las métricas en un JSON para que el frontend las lea sin cálculos."""
    out_path = os.path.join(OUT_DIR, "metricas.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metricas, f, ensure_ascii=False, indent=2)
    print(f"[EDA] Metricas guardadas -> {out_path}")
    print(f"      {metricas}")


def main() -> None:
    print("=" * 60)
    print("  SPIRE - Generador Offline de Charts EDA")
    print("=" * 60)

    _asegurar_directorio(OUT_DIR)
    df_movies, df_ratings = _cargar_datos()

    generar_top10_peliculas(df_movies)
    metricas = generar_distribucion_usuarios(df_ratings)
    generar_distribucion_puntuaciones(df_ratings)

    # Gráficos nuevos (Smart Selector + Sparsity + Géneros)
    metricas_smart = generar_smart_selector_grupos(df_ratings)
    metricas_sparsity = generar_sparsity_heatmap(df_ratings)
    generar_top_generos(df_movies)
    generar_diagrama_smart_selector()

    metricas.update(metricas_smart)
    metricas.update(metricas_sparsity)
    guardar_metricas(metricas)

    print("\n[EDA] Todos los charts generados correctamente.")
    print(f"       Carpeta: {os.path.abspath(OUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
