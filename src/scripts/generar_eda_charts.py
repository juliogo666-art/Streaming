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

import json
import os
import sys

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
    sns.countplot(x="rating", data=df_ratings, hue="rating", palette="coolwarm", legend=False, ax=ax)
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
    guardar_metricas(metricas)

    print("\n[EDA] Todos los charts generados correctamente.")
    print(f"       Carpeta: {os.path.abspath(OUT_DIR)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
