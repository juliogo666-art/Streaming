"""
##############################################################################################
#
#  MODELO 6: NCF (Neural Collaborative Filtering)
#  ===============================================
#  Arquitectura de Deep Learning que fusiona dos ramas complementarias:
#
#  1. GMF (Generalized Matrix Factorization):
#     - Producto elemento a elemento de embeddings de usuario e item.
#     - Captura interacciones lineales directas (similar a SVD pero más flexible).
#
#  2. MLP (Multi-Layer Perceptron):
#     - Concatena embeddings y los pasa por capas densas con ReLU.
#     - Captura interacciones NO lineales de orden superior.
#
#  La capa de salida combina ambas ramas en un logit que indica la probabilidad
#  de que al usuario le "guste" el item (entrenado con BCE sobre pares +/-).
#
#  Referencia: He et al., 2017 – "Neural Collaborative Filtering"
#
#  Exportación: el modelo se exporta a ONNX para inferencia sin dependencia de PyTorch.
#
##############################################################################################
"""

import pandas as pd  # Manejo de tablas (DataFrames) para leer el CSV de ratings
import numpy as np  # Operaciones numéricas rápidas (arrays, random, etc.)
import os  # Acceso al sistema de archivos (rutas, tamaño de ficheros)
import sys  # Manipulación del PATH de Python para imports relativos
import time  # Medición de tiempos de entrenamiento (perf_counter)
import json  # Serializar/deserializar diccionarios a archivos JSON
from typing import Dict, List, Tuple  # Anotaciones de tipo para funciones y variables

import torch  # Framework de Deep Learning (tensores, autograd, GPU)
import torch.nn as nn  # Módulos de redes neuronales (Linear, Embedding, etc.)
import torch.optim as optim  # Optimizadores de gradiente (Adam, SGD, etc.)
from torch.utils.data import (
    DataLoader,
    Dataset,
)  # DataLoader=iterador de lotes; Dataset=clase base de datos

# Sube 3 niveles (jj → models → src → raíz) para poder hacer import de src.utils
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)
from src.utils.registrar_metricas import (
    registrar_metricas,
)  # Función que guarda métricas en CSV histórico

##############################################################################################
#  CONFIGURACIÓN GLOBAL
##############################################################################################

# ── Ruta de entrada: CSV con columnas [userId, tmdb_id, rating] ──
RUTA_RATINGS = "src/data/ready/ratings_finales_ia.csv"

# ── Rutas de salida: artefactos que consumirá la API en producción ──
RUTA_MODELO_ONNX = (
    "artifacts/exports/modelo_6_ncf.onnx"  # Red neuronal serializada en formato ONNX
)
RUTA_USER2IDX = "artifacts/mappings/ncf_user2idx.json"  # Diccionario {userId_original: índice_interno}
RUTA_ITEM2IDX = "artifacts/mappings/ncf_item2idx.json"  # Diccionario {tmdb_id_original: índice_interno}

# ── Hiperparámetros del modelo ──
EMB_DIM = 32  # Cada usuario/item se representa como un vector de 32 números
BATCH_SIZE = 4096  # Nº de tripletas (u, pos, neg) que la GPU procesa de golpe
EPOCHS = 5  # Nº de pasadas completas sobre todo el dataset
NEG_SAMPLES = 4  # Por cada rating real, se inventan 4 "falsos" para contrastar
LEARNING_RATE = 1e-3  # Velocidad a la que Adam ajusta los pesos (0.001)
MIN_RATINGS_USER = 100  # Un usuario necesita ≥100 ratings para no ser descartado
MIN_RATINGS_ITEM = 100  # Una película necesita ≥100 ratings para no ser descartada

# ── Detección automática de dispositivo (GPU > CPU) ──
try:
    import torch_directml  # type: ignore  # Backend para GPUs AMD/Intel en Windows

    DEVICE = torch_directml.device()  # Crea el dispositivo DirectML
    print("[NCF] torch-directml detectado. Usando GPU AMD/Intel via DirectML.")
    _BACKEND = (
        f"DirectML (GPU) — {torch_directml.device_name(0)}"  # Nombre legible de la GPU
    )
except ImportError:  # Si no hay DirectML instalado...
    if torch.cuda.is_available():  # ...comprueba si hay GPU NVIDIA con CUDA
        DEVICE = torch.device("cuda")  # Usa la GPU NVIDIA
        _BACKEND = f"CUDA (GPU) — {torch.cuda.get_device_name(0)}"
    else:  # Sin GPU disponible
        DEVICE = torch.device("cpu")  # Fallback a CPU (más lento pero funcional)
        _BACKEND = "CPU"

print(f"[NCF] Backend de cómputo: {_BACKEND}")  # Log del dispositivo elegido


##############################################################################################
#  PASO 1: CARGA Y FILTRADO K-CORE
##############################################################################################


def cargar_y_filtrar(ruta: str) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    """Lee el CSV de ratings y aplica filtrado K-Core iterativo hasta convergencia.
    Retorna: (DataFrame filtrado y reindexado, mapeo_usuarios, mapeo_items)."""
    print("=" * 70)
    print("  MODELO 6: NCF — Carga y Filtrado K-Core")
    print("=" * 70)

    print(f"\n  Leyendo {ruta}...")
    # Lee el CSV forzando tipos ligeros (int32/float32) para ahorrar ~50% de RAM vs int64/float64
    df = pd.read_csv(
        ruta, dtype={"userId": "int32", "tmdb_id": "int32", "rating": "float32"}
    )
    print(f"  -> Filas originales: {len(df):,}")

    # ── Filtrado iterativo K-Core hasta convergencia ──
    # Elimina usuarios con <100 ratings y películas con <100 ratings.
    # Es iterativo porque al quitar un usuario, sus películas pierden ratings
    # y podrían caer bajo el umbral, generando una reacción en cadena.
    prev_len = -1  # Longitud previa (-1 fuerza al menos 1 iteración)
    iteration = 0  # Contador de pasadas K-Core
    while len(df) != prev_len:  # Repite hasta que no se elimine nada más (convergencia)
        prev_len = len(df)  # Guarda el tamaño actual para comparar al final del ciclo
        iteration += 1
        user_counts = df["userId"].value_counts()  # Cuenta ratings por usuario
        # Filtra: solo mantiene filas cuyo userId tiene ≥100 ratings
        df = df[df["userId"].isin(user_counts[user_counts >= MIN_RATINGS_USER].index)]
        item_counts = df["tmdb_id"].value_counts()  # Cuenta ratings por película
        # Filtra: solo mantiene filas cuyo tmdb_id tiene ≥100 ratings
        df = df[df["tmdb_id"].isin(item_counts[item_counts >= MIN_RATINGS_ITEM].index)]

    print(
        f"  -> Sub-muestra K-Core ({MIN_RATINGS_USER}/{MIN_RATINGS_ITEM}): "
        f"{len(df):,} filas | {df['userId'].nunique():,} usuarios | "
        f"{df['tmdb_id'].nunique():,} items ({iteration} iteraciones)"
    )

    # ── Reindexado: IDs originales (ej. 58423) → índices contiguos (0,1,2...) ──
    # Necesario porque nn.Embedding es una tabla indexada por posición [0, N)
    user2idx = {int(u): i for i, u in enumerate(sorted(df["userId"].unique()))}
    item2idx = {int(it): i for i, it in enumerate(sorted(df["tmdb_id"].unique()))}

    # Sustituye los IDs originales por los índices nuevos en el DataFrame
    df["userId"] = df["userId"].map(user2idx).astype("int32")
    df["tmdb_id"] = df["tmdb_id"].map(item2idx).astype("int32")

    # Resetea el índice del DataFrame y devuelve todo
    return df.reset_index(drop=True), user2idx, item2idx


##############################################################################################
#  PASO 2: DATASET PAIRWISE
##############################################################################################


class PairwiseDataset(Dataset):
    """
    Dataset que genera tripletas (usuario, item_positivo, item_negativo).
    Por cada interacción real (positiva), muestrea negativos aleatorios.
    No construye matrices densas — solo guarda listas de positivos por usuario.
    """

    def __init__(self, df: pd.DataFrame, n_items: int, neg_samples: int = NEG_SAMPLES):
        self.n_items = n_items  # Total de items en el catálogo (para generar negativos)
        self.neg_samples = (
            neg_samples  # Cuántos negativos generar por cada positivo (4)
        )
        # Agrupa: para cada usuario, un SET con los items que SI valoró
        self.user_positives = df.groupby("userId")["tmdb_id"].apply(set).to_dict()
        # Convierte las interacciones a un array numpy --> peliculas que vio el usuario
        self.interactions = df[["userId", "tmdb_id"]].values.astype(np.int64)

    def __len__(self) -> int:
        # Tamaño total = interacciones_reales × 4 negativos cada una
        return len(self.interactions) * self.neg_samples

    def __getitem__(self, idx: int):
        # Dado un índice global, calcula qué interacción real le corresponde
        pos_idx = (
            idx // self.neg_samples
        )  # División entera: idx 0,1,2,3 → interacción 0
        u, i_pos = self.interactions[pos_idx]  # Desempaqueta usuario e item positivo
        positives = self.user_positives[u]  # Set de items que este usuario ya valoró

        # Muestreo negativo por rechazo: elige un item al azar del catálogo
        i_neg = np.random.randint(self.n_items)  # Item aleatorio en [0, n_items)
        while i_neg in positives:  # Si el usuario ya lo vio, repetir
            i_neg = np.random.randint(
                self.n_items
            )  # (casi siempre acierta al 1er intento)

        # Devuelve la tripleta como tensores de PyTorch (enteros de 64 bits)
        return (
            torch.tensor(u, dtype=torch.long),  # ID del usuario
            torch.tensor(i_pos, dtype=torch.long),  # ID del item que SÍ vio (positivo)
            torch.tensor(i_neg, dtype=torch.long),  # ID del item que NO vio (negativo)
        )


##############################################################################################
#  PASO 3: ARQUITECTURA NCF (GMF + MLP)
##############################################################################################


class NCF(nn.Module):
    """
    Neural Collaborative Filtering con dos ramas fusionadas:
      - GMF: producto elemento a elemento de embeddings → captura interacciones lineales.
      - MLP: concatena embeddings + capas densas → captura interacciones no lineales.
    La capa de fusión combina ambas en un logit de relevancia.
    """

    def __init__(
        self,
        n_users: int,  # Total de usuarios únicos tras K-Core
        n_items: int,  # Total de items únicos tras K-Core
        emb_dim: int = EMB_DIM,  # Dimensión de cada vector embedding (32)
        mlp_layers: List[int] | None = None,  # Tamaños de las capas ocultas del MLP
    ):
        super().__init__()  # Inicializa la clase padre nn.Module

        # Si no se pasan capas MLP, usa: [64, 32, 16] (con emb_dim=32)
        if mlp_layers is None:
            mlp_layers = [emb_dim * 2, emb_dim, emb_dim // 2]

        # ── 4 tablas de embeddings separadas (2 por rama) ──
        # Cada una es una matriz de tamaño [n_entidades × 32]
        # Se usan tablas distintas para que cada rama aprenda representaciones especializadas
        self.gmf_user = nn.Embedding(n_users, emb_dim)  # Rama GMF: embedding de usuario
        self.gmf_item = nn.Embedding(n_items, emb_dim)  # Rama GMF: embedding de item
        self.mlp_user = nn.Embedding(n_users, emb_dim)  # Rama MLP: embedding de usuario
        self.mlp_item = nn.Embedding(n_items, emb_dim)  # Rama MLP: embedding de item

        # ── Bloque MLP: construye las capas densas dinámicamente ──
        mlp_modules: List[nn.Module] = []  # Lista temporal de capas
        in_size = emb_dim * 2  # Entrada inicial = 64 (32 user + 32 item concatenados)
        for out_size in mlp_layers:  # Itera: [64, 32, 16]
            mlp_modules += [
                nn.Linear(in_size, out_size),  # Capa densa: multiplica pesos + bias
                nn.ReLU(),  # Activación: max(0, x) → introduce no-linealidad
                nn.Dropout(0.1),  # Apaga 10% de neuronas al azar → anti-sobreajuste
            ]
            in_size = out_size  # La salida de esta capa es la entrada de la siguiente
        # Resultado: Linear(64→64) → ReLU → Dropout → Linear(64→32) → ReLU → Dropout → Linear(32→16) → ReLU → Dropout
        self.mlp = nn.Sequential(
            *mlp_modules
        )  # Encadena todas las capas en un solo módulo

        # ── Capa de fusión final ──
        # Recibe [gmf_out(32) || mlp_out(16)] = 48 dims → produce 1 logit escalar
        self.output_layer = nn.Linear(emb_dim + mlp_layers[-1], 1)  # Linear(48, 1)

        # ── Inicialización de pesos ──
        # Valores pequeños (std=0.01) para que el entrenamiento arranque estable
        for emb in (self.gmf_user, self.gmf_item, self.mlp_user, self.mlp_item):
            nn.init.normal_(emb.weight, std=0.01)  # Distribución normal con media=0

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Calcula el logit de relevancia para pares (usuario, item)."""
        # Rama GMF: producto elemento-a-elemento → [batch, 32]
        gmf_out = self.gmf_user(users) * self.gmf_item(items)
        # Rama MLP: concatena embeddings → [batch, 64] → pasa por capas densas → [batch, 16]
        mlp_in = torch.cat([self.mlp_user(users), self.mlp_item(items)], dim=1)
        mlp_out = self.mlp(mlp_in)
        # Fusión: concatena ambas ramas → [batch, 48] → capa lineal → [batch, 1] → squeeze → [batch]
        fused = torch.cat([gmf_out, mlp_out], dim=1)
        return self.output_layer(fused).squeeze(
            1
        )  # squeeze quita la dim extra → escalar por muestra

    def forward(self, users, pos_items, neg_items):
        """Función de entrenamiento: calcula BCE Loss sobre pares positivo/negativo."""
        pos_logits = self.score(
            users, pos_items
        )  # Scores para items que el usuario SÍ vio
        neg_logits = self.score(
            users, neg_items
        )  # Scores para items que el usuario NO vio
        # Concatena todos los scores y crea etiquetas: 1 para positivos, 0 para negativos
        logits = torch.cat([pos_logits, neg_logits])
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        # BCE con logits: penaliza si da score alto a negativos o bajo a positivos
        return nn.functional.binary_cross_entropy_with_logits(logits, labels)

    def predict(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Interfaz de inferencia (ONNX): solo 2 args → devuelve scores de relevancia."""
        return self.score(users, items)  # Reutiliza score() sin calcular loss


##############################################################################################
#  PASO 4: ENTRENAMIENTO
##############################################################################################


def entrenar(model, train_loader, optimizer, epochs):
    """Bucle de entrenamiento con gradient clipping."""
    model.train()  # Activa modo entrenamiento: Dropout ON, BatchNorm actualiza estadísticas
    print(f"\n  Iniciando entrenamiento por {epochs} épocas...")

    for epoch in range(1, epochs + 1):  # Épocas 1 a 5
        epoch_loss = 0.0  # Acumulador de loss para esta época
        t0 = time.perf_counter()  # Cronómetro de alta precisión

        for batch in train_loader:  # Itera sobre lotes de 4096 tripletas
            # Mueve los 3 tensores (users, pos, neg) del CPU a la GPU
            users, pos_items, neg_items = [t.to(DEVICE) for t in batch]
            optimizer.zero_grad()  # Limpia los gradientes del lote anterior
            loss = model(users, pos_items, neg_items)  # Forward pass → calcula BCE loss
            loss.backward()  # Backpropagation → calcula gradientes de cada peso
            # Recorta gradientes si su norma > 1.0 → previene "gradient explosion"
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # Adam actualiza los pesos usando los gradientes
            epoch_loss += loss.item()  # Acumula el loss escalar (sin grafo de cómputo)

        elapsed = time.perf_counter() - t0  # Tiempo de la época en segundos
        avg_loss = epoch_loss / len(train_loader)  # Loss medio por lote
        print(
            f"  Época {epoch}/{epochs} | Loss: {avg_loss:.5f} | Tiempo: {elapsed:.1f}s"
        )

    print("  Entrenamiento completado.\n")


##############################################################################################
#  PASO 5: EXPORTACIÓN A PRODUCCIÓN
##############################################################################################


def exportar_onnx(model):
    """
    Exporta el modelo NCF a formato ONNX.
    Usa un wrapper porque model.forward() acepta 3 args (entrenamiento)
    pero para inferencia necesitamos model.predict() con 2 args.
    """

    # Clase auxiliar que redirige forward() a predict() (2 argumentos)
    class _InferenceWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m  # Guarda referencia al modelo NCF real

        def forward(self, users, items):
            # ONNX traza esta función → solo necesita users e items
            return self.m.predict(users, items)

    # ONNX export requiere que el modelo esté en CPU
    original_device = next(
        model.parameters()
    ).device  # Guarda el dispositivo actual (GPU/CPU)
    cpu_model = model.to(torch.device("cpu"))  # Mueve todos los pesos a CPU
    cpu_model.eval()  # Desactiva Dropout y BatchNorm de entrenamiento

    wrapper = _InferenceWrapper(
        cpu_model
    )  # Envuelve el modelo con la interfaz de 2 args
    wrapper.eval()  # También en modo evaluación

    # Tensores dummy: ONNX necesita una entrada de ejemplo para trazar el grafo computacional
    dummy_users = torch.zeros(1, dtype=torch.long)  # Un usuario de ejemplo (ID=0)
    dummy_items = torch.zeros(1, dtype=torch.long)  # Un item de ejemplo (ID=0)

    print(f"  Exportando modelo a ONNX: {RUTA_MODELO_ONNX}")
    torch.onnx.export(
        wrapper,  # Módulo a exportar
        args=(dummy_users, dummy_items),  # Entradas de ejemplo para el trazado
        f=RUTA_MODELO_ONNX,  # Ruta del archivo .onnx de salida
        input_names=["user_ids", "item_ids"],  # Nombres legibles para las entradas
        output_names=["scores"],  # Nombre legible para la salida
        dynamic_axes={  # Permite batch_size variable en inferencia
            "user_ids": {0: "batch_size"},  # Eje 0 de user_ids es dinámico
            "item_ids": {0: "batch_size"},  # Eje 0 de item_ids es dinámico
            "scores": {0: "batch_size"},  # Eje 0 de scores es dinámico
        },
        opset_version=17,  # Versión del estándar ONNX (17 = estable y reciente)
        export_params=True,  # Incluir los pesos entrenados dentro del .onnx
        do_constant_folding=True,  # Optimización: pre-calcula expresiones constantes
    )

    model.to(original_device)  # Devuelve el modelo al dispositivo original (GPU)
    tamano_mb = os.path.getsize(RUTA_MODELO_ONNX) / (1024 * 1024)  # Tamaño en MB
    print(f"  -> Modelo ONNX guardado: {RUTA_MODELO_ONNX} ({tamano_mb:.1f} MB)")


def exportar_mappings(user2idx, item2idx):
    """Guarda los diccionarios de mapeo de IDs en formato JSON para la API."""
    # Guarda user2idx: la API lo usa para traducir userId real → índice interno del modelo
    # Las claves se convierten a str porque JSON no soporta claves enteras
    with open(RUTA_USER2IDX, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in user2idx.items()}, f)
    print(f"  -> user2idx.json guardado: {len(user2idx)} usuarios")

    # Guarda item2idx: la API lo usa para traducir tmdb_id real → índice interno del modelo
    with open(RUTA_ITEM2IDX, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in item2idx.items()}, f)
    print(f"  -> item2idx.json guardado: {len(item2idx)} items")


##############################################################################################
#  EJECUCIÓN PRINCIPAL
##############################################################################################

if __name__ == "__main__":
    # ======================================================================
    # PASO 1: Carga y filtrado K-Core
    # ======================================================================
    # Lee el CSV, filtra usuarios/items con <100 ratings, y reindexda IDs a [0, N)
    df, user2idx, item2idx = cargar_y_filtrar(RUTA_RATINGS)
    n_users = len(user2idx)  # Nº total de usuarios supervivientes al filtro K-Core
    n_items = len(item2idx)  # Nº total de items supervivientes al filtro K-Core
    print(f"  Vocabulario: {n_users} usuarios | {n_items} items")

    # ======================================================================
    # PASO 2: Crear Dataset y DataLoader
    # ======================================================================
    # PairwiseDataset genera tripletas (user, pos_item, neg_item) bajo demanda
    dataset = PairwiseDataset(df, n_items=n_items, neg_samples=NEG_SAMPLES)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,  # 4096 tripletas por lote
        shuffle=True,  # Aleatoriza el orden cada época (mejor generalización)
        num_workers=2,  # 2 hilos paralelos precargan datos mientras la GPU entrena
        pin_memory=True,  # Fija memoria en RAM → transfiere a GPU más rápido
    )
    print(
        f"  Dataset: {len(df):,} interacciones x {NEG_SAMPLES} negativos = {len(dataset):,} muestras"
    )

    # ======================================================================
    # PASO 3: Instanciar modelo y optimizador
    # ======================================================================
    # Crea la red NCF con los embeddings del tamaño correcto y la mueve a GPU
    model = NCF(n_users=n_users, n_items=n_items, emb_dim=EMB_DIM).to(DEVICE)
    # Adam: optimizador adaptativo que ajusta el lr por parámetro automáticamente
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Cuenta total de números entrenables en la red (pesos + biases)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  NCF instanciado: {total_params:,} parámetros entrenables")
    print(f"  Backend: {_BACKEND}")

    # ======================================================================
    # PASO 4: Entrenamiento
    # ======================================================================
    t_start = time.perf_counter()  # Inicia cronómetro global
    entrenar(model, train_loader, optimizer, epochs=EPOCHS)  # 5 épocas de BCE pairwise
    t_total = time.perf_counter() - t_start  # Tiempo total en segundos
    print(f"  Tiempo total: {t_total:.1f}s")

    # ======================================================================
    # PASO 5: Exportación a producción
    # ======================================================================
    print("\n" + "=" * 70)
    print("  MODELO 6: NCF — Exportación a Producción")
    print("=" * 70)
    exportar_onnx(model)
    exportar_mappings(user2idx, item2idx)

    # ======================================================================
    # PASO 6: Registrar métricas en el histórico
    # ======================================================================
    # Guarda hiperparámetros + tiempo en historial_metricas.csv para auditoría
    registrar_metricas(
        modelo="NCF",  # Nombre del modelo
        hiperparams={  # Diccionario con la configuración usada
            "embedding_dim": EMB_DIM,
            "n_epocas": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "neg_samples": NEG_SAMPLES,
            "min_ratings_user": MIN_RATINGS_USER,
            "min_ratings_item": MIN_RATINGS_ITEM,
        },
        metricas={},  # Vacío: NCF es ranking, MAE/RMSE no aplican
        dataset_size=len(df),  # Nº de filas tras K-Core
        train_time_s=t_total,  # Segundos de entrenamiento
        notas="Modelo de ranking (BCE pairwise). MAE/RMSE no aplican directamente.",
    )

    print("\n  ¡Modelo NCF exportado a ONNX con éxito!")
    print("=" * 70)
