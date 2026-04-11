# artifacts/

Directorio centralizado para todos los modelos de IA entrenados del proyecto.

## Estructura

```
artifacts/
├── weights/       # Modelos clásicos serializados (.pkl, .joblib)
│                  # SVD, KNN, TF-IDF, Implicit BPR
│
├── exports/       # Modelos exportados a ONNX para inferencia sin PyTorch
│                  # Wide&Deep, NCF, Two-Towers
│
├── checkpoints/   # Pesos de PyTorch (.pth) para reentrenamiento
│                  # Wide&Deep, Two-Towers
│
└── mappings/      # Mapeos de IDs internos <-> IDs reales (.json, .pkl)
                   # wnd_mappings, ncf_user2idx, ncf_item2idx, etc.
```

## Convenciones de Nombres

- Nombre base del modelo: `modelo_N_nombre.extension`
- Si se reentrena con parámetros distintos, añadir el cambio al final:
  - `modelo_3_wnd_r_100.onnx` → Reentrenado con min_ratings=100
  - `modelo_6_ncf_emb64.onnx` → Reentrenado con embedding_dim=64

## Archivos Actuales

### weights/
| Archivo | Modelo | Formato |
|---------|--------|---------|
| `modelo_1_SVD.pkl` / `.joblib` | SVD (Surprise) | Pickle / Joblib |
| `modelo_2_knn_cs.pkl` / `.joblib` | KNN Coseno (Surprise) | Pickle / Joblib |
| `modelo_2.5_knn_msd.pkl` / `.joblib` | KNN MSD (Surprise) | Pickle / Joblib |
| `modelo_4_tfidf.pkl` / `.joblib` | TF-IDF Vectorizer | Pickle / Joblib |
| `modelo_4_matriz.pkl` / `.joblib` | Matriz TF-IDF | Pickle / Joblib |
| `modelo_4_indices.pkl` / `.joblib` | Mapeo índices TF-IDF | Pickle / Joblib |
| `modelo_5_implicit.pkl` | BPR (librería implicit) | Pickle |
| `modelo_5_implicit_dataset.pkl` | Dataset del BPR | Pickle |

### exports/
| Archivo | Modelo | Formato |
|---------|--------|---------|
| `modelo_3_wnd.onnx` (+`.data`) | Wide & Deep | ONNX |
| `modelo_6_ncf.onnx` (+`.data`) | NCF-Lite (jj) | ONNX |
| `modelo_7_twotowers.onnx` (+`.data`) | Two-Towers | ONNX |
| `nil_ncf_model.onnx` | NCF-Lite (nil) | ONNX |

### checkpoints/
| Archivo | Modelo | Formato |
|---------|--------|---------|
| `modelo_3_wnd.pth` | Wide & Deep | PyTorch |
| `modelo_7_twotowers.pth` | Two-Towers | PyTorch |

### mappings/
| Archivo | Contenido | Formato |
|---------|-----------|---------|
| `wnd_mappings.pkl` | user2idx + movie2idx para W&D | Pickle |
| `twotowers_mappings.pkl` | user2idx + item2idx para TT | Pickle |
| `ncf_user2idx.json` | userId → idx para NCF (jj) | JSON |
| `ncf_item2idx.json` | tmdb_id → idx para NCF (jj) | JSON |