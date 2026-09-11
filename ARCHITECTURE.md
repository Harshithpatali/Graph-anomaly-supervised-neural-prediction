# Project Architecture

```
Fraud Ecommerce Dataset
        ↓
Data Quality & Temporal Audit          ← notebook 01
        ↓
Transaction → User / Device / IP Graph ← notebook 03
        ↓
Graph + Behavioural Features           ← notebook 04
        ↓
Temporal Dynamics                      ← notebooks 02, 08
        ↓
Statistical Anomaly Detection          ← notebook 05
        ↓
Isolation Forest / LOF                 ← notebook 06
        ↓
Graph Embeddings (lightweight)         ← notebook 07
        ↓
Anomaly Ranking / Hybrid               ← notebook 09
        ↓
Fraud Labels → Validation              ← notebooks 06-11
        ↓
Explainable Anomaly Investigation      ← notebook 12
        ↓
Streamlit Dashboard                    ← app/app.py
```

## Design Rules

1. **Fraud label is validation ground truth only** – never enters `build_feature_matrix`.
2. **Chronological train / validation / test** – test remains locked until notebook 11.
3. **Complex graph ML only if justified** – we start with degree & shared-entity features; full node2vec is optional.
4. **Deployment consumes frozen artefacts** produced after final evaluation.
5. **Every notebook is self-contained** with clear markdown objectives, code, and interpretation cells.

## Key Modules (`src/`)

| Module | Responsibility |
|--------|----------------|
| `data_utils` | Load, clean, IP→country, chronological split, parquet I/O |
| `temporal_features` | Account age, hour, velocity, instant-purchase flags |
| `graph_utils` | Multipartite graph construction, degree & sharing features |
| `graph_features` | End-to-end feature matrix builder |
| `anomaly` | IsolationForest / LOF / EllipticEnvelope wrappers + hybrid rank score |
| `evaluation` | ROC-AUC, PR-AUC, Precision@k, enrichment factor |
| `final_pipeline` | Inference entry-point for the Streamlit app |
| `embeddings` | Optional node2vec / simple log-degree surrogates |
