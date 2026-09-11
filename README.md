# Dynamic E-Commerce Graph Anomaly Detection & Fraud Intelligence

> A complete, reproducible data-science project that asks:  
> **Does modelling e-commerce activity as a dynamic multipartite graph (transactions ↔ users ↔ devices ↔ IPs) surface anomalous behaviour beyond transaction-level statistics?**

---

## Project Pipeline

```
Fraud Ecommerce Dataset
        ↓
Data Quality & Temporal Audit
        ↓
Transaction → User / Device / IP Graph
        ↓
Graph + Behavioural Features
        ↓
Temporal Dynamics
        ↓
Statistical Anomaly Detection
        ↓
Isolation Forest / LOF
        ↓
Graph Embeddings (lightweight)
        ↓
Anomaly Ranking
        ↓
Fraud Labels → Validation only
        ↓
Explainable Anomaly Investigation
        ↓
Interactive Streamlit Dashboard
```

## Scientific Principles

| Principle | Implementation |
|-----------|----------------|
| Label discipline | `class` is **never** a training feature; used only for post-hoc ranking metrics |
| Temporal integrity | Strict chronological train / val / test split on `purchase_time` |
| Locked test set | Final metrics reported only in notebook 11 |
| Complexity justification | Graph features are added only after tabular baselines are measured |
| Reproducibility | Fixed random seeds, frozen model artefacts, parquet intermediates |

## Repository Layout

```
├── data/
│   ├── raw/                  # original CSVs (do not modify)
│   └── processed/            # parquet intermediates
├── notebooks/                # 12 sequential, descriptive notebooks
├── src/                      # reusable library code
├── models/                   # frozen final model + config
├── app/                      # Streamlit dashboard
├── reports/                  # profiles, figures
├── tests/
├── ARCHITECTURE.md
├── data_dictionary.md
└── requirements.txt
```

## Quick Start

```bash
# 1. Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the analytical notebooks in order (01 → 12)
jupyter lab notebooks/

# 3. Launch the interactive dashboard
streamlit run app/app.py
```

## Notebook Map

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | Problem definition & dataset audit | Schema, quality, chronological split, leakage checklist |
| 02 | Temporal EDA | Volume, seasonality, account-age signals |
| 03 | Graph construction & network EDA | Degree distributions, shared-entity patterns |
| 04 | Node behaviour feature engineering | Full feature matrix (tabular + graph + temporal) |
| 05 | Statistical anomaly baselines | Robust z-score reference |
| 06 | Classical unsupervised ML | IsolationForest, LOF, EllipticEnvelope |
| 07 | Graph representation comparison | Ablation: tabular vs graph features |
| 08 | Temporal anomaly detection | Value of velocity features |
| 09 | Hybrid scoring & ablation | Rank-average ensemble |
| 10 | Synthetic anomaly stress testing | Controlled recovery experiment |
| 11 | Final evaluation | Locked test-set metrics + model freeze |
| 12 | Investigation & explainability | Feature deviation explanations, neighbourhood |

## Interactive Dashboard Features

- **Overview** – score distribution, top anomalies
- **Data Explorer** – filters, temporal charts
- **Graph Insights** – entity sharing statistics
- **Anomaly Ranking** – adjustable top-N + download
- **Investigation Workbench** – drill-down by rank / user / device
- **Model Performance** – ROC / PR curves, enrichment factor (labels for validation only)

## Data

- `Fraud_Data.csv` – 151 112 transactions, ~9.4 % fraud label
- `IpAddress_to_Country.csv` – IP range → country mapping

## Important Limitation

This application demonstrates **anomaly-ranking capability on the given dataset**.  
It is **not** evidence that the method detects fraud in the real world.

## License & Citation

Use freely for educational and research purposes.  
If you build on this work, please cite the repository.
