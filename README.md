# Fraud Graph Anomaly Intelligence

<p align="center">
  <strong>Graph Anomaly Detection · Neural Fraud Classification · Temporal Analysis · Investigator Workflows</strong>
</p>

<p align="center">
  <a href="https://graph-anomaly-supervised-neural-prediction-v1.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo" />
  </a>
  <a href="https://github.com/Harshithpatali/Graph-anomaly-supervised-neural-prediction">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" />
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/PyTorch-MLP-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/scikit--learn-ML%20%26%20Evaluation-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/PR--AUC-0.696-00C853?style=flat-square" alt="PR-AUC" />
  <img src="https://img.shields.io/badge/ROC--AUC-0.829-2196F3?style=flat-square" alt="ROC-AUC" />
</p>

<p align="center">
  <em>An end-to-end fraud intelligence system that combines relationship-aware anomaly detection with supervised neural risk scoring.</em>
</p>

---

## Overview

This project is a portfolio-grade **e-commerce fraud intelligence platform** built around two complementary modeling tracks:

| Track | Learning | Core question | Output |
|---|---|---|---|
| **Graph Anomaly** | Unsupervised | **Does this behavior look unusual?** | `anomaly_score`, `anomaly_rank` |
| **Neural Fraud** | Supervised | **How likely is this transaction to be fraudulent?** | `fraud_probability`, `predicted_class` |

Both tracks are exposed through a single **Streamlit investigation dashboard**.

The key design decision is to keep the two modeling problems separate. The graph anomaly model does **not** train on the fraud label; the label is used only for post-hoc evaluation. The neural classifier is explicitly supervised and is evaluated using a locked test set.

### What makes the project different?

- Relationship-aware fraud analysis using users, devices, IP addresses, and transaction activity.
- A dedicated unsupervised anomaly-detection track rather than relying only on classification.
- A final PyTorch MLP for supervised fraud prediction.
- Validation-based threshold selection instead of blindly using `0.50`.
- Locked test-set evaluation and precomputed test predictions.
- Explicit analysis of severe temporal distribution shift.
- Investigator-oriented ranking and drill-down workflows.
- Manual **new transaction prediction without CSV upload**.
- Frozen artifacts that preserve the training-to-serving feature pipeline.

---

## Live Demo

### [Open the Fraud Intelligence Dashboard](https://graph-anomaly-supervised-neural-prediction-v1.streamlit.app/)

The deployed application contains the complete analytical workflow:

| Dashboard page | Purpose |
|---|---|
| 🏠 **Overview** | Transaction volume, fraud rate, anomaly distribution, top anomalies |
| 📊 **Data Explorer** | Interactive filtering and transaction exploration |
| 🕸️ **Graph Insights** | User/device/IP relationships and entity reuse |
| 🚨 **Anomaly Ranking** | Highest-risk anomaly scores and ranked transactions |
| 🔍 **Investigation Workbench** | Drill into users, devices, ranks, and related activity |
| 🛡️ **Fraud Detection Model** | Locked test report and new-transaction prediction |
| 📈 **Model Performance** | Metrics, confusion matrix, review queue, evaluation views |
| ℹ️ **About** | Methodology, project context, and architecture |

### New transaction prediction

The application can score a single transaction directly from the UI. No CSV upload is required.

Inputs include:

```text
User ID
Signup time
Purchase time
Purchase value
Device ID
Source
Browser
Sex
Age
IP address
```

The frozen neural artifact returns:

- Fraud probability
- Risk band
- Decision against the frozen threshold
- Model name
- Submitted transaction payload

---

## Why Two Modeling Tracks?

Fraud is rarely a purely row-level problem.

A transaction can appear ordinary by itself while its relationships reveal suspicious behavior:

```text
                         Transaction
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
            User           Device             IP
              │               │               │
              └───────────────┼───────────────┘
                              │
                              ▼
                     Behavioral context
```

The project therefore answers two different questions:

| Question | Method | Fraud label used during training? |
|---|---|---:|
| **Is this behavior unusual?** | Graph anomaly detection | ❌ No |
| **Does this resemble historical fraud?** | Supervised PyTorch MLP | ✅ Yes |

This separation is important. **Anomaly does not mean fraud**, and a supervised classifier cannot guarantee that a novel fraud strategy will resemble historical examples.

Together, the models provide complementary evidence for investigation.

---

# System Architecture

```text
                         ┌─────────────────────────┐
                         │    E-Commerce Data      │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │       Data Audit        │
                         │ types · missingness     │
                         │ target · temporal data  │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
          ┌──────────────────┐                ┌──────────────────┐
          │  GRAPH / ANOMALY │                │ SUPERVISED NEURAL│
          │      TRACK       │                │      TRACK       │
          └────────┬─────────┘                └────────┬─────────┘
                   │                                   │
                   ▼                                   ▼
          Graph construction                   Feature preparation
          Entity relationships                 Encoding / scaling
          Behavioral features                  MLP training
          Temporal features                    Validation selection
          Graph embeddings                      Threshold selection
          Anomaly scoring                       Locked test
                   │                                   │
                   ▼                                   ▼
          anomaly_score                         fraud_probability
          anomaly_rank                          predicted_class
                   │                                   │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
                         ┌─────────────────────────┐
                         │  FRAUD INTELLIGENCE    │
                         │       DASHBOARD        │
                         └────────────┬────────────┘
                                      │
                ┌─────────────────────┼─────────────────────┐
                ▼                     ▼                     ▼
         Explore / Explain       Investigate         Score New Tx
```

---

# Dataset

The project uses transaction-level e-commerce fraud data with the following core fields:

| Feature | Description |
|---|---|
| `user_id` | User identifier |
| `signup_time` | Account signup timestamp |
| `purchase_time` | Transaction timestamp |
| `purchase_value` | Transaction value |
| `device_id` | Device identifier |
| `source` | Acquisition / traffic source |
| `browser` | Browser used |
| `sex` | User sex |
| `age` | User age |
| `ip_address` | IP address |
| `class` | Historical fraud label: `0` legitimate, `1` fraud |

### Dataset size

```text
Transactions      151,112
Legitimate        136,961
Fraud              14,151
Fraud prevalence      9.36%
```

The positive class is therefore sufficiently imbalanced that **accuracy alone is not an appropriate primary model-selection metric**.

---

# Critical Finding: Temporal Regime Shift

One of the most important findings in the project is that fraud behavior is **not stationary over time**.

| Period | Fraud rate |
|---|---:|
| **January 2015** | **76.49%** |
| February onward | **~4.5%** |

This creates an important distinction between two evaluation strategies:

- **Stratified random split** — useful for controlled model development and benchmarking.
- **Chronological split** — useful for testing generalization to a future behavioral regime.

The chronological experiments showed substantial degradation under temporal shift.

The January regime was **not silently removed** to make the model look better. It is treated as a genuine distribution-shift finding and an important limitation of the dataset/modeling setup.

---

# Supervised Modeling

## Development split

The supervised benchmark uses a stratified random split:

```text
Train        105,778
Validation    22,667
Test          22,667   ← locked during development
```

The responsibilities are deliberately separated:

```text
TRAIN
  └── Learn model parameters

VALIDATION
  ├── Compare models
  ├── Tune development choices
  └── Select operating threshold

TEST
  └── One-time final evaluation after decisions are frozen
```

The test set is not used for iterative model selection or threshold tuning.

---

# Final Neural Champion

The final supervised model is a **PyTorch MLP**.

### Locked test results

| Metric | Result |
|---|---:|
| **Primary metric: PR-AUC** | **0.695629** |
| ROC-AUC | 0.829392 |
| Accuracy | 0.955927 |
| Precision | 0.979505 |
| Recall | 0.540528 |
| F1 | 0.696629 |
| Balanced Accuracy | 0.769680 |
| Decision threshold | **0.937358** |

### Confusion matrix

```text
                         Predicted
                    Legitimate     Fraud
Actual Legitimate     20,521         24
Actual Fraud             975      1,147
```

Therefore:

- **True negatives:** 20,521
- **False positives:** 24
- **False negatives:** 975
- **True positives:** 1,147

The operating point gives very high precision but moderate recall. This makes the model useful as a **high-confidence review signal**, while clearly leaving room for investigation and complementary anomaly detection.

---

# Why PR-AUC?

Fraud detection is an imbalanced classification problem.

A model can achieve high accuracy simply by predicting the majority class frequently. That does not mean it is useful for fraud investigation.

The project therefore uses **Precision-Recall AUC (PR-AUC)** as the primary model-selection metric because it focuses directly on the quality of positive-class retrieval.

ROC-AUC remains useful as a secondary ranking metric, while threshold-dependent metrics are reported separately.

---

# Threshold Selection

The final decision threshold is:

```text
0.9373584389686584
```

The threshold was selected on validation data and frozen before the locked test evaluation.

```text
fraud_probability >= 0.9373584389686584
                    │
              ┌─────┴─────┐
              ▼           ▼
        FRAUD / REVIEW   LEGITIMATE
```

This is intentionally different from the default `0.50` threshold.

The dashboard displays both:

1. the continuous **fraud probability**, and
2. the resulting **binary decision**.

That distinction is important because probability is evidence, while the threshold is an operational policy choice.

---

# Risk Bands

The dashboard uses the following presentation bands:

| Risk band | Probability |
|---|---:|
| Very Low | `< 10%` |
| Low | `10–25%` |
| Moderate | `25–50%` |
| Elevated | `50–75%` |
| High | `75–90%` |
| **Critical** | **≥ 90%** |

These bands are **UI presentation categories**, not additional model classes. The actual binary decision continues to use the frozen validation-selected threshold.

---

# Graph Anomaly Track

The unsupervised track models the transaction ecosystem as a relationship structure involving entities such as:

```text
Transaction
   ├── User
   ├── Device
   ├── IP address
   ├── Source
   ├── Browser
   └── Temporal context
```

### Signals the graph can surface

- One device associated with many users.
- One IP associated with multiple accounts.
- Rare entity combinations.
- Entity reuse across suspicious activity.
- Unusual behavioral patterns relative to the graph population.

The anomaly model produces:

```text
anomaly_score
anomaly_rank
```

These values are used to prioritize investigation.

> **Important:** the fraud label is not used to train the anomaly detector. Fraud labels are used only for post-hoc evaluation such as PR-AUC, ROC-AUC, precision@k, and enrichment.

---

# Investigator Workflow

The application is designed around a practical investigation sequence:

```text
1. Understand the population
        ↓
2. Identify unusual behavior
        ↓
3. Rank suspicious transactions
        ↓
4. Drill into graph relationships
        ↓
5. Review supervised fraud probability
        ↓
6. Score new transactions when needed
        ↓
7. Make an informed investigation decision
```

### Overview

High-level operational view including transaction volume, fraud rate, anomaly distribution, and top anomalies.

### Data Explorer

Interactive exploration of transaction attributes and population segments.

### Graph Insights

Relationship-level analysis across users, devices, IP addresses, and transaction activity.

### Anomaly Ranking

Prioritizes transactions by anomaly score.

### Investigation Workbench

Supports drill-down by anomaly rank, user, and device, with related activity and contextual signals.

### Fraud Detection Model

Provides the locked test report and manual new-transaction prediction.

### Model Performance

Presents evaluation metrics, confusion matrix information, and fraud review outputs.

---

# New Transaction Prediction

A key application feature is the ability to score a transaction that was **not part of the original test CSV**.

The investigator enters the transaction directly in the UI.

```text
Raw transaction
      │
      ▼
Saved feature transformer
      │
      ▼
Saved preprocessing
      │
      ▼
Frozen PyTorch MLP
      │
      ▼
Fraud probability
      │
      ▼
Frozen decision threshold
      │
      ▼
Risk / review decision
```

The model is not retrained during this process.

---

# Training-to-Serving Contract

The final supervised artifact is designed to preserve the same transformation logic used during development.

```text
Training
────────
Raw fields
   ↓
Feature engineering
   ↓
Preprocessing
   ↓
MLP
   ↓
Validation threshold
   ↓
Frozen artifact

Serving
───────
New raw transaction
   ↓
Same artifact-owned transformation
   ↓
Same MLP
   ↓
Fraud probability
   ↓
Frozen threshold
```

This avoids re-implementing training-time feature engineering inside the dashboard and reduces the risk of training/serving mismatch.

---

# Data Leakage Controls

| Area | Control |
|---|---|
| Unsupervised anomaly model | Fraud label is not used during training |
| Supervised development | Test set is kept locked |
| Threshold selection | Selected using validation data before test evaluation |
| Test reporting | Uses precomputed locked predictions |
| New transaction scoring | Inference only; no retraining |

The distinction between **model training**, **validation decisions**, and **final test evaluation** is intentionally preserved throughout the project.

---

# Notebook Roadmap

The project is organized as a progressive analytical workflow rather than jumping directly to a final model.

## Graph / Unsupervised Track

| Notebook | Focus |
|---|---|
| **01** | Data Audit |
| **02** | Temporal EDA |
| **03** | Graph Construction / Network EDA |
| **04** | Node & Behavioral Feature Engineering |
| **05** | Statistical Anomaly Baselines |
| **06** | Classical Unsupervised ML |
| **07** | Graph Representation / Embeddings |
| **08** | Temporal Anomaly Detection |
| **09** | Hybrid Scoring / Ablation |
| **10** | Synthetic Anomaly Stress Testing |
| **11** | Final Evaluation / Model Freeze |
| **12** | Investigation / Explainability |

## Supervised Neural Track

### Notebook 13 — Neural Fraud Detection

```text
Load train / validation / test
        ↓
Feature preparation
        ↓
MLP experiments
        ↓
Validation model selection
        ↓
Threshold selection
        ↓
Lock champion
        ↓
One-time test evaluation
        ↓
Save serving artifact
        ↓
Generate final test predictions
```

---

# Key Artifacts

```text
artifacts/
│
├── final_model.joblib
│     └── Frozen graph/anomaly model and transformation components
│
├── final_supervised_neural_model.joblib
│     └── Frozen supervised neural serving artifact
│
├── final_test_predictions.csv
│     └── Locked test predictions used by the dashboard
│
├── graph_svd.joblib
├── graph_train_embedding.npy
├── graph_validation_embedding.npy
│
├── train_features.parquet
├── validation_features.parquet
├── test_features.parquet
│
├── validation_ranked.parquet
├── investigation_ranked.parquet
├── temporal_daily_scores.parquet
│
├── classical_models.joblib
├── classical_preprocessor.joblib
├── feature_config.json
└── hybrid_config.json
```

### Locked test prediction schema

The final prediction file contains the original transaction fields plus:

```text
actual_class
fraud_probability
predicted_class
```

This allows the dashboard to build a reproducible test report without repeatedly re-scoring the locked test set.

---

# Repository Structure

```text
Graph-anomaly-supervised-neural-prediction/
│
├── app/
│   └── app.py                         # Streamlit application
│
├── artifacts/                         # Frozen models and generated artifacts
│
├── data/
│   ├── raw/                           # Raw dataset
│   └── processed/
│       └── dashboard.parquet          # Precomputed dashboard table
│
├── notebooks/                         # 01 → 13 research workflow
│
├── reports/                           # Evaluation reports and model card
│
├── src/
│   ├── anomaly.py                     # Anomaly methods
│   ├── data_utils.py                  # Data loading / persistence
│   ├── embeddings.py                  # Graph representations
│   ├── evaluation.py                  # Evaluation utilities
│   ├── final_pipeline.py              # Final anomaly scoring pipeline
│   ├── graph_features.py              # Graph-derived features
│   ├── graph_utils.py                 # Graph construction utilities
│   ├── pipeline_utils.py              # Shared pipeline helpers
│   ├── supervised_fraud.py            # Neural fraud model / serving artifact
│   └── temporal_features.py           # Time-aware features
│
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone

```bash
git clone https://github.com/Harshithpatali/Graph-anomaly-supervised-neural-prediction.git
cd Graph-anomaly-supervised-neural-prediction
```

## 2. Create a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

Core stack:

- Python
- Pandas
- NumPy
- SciPy
- scikit-learn
- PyTorch
- Plotly
- Matplotlib
- Streamlit
- Joblib
- Jupyter

---

# Run the Dashboard

From the repository root:

```bash
python -m streamlit run app/app.py
```

If port `8501` is occupied:

```bash
python -m streamlit run app/app.py --server.port 8504
```

Then open:

```text
http://localhost:8501
```

or:

```text
http://localhost:8504
```

### Required runtime artifacts

For the complete dashboard experience, the project expects:

```text
artifacts/final_model.joblib
artifacts/final_supervised_neural_model.joblib
artifacts/final_test_predictions.csv
data/processed/dashboard.parquet
```

---

# Reproducibility

The modeling workflow uses fixed random seeds where appropriate and records the important decisions required to reproduce the final artifacts.

The final workflow separates:

```text
Experimentation
      ↓
Validation decisions
      ↓
Model freeze
      ↓
Locked test evaluation
      ↓
Serving artifact
```

The Streamlit application consumes the frozen outputs rather than depending on an active notebook session.

---

# Limitations

This project is intentionally explicit about what the models can and cannot establish.

### Temporal drift

Random-split performance does not guarantee future performance. The dataset contains a major temporal regime shift.

### False negatives

The final neural model produces **975 false negatives** on the locked test set. It should therefore not be treated as a complete fraud filter.

### Historical labels

The supervised classifier learns from historical fraud labels. Label quality and historical investigation policies can influence the learned decision boundary.

### Novel fraud

Previously unseen fraud strategies may not resemble historical examples. The anomaly track provides a complementary mechanism for surfacing unusual behavior.

### Human review

A model probability is evidence, not proof. Fraud decisions should incorporate graph relationships, transaction history, business rules, and investigator judgment.

---

# Design Principles

1. **Separate discovery from classification.** Anomaly detection and supervised classification answer different questions.
2. **Use PR-AUC for imbalanced fraud modeling.** Accuracy alone is not sufficient.
3. **Keep the test set locked.** Do not turn final evaluation into another tuning loop.
4. **Report distribution shift.** Do not hide temporal instability to improve headline metrics.
5. **Preserve the training-to-serving contract.** The saved artifact owns the transformation pipeline.
6. **Design for investigation.** Scores should lead to useful queues and contextual evidence.
7. **Treat predictions as evidence.** A probability is not a confirmation of fraud.

---

# Technology Stack

| Technology | Role |
|---|---|
| **Python** | Core language |
| **Pandas / NumPy / SciPy** | Data manipulation and numerical computing |
| **scikit-learn** | Classical ML utilities and evaluation |
| **PyTorch** | Final supervised neural model |
| **Joblib** | Model and artifact serialization |
| **Plotly** | Interactive visualizations |
| **Matplotlib** | Research notebook visualizations |
| **Streamlit** | Investigator-facing application |
| **Jupyter** | Research and experimentation |
| **Parquet** | Efficient tabular artifact storage |

---

# Project Links

- 🚀 **Live Demo:** https://graph-anomaly-supervised-neural-prediction-v1.streamlit.app/
- 💻 **GitHub:** https://github.com/Harshithpatali/Graph-anomaly-supervised-neural-prediction

---

# Final Takeaway

The complete system follows this path:

```text
Raw transaction data
        ↓
Data audit + temporal analysis
        ↓
Graph construction + behavioral features
        ↓
Unsupervised anomaly detection
        ↓
Anomaly ranking + investigation
        ↓
Supervised neural fraud classification
        ↓
Validation threshold selection
        ↓
Locked test evaluation
        ↓
Frozen model artifact
        ↓
Interactive fraud intelligence dashboard
        ↓
Manual new-transaction prediction
```

> **Use graph anomaly detection to discover unusual behavior, supervised learning to estimate fraud risk from historical evidence, and an investigator-oriented dashboard to turn both signals into actionable decisions.**

This project is designed as **fraud intelligence and decision support**, not as an autonomous fraud adjudication system.

---

## Author

**Harshith Patali**

[GitHub](https://github.com/Harshithpatali)
