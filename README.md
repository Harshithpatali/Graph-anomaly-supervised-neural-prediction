```markdown
# 🛡️ Fraud Graph Anomaly Intelligence

### E-Commerce Fraud Detection with Graph Anomaly Detection · Neural Fraud Classification · Temporal Analysis · Investigator Workflows

<p align="center">
  <a href="https://graph-anomaly-supervised-neural-prediction-v1.streamlit.app/">
    <img src="https://img.shields.io/badge/🔴_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo"/>
  </a>
  &nbsp;
  <a href="https://github.com/Harshithpatali/Graph-anomaly-supervised-neural-prediction">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-MLP-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-Anomaly-F7931E?style=flat-square&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/PR--AUC-0.696-00C853?style=flat-square"/>
  <img src="https://img.shields.io/badge/ROC--AUC-0.829-2196F3?style=flat-square"/>
</p>

---

A **portfolio-grade** e-commerce fraud intelligence system that combines:

| Track | Type | Core Question |
|-------|------|---------------|
| **🕸️ Graph Anomaly** | Unsupervised | *Does this transaction / entity behave unusually?* |
| **🧠 Neural Fraud** | Supervised | *How likely is this transaction to be fraudulent given history?* |

Both tracks feed a single **investigator-oriented Streamlit dashboard**.

---

## 🚀 Live Application

**→ [Open the Live Demo](https://graph-anomaly-supervised-neural-prediction-v1.streamlit.app/)**

**Main navigation**

| Page | Purpose |
|------|---------|
| 🏠 Overview | Volume, fraud rate, anomaly distribution, top anomalies |
| 📊 Data Explorer | Filter by country, label, value, time, attributes |
| 🕸️ Graph Insights | User ↔ Device ↔ IP relationships & reuse |
| 🚨 Anomaly Ranking | Ranked list by anomaly score |
| 🔍 Investigation Workbench | Drill-down by rank / user / device + related activity |
| 🛡️ Fraud Detection Model | Locked test report + **New Transaction Prediction** |
| 📈 Model Performance | Full metrics, confusion matrix, review queue |
| ℹ️ About | Project context & methodology |

### 🔮 New Transaction Prediction (no CSV needed)

Enter a single transaction manually:

```
User ID · Signup time · Purchase time · Purchase value
Device ID · Source · Browser · Sex · Age · IP address
```

The frozen neural artifact returns:

- Fraud probability  
- Risk band  
- Decision vs frozen threshold  
- Model name + full transaction payload  

---

## 🎯 Project Objective

Fraud is rarely a pure transaction-level problem.

A transaction can look normal in isolation yet become suspicious when relationships are considered:

- Multiple users sharing one device  
- Multiple accounts appearing from the same IP  
- Unusual timing or purchase-value patterns  
- Entity reuse and rare combinatorial signals  

The system therefore answers **two distinct questions**:

| Question | Method | Label used in training? |
|----------|--------|-------------------------|
| *Is this behavior unusual?* | Graph anomaly detection | ❌ No (post-hoc validation only) |
| *Does this resemble historical fraud?* | Supervised PyTorch MLP | ✅ Yes |

Keeping the tracks separate makes the methodology more defensible.

---

## 🧠 System Architecture

```text
                         ┌──────────────────────┐
                         │   E-Commerce Data    │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │      Data Audit       │
                         │ Missingness / types   │
                         │ Temporal structure    │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┴──────────────────┐
                 │                                     │
                 ▼                                     ▼
       ┌────────────────────┐                ┌────────────────────┐
       │  Graph / Anomaly   │                │ Supervised Neural  │
       │       Track        │                │       Track        │
       └─────────┬──────────┘                └─────────┬──────────┘
                 │                                     │
                 ▼                                     ▼
       Graph construction                    Feature transformation
       Entity relationships                  Categorical encoding
       Behavioral features                   Numeric features
       Temporal features                     MLP training
       Embeddings                            Threshold selection
       Anomaly scoring                       Locked test evaluation
                 │                                     │
                 ▼                                     ▼
       anomaly_score                           fraud_probability
       anomaly_rank                            predicted_class
                 │                                     │
                 └──────────────────┬──────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Fraud Intelligence   │
                         │     Dashboard        │
                         └──────────┬───────────┘
                                    │
                 ┌──────────────────┼──────────────────┐
                 │                  │                  │
                 ▼                  ▼                  ▼
          Data exploration    Investigation      New transaction
          Graph insights     anomaly ranking       prediction
          Model analysis     entity drill-down     risk scoring
```

---

## 🔬 Modeling Philosophy

### Track A — Unsupervised Graph Anomaly Detection

The ecosystem is modeled as a multipartite relationship structure:

```text
Transaction
   ├── User
   ├── Device
   ├── IP address
   ├── Source
   ├── Browser
   └── Temporal context
```

**Key signals the graph can surface**

- One device → many users  
- One IP → multiple accounts  
- Rare entity combinations  
- Entities linked to already anomalous transactions  
- Behavior that is rare relative to the rest of the graph  

> The anomaly score is an **investigation ranking signal**, not proof of fraud.  
> The fraud label is **never** used during training — only for post-hoc evaluation (ROC-AUC, PR-AUC, precision@k, enrichment).

### Track B — Supervised Neural Fraud Detection

```text
Raw transaction data
        │
        ▼
Feature preparation
        │
        ▼
Stratified development split
        │
        ├───────────────┐
        ▼               ▼
      Train           Validation
        │               │
        └───────┬───────┘
                ▼
          Model selection
                │
                ▼
       Threshold selection
                │
                ▼
        Lock final model
                │
                ▼
          One-time test
          evaluation
                │
                ▼
       Saved neural artifact
```

The saved artifact owns the complete **feature-engineering → preprocessing → model → threshold** contract, eliminating training/serving skew.

---

## 📊 Dataset Snapshot

| Feature | Description |
|---------|-------------|
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
| `class` | Historical fraud label (`0` = legitimate, `1` = fraud) |

```text
Total transactions     151,112
Legitimate             136,961
Fraud                   14,151
Fraud prevalence         9.36 %
```

---

## ⚠️ Temporal Regime Shift (Critical Finding)

Fraud behavior is **not stationary**.

| Period | Fraud Rate |
|--------|------------|
| **January 2015** | **76.49 %** |
| February onward | ~4.5 % |

- Random stratified splits → good for controlled benchmarking  
- Chronological splits → reveal substantial temporal degradation  

The January regime was **not** silently removed to inflate metrics. It is treated as a robustness / distribution-shift finding.

---

## 🧪 Development Split

```text
Train        105,778
Validation    22,667
Test          22,667   ← locked during model development
```

Validation is used for model comparison, diagnostics, and **threshold selection**.  
The test set is evaluated only after every decision is frozen.

---

## 🏆 Final Neural Model — Locked Test Results

| Metric | Value |
|--------|------:|
| **Model** | PyTorch MLP |
| **Primary metric** | PR-AUC |
| **Decision threshold** | **0.937358** |
| **PR-AUC** | **0.695629** |
| **ROC-AUC** | **0.829392** |
| Accuracy | 0.955927 |
| Precision | 0.979505 |
| Recall | 0.540528 |
| F1 | 0.696629 |
| Balanced Accuracy | 0.769680 |

### Confusion Matrix (Locked Test)

```text
                       Predicted
                    Legitimate   Fraud
Actual Legitimate     20,521       24
Actual Fraud             975    1,147
```

- False Positives = **24**  
- False Negatives = **975**  

→ High precision, moderate recall → designed as a **decision-support** system, not an autonomous adjudicator.

---

## 🎚️ Decision Threshold Logic

```text
fraud_probability  ≥  0.9373584389686584
          │
┌─────────┴─────────┐
▼                   ▼
FRAUD — REVIEW     LEGITIMATE
```

Probability and binary decision are always shown separately.

---

## 🚨 Risk Bands (UI Presentation Only)

| Band | Probability Range |
|------|-------------------|
| Very Low | < 10 % |
| Low | 10 – 25 % |
| Moderate | 25 – 50 % |
| Elevated | 50 – 75 % |
| High | 75 – 90 % |
| **Critical** | **≥ 90 %** |

These bands are for presentation; the binary decision still uses the frozen threshold.

---

## 🔍 Investigator Workflow

1. **Overview** → volume, fraud rate, anomaly distribution, top anomalies  
2. **Data Explorer** → slice by country, label, value, time  
3. **Graph Insights** → shared devices & IPs, connectivity  
4. **Anomaly Ranking** → focus on highest anomaly scores  
5. **Investigation Workbench** → drill by rank / user / device + same-user / same-device / same-IP activity  
6. **Fraud Detection Model** → locked test report + live new-transaction scoring  
7. **Fraud Review Queue** → predicted-fraud transactions sorted by probability  

---

## 📦 Key Artifacts

```text
artifacts/
├── final_model.joblib                        # frozen unsupervised graph/anomaly model
├── final_supervised_neural_model.joblib      # neural serving artifact
├── final_test_predictions.csv                # locked test predictions for dashboard
├── classical_models.joblib
├── classical_preprocessor.joblib
├── graph_svd.joblib
├── graph_train_embedding.npy
├── graph_validation_embedding.npy
├── train_features.parquet
├── validation_features.parquet
├── test_features.parquet
├── validation_ranked.parquet
├── investigation_ranked.parquet
├── temporal_daily_scores.parquet
├── feature_config.json
└── hybrid_config.json
```

---

## 📁 Project Structure

```text
fraud_ecommerce_graph_anomaly_and_ml/
├── app/
│   └── app.py
├── artifacts/                  # all frozen models & predictions
├── data/
│   ├── raw/
│   └── processed/
│       └── dashboard.parquet
├── notebooks/                  # 01 → 13 progressive workflow
├── reports/
│   ├── data_audit.json
│   ├── final_test_evaluation.json
│   ├── model_card.md
│   └── ...
├── src/
│   ├── anomaly.py
│   ├── data_utils.py
│   ├── embeddings.py
│   ├── evaluation.py
│   ├── final_pipeline.py
│   ├── graph_features.py
│   ├── graph_utils.py
│   ├── pipeline_utils.py
│   ├── supervised_fraud.py
│   └── temporal_features.py
├── requirements.txt
└── README.md
```

---

## 📓 Notebook Workflow

**Unsupervised / Graph Track**

```text
01 Data Audit
02 Temporal EDA
03 Graph Construction / Network EDA
04 Node / Behavioral Feature Engineering
05 Statistical Anomaly Baselines
06 Classical Unsupervised ML
07 Graph Representation / Embeddings
08 Temporal Anomaly Detection
09 Hybrid Scoring / Ablation
10 Synthetic Anomaly Stress Testing
11 Final Evaluation / Model Freeze
12 Investigation / Explainability
```

**Supervised Neural Track**

```text
13 Neural Fraud Detection
   → Load splits → Feature prep → MLP experiments
   → Validation selection → Threshold selection
   → Locked test → Save artifact → Generate predictions
```

---

## 🔐 Data Leakage Controls

| Concern | Control |
|---------|---------|
| Unsupervised model | Fraud label never used in training |
| Supervised model | Test set never used for selection or threshold |
| Dashboard | Reads frozen `final_test_predictions.csv` (no re-prediction) |

---

## 📈 Why Two Models?

```text
             Transaction
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 Graph anomaly         Neural classifier
 "Is this unusual?"    "Does this resemble fraud?"
        │                   │
        └─────────┬─────────┘
                  ▼
          Investigator review
```

| Strength | Limitation |
|----------|------------|
| Surfaces novel / unusual behavior | Anomalous ≠ fraudulent |
| Learns historical fraud patterns | Can struggle with novel attacks & distribution shift |

Together they give complementary evidence.

---

## ⚠️ Limitations

1. **Temporal drift** — random-split performance does not guarantee future performance.  
2. **False negatives** — 975 on the locked test set; the model is not a complete filter.  
3. **Label dependence** — supervised boundary reflects historical investigation policies.  
4. **Novel fraud** — previously unseen strategies may not resemble past examples.  
5. **Manual prediction** — a dashboard score is evidence, not confirmation.

---

## 🛠️ Installation & Run

```bash
# Clone
git clone https://github.com/Harshithpatali/Graph-anomaly-supervised-neural-prediction.git
cd Graph-anomaly-supervised-neural-prediction

# Virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1

# Dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app/app.py
# or
python -m streamlit run app/app.py --server.port 8504
```

**Required runtime artifacts**

```text
artifacts/final_model.joblib
artifacts/final_supervised_neural_model.joblib
artifacts/final_test_predictions.csv
data/processed/dashboard.parquet   (or .pkl)
```

---

## 🧱 Training-to-Serving Contract

```text
Notebook
   │
   ├── feature engineering
   ├── preprocessing
   ├── model
   └── threshold
           │
           ▼
final_supervised_neural_model.joblib
           │
           ▼
       Streamlit
           │
           ▼
   predict_proba(raw transaction)
```

The application never re-implements training-time feature engineering.

---

## 🧭 Design Principles

1. Separate discovery (anomaly) from classification (supervised).  
2. Prefer **PR-AUC** for imbalanced fraud problems.  
3. Lock the test set — never use it for iterative tuning.  
4. Report temporal findings; do not hide distribution shift.  
5. Keep inference identical to training via the frozen artifact.  
6. Make outputs investigator-friendly (queues + context).  
7. Treat probability as evidence, not certainty.

---

## 🛠️ Technology Stack

| Technology | Role |
|------------|------|
| Python | Core language |
| Pandas / NumPy / SciPy | Data & numerics |
| scikit-learn | Classical ML & evaluation |
| **PyTorch** | Neural fraud model |
| Joblib | Artifact serialization |
| Plotly | Interactive dashboard charts |
| Matplotlib | Notebook visualization |
| **Streamlit** | Investigator-facing app |
| Jupyter | Research & experimentation |
| Parquet | Efficient tabular storage |

---

## 🌐 Links

- **Live Demo** → https://graph-anomaly-supervised-neural-prediction-v1.streamlit.app/  
- **GitHub** → https://github.com/Harshithpatali/Graph-anomaly-supervised-neural-prediction  
- **Author** → [Harshith Patali](https://github.com/Harshithpatali)

---

## 📌 Final Takeaway

```text
Raw data
   ↓
Data audit + Temporal understanding
   ↓
Graph construction + Behavioral features
   ↓
Unsupervised anomaly detection + Ranking
   ↓
Investigation workflows
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

> **Use graph anomaly detection to discover unusual behavior,  
> supervised learning to estimate fraud risk from historical evidence,  
> and an investigator-oriented dashboard to turn both signals into actionable decisions.**

This is **fraud intelligence and decision support** — not an autonomous adjudication system.
```
