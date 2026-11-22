# Model Card — Customer Graph & Anomaly Intelligence

## Intended use

Exploratory customer-behavior analytics and analyst-assisted investigation prioritization on the public Fraud_Data benchmark. The dashboard surfaces behavioral profiles, unsupervised anomaly ranks, customer segments and user-device-IP connectivity.

## Out-of-scope use

Do not use scores as proof of fraud, calibrated fraud probabilities, automatic account blocks, accusations, or adverse customer decisions. Human review and independent evidence are required.

## Data and target

- Source: public Fraud_Data transaction benchmark.
- Primary customer-intelligence pipeline deliberately drops the historical `class` label before feature generation.
- Customer profiles are descriptive aggregates through an explicit `as_of` timestamp.
- Entity graph nodes represent users, devices and IP addresses; edges represent observed user-device or user-IP associations.
- The graph builder is capped at the first 50,000 input transactions in the current workflow.

## Methods

- Customer-level behavioral aggregation and robust numeric feature handling.
- K-means segmentation; cluster metrics include silhouette and Davies–Bouldin diagnostics.
- Isolation Forest and Local Outlier Factor anomaly scoring, combined as 60% / 40% percentile ranks.
- A separate 0–100 cohort-relative behavioral review index with explicit heuristic weights.
- NetworkX entity graph with degree, connected-component, betweenness and community summaries.
- New graph validation report measures node/edge type integrity, component structure, shared-entity connectivity, density and isolated nodes.

## Validation and reporting

Run `pytest -q` and `python scripts/build_customer_intelligence.py`. The workflow exports graph metrics to `reports/customer_intelligence/graph_validation_metrics.json` and records them in `run_metadata.json`.

Graph metrics are structural diagnostics, not supervised performance metrics. Because the primary pipeline intentionally does not use fraud labels, precision, recall, PR-AUC and ROC-AUC are not valid claims for the graph/anomaly product. A future labeled evaluation must define a time-based split, entity-disjoint leakage checks, analyst-review capacity and a label-maturation window before reporting detection performance.

## Known limitations

- Graph construction uses a bounded transaction sample, so results depend on input ordering and the 50,000-row cap.
- Full-snapshot entity frequency features are descriptive, not point-in-time predictive features.
- Shared devices/IPs can arise from households, workplaces, carrier NAT, public networks or shared infrastructure.
- Louvain communities and centrality depend on graph composition and sampling.
- Anomaly scores are relative rankings; their scale is not calibrated.
- The benchmark is not a production customer population.

## Business interpretation

The system helps an analyst answer: which customers or shared entities merit a closer look, what behavioral attributes contributed to an anomaly rank, and how are users connected through observed devices/IPs? It does not answer whether a customer committed fraud. Operational value must be validated through analyst-confirmed outcomes, precision at review capacity, false-positive burden and review-time savings.

## Reproducibility

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
pytest -q
python scripts/build_customer_intelligence.py
streamlit run app/app.py
```

The raw benchmark files must be placed under `data/raw/` as described in the README.