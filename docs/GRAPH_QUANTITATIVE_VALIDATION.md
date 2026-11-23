# Graph intelligence: quantitative validation

## Scope

The current dashboard is an **unsupervised customer-intelligence and graph-investigation tool**. Graph statistics are structural diagnostics, not a fraud classifier. The historical `class` field is excluded before profile generation, segmentation, anomaly ranking and graph construction.

## Metrics emitted by the research build

Running `python scripts/build_customer_intelligence.py` writes `reports/customer_intelligence/graph_validation_metrics.json`. The graph is bounded to the first 50,000 input transactions.

| Metric | Definition | Interpretation |
|---|---|---|
| Node/edge counts | Number of unique entity nodes and undirected relationships | Graph size, not model quality |
| Node-type counts | Users, devices and IPs | Schema coverage |
| Edge-type counts | User-device and user-IP links | Relationship integrity |
| Invalid edges | Edges that do not match the declared bipartite relation types | Must be zero for the current graph contract |
| Connected components | Number of disconnected subgraphs | Network fragmentation |
| Largest-component share | Nodes in largest component / all nodes | Concentration of connectivity |
| Isolated nodes | Nodes with degree zero | Unconnected entities |
| Shared-device/IP counts | Device/IP nodes linked to more than one user | Potential investigation pivots |
| Users linked to shared entities | Unique users connected to a shared device or IP | Exposure volume, not guilt |

## Deterministic structural acceptance checks

The test suite uses a small hand-constructed graph with three users, two devices and two IPs. It asserts exact node/edge counts, node types, edge types, component count, shared-entity counts, and zero invalid edges. It also checks empty-graph behavior. These are unit-test fixtures, **not empirical dataset findings**.

Run:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/test_customer_intelligence.py
python scripts/build_customer_intelligence.py
```

The source file `data/raw/Fraud_Data.csv` is required for the full report build. The workflow uploads the generated report bundle as a GitHub Actions artifact.

## What would establish incremental graph value?

Structural validity alone does not show that graph features improve detection. A defensible next experiment should:

1. Sort transactions by event time and define train, validation and untouched test periods before computing any label-linked statistic.
2. Build graph state only from transactions available before each prediction timestamp. Never use future edges or full-snapshot entity frequencies for a predictive claim.
3. Compare a tabular-only baseline with the same model plus point-in-time graph features (e.g. prior shared-device users, prior shared-IP users, prior entity degree and component size).
4. Keep the exact same temporal rows, folds, model family and tuning budget for both conditions.
5. Report PR-AUC, ROC-AUC, precision/recall at fixed review capacity, calibration, and paired bootstrap confidence intervals by user or time block.
6. Report cold-start performance for entities not seen in the graph history.
7. Run an ablation removing device and IP features separately.

No incremental graph lift is claimed in this report. The current metrics describe graph structure only.

## Analyst guardrails

A shared IP/device may reflect households, workplaces, carrier NAT, public Wi-Fi, shared devices or benign infrastructure. Do not treat connectivity, centrality or anomaly rank as proof of fraud, and do not automatically block or penalize a customer. Human review and independent evidence are required.
