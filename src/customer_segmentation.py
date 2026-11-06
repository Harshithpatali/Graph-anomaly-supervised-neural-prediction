"""Reproducible unsupervised customer segmentation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler


def segment_customers(profiles: pd.DataFrame, feature_columns: list[str] | None = None,
                      cluster_range: range = range(2, 9), random_state: int = 42):
    if len(profiles) < 3:
        raise ValueError("At least three customer profiles are required.")
    excluded = {"user_id", "first_purchase", "last_purchase", "as_of", "segment_id", "behavioral_risk_index"}
    if feature_columns is None:
        feature_columns = [c for c in profiles.select_dtypes(include=[np.number]).columns if c not in excluded]
    if not feature_columns:
        raise ValueError("No numeric customer features available for clustering.")
    missing = set(feature_columns).difference(profiles.columns)
    if missing:
        raise ValueError(f"Missing clustering features: {sorted(missing)}")
    X = profiles[feature_columns].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).fillna(0.0)
    scaled = RobustScaler().fit_transform(X)
    candidates = [k for k in cluster_range if 2 <= k < len(profiles)]
    if not candidates:
        raise ValueError("No valid cluster counts for this customer sample.")
    sample_n = min(5000, len(profiles))
    rng = np.random.default_rng(random_state)
    sample_idx = np.sort(rng.choice(len(profiles), sample_n, replace=False)) if sample_n < len(profiles) else np.arange(len(profiles))
    rows, fitted = [], []
    for k in candidates:
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(scaled)
        sample_labels = labels[sample_idx]
        sil = silhouette_score(scaled[sample_idx], sample_labels) if len(np.unique(sample_labels)) > 1 else -1.0
        db = davies_bouldin_score(scaled, labels)
        rows.append({"k": k, "silhouette_sample": float(sil), "davies_bouldin": float(db), "inertia": float(model.inertia_)})
        fitted.append((model, labels))
    metrics = pd.DataFrame(rows).sort_values(["silhouette_sample", "davies_bouldin"], ascending=[False, True])
    best_k = int(metrics.iloc[0]["k"])
    labels = next(labels for model, labels in fitted if model.n_clusters == best_k)
    output = profiles.copy()
    output["segment_id"] = labels.astype(int)
    aggregations = {
        "customers": ("user_id", "nunique"), "transactions": ("transaction_count", "sum"),
        "median_transactions": ("transaction_count", "median"),
        "median_total_spend": ("total_spend", "median"),
        "median_avg_ticket": ("mean_transaction_value", "median"),
        "median_recency_days": ("days_since_last_purchase", "median"),
    }
    if "distinct_device_id" in output:
        aggregations["median_devices"] = ("distinct_device_id", "median")
    summary = output.groupby("segment_id").agg(**aggregations).reset_index()
    metadata = {"selected_k": best_k, "feature_columns": feature_columns,
                "random_state": random_state,
                "selection_metric": "highest sample silhouette, tie-broken by Davies-Bouldin"}
    return output, metrics, {"metadata": metadata, "segment_summary": summary.to_dict(orient="records")}
