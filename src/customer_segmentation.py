"""Reproducible unsupervised customer segmentation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler


def segment_customers(
    profiles: pd.DataFrame,
    feature_columns: list[str] | None = None,
    cluster_range: range = range(2, 9),
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if len(profiles) < 3:
        raise ValueError("At least three customer profiles are required.")
    excluded = {"user_id", "first_purchase", "last_purchase", "as_of", "segment_id"}
    if feature_columns is None:
        feature_columns = [c for c in profiles.select_dtypes(include=[np.number]).columns if c not in excluded]
    if not feature_columns:
        raise ValueError("No numeric customer features available for clustering.")
    X = profiles[feature_columns].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).fillna(0.0)
    scaler = RobustScaler()
    scaled = scaler.fit_transform(X)
    max_k = min(max(cluster_range), len(profiles) - 1)
    candidates = [k for k in cluster_range if 2 <= k <= max_k]
    if not candidates:
        raise ValueError("Not enough customers for the requested cluster range.")
    rows, fitted = [], []
    sample_size = min(5000, len(profiles))
    rng = np.random.default_rng(random_state)
    sample_idx = np.sort(rng.choice(len(profiles), sample_size, replace=False)) if sample_size < len(profiles) else np.arange(len(profiles))
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
    best_model, best_labels = next(pair for pair in fitted if pair[0].n_clusters == best_k)
    output = profiles.copy()
    output["segment_id"] = best_labels.astype(int)
    summary = output.groupby("segment_id").agg(
        customers=("user_id", "nunique"),
        transactions=("transaction_count", "sum"),
        median_transactions=("transaction_count", "median"),
        median_total_spend=("total_spend", "median"),
        median_avg_ticket=("mean_transaction_value", "median"),
        median_recency_days=("days_since_last_purchase", "median"),
        median_devices=("distinct_device_id", "median") if "distinct_device_id" in output else ("transaction_count", "median"),
    ).reset_index()
    metadata = {"selected_k": best_k, "feature_columns": feature_columns, "random_state": random_state, "selection_metric": "highest sample silhouette, tie-broken by Davies-Bouldin"}
    return output, metrics, {"metadata": metadata, "segment_summary": summary.to_dict(orient="records")}
