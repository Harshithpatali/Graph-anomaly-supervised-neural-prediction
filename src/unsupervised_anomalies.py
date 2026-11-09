"""Unsupervised transaction/customer anomaly scoring."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler


def score_anomalies(frame: pd.DataFrame, feature_columns: list[str] | None = None,
                    contamination: float = .02, random_state: int = 42):
    if len(frame) < 10:
        raise ValueError("At least 10 rows are required for anomaly scoring.")
    excluded = {"user_id", "first_purchase", "last_purchase", "as_of", "class",
                "anomaly_score", "anomaly_rank", "segment_id", "behavioral_risk_index"}
    if feature_columns is None:
        feature_columns = [c for c in frame.select_dtypes(include=[np.number]).columns if c not in excluded]
    if not feature_columns:
        raise ValueError("No numeric features available.")
    missing = set(feature_columns).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing anomaly features: {sorted(missing)}")
    X = frame[feature_columns].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median()).fillna(0.0)
    Z = RobustScaler().fit_transform(X)
    iso = IsolationForest(n_estimators=300, contamination=contamination,
                          random_state=random_state, n_jobs=-1).fit(Z)
    iso_score = -iso.score_samples(Z)
    fit_n = min(30000, len(Z))
    fit_idx = np.linspace(0, len(Z) - 1, fit_n, dtype=int)
    lof = LocalOutlierFactor(n_neighbors=min(35, len(fit_idx) - 1),
                             contamination=contamination, novelty=True, n_jobs=-1).fit(Z[fit_idx])
    lof_score = -lof.score_samples(Z)

    def percentile(values):
        return pd.Series(values).rank(pct=True, method="average").to_numpy()

    combined = .60 * percentile(iso_score) + .40 * percentile(lof_score)
    result = frame.copy()
    result["isolation_forest_score"] = iso_score
    result["lof_score"] = lof_score
    result["anomaly_score"] = combined
    # Use the combined ranking, not fixed score cutoffs. The weighted average
    # of two percentile scores is NOT itself uniformly distributed; applying
    # 0.90/0.98 score thresholds can therefore produce misleading bucket sizes.
    result["anomaly_rank"] = result["anomaly_score"].rank(
        ascending=False, method="first"
    ).astype(int)
    n = len(result)
    extreme_n = max(1, int(np.ceil(0.02 * n)))
    review_n = max(extreme_n, int(np.ceil(0.10 * n)))
    result["anomaly_band"] = np.select(
        [result["anomaly_rank"] <= extreme_n, result["anomaly_rank"] <= review_n],
        ["Extreme", "Unusual"],
        default="Typical"
    )
    band_counts = result["anomaly_band"].value_counts().to_dict()
    return result.sort_values(["anomaly_rank", "anomaly_score"], ascending=[True, False]), {
        "rows_scored": int(n), "feature_columns": feature_columns,
        "contamination_parameter": contamination,
        "band_policy": "Top 2% Extreme; next 8% Unusual; remaining 90% Typical (rank-based, deterministic tie break).",
        "anomaly_band_counts": {k: int(v) for k, v in band_counts.items()},
        "anomaly_band_shares": {k: round(float(v) / n, 4) for k, v in band_counts.items()},
        "score_definition": "0.60 * IsolationForest percentile + 0.40 * LOF percentile; bands use combined rank, not raw score cutoffs.",
        "warning": "Relative unsupervised ranking; not a fraud probability or confirmed fraud label."
    }
