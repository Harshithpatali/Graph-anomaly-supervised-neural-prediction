"""
Classical unsupervised anomaly detectors and hybrid scoring.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.covariance import EllipticEnvelope


class AnomalyModel:
    """
    Thin wrapper that stores a fitted detector + scaler and exposes
    a consistent predict_anomaly_score interface (higher = more anomalous).
    """

    def __init__(self, name: str, detector, scaler=None):
        self.name = name
        self.detector = detector
        self.scaler = scaler
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, feature_names: list[str] | None = None):
        self.feature_names_ = feature_names or (
            list(X.columns) if hasattr(X, "columns") else None
        )
        X_arr = np.asarray(X, dtype=float)
        if self.scaler is not None:
            X_arr = self.scaler.fit_transform(X_arr)
        self.detector.fit(X_arr)
        return self

    def predict_anomaly_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=float)
        if self.scaler is not None:
            X_arr = self.scaler.transform(X_arr)

        if hasattr(self.detector, "decision_function"):
            # IsolationForest / EllipticEnvelope: higher decision_function = more normal
            raw = self.detector.decision_function(X_arr)
            scores = -raw  # invert so higher = more anomalous
        elif hasattr(self.detector, "score_samples"):
            raw = self.detector.score_samples(X_arr)
            scores = -raw
        else:
            # LOF novelty=False returns -1 / 1; negative_outlier_factor_ is available after fit
            # For predict we use decision_function if available, else fall back
            scores = -self.detector.decision_function(X_arr)
        return scores

    def predict_label(self, X: pd.DataFrame | np.ndarray, threshold: float | None = None) -> np.ndarray:
        scores = self.predict_anomaly_score(X)
        if threshold is None:
            # Use the contamination-based prediction if available
            if hasattr(self.detector, "predict"):
                X_arr = np.asarray(X, dtype=float)
                if self.scaler is not None:
                    X_arr = self.scaler.transform(X_arr)
                pred = self.detector.predict(X_arr)
                return (pred == -1).astype(int)
            threshold = np.percentile(scores, 90)
        return (scores >= threshold).astype(int)


def fit_isolation_forest(
    X: pd.DataFrame,
    contamination: float = 0.05,
    n_estimators: int = 200,
    random_state: int = 42,
) -> AnomalyModel:
    scaler = RobustScaler()
    clf = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
        max_samples="auto",
    )
    model = AnomalyModel("IsolationForest", clf, scaler)
    model.fit(X, feature_names=list(X.columns))
    return model


def fit_lof(
    X: pd.DataFrame,
    n_neighbors: int = 20,
    contamination: float = 0.05,
) -> AnomalyModel:
    # LOF does not support predict on new data well when novelty=False;
    # we fit with novelty=True for scoring new points.
    scaler = RobustScaler()
    clf = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
        n_jobs=-1,
    )
    model = AnomalyModel("LOF", clf, scaler)
    model.fit(X, feature_names=list(X.columns))
    return model


def fit_elliptic_envelope(
    X: pd.DataFrame,
    contamination: float = 0.05,
    random_state: int = 42,
) -> AnomalyModel:
    scaler = RobustScaler()
    clf = EllipticEnvelope(
        contamination=contamination,
        random_state=random_state,
        support_fraction=0.8,
    )
    model = AnomalyModel("EllipticEnvelope", clf, scaler)
    model.fit(X, feature_names=list(X.columns))
    return model


def statistical_zscore_scores(X: pd.DataFrame) -> np.ndarray:
    """Simple multivariate z-score magnitude (robust)."""
    med = X.median()
    mad = (X - med).abs().median().replace(0, 1e-9)
    z = (X - med) / (1.4826 * mad)
    return np.sqrt((z ** 2).sum(axis=1)).values


def hybrid_score(
    scores_dict: Dict[str, np.ndarray],
    weights: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Rank-average hybrid of multiple anomaly scores.
    Higher final score = more anomalous.
    """
    ranks = {}
    for name, s in scores_dict.items():
        # rank 1 = most anomalous
        ranks[name] = pd.Series(s).rank(ascending=False, method="average").values

    if weights is None:
        weights = {k: 1.0 for k in ranks}

    total_w = sum(weights.get(k, 1.0) for k in ranks)
    combined = np.zeros(len(next(iter(ranks.values()))))
    for name, r in ranks.items():
        w = weights.get(name, 1.0) / total_w
        combined += w * r
    # Invert so higher = more anomalous (lower average rank)
    return -combined
