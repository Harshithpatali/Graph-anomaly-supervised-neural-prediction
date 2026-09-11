"""
Evaluation utilities that treat the fraud label strictly as validation ground truth.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_anomaly_scores(
    y_true: np.ndarray | pd.Series,
    scores: np.ndarray | pd.Series,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute ranking metrics.  Higher score must mean more anomalous.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)

    metrics = {}
    try:
        metrics[f"{prefix}roc_auc"] = float(roc_auc_score(y, s))
    except ValueError:
        metrics[f"{prefix}roc_auc"] = float("nan")

    try:
        metrics[f"{prefix}pr_auc"] = float(average_precision_score(y, s))
    except ValueError:
        metrics[f"{prefix}pr_auc"] = float("nan")

    # Precision / Recall at top-k (k = number of positives)
    n_pos = int(y.sum())
    if n_pos > 0:
        order = np.argsort(-s)
        topk = order[:n_pos]
        metrics[f"{prefix}precision_at_k"] = float(y[topk].mean())
        metrics[f"{prefix}recall_at_k"] = float(y[topk].sum() / n_pos)
    else:
        metrics[f"{prefix}precision_at_k"] = float("nan")
        metrics[f"{prefix}recall_at_k"] = float("nan")

    return metrics


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores)[:k]
    return float(np.asarray(y_true)[order].mean())


def enrichment_factor(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """How many times better than random at top-k."""
    base_rate = np.asarray(y_true).mean()
    if base_rate == 0:
        return float("nan")
    return precision_at_k(y_true, scores, k) / base_rate


def summary_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(results).T.round(4)
