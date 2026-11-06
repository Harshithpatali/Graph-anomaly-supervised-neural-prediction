"""Explainable behavioral risk indicators; never a fraud probability."""
from __future__ import annotations
import numpy as np
import pandas as pd


def build_behavioral_risk(profiles: pd.DataFrame) -> pd.DataFrame:
    """Create a transparent 0-100 relative review-priority index.

    Components are cross-sectional percentile ranks. Weights are explicit
    heuristics and must be calibrated with investigator feedback before use.
    """
    x = profiles.copy()

    def pct(col: str) -> pd.Series:
        if col not in x:
            return pd.Series(0.0, index=x.index)
        values = pd.to_numeric(x[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        return values.rank(pct=True, method="average").fillna(0.0)

    components = {
        "activity_intensity": pct("transactions_per_active_day"),
        "spend_deviation": pct("transaction_value_cv"),
        "spend_change": pct("spend_change_late_vs_early").abs(),
        "device_diversity": pct("distinct_device_id"),
        "ip_diversity": pct("distinct_ip_address"),
        "recent_activity": 1.0 - pct("days_since_last_purchase"),
    }
    weights = {"activity_intensity": .20, "spend_deviation": .20, "spend_change": .20,
               "device_diversity": .15, "ip_diversity": .15, "recent_activity": .10}
    score = sum(components[k] * weights[k] for k in weights) * 100.0
    x["behavioral_risk_index"] = score.round(2)
    x["behavioral_risk_band"] = pd.cut(score, bins=[-.01, 40, 70, 100.01],
                                       labels=["Low", "Medium", "High"]).astype(str)
    explanations = []
    for i in range(len(x)):
        flags = [label for label, key in [
            ("high activity intensity", "activity_intensity"),
            ("unusual transaction-value variability", "spend_deviation"),
            ("large spending change", "spend_change"),
            ("multiple devices", "device_diversity"),
            ("multiple IP addresses", "ip_diversity"),
        ] if components[key].iloc[i] >= .90]
        explanations.append("; ".join(flags) or "No extreme behavioral component among selected indicators")
    x["risk_explanations"] = explanations
    return x
