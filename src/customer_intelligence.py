"""Unsupervised customer intelligence features for the Fraud_Data dataset.

No fraud labels are read or used. All outputs are descriptive behavioral signals,
not probabilities of fraud. Customer features are calculated as of a configurable
snapshot timestamp and are intended for segmentation and risk exploration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED = {"user_id", "purchase_time", "purchase_value"}
ENTITY_COLUMNS = ("device_id", "ip_address", "source", "browser", "country")


def _safe_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def build_customer_profiles(
    transactions: pd.DataFrame,
    as_of: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Aggregate transaction behavior into one row per user.

    The optional as_of timestamp makes recency explicit. If omitted, the maximum
    observed purchase timestamp is used. The fraud label is deliberately excluded.
    """
    missing = REQUIRED.difference(transactions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    x = transactions.copy()
    x["purchase_time"] = _safe_datetime(x["purchase_time"])
    x["purchase_value"] = pd.to_numeric(x["purchase_value"], errors="coerce")
    x = x.dropna(subset=["user_id", "purchase_time", "purchase_value"])
    x = x[x["purchase_value"] > 0].copy()
    if x.empty:
        raise ValueError("No valid transactions remain after cleaning.")
    cutoff = _safe_datetime(pd.Series([as_of])).iloc[0] if as_of is not None else x["purchase_time"].max()
    if pd.isna(cutoff):
        raise ValueError("as_of must be a valid timestamp.")
    x = x[x["purchase_time"] <= cutoff].copy()
    if x.empty:
        raise ValueError("No transactions exist at or before the requested as_of timestamp.")

    x["_hour"] = x["purchase_time"].dt.hour
    x["_weekend"] = (x["purchase_time"].dt.dayofweek >= 5).astype("int8")
    x["_night"] = x["_hour"].isin([0, 1, 2, 3, 4, 5, 23]).astype("int8")
    x["_day"] = x["purchase_time"].dt.floor("D")

    agg = x.groupby("user_id", sort=False).agg(
        transaction_count=("purchase_value", "size"),
        total_spend=("purchase_value", "sum"),
        mean_transaction_value=("purchase_value", "mean"),
        median_transaction_value=("purchase_value", "median"),
        std_transaction_value=("purchase_value", "std"),
        max_transaction_value=("purchase_value", "max"),
        min_transaction_value=("purchase_value", "min"),
        first_purchase=("purchase_time", "min"),
        last_purchase=("purchase_time", "max"),
        active_days=("_day", "nunique"),
        night_transaction_rate=("_night", "mean"),
        weekend_transaction_rate=("_weekend", "mean"),
    )
    agg["std_transaction_value"] = agg["std_transaction_value"].fillna(0.0)
    agg["customer_tenure_days"] = (agg["last_purchase"] - agg["first_purchase"]).dt.total_seconds() / 86400
    agg["customer_tenure_days"] = agg["customer_tenure_days"].clip(lower=0)
    agg["days_since_last_purchase"] = (cutoff - agg["last_purchase"]).dt.total_seconds() / 86400
    agg["transactions_per_active_day"] = agg["transaction_count"] / agg["active_days"].clip(lower=1)
    agg["spend_per_active_day"] = agg["total_spend"] / agg["active_days"].clip(lower=1)
    agg["transaction_value_cv"] = agg["std_transaction_value"] / agg["mean_transaction_value"].replace(0, np.nan)
    agg["transaction_value_cv"] = agg["transaction_value_cv"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Within-customer temporal trend: compare earlier and later halves by event time.
    trend_rows = []
    for user, group in x.groupby("user_id", sort=False):
        ordered = group.sort_values("purchase_time")
        split = max(1, len(ordered) // 2)
        early = ordered.iloc[:split]["purchase_value"].mean()
        late = ordered.iloc[split:]["purchase_value"].mean() if len(ordered) > split else early
        trend_rows.append((user, float((late - early) / max(abs(early), 1e-9))))
    trend = pd.DataFrame(trend_rows, columns=["user_id", "spend_change_late_vs_early"]).set_index("user_id")
    agg = agg.join(trend)

    for col in ENTITY_COLUMNS:
        if col in x.columns:
            counts = x.groupby("user_id")[col].nunique(dropna=True)
            agg[f"distinct_{col}"] = counts.reindex(agg.index).fillna(0).astype(float)

    if "age" in x.columns:
        age = pd.to_numeric(x["age"], errors="coerce")
        agg["customer_age"] = x.assign(_age=age).groupby("user_id")["_age"].median().reindex(agg.index)
    agg.index.name = "user_id"
    result = agg.reset_index()
    result["as_of"] = cutoff.isoformat()
    return result.replace([np.inf, -np.inf], np.nan)


def numeric_customer_features(profiles: pd.DataFrame) -> pd.DataFrame:
    """Return stable numeric modeling columns, excluding identifiers/timestamps."""
    excluded = {"user_id", "first_purchase", "last_purchase", "as_of"}
    features = profiles.drop(columns=[c for c in excluded if c in profiles], errors="ignore")
    features = features.select_dtypes(include=[np.number]).copy()
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.fillna(features.median()).fillna(0.0)
