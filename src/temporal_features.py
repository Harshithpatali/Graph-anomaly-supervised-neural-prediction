"""
Temporal feature engineering for e-commerce fraud / anomaly detection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add account age, hour-of-day, day-of-week, and related temporal signals.
    """
    x = df.copy()
    for c in ["signup_time", "purchase_time"]:
        if c in x.columns and not pd.api.types.is_datetime64_any_dtype(x[c]):
            x[c] = pd.to_datetime(x[c], errors="coerce", utc=True)

    if {"signup_time", "purchase_time"} <= set(x.columns):
        delta = x["purchase_time"] - x["signup_time"]
        x["account_age_hours"] = delta.dt.total_seconds() / 3600.0
        x["account_age_seconds"] = delta.dt.total_seconds()
        # Very short time between signup and purchase is a classic fraud signal
        x["instant_purchase"] = (x["account_age_seconds"] <= 1).astype(int)
        x["same_day_purchase"] = (x["account_age_hours"] <= 24).astype(int)

    if "purchase_time" in x.columns:
        x["purchase_hour"] = x["purchase_time"].dt.hour
        x["purchase_dayofweek"] = x["purchase_time"].dt.dayofweek
        x["purchase_day"] = x["purchase_time"].dt.floor("D")
        x["is_weekend"] = x["purchase_dayofweek"].isin([5, 6]).astype(int)
        x["is_night"] = x["purchase_hour"].isin(list(range(0, 6)) + [23]).astype(int)

    return x


def add_velocity_features(df: pd.DataFrame, time_col: str = "purchase_time") -> pd.DataFrame:
    """
    Per-entity velocity / frequency features computed chronologically.
    These are computed on the full frame for simplicity; for production
    streaming they would be maintained as rolling state.
    """
    x = df.copy()
    if time_col not in x.columns:
        return x

    x = x.sort_values(time_col)

    for entity, prefix in [
        ("user_id", "user"),
        ("device_id", "device"),
        ("ip_address", "ip"),
    ]:
        if entity not in x.columns:
            continue
        # Transaction count (global for this snapshot)
        counts = x[entity].value_counts()
        x[f"{prefix}_tx_count"] = x[entity].map(counts).astype(float)

        # Time since previous transaction for same entity
        x[f"{prefix}_time_since_prev"] = (
            x.groupby(entity)[time_col].diff().dt.total_seconds()
        )
        # Fill first occurrence with large value (no previous)
        x[f"{prefix}_time_since_prev"] = x[f"{prefix}_time_since_prev"].fillna(1e7)

    return x
