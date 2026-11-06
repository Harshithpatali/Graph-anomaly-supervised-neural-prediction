"""
Temporal feature engineering for e-commerce fraud / anomaly detection.

The historical batch helpers are retained for descriptive graph exploration.
They are not suitable as pre-transaction features for future-period validation:
full-frame entity totals include events occurring after individual rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar fields and account age for each transaction."""
    x = df.copy()
    for c in ["signup_time", "purchase_time"]:
        if c in x.columns and not pd.api.types.is_datetime64_any_dtype(x[c]):
            x[c] = pd.to_datetime(x[c], errors="coerce", utc=True)

    if {"signup_time", "purchase_time"} <= set(x.columns):
        delta = x["purchase_time"] - x["signup_time"]
        hours = delta.dt.total_seconds() / 3600.0
        x["account_age_hours"] = hours.where(hours >= 0)
        x["account_age_seconds"] = delta.dt.total_seconds().where(hours >= 0)
        x["instant_purchase"] = (x["account_age_seconds"] <= 1).astype(int)
        x["same_day_purchase"] = (x["account_age_hours"] <= 24).astype(int)

    if "purchase_time" in x.columns:
        x["purchase_hour"] = x["purchase_time"].dt.hour
        x["purchase_dayofweek"] = x["purchase_time"].dt.dayofweek
        x["purchase_day"] = x["purchase_time"].dt.floor("D")
        x["is_weekend"] = x["purchase_dayofweek"].isin([5, 6]).astype(int)
        x["is_night"] = x["purchase_hour"].isin(list(range(0, 6)) + [23]).astype(int)

    return x


def add_historical_velocity_features(
    df: pd.DataFrame,
    time_col: str = "purchase_time",
) -> pd.DataFrame:
    """
    Calculate entity counts and time-since-previous using strictly prior events.

    Rows sharing an entity and timestamp are treated as simultaneous. They
    receive the same prior count and the same time gap from the previous
    distinct timestamp; events at the current timestamp are not counted yet.
    """
    x = df.copy()
    if time_col not in x.columns:
        return x

    x[time_col] = pd.to_datetime(x[time_col], errors="coerce", utc=True)
    original_order = np.arange(len(x))
    x["_original_order"] = original_order

    entities = [
        ("user_id", "user"),
        ("device_id", "device"),
        ("ip_address", "ip"),
    ]
    for entity, prefix in entities:
        if entity not in x.columns:
            continue

        # Unique entity/timestamp rows ensure simultaneous transactions share
        # identical history. The shift excludes the current timestamp group.
        timeline = (
            x.loc[x[entity].notna() & x[time_col].notna(), [entity, time_col]]
             .groupby([entity, time_col], dropna=False)
             .size()
             .rename("events_at_time")
             .reset_index()
             .sort_values([entity, time_col], kind="mergesort")
        )
        grouped = timeline.groupby(entity, sort=False, dropna=False)
        timeline[f"{prefix}_prior_count"] = (
            grouped["events_at_time"].cumsum() - timeline["events_at_time"]
        ).astype(float)
        timeline[f"{prefix}_previous_time"] = grouped[time_col].shift()
        timeline[f"{prefix}_time_since_prev_seconds"] = (
            timeline[time_col] - timeline[f"{prefix}_previous_time"]
        ).dt.total_seconds().fillna(1e7)

        lookup = timeline.set_index([entity, time_col])
        keys = pd.MultiIndex.from_arrays([x[entity], x[time_col]])
        aligned = lookup.reindex(keys)
        x[f"{prefix}_tx_count_prior"] = (
            aligned[f"{prefix}_prior_count"].fillna(0).to_numpy(dtype=float)
        )
        x[f"{prefix}_time_since_prev_seconds"] = (
            aligned[f"{prefix}_time_since_prev_seconds"].fillna(1e7).to_numpy(dtype=float)
        )

    return x.sort_values("_original_order", kind="mergesort").drop(columns="_original_order")


def add_velocity_features(
    df: pd.DataFrame,
    time_col: str = "purchase_time",
) -> pd.DataFrame:
    """
    Backwards-compatible alias for historical velocity features.

    Unlike the legacy full-frame counts, values are based strictly on records
    at earlier timestamps. Prefer add_historical_velocity_features explicitly.
    """
    return add_historical_velocity_features(df, time_col=time_col)
