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
    Calculate entity counts and time-since-previous using only prior rows.

    A stable chronological sort is used. Ties at the same timestamp are treated
    as simultaneous: all rows at that timestamp see events strictly earlier
    than the timestamp. The counts therefore do not depend on row ordering
    among simultaneous transactions.
    """
    x = df.copy()
    if time_col not in x.columns:
        return x

    x[time_col] = pd.to_datetime(x[time_col], errors="coerce", utc=True)
    x["_original_order"] = np.arange(len(x))
    x = x.sort_values([time_col, "_original_order"], kind="mergesort")

    for entity, prefix in [
        ("user_id", "user"),
        ("device_id", "device"),
        ("ip_address", "ip"),
    ]:
        if entity not in x.columns:
            continue

        # Count strictly earlier timestamps, not the current batch's future rows.
        x[f"{prefix}_tx_count_prior"] = (
            x.groupby(entity, dropna=False)[time_col]
             .transform(lambda s: s.groupby(s).transform("size").cumsum() - s.groupby(s).transform("size"))
        ).astype(float)

        previous_time = x.groupby(entity, dropna=False)[time_col].transform(
            lambda s: s.groupby(s).transform("first").shift()
        )
        # The groupby-on-time formulation above can be ambiguous for duplicated
        # timestamps. Recompute exact prior timestamp via per-entity unique times.
        unique_times = (
            x.loc[x[entity].notna(), [entity, time_col]]
             .drop_duplicates()
             .sort_values([entity, time_col], kind="mergesort")
        )
        unique_times[f"{prefix}_previous_time"] = unique_times.groupby(entity, dropna=False)[time_col].shift()
        previous_lookup = unique_times.set_index([entity, time_col])[f"{prefix}_previous_time"]
        keys = pd.MultiIndex.from_arrays([x[entity], x[time_col]])
        previous = previous_lookup.reindex(keys).to_numpy()
        x[f"{prefix}_time_since_prev_seconds"] = (
            (x[time_col].to_numpy() - previous) / np.timedelta64(1, "s")
        )
        x[f"{prefix}_time_since_prev_seconds"] = (
            pd.to_numeric(x[f"{prefix}_time_since_prev_seconds"], errors="coerce")
              .fillna(1e7)
        )

    x = x.sort_values("_original_order", kind="mergesort").drop(columns="_original_order")
    return x


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
