"""Causal event-time transaction graph features.

Every feature for a transaction at time t is computed from events strictly
before t. Same-timestamp transactions are simultaneous and cannot leak into
one another. Labels are never used in feature construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


ENTITY_COLUMNS = {
    "user_id": "user",
    "device_id": "device",
    "ip_address": "ip",
}


def build_causal_graph_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"purchase_time", *ENTITY_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns for causal graph features: {sorted(missing)}")

    x = frame.copy()
    x["_row_order"] = np.arange(len(x))
    x["purchase_time"] = pd.to_datetime(x["purchase_time"], errors="coerce", utc=True)
    if x["purchase_time"].isna().any():
        raise ValueError("purchase_time must be valid before causal graph feature building")

    # Transaction-level calendar/amount features.
    ts = x["purchase_time"]
    x["event_hour"] = ts.dt.hour.astype("int16")
    x["event_dow"] = ts.dt.dayofweek.astype("int8")
    x["event_month"] = ts.dt.month.astype("int8")
    x["event_is_weekend"] = (ts.dt.dayofweek >= 5).astype("int8")
    value = pd.to_numeric(x.get("purchase_value", 0), errors="coerce")
    x["purchase_value_log1p"] = np.log1p(value.clip(lower=0))

    # Account age is observed at transaction time, not inferred from future rows.
    if {"signup_time", "purchase_time"}.issubset(x.columns):
        signup = pd.to_datetime(x["signup_time"], errors="coerce", utc=True)
        age = (ts - signup).dt.total_seconds() / 3600
        x["account_age_hours_causal"] = age.where(age >= 0)

    # Entity timelines aggregate simultaneous transactions before shifting.
    for entity, prefix in ENTITY_COLUMNS.items():
        timeline = (
            x.loc[x[entity].notna(), [entity, "purchase_time"]]
             .groupby([entity, "purchase_time"], sort=False)
             .size().rename("events").reset_index()
             .sort_values([entity, "purchase_time"], kind="mergesort")
        )
        by_entity = timeline.groupby(entity, sort=False)
        timeline[f"{prefix}_events_prior"] = (
            by_entity["events"].cumsum() - timeline["events"]
        ).astype("float32")
        timeline[f"{prefix}_seconds_since_prior"] = (
            by_entity["purchase_time"].diff().dt.total_seconds().fillna(1e7)
        ).astype("float32")

        # Vectorized grouped time rolling; the left-closed boundary excludes
        # all transactions sharing the current timestamp.
        for window in ("7D", "30D"):
            rolled = (
                timeline.groupby(entity, sort=False)
                        .rolling(window, on="purchase_time", closed="left")["events"]
                        .sum()
            )
            keys = pd.MultiIndex.from_frame(timeline[[entity, "purchase_time"]])
            timeline[f"{prefix}_events_prior_{window.lower()}"] = (
                rolled.reindex(keys).fillna(0).to_numpy(dtype=np.float32)
            )

        lookup_cols = [
            entity, "purchase_time", f"{prefix}_events_prior",
            f"{prefix}_seconds_since_prior", f"{prefix}_events_prior_7d",
            f"{prefix}_events_prior_30d",
        ]
        keys = x[[entity, "purchase_time"]].copy()
        aligned = keys.merge(timeline[lookup_cols], on=[entity, "purchase_time"],
                             how="left", sort=False, validate="many_to_one")
        for col in lookup_cols[2:]:
            x[col] = aligned[col].fillna(0 if "events" in col else 1e7).to_numpy()

    # Historical cross-entity degree: prior users seen on this device/IP.
    for shared_entity, counterpart, name in [
        ("device_id", "user_id", "device_prior_distinct_users"),
        ("ip_address", "user_id", "ip_prior_distinct_users"),
        ("user_id", "device_id", "user_prior_distinct_devices"),
        ("user_id", "ip_address", "user_prior_distinct_ips"),
    ]:
        valid = x[[shared_entity, counterpart, "purchase_time"]].dropna()
        first_seen = (
            valid.groupby([shared_entity, counterpart], sort=False)["purchase_time"]
                 .min().rename("first_seen").reset_index()
        )
        events = (
            first_seen.groupby([shared_entity, "first_seen"], sort=False)
                      .size().rename("new_links").reset_index()
                      .sort_values([shared_entity, "first_seen"], kind="mergesort")
        )
        events["prior_links"] = (
            events.groupby(shared_entity, sort=False)["new_links"].cumsum()
            - events["new_links"]
        )
        # As-of join returns the latest link-creation timestamp strictly
        # before each transaction; sorting by event time is required by pandas.
        left = x[[shared_entity, "purchase_time"]].copy()
        left["_row"] = np.arange(len(left))
        right = events[[shared_entity, "first_seen", "new_links"]].copy()
        right["prior_links"] = right.groupby(shared_entity, sort=False)["new_links"].cumsum()
        aligned = pd.merge_asof(
            left.sort_values("purchase_time"),
            right.sort_values("first_seen"),
            left_on="purchase_time", right_on="first_seen",
            by=shared_entity, direction="backward", allow_exact_matches=False
        ).sort_values("_row")
        x[name] = aligned["prior_links"].fillna(0).to_numpy(dtype=np.float32)

    return x.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)
