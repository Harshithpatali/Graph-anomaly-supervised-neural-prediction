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

        for window in ("7D", "30D"):
            out = np.zeros(len(timeline), dtype=np.float32)
            # Iterate over entity groups to ensure time windows never cross entities.
            for _, idx in timeline.groupby(entity, sort=False).groups.items():
                g = timeline.loc[idx].sort_values("purchase_time")
                series = g.set_index("purchase_time")["events"]
                rolled = series.rolling(window, closed="left").sum()
                out[timeline.index.get_indexer(g.index)] = rolled.fillna(0).to_numpy(dtype=np.float32)
            timeline[f"{prefix}_events_prior_{window.lower()}"] = out

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
        join = x[[shared_entity, "purchase_time"]].merge(
            events, left_on=[shared_entity, "purchase_time"],
            right_on=[shared_entity, "first_seen"], how="left", sort=False
        )
        # Sum only links whose first appearance precedes this event timestamp.
        prior = np.zeros(len(x), dtype=np.float32)
        for key, idx in x.groupby(shared_entity, sort=False, dropna=True).groups.items():
            e = events[events[shared_entity] == key]
            if e.empty:
                continue
            times = e["first_seen"].astype("int64").to_numpy()
            cumulative = e["new_links"].cumsum().to_numpy()
            row_times = x.loc[idx, "purchase_time"].astype("int64").to_numpy()
            positions = np.searchsorted(times, row_times, side="left") - 1
            valid_pos = positions >= 0
            vals = np.zeros(len(idx), dtype=np.float32)
            vals[valid_pos] = cumulative[positions[valid_pos]]
            prior[x.index.get_indexer(idx)] = vals
        x[name] = prior

    return x.sort_values("_row_order").drop(columns="_row_order").reset_index(drop=True)
