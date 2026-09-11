"""
High-level feature engineering combining behavioural, temporal and graph signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from temporal_features import add_temporal_features, add_velocity_features
from graph_utils import entity_degree_features, shared_entity_features


def add_behavioral_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core behavioural + frequency features used across notebooks.
    """
    x = df.copy()
    if "user_id" in x.columns:
        vc = x["user_id"].value_counts()
        x["user_transaction_count"] = x["user_id"].map(vc).astype(float)
    if "device_id" in x.columns:
        vc = x["device_id"].value_counts()
        x["device_transaction_count"] = x["device_id"].map(vc).astype(float)
    if "ip_address" in x.columns:
        vc = x["ip_address"].value_counts()
        x["ip_transaction_count"] = x["ip_address"].map(vc).astype(float)
    if "purchase_value" in x.columns:
        x["purchase_value_log1p"] = np.log1p(x["purchase_value"].clip(lower=0))
    return x


ID_COLS = ["user_id", "device_id", "ip_address", "signup_time", "purchase_time", "purchase_day"]


def build_feature_matrix(
    df: pd.DataFrame,
    categorical_cols: list | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Produce a numeric feature matrix ready for unsupervised models.
    Returns (feature_df, feature_names).
    """
    x = df.copy()
    x = add_temporal_features(x)
    x = add_behavioral_features(x)
    x = add_velocity_features(x)

    # Graph-style features
    deg = entity_degree_features(x)
    shared = shared_entity_features(x)
    x = pd.concat([x, deg, shared], axis=1)

    # Simple one-hot for low-cardinality categoricals
    if categorical_cols is None:
        categorical_cols = [c for c in ["source", "browser", "sex", "country"] if c in x.columns]

    cat_dummies = []
    for col in categorical_cols:
        dummies = pd.get_dummies(x[col].astype(str), prefix=col, drop_first=True)
        cat_dummies.append(dummies)
        x = x.drop(columns=[col], errors="ignore")

    if cat_dummies:
        x = pd.concat([x] + cat_dummies, axis=1)

    # Drop raw identifiers and timestamps
    x = x.drop(columns=[c for c in ID_COLS if c in x.columns], errors="ignore")

    # Select numeric only
    numeric = x.select_dtypes(include=[np.number]).copy()
    # Drop the raw label if present – never used for training
    if "class" in numeric.columns:
        numeric = numeric.drop(columns=["class"])

    # Fill remaining NaNs with median
    numeric = numeric.fillna(numeric.median(numeric_only=True))
    # Replace inf
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(0)

    feature_names = numeric.columns.tolist()
    return numeric, feature_names


FEATURE_CANDIDATES = [
    "purchase_value",
    "purchase_value_log1p",
    "age",
    "account_age_hours",
    "account_age_seconds",
    "instant_purchase",
    "same_day_purchase",
    "purchase_hour",
    "purchase_dayofweek",
    "is_weekend",
    "is_night",
    "user_transaction_count",
    "device_transaction_count",
    "ip_transaction_count",
    "user_degree",
    "device_degree",
    "ip_degree",
    "users_on_device",
    "users_on_ip",
    "ips_on_device",
    "user_time_since_prev",
    "device_time_since_prev",
    "ip_time_since_prev",
]
