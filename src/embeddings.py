"""
Graph embedding helpers.  Node2Vec is optional; if unavailable we fall back
to simple degree + community proxies.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def node2vec_available() -> bool:
    try:
        import node2vec  # noqa: F401
        return True
    except ImportError:
        return False


def compute_simple_graph_embedding_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight surrogate for graph embeddings:
    - log-degree of each entity
    - pairwise co-occurrence intensity
    These are already partly covered by graph_features; this function exists
    for explicit comparison experiments.
    """
    feats = pd.DataFrame(index=df.index)
    for col, name in [("user_id", "user"), ("device_id", "device"), ("ip_address", "ip")]:
        if col in df.columns:
            deg = df[col].map(df[col].value_counts()).astype(float)
            feats[f"{name}_log_degree"] = np.log1p(deg)
        else:
            feats[f"{name}_log_degree"] = 0.0
    return feats
