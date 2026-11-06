"""Regression tests for event-time graph features and inductive embeddings."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _events():
    return pd.DataFrame({
        "user_id": [1, 1, 2, 1],
        "device_id": ["d1", "d1", "d1", "d2"],
        "ip_address": [10, 10, 10, 11],
        "signup_time": [
            "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z", "2024-01-01T00:00:00Z",
        ],
        "purchase_time": [
            "2024-01-01T01:00:00Z", "2024-01-01T01:00:00Z",
            "2024-01-01T02:00:00Z", "2024-01-02T01:00:00Z",
        ],
        "purchase_value": [10, 20, 30, 40],
    })


def test_causal_features_exclude_same_timestamp_and_future_events():
    from causal_graph_features import build_causal_graph_features

    base = _events().iloc[:2].copy()
    extended = pd.concat([
        base,
        pd.DataFrame({
            "user_id": [1], "device_id": ["d1"], "ip_address": [10],
            "signup_time": ["2024-01-01T00:00:00Z"],
            "purchase_time": ["2024-01-03T00:00:00Z"],
            "purchase_value": [50],
        })
    ], ignore_index=True)
    a = build_causal_graph_features(base)
    b = build_causal_graph_features(extended)
    assert a["user_events_prior"].tolist() == [0.0, 0.0]
    assert b.loc[:1, "user_events_prior"].tolist() == [0.0, 0.0]
    assert b.loc[2, "user_events_prior"] == 2.0


def test_causal_features_count_prior_events_and_rolling_windows():
    from causal_graph_features import build_causal_graph_features

    out = build_causal_graph_features(_events())
    assert out.loc[2, "device_events_prior"] == 2.0
    assert out.loc[2, "device_events_prior_7d"] == 2.0
    assert out.loc[3, "user_events_prior"] == 2.0
    assert out.loc[3, "user_seconds_since_prior"] == 86400.0


def test_graph_svd_is_inductive_for_unseen_entities():
    from graph_embedding import TransactionGraphSVD

    train = _events().iloc[:3]
    future = pd.DataFrame({
        "user_id": [999], "device_id": ["new-device"], "ip_address": [9999]
    })
    model = TransactionGraphSVD(n_components=4, random_state=7)
    train_embedding = model.fit_transform(train)
    future_embedding = model.transform(future)
    assert train_embedding.shape == (3, 4)
    assert future_embedding.shape == (1, 4)
    assert np.isfinite(future_embedding).all()

def test_temporal_edge_and_neighborhood_features_are_causal():
    from causal_graph_features import build_causal_graph_features

    base = _events().iloc[:3].copy()
    extended = pd.concat([
        base,
        pd.DataFrame({
            "user_id": [1], "device_id": ["d1"], "ip_address": [10],
            "signup_time": ["2024-01-01T00:00:00Z"],
            "purchase_time": ["2024-01-04T00:00:00Z"],
            "purchase_value": [55],
        })
    ], ignore_index=True)
    a = build_causal_graph_features(base)
    b = build_causal_graph_features(extended)
    assert a.loc[2, "device_ip_events_prior"] == 2.0
    assert b.loc[:2, "device_ip_events_prior"].tolist() == a["device_ip_events_prior"].tolist()
    assert b.loc[2, "device_ip_novelty"] == 0
    assert b.loc[2, "device_ip_neighborhood_degree"] >= 0
    assert np.isfinite(b.select_dtypes(include=[np.number]).to_numpy()).all()
