"""Core smoke tests and temporal feature regression tests."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_load_raw():
    from data_utils import load_raw
    fraud, ipmap = load_raw()
    assert fraud.shape[0] > 100000
    assert "class" in fraud.columns
    assert "ip_address" in fraud.columns


def test_feature_matrix():
    from data_utils import load_raw, parse_timestamps, enrich_with_country, basic_clean
    from graph_features import build_feature_matrix
    fraud, ipmap = load_raw()
    fraud = basic_clean(fraud.head(1000))
    fraud = parse_timestamps(fraud)
    fraud = enrich_with_country(fraud, ipmap)
    X, names = build_feature_matrix(fraud)
    assert X.shape[0] == 1000
    assert "class" not in names
    assert "user_id" not in names
    assert len(names) > 10


def test_anomaly_model():
    from data_utils import load_raw, parse_timestamps, enrich_with_country, basic_clean
    from graph_features import build_feature_matrix
    from anomaly import fit_isolation_forest
    fraud, ipmap = load_raw()
    fraud = basic_clean(fraud.head(2000))
    fraud = parse_timestamps(fraud)
    fraud = enrich_with_country(fraud, ipmap)
    X, _ = build_feature_matrix(fraud)
    model = fit_isolation_forest(X, contamination=0.1, n_estimators=50)
    scores = model.predict_anomaly_score(X)
    assert len(scores) == len(X)
    assert scores.std() > 0


def test_historical_velocity_uses_only_strictly_prior_events():
    from temporal_features import add_historical_velocity_features

    frame = pd.DataFrame({
        "user_id": [1, 1, 1, 2],
        "device_id": ["d1", "d1", "d1", "d2"],
        "ip_address": [100, 100, 100, 200],
        "purchase_time": [
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:00Z",
            "2024-01-01T01:00:00Z",
            "2024-01-01T00:30:00Z",
        ],
    })
    out = add_historical_velocity_features(frame)
    assert out["user_tx_count_prior"].tolist() == [0.0, 0.0, 2.0, 0.0]
    assert out["device_tx_count_prior"].tolist() == [0.0, 0.0, 2.0, 0.0]
    assert out.loc[2, "user_time_since_prev_seconds"] == 3600.0
    assert out.loc[0, "user_time_since_prev_seconds"] == 1e7


def test_historical_velocity_preserves_original_row_order_and_does_not_depend_on_future_rows():
    from temporal_features import add_historical_velocity_features

    base = pd.DataFrame({
        "user_id": [1, 1],
        "device_id": ["d1", "d1"],
        "ip_address": [100, 100],
        "purchase_time": [
            "2024-01-01T00:00:00Z",
            "2024-01-01T01:00:00Z",
        ],
    })
    extended = pd.concat([
        base,
        pd.DataFrame({
            "user_id": [1], "device_id": ["d1"], "ip_address": [100],
            "purchase_time": ["2024-01-02T00:00:00Z"],
        })
    ], ignore_index=True)

    base_out = add_historical_velocity_features(base)
    extended_out = add_historical_velocity_features(extended)
    pd.testing.assert_series_equal(
        base_out["user_tx_count_prior"],
        extended_out.loc[:1, "user_tx_count_prior"],
        check_names=False,
    )
    assert base_out["user_tx_count_prior"].tolist() == [0.0, 1.0]
    assert extended_out.loc[2, "user_tx_count_prior"] == 2.0
