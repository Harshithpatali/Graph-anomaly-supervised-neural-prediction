"""Regression tests for supervised fraud features and evaluation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supervised_fraud import FraudFeatureTransformer, evaluate_model, select_threshold


@pytest.fixture
def sample_transactions():
    return pd.DataFrame({
        "user_id": [1, 2, 1, 3],
        "signup_time": [
            "2024-01-01 00:00:00", "2024-01-01 00:00:00",
            "2024-01-02 00:00:00", "2024-01-01 00:00:00"
        ],
        "purchase_time": [
            "2024-01-01 00:01:00", "2024-01-01 01:00:00",
            "2024-01-02 00:01:00", "2024-01-01 02:00:00"
        ],
        "purchase_value": [10.0, 20.0, 30.0, 40.0],
        "device_id": ["d1", "d2", "d1", "d3"],
        "source": ["SEO", "Ads", "SEO", None],
        "browser": ["Chrome", "Safari", "Chrome", "Firefox"],
        "sex": ["M", "F", "M", "F"],
        "age": [20, 30, 20, 40],
        "ip_address": [100.0, 200.0, 100.0, 300.0],
    })


def test_transformer_frequencies_are_fit_only_and_batch_invariant(sample_transactions):
    train = sample_transactions.iloc[:3].copy()
    transformer = FraudFeatureTransformer().fit(train)
    row = sample_transactions.iloc[[3]].copy()
    first = transformer.transform(row)
    augmented = pd.concat([row, row.assign(user_id=999, device_id="new", ip_address=9999)], ignore_index=True)
    second = transformer.transform(augmented).iloc[[0]]
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
    assert first.iloc[0]["device_frequency"] == 0


def test_transformer_handles_negative_account_age_and_missing_categories(sample_transactions):
    transformer = FraudFeatureTransformer().fit(sample_transactions.iloc[:3])
    bad = sample_transactions.iloc[[0]].copy()
    bad["signup_time"] = "2025-01-01 00:00:00"
    bad["source"] = None
    features = transformer.transform(bad)
    assert pd.isna(features.iloc[0]["account_age_hours"])
    assert features.iloc[0]["source"] == "Unknown"


def test_transformer_outputs_expected_feature_contract(sample_transactions):
    transformer = FraudFeatureTransformer().fit(sample_transactions)
    features = transformer.transform(sample_transactions)
    assert set(["account_age_hours", "user_frequency", "device_frequency",
                "ip_frequency", "ip_country"]).issubset(features.columns)
    assert len(features) == len(sample_transactions)


def test_evaluation_metrics_and_threshold():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold = select_threshold(y, p)
    metrics = evaluate_model(y, p, threshold)
    assert metrics["pr_auc"] == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert 0.0 <= threshold <= 1.0
    assert metrics["tp"] + metrics["tn"] + metrics["fp"] + metrics["fn"] == len(y)
