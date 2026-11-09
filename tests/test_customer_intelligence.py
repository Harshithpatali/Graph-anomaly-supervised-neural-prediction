import pandas as pd
from src.customer_intelligence import build_customer_profiles
from src.customer_risk import build_behavioral_risk
from src.customer_segmentation import segment_customers
from src.unsupervised_anomalies import score_anomalies


def sample():
    return pd.DataFrame([{'user_id': u, 'purchase_time': f'2024-01-{d:02d}T10:00:00Z', 'purchase_value': float(10+u*d), 'device_id': f'd{u%4}', 'ip_address': f'ip{u%3}', 'age': 25+u%10, 'class': u%2} for u in range(1, 16) for d in range(1, 4)])


def test_profiles_exclude_label_and_are_user_level():
    profiles = build_customer_profiles(sample())
    assert profiles.user_id.is_unique and len(profiles) == 15
    assert 'class' not in profiles.columns
    assert profiles.transaction_count.eq(3).all()


def test_behavioral_risk_is_not_probability():
    risk = build_behavioral_risk(build_customer_profiles(sample()))
    assert risk.behavioral_risk_index.between(0, 100).all()
    assert risk.risk_explanations.notna().all()


def test_segmentation_quality_metrics():
    segmented, metrics, meta = segment_customers(build_customer_profiles(sample()), cluster_range=range(2, 4))
    assert segmented.segment_id.nunique() in (2, 3)
    assert {'silhouette_sample', 'davies_bouldin'}.issubset(metrics.columns)


def test_unsupervised_anomaly_ranking():
    scored, meta = score_anomalies(build_customer_profiles(sample()))
    assert scored.anomaly_rank.min() == 1
    assert scored.anomaly_score.between(0, 1).all()
    assert 'not a fraud probability' in meta['warning']


def test_anomaly_model_excludes_heuristic_risk_index():
    profiles = build_behavioral_risk(build_customer_profiles(sample()))
    scored, meta = score_anomalies(profiles)
    assert "behavioral_risk_index" not in meta["feature_columns"]
