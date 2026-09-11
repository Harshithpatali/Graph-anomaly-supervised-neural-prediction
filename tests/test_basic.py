"""Basic smoke tests for the project library."""
import sys
from pathlib import Path

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
