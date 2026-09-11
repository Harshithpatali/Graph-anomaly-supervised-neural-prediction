"""
Inference pipeline used by the Streamlit app.
Artifacts are produced only after the analytical notebooks have been executed
and the final model has been frozen.
"""
from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


def load_artifacts():
    model_path = MODELS_DIR / "final_model.joblib"
    config_path = MODELS_DIR / "inference_config.json"
    if not model_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            "Final artifacts do not exist yet. Complete the analytical notebooks "
            "and final evaluation before deployment. Run notebooks 01→11 first."
        )
    model = joblib.load(model_path)
    cfg = json.loads(config_path.read_text())
    return model, cfg


def score_new_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a new batch of transactions.
    Expects the same raw columns as Fraud_Data.csv (minus class).
    """
    from graph_features import build_feature_matrix
    from data_utils import parse_timestamps, enrich_with_country, load_raw

    model, cfg = load_artifacts()
    features = cfg["features"]

    # Minimal enrichment
    x = parse_timestamps(df)
    try:
        _, ipmap = load_raw()
        x = enrich_with_country(x, ipmap)
    except Exception:
        x["country"] = "Unknown"

    feat_df, _ = build_feature_matrix(x)
    # Align columns
    for c in features:
        if c not in feat_df.columns:
            feat_df[c] = 0.0
    feat_df = feat_df[features]

    scores = model.predict_anomaly_score(feat_df)
    result = df.copy()
    result["anomaly_score"] = scores
    result["anomaly_rank"] = (
        result["anomaly_score"].rank(method="first", ascending=False).astype(int)
    )
    return result.sort_values("anomaly_score", ascending=False)
