#!/usr/bin/env python3
"""Tune on earlier validation windows; calibrate and evaluate the latest test once."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_walk_forward_benchmark import DATA, OUT, ART, build_folds, make_features, metric_at_capacity

SEED = 42
CAPACITIES = (0.01, 0.02, 0.05)
TUNING_GRID = [
    {"n_estimators": 250, "max_depth": 3, "learning_rate": 0.03, "min_child_weight": 5, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 5},
    {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03, "min_child_weight": 8, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 8},
    {"n_estimators": 350, "max_depth": 5, "learning_rate": 0.04, "min_child_weight": 5, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 5},
    {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.025, "min_child_weight": 10, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 10},
]


def _safe_probs(values):
    return np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)


def expected_calibration_error(y, p, bins=10):
    y = np.asarray(y, dtype=int)
    p = _safe_probs(p)
    edges = np.linspace(0, 1, bins + 1)
    total = len(y)
    error = 0.0
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if not mask.any():
            continue
        confidence = float(p[mask].mean())
        observed = float(y[mask].mean())
        weight = float(mask.sum() / max(total, 1))
        error += weight * abs(confidence - observed)
        rows.append({"bin_left": float(lo), "bin_right": float(hi),
                     "count": int(mask.sum()), "mean_probability": confidence,
                     "observed_rate": observed})
    return float(error), rows


def psi_numeric(reference, current, bins=10, epsilon=1e-6):
    """Quantile-binned numeric PSI with explicit under/overflow buckets."""
    a = pd.to_numeric(pd.Series(reference), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    b = pd.to_numeric(pd.Series(current), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if not len(a) or not len(b):
        return None
    quantiles = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(quantiles) <= 1:
        return 0.0
    inner = quantiles[1:-1]
    edges = np.concatenate(([-np.inf], inner, [np.inf]))
    ea = np.histogram(a, bins=edges)[0].astype(float) / len(a)
    eb = np.histogram(b, bins=edges)[0].astype(float) / len(b)
    ea = np.clip(ea, epsilon, None)
    eb = np.clip(eb, epsilon, None)
    return float(np.sum((eb - ea) * np.log(eb / ea)))


def psi_categorical(reference, current, epsilon=1e-6):
    """Categorical PSI over the union of categories, including missing values."""
    a = pd.Series(reference).astype("string").fillna("__MISSING__")
    b = pd.Series(current).astype("string").fillna("__MISSING__")
    if a.empty or b.empty:
        return None
    categories = sorted(set(a.unique()).union(set(b.unique())))
    pa = a.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(float)
    pb = b.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy(float)
    pa = np.clip(pa, epsilon, None)
    pb = np.clip(pb, epsilon, None)
    return float(np.sum((pb - pa) * np.log(pb / pa)))


def calculate_psi(reference, current, column):
    # Calendar fields and low-cardinality integer features are categorical;
    # graph SVD coordinates are deliberately handled by separate diagnostics.
    categorical_names = {"event_hour", "event_dow", "event_month", "event_is_weekend"}
    s = pd.Series(reference)
    low_cardinality = s.nunique(dropna=True) <= 20
    if column in categorical_names or not pd.api.types.is_numeric_dtype(s) or low_cardinality:
        return {"method": "categorical", "psi": psi_categorical(reference, current)}
    return {"method": "numeric_quantile", "psi": psi_numeric(reference, current)}


def graph_embedding_diagnostics(train, future, graph):
    diagnostics = {}
    for col in ("user_id", "device_id", "ip_address"):
        known = set(train[col].dropna().astype(str))
        values = future[col].dropna().astype(str)
        diagnostics[f"{col}_unseen_rate"] = float((~values.isin(known)).mean()) if len(values) else 0.0
    z = graph.transform(future)
    diagnostics["future_zero_embedding_rate"] = float(np.all(np.isclose(z, 0), axis=1).mean())
    diagnostics["train_svd_explained_variance_ratio"] = [
        float(v) for v in getattr(graph.svd_, "explained_variance_ratio_", [])
    ]
    diagnostics["train_svd_total_explained_variance_ratio"] = float(
        np.sum(getattr(graph.svd_, "explained_variance_ratio_", []))
    )
    diagnostics["future_component_mean"] = [float(v) for v in z.mean(axis=0)]
    diagnostics["future_component_std"] = [float(v) for v in z.std(axis=0)]
    return diagnostics


def build_model(params, seed):
    return XGBClassifier(
        **params, objective="binary:logistic", eval_metric="aucpr",
        tree_method="hist", n_jobs=2, random_state=seed,
        scale_pos_weight=1.0
    )


def main():
    started = time.time()
    df = pd.read_csv(DATA)
    df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce", utc=True)
    df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["purchase_time", "class"]).sort_values("purchase_time", kind="mergesort").reset_index(drop=True)
    df["class"] = df["class"].astype(int)
    fold_bounds = list(build_folds(len(df)))
    tuning_rows = []

    # Only folds 1 and 2 validation partitions participate in model selection.
    for fold, train_end, val_end, _test_end in fold_bounds[:2]:
        train, valid = df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy()
        Xtr, Xv, _unused, _graph, _medians = make_features(train, valid, df.iloc[val_end:val_end].copy())
        ytr, yv = train["class"].to_numpy(), valid["class"].to_numpy()
        for config_id, params in enumerate(TUNING_GRID, 1):
            model = build_model(params, SEED + fold * 100 + config_id)
            model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
            p = model.predict_proba(Xv)[:, 1]
            tuning_rows.append({"fold": fold, "config_id": config_id,
                                "validation_pr_auc": float(average_precision_score(yv, p)),
                                "validation_roc_auc": float(roc_auc_score(yv, p)),
                                "params": params})
            print(f"tuning fold={fold} config={config_id} PR-AUC={tuning_rows[-1]['validation_pr_auc']:.5f}", flush=True)

    scores = {}
    for row in tuning_rows:
        scores.setdefault(row["config_id"], []).append(row["validation_pr_auc"])
    selected_id = max(scores, key=lambda k: (float(np.mean(scores[k])), -k))
    selected_params = TUNING_GRID[selected_id - 1]

    # Final fold is reserved: fit once, calibrate on its validation segment,
    # then score its test segment once. No test-driven model/threshold choices.
    fold, train_end, val_end, test_end = fold_bounds[2]
    train = df.iloc[:train_end].copy()
    valid = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:test_end].copy()
    Xtr, Xv, Xt, graph, medians = make_features(train, valid, test)
    ytr, yv, yt = train["class"].to_numpy(), valid["class"].to_numpy(), test["class"].to_numpy()
    model = build_model(selected_params, SEED + 300)
    model.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
    p_val_raw = _safe_probs(model.predict_proba(Xv)[:, 1])
    p_test_raw = _safe_probs(model.predict_proba(Xt)[:, 1])

    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(p_val_raw, yv)
    p_val_cal = _safe_probs(calibrator.predict(p_val_raw))
    p_test_cal = _safe_probs(calibrator.predict(p_test_raw))

    threshold_rows = {}
    for capacity in CAPACITIES:
        k = max(1, int(np.ceil(len(p_val_cal) * capacity)))
        threshold = float(np.sort(p_val_cal)[-k])
        threshold_rows[str(capacity)] = {
            "validation_threshold": threshold,
            "validation_metrics": metric_at_capacity(yv, p_val_cal, capacity),
            "test_threshold_metrics": {
                "precision": float(((p_test_cal >= threshold) & (yt == 1)).sum() / max((p_test_cal >= threshold).sum(), 1)),
                "recall": float(((p_test_cal >= threshold) & (yt == 1)).sum() / max(int(yt.sum()), 1)),
                "review_rate": float((p_test_cal >= threshold).mean()),
            },
            "test_top_k_metrics": metric_at_capacity(yt, p_test_cal, capacity),
        }

    ece_raw, bins_raw = expected_calibration_error(yv, p_val_raw)
    ece_cal, bins_cal = expected_calibration_error(yv, p_val_cal)
    feature_drift = {}
    for col in Xtr.columns:
        if col.startswith("graph_svd_"):
            continue
        feature_drift[col] = calculate_psi(Xtr[col], Xt[col], col)
    embedding = graph_embedding_diagnostics(train, test, graph)
    report = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "dataset_rows": int(len(df)),
        "fraud_rate": float(df["class"].mean()),
        "protocol": "config selection on folds 1-2 validation only; fold 3 untouched until one final evaluation; isotonic calibration fitted on fold 3 validation",
        "selected_config_id": selected_id,
        "selected_params": selected_params,
        "tuning": tuning_rows,
        "tuning_config_mean_validation_pr_auc": {str(k): float(np.mean(v)) for k, v in scores.items()},
        "final_fold": {
            "fold": fold,
            "train": {"rows": len(train), "fraud_rate": float(ytr.mean()), "start": str(train.purchase_time.min()), "end": str(train.purchase_time.max())},
            "validation": {"rows": len(valid), "fraud_rate": float(yv.mean()), "start": str(valid.purchase_time.min()), "end": str(valid.purchase_time.max())},
            "test": {"rows": len(test), "fraud_rate": float(yt.mean()), "start": str(test.purchase_time.min()), "end": str(test.purchase_time.max())},
            "metrics": {
                "raw": {"pr_auc": float(average_precision_score(yt, p_test_raw)), "roc_auc": float(roc_auc_score(yt, p_test_raw)), "brier": float(brier_score_loss(yt, p_test_raw))},
                "isotonic_calibrated": {"pr_auc": float(average_precision_score(yt, p_test_cal)), "roc_auc": float(roc_auc_score(yt, p_test_cal)), "brier": float(brier_score_loss(yt, p_test_cal))},
                "validation_calibration": {"raw_brier": float(brier_score_loss(yv, p_val_raw)), "calibrated_brier": float(brier_score_loss(yv, p_val_cal)), "raw_ece": ece_raw, "calibrated_ece": ece_cal, "raw_bins": bins_raw, "calibrated_bins": bins_cal},
                "capacity_thresholds": threshold_rows,
            },
            "feature_drift": feature_drift,
            "graph_embedding_diagnostics": embedding,
        },
        "runtime_seconds": round(time.time() - started, 2),
        "interpretation": "The final test is reported once after all model and calibration choices are frozen. Calibration quality is assessed on validation; future test metrics are a one-shot generalization estimate.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    (OUT / "calibrated_temporal_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(tuning_rows).drop(columns=["params"]).to_csv(OUT / "tuning_validation_summary.csv", index=False)
    pd.DataFrame([{"feature": k, **v} for k, v in feature_drift.items()]).to_csv(OUT / "final_feature_psi.csv", index=False)
    joblib.dump({"model": model, "calibrator": calibrator, "graph_svd": graph,
                 "feature_columns": list(Xtr.columns), "medians": medians,
                 "selected_params": selected_params, "validation_thresholds": threshold_rows},
                ART / "calibrated_xgboost_final_fold.joblib")
    pd.DataFrame({"actual": yt, "probability_raw": p_test_raw,
                  "probability_calibrated": p_test_cal}).to_csv(OUT / "final_fold_predictions.csv", index=False)
    print(json.dumps({"selected_config_id": selected_id,
                      "tuning_mean_pr_auc": report["tuning_config_mean_validation_pr_auc"],
                      "final_test": report["final_fold"]["metrics"]["isotonic_calibrated"],
                      "runtime_seconds": report["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
