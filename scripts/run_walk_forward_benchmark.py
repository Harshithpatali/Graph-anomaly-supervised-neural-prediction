#!/usr/bin/env python3
"""Walk-forward benchmark: causal graph features + CatBoost/XGBoost/MLP."""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from causal_graph_features import build_causal_graph_features
from graph_embedding import TransactionGraphSVD
from supervised_fraud import train_mlp, predict_mlp

warnings.filterwarnings("ignore")
SEED = 42
DATA = ROOT / "data/raw/Fraud_Data.csv"
OUT = ROOT / "reports/research"
ART = ROOT / "artifacts/research"
OUT.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)
CAPACITIES = (0.01, 0.02, 0.05)


def metric_at_capacity(y, p, fraction):
    k = max(1, int(np.ceil(len(y) * fraction)))
    idx = np.argsort(-p)[:k]
    tp = int(np.asarray(y)[idx].sum())
    positives = max(int(np.asarray(y).sum()), 1)
    precision = tp / k
    recall = tp / positives
    return {"queue_fraction": fraction, "queue_size": k, "precision": precision,
            "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12)}


def capacity_threshold(y, p, fraction=0.02):
    k = max(1, int(np.ceil(len(p) * fraction)))
    return float(np.sort(p)[-k])


def psi(expected, actual, bins=10):
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, bins=edges)[0] / len(expected)
    a = np.histogram(actual, bins=edges)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def make_features(train, valid, test):
    raw_cols = ["user_id", "device_id", "ip_address", "signup_time", "purchase_time",
                "purchase_value", "source", "browser", "sex", "age"]
    causal = build_causal_graph_features(pd.concat([train, valid, test], ignore_index=True))
    n_train, n_val = len(train), len(valid)
    a = causal.iloc[:n_train].copy()
    b = causal.iloc[n_train:n_train+n_val].copy()
    c = causal.iloc[n_train+n_val:].copy()

    # The graph SVD vocabulary and components are learned from this fold's
    # training window only; future transactions are projected inductively.
    graph = TransactionGraphSVD(n_components=12, random_state=SEED)
    ga = graph.fit_transform(a)
    gb = graph.transform(b)
    gc = graph.transform(c)
    for j in range(ga.shape[1]):
        a[f"graph_svd_{j:02d}"] = ga[:, j]
        b[f"graph_svd_{j:02d}"] = gb[:, j]
        c[f"graph_svd_{j:02d}"] = gc[:, j]

    drop = {"class", "user_id", "device_id", "ip_address", "signup_time", "purchase_time"}
    for frame in (a, b, c):
        for col in ["source", "browser", "sex"]:
            if col in frame:
                frame[col] = frame[col].fillna("Unknown").astype(str)
        frame.drop(columns=[col for col in drop if col in frame], inplace=True)

    # Fit category vocabulary and numeric imputation on training only.
    train_dummies = pd.get_dummies(a, columns=["source", "browser", "sex"], dummy_na=False)
    val_dummies = pd.get_dummies(b, columns=["source", "browser", "sex"], dummy_na=False)
    test_dummies = pd.get_dummies(c, columns=["source", "browser", "sex"], dummy_na=False)
    val_dummies = val_dummies.reindex(columns=train_dummies.columns, fill_value=0)
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)
    medians = train_dummies.median(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0)
    train_dummies = train_dummies.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0)
    val_dummies = val_dummies.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0)
    test_dummies = test_dummies.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0)
    return (train_dummies.astype(np.float32), val_dummies.astype(np.float32),
            test_dummies.astype(np.float32), graph, medians)


def build_folds(n):
    # Three expanding-window folds. The final 10% of the timeline remains a
    # test set in the last fold; earlier folds provide stability diagnostics.
    # Non-overlapping test windows. Each later fold may use the prior
    # fold's matured validation/test history as training data.
    specs = [(0.50, 0.60, 0.73), (0.60, 0.73, 0.86), (0.73, 0.86, 1.00)]
    for i, (tr, va, te) in enumerate(specs, 1):
        yield i, int(n * tr), int(n * va), int(n * te)


def run():
    started = time.time()
    df = pd.read_csv(DATA)
    df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce", utc=True)
    df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["purchase_time", "class"]).sort_values("purchase_time", kind="mergesort").reset_index(drop=True)
    df["class"] = df["class"].astype(int)
    results = []

    for fold, train_end, val_end, test_end in build_folds(len(df)):
        train = df.iloc[:train_end].copy()
        valid = df.iloc[train_end:val_end].copy()
        test = df.iloc[val_end:test_end].copy()
        if train["class"].nunique() < 2 or valid["class"].nunique() < 2 or test["class"].nunique() < 2:
            raise ValueError(f"Fold {fold} does not contain both classes in every partition")

        Xtr, Xv, Xt, graph, medians = make_features(train, valid, test)
        ytr, yv, yt = train["class"].to_numpy(), valid["class"].to_numpy(), test["class"].to_numpy()
        scaler = StandardScaler()
        Xtr_scaled = scaler.fit_transform(Xtr).astype(np.float32)
        Xv_scaled = scaler.transform(Xv).astype(np.float32)
        Xt_scaled = scaler.transform(Xt).astype(np.float32)

        candidates = {}
        xgb = XGBClassifier(
            n_estimators=350, max_depth=5, learning_rate=0.04,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=5,
            min_child_weight=5, objective="binary:logistic",
            eval_metric="aucpr", tree_method="hist", n_jobs=2,
            random_state=SEED + fold
        )
        xgb.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
        candidates["xgboost"] = (xgb, xgb.predict_proba(Xv)[:, 1], lambda z: xgb.predict_proba(z)[:, 1])

        cat = CatBoostClassifier(
            iterations=450, depth=7, learning_rate=0.04,
            loss_function="Logloss", eval_metric="PRAUC",
            auto_class_weights="Balanced", l2_leaf_reg=6,
            random_seed=SEED + fold, thread_count=2, verbose=False,
            allow_writing_files=False
        )
        cat.fit(Xtr, ytr, eval_set=(Xv, yv), early_stopping_rounds=60, verbose=False)
        candidates["catboost"] = (cat, cat.predict_proba(Xv)[:, 1], lambda z: cat.predict_proba(z)[:, 1])

        mlp_cfg = {"hidden": (128, 64), "dropout": 0.25, "lr": 5e-4,
                   "weight_decay": 1e-4, "batch_size": 2048, "epochs": 16, "patience": 4}
        mlp, _, epoch, device = train_mlp(
            Xtr_scaled, ytr, Xv_scaled, yv, config=mlp_cfg,
            seed=SEED + fold, device="cpu"
        )
        candidates["mlp"] = (mlp, predict_mlp(mlp, Xv_scaled, device),
                            lambda z: predict_mlp(mlp, z, device))

        fold_result = {
            "fold": fold,
            "train": {"rows": len(train), "fraud_rate": float(ytr.mean()),
                      "start": str(train.purchase_time.min()), "end": str(train.purchase_time.max())},
            "validation": {"rows": len(valid), "fraud_rate": float(yv.mean()),
                           "start": str(valid.purchase_time.min()), "end": str(valid.purchase_time.max())},
            "test": {"rows": len(test), "fraud_rate": float(yt.mean()),
                     "start": str(test.purchase_time.min()), "end": str(test.purchase_time.max())},
            "models": {}
        }

        for name, (model, p_val, predict_fn) in candidates.items():
            threshold = capacity_threshold(yv, p_val, 0.02)
            p_test = predict_fn(Xt_scaled if name == "mlp" else Xt)
            metrics = {
                "pr_auc": float(average_precision_score(yt, p_test)),
                "roc_auc": float(roc_auc_score(yt, p_test)),
                "threshold_at_validation_2pct_capacity": threshold,
                "precision_at_validation_threshold": float(((p_test >= threshold) & (yt == 1)).sum() / max((p_test >= threshold).sum(), 1)),
                "recall_at_validation_threshold": float(((p_test >= threshold) & (yt == 1)).sum() / max(yt.sum(), 1)),
                "capacity_metrics": [metric_at_capacity(yt, p_test, cap) for cap in CAPACITIES],
            }
            fold_result["models"][name] = metrics
            if fold == 3:
                joblib.dump({"model": model, "scaler": scaler if name == "mlp" else None,
                             "graph_svd": graph, "feature_columns": list(Xtr.columns),
                             "medians": medians, "threshold": threshold},
                            ART / f"{name}_fold3.joblib")

        # Feature drift PSI, evaluated without using labels.
        drift_cols = [c for c in Xtr.columns if c in Xt.columns][:30]
        fold_result["feature_drift_psi"] = {
            col: psi(Xtr[col], Xt[col]) for col in drift_cols
        }
        max_psi = max(fold_result["feature_drift_psi"].values(), default=0.0)
        fold_result["drift_status"] = (
            "high" if max_psi >= 0.30 else "moderate" if max_psi >= 0.20 else "low"
        )
        results.append(fold_result)
        print(f"fold={fold} rows={len(train):,}/{len(valid):,}/{len(test):,} complete", flush=True)

    # Aggregate mean test performance over three genuinely future windows.
    aggregate = {}
    for model_name in ["xgboost", "catboost", "mlp"]:
        aggregate[model_name] = {
            metric: float(np.mean([r["models"][model_name][metric] for r in results]))
            for metric in ["pr_auc", "roc_auc", "precision_at_validation_threshold",
                           "recall_at_validation_threshold"]
        }
        aggregate[model_name]["capacity_metrics"] = {
            str(cap): {
                key: float(np.mean([
                    next(m for m in r["models"][model_name]["capacity_metrics"]
                         if m["queue_fraction"] == cap)[key] for r in results
                ]))
                for key in ["precision", "recall", "f1"]
            } for cap in CAPACITIES
        }

    report = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "dataset_rows": int(len(df)),
        "fraud_rate": float(df["class"].mean()),
        "protocol": "three expanding-window walk-forward folds; train-only graph SVD; causal event-time features",
        "models": aggregate,
        "folds": results,
        "runtime_seconds": round(time.time() - started, 2),
        "interpretation": "Select models by stability across future windows and queue-capacity metrics. Never use the final fold for tuning."
    }
    (OUT / "walk_forward_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pd.DataFrame([
        {"fold": r["fold"], "model": model, **{k: v for k, v in metrics.items() if isinstance(v, (int, float))}}
        for r in results for model, metrics in r["models"].items()
    ]).to_csv(OUT / "walk_forward_summary.csv", index=False)
    print(json.dumps({"models": aggregate, "runtime_seconds": report["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    run()
