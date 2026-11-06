#!/usr/bin/env python3
"""Reproducible fraud-model audit executed on GitHub Actions.

Runs on the repository's committed Fraud_Data.csv and IP map. It records
random-stratified and chronological evaluation separately. Thresholds are
selected on validation data only. The test partitions are evaluated once.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from supervised_fraud import (
    BASE_REQUIRED, TARGET, SEED, FraudFeatureTransformer,
    make_mlp_matrix, train_mlp, predict_mlp, evaluate_model,
    select_threshold, NeuralFraudArtifact, save_artifact
)

warnings.filterwarnings("ignore", category=FutureWarning)
BENCHMARK = os.getenv("BENCHMARK", "both").lower()
OUT_REPORT = ROOT / "reports" / "audit"
OUT_ARTIFACT = ROOT / "artifacts" / "audit"
OUT_REPORT.mkdir(parents=True, exist_ok=True)
OUT_ARTIFACT.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    {"name": "mlp_64_32", "hidden": (64, 32), "dropout": 0.20,
     "lr": 1e-3, "weight_decay": 1e-4, "batch_size": 2048,
     "epochs": 18, "patience": 4},
    {"name": "mlp_128_64", "hidden": (128, 64), "dropout": 0.25,
     "lr": 5e-4, "weight_decay": 1e-4, "batch_size": 2048,
     "epochs": 22, "patience": 5},
]

def dump_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=float) + "\n")

def assert_data(df: pd.DataFrame, ip_map: pd.DataFrame) -> None:
    required = BASE_REQUIRED + [TARGET]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Fraud dataset is missing columns: {missing}")
    if not set(pd.to_numeric(df[TARGET], errors="coerce").dropna().unique()).issubset({0, 1}):
        raise ValueError("Target class must contain only 0 and 1.")
    if df[TARGET].isna().any():
        raise ValueError("Target class contains missing labels.")
    for col in ["lower_bound_ip_address", "upper_bound_ip_address", "country"]:
        if col not in ip_map:
            raise ValueError(f"IP map missing required column: {col}")
    if df[TARGET].nunique() != 2:
        raise ValueError("Both classes must be present.")

def split_data(df: pd.DataFrame, protocol: str):
    if protocol == "random":
        train, temp = train_test_split(
            df, test_size=0.30, stratify=df[TARGET], random_state=SEED
        )
        val, test = train_test_split(
            temp, test_size=0.50, stratify=temp[TARGET], random_state=SEED
        )
    else:
        ordered = df.sort_values("purchase_time", kind="mergesort").reset_index(drop=True)
        n = len(ordered)
        train = ordered.iloc[: int(n * 0.60)].copy()
        val = ordered.iloc[int(n * 0.60): int(n * 0.80)].copy()
        test = ordered.iloc[int(n * 0.80):].copy()
        if train.empty or val.empty or test.empty:
            raise ValueError("Chronological split produced an empty partition.")
    return train.copy(), val.copy(), test.copy()

def split_info(name: str, part: pd.DataFrame) -> dict:
    times = pd.to_datetime(part["purchase_time"], errors="coerce", utc=True)
    return {
        "name": name, "rows": int(len(part)), "fraud": int(part[TARGET].sum()),
        "fraud_rate": float(part[TARGET].mean()),
        "start_time": str(times.min()), "end_time": str(times.max())
    }

def run_protocol(df: pd.DataFrame, ip_map: pd.DataFrame, protocol: str) -> dict:
    started = time.time()
    train, val, test = split_data(df, protocol)
    X_train, y_train = train[BASE_REQUIRED].copy(), train[TARGET].astype(int).to_numpy()
    X_val, y_val = val[BASE_REQUIRED].copy(), val[TARGET].astype(int).to_numpy()
    X_test, y_test = test[BASE_REQUIRED].copy(), test[TARGET].astype(int).to_numpy()

    transformer = FraudFeatureTransformer(ip_map=ip_map)
    F_train = transformer.fit_transform(X_train)
    F_val = transformer.transform(X_val)
    F_test = transformer.transform(X_test)

    Xtr, scaler, low_maps = make_mlp_matrix(F_train, fit=True)
    Xv, _, _ = make_mlp_matrix(F_val, scaler=scaler, maps=low_maps, fit=False)
    Xt, _, _ = make_mlp_matrix(F_test, scaler=scaler, maps=low_maps, fit=False)

    candidates = []
    best = None
    for index, cfg in enumerate(CONFIGS, start=1):
        model, val_ap_from_training, epoch, device = train_mlp(
            Xtr, y_train, Xv, y_val,
            config={k: v for k, v in cfg.items() if k != "name"},
            seed=SEED + index, device="cpu"
        )
        p_val = predict_mlp(model, Xv, device)
        val_metrics = evaluate_model(y_val, p_val, 0.5)
        entry = {
            "model": cfg["name"],
            "validation_pr_auc": float(average_precision_score(y_val, p_val)),
            "validation_roc_auc": float(roc_auc_score(y_val, p_val)),
            "best_epoch": int(epoch),
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items() if k != "name"}
        }
        candidates.append(entry)
        if best is None or entry["validation_pr_auc"] > best["entry"]["validation_pr_auc"]:
            best = {"entry": entry, "model": model, "device": device,
                    "p_val": p_val, "config": cfg}

    threshold = select_threshold(y_val, best["p_val"])
    val_metrics = evaluate_model(y_val, best["p_val"], threshold)
    p_test = predict_mlp(best["model"], Xt, best["device"])
    test_metrics = evaluate_model(y_test, p_test, threshold)

    # Capacity-based top-k ranking complements threshold metrics.
    k = max(1, int(y_test.sum()))
    order = np.argsort(-p_test)
    topk = order[:k]
    precision_at_k = float(y_test[topk].mean())
    base_rate = float(y_test.mean())
    test_metrics["precision_at_k"] = precision_at_k
    test_metrics["recall_at_k"] = float(y_test[topk].sum() / max(int(y_test.sum()), 1))
    test_metrics["enrichment_at_k"] = float(precision_at_k / base_rate) if base_rate else 0.0
    test_metrics["k"] = int(k)
    test_metrics["majority_baseline_accuracy"] = float(max(np.mean(y_test == 0), np.mean(y_test == 1)))

    # Save model artifact with train-only maps/scaler/frequency stats.
    artifact_metadata = {
        "scaler": scaler, "low_maps": low_maps, "device": best["device"],
        "threshold": float(threshold), "config": {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in best["config"].items() if k != "name"
        },
        "best_epoch": int(best["entry"]["best_epoch"]),
        "evaluation_protocol": protocol,
        "training_rows": int(len(train)),
    }
    artifact = NeuralFraudArtifact(transformer, "mlp", best["model"], artifact_metadata)
    model_path = OUT_ARTIFACT / f"{protocol}_neural_model.joblib"
    save_artifact(artifact, model_path)

    pred = test[BASE_REQUIRED + [TARGET]].copy()
    pred["fraud_probability"] = p_test
    pred["predicted_class"] = (p_test >= threshold).astype(int)
    pred = pred.rename(columns={TARGET: "actual_class"})
    pred.to_csv(OUT_ARTIFACT / f"{protocol}_test_predictions.csv", index=False)

    result = {
        "protocol": protocol,
        "seed": SEED,
        "split": [split_info("train", train), split_info("validation", val), split_info("test", test)],
        "candidate_models": candidates,
        "selected_model": best["entry"]["model"],
        "threshold_selected_on_validation": float(threshold),
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "artifacts": {
            "model": str(model_path.relative_to(ROOT)),
            "test_predictions": str((OUT_ARTIFACT / f"{protocol}_test_predictions.csv").relative_to(ROOT))
        },
        "runtime_seconds": round(time.time() - started, 2),
        "warning": (
            "Random split is an IID benchmark only; it must not be interpreted as future-period generalization."
            if protocol == "random" else
            "Chronological holdout tests future-period generalization; score may be lower because the dataset exhibits severe time drift."
        )
    }
    dump_json(OUT_REPORT / f"{protocol}_metrics.json", result)
    return result

def main() -> None:
    data_path = ROOT / "data" / "raw" / "Fraud_Data.csv"
    ip_path = ROOT / "data" / "raw" / "IpAddress_to_Country.csv"
    if not data_path.exists() or not ip_path.exists():
        raise FileNotFoundError("Committed raw data files are required for GitHub Actions retraining.")

    print("Loading dataset...")
    df = pd.read_csv(data_path)
    ip_map = pd.read_csv(ip_path)
    print(f"Loaded fraud rows={len(df):,}, map rows={len(ip_map):,}", flush=True)
    df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce", utc=True)
    df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["purchase_time", "class"]).copy()
    df[TARGET] = df[TARGET].astype(int)
    assert_data(df, ip_map)

    protocols = ["random", "chronological"] if BENCHMARK == "both" else [BENCHMARK]
    if any(p not in {"random", "chronological"} for p in protocols):
        raise ValueError(f"Unsupported BENCHMARK={BENCHMARK!r}")

    results = []
    for protocol in protocols:
        print(f"Running {protocol} benchmark ...")
        result = run_protocol(df, ip_map, protocol)
        results.append(result)
        print(
            f"{protocol}: test PR-AUC={result['test_metrics']['pr_auc']:.6f}, "
            f"ROC-AUC={result['test_metrics']['roc_auc']:.6f}, "
            f"Precision@k={result['test_metrics']['precision_at_k']:.6f}",
            flush=True,
        )

    manifest = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "dataset_rows": int(len(df)),
        "dataset_columns": list(df.columns),
        "fraud_rate": float(df[TARGET].mean()),
        "results": [{"protocol": x["protocol"], "test_metrics": x["test_metrics"],
                     "validation_metrics": x["validation_metrics"],
                     "selected_model": x["selected_model"],
                     "threshold": x["threshold_selected_on_validation"]} for x in results],
        "interpretation": "Compare like-for-like protocols. Do not mix random and chronological scores."
    }
    dump_json(OUT_REPORT / "audit_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    main()
