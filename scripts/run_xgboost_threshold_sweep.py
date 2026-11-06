#!/usr/bin/env python3
"""XGBoost-only probability threshold sweep on chronological folds.

Threshold selection is performed on validation data only. Test metrics are
reported for the full predeclared 0.50..1.00 grid without selecting on test.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, confusion_matrix)
from run_expanded_fraud_research import DATA, OUT, CANDIDATES, SEED, make_model, fit_candidate
from run_walk_forward_benchmark import build_folds, make_features

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = np.round(np.arange(0.50, 1.001, 0.01), 2)

def metrics(y, p, threshold):
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "review_rate": float(pred.mean()),
        "predicted_positive": int(pred.sum()),
        "true_positive": int(tp), "false_positive": int(fp),
        "false_negative": int(fn), "true_negative": int(tn),
    }

def main():
    started = time.time()
    df = pd.read_csv(DATA)
    df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce", utc=True)
    df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["purchase_time", "class"]).sort_values("purchase_time", kind="mergesort").reset_index(drop=True)
    df["class"] = df["class"].astype(int)
    xgb_candidates = [c for c in CANDIDATES if c["name"] == "xgboost"]
    all_rows, fold_reports = [], []
    for fold, train_end, valid_end, test_end in build_folds(len(df)):
        train, valid, test = (df.iloc[:train_end].copy(), df.iloc[train_end:valid_end].copy(),
                              df.iloc[valid_end:test_end].copy())
        Xtr, Xv, Xt, _, _ = make_features(train, valid, test)
        graph_cols = [c for c in Xtr.columns if c.startswith("graph_svd_")]
        Xtr, Xv, Xt = (x.drop(columns=graph_cols) for x in (Xtr, Xv, Xt))
        ytr, yv, yt = (x["class"].to_numpy() for x in (train, valid, test))
        candidates_scored = []
        for j, candidate in enumerate(xgb_candidates):
            model = make_model(candidate, SEED + fold * 100 + j)
            fit_candidate(model, "xgboost", Xtr, ytr, Xv, yv)
            pv = model.predict_proba(Xv)[:,1]
            candidates_scored.append((average_precision_score(yv, pv), candidate))
        best_val_ap, chosen = max(candidates_scored, key=lambda z: z[0])
        model = make_model(chosen, SEED + fold * 1000 + 77)
        fit_candidate(model, "xgboost", Xtr, ytr, Xv, yv)
        pv, pt = model.predict_proba(Xv)[:,1], model.predict_proba(Xt)[:,1]
        val_rows = [metrics(yv, pv, t) for t in THRESHOLDS]
        test_rows = [metrics(yt, pt, t) for t in THRESHOLDS]
        # Threshold is chosen by validation F1; test is used only once for
        # reporting the transferred operating point and the fixed grid.
        best_idx = max(range(len(val_rows)), key=lambda i: (val_rows[i]["f1"], val_rows[i]["precision"], -val_rows[i]["threshold"]))
        selected = float(THRESHOLDS[best_idx])
        transferred = metrics(yt, pt, selected)
        for v, te in zip(val_rows, test_rows):
            all_rows.append({
                "fold": fold, "model": "xgboost", "config": chosen["config"],
                "validation_pr_auc": float(best_val_ap),
                "validation_threshold_selected_by_f1": selected,
                "validation_threshold_metrics": v,
                "test_threshold_metrics": te,
                "is_validation_selected_threshold": bool(v["threshold"] == selected),
            })
        fold_reports.append({
            "fold": fold, "selected_config": chosen["config"],
            "validation_pr_auc": float(best_val_ap),
            "test_pr_auc": float(average_precision_score(yt, pt)),
            "test_roc_auc": float(roc_auc_score(yt, pt)),
            "test_prevalence": float(yt.mean()),
            "validation_selected_threshold": selected,
            "test_metrics_at_transferred_threshold": transferred,
            "validation_metrics_at_selected_threshold": val_rows[best_idx],
            "train_rows": len(train), "validation_rows": len(valid), "test_rows": len(test),
            "validation_start": str(valid.purchase_time.min()), "validation_end": str(valid.purchase_time.max()),
            "test_start": str(test.purchase_time.min()), "test_end": str(test.purchase_time.max()),
        })
        print(f"fold={fold} XGBoost PR-AUC={fold_reports[-1]['test_pr_auc']:.5f} "
              f"validation-selected threshold={selected:.2f} "
              f"test precision={transferred['precision']:.4f} recall={transferred['recall']:.4f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    flat = []
    for row in all_rows:
        flat.append({
            "fold": row["fold"], "model": row["model"], "config": row["config"],
            "validation_pr_auc": row["validation_pr_auc"],
            "threshold": row["validation_threshold_metrics"]["threshold"],
            **{f"validation_{k}":v for k,v in row["validation_threshold_metrics"].items() if k != "threshold"},
            **{f"test_{k}":v for k,v in row["test_threshold_metrics"].items() if k != "threshold"},
            "validation_selected_threshold": row["validation_threshold_selected_by_f1"],
            "is_validation_selected_threshold": row["is_validation_selected_threshold"],
        })
    pd.DataFrame(flat).to_csv(OUT / "xgboost_threshold_sweep.csv", index=False)
    report = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_sha": os.getenv("GITHUB_SHA","local"),
        "model_scope": "XGBoost only; no MLP, LightGBM or CatBoost training",
        "threshold_grid": [float(t) for t in THRESHOLDS],
        "selection_protocol": "For each chronological fold, choose the threshold maximizing validation F1 (tie-break: higher precision, then lower threshold). Apply that threshold once to the subsequent test window. The full fixed threshold grid is also reported on test for descriptive comparison; do not select a threshold from test results.",
        "fold_results": fold_reports,
        "mean_test_pr_auc": float(np.mean([r["test_pr_auc"] for r in fold_reports])),
        "mean_test_roc_auc": float(np.mean([r["test_roc_auc"] for r in fold_reports])),
        "mean_test_precision_at_transferred_threshold": float(np.mean([r["test_metrics_at_transferred_threshold"]["precision"] for r in fold_reports])),
        "mean_test_recall_at_transferred_threshold": float(np.mean([r["test_metrics_at_transferred_threshold"]["recall"] for r in fold_reports])),
        "runtime_seconds": round(time.time()-started,2),
    }
    (OUT / "xgboost_threshold_sweep.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2), flush=True)

if __name__ == "__main__":
    main()
