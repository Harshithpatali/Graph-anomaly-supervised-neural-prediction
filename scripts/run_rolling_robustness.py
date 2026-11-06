#!/usr/bin/env python3
"""Rolling temporal robustness, uncertainty, calibration and queue evaluation."""
from __future__ import annotations
import json, os, time, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, log_loss
from run_expanded_fraud_research import (
    DATA, OUT, CANDIDATES, SEED, make_model, fit_candidate, topk, CAPACITIES
)
from run_walk_forward_benchmark import build_folds, make_features
from supervised_fraud import train_mlp, predict_mlp
from sklearn.preprocessing import StandardScaler

BOOTSTRAPS = int(os.getenv("ROBUSTNESS_BOOTSTRAPS", "1000"))
RNG = np.random.default_rng(20261106)

def ece(y, p, bins=10):
    y, p = np.asarray(y, int), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & ((p < edges[i+1]) if i < bins-1 else (p <= edges[i+1]))
        if mask.any():
            total += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)

def calibration_slope_intercept(y, p):
    from sklearn.linear_model import LogisticRegression
    eps = 1e-6
    z = np.log(np.clip(p, eps, 1-eps) / (1-np.clip(p, eps, 1-eps))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(z, y)
    return {"intercept": float(lr.intercept_[0]), "slope": float(lr.coef_[0, 0])}

def bootstrap_ci(y, p, metric, n=BOOTSTRAPS):
    y, p = np.asarray(y, int), np.asarray(p, float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) < 2 or len(neg) < 2:
        return {"lower_95": None, "upper_95": None, "valid_replicates": 0}
    vals = []
    for _ in range(n):
        idx = np.concatenate([RNG.choice(pos, len(pos), replace=True),
                              RNG.choice(neg, len(neg), replace=True)])
        yy, pp = y[idx], p[idx]
        if metric == "pr_auc":
            vals.append(average_precision_score(yy, pp))
        elif metric == "precision_at_2pct":
            k = max(1, int(np.ceil(len(yy) * .02)))
            vals.append(float(yy[np.argsort(-pp, kind="mergesort")[:k]].mean()))
    if not vals:
        return {"lower_95": None, "upper_95": None, "valid_replicates": 0}
    lo, hi = np.quantile(vals, [.025, .975])
    return {"lower_95": float(lo), "upper_95": float(hi), "valid_replicates": len(vals)}

def main():
    started = time.time()
    df = pd.read_csv(DATA)
    df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce", utc=True)
    df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["purchase_time", "class"]).sort_values("purchase_time", kind="mergesort").reset_index(drop=True)
    df["class"] = df["class"].astype(int)
    rows, details = [], []
    for fold, train_end, valid_end, test_end in build_folds(len(df)):
        train, valid, test = (df.iloc[:train_end].copy(), df.iloc[train_end:valid_end].copy(),
                              df.iloc[valid_end:test_end].copy())
        if any(x["class"].nunique() < 2 for x in (train, valid, test)):
            raise ValueError(f"Fold {fold}: each partition must contain both classes")
        Xtr, Xv, Xt, _, _ = make_features(train, valid, test)
        drop = [c for c in Xtr.columns if c.startswith("graph_svd_")]
        Xtr, Xv, Xt = (x.drop(columns=drop) for x in (Xtr, Xv, Xt))
        ytr, yv, yt = (x["class"].to_numpy() for x in (train, valid, test))
        for family in ("xgboost", "lightgbm"):
            options = [c for c in CANDIDATES if c["name"] == family]
            scored = []
            for j, candidate in enumerate(options):
                m = make_model(candidate, SEED + fold * 100 + j)
                fit_candidate(m, family, Xtr, ytr, Xv, yv)
                pv = m.predict_proba(Xv)[:, 1]
                scored.append((average_precision_score(yv, pv), candidate))
            best_score, chosen = max(scored, key=lambda z: z[0])
            model = make_model(chosen, SEED + fold * 1000 + 77)
            fit_candidate(model, family, Xtr, ytr, Xv, yv)
            pv, pt = model.predict_proba(Xv)[:, 1], model.predict_proba(Xt)[:, 1]
            threshold = float(np.sort(pv)[-max(1, int(np.ceil(len(pv)*.02)))])
            transferred = pt >= threshold
            k = max(1, int(np.ceil(len(yt)*.02)))
            exact_idx = np.argsort(-pt, kind="mergesort")[:k]
            exact_precision = float(yt[exact_idx].mean())
            exact_recall = float(yt[exact_idx].sum()/max(int(yt.sum()),1))
            result = {
                "fold": fold, "model": family, "selected_config": chosen["config"],
                "validation_pr_auc": float(best_score),
                "test_pr_auc": float(average_precision_score(yt, pt)),
                "test_roc_auc": float(roc_auc_score(yt, pt)),
                "test_prevalence": float(yt.mean()),
                "test_pr_auc_lift": float(average_precision_score(yt, pt)/max(float(yt.mean()),1e-12)),
                "test_brier": float(brier_score_loss(yt, pt)),
                "test_log_loss": float(log_loss(yt, np.clip(pt,1e-7,1-1e-7))),
                "test_ece_10bin": ece(yt, pt),
                "test_calibration": calibration_slope_intercept(yt, pt),
                "test_precision_at_validation_2pct_threshold": float(yt[transferred].mean()) if transferred.any() else 0.0,
                "test_recall_at_validation_2pct_threshold": float(yt[transferred].sum()/max(int(yt.sum()),1)),
                "test_review_rate_at_validation_threshold": float(transferred.mean()),
                "test_precision_exact_2pct": exact_precision,
                "test_recall_exact_2pct": exact_recall,
                "pr_auc_bootstrap_95ci": bootstrap_ci(yt, pt, "pr_auc"),
                "precision_exact_2pct_bootstrap_95ci": bootstrap_ci(yt, pt, "precision_at_2pct"),
                "capacity": {str(cap): topk(yt, pt, cap) for cap in CAPACITIES},
                "train_start": str(train.purchase_time.min()), "train_end": str(train.purchase_time.max()),
                "validation_start": str(valid.purchase_time.min()), "validation_end": str(valid.purchase_time.max()),
                "test_start": str(test.purchase_time.min()), "test_end": str(test.purchase_time.max()),
                "train_rows": len(train), "validation_rows": len(valid), "test_rows": len(test),
            }
            details.append(result)
            rows.append({k:v for k,v in result.items() if k not in ("capacity","test_calibration","pr_auc_bootstrap_95ci","precision_exact_2pct_bootstrap_95ci")})
            print(f"fold={fold} {family} PR-AUC={result['test_pr_auc']:.5f} Brier={result['test_brier']:.5f}", flush=True)
    summary = {}
    for family in ("xgboost","lightgbm"):
        rr = [r for r in details if r["model"] == family]
        summary[family] = {
            "mean_test_pr_auc": float(np.mean([r["test_pr_auc"] for r in rr])),
            "std_test_pr_auc": float(np.std([r["test_pr_auc"] for r in rr], ddof=1)),
            "mean_test_roc_auc": float(np.mean([r["test_roc_auc"] for r in rr])),
            "mean_test_brier": float(np.mean([r["test_brier"] for r in rr])),
            "mean_test_log_loss": float(np.mean([r["test_log_loss"] for r in rr])),
            "mean_test_ece_10bin": float(np.mean([r["test_ece_10bin"] for r in rr])),
            "mean_precision_exact_2pct": float(np.mean([r["test_precision_exact_2pct"] for r in rr])),
            "mean_recall_exact_2pct": float(np.mean([r["test_recall_exact_2pct"] for r in rr])),
            "folds_won_by_pr_auc": sum(1 for r in rr if r["test_pr_auc"] > next(z["test_pr_auc"] for z in details if z["fold"] == r["fold"] and z["model"] != family)),
        }
    report = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_sha": os.getenv("GITHUB_SHA","local"),
        "protocol": "Three expanding-window chronological test windows. Each fold selects one of two predeclared configurations on its validation partition, then evaluates once on the subsequent test partition. Graph SVD excluded. Bootstrap intervals are stratified within each test window.",
        "bootstrap_replicates": BOOTSTRAPS, "fold_results": details, "summary": summary,
        "external_validation": "Not performed; requires independent later-period labeled data.",
        "runtime_seconds": round(time.time()-started,2)
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"rolling_robustness_report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    pd.DataFrame(rows).to_csv(OUT/"rolling_robustness_summary.csv",index=False)
    pd.DataFrame([{"fold":r["fold"],"model":r["model"],"capacity":cap,**v} for r in details for cap,v in r["capacity"].items()]).to_csv(OUT/"rolling_robustness_capacity.csv",index=False)
    print(json.dumps({"summary":summary,"runtime_seconds":report["runtime_seconds"]},indent=2),flush=True)

if __name__ == "__main__":
    main()
