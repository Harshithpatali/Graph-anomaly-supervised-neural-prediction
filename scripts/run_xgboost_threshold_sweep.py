#!/usr/bin/env python3
"""XGBoost-only validation-selected probability cutoffs on chronological folds.

For each fold, derive score cutoffs from validation score quantiles at 1%, 2%,
5%, and 10% investigation capacity. Transfer each cutoff unchanged to the
subsequent test window. Test labels never influence cutoff selection.
"""
from __future__ import annotations
import json, os, time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix,
)
from run_expanded_fraud_research import DATA, OUT, CANDIDATES, SEED, make_model, fit_candidate
from run_walk_forward_benchmark import build_folds, make_features

CAPACITIES = (0.01, 0.02, 0.05, 0.10)
# Sensitivity scenarios: one missed fraud is assigned 2x, 5x, 10x, 20x,
# or 50x the cost of investigating one alert. These are assumptions, not facts.
MISSED_FRAUD_COST_MULTIPLIERS = (2, 5, 10, 20, 50)

def score_distribution(p):
    p = np.asarray(p, dtype=float)
    return {
        "min": float(np.min(p)), "p50": float(np.quantile(p, .50)),
        "p75": float(np.quantile(p, .75)), "p90": float(np.quantile(p, .90)),
        "p95": float(np.quantile(p, .95)), "p99": float(np.quantile(p, .99)),
        "max": float(np.max(p)),
    }

def evaluate(y, p, threshold):
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
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
    df = df.dropna(subset=["purchase_time", "class"]).sort_values(
        "purchase_time", kind="mergesort"
    ).reset_index(drop=True)
    df["class"] = df["class"].astype(int)

    candidates = [c for c in CANDIDATES if c["name"] == "xgboost"]
    rows, fold_reports = [], []
    for fold, train_end, valid_end, test_end in build_folds(len(df)):
        train = df.iloc[:train_end].copy()
        valid = df.iloc[train_end:valid_end].copy()
        test = df.iloc[valid_end:test_end].copy()
        Xtr, Xv, Xt, _, _ = make_features(train, valid, test)
        graph_cols = [c for c in Xtr.columns if c.startswith("graph_svd_")]
        Xtr, Xv, Xt = (x.drop(columns=graph_cols) for x in (Xtr, Xv, Xt))
        ytr, yv, yt = (x["class"].to_numpy() for x in (train, valid, test))

        scored = []
        for j, candidate in enumerate(candidates):
            candidate_model = make_model(candidate, SEED + fold * 100 + j)
            fit_candidate(candidate_model, "xgboost", Xtr, ytr, Xv, yv)
            validation_scores = candidate_model.predict_proba(Xv)[:, 1]
            scored.append((average_precision_score(yv, validation_scores), candidate))
        validation_ap, chosen = max(scored, key=lambda item: item[0])

        model = make_model(chosen, SEED + fold * 1000 + 77)
        fit_candidate(model, "xgboost", Xtr, ytr, Xv, yv)
        pv = model.predict_proba(Xv)[:, 1]
        pt = model.predict_proba(Xt)[:, 1]

        capacity_results = []
        for capacity in CAPACITIES:
            # A validation-only order statistic defines the probability cutoff.
            k_valid = max(1, int(np.ceil(len(pv) * capacity)))
            threshold = float(np.sort(pv)[-k_valid])
            val_metrics = evaluate(yv, pv, threshold)
            test_metrics = evaluate(yt, pt, threshold)
            result = {
                "fold": fold, "model": "xgboost", "config": chosen["config"],
                "target_validation_capacity": capacity,
                "validation_selected_threshold": threshold,
                "validation_metrics": val_metrics,
                "transferred_test_metrics": test_metrics,
            }
            capacity_results.append(result)
            rows.append({
                "fold": fold, "model": "xgboost", "config": chosen["config"],
                "target_validation_capacity": capacity,
                "validation_selected_threshold": threshold,
                **{f"validation_{k}": v for k, v in val_metrics.items() if k != "threshold"},
                **{f"test_{k}": v for k, v in test_metrics.items() if k != "threshold"},
            })

        # Cost-sensitive operating-point selection. Select only on validation labels,
        # then transfer the chosen validation threshold unchanged to the test window.
        cost_scenarios = []
        for missed_cost_multiplier in MISSED_FRAUD_COST_MULTIPLIERS:
            candidates_for_cost = []
            for result in capacity_results:
                vm = result["validation_metrics"]
                validation_cost = (
                    vm["predicted_positive"]
                    + missed_cost_multiplier * vm["false_negative"]
                )
                candidates_for_cost.append((validation_cost, result))
            selected_validation_cost, selected_result = min(
                candidates_for_cost, key=lambda item: item[0]
            )
            selected_threshold = selected_result["validation_selected_threshold"]
            validation_metrics = selected_result["validation_metrics"]
            test_metrics = evaluate(yt, pt, selected_threshold)
            validation_cost_per_1000 = (
                1000.0 * selected_validation_cost / max(len(yv), 1)
            )
            test_cost = (
                test_metrics["predicted_positive"]
                + missed_cost_multiplier * test_metrics["false_negative"]
            )
            test_cost_per_1000 = 1000.0 * test_cost / max(len(yt), 1)
            cost_scenarios.append({
                "missed_fraud_cost_in_review_costs": missed_cost_multiplier,
                "selected_validation_capacity": selected_result["target_validation_capacity"],
                "validation_selected_threshold": selected_threshold,
                "validation_estimated_cost_per_1000": validation_cost_per_1000,
                "validation_review_rate": validation_metrics["review_rate"],
                "validation_precision": validation_metrics["precision"],
                "validation_recall": validation_metrics["recall"],
                "test_estimated_cost_per_1000": test_cost_per_1000,
                "test_review_rate": test_metrics["review_rate"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_false_positives": test_metrics["false_positive"],
                "test_false_negatives": test_metrics["false_negative"],
                "test_cases_investigated": test_metrics["predicted_positive"],
            })

        fold_report = {
            "fold": fold, "selected_config": chosen["config"],
            "validation_pr_auc": float(validation_ap),
            "test_pr_auc": float(average_precision_score(yt, pt)),
            "test_roc_auc": float(roc_auc_score(yt, pt)),
            "test_prevalence": float(yt.mean()),
            "validation_score_distribution": score_distribution(pv),
            "test_score_distribution": score_distribution(pt),
            "capacity_cutoffs": capacity_results,
            "cost_sensitive_scenarios": cost_scenarios,
            "train_rows": len(train), "validation_rows": len(valid), "test_rows": len(test),
            "validation_start": str(valid.purchase_time.min()),
            "validation_end": str(valid.purchase_time.max()),
            "test_start": str(test.purchase_time.min()),
            "test_end": str(test.purchase_time.max()),
        }
        fold_reports.append(fold_report)
        print(
            f"fold={fold} XGBoost PR-AUC={fold_report['test_pr_auc']:.5f} "
            + " | ".join(
                f"{int(x['target_validation_capacity']*100)}% cutoff={x['validation_selected_threshold']:.6g}, "
                f"test precision={x['transferred_test_metrics']['precision']:.4f}, "
                f"recall={x['transferred_test_metrics']['recall']:.4f}, "
                f"review={x['transferred_test_metrics']['review_rate']:.3%}"
                for x in capacity_results
            ),
            flush=True,
        )

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "xgboost_threshold_sweep.csv", index=False)
    report = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "model_scope": "XGBoost only; model family and causal features unchanged",
        "selection_protocol": (
            "For each chronological fold, derive probability cutoffs from validation "
            "score order statistics at 1%, 2%, 5%, and 10% target review capacities. "
            "Transfer each cutoff unchanged to the next test window. Test labels are "
            "never used to select cutoffs. Test review rates may differ from targets."
        ),
        "fold_results": fold_reports,
        "cost_model": {
            "formula": "investigated_cases * review_cost + missed_frauds * missed_fraud_cost",
            "normalization": "review_cost = 1; missed_fraud_cost is varied by scenario",
            "scenarios": list(MISSED_FRAUD_COST_MULTIPLIERS),
            "selection": "minimum validation estimated cost among the 1%, 2%, 5%, and 10% capacity-derived thresholds; test labels are not used for selection",
            "caveat": "Illustrative sensitivity analysis only. Real investigation cost, fraud loss, recovery, prevention effectiveness, and review capacity must be supplied by the business before operational use.",
        },
        "mean_test_estimated_cost_per_1000_by_missed_fraud_cost_ratio": {
            str(multiplier): float(np.mean([
                next(x for x in fold["cost_sensitive_scenarios"]
                     if x["missed_fraud_cost_in_review_costs"] == multiplier)["test_estimated_cost_per_1000"]
                for fold in fold_reports
            ]))
            for multiplier in MISSED_FRAUD_COST_MULTIPLIERS
        },
        "validation_selected_capacity_by_cost_scenario": {
            str(multiplier): [
                next(x for x in fold["cost_sensitive_scenarios"]
                     if x["missed_fraud_cost_in_review_costs"] == multiplier)["selected_validation_capacity"]
                for fold in fold_reports
            ]
            for multiplier in MISSED_FRAUD_COST_MULTIPLIERS
        },
        "mean_test_pr_auc": float(np.mean([r["test_pr_auc"] for r in fold_reports])),
        "mean_test_roc_auc": float(np.mean([r["test_roc_auc"] for r in fold_reports])),
        "mean_test_metrics_by_validation_capacity": {
            str(cap): {
                metric: float(np.mean([
                    next(x for x in fold["capacity_cutoffs"]
                         if x["target_validation_capacity"] == cap)["transferred_test_metrics"][metric]
                    for fold in fold_reports
                ]))
                for metric in ("precision", "recall", "f1", "specificity", "review_rate")
            }
            for cap in CAPACITIES
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    (OUT / "xgboost_threshold_sweep.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)

if __name__ == "__main__":
    main()
