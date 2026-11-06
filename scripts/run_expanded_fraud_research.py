#!/usr/bin/env python3
"""Leakage-safe model comparison and capacity evaluation on chronological folds."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_walk_forward_benchmark import DATA, OUT, ART, build_folds, make_features

SEED = 2026
CAPACITIES = (0.01, 0.02, 0.05, 0.10)
CANDIDATES = [
    {"name": "xgboost", "config": 1, "params": {"n_estimators": 350, "max_depth": 4, "learning_rate": 0.035, "min_child_weight": 8, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 8}},
    {"name": "xgboost", "config": 2, "params": {"n_estimators": 600, "max_depth": 3, "learning_rate": 0.025, "min_child_weight": 10, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 12}},
    {"name": "catboost", "config": 1, "params": {"iterations": 600, "depth": 6, "learning_rate": 0.035, "l2_leaf_reg": 8}},
    {"name": "catboost", "config": 2, "params": {"iterations": 800, "depth": 5, "learning_rate": 0.025, "l2_leaf_reg": 12}},
    {"name": "lightgbm", "config": 1, "params": {"n_estimators": 600, "num_leaves": 15, "max_depth": 5, "learning_rate": 0.025, "min_child_samples": 80, "reg_lambda": 8}},
    {"name": "lightgbm", "config": 2, "params": {"n_estimators": 800, "num_leaves": 23, "max_depth": 6, "learning_rate": 0.02, "min_child_samples": 100, "reg_lambda": 12}},
]


def make_model(candidate, seed):
    name, p = candidate["name"], candidate["params"]
    if name == "xgboost":
        return XGBClassifier(**p, objective="binary:logistic", eval_metric="aucpr",
                             tree_method="hist", n_jobs=2, random_state=seed)
    if name == "catboost":
        return CatBoostClassifier(**p, loss_function="Logloss", eval_metric="PRAUC",
                                   auto_class_weights="Balanced", random_seed=seed,
                                   thread_count=2, verbose=False, allow_writing_files=False)
    return LGBMClassifier(**p, objective="binary", metric="average_precision",
                          class_weight="balanced", random_state=seed,
                          n_jobs=2, verbosity=-1)


def fit_candidate(model, name, x_train, y_train, x_valid, y_valid):
    if name == "catboost":
        model.fit(x_train, y_train, eval_set=(x_valid, y_valid),
                  early_stopping_rounds=70, verbose=False)
    elif name == "lightgbm":
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)],
                  callbacks=[early_stopping(70, verbose=False), log_evaluation(0)])
    else:
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    return model


def topk(y, scores, fraction):
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    k = max(1, int(np.ceil(len(y) * fraction)))
    chosen = np.argsort(-scores, kind="mergesort")[:k]
    tp = int(y[chosen].sum())
    precision, recall = tp / k, tp / max(int(y.sum()), 1)
    return {"capacity": fraction, "queue_size": k, "precision": precision,
            "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12)}


def threshold_transfer(y_val, p_val, y_test, p_test, fraction):
    val = topk(y_val, p_val, fraction)
    k = val["queue_size"]
    order = np.argsort(-p_val, kind="mergesort")
    cutoff = float(p_val[order[k - 1]])
    # Stable secondary order preserves exact queue capacity in validation;
    # test transfer uses the learned cutoff, and test top-k is reported separately.
    selected = p_test >= cutoff
    return {"validation_threshold": cutoff, "validation_topk": val,
            "test_transferred_threshold": {
                "precision": float((selected & (np.asarray(y_test) == 1)).sum() / max(int(selected.sum()), 1)),
                "recall": float((selected & (np.asarray(y_test) == 1)).sum() / max(int(np.asarray(y_test).sum()), 1)),
                "review_rate": float(selected.mean()),
            },
            "test_exact_capacity": topk(y_test, p_test, fraction)}


def main():
    started = time.time()
    df = pd.read_csv(DATA)
    df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce", utc=True)
    df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["purchase_time", "class"]).sort_values("purchase_time", kind="mergesort").reset_index(drop=True)
    df["class"] = df["class"].astype(int)
    folds = list(build_folds(len(df)))
    validation_rows = []

    # Model family/config selection uses only validation partitions from folds 1 and 2.
    for fold, train_end, valid_end, _ in folds[:2]:
        train, valid = df.iloc[:train_end].copy(), df.iloc[train_end:valid_end].copy()
        x_train, x_valid, _, _, _ = make_features(train, valid, df.iloc[valid_end:valid_end + 1].copy())
        # The old inductive SVD embeddings are intentionally excluded. New
        # event-time edge, neighborhood, velocity and novelty features remain.
        graph_cols = [c for c in x_train.columns if c.startswith("graph_svd_")]
        x_train = x_train.drop(columns=graph_cols)
        x_valid = x_valid.drop(columns=graph_cols)
        y_train, y_valid = train["class"].to_numpy(), valid["class"].to_numpy()
        for idx, candidate in enumerate(CANDIDATES):
            model = make_model(candidate, SEED + fold * 100 + idx)
            fit_candidate(model, candidate["name"], x_train, y_train, x_valid, y_valid)
            p = model.predict_proba(x_valid)[:, 1]
            validation_rows.append({
                "fold": fold, "model": candidate["name"], "config": candidate["config"],
                "validation_pr_auc": float(average_precision_score(y_valid, p)),
                "validation_roc_auc": float(roc_auc_score(y_valid, p)),
                "params": candidate["params"],
            })
            print(f"fold={fold} {candidate['name']} config={candidate['config']} "
                  f"PR-AUC={validation_rows[-1]['validation_pr_auc']:.5f}", flush=True)

    summary = {}
    for name in sorted({c["name"] for c in CANDIDATES}):
        for config in sorted({c["config"] for c in CANDIDATES if c["name"] == name}):
            rows = [r for r in validation_rows if r["model"] == name and r["config"] == config]
            summary[f"{name}:{config}"] = float(np.mean([r["validation_pr_auc"] for r in rows]))
    selected_key = max(summary, key=summary.get)
    selected_name, selected_config = selected_key.split(":")
    selected = next(c for c in CANDIDATES if c["name"] == selected_name and c["config"] == int(selected_config))

    # Untouched third chronological fold: evaluate the validation-selected
    # configuration from each model family once on the same final test window.
    fold, train_end, valid_end, test_end = folds[2]
    train, valid, test = (df.iloc[:train_end].copy(), df.iloc[train_end:valid_end].copy(),
                          df.iloc[valid_end:test_end].copy())
    x_train, x_valid, x_test, graph, medians = make_features(train, valid, test)
    graph_cols = [c for c in x_train.columns if c.startswith("graph_svd_")]
    x_train, x_valid, x_test = (x.drop(columns=graph_cols) for x in (x_train, x_valid, x_test))
    y_train, y_valid, y_test = (x["class"].to_numpy() for x in (train, valid, test))
    selected_by_model = {}
    for name in sorted({candidate["name"] for candidate in CANDIDATES}):
        options = [candidate for candidate in CANDIDATES if candidate["name"] == name]
        selected_by_model[name] = max(
            options, key=lambda candidate: summary[f"{name}:{candidate['config']}"]
        )
    final_models = {}
    for idx, (name, candidate) in enumerate(selected_by_model.items()):
        fitted = make_model(candidate, SEED + 999 + idx)
        fit_candidate(fitted, name, x_train, y_train, x_valid, y_valid)
        p_valid_family = fitted.predict_proba(x_valid)[:, 1]
        p_test_family = fitted.predict_proba(x_test)[:, 1]
        cap_results = {str(cap): threshold_transfer(y_valid, p_valid_family, y_test, p_test_family, cap)
                       for cap in CAPACITIES}
        final_models[name] = {
            "selected_config": candidate,
            "validation_pr_auc": summary[f"{name}:{candidate['config']}"],
            "test_pr_auc": float(average_precision_score(y_test, p_test_family)),
            "test_roc_auc": float(roc_auc_score(y_test, p_test_family)),
            "pr_auc_lift_over_prevalence": float(average_precision_score(y_test, p_test_family) / max(float(y_test.mean()), 1e-12)),
            "capacity": cap_results,
            "_model": fitted,
            "_p_valid": p_valid_family,
            "_p_test": p_test_family,
        }

    champion_name = max(summary, key=summary.get).split(":")[0]
    selected = selected_by_model[champion_name]
    selected_name = champion_name
    model = final_models[champion_name]["_model"]
    p_valid = final_models[champion_name]["_p_valid"]
    p_test = final_models[champion_name]["_p_test"]
    capacities = final_models[champion_name]["capacity"]
    prevalence = float(y_test.mean())
    final_test_comparison = {
        name: {k: v for k, v in result.items() if not k.startswith("_")}
        for name, result in final_models.items()
    }
    report = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "dataset": str(DATA.relative_to(ROOT)),
        "dataset_rows": int(len(df)),
        "overall_fraud_prevalence": float(df["class"].mean()),
        "final_test_prevalence": prevalence,
        "random_pr_auc_baseline": prevalence,
        "final_fold": {
            "fold": fold,
            "train": {"rows": len(train), "start": str(train.purchase_time.min()), "end": str(train.purchase_time.max())},
            "validation": {"rows": len(valid), "start": str(valid.purchase_time.min()), "end": str(valid.purchase_time.max())},
            "test": {"rows": len(test), "start": str(test.purchase_time.min()), "end": str(test.purchase_time.max())},
            "selected_model": selected_name,
            "selected_config": selected,
            "model_comparison": final_test_comparison,
            "metrics": {"pr_auc": float(average_precision_score(y_test, p_test)),
                        "roc_auc": float(roc_auc_score(y_test, p_test)),
                        "pr_auc_lift_over_prevalence": float(average_precision_score(y_test, p_test) / max(prevalence, 1e-12)),
                        "capacity": capacities},
        },
        "validation_selection": summary,
        "final_test_model_comparison": final_test_comparison,
        "validation_rows": [{k: v for k, v in row.items() if k != "params"} for row in validation_rows],
        "feature_groups": {
            "temporal_behavior": [c for c in x_train.columns if any(s in c for s in ("events_prior", "seconds_since", "events_per_day", "activity_ratio", "burst"))],
            "temporal_edges": [c for c in x_train.columns if c.startswith(("user_device_", "user_ip_", "device_ip_"))],
            "graph_neighborhood": [c for c in x_train.columns if any(s in c for s in ("distinct_users", "neighborhood", "novelty"))],
        },
        "graph_svd_used": False,
        "protocol": "Candidate selection uses only folds 1-2 validation PR-AUC. Fold 3 test is untouched until one final evaluation. New graph features are causal event-time edge histories and local neighborhood aggregates; old graph SVD coordinates are excluded.",
        "external_validation": {
            "status": "blocked_pending_independent_labeled_data",
            "reason": "The repository dataset ends in 2015. IEEE-CIS 2019 requires Kaggle-authenticated files; no licensed later-period raw dataset is present in this repository. Do not treat a different-domain or verification-biased dataset as a comparable temporal holdout.",
            "required_files": ["data/external/Fraud_Data_later.csv"],
            "expected_schema": "Same e-commerce schema and independently adjudicated class labels, with purchase_time strictly later than the existing dataset.",
        },
        "runtime_seconds": round(time.time() - started, 2),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    ART.mkdir(parents=True, exist_ok=True)
    (OUT / "expanded_model_comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(validation_rows).drop(columns=["params"]).to_csv(OUT / "expanded_model_validation.csv", index=False)
    pd.DataFrame([{"capacity": float(cap), **v["test_exact_capacity"]}
                  for cap, v in capacities.items()]).to_csv(OUT / "expanded_capacity_metrics.csv", index=False)
    import joblib
    joblib.dump({"model": model, "feature_columns": list(x_train.columns), "medians": medians,
                 "selected_model": selected_name, "selected_config": selected},
                ART / "expanded_temporal_champion.joblib")
    pd.DataFrame({"actual": y_test, "score": p_test}).to_csv(OUT / "expanded_final_predictions.csv", index=False)
    pd.DataFrame([{"model": name, "test_pr_auc": result["test_pr_auc"],
                   "test_roc_auc": result["test_roc_auc"],
                   "validation_pr_auc": result["validation_pr_auc"]}
                  for name, result in final_test_comparison.items()]).to_csv(
                      OUT / "expanded_final_model_comparison.csv", index=False)
    print(json.dumps({"selected": selected_name, "config": selected_config,
                      "final_test": report["final_fold"]["metrics"],
                      "final_test_model_comparison": report["final_test_model_comparison"],
                      "external_validation": report["external_validation"],
                      "runtime_seconds": report["runtime_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
