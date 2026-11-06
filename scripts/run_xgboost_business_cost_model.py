#!/usr/bin/env python3
"""Business-calibrated, validation-selected XGBoost investigation economics.

The XGBoost model, causal features, and chronological folds are unchanged.
Business assumptions are supplied at runtime; no illustrative rupee values are
silently substituted. Test labels are used only for final evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from run_expanded_fraud_research import DATA, OUT, CANDIDATES, SEED, make_model, fit_candidate
from run_walk_forward_benchmark import build_folds, make_features

CAPACITIES = (0.00, 0.01, 0.02, 0.05, 0.10)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyst-cost", type=float, required=True,
                        help="Fully loaded cost of one investigation, in currency units")
    parser.add_argument("--fraud-loss", type=float, required=True,
                        help="Average financial loss per fraudulent transaction, same currency")
    parser.add_argument("--recovery-rate", type=float, required=True,
                        help="Fraction of fraud loss recovered after investigation, 0..1")
    parser.add_argument("--prevention-rate", type=float, required=True,
                        help="Fraction of the unrecovered fraud loss prevented after investigation, 0..1")
    parser.add_argument("--max-cases-per-day", type=int, required=True,
                        help="Hard maximum number of cases investigators can review per calendar day")
    args = parser.parse_args()
    if args.analyst_cost < 0 or args.fraud_loss < 0:
        parser.error("analyst cost and fraud loss must be non-negative")
    if not 0 <= args.recovery_rate <= 1 or not 0 <= args.prevention_rate <= 1:
        parser.error("recovery and prevention rates must be between 0 and 1")
    if args.max_cases_per_day < 0:
        parser.error("max-cases-per-day must be non-negative")
    if args.analyst_cost == 0 and args.fraud_loss == 0:
        parser.error("at least one of analyst cost or fraud loss must be positive")
    return args


def calendar_days(times):
    dates = pd.to_datetime(times, utc=True).dt.floor("D")
    return max(1, int((dates.max() - dates.min()).days) + 1)


def select_scores(scores, times, threshold, max_per_day):
    """Apply a frozen score cutoff and enforce a per-calendar-day review cap."""
    scores = np.asarray(scores, dtype=float)
    dates = pd.to_datetime(pd.Series(times).reset_index(drop=True), utc=True).dt.floor("D")
    selected = np.zeros(len(scores), dtype=bool)
    if threshold is None or max_per_day == 0:
        return selected
    eligible = scores >= float(threshold)
    for day in dates[eligible].drop_duplicates():
        idx = np.flatnonzero((dates == day).to_numpy() & eligible)
        if len(idx) <= max_per_day:
            selected[idx] = True
        else:
            keep = idx[np.argsort(scores[idx], kind="mergesort")[-max_per_day:]]
            selected[keep] = True
    return selected


def evaluate_policy(y, scores, times, threshold, args):
    y = np.asarray(y, dtype=int)
    selected = select_scores(scores, times, threshold, args.max_cases_per_day)
    tp = int(np.sum(selected & (y == 1)))
    fp = int(np.sum(selected & (y == 0)))
    fn = int(np.sum((~selected) & (y == 1)))
    tn = int(np.sum((~selected) & (y == 0)))
    investigated = tp + fp
    # Recovery applies first; prevention applies to the remaining unrecovered loss.
    effective_loss_mitigation = args.recovery_rate + (
        1.0 - args.recovery_rate
    ) * args.prevention_rate
    gross_mitigated_loss = tp * args.fraud_loss * effective_loss_mitigation
    investigation_cost = investigated * args.analyst_cost
    net_benefit = gross_mitigated_loss - investigation_cost
    return {
        "threshold": None if threshold is None else float(threshold),
        "investigated_cases": investigated,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": tp / investigated if investigated else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "review_rate": investigated / len(y) if len(y) else 0.0,
        "gross_mitigated_fraud_loss": float(gross_mitigated_loss),
        "investigation_cost": float(investigation_cost),
        "net_benefit": float(net_benefit),
        "roi_on_investigation_cost": (
            float(net_benefit / investigation_cost) if investigation_cost > 0 else None
        ),
        "residual_missed_fraud_loss": float(fn * args.fraud_loss),
        "mean_cases_per_calendar_day": float(investigated / calendar_days(times)),
        "max_cases_per_day": int(args.max_cases_per_day),
    }


def main():
    args = parse_args()
    started = time.time()
    df = pd.read_csv(DATA)
    df["purchase_time"] = pd.to_datetime(df["purchase_time"], errors="coerce", utc=True)
    df["signup_time"] = pd.to_datetime(df["signup_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["purchase_time", "class"]).sort_values(
        "purchase_time", kind="mergesort"
    ).reset_index(drop=True)
    df["class"] = df["class"].astype(int)

    xgb_candidates = [c for c in CANDIDATES if c["name"] == "xgboost"]
    fold_reports = []
    for fold, train_end, valid_end, test_end in build_folds(len(df)):
        train = df.iloc[:train_end].copy()
        valid = df.iloc[train_end:valid_end].copy()
        test = df.iloc[valid_end:test_end].copy()
        Xtr, Xv, Xt, _, _ = make_features(train, valid, test)
        graph_cols = [c for c in Xtr.columns if c.startswith("graph_svd_")]
        Xtr, Xv, Xt = (x.drop(columns=graph_cols) for x in (Xtr, Xv, Xt))
        ytr, yv, yt = (x["class"].to_numpy() for x in (train, valid, test))

        validation_candidates = []
        for j, candidate in enumerate(xgb_candidates):
            candidate_model = make_model(candidate, SEED + fold * 100 + j)
            fit_candidate(candidate_model, "xgboost", Xtr, ytr, Xv, yv)
            pv = candidate_model.predict_proba(Xv)[:, 1]
            validation_candidates.append((average_precision_score(yv, pv), candidate))
        validation_pr_auc, chosen = max(validation_candidates, key=lambda x: x[0])

        model = make_model(chosen, SEED + fold * 1000 + 77)
        fit_candidate(model, "xgboost", Xtr, ytr, Xv, yv)
        pv = model.predict_proba(Xv)[:, 1]
        pt = model.predict_proba(Xt)[:, 1]

        policy_candidates = []
        for capacity in CAPACITIES:
            if capacity == 0:
                threshold = None
            else:
                k = max(1, int(np.ceil(len(pv) * capacity)))
                threshold = float(np.sort(pv)[-k])
            validation_metrics = evaluate_policy(
                yv, pv, valid["purchase_time"], threshold, args
            )
            policy_candidates.append({
                "target_validation_capacity": capacity,
                "validation_threshold": threshold,
                "validation_metrics": validation_metrics,
            })

        # Select maximum validation net benefit. Tie-break toward fewer reviews.
        selected = max(
            policy_candidates,
            key=lambda x: (
                x["validation_metrics"]["net_benefit"],
                -x["validation_metrics"]["investigated_cases"],
            ),
        )
        test_metrics = evaluate_policy(
            yt, pt, test["purchase_time"], selected["validation_threshold"], args
        )
        fold_reports.append({
            "fold": fold,
            "selected_xgboost_config": chosen["config"],
            "validation_pr_auc": float(validation_pr_auc),
            "test_pr_auc": float(average_precision_score(yt, pt)),
            "test_roc_auc": float(roc_auc_score(yt, pt)),
            "validation_rows": len(valid),
            "test_rows": len(test),
            "validation_days": calendar_days(valid["purchase_time"]),
            "test_days": calendar_days(test["purchase_time"]),
            "policy_candidates": policy_candidates,
            "selected_validation_capacity": selected["target_validation_capacity"],
            "selected_validation_threshold": selected["validation_threshold"],
            "selected_validation_metrics": selected["validation_metrics"],
            "transferred_test_metrics": test_metrics,
        })
        print(
            f"fold={fold} selected_capacity={selected['target_validation_capacity']:.0%} "
            f"validation_net={selected['validation_metrics']['net_benefit']:.2f} "
            f"test_net={test_metrics['net_benefit']:.2f} "
            f"test_cases/day={test_metrics['mean_cases_per_calendar_day']:.2f}",
            flush=True,
        )

    OUT.mkdir(parents=True, exist_ok=True)
    assumptions = {
        "currency_unit": "user-supplied; all monetary inputs must use the same unit",
        "analyst_cost_per_investigation": args.analyst_cost,
        "average_fraud_loss_per_transaction": args.fraud_loss,
        "recovery_rate": args.recovery_rate,
        "prevention_rate_on_unrecovered_loss": args.prevention_rate,
        "effective_loss_mitigation_rate": args.recovery_rate + (
            1 - args.recovery_rate
        ) * args.prevention_rate,
        "max_cases_per_calendar_day": args.max_cases_per_day,
    }
    report = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_sha": os.getenv("GITHUB_SHA", "local"),
        "model_scope": "XGBoost only; model family and causal features unchanged",
        "selection_protocol": (
            "For each chronological fold, evaluate no-review and 1%, 2%, 5%, 10% "
            "validation-score cutoffs. Enforce the daily investigator cap by ranking "
            "eligible scores within each calendar day. Select the policy with maximum "
            "validation net benefit (ties prefer fewer investigations), then transfer "
            "the frozen threshold to the next test window. Test labels are never used "
            "for policy selection."
        ),
        "financial_method": {
            "gross_mitigated_loss": "true_positives * average_fraud_loss * (recovery_rate + (1-recovery_rate)*prevention_rate)",
            "investigation_cost": "investigated_cases * analyst_cost_per_investigation",
            "net_benefit": "gross_mitigated_loss - investigation_cost",
            "residual_missed_fraud_loss": "false_negatives * average_fraud_loss",
            "assumptions": assumptions,
            "caveats": [
                "A model true positive is treated as an investigated fraud case; real confirmation and action rates may be lower.",
                "Recovery is applied first; prevention is applied only to the remaining unrecovered loss to avoid double counting.",
                "Average fraud loss is assumed constant across transactions; transaction-level loss values would improve the model.",
                "Net benefit is incremental benefit from detected cases minus investigation cost; residual missed-fraud loss is reported separately and is not subtracted again.",
                "Daily caps are enforced by selecting the highest scores within each calendar day.",
                "These are estimates based on supplied assumptions, not realized savings or a production authorization."
            ],
        },
        "fold_results": fold_reports,
        "mean_test_net_benefit": float(np.mean([
            f["transferred_test_metrics"]["net_benefit"] for f in fold_reports
        ])),
        "mean_test_gross_mitigated_fraud_loss": float(np.mean([
            f["transferred_test_metrics"]["gross_mitigated_fraud_loss"] for f in fold_reports
        ])),
        "mean_test_investigation_cost": float(np.mean([
            f["transferred_test_metrics"]["investigation_cost"] for f in fold_reports
        ])),
        "mean_test_residual_missed_fraud_loss": float(np.mean([
            f["transferred_test_metrics"]["residual_missed_fraud_loss"] for f in fold_reports
        ])),
        "selected_validation_capacities": [
            f["selected_validation_capacity"] for f in fold_reports
        ],
        "runtime_seconds": round(time.time() - started, 2),
    }
    json_path = OUT / "xgboost_business_cost_model.json"
    csv_path = OUT / "xgboost_business_cost_model.csv"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    flat_rows = []
    for fold in fold_reports:
        for candidate in fold["policy_candidates"]:
            flat_rows.append({
                "fold": fold["fold"],
                "target_validation_capacity": candidate["target_validation_capacity"],
                "validation_threshold": candidate["validation_threshold"],
                "validation_net_benefit": candidate["validation_metrics"]["net_benefit"],
                "validation_investigated_cases": candidate["validation_metrics"]["investigated_cases"],
                "validation_precision": candidate["validation_metrics"]["precision"],
                "validation_recall": candidate["validation_metrics"]["recall"],
                "selected": candidate["target_validation_capacity"] == fold["selected_validation_capacity"],
                "test_net_benefit_transferred_policy": (
                    fold["transferred_test_metrics"]["net_benefit"]
                    if candidate["target_validation_capacity"] == fold["selected_validation_capacity"]
                    else None
                ),
            })
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
