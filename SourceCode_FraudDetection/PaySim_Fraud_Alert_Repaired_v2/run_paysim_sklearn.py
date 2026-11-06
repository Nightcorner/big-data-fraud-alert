from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from src.paysim_features import (
    BALANCE_FEATURES, CORE_FEATURES, DATA_PATH, build_features, load_paysim,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "sklearn"


def choose_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, dict]:
    """Choose threshold using validation only; never use test labels here."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, {"validation_threshold_selection": "fallback_0.5_no_thresholds"}
    f1s = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
                    out=np.zeros_like(precision[:-1]), where=(precision[:-1] + recall[:-1]) > 0)
    ix = int(np.nanargmax(f1s))
    t = float(thresholds[ix])
    pred = (scores >= t).astype(np.int8)
    return t, {
        "validation_threshold_selection": "maximum_F1_on_validation",
        "validation_threshold": t,
        "validation_precision_at_threshold": float(precision_score(y_true, pred, zero_division=0)),
        "validation_recall_at_threshold": float(recall_score(y_true, pred, zero_division=0)),
        "validation_f1_at_threshold": float(f1_score(y_true, pred, zero_division=0)),
        "validation_predicted_alert_count": int(pred.sum()),
    }


def calc_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel()]
    return {
        "rows": int(len(y)), "fraud_actual": int(y.sum()), "nonfraud_actual": int(len(y) - y.sum()),
        "threshold": float(threshold), "alerts": int(pred.sum()),
        "true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, scores)),
    }


def calc_budget(y: np.ndarray, scores: np.ndarray) -> list[dict]:
    rows = []
    n = len(y)
    total_fraud = int(y.sum())
    order = np.argsort(-scores, kind="mergesort")
    for pct in [0.005, 0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.50]:
        k = max(1, int(round(n * pct)))
        tp = int(y[order[:k]].sum())
        fp = k - tp
        rows.append({
            "alert_budget_percent": pct * 100, "alerts": k,
            "true_positive": tp, "false_positive": fp,
            "false_negative": total_fraud - tp,
            "precision": tp / k if k else 0.0,
            "recall": tp / total_fraud if total_fraud else 0.0,
            "random_expected_true_positive": k * total_fraud / n,
            "lift_over_random": (tp / (k * total_fraud / n)) if k and total_fraud else 0.0,
        })
    return rows


def make_model(seed: int = 42) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=0.05, num_leaves=31,
        max_depth=-1, min_child_samples=80, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.9, reg_lambda=2.0, max_bin=127,
        random_state=seed, n_jobs=3, verbosity=-1, force_col_wise=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-aware PaySim ML experiment using scikit-learn API + LightGBM.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to PaySim CSV")
    parser.add_argument("--train-end-step", type=int, default=520)
    parser.add_argument("--validation-end-step", type=int, default=600)
    parser.add_argument("--skip-core-ablation", action="store_true", help="Skip simple core feature comparison")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Loading PaySim: {args.data}", flush=True)
    raw = load_paysim(args.data, usecols=["step", "type", "amount", "oldbalanceOrg", "oldbalanceDest", "isFraud"])
    step = raw["step"].astype(int)
    masks = {
        "train": step <= args.train_end_step,
        "validation": (step > args.train_end_step) & (step <= args.validation_end_step),
        "test": step > args.validation_end_step,
    }
    partition_summary = []
    for name, mask in masks.items():
        part_y = raw.loc[mask, "isFraud"].astype(int)
        part_step = raw.loc[mask, "step"]
        partition_summary.append({
            "partition": name, "rows": int(mask.sum()), "fraud_count": int(part_y.sum()),
            "nonfraud_count": int(len(part_y) - part_y.sum()),
            "step_min": int(part_step.min()), "step_max": int(part_step.max()),
        })
        if part_y.nunique() < 2:
            raise ValueError(f"Partisi {name} tidak memiliki kedua kelas. Periksa batas step.")
    pd.DataFrame(partition_summary).to_csv(OUT / "temporal_split_summary.csv", index=False)
    print(pd.DataFrame(partition_summary).to_string(index=False), flush=True)

    y_train = raw.loc[masks["train"], "isFraud"].astype(np.int8).to_numpy()
    y_val = raw.loc[masks["validation"], "isFraud"].astype(np.int8).to_numpy()
    y_test = raw.loc[masks["test"], "isFraud"].astype(np.int8).to_numpy()

    candidates = ["pre_event_core"]
    if not args.skip_core_ablation:
        candidates.append("pre_event_balance")

    candidate_rows: list[dict] = []
    best_model = None
    best_features = None
    best_name = None
    best_val_ap = -1.0
    best_val_scores = None
    best_val_metrics = None
    best_fit_seconds = None
    candidate_models = {}
    candidate_features = {}
    candidate_thresholds = {}

    for candidate_name in candidates:
        include_balance = candidate_name == "pre_event_balance"
        features_all = build_features(raw, include_balance_features=include_balance)
        feat_cols = BALANCE_FEATURES if include_balance else CORE_FEATURES
        train_mask = masks["train"].to_numpy()
        val_mask = masks["validation"].to_numpy()
        X_train = features_all.loc[train_mask, feat_cols]
        X_val = features_all.loc[val_mask, feat_cols]

        # Check forbidden columns before fitting.
        forbidden = {"isFraud", "isFlaggedFraud", "newbalanceOrig", "newbalanceDest"}.intersection(X_train.columns)
        if forbidden:
            raise AssertionError(f"Forbidden/leakage columns found in X: {sorted(forbidden)}")
        print(f"\nTraining {candidate_name} with {len(feat_cols)} features...", flush=True)
        model = make_model(seed=42)
        start = time.perf_counter()
        model.fit(
            X_train, y_train,
            categorical_feature=["type"],
            eval_set=[(X_val, y_val)],
            eval_metric="auc",
            callbacks=[early_stopping(stopping_rounds=35, verbose=True), log_evaluation(period=50)],
        )
        fit_seconds = time.perf_counter() - start
        val_scores = model.predict_proba(X_val, num_iteration=model.best_iteration_)[:, 1]
        val_ap = float(average_precision_score(y_val, val_scores))
        val_auc = float(roc_auc_score(y_val, val_scores))
        threshold, threshold_info = choose_f1_threshold(y_val, val_scores)
        val_at_threshold = calc_metrics(y_val, val_scores, threshold)
        candidate_rows.append({
            "candidate": candidate_name, "feature_count": len(feat_cols),
            "best_iteration": int(model.best_iteration_ or model.n_estimators),
            "fit_seconds": fit_seconds, "validation_roc_auc": val_auc,
            "validation_average_precision": val_ap,
            "validation_precision_at_f1_threshold": val_at_threshold["precision"],
            "validation_recall_at_f1_threshold": val_at_threshold["recall"],
            "validation_f1_at_f1_threshold": val_at_threshold["f1"],
            "validation_alerts_at_f1_threshold": val_at_threshold["alerts"],
            "validation_threshold": threshold,
        })
        print(f"{candidate_name}: val ROC-AUC={val_auc:.6f}; AP={val_ap:.6f}; threshold={threshold:.8f}; fit={fit_seconds:.1f}s", flush=True)

        candidate_models[candidate_name] = model
        candidate_features[candidate_name] = feat_cols.copy()
        candidate_thresholds[candidate_name] = threshold

        if val_ap > best_val_ap:
            best_model = model
            best_features = feat_cols.copy()
            best_name = candidate_name
            best_val_ap = val_ap
            best_val_scores = val_scores.copy()
            best_val_metrics = {**threshold_info, **val_at_threshold, "roc_auc": val_auc, "average_precision": val_ap}
            best_fit_seconds = fit_seconds
        # release feature matrices before next candidate to control memory
        del features_all, X_train, X_val, val_scores

    candidates_df = pd.DataFrame(candidate_rows).sort_values("validation_average_precision", ascending=False)
    candidates_df.to_csv(OUT / "validation_candidate_comparison.csv", index=False)
    print("\nCandidate comparison (model selected by validation Average Precision):", flush=True)
    print(candidates_df.to_string(index=False), flush=True)

    # After selecting the candidate from validation AP only, run a single temporal
    # test comparison to document the feature-ablation effect. Test labels are not
    # used for selecting the final model or its threshold.
    test_raw = raw.loc[masks["test"]]
    test_ablation_rows = []
    test_score_by_candidate = {}
    for candidate_name in candidates:
        use_balance = candidate_name == "pre_event_balance"
        feature_names = candidate_features[candidate_name]
        test_feature_frame = build_features(test_raw, include_balance_features=use_balance)
        X_test_candidate = test_feature_frame.loc[:, feature_names]
        model_candidate = candidate_models[candidate_name]
        scores_candidate = model_candidate.predict_proba(X_test_candidate, num_iteration=model_candidate.best_iteration_)[:, 1]
        threshold_candidate = candidate_thresholds[candidate_name]
        candidate_test_metrics = calc_metrics(y_test, scores_candidate, threshold_candidate)
        test_ablation_rows.append({
            "candidate": candidate_name,
            "validation_roc_auc": next(r["validation_roc_auc"] for r in candidate_rows if r["candidate"] == candidate_name),
            "validation_average_precision": next(r["validation_average_precision"] for r in candidate_rows if r["candidate"] == candidate_name),
            "validation_chosen_threshold": threshold_candidate,
            **{f"test_{k}": v for k, v in candidate_test_metrics.items()},
        })
        test_score_by_candidate[candidate_name] = scores_candidate
        del test_feature_frame, X_test_candidate, scores_candidate
    pd.DataFrame(test_ablation_rows).to_csv(OUT / "test_feature_ablation_comparison.csv", index=False)

    include_balance = best_name == "pre_event_balance"
    test_scores = test_score_by_candidate[best_name]
    threshold = candidate_thresholds[best_name]
    test_metrics_threshold = calc_metrics(y_test, test_scores, threshold)
    test_metrics_default = calc_metrics(y_test, test_scores, 0.5)
    budgets = calc_budget(y_test, test_scores)

    # Model feature importance with readable names.
    imp = pd.DataFrame({
        "feature": best_features,
        "importance_gain": best_model.booster_.feature_importance(importance_type="gain"),
        "importance_split": best_model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)
    imp.to_csv(OUT / "feature_importance.csv", index=False)
    best_model.booster_.save_model(str(OUT / "selected_lightgbm_model.txt"))

    test_result = {
        "dataset": args.data.name, "model_family": "LightGBM via scikit-learn API",
        "selected_candidate_by_validation_average_precision": best_name,
        "selected_features": best_features,
        "leakage_policy": {
            "excluded_isFlaggedFraud": True,
            "excluded_newbalanceOrig": True,
            "excluded_newbalanceDest": True,
            "excluded_nameOrig_nameDest_exact_ids": True,
            "included_old_balance_fields_if_selected": include_balance,
            "note": "PaySim is synthetic; amount==oldbalanceOrg is a strong simulator-specific pattern, not a universal banking rule.",
        },
        "split_strategy": "strict chronological by step: train<=train_end, validation next steps, test only later steps",
        "train_end_step": args.train_end_step, "validation_end_step": args.validation_end_step,
        "partitions": partition_summary,
        "chosen_validation_metrics": best_val_metrics,
        "decision_threshold_chosen_on_validation": threshold,
        "test_metrics_at_validation_f1_threshold": test_metrics_threshold,
        "test_metrics_at_default_threshold_0_5": test_metrics_default,
        "test_feature_ablation_comparison_file": "test_feature_ablation_comparison.csv",
        "test_fixed_alert_budgets": budgets,
        "fit_seconds_of_selected_model": best_fit_seconds,
        "warning": "PaySim scores can be very high due to synthetic generator patterns. Report as synthetic-data performance, not real-bank performance. Any future-time holdout inspected during development is diagnostic, not a fully independent external validation.",
    }
    (OUT / "final_test_metrics.json").write_text(json.dumps(test_result, indent=2), encoding="utf-8")
    pd.DataFrame([{**test_metrics_threshold, "selected_model": best_name,
                   "threshold_source": "validation maximum F1"}]).to_csv(OUT / "final_test_metrics.csv", index=False)
    pd.DataFrame(budgets).to_csv(OUT / "test_alert_budget_metrics.csv", index=False)
    cm = confusion_matrix(y_test, (test_scores >= threshold).astype(np.int8), labels=[0, 1])
    pd.DataFrame(cm, index=["actual_nonfraud", "actual_fraud"], columns=["pred_nonfraud", "pred_fraud"]).to_csv(OUT / "confusion_matrix.csv")
    test_ids = raw.loc[masks["test"], ["step", "type", "amount", "isFraud"]].copy()
    test_ids["score"] = test_scores
    test_ids["prediction_at_validation_threshold"] = (test_scores >= threshold).astype(np.int8)
    test_ids.to_csv(OUT / "test_predictions.csv", index=False)

    # Simple visualizations for the eventual report.
    fig, ax = plt.subplots(figsize=(7, 5))
    from sklearn.metrics import RocCurveDisplay
    RocCurveDisplay.from_predictions(y_test, test_scores, ax=ax)
    ax.set_title(f"PaySim temporal test ROC — {best_name}")
    fig.tight_layout(); fig.savefig(OUT / "test_roc_curve.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5))
    from sklearn.metrics import PrecisionRecallDisplay
    PrecisionRecallDisplay.from_predictions(y_test, test_scores, ax=ax)
    ax.set_title(f"PaySim temporal test Precision-Recall — {best_name}")
    fig.tight_layout(); fig.savefig(OUT / "test_pr_curve.png", dpi=150); plt.close(fig)

    print("\nSELECTED MODEL:", best_name, flush=True)
    print("TEST METRICS at validation-chosen threshold:", json.dumps(test_metrics_threshold, indent=2), flush=True)
    print("TEST FIXED BUDGETS:", pd.DataFrame(budgets).to_string(index=False), flush=True)
    print(f"Outputs saved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
