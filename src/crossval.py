"""Subject-grouped 5-fold cross-validation with fold-specific preprocessing.

Every recording is tested exactly once (out-of-fold), so pooling the held-out
predictions gives a full-length prediction vector per model for honest,
cross-validated metrics with subject-level bootstrap confidence intervals.

Guarantees enforced here:
  * splits are grouped by subject (no recording leaks across folds);
  * the StandardScaler and class weights are fit on each fold's inner-training
    subjects only;
  * decision thresholds for the neural nets are tuned on a subject-grouped inner
    validation split, never on the held-out test fold.

This module is evaluation-only. It does NOT read, retrain or overwrite the
deployed artifacts (``artifacts/``) or ``reports/metrics.json``.

Run with ``python -m src.crossval``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.data import (
    Dataset,
    assert_no_group_leakage,
    class_weights,
    cv_splits,
    fit_scaler,
    inner_subject_split,
    load_dataset,
    subject_labels,
)
from src.evaluate import plot_pr_curves, plot_roc_curves
from src.models import GroupAttentionNet, MLPBaseline
from src.reproducibility import repro_metadata
from src.stats import all_metrics, bootstrap_metrics
from src.train import train_torch_model, tune_threshold

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

ALL_MODELS = ("logistic_regression", "random_forest", "svm_rbf", "mlp", "attention")
TORCH_HPARAMS = {
    "mlp": {"lr": 1e-3, "weight_decay": 1e-4, "patience": 30},
    "attention": {"lr": 5e-4, "weight_decay": 1e-3, "patience": 40},
}


def _sklearn_models(seed: int) -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=5000, C=0.1, class_weight="balanced", random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=-1
        ),
        "svm_rbf": SVC(C=1.0, probability=True, class_weight="balanced", random_state=seed),
    }


def _torch_predict(model: torch.nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
    return torch.sigmoid(logits).numpy()


def _record(state: dict, name: str, test_idx: np.ndarray, groups: np.ndarray,
            y_test: np.ndarray, prob: np.ndarray, threshold: float) -> None:
    """Store OOF predictions and per-fold recording/subject metrics for one model."""
    pred = (prob >= threshold).astype(int)
    state["oof"][name]["prob"][test_idx] = prob
    state["oof"][name]["pred"][test_idx] = pred
    state["oof"][name]["thr"][test_idx] = threshold
    state["fold_thresholds"][name].append(float(threshold))
    state["per_fold_rec"][name].append(all_metrics(y_test, prob, pred))

    subs = np.unique(groups)
    y_subj = np.array([y_test[groups == s][0] for s in subs])
    p_subj = np.array([prob[groups == s].mean() for s in subs])
    d_subj = (p_subj >= threshold).astype(int)
    state["per_fold_subj"][name].append(all_metrics(y_subj, p_subj, d_subj))


def _summarise(fold_dicts: list[dict]) -> dict:
    keys = fold_dicts[0].keys()
    out = {}
    for key in keys:
        vals = [d[key] for d in fold_dicts]
        out[key] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "per_fold": [float(v) for v in vals],
        }
    return out


def _aggregate(data: Dataset, subjects: np.ndarray, subj_labels: np.ndarray,
               state: dict, name: str, *, n_boot: int, seed: int) -> dict:
    prob = state["oof"][name]["prob"]
    pred = state["oof"][name]["pred"].astype(int)
    groups = data.groups

    p_subj = np.array([prob[groups == s].mean() for s in subjects])
    thr_subj = np.array([state["oof"][name]["thr"][groups == s][0] for s in subjects])
    pred_subj = (p_subj >= thr_subj).astype(int)

    return {
        "recording_level": bootstrap_metrics(
            data.y, prob, pred, groups, n_boot=n_boot, seed=seed
        ),
        "subject_level": bootstrap_metrics(
            subj_labels, p_subj, pred_subj, subjects, n_boot=n_boot, seed=seed
        ),
        "per_fold_recording": _summarise(state["per_fold_rec"][name]),
        "per_fold_subject": _summarise(state["per_fold_subj"][name]),
        "fold_thresholds": state["fold_thresholds"][name],
    }


def run_cv(data: Dataset, *, n_splits: int = 5, seed: int = 42,
           epochs: int = 200, n_boot: int = 2000) -> tuple[dict, dict]:
    """Return ``(results, oof)`` for all models. ``oof`` holds pooled predictions."""
    n = len(data.y)
    group_sizes = [len(cols) for cols in data.feature_groups.values()]
    subjects, subj_labels = subject_labels(data)

    state = {
        "oof": {
            m: {
                "prob": np.full(n, np.nan),
                "pred": np.full(n, np.nan),
                "thr": np.full(n, np.nan),
            }
            for m in ALL_MODELS
        },
        "per_fold_rec": {m: [] for m in ALL_MODELS},
        "per_fold_subj": {m: [] for m in ALL_MODELS},
        "fold_thresholds": {m: [] for m in ALL_MODELS},
    }

    np.random.seed(seed)
    for fold, (train_all, test) in enumerate(cv_splits(data, n_splits=n_splits, seed=seed)):
        assert_no_group_leakage(train_all, test, data.groups)
        train, val = inner_subject_split(data, train_all, seed=seed)
        assert_no_group_leakage(train, test, data.groups)
        assert_no_group_leakage(val, test, data.groups)

        scaler = fit_scaler(data.X.iloc[train])
        x_train = scaler.transform(data.X.iloc[train].to_numpy())
        x_val = scaler.transform(data.X.iloc[val].to_numpy())
        x_test = scaler.transform(data.X.iloc[test].to_numpy())
        y_train, y_val, y_test = data.y[train], data.y[val], data.y[test]
        g_test = data.groups[test]

        weights = class_weights(y_train)
        pos_weight = float(weights[1] / weights[0])

        for name, model in _sklearn_models(seed).items():
            model.fit(x_train, y_train)
            prob = model.predict_proba(x_test)[:, 1]
            _record(state, name, test, g_test, y_test, prob, 0.5)

        torch.manual_seed(seed + fold)
        mlp = train_torch_model(
            MLPBaseline(x_train.shape[1]), x_train, y_train, x_val, y_val,
            pos_weight, epochs=epochs, **TORCH_HPARAMS["mlp"],
        )
        mlp_thr = tune_threshold(y_val, _torch_predict(mlp, x_val))
        _record(state, "mlp", test, g_test, y_test, _torch_predict(mlp, x_test), mlp_thr)

        torch.manual_seed(seed + 100 + fold)
        attn = train_torch_model(
            GroupAttentionNet(group_sizes), x_train, y_train, x_val, y_val,
            pos_weight, epochs=epochs, **TORCH_HPARAMS["attention"],
        )
        attn_thr = tune_threshold(y_val, _torch_predict(attn, x_val))
        _record(state, "attention", test, g_test, y_test, _torch_predict(attn, x_test), attn_thr)

    results = {
        name: _aggregate(data, subjects, subj_labels, state, name, n_boot=n_boot, seed=seed)
        for name in ALL_MODELS
    }
    return results, state["oof"]


def _print_summary(results: dict) -> None:
    print(f"\n{'model':<20} {'rec ROC-AUC [95% CI]':<26} {'subj ROC-AUC [95% CI]':<26}")
    for name, res in results.items():
        rec = res["recording_level"]["roc_auc"]
        sub = res["subject_level"]["roc_auc"]
        print(
            f"{name:<20} "
            f"{rec['point']:.3f} [{rec['ci_low']:.3f}, {rec['ci_high']:.3f}]".ljust(26)
            + " "
            + f"{sub['point']:.3f} [{sub['ci_low']:.3f}, {sub['ci_high']:.3f}]".ljust(26)
        )


def main(n_splits: int = 5, seed: int = 42, epochs: int = 200, n_boot: int = 2000) -> dict:
    REPORTS.mkdir(exist_ok=True)
    data = load_dataset()
    results, oof = run_cv(data, n_splits=n_splits, seed=seed, epochs=epochs, n_boot=n_boot)

    report = {
        "config": {
            "evaluation": "subject-grouped 5-fold cross-validation (pooled out-of-fold)",
            "n_splits": n_splits,
            "seed": seed,
            "epochs": epochs,
            "n_boot": n_boot,
            "n_features": int(data.X.shape[1]),
            "preprocessing": "StandardScaler fit on each fold's inner-training subjects only",
            "threshold_selection": (
                "max-F1 on subject-grouped inner validation (mlp, attention); "
                "fixed 0.5 for the class-weighted sklearn baselines"
            ),
            "ci_method": "percentile bootstrap (2.5-97.5%) resampling whole subjects",
            "note": (
                "Cross-validated generalisation estimates. These do NOT represent the deployed "
                "models: the served artifacts in artifacts/ and reports/metrics.json are a single "
                "fixed split and are left unchanged."
            ),
        },
        "reproducibility": repro_metadata(seed),
        "models": results,
    }
    (REPORTS / "cv_metrics.json").write_text(json.dumps(report, indent=2))

    curves = {m: (data.y, oof[m]["prob"]) for m in ALL_MODELS}
    plot_roc_curves(curves, REPORTS / "cv_roc_curves.png")
    plot_pr_curves(curves, REPORTS / "cv_pr_curves.png")

    _print_summary(results)
    print("\nWrote reports/cv_metrics.json, reports/cv_roc_curves.png, reports/cv_pr_curves.png")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    main(n_splits=args.n_splits, seed=args.seed, epochs=args.epochs, n_boot=args.n_boot)
