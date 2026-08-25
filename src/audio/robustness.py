"""Repeated subject-grouped cross-validation for the frozen audio baseline.

Phase 2B reported metrics from a *single* subject-grouped 5-fold split
(``seed=42``). This module repeats that entire leakage-safe pipeline across many
re-shuffled fold assignments to check whether the reported performance is an
artefact of one lucky split (Phase 2C objectives 3-5). Each repeat re-shuffles
subjects into folds with a different seed; the per-fold pipeline is *identical*
to Phase 2B:

    StandardScaler.fit(inner-train subjects only)
    model.fit(scaled inner-train)              # class_weight="balanced"
    Platt/sigmoid calibration on the inner-validation fold (cv="prefit")
    threshold = argmax-F1 on calibrated validation probs   # never sees test
    predict calibrated probs on the held-out test fold

Two complementary uncertainty views are kept deliberately distinct:

* the Phase 2B *single-split* subject-cluster bootstrap 95% CI (carried through
  from the frozen reference, unchanged), and
* the *across-repeats* distribution computed here (mean / std / 2.5-97.5%
  percentile band), which captures variability due to the choice of split.

Threshold stability is summarised over every fold of every repeat. Nothing here
reads, retrains or overwrites the Phase 2B ``artifacts/`` or metrics; outputs are
written by :mod:`src.audio.phase2c` under ``artifacts_audio/phase2c/``.
"""

from __future__ import annotations

import numpy as np

from src.audio.baseline_cv import (
    MODELS,
    _augmented_metrics,
    _fit_fold_model,
    make_models,
)
from src.data import (
    Dataset,
    assert_no_group_leakage,
    cv_splits,
    fit_scaler,
    inner_subject_split,
    subject_labels,
)

# Metrics tracked per repeat (subject-level). Threshold-free metrics first.
REPEAT_METRIC_NAMES = (
    "roc_auc",
    "pr_auc",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "brier_score",
    "accuracy",
    "f1",
)

REFERENCE_MODEL = "logistic_regression"


def _fold_predictions(data: Dataset, model_name: str, *, n_splits: int, seed: int):
    """Pooled out-of-fold prob / threshold per recording for one model + seed.

    Reproduces exactly one subject-grouped CV run of the Phase 2B pipeline. Every
    outer/inner split is checked for subject leakage before use.
    """
    n = len(data.y)
    prob = np.full(n, np.nan)
    thr = np.full(n, np.nan)
    fold_thresholds: list[float] = []
    calibrated: list[bool] = []
    for train_all, test in cv_splits(data, n_splits=n_splits, seed=seed):
        assert_no_group_leakage(train_all, test, data.groups)
        inner_train, val = inner_subject_split(data, train_all, seed=seed)
        assert_no_group_leakage(inner_train, test, data.groups)
        assert_no_group_leakage(val, test, data.groups)
        assert_no_group_leakage(inner_train, val, data.groups)

        scaler = fit_scaler(data.X.iloc[inner_train])  # training-only fit
        x_tr = scaler.transform(data.X.iloc[inner_train].to_numpy())
        x_val = scaler.transform(data.X.iloc[val].to_numpy())
        x_test = scaler.transform(data.X.iloc[test].to_numpy())
        base = make_models(seed)[model_name]
        p, t, was_cal = _fit_fold_model(
            base, x_tr, data.y[inner_train], x_val, data.y[val], x_test
        )
        prob[test] = p
        thr[test] = t
        fold_thresholds.append(float(t))
        calibrated.append(bool(was_cal))
    return prob, thr, fold_thresholds, calibrated


def reference_oof(
    data: Dataset, *, model_name: str = REFERENCE_MODEL, n_splits: int = 5, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical (seed=42) out-of-fold ``(prob, pred, threshold)`` per recording.

    Uses the same pipeline that produced the Phase 2B numbers, so downstream
    confound analysis reasons about the *reported* predictions.
    """
    prob, thr, _thresholds, _cal = _fold_predictions(
        data, model_name, n_splits=n_splits, seed=seed
    )
    pred = (prob >= thr).astype(int)
    return prob, pred, thr


def _subject_pool(data: Dataset, prob: np.ndarray, thr: np.ndarray):
    """Aggregate recording-level OOF to one prediction per subject."""
    subjects, subj_labels = subject_labels(data)
    p_subj = np.array([prob[data.groups == s].mean() for s in subjects])
    thr_subj = np.array([thr[data.groups == s][0] for s in subjects])
    pred_subj = (p_subj >= thr_subj).astype(int)
    return subj_labels, p_subj, pred_subj


def _one_repeat(data: Dataset, model_name: str, *, n_splits: int, seed: int):
    prob, thr, fold_thresholds, _cal = _fold_predictions(
        data, model_name, n_splits=n_splits, seed=seed
    )
    subj_labels, p_subj, pred_subj = _subject_pool(data, prob, thr)
    metrics = _augmented_metrics(subj_labels, p_subj, pred_subj)
    return metrics, fold_thresholds


def _distribution(values, *, thresholds: bool = False) -> dict:
    """Summary stats for a sample: mean/std and a 2.5-97.5 percentile band."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        base = dict.fromkeys(
            ("mean", "std", "min", "max", "p2_5", "p50", "p97_5"), float("nan")
        )
        base["n"] = 0
        return base
    out = {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p2_5": float(np.percentile(arr, 2.5)),
        "p50": float(np.percentile(arr, 50.0)),
        "p97_5": float(np.percentile(arr, 97.5)),
    }
    if thresholds:
        out["iqr"] = float(np.percentile(arr, 75.0) - np.percentile(arr, 25.0))
        out["frac_non_default"] = float(np.mean(np.abs(arr - 0.5) > 1e-9))
    return out


def repeated_grouped_cv(
    data: Dataset,
    *,
    models: tuple[str, ...] = MODELS,
    n_splits: int = 5,
    n_repeats: int = 50,
    base_seed: int = 42,
) -> dict:
    """Repeat subject-grouped CV across ``n_repeats`` re-shuffled fold assignments.

    Returns, per model, the across-repeats distribution of each metric plus a
    threshold-stability summary over all ``n_repeats * n_splits`` fold thresholds.
    Fully deterministic: repeat ``r`` uses ``base_seed + r`` for the fold split,
    the inner split and the estimator.
    """
    seeds = [base_seed + r for r in range(n_repeats)]
    results: dict[str, dict] = {}
    for name in models:
        per_repeat = {m: [] for m in REPEAT_METRIC_NAMES}
        all_thresholds: list[float] = []
        for seed in seeds:
            metrics, fold_thresholds = _one_repeat(data, name, n_splits=n_splits, seed=seed)
            for m in REPEAT_METRIC_NAMES:
                per_repeat[m].append(metrics[m])
            all_thresholds.extend(fold_thresholds)
        results[name] = {
            "n_repeats": n_repeats,
            "n_splits": n_splits,
            "base_seed": base_seed,
            "seeds": seeds,
            "metrics": {m: _distribution(per_repeat[m]) for m in REPEAT_METRIC_NAMES},
            "per_repeat": {m: [float(v) for v in per_repeat[m]] for m in REPEAT_METRIC_NAMES},
            "threshold_stability": _distribution(all_thresholds, thresholds=True),
            "fold_thresholds": [float(t) for t in all_thresholds],
        }
    return results
