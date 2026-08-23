"""Subject-level (cluster) bootstrap confidence intervals for evaluation metrics.

Recordings from the same subject are correlated, so resampling individual
recordings would understate uncertainty. Every bootstrap replicate here draws
whole subjects with replacement and pools their rows, which is the statistically
appropriate resampling unit for this grouped dataset.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

METRIC_NAMES = (
    "roc_auc",
    "pr_auc",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
)


def _counts(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    y = np.asarray(y)
    pred = np.asarray(pred)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    return tp, tn, fp, fn


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def threshold_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Confusion-matrix metrics from hard predictions (single-class safe)."""
    tp, tn, fp, fn = _counts(y, pred)
    sens = _ratio(tp, tp + fn)
    spec = _ratio(tn, tn + fp)
    prec = _ratio(tp, tp + fp)
    acc = _ratio(tp + tn, tp + tn + fp + fn)
    if np.isnan(prec) or np.isnan(sens) or (prec + sens) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * prec * sens / (prec + sens)
    return {
        "accuracy": acc,
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "f1": f1,
    }


def rank_metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    """Threshold-free ranking metrics; NaN when only one class is present."""
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
    }


def all_metrics(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    metrics = rank_metrics(y, prob)
    metrics.update(threshold_metrics(y, pred))
    return metrics


def bootstrap_metrics(
    y: np.ndarray,
    prob: np.ndarray,
    pred: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Percentile bootstrap CIs, resampling whole subjects (clusters).

    ``groups`` labels the resampling unit of each row. For subject-level inputs
    pass one row per subject with ``groups`` equal to the subject ids.
    """
    y = np.asarray(y)
    prob = np.asarray(prob)
    pred = np.asarray(pred)
    groups = np.asarray(groups)

    subjects = np.unique(groups)
    by_subject = [np.flatnonzero(groups == s) for s in subjects]
    rng = np.random.default_rng(seed)

    point = all_metrics(y, prob, pred)
    draws: dict[str, list[float]] = {k: [] for k in METRIC_NAMES}
    n = len(subjects)
    for _ in range(n_boot):
        pick = rng.integers(0, n, size=n)
        idx = np.concatenate([by_subject[i] for i in pick])
        replicate = all_metrics(y[idx], prob[idx], pred[idx])
        for name in METRIC_NAMES:
            draws[name].append(replicate[name])

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out: dict[str, dict[str, float]] = {}
    for name in METRIC_NAMES:
        arr = np.asarray(draws[name], dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size:
            out[name] = {
                "point": point[name],
                "ci_low": float(np.percentile(arr, lo_q)),
                "ci_high": float(np.percentile(arr, hi_q)),
                "n_boot_valid": int(arr.size),
            }
        else:
            out[name] = {
                "point": point[name],
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "n_boot_valid": 0,
            }
    return out
