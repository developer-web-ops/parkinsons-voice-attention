"""Subject-grouped cross-validation for the eGeMAPSv02 acoustic baseline.

Pipeline per outer fold (mirrors the Phase 1 methodology in ``src/crossval.py``,
but torch-free and audio-specific):

    outer split (subject-grouped)  -> assert no subject leakage
      inner split (subject-grouped) -> (inner-train, validation)
        StandardScaler.fit(inner-train)             # fit on training only
        model.fit(scaled inner-train)               # class_weight="balanced"
        CalibratedClassifierCV(frozen model).fit(validation)   # Platt / sigmoid
        threshold = argmax-F1 on calibrated validation probs   # never sees test
      predict calibrated probs on the held-out test fold
    pool out-of-fold predictions -> recording- and subject-level metrics + 95% CIs

Every recording is tested exactly once. Confidence intervals use the Phase 1
subject-cluster percentile bootstrap. Reported metrics reuse ``src.stats`` and
add balanced accuracy and Brier score (a calibration check).

This module writes only under ``artifacts_audio/``. It never reads, retrains or
overwrites the Phase 1 ``artifacts/`` or ``reports/`` deliverables.

Run with ``python -m src.audio.baseline_cv``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.audio.config import DEFAULT_CONFIG, AudioConfig
from src.audio.dataset import label_balance, load_dataset
from src.audio.features import FEATURE_CSV, load_feature_frame
from src.audio.reproducibility import audio_repro_metadata
from src.data import (
    Dataset,
    assert_no_group_leakage,
    cv_splits,
    fit_scaler,
    inner_subject_split,
    subject_labels,
)
from src.stats import METRIC_NAMES, all_metrics, threshold_metrics

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_AUDIO = ROOT / "artifacts_audio"
METRICS_DIR = ARTIFACTS_AUDIO / "metrics"
MODELS_DIR = ARTIFACTS_AUDIO / "models"
CV_METRICS_JSON = METRICS_DIR / "cv_metrics.json"

MODELS = ("logistic_regression", "svm_rbf", "random_forest")
AUDIO_METRIC_NAMES = (*METRIC_NAMES, "balanced_accuracy", "brier_score")
THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)


def make_models(seed: int) -> dict:
    """Simple, defensible baselines; all class-weighted for the PD/HC imbalance."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=5000, C=0.1, class_weight="balanced", random_state=seed
        ),
        "svm_rbf": SVC(C=1.0, kernel="rbf", class_weight="balanced", random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=1
        ),
    }


def _scores(estimator, X: np.ndarray) -> np.ndarray:
    """P(class=1) from a fitted estimator, with a decision-function fallback."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    s = estimator.decision_function(X)
    lo, hi = float(np.min(s)), float(np.max(s))
    return (s - lo) / (hi - lo + 1e-12)


def tune_threshold(y_val: np.ndarray, prob_val: np.ndarray) -> float:
    """Threshold maximising F1 on validation predictions (max-F1, like Phase 1)."""
    best_thr, best_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        pred = (prob_val >= t).astype(int)
        f1 = threshold_metrics(y_val, pred)["f1"]
        if not np.isnan(f1) and f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    return best_thr


def _augmented_metrics(y: np.ndarray, prob: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Phase 1 metrics + balanced accuracy + Brier score."""
    m = dict(all_metrics(y, prob, pred))
    sens, spec = m["sensitivity"], m["specificity"]
    m["balanced_accuracy"] = (
        float("nan") if (np.isnan(sens) or np.isnan(spec)) else 0.5 * (sens + spec)
    )
    y_arr = np.asarray(y, dtype=float)
    prob_arr = np.asarray(prob, dtype=float)
    m["brier_score"] = float(np.mean((prob_arr - y_arr) ** 2)) if y_arr.size else float("nan")
    return m


def bootstrap_audio_metrics(
    y: np.ndarray,
    prob: np.ndarray,
    pred: np.ndarray,
    groups: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Subject-cluster percentile bootstrap (as in ``src.stats``) over the
    augmented metric set. Whole subjects are resampled with replacement."""
    y, prob, pred, groups = map(np.asarray, (y, prob, pred, groups))
    subjects = np.unique(groups)
    by_subject = [np.flatnonzero(groups == s) for s in subjects]
    rng = np.random.default_rng(seed)
    n = len(subjects)

    point = _augmented_metrics(y, prob, pred)
    draws: dict[str, list[float]] = {k: [] for k in AUDIO_METRIC_NAMES}
    for _ in range(n_boot):
        pick = rng.integers(0, n, size=n)
        idx = np.concatenate([by_subject[i] for i in pick])
        rep = _augmented_metrics(y[idx], prob[idx], pred[idx])
        for name in AUDIO_METRIC_NAMES:
            draws[name].append(rep[name])

    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out: dict[str, dict[str, float]] = {}
    for name in AUDIO_METRIC_NAMES:
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


def _summarise(fold_dicts: list[dict]) -> dict:
    out = {}
    for key in fold_dicts[0]:
        vals = [d[key] for d in fold_dicts]
        out[key] = {
            "mean": float(np.nanmean(vals)),
            "std": float(np.nanstd(vals)),
            "per_fold": [float(v) for v in vals],
        }
    return out


def _fit_fold_model(base, x_tr, y_tr, x_val, y_val, x_test):
    """Fit on inner-train, calibrate + tune threshold on validation, score test.

    Platt (sigmoid) calibration is fit on the *whole* validation fold via
    ``cv="prefit"``. That is the appropriate choice for these small folds: it
    uses sklearn's Platt target-smoothing instead of an internal CV split that a
    ~4-subject validation set cannot support. The project pins scikit-learn
    1.6.0, where ``cv="prefit"`` is supported; its deprecation notice is silenced
    locally. Calibration and threshold selection see validation data only — never
    the held-out test fold.
    """
    base.fit(x_tr, y_tr)
    if len(np.unique(y_val)) >= 2:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*cv='prefit'.*", category=UserWarning)
            cal = CalibratedClassifierCV(estimator=base, method="sigmoid", cv="prefit")
            cal.fit(x_val, y_val)
        thr = tune_threshold(y_val, cal.predict_proba(x_val)[:, 1])
        return cal.predict_proba(x_test)[:, 1], thr, True
    # Degenerate single-class validation fold: skip calibration, use default threshold.
    return _scores(base, x_test), 0.5, False


def run_cv(
    data: Dataset, *, n_splits: int = 5, seed: int = 42, n_boot: int = 2000
) -> tuple[dict, dict]:
    n = len(data.y)
    subjects, subj_labels = subject_labels(data)
    oof = {
        m: {
            "prob": np.full(n, np.nan),
            "pred": np.full(n, np.nan),
            "thr": np.full(n, np.nan),
        }
        for m in MODELS
    }
    per_fold_rec = {m: [] for m in MODELS}
    per_fold_subj = {m: [] for m in MODELS}
    fold_thresholds = {m: [] for m in MODELS}
    calibrated_flags = {m: [] for m in MODELS}

    np.random.seed(seed)
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
        y_tr, y_val, y_test = data.y[inner_train], data.y[val], data.y[test]
        g_test = data.groups[test]

        for name, base in make_models(seed).items():
            prob, thr, was_cal = _fit_fold_model(base, x_tr, y_tr, x_val, y_val, x_test)
            pred = (prob >= thr).astype(int)
            oof[name]["prob"][test] = prob
            oof[name]["pred"][test] = pred
            oof[name]["thr"][test] = thr
            fold_thresholds[name].append(float(thr))
            calibrated_flags[name].append(bool(was_cal))
            per_fold_rec[name].append(_augmented_metrics(y_test, prob, pred))

            subs = np.unique(g_test)
            y_s = np.array([y_test[g_test == s][0] for s in subs])
            p_s = np.array([prob[g_test == s].mean() for s in subs])
            d_s = (p_s >= thr).astype(int)
            per_fold_subj[name].append(_augmented_metrics(y_s, p_s, d_s))

    results = {}
    for name in MODELS:
        prob = oof[name]["prob"]
        pred = oof[name]["pred"].astype(int)
        groups = data.groups
        p_subj = np.array([prob[groups == s].mean() for s in subjects])
        thr_subj = np.array([oof[name]["thr"][groups == s][0] for s in subjects])
        pred_subj = (p_subj >= thr_subj).astype(int)
        results[name] = {
            "recording_level": bootstrap_audio_metrics(
                data.y, prob, pred, groups, n_boot=n_boot, seed=seed
            ),
            "subject_level": bootstrap_audio_metrics(
                subj_labels, p_subj, pred_subj, subjects, n_boot=n_boot, seed=seed
            ),
            "per_fold_recording": _summarise(per_fold_rec[name]),
            "per_fold_subject": _summarise(per_fold_subj[name]),
            "fold_thresholds": fold_thresholds[name],
            "folds_calibrated": calibrated_flags[name],
        }
    return results, oof


def fit_full_models(data: Dataset, seed: int, fold_thresholds: dict) -> dict:
    """Fit calibrated models on ALL subjects for explainability/SHAP prep.

    These artifacts are NOT the source of the reported metrics (those come from
    the grouped CV above); they are a single all-data fit, saved with the scaler
    and the median CV operating threshold. Documented as such in the payload.
    """
    scaler = fit_scaler(data.X)
    x_all = scaler.transform(data.X.to_numpy())
    saved = {}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, base in make_models(seed).items():
        model = CalibratedClassifierCV(base, method="sigmoid", cv=5)
        model.fit(x_all, data.y)
        payload = {
            "model": model,
            "scaler": scaler,
            "threshold": float(np.median(fold_thresholds[name])),
            "feature_names": data.feature_names,
            "feature_group": "eGeMAPSv02",
            "note": (
                "Calibrated model fit on ALL subjects for explainability/deployment "
                "prep. Not the source of reported metrics; see cv_metrics.json for the "
                "subject-grouped cross-validated estimates."
            ),
        }
        path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(payload, path)
        saved[name] = path.name
    return saved


def _print_summary(results: dict) -> None:
    print(f"\n{'model':<22}{'subject ROC-AUC [95% CI]':<30}{'subject bal-acc [95% CI]':<30}")
    for name, res in results.items():
        r = res["subject_level"]["roc_auc"]
        b = res["subject_level"]["balanced_accuracy"]
        print(
            f"{name:<22}"
            + f"{r['point']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]".ljust(30)
            + f"{b['point']:.3f} [{b['ci_low']:.3f}, {b['ci_high']:.3f}]".ljust(30)
        )


def main(
    n_splits: int = 5,
    seed: int = 42,
    n_boot: int = 2000,
    config: AudioConfig = DEFAULT_CONFIG,
    feature_csv: Path = FEATURE_CSV,
    save_models: bool = True,
) -> dict:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_feature_frame(feature_csv)
    data = load_dataset(feature_csv)
    balance = label_balance(frame)

    results, _ = run_cv(data, n_splits=n_splits, seed=seed, n_boot=n_boot)
    saved_models = (
        fit_full_models(data, seed, {m: results[m]["fold_thresholds"] for m in MODELS})
        if save_models
        else {}
    )

    report = {
        "config": {
            "evaluation": "subject-grouped stratified 5-fold CV (pooled out-of-fold)",
            "n_splits": n_splits,
            "seed": seed,
            "n_boot": n_boot,
            "feature_set": config.feature_set,
            "feature_level": config.feature_level,
            "n_features": int(data.X.shape[1]),
            "preprocessing": "StandardScaler fit on each fold's inner-training subjects only",
            "calibration": "CalibratedClassifierCV (sigmoid/Platt) fit on inner validation",
            "threshold_selection": "max-F1 on calibrated inner-validation probabilities",
            "ci_method": "percentile bootstrap (2.5-97.5%) resampling whole subjects",
            "primary_metric_level": "subject_level",
            "models_saved_note": (
                "artifacts_audio/models/*.joblib are all-data fits for explainability; "
                "the metrics below are the grouped-CV generalisation estimates."
            ),
        },
        "data_balance": balance,
        "reproducibility": audio_repro_metadata(seed=seed, config=config, feature_csv=feature_csv),
        "saved_models": saved_models,
        "models": results,
    }
    CV_METRICS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_summary(results)
    print(f"\nWrote {CV_METRICS_JSON.relative_to(ROOT)}")
    if saved_models:
        print(f"Saved models: {', '.join(saved_models.values())}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--no-save-models", action="store_true")
    args = parser.parse_args()
    main(
        n_splits=args.n_splits,
        seed=args.seed,
        n_boot=args.n_boot,
        save_models=not args.no_save_models,
    )
