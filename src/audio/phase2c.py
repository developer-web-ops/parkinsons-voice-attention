"""Phase 2C orchestrator: robustness + explainability for the frozen baseline.

Reads the Phase 2B ``artifacts_audio/metrics/cv_metrics.json`` (never writes it),
freezes the Logistic-Regression block as the immutable reference, then runs and
persists every Phase 2C analysis under ``artifacts_audio/phase2c/``:

* ``frozen_baseline_reference.json`` — the frozen LR reference (+ provenance hash
  of the source metrics file and a snapshot of all three models).
* ``repeated_cv.json``             — repeated subject-grouped CV distributions
  and threshold stability (:mod:`src.audio.robustness`).
* ``explainability.json``          — coefficients, coefficient stability, SHAP,
  family aggregation, artefact-family share, coef/SHAP rank agreement.
* ``feature_importance.csv``       — per-feature merged importance table.
* ``confound_fairness.json``       — confound probes + task-based fairness
  (:mod:`src.audio.confounds`).
* ``phase2c_summary.json``         — everything above plus the comparison of the
  frozen single-split baseline against the repeated-CV distribution.
* ``top_features.png``             — optional SHAP bar chart (best-effort).

Nothing here modifies Phase 1 (``artifacts/``, ``reports/``) or the Phase 2B
baseline artifacts. Run with ``python -m src.audio.phase2c``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.audio import confounds as confound_mod
from src.audio import explain as explain_mod
from src.audio.baseline_cv import MODELS
from src.audio.config import DEFAULT_CONFIG, AudioConfig
from src.audio.dataset import label_balance, load_dataset
from src.audio.families import FAMILY_ORDER
from src.audio.features import FEATURE_CSV, load_feature_frame
from src.audio.reproducibility import audio_repro_metadata
from src.audio.robustness import REPEAT_METRIC_NAMES, repeated_grouped_cv

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_AUDIO = ROOT / "artifacts_audio"
CV_METRICS_JSON = ARTIFACTS_AUDIO / "metrics" / "cv_metrics.json"
PHASE2C_DIR = ARTIFACTS_AUDIO / "phase2c"

REFERENCE_MODEL = "logistic_regression"
# Subject-level metrics compared frozen-vs-repeated (present in both worlds).
COMPARE_METRICS = (
    "roc_auc",
    "pr_auc",
    "sensitivity",
    "specificity",
    "balanced_accuracy",
    "brier_score",
    "accuracy",
    "f1",
)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def freeze_baseline(cv_metrics_path: Path = CV_METRICS_JSON) -> dict:
    """Snapshot the Phase 2B LR block as the immutable Phase 2C reference.

    The source file is read and hashed but **never** modified; the returned dict
    records ``modified_phase2b=False`` explicitly. A compact subject-level
    snapshot of all three models is carried for context.
    """
    raw = Path(cv_metrics_path).read_text(encoding="utf-8")
    metrics = json.loads(raw)
    lr = metrics["models"][REFERENCE_MODEL]
    src_path = Path(cv_metrics_path)
    try:
        source_file = str(src_path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:  # path outside the repo (e.g. a test tmp dir)
        source_file = src_path.as_posix()
    snapshot = {
        m: {
            "roc_auc": metrics["models"][m]["subject_level"]["roc_auc"],
            "balanced_accuracy": metrics["models"][m]["subject_level"]["balanced_accuracy"],
        }
        for m in metrics["models"]
    }
    repro = metrics.get("reproducibility", {})
    return {
        "reference_model": REFERENCE_MODEL,
        "modified_phase2b": False,
        "source_file": source_file,
        "source_sha256": _sha256_text(raw),
        "primary_metric_level": metrics.get("config", {}).get("primary_metric_level"),
        "data_balance": metrics.get("data_balance"),
        "subject_level": lr["subject_level"],
        "recording_level": lr["recording_level"],
        "per_fold_subject": lr["per_fold_subject"],
        "per_fold_recording": lr["per_fold_recording"],
        "fold_thresholds": lr["fold_thresholds"],
        "folds_calibrated": lr["folds_calibrated"],
        "all_models_subject_snapshot": snapshot,
        "provenance": {
            "git_commit": repro.get("git_commit"),
            "feature_dataset_sha256": repro.get("feature_dataset_sha256"),
            "dataset_manifest_sha256": repro.get("dataset_manifest_sha256"),
        },
    }


def compare_to_repeated(frozen: dict, repeated_lr: dict) -> dict:
    """Section F: frozen single-split baseline vs the repeated-CV distribution.

    Two distinct uncertainty views are reported side by side: the frozen
    single-split subject-cluster bootstrap 95% CI, and the across-repeats
    distribution (mean / std / 2.5-97.5% band). For each metric we flag whether
    the frozen point sits inside the repeated band and the mean shift.
    """
    rows = {}
    for metric in COMPARE_METRICS:
        if metric not in frozen["subject_level"] or metric not in repeated_lr["metrics"]:
            continue
        fz = frozen["subject_level"][metric]
        rp = repeated_lr["metrics"][metric]
        point = fz["point"]
        in_band = (
            not np.isnan(rp["p2_5"])
            and not np.isnan(rp["p97_5"])
            and rp["p2_5"] <= point <= rp["p97_5"]
        )
        rows[metric] = {
            "frozen_point": point,
            "frozen_ci95": [fz["ci_low"], fz["ci_high"]],
            "repeated_mean": rp["mean"],
            "repeated_std": rp["std"],
            "repeated_band_2_5_97_5": [rp["p2_5"], rp["p97_5"]],
            "mean_minus_frozen": float(rp["mean"] - point),
            "frozen_point_in_repeated_band": bool(in_band),
        }
    return {
        "reference_model": REFERENCE_MODEL,
        "n_repeats": repeated_lr["n_repeats"],
        "metrics": rows,
        "note": (
            "frozen_ci95 is the Phase 2B single-split subject-cluster bootstrap CI "
            "(sampling variance at one split); repeated_band is the across-splits "
            "distribution (split-choice variance). They answer different questions "
            "and are both reported without conflation."
        ),
    }


def _feature_importance_table(coef_ranked, shap_ranked, stability) -> pd.DataFrame:
    """Merge the three per-feature views into one table, ordered by |coef|."""
    shap_by = {r["feature"]: r for r in (shap_ranked or [])}
    stab_by = {r["feature"]: r for r in stability}
    rows = []
    for r in coef_ranked:
        f = r["feature"]
        s = shap_by.get(f, {})
        st = stab_by.get(f, {})
        rows.append(
            {
                "feature": f,
                "family": r["family"],
                "coef": r["coef"],
                "abs_coef": r["abs_coef"],
                "direction": r["direction"],
                "mean_abs_shap": s.get("mean_abs_shap"),
                "mean_signed_shap": s.get("mean_signed_shap"),
                "mean_coef_cv": st.get("mean_coef"),
                "std_coef_cv": st.get("std_coef"),
                "sign_consistency_cv": st.get("sign_consistency"),
            }
        )
    return pd.DataFrame(rows)


def _explainability(data, seed: int) -> dict:
    """Run every explainability view; SHAP is best-effort (optional dependency)."""
    coef_ranked = explain_mod.coefficient_importance(data, seed=seed)
    stability = explain_mod.coefficient_stability(data, seed=seed)
    shap_ranked = None
    shap_error = None
    try:
        shap_ranked, _values, _names = explain_mod.shap_importance(data, seed=seed)
    except ImportError as exc:  # pragma: no cover - shap present in the pinned env
        shap_error = f"shap unavailable: {exc}"

    if shap_ranked is not None:
        family_rows = explain_mod.family_importance(shap_ranked)
        importance_basis = "mean_abs_shap"
    else:
        # Fall back to |coef| so family aggregation still works without shap.
        family_rows = explain_mod.family_importance(
            [{"feature": r["feature"], "family": r["family"], "mean_abs_shap": r["abs_coef"]}
             for r in coef_ranked]
        )
        importance_basis = "abs_coef (shap unavailable)"

    return {
        "coef_ranked": coef_ranked,
        "shap_ranked": shap_ranked,
        "shap_available": shap_ranked is not None,
        "shap_error": shap_error,
        "stability": stability,
        "family_importance": family_rows,
        "family_importance_basis": importance_basis,
        "artefact_family_share": explain_mod.artefact_family_share(family_rows),
        "rank_agreement": (
            explain_mod.rank_agreement(coef_ranked, shap_ranked)
            if shap_ranked is not None
            else None
        ),
    }


def _maybe_plot(shap_ranked, out_path: Path, *, top_k: int = 20) -> str | None:
    """Best-effort SHAP bar chart. Never fatal; returns the relpath or None."""
    if not shap_ranked:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = shap_ranked[:top_k][::-1]
        palette = dict(zip(FAMILY_ORDER, plt.cm.tab10.colors, strict=False))
        colors = [palette.get(r["family"], "#888888") for r in top]
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh([r["feature"] for r in top], [r["mean_abs_shap"] for r in top], color=colors)
        ax.set_xlabel("mean(|SHAP|)")
        ax.set_title(f"Top {top_k} eGeMAPS features (LR baseline)")
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=palette[f])
            for f in FAMILY_ORDER
            if any(r["family"] == f for r in top)
        ]
        labels = [f for f in FAMILY_ORDER if any(r["family"] == f for r in top)]
        ax.legend(handles, labels, fontsize=8, loc="lower right")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        return str(out_path.relative_to(ROOT)).replace("\\", "/")
    except Exception as exc:  # pragma: no cover - plotting is non-essential
        return f"plot_skipped: {exc}"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(
    *,
    n_repeats: int = 50,
    n_splits: int = 5,
    base_seed: int = 42,
    seed: int = 42,
    config: AudioConfig = DEFAULT_CONFIG,
    feature_csv: Path = FEATURE_CSV,
    cv_metrics_path: Path = CV_METRICS_JSON,
    out_dir: Path = PHASE2C_DIR,
    make_plot: bool = True,
) -> dict:
    """Execute the full Phase 2C analysis and write all artifacts under ``out_dir``."""
    frame = load_feature_frame(feature_csv)
    data = load_dataset(feature_csv)

    # A/B/C — freeze the reference, then repeated subject-grouped CV.
    frozen = freeze_baseline(cv_metrics_path)
    repeated = repeated_grouped_cv(
        data, models=MODELS, n_splits=n_splits, n_repeats=n_repeats, base_seed=base_seed
    )

    # D — explainability.
    explain_out = _explainability(data, seed=seed)

    # E — confounds + task-based fairness (uses the frozen model's OOF predictions).
    confound_out = confound_mod.analyze(
        frame, data, explain_out["coef_ranked"], n_splits=n_splits, seed=seed
    )

    # F — frozen single-split baseline vs repeated-CV distribution.
    comparison = compare_to_repeated(frozen, repeated[REFERENCE_MODEL])

    # Persist per-analysis artifacts.
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "frozen_baseline_reference.json", frozen)
    _write_json(out_dir / "repeated_cv.json", repeated)
    _write_json(out_dir / "explainability.json", explain_out)
    _write_json(out_dir / "confound_fairness.json", confound_out)

    table = _feature_importance_table(
        explain_out["coef_ranked"], explain_out["shap_ranked"], explain_out["stability"]
    )
    (out_dir / "feature_importance.csv").parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "feature_importance.csv", index=False)

    plot_rel = (
        _maybe_plot(explain_out["shap_ranked"], out_dir / "top_features.png")
        if make_plot
        else None
    )

    summary = {
        "phase": "2C",
        "scope": "robustness_and_explainability_only",
        "config": {
            "reference_model": REFERENCE_MODEL,
            "n_repeats": n_repeats,
            "n_splits": n_splits,
            "base_seed": base_seed,
            "seed": seed,
            "repeat_metric_names": list(REPEAT_METRIC_NAMES),
        },
        "data_balance": label_balance(frame),
        "reproducibility": audio_repro_metadata(seed=seed, config=config, feature_csv=feature_csv),
        "frozen_baseline_reference": frozen,
        "repeated_cv": repeated,
        "explainability": explain_out,
        "confound_fairness": confound_out,
        "comparison_frozen_vs_repeated": comparison,
        "artifacts": {
            "frozen_baseline_reference": "frozen_baseline_reference.json",
            "repeated_cv": "repeated_cv.json",
            "explainability": "explainability.json",
            "feature_importance_csv": "feature_importance.csv",
            "confound_fairness": "confound_fairness.json",
            "top_features_png": plot_rel,
        },
    }
    _write_json(out_dir / "phase2c_summary.json", summary)
    _print_summary(summary)
    return summary


def _print_summary(summary: dict) -> None:
    lr = summary["repeated_cv"][REFERENCE_MODEL]["metrics"]
    frozen = summary["frozen_baseline_reference"]["subject_level"]
    print("\nPhase 2C — robustness & explainability")
    print(
        f"  repeated CV: {summary['config']['n_repeats']} repeats x "
        f"{summary['config']['n_splits']} folds (LR reference)"
    )
    for m in ("roc_auc", "balanced_accuracy", "brier_score"):
        print(
            f"  {m:<18} frozen={frozen[m]['point']:.3f} "
            f"repeated={lr[m]['mean']:.3f}+/-{lr[m]['std']:.3f} "
            f"[{lr[m]['p2_5']:.3f}, {lr[m]['p97_5']:.3f}]"
        )
    ex = summary["explainability"]
    top = (ex["shap_ranked"] or ex["coef_ranked"])[:5]
    print("  top features: " + ", ".join(r["feature"] for r in top))
    share = ex["artefact_family_share"]["combined_share"]
    print(f"  artefact-susceptible family share: {share:.3f}")
    flag = summary["confound_fairness"]["flags"]["duration_confound_suspected"]
    print(f"  duration-confound suspected: {flag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-repeats", type=int, default=50)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    run(
        n_repeats=args.n_repeats,
        n_splits=args.n_splits,
        base_seed=args.base_seed,
        seed=args.seed,
        make_plot=not args.no_plot,
    )
