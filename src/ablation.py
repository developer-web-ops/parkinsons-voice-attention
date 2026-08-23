"""Gender ablation: acoustic-only (752) vs acoustic+gender (753).

Runs the full subject-grouped cross-validation twice on identical folds and
subjects, changing only whether the demographic ``gender`` column is present:

  * ``acoustic_only``        - 752 features, ``gender`` dropped from the
    Baseline block (acoustic measurements only);
  * ``acoustic_plus_gender`` - 753 features, ``gender`` kept in the Baseline
    block. This is the SAME feature set as the deployed model.

``StratifiedGroupKFold`` ignores feature values, so both configurations use the
exact same folds and subjects; the only difference is the feature set, making
the per-model delta a clean controlled comparison.

This is evaluation-only. The deployed artifacts (``artifacts/``) and
``reports/metrics.json`` are NOT read, retrained or overwritten, and these
cross-validated numbers do not represent the single-split deployed model.

Run with ``python -m src.ablation``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.crossval import ALL_MODELS, run_cv
from src.data import gender_vector, load_dataset, without_gender
from src.reproducibility import repro_metadata
from src.stats import bootstrap_metrics

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _delta(acoustic: dict, full: dict) -> dict:
    """Per-model, per-level point estimates for both configs and their delta."""
    out: dict = {}
    for model in ALL_MODELS:
        out[model] = {}
        for level in ("recording_level", "subject_level"):
            a_level, f_level = acoustic[model][level], full[model][level]
            row = {}
            for metric in a_level:
                a_pt, f_pt = a_level[metric]["point"], f_level[metric]["point"]
                valid = not (np.isnan(a_pt) or np.isnan(f_pt))
                row[metric] = {
                    "acoustic_only": a_pt,
                    "acoustic_plus_gender": f_pt,
                    "delta_gender_minus_acoustic": (f_pt - a_pt) if valid else float("nan"),
                }
            out[model][level] = row
    return out


def _subgroups(data, oof: dict, *, n_boot: int, seed: int) -> dict:
    """Recording-level performance within each gender code, with subject CIs.

    Gender is reported by its raw dataset code (gender_0 / gender_1); the mapping
    to male/female is not assumed here.
    """
    gender = gender_vector(data)
    if gender is None:
        return {"note": "gender column absent in this configuration; subgroup analysis skipped"}
    out: dict = {}
    for code in sorted(int(g) for g in np.unique(gender)):
        mask = gender == code
        y, groups = data.y[mask], data.groups[mask]
        entry = {
            "n_recordings": int(mask.sum()),
            "n_subjects": len(np.unique(groups)),
            "pd_prevalence_recordings": float(np.mean(y)),
            "models": {},
        }
        for model in ALL_MODELS:
            prob = oof[model]["prob"][mask]
            pred = oof[model]["pred"][mask].astype(int)
            entry["models"][model] = bootstrap_metrics(
                y, prob, pred, groups, n_boot=n_boot, seed=seed
            )
        out[f"gender_{code}"] = entry
    return out


def _point_table(results: dict) -> dict:
    """Compact recording+subject ROC-AUC / F1 point estimates per model."""
    table = {}
    for model in ALL_MODELS:
        table[model] = {
            level: {
                metric: results[model][level][metric]["point"]
                for metric in ("roc_auc", "pr_auc", "f1", "sensitivity", "specificity")
            }
            for level in ("recording_level", "subject_level")
        }
    return table


def main(n_splits: int = 5, seed: int = 42, epochs: int = 200, n_boot: int = 2000) -> dict:
    REPORTS.mkdir(exist_ok=True)
    data = load_dataset()
    acoustic_data = without_gender(data)

    print(f"acoustic+gender config: {data.X.shape[1]} features")
    print(f"acoustic-only  config: {acoustic_data.X.shape[1]} features")

    print("\n[1/2] cross-validating acoustic+gender (753) ...")
    full_results, full_oof = run_cv(
        data, n_splits=n_splits, seed=seed, epochs=epochs, n_boot=n_boot
    )
    print("[2/2] cross-validating acoustic-only (752) ...")
    acoustic_results, acoustic_oof = run_cv(
        acoustic_data, n_splits=n_splits, seed=seed, epochs=epochs, n_boot=n_boot
    )

    report = {
        "config": {
            "evaluation": "subject-grouped 5-fold CV, identical folds for both configurations",
            "n_splits": n_splits,
            "seed": seed,
            "epochs": epochs,
            "n_boot": n_boot,
            "note": (
                "StratifiedGroupKFold ignores feature values, so both configurations share "
                "the same folds and subjects; only the feature set differs."
            ),
        },
        "reproducibility": repro_metadata(seed),
        "documentation": {
            "acoustic_only_752": (
                "gender removed from the Baseline block; acoustic measurements only."
            ),
            "acoustic_plus_gender_753": (
                "gender retained in the Baseline block as demographic metadata; identical "
                "feature set to the deployed model."
            ),
            "deployed_model": (
                "The deployed 753-feature model was fit on a single fixed subject split and is "
                "NOT retrained or altered here. These cross-validated estimates measure "
                "generalisation and do not represent the deployed model's reported numbers."
            ),
            "gender_treatment": "demographic metadata, not an acoustic feature group.",
        },
        "configurations": {
            "acoustic_only": {
                "n_features": int(acoustic_data.X.shape[1]),
                "models": acoustic_results,
            },
            "acoustic_plus_gender": {
                "n_features": int(data.X.shape[1]),
                "models": full_results,
            },
        },
        "comparison_point_estimates": _delta(acoustic_results, full_results),
        "gender_subgroups": {
            "acoustic_plus_gender": _subgroups(data, full_oof, n_boot=n_boot, seed=seed),
            "acoustic_only": _subgroups(data, acoustic_oof, n_boot=n_boot, seed=seed),
        },
    }
    (REPORTS / "gender_ablation.json").write_text(json.dumps(report, indent=2))

    print("\nrecording-level ROC-AUC (point):")
    print(f"{'model':<20} {'acoustic-only':>14} {'+gender':>10} {'delta':>8}")
    cmp = report["comparison_point_estimates"]
    for model in ALL_MODELS:
        r = cmp[model]["recording_level"]["roc_auc"]
        print(
            f"{model:<20} {r['acoustic_only']:>14.3f} "
            f"{r['acoustic_plus_gender']:>10.3f} {r['delta_gender_minus_acoustic']:>+8.3f}"
        )
    print("\nWrote reports/gender_ablation.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    main(n_splits=args.n_splits, seed=args.seed, epochs=args.epochs, n_boot=args.n_boot)
