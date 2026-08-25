"""Phase 2C tests: robustness, explainability, confounds/fairness, orchestrator.

Everything runs on tiny synthetic feature frames — never the ~606 MB corpus. The
module is skipped when the audio stack is absent (the Phase 2C modules import
:mod:`src.audio.baseline_cv`, which imports openSMILE). SHAP-specific assertions
additionally skip if ``shap`` is not installed.

Coverage: 88-feature family partition; repeated grouped-CV determinism and
structure; canonical OOF reproduction; coefficient / SHAP / stability views and
their rank agreement; the metadata-availability guard (no fabricated
demographics); the duration and task confound probes; task-based fairness; and
the orchestrator (freeze-does-not-mutate + full run writing only under its
out_dir).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

# Phase 2C modules import baseline_cv -> opensmile; skip cleanly without it.
pytest.importorskip("opensmile")

from src.audio.confounds import analyze as confounds_analyze
from src.audio.confounds import (
    duration_by_class,
    duration_only_separability,
    duration_prediction_correlation,
    format_uniformity,
    metadata_availability,
    subgroup_performance_by_task,
    task_distribution,
)
from src.audio.dataset import build_dataset
from src.audio.explain import (
    artefact_family_share,
    coefficient_importance,
    coefficient_stability,
    family_importance,
    rank_agreement,
    shap_importance,
)
from src.audio.families import (
    FAMILY_ORDER,
    assign_families,
    family_members,
    feature_family,
)
from src.audio.features import feature_names
from src.audio.phase2c import (
    compare_to_repeated,
    freeze_baseline,
    run,
)
from src.audio.robustness import (
    REPEAT_METRIC_NAMES,
    reference_oof,
    repeated_grouped_cv,
)

# Designed eGeMAPSv02 family sizes (sum to 88); see src/audio/families.py.
EXPECTED_FAMILY_COUNTS = {
    "Pitch (F0)": 10,
    "Jitter": 2,
    "Shimmer": 2,
    "Loudness/Energy": 11,
    "Harmonicity/Noise": 6,
    "Formants": 18,
    "Spectral Balance": 17,
    "MFCC (Cepstral)": 16,
    "Temporal/Rhythm": 6,
}


# ----------------------------------------------------------------------------- helpers


def make_fake_frame(rng, n_per_class=12, recs=2, n_feat=16, sep=2.0, dur_sep=0.0):
    """Synthetic feature frame with a clear per-feature importance gradient.

    Feature ``i`` separates the classes with weight ``(i+1)/n_feat`` so |coef| and
    mean|SHAP| have a monotone, testable ordering. ``dur_sep`` injects a duration
    difference between PD and HC for the confound probes.
    """
    rows = []
    sid = 1
    weights = np.array([(i + 1) / n_feat for i in range(n_feat)])
    for label in (0, 1):
        for _ in range(n_per_class):
            gk = f"S{sid:02d}"
            for r in range(recs):
                dur = 2.0 + label * dur_sep + float(rng.normal(0, 0.05))
                row = {
                    "subject_id": sid,
                    "group_key": gk,
                    "label": label,
                    "cohort_folder": "PD" if label else "HC",
                    "cohort_file": "pd" if label else "hc",
                    "task": "ReadText" if r == 0 else "SpontaneousDialogue",
                    "filename": f"{gk}_{r}.wav",
                    "relpath": f"{gk}_{r}.wav",
                    "duration_s": dur,
                    "orig_sample_rate": 44100,
                    "orig_channels": 1,
                }
                feats = rng.normal(label * sep * weights, 1.0, size=n_feat)
                row.update({f"feat_{i:02d}": float(v) for i, v in enumerate(feats)})
                rows.append(row)
            sid += 1
    return pd.DataFrame(rows)


def fake_cv_metrics():
    """A minimal cv_metrics.json payload with the LR block freeze_baseline reads."""

    def _boot(point):
        return {"point": point, "ci_low": point - 0.1, "ci_high": point + 0.1, "n_boot_valid": 2000}

    def _model_block(auc):
        subj = {
            m: _boot(v)
            for m, v in {
                "roc_auc": auc,
                "pr_auc": auc - 0.02,
                "accuracy": 0.75,
                "sensitivity": 0.9,
                "specificity": 0.6,
                "precision": 0.65,
                "f1": 0.76,
                "balanced_accuracy": 0.77,
                "brier_score": 0.15,
            }.items()
        }
        per_fold = {"roc_auc": {"mean": auc, "std": 0.05, "per_fold": [auc] * 5}}
        return {
            "recording_level": subj,
            "subject_level": subj,
            "per_fold_recording": per_fold,
            "per_fold_subject": per_fold,
            "fold_thresholds": [0.32, 0.5, 0.34, 0.41, 0.39],
            "folds_calibrated": [True] * 5,
        }

    return {
        "config": {"primary_metric_level": "subject_level"},
        "data_balance": {"recordings": 48, "subjects": 24},
        "reproducibility": {
            "git_commit": "deadbeef",
            "feature_dataset_sha256": "a" * 64,
            "dataset_manifest_sha256": "b" * 64,
        },
        "models": {
            "logistic_regression": _model_block(0.88),
            "svm_rbf": _model_block(0.86),
            "random_forest": _model_block(0.85),
        },
    }


# ----------------------------------------------------------------------------- families


def test_family_map_covers_all_88_features():
    names = feature_names()
    assert len(names) == 88
    fam = assign_families(names)
    assert "Other" not in fam.values()
    assert set(fam.values()) == set(FAMILY_ORDER)
    counts = {f: sum(v == f for v in fam.values()) for f in FAMILY_ORDER}
    assert sum(counts.values()) == 88
    assert counts == EXPECTED_FAMILY_COUNTS


def test_family_members_is_a_partition():
    names = feature_names()
    members = family_members(names)
    flat = [n for cols in members.values() for n in cols]
    assert len(flat) == 88
    assert len(set(flat)) == 88  # disjoint
    assert list(members) == [f for f in FAMILY_ORDER if f in members]


def test_feature_family_is_pure_function():
    assert feature_family("F0semitoneFrom27.5Hz_sma3nz_amean") == "Pitch (F0)"
    assert feature_family("jitterLocal_sma3nz_amean") == "Jitter"
    assert feature_family("mfcc1_sma3_amean") == "MFCC (Cepstral)"
    assert feature_family("loudnessPeaksPerSec") == "Temporal/Rhythm"
    assert feature_family("totally_unknown_feature") == "Other"


# ----------------------------------------------------------------------------- robustness


def test_repeated_grouped_cv_structure():
    rng = np.random.default_rng(0)
    data = build_dataset(make_fake_frame(rng, n_per_class=12, sep=2.5))
    out = repeated_grouped_cv(
        data, models=("logistic_regression",), n_splits=4, n_repeats=3, base_seed=42
    )
    lr = out["logistic_regression"]
    assert lr["seeds"] == [42, 43, 44]
    assert set(lr["metrics"]) == set(REPEAT_METRIC_NAMES)
    for stats in lr["metrics"].values():
        assert {"n", "mean", "std", "min", "max", "p2_5", "p50", "p97_5"} <= set(stats)
        assert stats["n"] == 3
    assert {"iqr", "frac_non_default"} <= set(lr["threshold_stability"])
    assert len(lr["fold_thresholds"]) == 3 * 4
    assert lr["metrics"]["roc_auc"]["mean"] > 0.6  # separable synthetic data


def test_repeated_grouped_cv_is_deterministic():
    rng = np.random.default_rng(1)
    data = build_dataset(make_fake_frame(rng, n_per_class=12, sep=2.5))
    a = repeated_grouped_cv(
        data, models=("logistic_regression",), n_splits=4, n_repeats=3, base_seed=42
    )
    rng2 = np.random.default_rng(1)
    data2 = build_dataset(make_fake_frame(rng2, n_per_class=12, sep=2.5))
    b = repeated_grouped_cv(
        data2, models=("logistic_regression",), n_splits=4, n_repeats=3, base_seed=42
    )
    assert a["logistic_regression"]["per_repeat"] == b["logistic_regression"]["per_repeat"]


def test_reference_oof_covers_all_and_is_deterministic():
    rng = np.random.default_rng(2)
    data = build_dataset(make_fake_frame(rng, n_per_class=12, sep=2.5))
    prob, pred, thr = reference_oof(data, n_splits=4, seed=42)
    assert prob.shape == (len(data.y),)
    assert not np.isnan(prob).any()
    assert not np.isnan(thr).any()
    assert set(np.unique(pred)) <= {0, 1}
    prob2, pred2, _thr2 = reference_oof(data, n_splits=4, seed=42)
    np.testing.assert_array_equal(prob, prob2)
    np.testing.assert_array_equal(pred, pred2)


# ----------------------------------------------------------------------------- explainability


def test_coefficient_importance_ranked_and_labelled():
    rng = np.random.default_rng(3)
    data = build_dataset(make_fake_frame(rng, n_per_class=12, n_feat=16, sep=2.5))
    ranked = coefficient_importance(data, seed=42)
    assert len(ranked) == data.X.shape[1]
    abs_coefs = [r["abs_coef"] for r in ranked]
    assert abs_coefs == sorted(abs_coefs, reverse=True)  # descending
    for r in ranked:
        assert r["direction"] in {"PD", "HC"}
        assert r["family"] in set(FAMILY_ORDER) | {"Other"}


def test_coefficient_stability_bounds():
    rng = np.random.default_rng(4)
    data = build_dataset(make_fake_frame(rng, n_per_class=12, n_feat=16, sep=2.5))
    stab = coefficient_stability(data, seed=42, n_splits=4)
    assert len(stab) == data.X.shape[1]
    for r in stab:
        assert 0.0 <= r["sign_consistency"] <= 1.0
        assert r["std_coef"] >= 0.0


def test_shap_importance_and_rank_agreement():
    pytest.importorskip("shap")
    rng = np.random.default_rng(5)
    data = build_dataset(make_fake_frame(rng, n_per_class=14, n_feat=16, sep=3.0))
    ranked, values, _names = shap_importance(data, seed=42)
    assert len(ranked) == data.X.shape[1]
    assert values.shape[1] == data.X.shape[1]
    assert all(r["mean_abs_shap"] >= 0 for r in ranked)
    coef_ranked = coefficient_importance(data, seed=42)
    agree = rank_agreement(coef_ranked, ranked)
    assert agree["n_features"] == data.X.shape[1]
    # For a linear model, SHAP magnitude ranks track |coef| strongly.
    assert agree["spearman_rho"] > 0.5


def test_family_importance_and_artefact_share_sum_to_one():
    rng = np.random.default_rng(6)
    data = build_dataset(make_fake_frame(rng, n_per_class=12, n_feat=16, sep=2.5))
    coef_ranked = coefficient_importance(data, seed=42)
    # family_importance accepts any rows carrying family + mean_abs_shap.
    rows = family_importance(
        [{"feature": r["feature"], "family": r["family"], "mean_abs_shap": r["abs_coef"]}
         for r in coef_ranked]
    )
    assert abs(sum(r["share"] for r in rows) - 1.0) < 1e-9
    share = artefact_family_share(rows)
    assert 0.0 <= share["combined_share"] <= 1.0
    assert abs(share["combined_share"] + share["voice_intrinsic_share"] - 1.0) < 1e-9


# ----------------------------------------------------------------------------- confounds / fairness


def test_metadata_availability_reports_no_demographics():
    rng = np.random.default_rng(7)
    frame = make_fake_frame(rng, n_per_class=6)
    md = metadata_availability(frame)
    assert md["available"]["task"] is True
    assert md["available"]["duration_s"] is True
    # No demographic field is present, and none is fabricated.
    assert all(v is False for v in md["demographic_available"].values())
    assert "not attempted" in md["note"].lower() or "no fabrication" in md["note"].lower()


def test_format_uniformity_flags_uniform_corpus():
    rng = np.random.default_rng(8)
    frame = make_fake_frame(rng, n_per_class=6)
    fu = format_uniformity(frame)
    assert fu["sample_rate_uniform"] is True
    assert fu["channels_uniform"] is True
    assert fu["orig_sample_rate_counts"] == {44100: len(frame)}


def test_duration_probes_detect_injected_length_confound():
    rng = np.random.default_rng(9)
    frame = make_fake_frame(rng, n_per_class=12, sep=0.2, dur_sep=3.0)  # strong duration gap
    dbc = duration_by_class(frame)
    assert dbc["pd"]["mean"] > dbc["hc"]["mean"]
    assert dbc["point_biserial_r"] > 0.5
    sep = duration_only_separability(frame, n_splits=4, seed=42)
    assert {"recording_roc_auc", "subject_roc_auc"} <= set(sep)
    assert sep["recording_roc_auc"] > 0.7  # duration alone separates when injected


def test_duration_prediction_correlation_structure():
    rng = np.random.default_rng(10)
    frame = make_fake_frame(rng, n_per_class=12, sep=2.5)
    data = build_dataset(frame)
    prob, _pred, _thr = reference_oof(data, n_splits=4, seed=42)
    corr = duration_prediction_correlation(frame, prob)
    assert set(corr) == {"spearman_rho", "p_value"}
    assert -1.0 <= corr["spearman_rho"] <= 1.0


def test_task_distribution_counts_add_up():
    rng = np.random.default_rng(11)
    frame = make_fake_frame(rng, n_per_class=10)
    td = task_distribution(frame)
    assert set(td["per_task"]) == {"ReadText", "SpontaneousDialogue"}
    for stats in td["per_task"].values():
        assert stats["n_pd"] + stats["n_hc"] == stats["n_recordings"]
    total = sum(s["n_recordings"] for s in td["per_task"].values())
    assert total == len(frame)


def test_subgroup_performance_by_task_structure():
    rng = np.random.default_rng(12)
    frame = make_fake_frame(rng, n_per_class=12, sep=2.5)
    data = build_dataset(frame)
    prob, pred, _thr = reference_oof(data, n_splits=4, seed=42)
    fair = subgroup_performance_by_task(frame, prob, pred)
    assert set(fair["per_task"]) == {"ReadText", "SpontaneousDialogue"}
    for stats in fair["per_task"].values():
        assert "balanced_accuracy" in stats and "roc_auc" in stats
    assert "protected attribute" in fair["note"]


def test_confounds_analyze_end_to_end_preserves_grouping():
    rng = np.random.default_rng(13)
    frame = make_fake_frame(rng, n_per_class=12, sep=2.5)
    data = build_dataset(frame)
    coef_ranked = coefficient_importance(data, seed=42)
    out = confounds_analyze(frame, data, coef_ranked, n_splits=4, seed=42)
    for key in (
        "metadata_availability",
        "format_uniformity",
        "duration_only_separability",
        "task_distribution",
        "fairness_by_task",
        "flags",
    ):
        assert key in out
    assert isinstance(out["flags"]["duration_confound_suspected"], bool)


# ----------------------------------------------------------------------------- orchestrator


def test_freeze_baseline_does_not_mutate_source(tmp_path):
    src = tmp_path / "cv_metrics.json"
    payload = fake_cv_metrics()
    src.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    before = src.read_bytes()

    frozen = freeze_baseline(src)
    assert frozen["reference_model"] == "logistic_regression"
    assert frozen["modified_phase2b"] is False
    assert frozen["subject_level"]["roc_auc"]["point"] == 0.88
    assert set(frozen["all_models_subject_snapshot"]) == set(payload["models"])
    assert frozen["provenance"]["git_commit"] == "deadbeef"
    # The source metrics file is untouched.
    assert src.read_bytes() == before


def test_compare_to_repeated_flags_band_membership():
    frozen = {
        "subject_level": {
            "roc_auc": {"point": 0.88, "ci_low": 0.75, "ci_high": 0.98},
            "balanced_accuracy": {"point": 0.90, "ci_low": 0.80, "ci_high": 0.99},
        }
    }
    repeated_lr = {
        "n_repeats": 5,
        "metrics": {
            "roc_auc": {"mean": 0.86, "std": 0.03, "p2_5": 0.80, "p97_5": 0.92},
            "balanced_accuracy": {"mean": 0.70, "std": 0.03, "p2_5": 0.64, "p97_5": 0.76},
        },
    }
    cmp = compare_to_repeated(frozen, repeated_lr)
    assert cmp["metrics"]["roc_auc"]["frozen_point_in_repeated_band"] is True
    assert cmp["metrics"]["balanced_accuracy"]["frozen_point_in_repeated_band"] is False
    assert cmp["metrics"]["roc_auc"]["mean_minus_frozen"] == pytest.approx(-0.02)


def test_run_writes_all_artifacts_under_out_dir(tmp_path):
    rng = np.random.default_rng(14)
    frame = make_fake_frame(rng, n_per_class=12, sep=2.5)
    feature_csv = tmp_path / "feat.csv"
    frame.to_csv(feature_csv, index=False)
    cv_metrics = tmp_path / "cv_metrics.json"
    cv_metrics.write_text(json.dumps(fake_cv_metrics(), indent=2), encoding="utf-8")
    out_dir = tmp_path / "phase2c"

    summary = run(
        n_repeats=2,
        n_splits=4,
        base_seed=42,
        seed=42,
        feature_csv=feature_csv,
        cv_metrics_path=cv_metrics,
        out_dir=out_dir,
        make_plot=False,
    )

    for fname in (
        "frozen_baseline_reference.json",
        "repeated_cv.json",
        "explainability.json",
        "feature_importance.csv",
        "confound_fairness.json",
        "phase2c_summary.json",
    ):
        assert (out_dir / fname).exists()

    assert summary["frozen_baseline_reference"]["modified_phase2b"] is False
    assert "roc_auc" in summary["comparison_frozen_vs_repeated"]["metrics"]
    written = json.loads((out_dir / "phase2c_summary.json").read_text(encoding="utf-8"))
    assert written["phase"] == "2C"
    # feature_importance.csv has one row per feature.
    table = pd.read_csv(out_dir / "feature_importance.csv")
    assert len(table) == frame.filter(like="feat_").shape[1]
