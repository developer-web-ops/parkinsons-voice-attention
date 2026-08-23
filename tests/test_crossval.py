"""Fast unit tests for the metric and bootstrap helpers in ``src.stats``.

These run without any model training so they stay cheap; the heavy pipeline is
exercised by actually running ``python -m src.crossval``.
"""

import numpy as np

from src.stats import (
    METRIC_NAMES,
    all_metrics,
    bootstrap_metrics,
    rank_metrics,
    threshold_metrics,
)


def test_threshold_metrics_perfect():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 1, 1])
    m = threshold_metrics(y, pred)
    assert m["accuracy"] == 1.0
    assert m["sensitivity"] == 1.0
    assert m["specificity"] == 1.0
    assert m["precision"] == 1.0
    assert m["f1"] == 1.0


def test_sensitivity_specificity_directions():
    # tp=1 (idx0), fn=1 (idx1), tn=2 (idx2,3), fp=0
    y = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 0, 0])
    m = threshold_metrics(y, pred)
    assert m["sensitivity"] == 0.5  # caught 1 of 2 positives
    assert m["specificity"] == 1.0  # no false positives
    assert m["precision"] == 1.0


def test_threshold_metrics_no_positive_predictions():
    y = np.array([0, 1, 1])
    pred = np.array([0, 0, 0])
    m = threshold_metrics(y, pred)
    assert m["sensitivity"] == 0.0
    assert m["specificity"] == 1.0
    assert np.isnan(m["precision"])  # 0/0 → undefined, not 0
    assert np.isnan(m["f1"])


def test_rank_metrics_single_class_is_nan():
    y = np.ones(5, dtype=int)
    prob = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
    m = rank_metrics(y, prob)
    assert np.isnan(m["roc_auc"])
    assert np.isnan(m["pr_auc"])


def test_rank_metrics_perfect_separation():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.8, 0.9])
    m = rank_metrics(y, prob)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0


def test_all_metrics_has_every_named_metric():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.1, 0.2, 0.8, 0.9])
    pred = (prob >= 0.5).astype(int)
    m = all_metrics(y, prob, pred)
    assert set(METRIC_NAMES).issubset(m.keys())


def _grouped_sample():
    """40 subjects, 3 recordings each; probability correlated with label."""
    rng = np.random.default_rng(0)
    n_subj = 40
    groups = np.repeat(np.arange(n_subj), 3)
    subj_label = np.array([i % 2 for i in range(n_subj)])
    y = np.repeat(subj_label, 3)
    prob = np.clip(np.where(y == 1, 0.7, 0.3) + rng.normal(0, 0.1, size=len(y)), 0, 1)
    pred = (prob >= 0.5).astype(int)
    return y, prob, pred, groups


def test_bootstrap_ci_is_ordered_and_bounded():
    y, prob, pred, groups = _grouped_sample()
    res = bootstrap_metrics(y, prob, pred, groups, n_boot=200, seed=1)
    assert set(res.keys()) == set(METRIC_NAMES)
    for name in ("roc_auc", "accuracy", "sensitivity", "specificity"):
        ci = res[name]
        assert ci["ci_low"] <= ci["ci_high"]
        assert 0.0 <= ci["ci_low"] <= 1.0
        assert 0.0 <= ci["ci_high"] <= 1.0
        assert 0 < ci["n_boot_valid"] <= 200


def test_bootstrap_is_deterministic_for_a_seed():
    y, prob, pred, groups = _grouped_sample()
    a = bootstrap_metrics(y, prob, pred, groups, n_boot=200, seed=7)
    b = bootstrap_metrics(y, prob, pred, groups, n_boot=200, seed=7)
    assert a["roc_auc"]["ci_low"] == b["roc_auc"]["ci_low"]
    assert a["roc_auc"]["ci_high"] == b["roc_auc"]["ci_high"]
