"""Leakage guards for the subject-grouped cross-validation pipeline.

These tests fail loudly if a future change lets recordings from one subject
appear in two splits, if preprocessing is fit on anything other than the
training fold, or if the gender-ablation helper drops the wrong columns.
"""

import numpy as np
import pytest

from src.data import (
    GENDER_COL,
    assert_no_group_leakage,
    cv_splits,
    fit_scaler,
    inner_subject_split,
    load_dataset,
    without_gender,
)


def test_cv_splits_no_subject_leakage():
    data = load_dataset()
    for train_idx, test_idx in cv_splits(data, n_splits=5, seed=42):
        train_subj = set(data.groups[train_idx])
        test_subj = set(data.groups[test_idx])
        assert not (train_subj & test_subj)
    # every recording is tested exactly once across the folds
    tested = np.concatenate([test for _, test in cv_splits(data, n_splits=5, seed=42)])
    assert sorted(tested) == list(range(len(data.y)))


def test_inner_val_split_no_leakage():
    data = load_dataset()
    train_all, _ = cv_splits(data, n_splits=5, seed=42)[0]
    inner_train, val = inner_subject_split(data, train_all, seed=42)
    assert not (set(data.groups[inner_train]) & set(data.groups[val]))
    # the inner split partitions the training fold with nothing added or lost
    assert sorted(np.concatenate([inner_train, val])) == sorted(train_all)


def test_fold_scaler_fits_train_only():
    data = load_dataset()
    train_idx, test_idx = cv_splits(data, n_splits=5, seed=42)[0]
    scaler = fit_scaler(data.X.iloc[train_idx])
    # the scaler's statistics match the training fold, not the full dataset
    assert np.allclose(scaler.mean_, data.X.iloc[train_idx].mean().to_numpy())
    assert not np.allclose(scaler.mean_, data.X.mean().to_numpy())
    # and not the test fold either
    assert not np.allclose(scaler.mean_, data.X.iloc[test_idx].mean().to_numpy())


def test_leakage_canary_is_detected():
    """A deliberately leaky split must raise — proves the guard actually fires."""
    groups = np.array([0, 0, 1, 1, 2, 2])
    train_idx = np.array([0, 1, 2])  # subject 1 straddles both sides
    test_idx = np.array([3, 4, 5])
    with pytest.raises(ValueError, match="leakage"):
        assert_no_group_leakage(train_idx, test_idx, groups)


def test_assert_passes_on_clean_split():
    groups = np.array([0, 0, 1, 1, 2, 2])
    assert assert_no_group_leakage(np.array([0, 1, 2, 3]), np.array([4, 5]), groups) is None


def test_without_gender_drops_only_gender():
    data = load_dataset()
    assert GENDER_COL in data.X.columns
    reduced = without_gender(data)
    assert reduced.X.shape[1] == data.X.shape[1] - 1
    assert GENDER_COL not in reduced.X.columns
    # every other column is preserved, in order
    assert list(reduced.X.columns) == [c for c in data.X.columns if c != GENDER_COL]
    # labels and subject groups are untouched
    assert np.array_equal(reduced.y, data.y)
    assert np.array_equal(reduced.groups, data.groups)
    # gender came out of exactly one feature group and no group vanished emptied-out
    assert GENDER_COL not in {c for cols in reduced.feature_groups.values() for c in cols}
    assert all(cols for cols in reduced.feature_groups.values())
