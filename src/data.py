"""Loading and preprocessing for the UCI Parkinson's Disease Classification dataset.

The raw CSV has a two-row header: the first row names feature *groups* and the
second row names the individual features. Every subject (``id``) contributed
three voice recordings, so all splits are grouped by subject to avoid leakage.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "pd_speech_features.csv"

TARGET = "class"
GROUP_COL = "id"

# Canonical order of the feature-group blocks declared in the first header row.
GROUP_ORDER = [
    "Baseline Features",
    "Intensity Parameters",
    "Formant Frequencies",
    "Bandwidth Parameters",
    "Vocal Fold",
    "MFCC",
    "Wavelet Features",
    "TQWT Features",
]


@dataclass
class Dataset:
    X: pd.DataFrame
    y: np.ndarray
    groups: np.ndarray
    feature_groups: dict[str, list[str]]

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)


def read_feature_groups(path: Path = DATA_PATH) -> dict[str, list[str]]:
    """Map each feature-group block to its member column names."""
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        block_row = next(reader)
        name_row = next(reader)

    groups: dict[str, list[str]] = {}
    current: str | None = None
    for block, name in zip(block_row, name_row, strict=False):
        block = block.strip()
        if block:
            current = block
        if current is None or name in (TARGET, GROUP_COL):
            continue
        groups.setdefault(current, []).append(name)

    # ``gender`` sits before the first block header; treat it as a baseline feature.
    if "gender" in name_row:
        groups["Baseline Features"].insert(0, "gender")
    return {g: groups[g] for g in GROUP_ORDER if g in groups}


def load_dataset(path: Path = DATA_PATH) -> Dataset:
    df = pd.read_csv(path, header=1)
    feature_groups = read_feature_groups(path)
    features = [c for g in feature_groups.values() for c in g]
    return Dataset(
        X=df[features].astype("float64"),
        y=df[TARGET].to_numpy(dtype=np.int64),
        groups=df[GROUP_COL].to_numpy(),
        feature_groups=feature_groups,
    )


def subject_split(
    data: Dataset, test_size: float = 0.2, val_size: float = 0.15, seed: int = 42
) -> dict[str, np.ndarray]:
    """Split row indices into train/val/test so a subject appears in one split only."""
    subjects = np.unique(data.groups)
    # A subject's label is constant across their recordings.
    subject_label = np.array([data.y[data.groups == s][0] for s in subjects])

    train_subj, test_subj = train_test_split(
        subjects, test_size=test_size, stratify=subject_label, random_state=seed
    )
    train_label = np.array([data.y[data.groups == s][0] for s in train_subj])
    train_subj, val_subj = train_test_split(
        train_subj, test_size=val_size, stratify=train_label, random_state=seed
    )

    return {
        "train": np.flatnonzero(np.isin(data.groups, train_subj)),
        "val": np.flatnonzero(np.isin(data.groups, val_subj)),
        "test": np.flatnonzero(np.isin(data.groups, test_subj)),
    }


def cv_splits(data: Dataset, n_splits: int = 5, seed: int = 42):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(data.X, data.y, groups=data.groups))


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    return StandardScaler().fit(X_train.to_numpy())


def class_weights(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights for the imbalanced (192 vs 564) classes."""
    counts = np.bincount(y, minlength=2).astype(np.float64)
    return counts.sum() / (len(counts) * counts)
