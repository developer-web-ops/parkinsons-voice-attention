"""Adapt the extracted eGeMAPSv02 feature table into a Phase 1 ``src.data.Dataset``.

Reusing the Phase 1 container is deliberate: it lets the audio pipeline call the
exact same, already-tested subject-grouped CV helpers (:func:`cv_splits`,
:func:`inner_subject_split`, :func:`fit_scaler`, :func:`assert_no_group_leakage`,
:func:`subject_labels`) without copying or modifying any Phase 1 code. The group
key is the subject id, so every split is grouped by subject.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.audio.features import FEATURE_CSV, META_COLUMNS, load_feature_frame
from src.data import Dataset

FEATURE_GROUP = "eGeMAPSv02"


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """The acoustic-feature columns = everything that is not metadata."""
    return [c for c in frame.columns if c not in META_COLUMNS]


def build_dataset(frame: pd.DataFrame, names: list[str] | None = None) -> Dataset:
    """Construct a subject-grouped :class:`Dataset` from a feature frame."""
    names = names or feature_columns(frame)
    return Dataset(
        X=frame[names].astype("float64").reset_index(drop=True),
        y=frame["label"].to_numpy(dtype=np.int64),
        groups=frame["group_key"].to_numpy(),
        feature_groups={FEATURE_GROUP: list(names)},
    )


def load_dataset(csv_path: Path = FEATURE_CSV) -> Dataset:
    frame = load_feature_frame(csv_path)
    return build_dataset(frame)


def subject_table(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per subject with its (constant) label and recording count."""
    grp = frame.groupby("group_key")
    out = grp.agg(label=("label", "first"), n_recordings=("filename", "count"))
    # Guard: a subject's label must be constant across its recordings.
    inconsistent = grp["label"].nunique()
    bad = inconsistent[inconsistent > 1].index.tolist()
    if bad:
        raise ValueError(f"inconsistent label within subject(s): {bad}")
    return out.reset_index()


def label_balance(frame: pd.DataFrame) -> dict[str, int]:
    subj = subject_table(frame)
    return {
        "recordings": len(frame),
        "recordings_pd": int((frame["label"] == 1).sum()),
        "recordings_hc": int((frame["label"] == 0).sum()),
        "subjects": len(subj),
        "subjects_pd": int((subj["label"] == 1).sum()),
        "subjects_hc": int((subj["label"] == 0).sum()),
    }


def as_int_groups(groups: np.ndarray) -> np.ndarray:
    """Stable integer codes for subject keys (handy for some sklearn utilities)."""
    _, codes = np.unique(groups, return_inverse=True)
    return codes
