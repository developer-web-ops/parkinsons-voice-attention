"""eGeMAPSv02 feature extraction: raw WAV -> preprocessed signal -> 88 functionals.

openSMILE's eGeMAPSv02 *Functionals* set yields a fixed 88-dimensional vector of
named acoustic descriptors per recording (F0, jitter, shimmer, HNR, loudness,
spectral/MFCC statistics, ...). The extraction is deterministic and unsupervised,
so it is safe to run once, offline, over every recording; all label-aware fitting
(scaling, calibration, threshold) happens later, inside each CV training fold.

Named features are preserved verbatim (not renamed to ``f0..f87``) so the
downstream model stays explainable and SHAP-ready.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import opensmile
import pandas as pd

from src.audio.config import DEFAULT_CONFIG, AudioConfig
from src.audio.discovery import Recording, discover_recordings
from src.audio.preprocess import (
    ProcessedAudio,
    duration_exclusion_reason,
    preprocess_file,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_AUDIO = ROOT / "data_audio" / "mdvr_kcl"
RAW_DIR = DATA_AUDIO / "raw"
MANIFEST_PATH = DATA_AUDIO / "manifest.sha256.json"

ARTIFACTS_AUDIO = ROOT / "artifacts_audio"
FEATURES_DIR = ARTIFACTS_AUDIO / "features"
FEATURE_CSV = FEATURES_DIR / "egemaps_v02_functionals.csv"
FEATURE_META_JSON = FEATURES_DIR / "egemaps_v02_functionals.meta.json"

# Non-feature (metadata) columns carried alongside the 88 acoustic features.
META_COLUMNS = [
    "subject_id",
    "group_key",
    "label",
    "cohort_folder",
    "cohort_file",
    "task",
    "filename",
    "relpath",
    "duration_s",
    "orig_sample_rate",
    "orig_channels",
]


def build_smile(config: AudioConfig = DEFAULT_CONFIG) -> opensmile.Smile:
    """Construct the pinned eGeMAPSv02 Functionals extractor."""
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def feature_names(smile: opensmile.Smile | None = None) -> list[str]:
    """Canonical, ordered eGeMAPSv02 feature names (88 of them)."""
    smile = smile or build_smile()
    return list(smile.feature_names)


def extract_vector(
    processed: ProcessedAudio, smile: opensmile.Smile, config: AudioConfig = DEFAULT_CONFIG
) -> np.ndarray:
    """Return the 88-dim feature vector for a preprocessed signal (order = feature_names)."""
    df = smile.process_signal(processed.signal, processed.sample_rate)
    df = df.reindex(columns=smile.feature_names)
    vec = df.to_numpy(dtype=np.float64).reshape(-1)
    if vec.shape[0] != config.expected_feature_count:
        raise ValueError(
            f"expected {config.expected_feature_count} features, got {vec.shape[0]}"
        )
    return vec


def _row(rec: Recording, processed: ProcessedAudio, vec: np.ndarray, names: list[str]) -> dict:
    try:
        relpath = rec.path.relative_to(RAW_DIR).as_posix()
    except ValueError:
        relpath = rec.path.name
    row = {
        "subject_id": rec.subject_id,
        "group_key": rec.group_key,
        "label": rec.label,
        "cohort_folder": rec.cohort_folder,
        "cohort_file": rec.cohort_file,
        "task": rec.task,
        "filename": rec.filename,
        "relpath": relpath,
        "duration_s": round(processed.duration_s, 6),
        "orig_sample_rate": processed.orig_sample_rate,
        "orig_channels": processed.orig_channels,
    }
    row.update({name: float(v) for name, v in zip(names, vec, strict=True)})
    return row


def build_feature_frame(
    raw_dir: Path = RAW_DIR, config: AudioConfig = DEFAULT_CONFIG
) -> tuple[pd.DataFrame, list[dict], list[str]]:
    """Extract features for every usable recording under ``raw_dir``.

    Returns ``(frame, exclusions, names)`` where ``exclusions`` is a list of
    ``{"path", "reason", "detail"}`` dicts covering *every* recording that was
    discovered but not featurised (nothing is dropped silently).
    """
    recordings, discovery_exclusions = discover_recordings(raw_dir)
    smile = build_smile(config)
    names = feature_names(smile)

    exclusions: list[dict] = [
        {"path": e.path.as_posix(), "reason": e.reason, "detail": e.detail}
        for e in discovery_exclusions
    ]

    rows: list[dict] = []
    for rec in recordings:
        try:
            processed = preprocess_file(rec.path, config)
        except Exception as exc:
            exclusions.append(
                {"path": rec.path.as_posix(), "reason": "load_failed", "detail": repr(exc)}
            )
            continue

        reason = duration_exclusion_reason(processed, config)
        if reason is not None:
            exclusions.append({"path": rec.path.as_posix(), "reason": "duration", "detail": reason})
            continue

        vec = extract_vector(processed, smile, config)
        if not np.isfinite(vec).all():
            exclusions.append(
                {"path": rec.path.as_posix(), "reason": "nonfinite_features", "detail": ""}
            )
            continue

        rows.append(_row(rec, processed, vec, names))

    columns = META_COLUMNS + names
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        frame = frame.sort_values(["subject_id", "task", "filename"]).reset_index(drop=True)
    return frame, exclusions, names


def dataset_metadata(
    frame: pd.DataFrame, exclusions: list[dict], names: list[str], config: AudioConfig
) -> dict:
    """Summary + schema written next to the feature CSV (no timestamps in the CSV)."""
    n_pd = int((frame["label"] == 1).sum()) if not frame.empty else 0
    n_hc = int((frame["label"] == 0).sum()) if not frame.empty else 0
    subj = frame[["group_key", "label"]].drop_duplicates() if not frame.empty else pd.DataFrame()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_set": config.feature_set,
        "feature_level": config.feature_level,
        "opensmile_package": config.opensmile_package,
        "feature_count": len(names),
        "feature_names": names,
        "audio_config": config.as_dict(),
        "n_recordings": len(frame),
        "n_subjects": int(subj["group_key"].nunique()) if not subj.empty else 0,
        "recordings_pd": n_pd,
        "recordings_hc": n_hc,
        "subjects_pd": int((subj["label"] == 1).sum()) if not subj.empty else 0,
        "subjects_hc": int((subj["label"] == 0).sum()) if not subj.empty else 0,
        "n_excluded": len(exclusions),
        "exclusions": exclusions,
        "meta_columns": META_COLUMNS,
    }


def save_feature_dataset(
    frame: pd.DataFrame, meta: dict, out_csv: Path = FEATURE_CSV, out_meta: Path = FEATURE_META_JSON
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_feature_frame(csv_path: Path = FEATURE_CSV) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def main(raw_dir: Path = RAW_DIR, config: AudioConfig = DEFAULT_CONFIG) -> dict:
    """Extract eGeMAPSv02 features for the whole corpus and cache to artifacts_audio."""
    frame, exclusions, names = build_feature_frame(raw_dir=raw_dir, config=config)
    meta = dataset_metadata(frame, exclusions, names, config)
    save_feature_dataset(frame, meta)
    print(
        f"eGeMAPSv02 features: {meta['n_recordings']} recordings, "
        f"{meta['n_subjects']} subjects "
        f"(PD subj={meta['subjects_pd']}, HC subj={meta['subjects_hc']}; "
        f"PD rec={meta['recordings_pd']}, HC rec={meta['recordings_hc']}), "
        f"{meta['feature_count']} features, {meta['n_excluded']} excluded"
    )
    if exclusions:
        for e in exclusions:
            print(f"  excluded [{e['reason']}] {e['path']} {e['detail']}")
    print(f"Wrote {FEATURE_CSV.relative_to(ROOT)} and {FEATURE_META_JSON.relative_to(ROOT)}")
    return meta


if __name__ == "__main__":
    main()
