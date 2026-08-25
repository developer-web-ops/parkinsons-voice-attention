"""Reproducibility manifest for the Phase 2B audio pipeline (torch-free).

Mirrors :mod:`src.reproducibility` in spirit but records the things that make an
*audio* result reproducible: the openSMILE version and feature-set identity, the
:class:`AudioConfig`, the SHA-256 of the dataset manifest, and the SHA-256 of the
extracted feature table. It deliberately does not import torch or the UCI loader.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from src.audio.config import DEFAULT_CONFIG, AudioConfig
from src.audio.features import FEATURE_CSV, MANIFEST_PATH

ROOT = Path(__file__).resolve().parents[2]


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=ROOT,
            check=False,
        )
        return result.stdout.strip() or None
    except (OSError, ValueError):
        return None


def _sha256(path: Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pkg_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def audio_repro_metadata(
    seed: int = 42,
    config: AudioConfig = DEFAULT_CONFIG,
    feature_csv: Path = FEATURE_CSV,
    manifest_path: Path = MANIFEST_PATH,
) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "opensmile": _pkg_version("opensmile"),
            "audiofile": _pkg_version("audiofile"),
            "soundfile": _pkg_version("soundfile"),
        },
        "git_commit": _git_commit(),
        "feature_set": config.feature_set,
        "feature_level": config.feature_level,
        "audio_config": config.as_dict(),
        "dataset_manifest_file": Path(manifest_path).name,
        "dataset_manifest_sha256": _sha256(manifest_path),
        "feature_dataset_file": Path(feature_csv).name,
        "feature_dataset_sha256": _sha256(feature_csv),
    }
