"""Reproducibility manifest: seeds, library versions, git commit, dataset hash.

Embedded into every generated report so a result can be traced to the exact
code, dependencies and data that produced it.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from src.data import DATA_PATH

ROOT = Path(__file__).resolve().parents[1]


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


def dataset_sha256(path: Path = DATA_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repro_metadata(seed: int = 42) -> dict:
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
            "torch": torch.__version__,
        },
        "git_commit": _git_commit(),
        "dataset_file": DATA_PATH.name,
        "dataset_sha256": dataset_sha256(),
    }
