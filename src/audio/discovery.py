"""Discover MDVR-KCL WAV files and derive leakage-safe subject/label metadata.

Responsibilities (all deterministic, no audio decoding here):

* find every ``*.wav`` under the raw dataset directory;
* parse the **subject id** with a tolerant matcher (``ID(\\d+)``) so the ID22
  filename anomaly (``ID22hc_0_0_0.wav``, missing underscore) resolves to subject
  ``22`` rather than a phantom ``ID22hc`` — this is what keeps subject-grouped CV
  honest (see the Phase 2A manifest ``known_anomalies``);
* derive the **primary PD/HC label from the folder** (``HC``/``PD``) and assert
  it agrees with the cohort token in the filename (``hc``/``pd``);
* never silently discard a recording — anything unparseable or inconsistent is
  returned as an :class:`Exclusion` with a reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Tolerant subject-id matcher. Deliberately NOT str.split("_")[0]: that would
# turn "ID22hc_0_0_0.wav" into subject "ID22hc" and leak subject 22 across folds.
SUBJECT_RE = re.compile(r"ID(\d+)", re.IGNORECASE)
# Cohort token in the filename; optional underscore tolerates the ID22 anomaly.
COHORT_RE = re.compile(r"ID\d+_?(hc|pd)", re.IGNORECASE)

LABELS = {"hc": 0, "pd": 1}  # 0 = healthy control, 1 = Parkinson's disease
LABEL_NAMES = {0: "HC", 1: "PD"}

_TASK_CANON = {"readtext": "ReadText", "spontaneousdialogue": "SpontaneousDialogue"}


@dataclass(frozen=True)
class Recording:
    """One usable WAV file with parsed, leakage-safe metadata."""

    path: Path
    subject_id: int  # numeric subject id (group key for CV)
    label: int  # 0 = HC, 1 = PD (derived from the folder)
    cohort_folder: str  # "HC" | "PD"
    cohort_file: str  # "hc" | "pd" (asserted to agree with the folder)
    task: str  # "ReadText" | "SpontaneousDialogue" | raw folder name
    filename: str

    @property
    def group_key(self) -> str:
        """Stable subject key used to group recordings across folds."""
        return f"S{self.subject_id:02d}"


@dataclass(frozen=True)
class Exclusion:
    """A WAV that could not be turned into a valid :class:`Recording`."""

    path: Path
    reason: str
    detail: str = ""


def _find_in_parts(parts: tuple[str, ...], candidates: set[str]) -> str | None:
    """Return the first path component matching one of ``candidates`` (casefold)."""
    for part in parts:
        if part.casefold() in candidates:
            return part
    return None


def parse_subject_id(filename: str) -> int | None:
    m = SUBJECT_RE.search(filename)
    return int(m.group(1)) if m else None


def parse_cohort_from_filename(filename: str) -> str | None:
    m = COHORT_RE.search(filename)
    return m.group(1).lower() if m else None


def _classify(path: Path) -> Recording | Exclusion:
    parts = path.parts
    name = path.name

    subject_id = parse_subject_id(name)
    if subject_id is None:
        return Exclusion(path, "no_subject_id", f"no ID<digits> in {name!r}")

    folder = _find_in_parts(parts, {"hc", "pd"})
    if folder is None:
        return Exclusion(path, "no_cohort_folder", "no HC/PD ancestor directory")
    cohort_folder = folder.upper()

    cohort_file = parse_cohort_from_filename(name)
    if cohort_file is None:
        return Exclusion(path, "no_cohort_in_filename", f"no hc/pd token in {name!r}")

    # Primary label comes from the folder; the filename must agree.
    if cohort_folder.lower() != cohort_file:
        return Exclusion(
            path,
            "label_folder_filename_mismatch",
            f"folder={cohort_folder} but filename cohort={cohort_file}",
        )

    task_part = _find_in_parts(parts, set(_TASK_CANON))
    task = _TASK_CANON.get(task_part.casefold(), task_part) if task_part else "unknown"

    return Recording(
        path=path,
        subject_id=subject_id,
        label=LABELS[cohort_file],
        cohort_folder=cohort_folder,
        cohort_file=cohort_file,
        task=task,
        filename=name,
    )


def discover_recordings(raw_dir: Path) -> tuple[list[Recording], list[Exclusion]]:
    """Walk ``raw_dir`` for WAV files. Returns ``(recordings, exclusions)``.

    Results are sorted by path for deterministic ordering.
    """
    raw_dir = Path(raw_dir)
    wavs = sorted(p for p in raw_dir.rglob("*.wav") if p.is_file())
    recordings: list[Recording] = []
    exclusions: list[Exclusion] = []
    for wav in wavs:
        result = _classify(wav)
        if isinstance(result, Recording):
            recordings.append(result)
        else:
            exclusions.append(result)
    return recordings, exclusions
