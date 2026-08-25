"""Deterministic audio preprocessing + feature-extraction configuration.

Every choice that affects the extracted features lives here as an explicit,
serialisable value so a result can be traced to the exact configuration that
produced it. Importing this module has no side effects and does not require
openSMILE (the ``opensmile`` package is only needed by :mod:`src.audio.features`).

Design notes for the defaults (documented, not invented silently):

* ``target_sample_rate = 44100`` — the MDVR-KCL recordings are natively
  44.1 kHz. We keep the native rate rather than down-sampling: eGeMAPSv02 pitch
  and formant descriptors benefit from the full band, and avoiding a resample
  removes a lossy, implementation-dependent step. A resample *policy* is still
  implemented (:func:`src.audio.preprocess.resample_to`) and used only if a
  non-conforming file ever appears.
* ``normalization = "none"`` — eGeMAPSv02 is standardly applied to the signal
  as recorded, and reduced loudness (hypophonia) is itself a clinically relevant
  Parkinsonian sign, so we do not neutralise amplitude by default. ``"peak"`` is
  implemented and tested as a selectable alternative (useful if device-gain
  differences later prove to be a confound); the trade-off is that peak
  normalisation removes absolute-loudness information.
* ``silence = "none"`` — pause and hesitation structure is diagnostically
  informative in Parkinsonian speech, so no silence is trimmed by default. The
  field exists to make the policy explicit and revisitable.
* ``min_duration_s`` / ``max_duration_s`` — recordings outside this band are
  *excluded with a recorded reason*, never silently dropped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# openSMILE feature-set identity. Pinned so the manifest records exactly what was
# extracted; see requirements-audio.txt for the package version pin.
FEATURE_SET = "eGeMAPSv02"
FEATURE_LEVEL = "Functionals"
EXPECTED_FEATURE_COUNT = 88  # eGeMAPSv02 Functionals; asserted at extraction time
OPENSMILE_PACKAGE = "opensmile==2.6.0"

NormalizationPolicy = str  # "none" | "peak"
SilencePolicy = str  # "none" | "trim_edges"


@dataclass(frozen=True)
class AudioConfig:
    """Immutable description of the deterministic audio → features transform."""

    target_sample_rate: int = 44100
    mono: bool = True
    normalization: NormalizationPolicy = "none"
    peak_dbfs: float = -1.0  # only used when normalization == "peak"
    silence: SilencePolicy = "none"
    silence_threshold_dbfs: float = -40.0  # only used when silence == "trim_edges"
    min_duration_s: float = 0.5
    max_duration_s: float = 600.0
    feature_set: str = FEATURE_SET
    feature_level: str = FEATURE_LEVEL
    opensmile_package: str = OPENSMILE_PACKAGE
    expected_feature_count: int = EXPECTED_FEATURE_COUNT

    def __post_init__(self) -> None:
        if self.normalization not in ("none", "peak"):
            raise ValueError(f"unknown normalization policy: {self.normalization!r}")
        if self.silence not in ("none", "trim_edges"):
            raise ValueError(f"unknown silence policy: {self.silence!r}")
        if self.target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")
        if not (0 < self.min_duration_s < self.max_duration_s):
            raise ValueError("require 0 < min_duration_s < max_duration_s")

    def as_dict(self) -> dict:
        """JSON-serialisable view for manifests and reproducibility reports."""
        return asdict(self)


DEFAULT_CONFIG = AudioConfig()
