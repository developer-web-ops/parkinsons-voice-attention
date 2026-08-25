"""Group the 88 eGeMAPSv02 functionals into meaningful acoustic families.

The GeMAPS / eGeMAPS parameter set (Eyben et al., 2016, *IEEE Transactions on
Affective Computing*) organises its low-level descriptors into frequency,
energy/amplitude, spectral-balance and temporal groups. This module maps each of
the 88 *Functionals* feature names to a compact, human-readable acoustic family
so feature-importance results can be summarised by family (Phase 2C objective 8).

The mapping is a pure, deterministic function of the feature name; a test asserts
that every one of the 88 canonical names resolves to a known family (no
``"Other"``). The families are an interpretive grouping of the eGeMAPS parameters
and are used only to describe *what the model attends to* — not to make any
clinical claim.
"""

from __future__ import annotations

from collections.abc import Iterable

# Canonical display order for reports.
FAMILY_ORDER = [
    "Pitch (F0)",
    "Jitter",
    "Shimmer",
    "Loudness/Energy",
    "Harmonicity/Noise",
    "Formants",
    "Spectral Balance",
    "MFCC (Cepstral)",
    "Temporal/Rhythm",
]

# Voice-intrinsic families vs families that are more susceptible to recording
# conditions / utterance length (used by the confound analysis).
ARTEFACT_SUSCEPTIBLE_FAMILIES = ("Loudness/Energy", "Temporal/Rhythm")

_TEMPORAL_NAMES = frozenset(
    {
        "loudnessPeaksPerSec",
        "VoicedSegmentsPerSec",
        "MeanVoicedSegmentLengthSec",
        "StddevVoicedSegmentLengthSec",
        "MeanUnvoicedSegmentLength",
        "StddevUnvoicedSegmentLength",
    }
)


def feature_family(name: str) -> str:
    """Return the acoustic family for a single eGeMAPSv02 functional name."""
    if name.startswith("F0semitone"):
        return "Pitch (F0)"
    if name.startswith("jitterLocal"):
        return "Jitter"
    if name.startswith("shimmerLocaldB"):
        return "Shimmer"
    if name.startswith(("HNRdBACF", "logRelF0-H1")):
        return "Harmonicity/Noise"
    if name.startswith(("F1", "F2", "F3")):
        return "Formants"
    if name.startswith("mfcc"):
        return "MFCC (Cepstral)"
    if name.startswith(("alphaRatio", "hammarbergIndex", "slopeV", "slopeUV", "spectralFlux")):
        return "Spectral Balance"
    if name.startswith("loudness_sma3") or name.startswith("equivalentSoundLevel"):
        return "Loudness/Energy"
    if name in _TEMPORAL_NAMES:
        return "Temporal/Rhythm"
    return "Other"


def assign_families(names: Iterable[str]) -> dict[str, str]:
    """Map each feature name to its family (input order preserved by the caller)."""
    return {n: feature_family(n) for n in names}


def family_members(names: Iterable[str]) -> dict[str, list[str]]:
    """Invert to ``family -> [feature names]``, families in :data:`FAMILY_ORDER`."""
    members: dict[str, list[str]] = {}
    for n in names:
        members.setdefault(feature_family(n), []).append(n)
    ordered = {fam: members[fam] for fam in FAMILY_ORDER if fam in members}
    # Append any non-canonical families (e.g. "Other") after the known ones.
    for fam, cols in members.items():
        if fam not in ordered:
            ordered[fam] = cols
    return ordered
