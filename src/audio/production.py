"""Production scoring + explainability for the eGeMAPSv02 + calibrated-LR model.

This module is the single, framework-free home for the *scientific* logic of the
production audio model:

* loading the frozen deployable bundle (``artifacts_audio/models/logistic_regression.joblib``),
* validating that the live openSMILE feature layout still matches the layout the
  model was trained on (feature ordering is immutable and checked, never assumed),
* turning an 88-dim eGeMAPSv02 vector into a calibrated probability, a decision at
  the frozen operating threshold, and a per-feature local explanation,
* assembling the versioned model card from the frozen Phase 2B/2C/2D artifacts.

It deliberately contains **no** file-upload / HTTP / audio-I/O concerns — those
live in :mod:`app.audio_inference`. Keeping the numeric core here makes it unit
testable without FastAPI or a WAV on disk.

Explainability note (honest by construction): the deployed classifier is a
``CalibratedClassifierCV`` wrapping five Platt-scaled logistic-regression folds.
The *representative coefficients* used for explanation are the mean of those five
folds' standardised LR coefficients. A local contribution is ``coef * x_std`` —
the additive log-odds effect of each feature on the model's internal linear
score, which the calibration layer then maps *monotonically* to the reported
probability. So contribution signs and relative magnitudes are faithful; the
contributions do not sum exactly to ``logit(probability)`` because of the Platt
maps and fold averaging. Describing what a linear model weighs is a statement
about the model, never a claim about Parkinson's pathophysiology.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.audio.config import (
    DEFAULT_CONFIG,
    EXPECTED_FEATURE_COUNT,
    FEATURE_LEVEL,
    FEATURE_SET,
)
from src.audio.families import feature_family

logger = logging.getLogger(__name__)

# --- Identity of the production model (immutable, versioned) -----------------
PRODUCTION_MODEL_ID = "egemaps-lr"
PRODUCTION_MODEL_VERSION = "1.0.0"
PRODUCTION_MODEL_DISPLAY_NAME = "eGeMAPSv02 + Calibrated Logistic Regression"
# Human-facing identifier returned by the API, e.g. "egemaps-lr@1.0.0".
PRODUCTION_MODEL_TAG = f"{PRODUCTION_MODEL_ID}@{PRODUCTION_MODEL_VERSION}"

# --- Frozen artifact locations (repo-relative) -------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_PATH = _REPO_ROOT / "artifacts_audio" / "models" / "logistic_regression.joblib"
FROZEN_REFERENCE_PATH = (
    _REPO_ROOT / "artifacts_audio" / "phase2c" / "frozen_baseline_reference.json"
)
REPEATED_CV_PATH = _REPO_ROOT / "artifacts_audio" / "phase2c" / "repeated_cv.json"
CNN_SUMMARY_PATH = _REPO_ROOT / "artifacts_audio" / "cnn" / "cnn_summary.json"
MODEL_CARD_PATH = _REPO_ROOT / "artifacts_audio" / "production" / "model_card.json"

# --- Explanation / reliability parameters ------------------------------------
# Number of feature contributions surfaced per prediction.
TOP_K_CONTRIBUTIONS = 8
# Per-feature |z| above which a standardised feature is treated as far outside
# the range seen in the 73-recording training set. This is a conservative,
# heuristic in-domain guard (NOT a validated OOD detector): real speech features
# standardised by the frozen scaler rarely exceed a few SD, whereas non-speech
# audio (silence, tones, noise, corrupt files) produces extreme values for which
# the calibrated probability is not trustworthy. Inference never clips features
# (that would break train/inference parity); this only raises a reliability flag.
EXTREME_Z = 8.0


class ProductionModelError(RuntimeError):
    """Raised when the frozen production bundle cannot be loaded/validated."""


def representative_coefficients(model: Any) -> tuple[np.ndarray, float]:
    """Mean standardised LR coefficients (and intercept) across calibration folds.

    The deployed model is a ``CalibratedClassifierCV`` with ``cv=5``; each of the
    five ``calibrated_classifiers_`` wraps a fitted base ``LogisticRegression``.
    Averaging their ``coef_`` gives a single representative linear direction for
    explanation. Falls back to ``model.coef_`` if the object is a plain linear
    model (used by tests with a bare estimator).
    """
    subs = getattr(model, "calibrated_classifiers_", None)
    if subs:
        coefs = np.vstack([np.asarray(cc.estimator.coef_).reshape(-1) for cc in subs])
        intercepts = np.array(
            [float(np.asarray(cc.estimator.intercept_).reshape(-1)[0]) for cc in subs]
        )
        return coefs.mean(axis=0), float(intercepts.mean())
    coef = np.asarray(model.coef_).reshape(-1)
    intercept = float(np.asarray(model.intercept_).reshape(-1)[0])
    return coef, intercept


@lru_cache(maxsize=1)
def load_bundle() -> dict[str, Any]:
    """Load and validate the frozen deployable bundle.

    Returns a dict augmented with ``rep_coef`` / ``rep_intercept`` (representative
    coefficients) alongside the persisted ``model``/``scaler``/``threshold``/
    ``feature_names``. Raises :class:`ProductionModelError` on any structural
    problem so deployment fails loudly rather than scoring with a wrong layout.
    """
    logger.info("Production model path: %s", BUNDLE_PATH)
    logger.info("Production model exists: %s", BUNDLE_PATH.exists())

    if BUNDLE_PATH.exists():
        logger.info(
            "Production model size: %d bytes",
            BUNDLE_PATH.stat().st_size,
        )

    if not BUNDLE_PATH.exists():
        raise ProductionModelError(
            f"Production model bundle is missing: {BUNDLE_PATH.name}"
        )

    try:
        bundle = joblib.load(BUNDLE_PATH)
    except Exception as exc:
        logger.exception("Could not load production model bundle")
        raise ProductionModelError(
            f"Could not load production model bundle: {exc}"
        ) from exc

    required = {"model", "scaler", "threshold", "feature_names", "feature_group"}
    missing = required - set(bundle)
    if missing:
        raise ProductionModelError(
            f"Production bundle missing keys: {sorted(missing)}"
        )

    names = list(bundle["feature_names"])

    if len(names) != EXPECTED_FEATURE_COUNT:
        raise ProductionModelError(
            f"Production bundle has {len(names)} features, "
            f"expected {EXPECTED_FEATURE_COUNT}."
        )

    if bundle["feature_group"] != FEATURE_SET:
        raise ProductionModelError(
            f"Production bundle feature_group={bundle['feature_group']!r}, "
            f"expected {FEATURE_SET!r}."
        )

    rep_coef, rep_intercept = representative_coefficients(bundle["model"])

    if rep_coef.shape[0] != len(names):
        raise ProductionModelError(
            "Representative coefficient length does not match feature count."
        )

    bundle = dict(bundle)
    bundle["rep_coef"] = rep_coef
    bundle["rep_intercept"] = rep_intercept
    bundle["feature_names"] = names

    return bundle


@lru_cache(maxsize=1)
def _smile():
    """Build the eGeMAPSv02 extractor once and assert its layout matches the model.

    openSMILE is imported lazily here so importing this module (e.g. to build the
    model card) does not require the audio extras. The immutability of feature
    ordering is enforced: if the installed openSMILE ever emits a different set
    or order of names, scoring is refused rather than silently misaligned.
    """
    from src.audio.features import build_smile

    smile = build_smile(DEFAULT_CONFIG)
    bundle = load_bundle()
    live = list(smile.feature_names)
    if live != bundle["feature_names"]:
        raise ProductionModelError(
            "Installed openSMILE eGeMAPSv02 feature layout does not match the frozen "
            "model's feature ordering; refusing to score. Check the openSMILE version."
        )
    return smile


def get_smile():
    """Public accessor for the validated, cached eGeMAPSv02 extractor."""
    return _smile()


def local_contributions(
    x_std: np.ndarray,
    bundle: dict[str, Any],
    top_k: int = TOP_K_CONTRIBUTIONS,
) -> list[dict[str, Any]]:
    """Per-feature log-odds contributions ``coef * x_std``, ranked by |contribution|.

    Each entry names the eGeMAPS feature, its acoustic family, the standardised
    input value ``z``, the additive contribution to the model's internal linear
    score, and the direction that magnitude pushes toward (PD/HC).
    """
    coef = bundle["rep_coef"]
    names = bundle["feature_names"]
    x = np.asarray(x_std, dtype=float).reshape(-1)
    contrib = coef * x
    order = np.argsort(-np.abs(contrib))[: max(0, int(top_k))]
    rows: list[dict[str, Any]] = []
    for i in order:
        rows.append(
            {
                "feature": names[i],
                "family": feature_family(names[i]),
                "z": float(x[i]),
                "contribution": float(contrib[i]),
                "direction": "PD" if contrib[i] > 0 else "HC",
            }
        )
    return rows


def input_domain_report(x_std: np.ndarray) -> dict[str, Any]:
    """Reliability flag for how far a standardised input sits outside training range."""
    x = np.abs(np.asarray(x_std, dtype=float).reshape(-1))
    max_abs_z = float(x.max()) if x.size else 0.0
    n_extreme = int((x > EXTREME_Z).sum())
    in_domain = n_extreme == 0
    return {
        "in_domain": in_domain,
        "max_abs_z": max_abs_z,
        "n_features_out_of_range": n_extreme,
        "extreme_z_threshold": EXTREME_Z,
        "note": (
            "Standardised acoustic features are within the range seen in the training set."
            if in_domain
            else (
                "One or more standardised features fall far outside the training range; "
                "this input does not resemble the speech the model was validated on "
                "(e.g. non-speech audio, silence, tones, or a corrupt recording), so the "
                "probability is unreliable and should not be interpreted."
            )
        ),
    }


def predict_from_vector(vector: np.ndarray, bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    """Score one raw 88-dim eGeMAPSv02 vector into the full prediction payload.

    Steps mirror training exactly: standardise with the frozen scaler, take the
    calibrated positive-class probability, decide at the frozen operating
    threshold, then attach the local explanation and the in-domain reliability
    report. No feature clipping is applied (train/inference parity).
    """
    bundle = bundle or load_bundle()
    names = bundle["feature_names"]
    vec = np.asarray(vector, dtype=float).reshape(-1)
    if vec.shape[0] != len(names):
        raise ProductionModelError(
            f"Feature vector has {vec.shape[0]} values, expected {len(names)}."
        )
    if not np.all(np.isfinite(vec)):
        raise ProductionModelError("Feature vector contains non-finite values (NaN/Inf).")

    x_std = bundle["scaler"].transform(vec.reshape(1, -1))
    probability = float(bundle["model"].predict_proba(x_std)[0, 1])
    threshold = float(bundle["threshold"])
    prediction = int(probability >= threshold)
    domain = input_domain_report(x_std)

    return {
        "prediction": prediction,
        "label": "Parkinson's indicated" if prediction == 1 else "Healthy control indicated",
        "probability": probability,
        "threshold": threshold,
        "model_id": PRODUCTION_MODEL_ID,
        "model_version": PRODUCTION_MODEL_VERSION,
        "model": PRODUCTION_MODEL_DISPLAY_NAME,
        "model_tag": PRODUCTION_MODEL_TAG,
        "feature_set": FEATURE_SET,
        "feature_level": FEATURE_LEVEL,
        "n_features": len(names),
        "input_domain": domain,
        "explanation": {
            "method": "mean standardised LR coefficients x standardised feature (log-odds)",
            "basis": (
                "Additive contribution to the model's internal linear score, averaged over the "
                "five calibration folds; the calibration layer maps that score monotonically to "
                "the probability. Model explanation only, not a medical/causal claim."
            ),
            "top_contributions": local_contributions(x_std, bundle),
        },
    }


# --- Model card assembly (Phase A / H) ---------------------------------------
def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_card() -> dict[str, Any]:
    """Assemble the versioned model card from the frozen artifacts (no fabrication).

    Every number is read live from the frozen Phase 2B/2C/2D artifacts so the card
    cannot drift from the evidence. Regenerate with ``write_model_card()``.
    """
    bundle = load_bundle()
    ref = _read_json(FROZEN_REFERENCE_PATH)
    rcv = _read_json(REPEATED_CV_PATH)["logistic_regression"]
    cnn = _read_json(CNN_SUMMARY_PATH)

    lr_roc = rcv["metrics"]["roc_auc"]
    single = ref["subject_level"]["roc_auc"]

    return {
        "model_id": PRODUCTION_MODEL_ID,
        "version": PRODUCTION_MODEL_VERSION,
        "tag": PRODUCTION_MODEL_TAG,
        "display_name": PRODUCTION_MODEL_DISPLAY_NAME,
        "task": (
            "Binary classification of Parkinson's disease vs healthy control "
            "from a voice recording."
        ),
        "intended_use": (
            "Research / educational demonstration of an acoustic ML pipeline. NOT a medical "
            "device and NOT a diagnostic tool. Outputs must not be used for clinical decisions."
        ),
        "feature_extractor": {
            "set": FEATURE_SET,
            "level": FEATURE_LEVEL,
            "n_features": EXPECTED_FEATURE_COUNT,
            "toolkit": "openSMILE 2.6.0",
        },
        "classifier": {
            "estimator": (
                "CalibratedClassifierCV(LogisticRegression(C=0.1, "
                "class_weight='balanced', max_iter=5000, random_state=42), "
                "method='sigmoid', cv=5)"
            ),
            "preprocessing": "StandardScaler on the 88 eGeMAPSv02 functionals",
            "operating_threshold": float(bundle["threshold"]),
            "threshold_rule": "median of the seed=42 per-fold max-F1 thresholds",
        },
        "evaluation": {
            "methodology": (
                "Subject-grouped StratifiedGroupKFold; scaler, Platt calibration and decision "
                "threshold all fit strictly within training folds (leakage-safe). No subject "
                "appears in both train and test of any fold."
            ),
            "primary_metric_level": "subject_level",
            "headline_repeated_cv": {
                "description": "50x repeated 5-fold subject-grouped CV (the honest headline).",
                "roc_auc_mean": lr_roc["mean"],
                "roc_auc_std": lr_roc["std"],
                "roc_auc_p2_5": lr_roc["p2_5"],
                "roc_auc_p50": lr_roc["p50"],
                "roc_auc_p97_5": lr_roc["p97_5"],
                "n_repeats": rcv["n_repeats"],
                "n_splits": rcv["n_splits"],
            },
            "historical_single_split": {
                "description": (
                    "Phase 2B seed=42 single split with subject-cluster bootstrap CI. This "
                    "estimate lies within the variability of the repeated subject-grouped "
                    "splits but toward the favorable end of that distribution; it is not the "
                    "expected performance."
                ),
                "roc_auc_point": single["point"],
                "roc_auc_ci_low": single["ci_low"],
                "roc_auc_ci_high": single["ci_high"],
            },
            "secondary_repeated_cv": {
                "balanced_accuracy_mean": rcv["metrics"]["balanced_accuracy"]["mean"],
                "sensitivity_mean": rcv["metrics"]["sensitivity"]["mean"],
                "specificity_mean": rcv["metrics"]["specificity"]["mean"],
                "brier_score_mean": rcv["metrics"]["brier_score"]["mean"],
            },
        },
        "rejected_alternatives": {
            "compact_cnn_phase2d": {
                "repeated_roc_auc_mean": cnn["cnn_repeated_roc_auc"]["mean"],
                "repeated_roc_auc_std": cnn["cnn_repeated_roc_auc"]["std"],
                "single_split_roc_auc": cnn["cnn_single_split_subject_roc_auc"]["point"],
                "mean_delta_vs_lr": cnn["verdict"]["mean_delta_cnn_minus_lr"],
                "paired_folds_won": cnn["verdict"]["paired_folds_won"],
                "parameters": cnn["parameters"],
                "decision": "rejected",
                "reason": cnn["verdict"]["statement"],
            },
            "pretrained_embeddings_wav2vec2_wavlm": {
                "decision": "not run (documented skip)",
                "reason": (
                    "Large self-supervised speech transformers (~95M-300M params) require heavy "
                    "downloads and CPU inference unsuitable for the openSMILE deployment path, "
                    "would fragment the feature pipeline, and are at high overfitting risk on a "
                    "73-recording / 37-subject corpus where even a 6033-parameter CNN failed to "
                    "beat the linear baseline under repeated subject-grouped CV. Expected value "
                    "does not justify the compute; skipped per the model-decision rules."
                ),
            },
        },
        "data": {
            "corpus": "MDVR-KCL (Mobile Device Voice Recordings at King's College London)",
            "recordings": ref["data_balance"]["recordings"],
            "subjects": ref["data_balance"]["subjects"],
            "recordings_pd": ref["data_balance"]["recordings_pd"],
            "recordings_hc": ref["data_balance"]["recordings_hc"],
            "license": "CC-BY-4.0",
        },
        "provenance": ref["provenance"],
        "limitations": [
            "Very small corpus (37 subjects); metrics have wide uncertainty bands.",
            "Single-site, mobile-device recordings; not validated across devices, "
            "languages, or clinics.",
            "Predicts a statistical association with a diagnostic label, not disease presence.",
            "Out-of-domain audio (non-speech, silence, tones) yields unreliable probabilities; "
            "flagged via the in-domain reliability report.",
        ],
        "disclaimer": (
            "This is a research/ML prediction system, NOT a clinical diagnostic device. It has "
            "not been evaluated by any regulatory body. Do not use it to make medical decisions."
        ),
    }


def write_model_card(path: Path | None = None) -> Path:
    """Serialise :func:`build_model_card` to ``artifacts_audio/production/model_card.json``."""
    path = path or MODEL_CARD_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_model_card(), indent=2) + "\n", encoding="utf-8")
    return path


def model_card_summary() -> dict[str, Any]:
    """Compact model-card view for the API ``/api/audio/info`` endpoint."""
    card = build_model_card()
    return {
        "model_id": card["model_id"],
        "version": card["version"],
        "tag": card["tag"],
        "display_name": card["display_name"],
        "feature_extractor": card["feature_extractor"],
        "operating_threshold": card["classifier"]["operating_threshold"],
        "headline_metric": {
            "name": "ROC-AUC (50x repeated 5-fold subject-grouped CV)",
            "mean": card["evaluation"]["headline_repeated_cv"]["roc_auc_mean"],
            "std": card["evaluation"]["headline_repeated_cv"]["roc_auc_std"],
            "p2_5": card["evaluation"]["headline_repeated_cv"]["roc_auc_p2_5"],
            "p97_5": card["evaluation"]["headline_repeated_cv"]["roc_auc_p97_5"],
        },
        "data": card["data"],
        "disclaimer": card["disclaimer"],
    }
