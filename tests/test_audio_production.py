"""Tests for the production scoring core (:mod:`src.audio.production`).

These exercise the *numeric* production model — bundle loading/validation,
representative-coefficient averaging, single-vector scoring, local
explanations, the in-domain reliability flag, and model-card assembly. They
require only the Phase 1 scientific stack (numpy / scikit-learn / joblib) plus
the frozen ``artifacts_audio/`` bundle; they do **not** need openSMILE, so the
core stays testable in a minimal environment. The one test that touches the
live eGeMAPSv02 extractor is individually gated.

Guard rails honoured here:
* The synthetic / out-of-domain inputs are only ever asserted on *structure*
  and *determinism* — never on a predicted class label.
* Every model-card number is checked against the frozen artifact it is read
  from, so the card cannot silently drift from the evidence.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.audio.config import EXPECTED_FEATURE_COUNT, FEATURE_SET
from src.audio.production import (
    PRODUCTION_MODEL_ID,
    PRODUCTION_MODEL_TAG,
    PRODUCTION_MODEL_VERSION,
    REPEATED_CV_PATH,
    ProductionModelError,
    build_model_card,
    input_domain_report,
    load_bundle,
    local_contributions,
    model_card_summary,
    predict_from_vector,
    representative_coefficients,
)

# ---------------------------------------------------------------- fixtures/helpers


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


def _real_feature_vector(bundle) -> np.ndarray:
    """One genuine in-domain 88-dim vector, read from the frozen feature CSV.

    Using a real training-corpus row (not synthetic noise) means the standardised
    values sit inside the range the scaler was fit on, so this exercises the
    normal, in-domain scoring path.
    """
    import pandas as pd

    from src.audio.features import FEATURE_CSV

    frame = pd.read_csv(FEATURE_CSV)
    return frame.loc[0, bundle["feature_names"]].to_numpy(dtype=float)


# ---------------------------------------------------------------- bundle loading


def test_bundle_loads_and_validates(bundle):
    for key in ("model", "scaler", "threshold", "feature_names", "feature_group"):
        assert key in bundle
    assert bundle["feature_group"] == FEATURE_SET
    assert len(bundle["feature_names"]) == EXPECTED_FEATURE_COUNT
    assert 0.0 < float(bundle["threshold"]) < 1.0


def test_bundle_is_cached_singleton():
    assert load_bundle() is load_bundle()


def test_representative_coefficients_shape_and_augmentation(bundle):
    assert bundle["rep_coef"].shape == (EXPECTED_FEATURE_COUNT,)
    assert np.isfinite(bundle["rep_coef"]).all()
    assert np.isfinite(bundle["rep_intercept"])


def test_representative_coefficients_bare_estimator_fallback():
    """A plain linear model (no calibration folds) uses its own coef_/intercept_."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 5))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    lr = LogisticRegression().fit(x, y)
    coef, intercept = representative_coefficients(lr)
    assert coef.shape == (5,)
    np.testing.assert_allclose(coef, lr.coef_.reshape(-1))
    assert intercept == pytest.approx(float(lr.intercept_[0]))


# ---------------------------------------------------------------- scoring


def test_predict_from_vector_payload_structure(bundle):
    result = predict_from_vector(_real_feature_vector(bundle), bundle)

    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["threshold"] == pytest.approx(float(bundle["threshold"]))
    assert result["model_id"] == PRODUCTION_MODEL_ID
    assert result["model_version"] == PRODUCTION_MODEL_VERSION
    assert result["model_tag"] == PRODUCTION_MODEL_TAG
    assert result["feature_set"] == FEATURE_SET
    assert result["n_features"] == EXPECTED_FEATURE_COUNT
    # Decision is consistent with the reported probability and threshold.
    assert result["prediction"] == int(result["probability"] >= result["threshold"])

    dom = result["input_domain"]
    assert set(dom) == {
        "in_domain",
        "max_abs_z",
        "n_features_out_of_range",
        "extreme_z_threshold",
        "note",
    }

    contribs = result["explanation"]["top_contributions"]
    assert len(contribs) == 8
    for c in contribs:
        assert set(c) == {"feature", "family", "z", "contribution", "direction"}
        assert c["direction"] in ("PD", "HC")


def test_real_corpus_vector_is_in_domain(bundle):
    """A genuine training-corpus recording must not trip the OOD reliability flag."""
    result = predict_from_vector(_real_feature_vector(bundle), bundle)
    assert result["input_domain"]["in_domain"] is True


def test_predict_is_deterministic(bundle):
    vec = _real_feature_vector(bundle)
    a = predict_from_vector(vec, bundle)
    b = predict_from_vector(vec, bundle)
    assert a["probability"] == b["probability"]
    assert a["prediction"] == b["prediction"]


def test_predict_rejects_wrong_length(bundle):
    with pytest.raises(ProductionModelError):
        predict_from_vector(np.zeros(EXPECTED_FEATURE_COUNT - 1), bundle)


def test_predict_rejects_non_finite(bundle):
    vec = _real_feature_vector(bundle)
    vec[3] = np.nan
    with pytest.raises(ProductionModelError):
        predict_from_vector(vec, bundle)


# ---------------------------------------------------------------- explanation


def test_local_contributions_ranked_and_signed(bundle):
    """Contributions are coef*z, ranked by |value|, with a matching direction."""
    x_std = np.zeros(EXPECTED_FEATURE_COUNT)
    coef = bundle["rep_coef"]
    pos = int(np.argmax(coef))  # a feature with a positive (PD-ward) weight
    neg = int(np.argmin(coef))
    x_std[pos] = 3.0
    x_std[neg] = 3.0
    rows = local_contributions(x_std, bundle, top_k=8)

    magnitudes = [abs(r["contribution"]) for r in rows]
    assert magnitudes == sorted(magnitudes, reverse=True)
    for r in rows:
        assert r["direction"] == ("PD" if r["contribution"] > 0 else "HC")


def test_local_contributions_respects_top_k(bundle):
    x_std = np.ones(EXPECTED_FEATURE_COUNT)
    assert len(local_contributions(x_std, bundle, top_k=3)) == 3


# ---------------------------------------------------------------- domain report


def test_input_domain_report_in_range():
    rep = input_domain_report(np.linspace(-2.0, 2.0, EXPECTED_FEATURE_COUNT))
    assert rep["in_domain"] is True
    assert rep["n_features_out_of_range"] == 0
    assert rep["max_abs_z"] <= 2.0 + 1e-9


def test_input_domain_report_flags_extreme():
    x = np.zeros(EXPECTED_FEATURE_COUNT)
    x[0] = 50.0  # far beyond EXTREME_Z
    rep = input_domain_report(x)
    assert rep["in_domain"] is False
    assert rep["n_features_out_of_range"] == 1
    assert rep["max_abs_z"] == pytest.approx(50.0)
    assert "unreliable" in rep["note"].lower()


# ---------------------------------------------------------------- model card


def test_model_card_matches_frozen_repeated_cv():
    """The headline number is exactly the value in the frozen repeated-CV artifact."""
    card = build_model_card()
    frozen = json.loads(REPEATED_CV_PATH.read_text(encoding="utf-8"))
    lr = frozen["logistic_regression"]["metrics"]["roc_auc"]
    headline = card["evaluation"]["headline_repeated_cv"]
    assert headline["roc_auc_mean"] == lr["mean"]
    assert headline["roc_auc_std"] == lr["std"]
    # Sanity: the honest headline sits around the documented ~0.76, not 0.884.
    assert 0.70 <= headline["roc_auc_mean"] <= 0.82


def test_model_card_documents_rejected_cnn_and_skipped_embeddings():
    card = build_model_card()
    rejected = card["rejected_alternatives"]
    assert rejected["compact_cnn_phase2d"]["decision"] == "rejected"
    assert "not run" in rejected["pretrained_embeddings_wav2vec2_wavlm"]["decision"]


def test_model_card_disclaims_clinical_use():
    card = build_model_card()
    text = (card["disclaimer"] + card["intended_use"]).lower()
    assert "not a" in text
    assert "diagnostic" in text or "clinical" in text or "medical" in text


def test_model_card_summary_headline_is_repeated_cv():
    summary = model_card_summary()
    assert summary["tag"] == PRODUCTION_MODEL_TAG
    assert summary["feature_extractor"]["n_features"] == EXPECTED_FEATURE_COUNT
    assert "repeated" in summary["headline_metric"]["name"].lower()
    assert summary["headline_metric"]["mean"] == (
        build_model_card()["evaluation"]["headline_repeated_cv"]["roc_auc_mean"]
    )


# ---------------------------------------------------------------- live extractor (gated)


def test_live_smile_layout_matches_frozen_model(bundle):
    """The installed openSMILE must emit exactly the frozen feature ordering."""
    pytest.importorskip("opensmile")
    from src.audio.production import get_smile

    smile = get_smile()  # raises ProductionModelError on any layout mismatch
    assert list(smile.feature_names) == bundle["feature_names"]
