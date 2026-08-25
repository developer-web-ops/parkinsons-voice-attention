"""Explainability for the frozen Logistic-Regression audio baseline.

The reference model is a linear classifier on standardised eGeMAPSv02 features,
so three convergent, deterministic views of feature influence are produced
(Phase 2C objectives 6-8):

1. **Standardised coefficients** — the exact log-odds effect per standard
   deviation of each feature (native to a linear model).
2. **SHAP values** (``shap.LinearExplainer``) — mean(|SHAP|) per feature. For a
   linear model SHAP values are proportional to ``coef * (x - E[x])``, so their
   ranking should closely track |coef|; the Spearman agreement between the two is
   reported as a sanity check rather than treated as independent evidence.
3. **Coefficient stability across folds** — mean, std and sign-consistency of
   each standardised coefficient across the 5 folds of the canonical seed=42
   split. Features that are both large and sign-stable are the robustly
   influential ones, tying explainability to the Phase 2C robustness theme.

Effects are aggregated to acoustic families (:mod:`src.audio.families`) for a
compact summary. SHAP is applied to the *uncalibrated* logistic regression
(calibration is a monotone post-hoc map that does not change feature ranking).
Describing a linear model's coefficients states *what the model uses*; it is not
a claim about Parkinson's pathophysiology.
"""

from __future__ import annotations

import numpy as np

from src.audio.baseline_cv import make_models
from src.audio.families import ARTEFACT_SUSCEPTIBLE_FAMILIES, FAMILY_ORDER, feature_family
from src.data import (
    Dataset,
    cv_splits,
    fit_scaler,
    inner_subject_split,
)

REFERENCE_MODEL = "logistic_regression"


def fit_reference(data: Dataset, seed: int = 42):
    """Fit the frozen LR configuration on standardised full data.

    Returns ``(model, scaler, X_standardised)``. This all-data fit is used only
    for explainability; the *reported* generalisation metrics come from the
    grouped cross-validation, never from this fit.
    """
    scaler = fit_scaler(data.X)
    x_std = scaler.transform(data.X.to_numpy())
    model = make_models(seed)[REFERENCE_MODEL]
    model.fit(x_std, data.y)
    return model, scaler, x_std


def coefficient_importance(data: Dataset, seed: int = 42) -> list[dict]:
    """Standardised LR coefficients, ranked by |coef| (descending)."""
    model, _scaler, _x = fit_reference(data, seed)
    coef = np.asarray(model.coef_).reshape(-1)
    names = data.feature_names
    order = np.argsort(-np.abs(coef))
    return [
        {
            "feature": names[i],
            "family": feature_family(names[i]),
            "coef": float(coef[i]),
            "abs_coef": float(abs(coef[i])),
            "direction": "PD" if coef[i] > 0 else "HC",
        }
        for i in order
    ]


def shap_importance(data: Dataset, seed: int = 42):
    """mean(|SHAP|) per feature via ``shap.LinearExplainer`` on the standardised model.

    Returns ``(ranked, shap_values, names)`` where ``ranked`` is sorted by
    mean(|SHAP|) descending. Requires the optional ``shap`` dependency.
    """
    import shap  # optional dependency; imported lazily

    model, _scaler, x_std = fit_reference(data, seed)
    explainer = shap.LinearExplainer(model, x_std)
    values = np.asarray(explainer.shap_values(x_std))
    if values.ndim == 3:  # some versions return (n, features, classes)
        values = values[..., -1]
    mean_abs = np.abs(values).mean(axis=0)
    mean_signed = values.mean(axis=0)
    # A centered LinearExplainer has ~zero-mean SHAP per feature, so mean_signed
    # is uninformative for direction; the model coefficient sign is authoritative
    # for a linear model and is used for the PD/HC direction instead.
    coef = np.asarray(model.coef_).reshape(-1)
    names = data.feature_names
    order = np.argsort(-mean_abs)
    ranked = [
        {
            "feature": names[i],
            "family": feature_family(names[i]),
            "mean_abs_shap": float(mean_abs[i]),
            "mean_signed_shap": float(mean_signed[i]),
            "direction": "PD" if coef[i] > 0 else "HC",
        }
        for i in order
    ]
    return ranked, values, names


def coefficient_stability(data: Dataset, seed: int = 42, n_splits: int = 5) -> list[dict]:
    """Std and sign-consistency of standardised LR coefficients across CV folds.

    Each fold fits the LR on that fold's inner-training subjects only (leakage
    safe, mirroring the evaluation pipeline). ``sign_consistency`` is 1.0 when a
    coefficient keeps the same sign in every fold.
    """
    names = data.feature_names
    per_fold = []
    for train_all, _test in cv_splits(data, n_splits=n_splits, seed=seed):
        inner_train, _val = inner_subject_split(data, train_all, seed=seed)
        scaler = fit_scaler(data.X.iloc[inner_train])
        x_tr = scaler.transform(data.X.iloc[inner_train].to_numpy())
        model = make_models(seed)[REFERENCE_MODEL]
        model.fit(x_tr, data.y[inner_train])
        per_fold.append(np.asarray(model.coef_).reshape(-1))
    coefs = np.vstack(per_fold)  # (n_folds, n_features)
    mean = coefs.mean(axis=0)
    std = coefs.std(axis=0, ddof=1) if coefs.shape[0] > 1 else np.zeros(coefs.shape[1])
    sign_consistency = np.abs(np.sign(coefs).sum(axis=0)) / coefs.shape[0]
    order = np.argsort(-np.abs(mean))
    return [
        {
            "feature": names[i],
            "family": feature_family(names[i]),
            "mean_coef": float(mean[i]),
            "std_coef": float(std[i]),
            "sign_consistency": float(sign_consistency[i]),
        }
        for i in order
    ]


def family_importance(shap_ranked: list[dict]) -> list[dict]:
    """Aggregate mean(|SHAP|) by acoustic family and report each family's share."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in shap_ranked:
        fam = row["family"]
        totals[fam] = totals.get(fam, 0.0) + row["mean_abs_shap"]
        counts[fam] = counts.get(fam, 0) + 1
    grand = sum(totals.values()) or 1.0
    families = [f for f in FAMILY_ORDER if f in totals]
    families += [f for f in totals if f not in FAMILY_ORDER]
    return [
        {
            "family": f,
            "sum_mean_abs_shap": float(totals[f]),
            "share": float(totals[f] / grand),
            "n_features": counts[f],
        }
        for f in families
    ]


def artefact_family_share(family_rows: list[dict]) -> dict:
    """Share of total importance sitting in recording-condition-susceptible families.

    Loudness/Energy and Temporal/Rhythm can reflect recording gain or utterance
    length rather than voice pathology; a large share there is a confound flag.
    """
    susceptible = sum(
        r["share"] for r in family_rows if r["family"] in ARTEFACT_SUSCEPTIBLE_FAMILIES
    )
    return {
        "artefact_susceptible_families": list(ARTEFACT_SUSCEPTIBLE_FAMILIES),
        "combined_share": float(susceptible),
        "voice_intrinsic_share": float(1.0 - susceptible),
    }


def rank_agreement(coef_ranked: list[dict], shap_ranked: list[dict]) -> dict:
    """Spearman rank correlation between |coef| and mean(|SHAP|) orderings."""
    from scipy.stats import spearmanr

    abs_coef = {r["feature"]: r["abs_coef"] for r in coef_ranked}
    abs_shap = {r["feature"]: r["mean_abs_shap"] for r in shap_ranked}
    feats = list(abs_coef)
    rho, p = spearmanr([abs_coef[f] for f in feats], [abs_shap[f] for f in feats])
    return {"spearman_rho": float(rho), "p_value": float(p), "n_features": len(feats)}
