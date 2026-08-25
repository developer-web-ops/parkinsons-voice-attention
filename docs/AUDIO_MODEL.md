# Native-Audio Parkinson's Voice Model — Technical Documentation

Production/reference model: **`egemaps-lr@1.0.0`** — openSMILE **eGeMAPSv02** functionals
(88 acoustic features) → **StandardScaler** → **calibrated logistic regression**, selected and
evaluated under **subject-grouped** cross-validation.

> **This is a research / educational ML system, NOT a medical device and NOT a diagnostic tool.**
> It has not been evaluated or cleared by any regulatory body. It predicts a *statistical
> association* with a diagnostic label from a very small corpus; it does not detect disease. Do not
> use it for any clinical decision. See [Limitations](#12-limitations) and [Disclaimer](#15-disclaimer).

This document is the comprehensive reference for the native-audio system (Phase 2). The Phase 1
tabular UCI system is documented in the top-level [`README.md`](../README.md) and is unchanged.

---

## Table of contents

1. [System architecture](#1-system-architecture)
2. [Data provenance and license](#2-data-provenance-and-license)
3. [Preprocessing](#3-preprocessing)
4. [eGeMAPSv02 feature extraction](#4-egemapsv02-feature-extraction)
5. [Model selection rationale](#5-model-selection-rationale)
6. [Subject-grouped evaluation methodology](#6-subject-grouped-evaluation-methodology)
7. [Final metrics and uncertainty](#7-final-metrics-and-uncertainty)
8. [Why the compact CNN was rejected](#8-why-the-compact-cnn-was-rejected)
9. [Why pretrained embeddings were not deployed](#9-why-pretrained-embeddings-were-not-deployed)
10. [API](#10-api)
11. [Frontend](#11-frontend)
12. [Limitations](#12-limitations)
13. [Local setup](#13-local-setup)
14. [Deployment](#14-deployment)
15. [Reproducibility and versioning](#15-reproducibility-and-versioning)
16. [Disclaimer](#16-disclaimer)

---

## 1. System architecture

```
        Browser (app/static)                     FastAPI service (app/main.py)
   ┌───────────────────────────┐            ┌──────────────────────────────────────┐
   │ Audio (native) tab        │  multipart │ POST /api/audio/predict               │
   │  • .wav upload / drop     │ ─────────► │  app/audio_inference.py               │
   │  • renders probability,   │            │   size guard → WAV validate (libsndfile)│
   │    threshold, top acoustic │            │   → preprocess (src/audio/preprocess) │
   │    contributions, OOD flag│ ◄───────── │   → eGeMAPSv02 (src/audio/features)   │
   └───────────────────────────┘   JSON     │   → src/audio/production.py            │
                                             │       scaler → calibrated LR →        │
   GET /api/audio/info  ◄────── model card   │       threshold → explanation + OOD   │
                                             └──────────────────────────────────────┘
                                                 frozen bundle: artifacts_audio/…
```

Two layers with a deliberate split of concerns:

- **`src/audio/production.py`** — the framework-free *scientific core*. Loads and validates the
  frozen bundle, enforces that the live openSMILE feature ordering matches the trained ordering,
  turns an 88-dim vector into a calibrated probability + decision + per-feature explanation +
  reliability flag, and assembles the versioned model card. No HTTP, no file I/O — unit-testable
  without FastAPI or a WAV on disk.
- **`app/audio_inference.py`** — the *serving layer*. Owns upload size limits, WAV validation,
  temp-file lifecycle, typed user-facing errors, and orchestration of the pipeline. It never lets
  an internal path or stack trace escape into a response.
- **`app/main.py`** — thin FastAPI endpoints (`/api/audio/info`, `/api/audio/predict`) that map the
  typed errors to HTTP status codes. The Phase 1 tabular endpoints are untouched.

The training pipeline (`src/audio/features.py`, `baseline_cv.py`, `phase2c.py`) and the serving
pipeline **share the same preprocessing + extraction code and the same `DEFAULT_CONFIG`**, so there
is no train/serve skew.

## 2. Data provenance and license

| | |
| --- | --- |
| Corpus | **MDVR-KCL** — Mobile Device Voice Recordings at King's College London |
| Recordings | **73** (31 Parkinson's, 42 healthy control) |
| Subjects | **37** (16 Parkinson's, 21 healthy control) |
| Native format | 44.1 kHz, 24-bit PCM, mono WAV |
| Tasks | Read text + spontaneous dialogue |
| License | **CC-BY-4.0** |

The raw audio and the ~606 MB source archive are **never committed** (enforced by `.gitignore` and
`.gitattributes`). What *is* committed is the non-audio provenance under
[`data_audio/mdvr_kcl/`](../data_audio/mdvr_kcl/): the license text (`LICENSE.txt`), a provenance
note (`PROVENANCE.md`), and a **SHA-256 manifest** (`manifest.sha256.json`) that pins the exact
bytes of every source file so a reviewer can confirm they obtained the identical corpus.

Label derivation and the known dataset quirks (e.g. the `ID22` filename anomaly, folder/filename
label agreement) are handled in `src/audio/discovery.py` and covered by tests. Labels are derived
from the cohort folder and cross-checked against the filename; inconsistent recordings are flagged,
never silently trusted.

## 3. Preprocessing

Every transform (`src/audio/preprocess.py`) is a pure, deterministic function of the input signal
and the immutable `AudioConfig` (`src/audio/config.py`), so the same WAV always yields the same
processed signal. Because extraction is deterministic and unsupervised it cannot leak label
information, and it is identical between training and inference.

Defaults (each an explicit, documented choice — not an accident):

- **`target_sample_rate = 44100`** — the corpus is natively 44.1 kHz. Non-conforming uploads are
  polyphase-resampled to 44.1 kHz (`resample_poly`); native files are a bit-exact no-op.
- **`mono = True`** — multi-channel input is averaged to mono.
- **`normalization = "none"`** — reduced loudness (hypophonia) is itself a Parkinsonian sign, so
  amplitude is not neutralised by default (`"peak"` is implemented and tested as an alternative).
- **`silence = "none"`** — pause/hesitation structure is diagnostically informative, so no silence
  is trimmed by default (`"trim_edges"` is implemented and tested).
- **Duration gate `0.5 s … 600 s`** — recordings outside the band are excluded *with a recorded
  reason*, never silently dropped. At inference this maps to HTTP `422`.

## 4. eGeMAPSv02 feature extraction

The extended Geneva Minimalistic Acoustic Parameter Set v02 (**eGeMAPSv02**), *Functionals* level,
via **openSMILE 2.6.0** (`opensmile==2.6.0`, pinned in `requirements-audio.txt`). It yields a fixed
**88-dimensional** vector of named descriptors (F0, jitter, shimmer, HNR, loudness, spectral/MFCC
statistics, voiced-segment timing, …).

Named features are preserved verbatim (not renamed to `f0…f87`) so the linear model stays
explainable. **Feature ordering is immutable and validated, never assumed**: at first use the
serving layer asserts that the installed openSMILE emits exactly the frozen ordering
(`src/audio/production.py::_smile`) and refuses to score on any mismatch, so a library upgrade can
never silently misalign the columns.

## 5. Model selection rationale

The production model is **eGeMAPSv02 + calibrated logistic regression**:

```python
CalibratedClassifierCV(
    LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000, random_state=42),
    method="sigmoid", cv=5,
)
# fit on StandardScaler-standardised 88 eGeMAPSv02 functionals
```

- **Standardisation** with `StandardScaler`, fit strictly on training folds.
- **Probability calibration** via 5-fold Platt scaling (`method="sigmoid"`) so the reported
  probability is meaningful, not just a ranking score.
- **Operating threshold = 0.39**, the *median of the per-fold max-F1 thresholds* from the seed-42
  subject-grouped split (not tuned on the test data).

Why a linear model over the alternatives on this corpus:

1. **It is the strongest defensible option under subject-grouped evaluation.** On 37 subjects, a
   well-regularised linear model over compact, interpretable acoustic descriptors is hard to beat;
   both a compact CNN (§8) and the case for large pretrained embeddings (§9) failed to justify
   themselves against it.
2. **It is honest about uncertainty and calibrated.** Platt scaling + a threshold chosen inside
   training folds avoids the classic small-data trap of a favorable single split.
3. **It is explainable by construction** (§10), which matters for a research tool that must not be
   mistaken for a black-box diagnostic.

Selection followed strict rules: never select on a favorable single split; select on the strongest
subject-grouped evidence; do not deploy a more complex model merely because it exists.

## 6. Subject-grouped evaluation methodology

The overriding risk on a 37-subject corpus is **subject leakage** — the same person's recordings
appearing in both train and test, which inflates every metric. All evaluation therefore uses
**`StratifiedGroupKFold` grouped by subject**, so no subject is ever in both the train and test side
of a fold. Within each training fold, the **scaler, the Platt calibration, and the decision
threshold are all fit strictly on training subjects only** — leakage-safe by construction, with an
explicit leakage canary in the tests (`tests/test_leakage.py`, `tests/test_audio_pipeline.py`).

Two complementary uncertainty lenses are reported:

- **50× repeated 5-fold subject-grouped CV** — the honest *headline*: the full distribution of
  performance across many different subject partitions.
- **Single-split subject-cluster bootstrap CI** — a historical point estimate with a bootstrap
  confidence interval, kept for traceability but explicitly *not* the headline.

## 7. Final metrics and uncertainty

**Headline — 50 × 5-fold subject-grouped repeated CV** (`artifacts_audio/phase2c/repeated_cv.json`):

| Metric | Mean | Std | 2.5% | Median | 97.5% |
| --- | --- | --- | --- | --- | --- |
| **ROC-AUC** | **0.761** | 0.118 | 0.457 | 0.798 | 0.906 |
| Balanced accuracy | 0.680 | 0.097 | 0.466 | 0.708 | 0.809 |
| Sensitivity (recall, PD) | 0.779 | 0.141 | 0.500 | 0.813 | 1.000 |
| Specificity (HC) | 0.581 | 0.147 | 0.296 | 0.571 | 0.894 |
| Brier score (↓ better) | 0.194 | 0.042 | 0.143 | 0.180 | 0.285 |

**The honest headline is ROC-AUC ≈ 0.761, with a wide 95% band of roughly 0.46–0.91.** That band is
the point: on 37 subjects, performance depends heavily on which subjects land in the test fold.

**Historical single split** (seed 42, subject level;
`artifacts_audio/phase2c/frozen_baseline_reference.json`): ROC-AUC **0.884** (bootstrap 95% CI
0.750–0.988). This estimate lies *within* the variability of the repeated subject-grouped splits but
toward the **favorable end** of that distribution — it is **not** the expected performance and is
never quoted as the headline.

## 8. Why the compact CNN was rejected

A small log-mel CNN was built and evaluated head-to-head (Phase 2D,
`artifacts_audio/cnn/cnn_summary.json`):

| | Compact CNN | eGeMAPS + LR |
| --- | --- | --- |
| Parameters | 6,033 | 89 (88 coef + intercept) |
| Repeated subject-grouped ROC-AUC (mean) | **0.677** ± 0.141 | **0.761** ± 0.118 |
| Single-split subject ROC-AUC | 0.795 | 0.884 |
| Paired folds won vs LR | **0/5** | — |
| Mean Δ(CNN − LR) | **−0.084** | — |

**Verdict (frozen):** *"The compact CNN does NOT show a consistent improvement over the frozen
eGeMAPS+LR baseline on roc_auc (mean delta −0.084, 0/5 paired folds won). The baseline remains the
reference; any single-split advantage is within the variability of repeated subject-grouped splits
and is not treated as a real gain."* The CNN is retained only as a documented negative result; it is
**not deployed**.

## 9. Why pretrained embeddings were not deployed

Large self-supervised speech transformers (wav2vec2 / WavLM, ~95M–300M parameters) were considered
as an offline research benchmark and **deliberately not run / not deployed**:

- They require heavy model downloads and CPU inference unsuitable for the lightweight openSMILE
  deployment path, and would fragment the feature pipeline.
- They are at **high overfitting risk on a 73-recording / 37-subject corpus** — where even a
  6,033-parameter CNN failed to beat the linear baseline under repeated subject-grouped CV.
- Per the model-decision rules, an embedding model that cannot show a *clear, defensible* subject-
  grouped improvement over eGeMAPS+LR must be abandoned and the reason documented. The expected
  value did not justify the compute, so it was skipped rather than deployed on hype.

## 10. API

New audio endpoints (Phase 1 tabular endpoints in [`README.md`](../README.md#api) are unchanged and
still serve 753-feature predictions):

| Endpoint | Purpose |
| --- | --- |
| `GET /api/audio/info` | Model card summary: tag, feature extractor, operating threshold, headline metric, upload limit, disclaimer |
| `POST /api/audio/predict` | Score one uploaded `.wav`; returns the structured payload below |

**Request:** `multipart/form-data` with a single `file` field containing a WAV.

```bash
curl -X POST http://localhost:8000/api/audio/predict -F "file=@recording.wav"
```

**Response (abridged):**

```json
{
  "prediction": 0,
  "label": "Healthy control indicated",
  "probability": 0.31,
  "threshold": 0.39,
  "model_id": "egemaps-lr",
  "model_version": "1.0.0",
  "model_tag": "egemaps-lr@1.0.0",
  "feature_set": "eGeMAPSv02",
  "n_features": 88,
  "input_domain": { "in_domain": true, "max_abs_z": 2.7, "note": "…" },
  "explanation": {
    "method": "mean standardised LR coefficients x standardised feature (log-odds)",
    "top_contributions": [
      { "feature": "…", "family": "…", "z": 1.9, "contribution": 0.42, "direction": "PD" }
    ]
  },
  "audio": { "filename": "recording.wav", "duration_s": 12.3, "original_sample_rate": 44100, "resampled": false },
  "disclaimer": "Research/ML prediction only - NOT a medical diagnosis. …"
}
```

**Explainability is honest by construction.** A contribution is `coef × z` — the additive log-odds
effect of each standardised feature on the model's internal linear score, using the *mean* of the
five calibration folds' coefficients. The calibration layer maps that score *monotonically* to the
probability, so contribution signs and relative magnitudes are faithful, but they do not sum exactly
to `logit(probability)`. This describes **what the linear model weighs — it is not a claim about
Parkinson's pathophysiology.**

**Out-of-domain reliability flag.** Inference **never clips** features (that would break train/serve
parity). Instead, if any standardised feature exceeds `|z| > 8`, `input_domain.in_domain` is set to
`false` with a plain-language warning. Non-speech audio (silence, pure tones, corrupt files)
produces extreme z-scores and a saturated, **untrustworthy** probability — the flag surfaces this
rather than hiding it.

**Error handling** (safe messages only; no internal paths): empty upload `400`, oversize `413`,
non-WAV extension / container `415`, unreadable/corrupt audio `400`, out-of-band duration `422`,
feature-extraction failure `500`, model bundle unavailable `503`.

## 11. Frontend

`app/static` gains an **"Audio (native)"** tab alongside the existing Predict / Performance /
Explainability / About tabs (all preserved). The tab shows a prominent research-not-diagnosis
banner, a drag-and-drop `.wav` dropzone with live file status, and — after analysis — the
probability, PD/HC badge, decision threshold, model name/version, the top acoustic-family
contributions, audio metadata, and the out-of-domain warning when it fires. Invalid files and API
failures render an inline error message rather than failing silently.

`app/static/config.js` sets `window.API_BASE`: empty means same-origin (single-service). If the
frontend is hosted separately, set it to the backend origin; `app.js` also rewrites report `<img>`
sources to that origin.

## 12. Limitations

- **Very small corpus** (37 subjects); metrics carry wide uncertainty bands (ROC-AUC 95% band
  ≈ 0.46–0.91). Treat any single number with caution.
- **Single-site, mobile-device recordings**; not validated across devices, microphones, languages,
  accents, or clinical populations.
- The model predicts a **statistical association with a diagnostic label**, not the presence of
  disease, and specificity in particular is modest (mean ≈ 0.58).
- **Out-of-domain audio** (non-speech, silence, tones, corruption) yields unreliable probabilities;
  these are flagged but the underlying score should not be interpreted.
- No causal or biomarker claim is made; feature contributions explain the model, not the disease.

## 13. Local setup

```bash
# Phase 1 (tabular) stack, then the audio extras, holding Phase 1 pins constant:
pip install -r requirements.txt
pip install -r requirements-audio.txt -c requirements.txt

# Run the service (serves the API and the frontend):
uvicorn app.main:app --reload
# open http://localhost:8000  →  "Audio (native)" tab
```

The frozen bundle and artifacts are committed under `artifacts_audio/`, so the audio endpoint runs
without the raw corpus and without retraining. Regenerate the model card with
`python -c "from src.audio.production import write_model_card; write_model_card()"`.

## 14. Deployment

The system supports two topologies; both use the committed artifacts and require no secrets.

**A. Single service (simplest) — Render.** `render.yaml` builds with
`pip install -r requirements.txt && pip install -r requirements-audio.txt -c requirements.txt`,
starts `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, and health-checks `/api/health`. The
FastAPI service serves the frontend itself, so `config.js` stays `API_BASE = ""` and no CORS
configuration is needed.

**B. Split — Vercel frontend + Render backend.** Deploy the Python API on Render as above. Deploy
the static frontend on Vercel (`vercel.json` serves `app/static/`, or set the project's Root
Directory to `app/static`). Then:

1. In `app/static/config.js`, set `window.API_BASE = "https://<your-api>.onrender.com"`.
2. On the Render service, set `ALLOWED_ORIGINS = "https://<your-project>.vercel.app"` so the API's
   CORS allow-list admits the Vercel origin.

**Environment variables** (all optional; safe defaults; **no secrets**):

| Var | Default | Purpose |
| --- | --- | --- |
| `PYTHON_VERSION` | `3.11.9` | Render runtime |
| `OMP_NUM_THREADS` | `1` | Deterministic, low-memory CPU inference |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS allow-list (set to the frontend origin in split mode) |
| `AUDIO_MAX_UPLOAD_MB` | `25` | Upload size guard for `/api/audio/predict` |

The audio path was verified end-to-end against a real running server (real WAV over HTTP → `200`
with the full payload; env-driven CORS confirmed; `.mp3` rejected `415`).

## 15. Reproducibility and versioning

- **Versioned identity:** the model is tagged `egemaps-lr@1.0.0`; the same tag is returned by the
  API and recorded in the model card.
- **Provenance pinned in the frozen reference** (`frozen_baseline_reference.json`): Phase 1 git
  commit `d88dd1f…`, feature-dataset SHA-256 `044946b5…`, corpus manifest SHA-256 `7c9ea7aa…`.
- **The model card is assembled live from the frozen artifacts** (`build_model_card()`), so it
  cannot drift from the evidence; a test asserts the headline number equals the value in
  `repeated_cv.json` (it is *read*, never re-typed).
- **Determinism:** preprocessing and extraction are pure functions; `random_state=42` fixes the
  classifier; a test asserts identical bytes yield an identical probability.
- **Pinned dependencies** in `requirements-audio.txt`; the `-c requirements.txt` install guarantees
  the Phase 1 scientific pins are never moved.

## 16. Disclaimer

**This is a research / ML prediction system, NOT a clinical diagnostic device.** It has not been
evaluated by any regulatory body, is trained on a very small single-site corpus, and predicts a
statistical association with a label rather than the presence of disease. Its outputs, probabilities,
and feature explanations must **not** be used for medical decisions or presented as a diagnosis.
