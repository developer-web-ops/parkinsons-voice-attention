<!-- The YAML block below configures the Hugging Face Space (Docker SDK). It is
     ignored by GitHub except for rendering a small metadata table. -->
---
title: Parkinsons Voice Attention
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Parkinson's Disease Detection from Voice Biomarkers Using an Attention-Based Neural Network

End-to-end system that predicts Parkinson's disease from sustained-phonation voice biomarkers,
compares an MLP baseline against a group-attention network, explains predictions with attention
weights and SHAP, and serves everything through a FastAPI backend with a plain HTML/CSS/JS frontend.

| Layer | Stack |
| --- | --- |
| Data | [UCI Parkinson's Disease Classification](https://archive.ics.uci.edu/dataset/470/parkinson+s+disease+classification) (756 recordings, 252 subjects, 753 features) |
| Classical ML | scikit-learn (logistic regression, random forest, RBF-SVM) |
| Deep learning | PyTorch (MLP baseline + group-attention network) |
| Explainability | Attention weights + SHAP `GradientExplainer` |
| Backend | FastAPI + Uvicorn |
| Frontend | Static HTML / CSS / vanilla JS |
| Deployment | Render (`render.yaml`) |

> ### 🎙️ Native-audio model (Phase 2)
>
> Beyond the tabular UCI system below, this project also includes a **native-audio** model that
> predicts directly from an uploaded `.wav`: **openSMILE eGeMAPSv02** (88 acoustic features) →
> **calibrated logistic regression**, tagged **`egemaps-lr@1.0.0`**. Its honest headline is
> **ROC-AUC ≈ 0.761** (50× repeated 5-fold *subject-grouped* CV; 95% band ≈ 0.46–0.91) on the
> **MDVR-KCL** corpus (73 recordings, 37 subjects, CC-BY-4.0). A compact CNN and large pretrained
> speech embeddings were evaluated and **not** deployed (neither beat the linear baseline under
> subject-grouped evaluation). Try it in the **"Audio (native)"** tab, or via
> `POST /api/audio/predict`.
>
> **Full technical documentation: [`docs/AUDIO_MODEL.md`](docs/AUDIO_MODEL.md)** — architecture,
> provenance, preprocessing, feature extraction, model-selection rationale, subject-grouped
> methodology, metrics + uncertainty, the CNN/embedding decisions, API, deployment, reproducibility,
> and limitations.
>
> ⚠️ Research / educational ML only — **not a medical device and not a diagnostic tool.**

## Quick start

```bash
pip install -r requirements.txt
python -m src.train      # trains all models, writes artifacts/ and reports/
python -m src.explain    # SHAP + attention analysis
uvicorn app.main:app --reload
# open http://localhost:8000
```

Trained artifacts and reports are committed, so the API runs without retraining.

## Data and splitting

Each of the 252 subjects contributed three recordings of the vowel /a/. Splitting rows at random
therefore leaks a subject across train and test and inflates every metric. All splits here are
**grouped by subject** (`src/data.py:subject_split`), and metrics are reported both per recording
and per subject (averaging a subject's three predictions).

Standardisation is fitted on training subjects only. Class imbalance (564 PD vs 192 healthy) is
handled with `pos_weight` in the loss for the neural nets and `class_weight="balanced"` for the
scikit-learn models. Decision thresholds for the neural nets are tuned on the validation split.

## Models

- **MLP baseline** (`MLPBaseline`) — 753 → 256 → 64 → 1 with batch norm and dropout.
- **Attention network** (`GroupAttentionNet`) — the features are partitioned into the eight
  acoustic families the dataset defines (baseline jitter/shimmer, intensity, formants, bandwidths,
  vocal fold, MFCC, wavelet, TQWT). Each family gets its own encoder producing a 64-d embedding;
  an additive-attention head scores the embeddings and the softmax weights fuse them into one
  context vector that feeds the classifier. Those weights are the model's own explanation of which
  acoustic family drove the prediction, available per request via `forward_with_attention`.

Both are trained with AdamW and early stopping on validation ROC-AUC.

## Results (held-out subjects, recording level)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.739 | 0.830 | 0.816 | 0.823 | 0.804 |
| Random forest | 0.771 | 0.780 | 0.965 | 0.863 | 0.741 |
| RBF-SVM | 0.758 | 0.808 | 0.886 | 0.845 | 0.743 |
| MLP baseline | 0.804 | 0.813 | 0.956 | 0.879 | 0.807 |
| Attention network | 0.725 | 0.816 | 0.816 | 0.816 | 0.793 |

Full numbers, confusion matrices and subject-level aggregates: [`reports/metrics.json`](reports/metrics.json).
Plots: [`reports/roc_curves.png`](reports/roc_curves.png),
[`reports/confusion_matrix_attention.png`](reports/confusion_matrix_attention.png).

The attention network trades a little accuracy for interpretability; it stays within noise of the
MLP on ROC-AUC while exposing per-prediction attribution over acoustic families.

## Explainability

Mean attention on the test set concentrates on the TQWT (0.43) and MFCC (0.26) families, matching
the SHAP ranking, whose top attributions are MFCC delta/delta-delta standard deviations and TQWT
kurtosis coefficients. See [`reports/explainability.json`](reports/explainability.json),
[`reports/attention_weights.png`](reports/attention_weights.png) and
[`reports/shap_top_features.png`](reports/shap_top_features.png).

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Liveness + loaded models |
| `GET /api/features` | Feature groups, medians, min/max (drives the form) |
| `GET /api/examples` | One held-out recording per class for demos |
| `POST /api/predict` | Score one sample; returns probability, attention weights and SHAP values |
| `POST /api/predict/csv` | Score up to 200 rows of an uploaded CSV |
| `GET /api/metrics` | Evaluation report |
| `GET /api/explainability` | Global attention + SHAP summary |
| `GET /api/audio/info` | Native-audio model card (see [`docs/AUDIO_MODEL.md`](docs/AUDIO_MODEL.md)) |
| `POST /api/audio/predict` | Score an uploaded `.wav` with the `egemaps-lr@1.0.0` model |

```bash
curl -X POST localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"features": {"PPE": 0.85, "DFA": 0.72, "locPctJitter": 0.008}, "model": "attention"}'
```

Unspecified features fall back to the dataset median, so partial payloads are valid.

## Frontend

`app/static` serves a four-tab single page: **Predict** (editable biomarkers, one-click held-out
examples, CSV batch scoring, live attention and SHAP bars), **Model performance** (metrics table,
ROC curves, confusion matrix), **Explainability** (global attention and SHAP) and **About**.

## Deployment (Render)

`render.yaml` is a Render blueprint: point Render at this repository, choose *New → Blueprint*, and
it builds with `pip install -r requirements.txt && pip install -r requirements-audio.txt -c
requirements.txt` (the tabular stack plus the openSMILE audio extras, holding the Phase 1 pins
constant) and starts `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, health-checking
`/api/health`. All environment variables have safe defaults and **no secrets are required**; a
frontend can optionally be split onto Vercel (`vercel.json`) with `ALLOWED_ORIGINS` + `config.js`.
See [`docs/AUDIO_MODEL.md` §14](docs/AUDIO_MODEL.md#14-deployment) for both deployment topologies and
the full environment-variable table.

## Tests

```bash
python -m pytest      # 145 tests: data integrity, no subject leakage, model shapes, tabular + audio API, audio pipeline/production/CNN/provenance
ruff check .
```

The audio-specific suites (`tests/test_audio_*.py`) use tiny synthetic WAVs and never require the
MDVR-KCL corpus; they skip cleanly if the audio stack is not installed. They assert response
structure and determinism and, per policy, never assert a class label for synthetic audio.

## Disclaimer

Research demonstration only. This is not a medical device and must not be used for diagnosis.
