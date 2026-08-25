# `artifacts_audio/` — Phase 2 raw-audio pipeline outputs

This namespace holds artifacts produced by the **Phase 2 raw-audio pipeline**
(features, calibrated models and related outputs). It is kept separate from
`artifacts/`, which contains the **deployed Phase 1** 753-feature tabular models
and must not be modified by Phase 2.

## Status

Empty in Phase 2A (data acquisition only). Feature extraction (eGeMAPSv02) and
model artifacts arrive in Phase 2B; this directory and its namespace are
established now so the two pipelines never share output paths.

## Separation from Phase 1

- `artifacts/` — Phase 1 tabular models (logistic regression, random forest,
  RBF-SVM, MLP, attention network), scaler and SHAP background. **Unchanged.**
- `artifacts_audio/` — Phase 2 audio-pipeline outputs only.
