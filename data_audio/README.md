# `data_audio/` — raw-audio datasets (Phase 2)

This namespace holds the **raw-audio** datasets for the Phase 2 pipeline. It is
deliberately separate from `data/`, which contains the Phase 1 tabular UCI
dataset (`pd_speech_features.csv`) and is untouched by Phase 2.

## Contents

```
data_audio/
└── mdvr_kcl/
    ├── PROVENANCE.md          # origin, DOI, licence, acquisition method  (committed)
    ├── LICENSE.txt            # CC-BY-4.0 terms + required attribution     (committed)
    ├── manifest.sha256.json   # SHA-256 provenance manifest                (committed)
    ├── 26_29_09_2017_KCL.zip  # downloaded archive          (git-ignored, NEVER committed)
    └── raw/                   # extracted *.wav recordings  (git-ignored, NEVER committed)
```

## Rules

- **Raw audio and archives are never committed to Git.** The `~606 MB`
  `26_29_09_2017_KCL.zip` and all `*.wav` files are excluded by `.gitignore`.
- **Only metadata is committed:** provenance, licence and the SHA-256 manifest.
- Datasets are acquired by an **explicit command**, never as a side effect of
  importing code.

## Acquiring MDVR-KCL

```bash
python -m src.audio.fetch_mdvr_kcl --data-dir data_audio/mdvr_kcl
```

The command downloads the archive, verifies it against the MD5 published on the
Zenodo record, extracts the audio into `raw/`, and (re)writes
`manifest.sha256.json`. See [`mdvr_kcl/PROVENANCE.md`](mdvr_kcl/PROVENANCE.md).
