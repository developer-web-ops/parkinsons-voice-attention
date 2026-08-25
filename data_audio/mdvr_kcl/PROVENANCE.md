# MDVR-KCL — Provenance

This file documents the origin, licensing and acquisition of the **MDVR-KCL**
audio corpus used by the Phase 2 raw-audio pipeline. It records only what the
authoritative Zenodo record supports.

## Dataset

| Field | Value |
| --- | --- |
| Title | Mobile Device Voice Recordings at King's College London (MDVR-KCL) from both early and advanced Parkinson's disease patients and healthy controls |
| Authors / creators | Hagen Jaeger; Dhaval Trivedi; Michael Stadtschnitzer |
| Version | v1 |
| Version DOI | [10.5281/zenodo.2867216](https://doi.org/10.5281/zenodo.2867216) |
| Concept DOI (all versions) | [10.5281/zenodo.2867215](https://doi.org/10.5281/zenodo.2867215) |
| Zenodo record | https://zenodo.org/records/2867216 |
| Publication date | 2019-05-17 |
| Access | open |
| License | **CC-BY-4.0** — https://creativecommons.org/licenses/by/4.0/ |
| Audio format (source-reported) | WAV, 44.1 kHz, 16-bit — see [Audio format](#audio-format) (measured bit depth differs) |

Subject IDs and Parkinson's disease (PD) vs. healthy-control (HC) labels are
documented by the dataset itself.

## Audio format

Two clearly-distinguished records are kept, because they disagree on bit depth.
Both appear in [`manifest.sha256.json`](manifest.sha256.json) under `audio_format`;
the source-reported value is **preserved, not overwritten**.

| | Sample rate | Bit depth | Channels | PCM code | Source |
| --- | --- | --- | --- | --- | --- |
| **Source-reported** | 44.1 kHz | **16-bit** | — | — | Zenodo record 2867216 metadata |
| **Empirically verified** | 44,100 Hz | **24-bit** | 1 (mono) | 1 (PCM) | WAV `fmt ` chunk headers of the extracted files |

- **Discrepancy:** the Zenodo record states 16-bit, but every one of the 73
  extracted WAV files inspected is **24-bit PCM**. The verification read the WAV
  `fmt ` chunk headers with the Python standard library (the `wave` module plus a
  raw byte parse of `audio_format` / `channels` / `sample_rate` / `bits_per_sample`);
  all 73 files agreed.
- The source-reported value is retained verbatim so the mismatch is auditable
  rather than silently corrected. Phase 2B feature extraction should treat the
  files as **44.1 kHz / 24-bit / mono PCM** (the measured format).

## Known anomalies

These are valid recordings; the note is a hand-off to Phase 2B, not a defect list.

- **`SpontaneousDialogue/HC/ID22hc_0_0_0.wav`** — the filename is missing the
  underscore between the subject ID and cohort (`ID22hc`), whereas every other
  file uses the `ID<NN>_<hc|pd>` convention. The audio itself is valid 24-bit PCM.
  Phase 2B **must** derive the subject ID with a tolerant matcher (e.g. the regex
  `ID(\d+)`), **not** `str.split("_")[0]`. A naive split would mint a phantom
  subject `ID22hc` and split ID22's recordings across folds, leaking the subject
  between train and test in grouped cross-validation.

## Archive

| Field | Value |
| --- | --- |
| Filename | `26_29_09_2017_KCL.zip` |
| Size | 606,144,431 bytes |
| MD5 (published on the record) | `98c51bdd2b092b93f8bb038dea4505fa` |

The MD5 above was verified by the project owner against the Zenodo record on
**2026-08-23**.

## Acquisition

Acquisition is an explicit, opt-in command — importing the code never downloads
anything:

```bash
python -m src.audio.fetch_mdvr_kcl --data-dir data_audio/mdvr_kcl
```

The command:

1. downloads `26_29_09_2017_KCL.zip` from the Zenodo record,
2. **refuses to proceed unless** its MD5 matches
   `98c51bdd2b092b93f8bb038dea4505fa`,
3. extracts the WAV files into `data_audio/mdvr_kcl/raw/`, and
4. writes `data_audio/mdvr_kcl/manifest.sha256.json` — a SHA-256 manifest of the
   archive and every extracted file, plus the acquisition timestamp.

The exact acquisition date, the archive SHA-256 and the per-file SHA-256 digests
are recorded in `manifest.sha256.json`. The committed manifest records a
**completed acquisition** (`"status": "acquired"`) of the verified release: the
archive SHA-256, the acquisition timestamp, and a SHA-256 for each of the 73
extracted WAV files. The audio bytes themselves remain un-committed (see below).

## What is and isn't committed

- **Committed:** this file, `LICENSE.txt`, and `manifest.sha256.json` (metadata
  only).
- **Never committed:** the `26_29_09_2017_KCL.zip` archive and all raw `*.wav`
  audio under `raw/`. These are excluded by `.gitignore`.

## Attribution (required by CC-BY-4.0)

> Jaeger, H., Trivedi, D., & Stadtschnitzer, M. (2019). *Mobile Device Voice
> Recordings at King's College London (MDVR-KCL) from both early and advanced
> Parkinson's disease patients and healthy controls* [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.2867216

See [`LICENSE.txt`](LICENSE.txt) for the licensing terms that apply to the
dataset.
