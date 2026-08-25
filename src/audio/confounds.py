"""Confound / dataset-artefact probes and the *available* fairness analysis.

Two Phase 2C objectives are addressed here:

* **Objective 9 — confounds.** Does the frozen baseline appear to lean on
  recording artefacts rather than voice content? Probes: recording-format
  uniformity, recordings-per-subject balance, whether utterance *duration* alone
  separates the classes (a duration-only grouped-CV model) and how strongly
  duration correlates with the model's predictions, whether the recording *task*
  is confounded with the label, and how much predictive weight sits in
  recording-condition-susceptible feature families.

* **Objective 10 — fairness on available metadata only.** The extracted corpus
  carries, per recording: subject id, PD/HC label, task
  (ReadText | SpontaneousDialogue), duration, original sample rate and channel
  count. It ships **no age, sex, gender, ethnicity, device or severity**
  metadata, so a demographic fairness analysis would require inventing data and
  is deliberately **not** attempted — its absence is reported instead. The
  available subgroup-parity analysis is by task (not a protected attribute; the
  closest stratum the data supports).

Subject grouping is preserved everywhere: the duration-only model uses the same
leakage-checked subject-grouped CV, and all correlations/tests operate on the
out-of-fold predictions of the reported baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu, pointbiserialr, spearmanr

from src.audio.baseline_cv import _augmented_metrics
from src.audio.families import feature_family
from src.audio.robustness import reference_oof
from src.data import Dataset, subject_labels
from src.stats import rank_metrics

# Demographic / device metadata that a fairness audit would need but the corpus
# does not provide. Reported as unavailable rather than fabricated.
UNAVAILABLE_METADATA = ("age", "sex", "gender", "ethnicity", "recording_device", "disease_severity")


def metadata_availability(frame: pd.DataFrame) -> dict:
    """Enumerate which per-recording metadata exist, and which do not."""
    available = {
        "subject_id": "subject_id" in frame.columns,
        "label_pd_hc": "label" in frame.columns,
        "task": "task" in frame.columns,
        "duration_s": "duration_s" in frame.columns,
        "orig_sample_rate": "orig_sample_rate" in frame.columns,
        "orig_channels": "orig_channels" in frame.columns,
    }
    demographic = {field: (field in frame.columns) for field in UNAVAILABLE_METADATA}
    return {
        "available": available,
        "demographic_available": demographic,
        "note": (
            "No age/sex/gender/ethnicity/device/severity metadata is present in "
            "the extracted dataset; a demographic fairness analysis is not "
            "attempted (no fabrication). Task is the only available non-target "
            "stratum and is not a protected attribute."
        ),
    }


def format_uniformity(frame: pd.DataFrame) -> dict:
    """Distribution of original sample rate and channel count across recordings."""
    sr = frame["orig_sample_rate"].value_counts().to_dict()
    ch = frame["orig_channels"].value_counts().to_dict()
    return {
        "orig_sample_rate_counts": {int(k): int(v) for k, v in sr.items()},
        "orig_channels_counts": {int(k): int(v) for k, v in ch.items()},
        "sample_rate_uniform": len(sr) == 1,
        "channels_uniform": len(ch) == 1,
        "note": (
            "Uniform format across all recordings means the classifier cannot "
            "exploit sample-rate or channel-count as a class proxy."
        ),
    }


def recordings_per_subject(frame: pd.DataFrame) -> dict:
    """Recordings-per-subject counts, and whether they are balanced across classes."""
    per = frame.groupby("group_key").agg(
        label=("label", "first"), n=("filename", "count")
    )
    pd_counts = per.loc[per["label"] == 1, "n"]
    hc_counts = per.loc[per["label"] == 0, "n"]
    return {
        "distribution": {int(k): int(v) for k, v in per["n"].value_counts().items()},
        "pd_mean": float(pd_counts.mean()) if len(pd_counts) else float("nan"),
        "hc_mean": float(hc_counts.mean()) if len(hc_counts) else float("nan"),
        "balanced": bool(abs(float(pd_counts.mean()) - float(hc_counts.mean())) < 0.5)
        if len(pd_counts) and len(hc_counts)
        else False,
    }


def duration_by_class(frame: pd.DataFrame) -> dict:
    """Compare utterance duration between PD and HC (a classic length confound)."""
    pd_dur = frame.loc[frame["label"] == 1, "duration_s"].to_numpy(dtype=float)
    hc_dur = frame.loc[frame["label"] == 0, "duration_s"].to_numpy(dtype=float)

    def _summary(a: np.ndarray) -> dict:
        return {
            "n": int(a.size),
            "mean": float(np.mean(a)) if a.size else float("nan"),
            "median": float(np.median(a)) if a.size else float("nan"),
            "std": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
            "min": float(np.min(a)) if a.size else float("nan"),
            "max": float(np.max(a)) if a.size else float("nan"),
        }

    labels = frame["label"].to_numpy(dtype=float)
    durations = frame["duration_s"].to_numpy(dtype=float)
    r_pb, p_pb = pointbiserialr(labels, durations)
    mwu_p = float("nan")
    if pd_dur.size and hc_dur.size:
        mwu_p = float(mannwhitneyu(pd_dur, hc_dur, alternative="two-sided").pvalue)
    return {
        "pd": _summary(pd_dur),
        "hc": _summary(hc_dur),
        "point_biserial_r": float(r_pb),
        "point_biserial_p": float(p_pb),
        "mannwhitney_p": mwu_p,
    }


def _duration_dataset(frame: pd.DataFrame) -> Dataset:
    return Dataset(
        X=frame[["duration_s"]].astype("float64").reset_index(drop=True),
        y=frame["label"].to_numpy(dtype=np.int64),
        groups=frame["group_key"].to_numpy(),
        feature_groups={"duration": ["duration_s"]},
    )


def duration_only_separability(frame: pd.DataFrame, *, n_splits: int = 5, seed: int = 42) -> dict:
    """ROC/PR-AUC of a subject-grouped CV model that sees *only* utterance duration.

    If duration alone already separates the classes well, the main model's
    performance may be partly a length artefact rather than voice content.
    """
    data = _duration_dataset(frame)
    prob, _pred, _thr = reference_oof(data, n_splits=n_splits, seed=seed)
    subjects, subj_labels = subject_labels(data)
    p_subj = np.array([prob[data.groups == s].mean() for s in subjects])
    rec = rank_metrics(data.y, prob)
    subj = rank_metrics(subj_labels, p_subj)
    return {
        "recording_roc_auc": rec["roc_auc"],
        "recording_pr_auc": rec["pr_auc"],
        "subject_roc_auc": subj["roc_auc"],
        "subject_pr_auc": subj["pr_auc"],
        "note": "Grouped-CV LR on duration_s only; high AUC would indicate a length confound.",
    }


def duration_prediction_correlation(frame: pd.DataFrame, prob: np.ndarray) -> dict:
    """Spearman correlation between utterance duration and the model's OOF probability."""
    durations = frame["duration_s"].to_numpy(dtype=float)
    rho, p = spearmanr(durations, np.asarray(prob, dtype=float))
    return {"spearman_rho": float(rho), "p_value": float(p)}


def top_feature_duration_correlation(
    frame: pd.DataFrame, coef_ranked: list[dict], *, top_k: int = 10
) -> dict:
    """Do the most influential features merely track duration?

    For each of the top-``top_k`` coefficient features, report |Spearman| against
    duration. Strong correlations would mean the model's "voice" signal is partly
    duration in disguise.
    """
    durations = frame["duration_s"].to_numpy(dtype=float)
    rows = []
    for entry in coef_ranked[:top_k]:
        feat = entry["feature"]
        if feat not in frame.columns:
            continue
        rho, p = spearmanr(frame[feat].to_numpy(dtype=float), durations)
        rows.append(
            {
                "feature": feat,
                "family": feature_family(feat),
                "spearman_rho_vs_duration": float(rho),
                "abs_rho": float(abs(rho)),
                "p_value": float(p),
            }
        )
    max_abs = max((r["abs_rho"] for r in rows), default=float("nan"))
    return {"per_feature": rows, "max_abs_rho": float(max_abs)}


def task_distribution(frame: pd.DataFrame) -> dict:
    """Recording counts per task and whether task is associated with the label."""
    per_task = {}
    for task, sub in frame.groupby("task"):
        n_pd = int((sub["label"] == 1).sum())
        n_hc = int((sub["label"] == 0).sum())
        per_task[str(task)] = {
            "n_recordings": len(sub),
            "n_pd": n_pd,
            "n_hc": n_hc,
            "pd_fraction": float(n_pd / len(sub)) if len(sub) else float("nan"),
            "n_subjects": int(sub["group_key"].nunique()),
        }
    # Chi-square: is task associated with PD/HC? (both cohorts should span tasks)
    ct = pd.crosstab(frame["task"], frame["label"])
    chi_p = float("nan")
    if ct.shape[0] > 1 and ct.shape[1] > 1:
        chi_p = float(chi2_contingency(ct.to_numpy())[1])
    return {"per_task": per_task, "task_label_chi2_p": chi_p}


def subgroup_performance_by_task(
    frame: pd.DataFrame, prob: np.ndarray, pred: np.ndarray
) -> dict:
    """Recording-level metrics of the frozen baseline within each task subgroup.

    Framed as fairness-style parity across the only available stratum. The
    balanced-accuracy / sensitivity / specificity gap between tasks is reported.
    """
    y = frame["label"].to_numpy(dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = np.asarray(pred, dtype=int)
    per_task = {}
    for task, sub in frame.groupby("task"):
        idx = sub.index.to_numpy()
        per_task[str(task)] = {
            "n_recordings": len(idx),
            **{k: float(v) for k, v in _augmented_metrics(y[idx], prob[idx], pred[idx]).items()},
        }
    gaps = {}
    tasks = list(per_task)
    if len(tasks) == 2:
        a, b = tasks
        for metric in ("balanced_accuracy", "sensitivity", "specificity", "roc_auc"):
            va, vb = per_task[a].get(metric), per_task[b].get(metric)
            if va is not None and vb is not None and not (np.isnan(va) or np.isnan(vb)):
                gaps[metric] = float(abs(va - vb))
    return {"per_task": per_task, "max_gap": gaps, "note": "Task is not a protected attribute."}


def analyze(
    frame: pd.DataFrame,
    data: Dataset,
    coef_ranked: list[dict],
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """Run every confound/fairness probe for the frozen baseline on ``frame``.

    ``data`` is the full 88-feature :class:`Dataset`; ``coef_ranked`` is the
    coefficient ranking from :mod:`src.audio.explain`. Uses the reported model's
    out-of-fold predictions throughout.
    """
    prob, pred, _thr = reference_oof(data, n_splits=n_splits, seed=seed)
    duration_only = duration_only_separability(frame, n_splits=n_splits, seed=seed)
    dur_pred_corr = duration_prediction_correlation(frame, prob)

    duration_flag = (
        (not np.isnan(duration_only["subject_roc_auc"]))
        and duration_only["subject_roc_auc"] >= 0.70
        and abs(dur_pred_corr["spearman_rho"]) >= 0.40
    )
    return {
        "metadata_availability": metadata_availability(frame),
        "format_uniformity": format_uniformity(frame),
        "recordings_per_subject": recordings_per_subject(frame),
        "duration_by_class": duration_by_class(frame),
        "duration_only_separability": duration_only,
        "duration_prediction_correlation": dur_pred_corr,
        "top_feature_duration_correlation": top_feature_duration_correlation(frame, coef_ranked),
        "task_distribution": task_distribution(frame),
        "fairness_by_task": subgroup_performance_by_task(frame, prob, pred),
        "flags": {
            "duration_confound_suspected": bool(duration_flag),
            "sample_rate_uniform": format_uniformity(frame)["sample_rate_uniform"],
            "channels_uniform": format_uniformity(frame)["channels_uniform"],
        },
    }
