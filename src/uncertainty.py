"""Prediction confidence and uncertainty reporting (M3 Part 5).

WHAT THESE NUMBERS ARE
----------------------
A classifier's `predict_proba` output is a score conditional on the training
data, the feature set and the model family. It is NOT a probability that a
piece of ground is degraded, and it is NOT certainty. Random Forest
probabilities are the fraction of trees voting for a class; softmax outputs
of a neural network are normalised logits. Both are typically miscalibrated,
and neither carries information about the things that actually dominate
real-world error: sensor artefacts, label error, and land-cover types absent
from the training set.

What they are useful for is RELATIVE ranking - flagging which predictions the
model itself found marginal, so those pixels can be excluded from a summary
statistic or sent for manual checking.

Three complementary measures are reported, because the maximum probability
alone hides the shape of the distribution:

confidence  max class probability. High for a confident single class.
margin      top probability minus runner-up. Low where two classes compete,
            even if the top probability looks high in a 5-class problem.
entropy     Shannon entropy normalised to [0, 1] by log(n_classes). High
            where the model spreads its vote over many classes.

A prediction is flagged `uncertain` when confidence falls below
`UncertaintyConfig.confidence_threshold` OR the margin falls below
`margin_threshold`. Both thresholds are reporting conventions, documented in
README.md, not decision rules validated against ground truth.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import UncertaintyConfig

__all__ = ["prediction_confidence", "uncertainty_table", "uncertainty_summary",
           "CONFIDENCE_DISCLAIMER"]

CONFIDENCE_DISCLAIMER = (
    "Model confidence estimates conditional on the training data, feature "
    "set and model family. They are relative scores, not calibrated "
    "probabilities and not certainty about the land surface.")


def _validate(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype="float64")
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("probabilities must be a (n_samples, n_classes) "
                         "array with at least two classes")
    rows = np.isfinite(p).all(axis=1)
    if not rows.any():
        raise ValueError("probabilities contain no finite rows")
    if (p[rows] < -1e-9).any():
        raise ValueError("probabilities must not be negative")
    if not np.allclose(p[rows].sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("probability rows must sum to 1")
    return p


def prediction_confidence(probabilities, *, cfg: UncertaintyConfig | None = None
                          ) -> dict:
    """Confidence, margin, normalised entropy and the uncertain flag.

    Rows that are entirely non-finite (a sample no fold predicted) yield NaN
    measures and are flagged uncertain, never silently treated as confident.
    """
    cfg = cfg or UncertaintyConfig()
    p = _validate(probabilities)
    finite = np.isfinite(p).all(axis=1)
    n_classes = p.shape[1]

    confidence = np.full(len(p), np.nan)
    margin = np.full(len(p), np.nan)
    entropy = np.full(len(p), np.nan)

    values = p[finite]
    ordered = np.sort(values, axis=1)[:, ::-1]
    confidence[finite] = ordered[:, 0]
    margin[finite] = ordered[:, 0] - ordered[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(values > 0, values * np.log(values), 0.0)
    entropy[finite] = -terms.sum(axis=1) / np.log(n_classes)

    uncertain = ~finite | (confidence < cfg.confidence_threshold) \
        | (margin < cfg.margin_threshold)
    return {"confidence": confidence, "margin": margin, "entropy": entropy,
            "uncertain": uncertain, "n_classes": int(n_classes),
            "disclaimer": CONFIDENCE_DISCLAIMER}


def uncertainty_table(predictions, probabilities, classes, *, truth=None,
                      cfg: UncertaintyConfig | None = None) -> pd.DataFrame:
    """Tidy per-sample table: prediction, per-class score, and uncertainty."""
    measures = prediction_confidence(probabilities, cfg=cfg)
    frame = pd.DataFrame({"prediction": np.asarray(predictions)})
    if truth is not None:
        frame.insert(0, "truth", np.asarray(truth))
        frame["correct"] = frame["truth"].to_numpy() == frame["prediction"].to_numpy()
    for i, name in enumerate(classes):
        frame[f"probability_{name}"] = np.asarray(probabilities)[:, i]
    for key in ("confidence", "margin", "entropy", "uncertain"):
        frame[key] = measures[key]
    return frame


def uncertainty_summary(probabilities, *, truth=None, predictions=None,
                        cfg: UncertaintyConfig | None = None) -> dict:
    """Aggregate uncertainty, and how accuracy splits by the flag.

    When labels are supplied, the summary reports accuracy on the confident
    subset and on the flagged subset. A useful confidence measure should show
    higher accuracy where it is confident; if it does not, the measure is not
    informative for this problem and the summary makes that visible instead
    of hiding it.
    """
    cfg = cfg or UncertaintyConfig()
    measures = prediction_confidence(probabilities, cfg=cfg)
    uncertain = measures["uncertain"]
    summary = {
        "n_samples": int(len(uncertain)),
        "n_uncertain": int(uncertain.sum()),
        "uncertain_fraction": float(uncertain.mean()),
        "confidence_threshold": float(cfg.confidence_threshold),
        "margin_threshold": float(cfg.margin_threshold),
        "mean_confidence": float(np.nanmean(measures["confidence"])),
        "median_confidence": float(np.nanmedian(measures["confidence"])),
        "mean_margin": float(np.nanmean(measures["margin"])),
        "mean_normalised_entropy": float(np.nanmean(measures["entropy"])),
        "disclaimer": CONFIDENCE_DISCLAIMER,
    }
    if truth is not None and predictions is not None:
        correct = np.asarray(truth) == np.asarray(predictions)
        confident = ~uncertain
        summary["accuracy_confident_subset"] = (
            float(correct[confident].mean()) if confident.any() else None)
        summary["accuracy_uncertain_subset"] = (
            float(correct[uncertain].mean()) if uncertain.any() else None)
        summary["confidence_is_informative"] = bool(
            confident.any() and uncertain.any()
            and correct[confident].mean() > correct[uncertain].mean())
    return summary
