"""Temporal holdout infrastructure (M2 Part 9).

Concept
-------
    historical period  ->  features -> model training
    later unseen period ->  features -> evaluation

The point is to test whether a feature-to-trajectory mapping learned on one
period still holds on a later, unseen one. The cutoff is configurable
(`config.TemporalHoldoutConfig`).

No-lookahead guarantee
----------------------
The two windows are materialised as separate arrays BEFORE any estimator
runs (`split_series`), and features for each window are built from that
window only. No statistic - not the mean, not the trend, not the spectral
band power - can therefore see an observation from the other window. This is
enforced by tests, not only by convention.

LIMITATION ON THE CURRENT SYNTHETIC DATA (read before quoting any number)
-------------------------------------------------------------------------
The bundled synthetic dataset has one static label per pixel for the whole
record, and its archetypes are defined over the full 36 steps (for example a
declining pixel breaks somewhere in the middle, and a jhum pixel's period
shortens across the record). Splitting it produces two short windows in
which:

  * a window may contain no breakpoint at all, so disturbance and recovery
    features are structurally absent rather than informative;
  * a ~12-24 step window is at or below the minimum length the cyclicity
    and breakpoint estimators need, so those features degrade sharply;
  * the label still describes the WHOLE record, so a "declining" pixel may
    be genuinely flat inside the historical window.

The infrastructure is therefore complete and tested, but a temporal-holdout
score computed on synthetic data measures the synthetic generator, not
temporal generalization. Real multi-decadal data (M6/M7) are required before
any temporal-generalization claim can be made. Results carry
`meaningful_on_this_dataset: false` for exactly this reason.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, RFExperimentConfig, TemporalHoldoutConfig
from .features import build_feature_table, feature_names
from .validation import (classification_metrics, fit_random_forest,
                         spatial_block_folds)

__all__ = ["resolve_cutoff", "temporal_holdout_indices", "split_series",
           "run_temporal_holdout"]


def resolve_cutoff(n_time: int, cfg: TemporalHoldoutConfig | None = None
                   ) -> int:
    """Resolve the configured cutoff to a concrete index.

    `cutoff_index` wins when set (>= 0); otherwise the cutoff is
    round(cutoff_fraction * T). The result must leave at least
    `min_history` historical and `min_future` later observations.
    """
    cfg = cfg or TemporalHoldoutConfig()
    if n_time < cfg.min_history + cfg.min_future:
        raise ValueError(
            f"a temporal holdout needs at least "
            f"{cfg.min_history + cfg.min_future} time steps, got {n_time}")
    cutoff = int(cfg.cutoff_index) if cfg.cutoff_index is not None \
        and cfg.cutoff_index >= 0 else int(round(cfg.cutoff_fraction * n_time))
    if cutoff < cfg.min_history or n_time - cutoff < cfg.min_future:
        raise ValueError(
            f"cutoff {cutoff} leaves {cutoff} historical and "
            f"{n_time - cutoff} later steps, violating min_history="
            f"{cfg.min_history} / min_future={cfg.min_future}")
    return cutoff


def temporal_holdout_indices(n_time: int, cutoff: int):
    """Chronology-safe (historical, later) index arrays."""
    if not 1 <= cutoff < n_time:
        raise ValueError("cutoff must leave historical and later observations")
    return np.arange(cutoff), np.arange(cutoff, n_time)


def split_series(ndvi, rain, cutoff: int):
    """Materialise the historical and later windows as separate arrays.

    Returned arrays are copies: a later mutation of one window cannot reach
    the other, and no estimator can index past its own window.
    """
    ndvi = np.asarray(ndvi, dtype="float64")
    rain = np.asarray(rain, dtype="float64")
    historical, later = temporal_holdout_indices(ndvi.shape[0], cutoff)
    return ((ndvi[historical].copy(), rain[historical].copy()),
            (ndvi[later].copy(), rain[later].copy()))


def run_temporal_holdout(ndvi, rain, labels, output_dir,
                         cfg: Config | None = None, *,
                         sample_mask=None, grid_shape=None,
                         rf_cfg: RFExperimentConfig | None = None,
                         fold_grid=None, logger=None) -> dict:
    """Train on historical-window features, evaluate on later-window ones.

    When `cfg.research.holdout.spatially_separated` is set and a fold
    assignment is available, the evaluation samples are also a held-out
    spatial fold, so the test set differs from the training set in BOTH
    time and space. Otherwise the same pixels are used in both windows and
    only the period differs.

    Writes metrics, predictions, probabilities and the limitation note to
    `output_dir`, and returns the result dictionary.
    """
    cfg = cfg or Config()
    holdout_cfg = cfg.research.holdout
    rf_cfg = rf_cfg or cfg.research.model
    ndvi = np.asarray(ndvi, dtype="float64")
    rain = np.asarray(rain, dtype="float64")
    labels = np.asarray(labels)

    cutoff = resolve_cutoff(ndvi.shape[0], holdout_cfg)
    (hist_ndvi, hist_rain), (late_ndvi, late_rain) = split_series(
        ndvi, rain, cutoff)

    historical_features, _ = build_feature_table(hist_ndvi, hist_rain, cfg)
    later_features, _ = build_feature_table(late_ndvi, late_rain, cfg)
    columns = feature_names(cfg.research.features.groups)

    n = len(labels)
    mask = np.ones(n, bool) if sample_mask is None \
        else np.asarray(sample_mask, bool)
    train_mask, test_mask = mask.copy(), mask.copy()
    spatial_note = "same pixels in both windows; separation is temporal only"
    if holdout_cfg.spatially_separated:
        if fold_grid is None and grid_shape is not None:
            fold_grid = spatial_block_folds(
                *grid_shape, cfg.research.spatial_cv)[1]
        if fold_grid is not None:
            folds = np.asarray(fold_grid).reshape(-1)
            if folds.size != n:
                raise ValueError("fold assignment and labels must describe "
                                 "the same samples")
            test_fold = int(np.unique(folds[mask])[0])
            test_mask = mask & (folds == test_fold)
            train_mask = mask & (folds != test_fold)
            spatial_note = (f"evaluation restricted to spatial fold "
                            f"{test_fold}; training uses the other folds")

    if not train_mask.any() or not test_mask.any():
        raise ValueError("temporal holdout produced an empty train or test set")

    x_train = historical_features.loc[:, columns].to_numpy(dtype="float64")
    x_test = later_features.loc[:, columns].to_numpy(dtype="float64")
    imputer, model = fit_random_forest(x_train[train_mask],
                                       labels[train_mask], rf_cfg)
    x_eval = imputer.transform(x_test[test_mask])
    predictions = model.predict(x_eval)
    probabilities = model.predict_proba(x_eval)
    metrics = classification_metrics(labels[test_mask], predictions,
                                     labels=np.unique(labels[mask]))
    metrics.update({
        "validation": "temporal_holdout",
        "cutoff_index": int(cutoff),
        "n_historical_steps": int(cutoff),
        "n_later_steps": int(ndvi.shape[0] - cutoff),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "spatial_separation": spatial_note,
        "meaningful_on_this_dataset": False,
        "limitation": (
            "Synthetic pixels carry one static label for the whole record "
            "and the windows are short, so this score characterises the "
            "synthetic generator, not temporal generalization. Real "
            "multi-decadal data (M6/M7) are required before quoting a "
            "temporal-generalization result."),
    })

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "configuration.json").write_text(json.dumps({
        "holdout": asdict(holdout_cfg), "model": asdict(rf_cfg),
        "features": columns, "cutoff_index": int(cutoff)}, indent=2))
    pd.DataFrame({"truth": labels[test_mask], "prediction": predictions}
                 ).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(probabilities,
                 columns=[f"probability_{c}" for c in model.classes_]
                 ).to_csv(out / "probabilities.csv", index=False)
    (out / "LIMITATION.txt").write_text(metrics["limitation"] + "\n")
    if logger is not None:
        logger.info("temporal holdout cutoff=%d (%d historical / %d later "
                    "steps): accuracy %.4f, macro F1 %.4f - NOT a "
                    "temporal-generalization result on synthetic data",
                    cutoff, cutoff, ndvi.shape[0] - cutoff,
                    metrics["accuracy"], metrics["f1_macro"])
    return {"metrics": metrics, "predictions": predictions,
            "probabilities": probabilities, "cutoff": cutoff,
            "train_mask": train_mask, "test_mask": test_mask,
            "historical_features": historical_features,
            "later_features": later_features}
