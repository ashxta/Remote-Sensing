"""Random Forest research pipeline and spatial block cross-validation
(M2 Parts 4 and 5).

Why spatial blocks
------------------
Neighbouring pixels of a raster are spatially autocorrelated: a random
pixel-level split puts near-duplicate samples on both sides of the split and
reports an optimistic accuracy that does not describe performance on new
ground. Spatial block cross-validation assigns whole square blocks of the
grid to folds, so every test pixel is separated from the training set by at
least a block boundary, and optionally by a configurable buffer of blocks.

Random pixel splitting is retained ONLY as a baseline for comparison
(`random_split_baseline`); it is never the primary validation.

Leakage policy
--------------
Anything that learns parameters from data - here the imputer used to fill
non-finite features - is fitted inside the training fold and only applied to
the test fold. Feature engineering itself is per-pixel and involves no
cross-pixel fitting, so it cannot leak between folds.

Probabilities
-------------
`predict_proba` outputs are model CONFIDENCE ESTIMATES conditional on the
training data and the feature set. They are not calibrated probabilities of
real-world land degradation and must never be reported as certainty.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

from .config import RFExperimentConfig, SpatialCVConfig

__all__ = ["SpatialCVConfig", "RFExperimentConfig", "spatial_block_folds",
           "block_coordinates", "buffered_training_mask",
           "classification_metrics", "aggregate_fold_metrics",
           "spatial_cv_rf", "random_split_baseline", "fit_random_forest"]


# --------------------------------------------------------------------- blocks
def block_coordinates(height: int, width: int, block_size: int):
    """Per-pixel (block_row, block_col, block_id) grids."""
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    rows, cols = np.indices((height, width))
    brow, bcol = rows // block_size, cols // block_size
    n_block_cols = int(np.ceil(width / block_size))
    return brow, bcol, brow * n_block_cols + bcol


def spatial_block_folds(height: int, width: int,
                        cfg: SpatialCVConfig | None = None):
    """Assign whole spatial blocks to deterministic folds.

    Returns (block_id_grid, fold_grid). A block is never split across folds,
    and the assignment depends only on (height, width, cfg), so it is
    reproducible across runs and machines.
    """
    cfg = cfg or SpatialCVConfig()
    if cfg.block_size < 1 or cfg.n_folds < 2:
        raise ValueError("block_size must be >=1 and n_folds must be >=2")
    _, _, block_id = block_coordinates(height, width, cfg.block_size)
    unique = np.unique(block_id)
    if unique.size < cfg.n_folds:
        raise ValueError(f"{unique.size} blocks cannot fill {cfg.n_folds} "
                         "folds; reduce block_size or n_folds")
    ordered = np.random.default_rng(cfg.seed).permutation(unique)
    mapping = {int(b): i % cfg.n_folds for i, b in enumerate(ordered)}
    folds = np.vectorize(mapping.get)(block_id).astype("int16")
    return block_id, folds


def buffered_training_mask(train: np.ndarray, test: np.ndarray,
                           block_row: np.ndarray, block_col: np.ndarray,
                           buffer_blocks: int) -> np.ndarray:
    """Drop training samples within `buffer_blocks` of any test block.

    Distance is Chebyshev distance in block units, so `buffer_blocks=1`
    removes the ring of blocks touching a test block. With `buffer_blocks=0`
    the training mask is returned unchanged (plain block CV, in which train
    and test still never share a pixel or a block).
    """
    if buffer_blocks <= 0:
        return train
    brow = np.asarray(block_row).reshape(-1)
    bcol = np.asarray(block_col).reshape(-1)
    test_blocks = np.unique(np.c_[brow[test], bcol[test]], axis=0)
    keep = np.ones(train.shape, dtype=bool)
    for r, c in test_blocks:
        keep &= ~((np.abs(brow - r) <= buffer_blocks)
                  & (np.abs(bcol - c) <= buffer_blocks))
    return train & keep


# -------------------------------------------------------------------- metrics
def classification_metrics(y_true, y_pred, labels=None) -> dict:
    """Full metric set. Accuracy alone is never reported on its own."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.asarray(sorted(np.unique(np.r_[y_true, y_pred]))
                        if labels is None else labels)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    weights = s if s.sum() else None
    return {
        "n_samples": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "precision_macro": float(np.mean(p)),
        "recall_macro": float(np.mean(r)),
        "f1_macro": float(np.mean(f)),
        "precision_weighted": float(np.average(p, weights=weights))
        if weights is not None else 0.0,
        "recall_weighted": float(np.average(r, weights=weights))
        if weights is not None else 0.0,
        "f1_weighted": float(np.average(f, weights=weights))
        if weights is not None else 0.0,
        "per_class": {
            str(k): {"precision": float(a), "recall": float(b),
                     "f1": float(c), "support": int(d)}
            for k, a, b, c, d in zip(labels, p, r, f, s)},
        "confusion_matrix": confusion_matrix(y_true, y_pred,
                                             labels=labels).tolist(),
        "labels": labels.tolist(),
    }


AGGREGATED = ("accuracy", "precision_macro", "recall_macro", "f1_macro",
              "precision_weighted", "recall_weighted", "f1_weighted",
              "cohen_kappa")


def aggregate_fold_metrics(fold_metrics: Sequence[dict]) -> dict:
    """Mean +/- standard deviation across folds, per metric."""
    out = {}
    for key in AGGREGATED:
        values = [m[key] for m in fold_metrics if key in m]
        if not values:
            continue
        out[f"{key}_mean"] = float(np.mean(values))
        out[f"{key}_std"] = float(np.std(values))
    out["n_folds"] = len(fold_metrics)
    return out


# ---------------------------------------------------------------------- model
def fit_random_forest(x_train, y_train, cfg: RFExperimentConfig):
    """Fit an imputer and a Random Forest on training data only."""
    imputer = SimpleImputer(strategy=cfg.imputation_strategy,
                            keep_empty_features=True).fit(x_train)
    model = RandomForestClassifier(
        n_estimators=cfg.n_estimators,
        min_samples_leaf=cfg.min_samples_leaf,
        max_features=cfg.max_features,
        class_weight=None if cfg.class_weight == "none" else cfg.class_weight,
        random_state=cfg.seed, n_jobs=cfg.n_jobs,
    ).fit(imputer.transform(x_train), y_train)
    return imputer, model


def _design_matrix(features: pd.DataFrame, feature_names: Sequence[str]):
    missing = [c for c in feature_names if c not in features.columns]
    if missing:
        raise ValueError(f"feature table is missing requested columns: "
                         f"{missing}")
    x = features.loc[:, list(feature_names)].to_numpy(dtype="float64")
    return np.where(np.isfinite(x), x, np.nan)


def spatial_cv_rf(features: pd.DataFrame, labels, fold_grid, *,
                  sample_mask=None, feature_names=None,
                  cfg: RFExperimentConfig | None = None,
                  block_row=None, block_col=None):
    """Spatial block cross-validated Random Forest.

    Each fold is held out in turn; the imputer and the forest are fitted on
    the remaining folds only. Optionally, training samples within
    `cfg.block_cv.buffer_blocks` of a test block are dropped, which requires
    `block_row`/`block_col` for the same samples.

    Returns a dict with predictions, probabilities (model confidence),
    per-fold and pooled metrics, mean +/- std across folds, and the mean
    feature importance.
    """
    cfg = cfg or RFExperimentConfig()
    if feature_names is None:
        feature_names = [c for c in features.columns]
    feature_names = list(feature_names)

    x = _design_matrix(features, feature_names)
    y = np.asarray(labels)
    folds = np.asarray(fold_grid).reshape(-1)
    mask = np.ones(len(y), bool) if sample_mask is None \
        else np.asarray(sample_mask, bool)
    if not (len(x) == len(y) == len(folds) == len(mask)):
        raise ValueError("features, labels, folds and sample_mask must "
                         "describe the same samples")
    if not mask.any():
        raise ValueError("sample_mask selects no samples")

    classes = np.unique(y[mask])
    class_index = {c: i for i, c in enumerate(classes)}
    # zeros_like (not empty_like) so unevaluated entries are deterministic;
    # `evaluated` below is what says which entries carry a prediction.
    predictions = np.zeros_like(y)
    assigned = np.zeros(len(y), dtype=bool)
    probabilities = np.full((len(y), len(classes)), np.nan)
    fold_metrics, importances, fold_sizes = [], [], []

    for fold in sorted(np.unique(folds[mask])):
        test = mask & (folds == fold)
        train = mask & ~test
        if block_row is not None and block_col is not None:
            train = buffered_training_mask(train, test, block_row, block_col,
                                           cfg.block_cv.buffer_blocks)
        if not train.any() or not test.any():
            continue
        if np.unique(y[train]).size < 2:
            continue
        imputer, model = fit_random_forest(x[train], y[train], cfg)
        xt = imputer.transform(x[test])
        predictions[test] = model.predict(xt)
        assigned |= test
        columns = [class_index[c] for c in model.classes_]
        probabilities[np.ix_(test, columns)] = model.predict_proba(xt)
        metrics = classification_metrics(y[test], predictions[test],
                                         labels=classes)
        metrics["fold"] = int(fold)
        metrics["n_train"] = int(train.sum())
        metrics["n_test"] = int(test.sum())
        fold_metrics.append(metrics)
        fold_sizes.append(int(test.sum()))
        importances.append(model.feature_importances_)

    evaluated = mask & assigned
    if not evaluated.any():
        raise ValueError("spatial CV produced no evaluable folds")
    summary = classification_metrics(y[evaluated], predictions[evaluated],
                                     labels=classes)
    summary["validation"] = "spatial_block_cv"
    summary["fold_metrics"] = fold_metrics
    summary["fold_summary"] = aggregate_fold_metrics(fold_metrics)
    summary["n_evaluated"] = int(evaluated.sum())
    summary["n_unevaluated"] = int(mask.sum() - evaluated.sum())
    # Kept for backwards compatibility with the first M2 implementation.
    summary["fold_accuracy_mean"] = summary["fold_summary"].get(
        "accuracy_mean", float("nan"))
    summary["fold_accuracy_std"] = summary["fold_summary"].get(
        "accuracy_std", float("nan"))

    importance = pd.Series(np.mean(importances, axis=0), index=feature_names,
                           name="importance").sort_values(ascending=False)
    return {"predictions": predictions, "probabilities": probabilities,
            "classes": classes, "metrics": summary, "importance": importance,
            "evaluated": evaluated, "feature_names": feature_names,
            "fold_sizes": fold_sizes}


def random_split_baseline(features: pd.DataFrame, labels, *,
                          sample_mask=None, feature_names=None,
                          cfg: RFExperimentConfig | None = None,
                          test_size: float = 0.30) -> dict:
    """Random pixel-level split, kept ONLY as an optimistic baseline.

    Spatial autocorrelation means the held-out pixels are neighbours of
    training pixels, so this number is expected to exceed the spatially
    validated one. It is reported as a baseline, never as evidence of
    spatial generalization.
    """
    cfg = cfg or RFExperimentConfig()
    if feature_names is None:
        feature_names = list(features.columns)
    x = _design_matrix(features, list(feature_names))
    y = np.asarray(labels)
    mask = np.ones(len(y), bool) if sample_mask is None \
        else np.asarray(sample_mask, bool)
    idx = np.flatnonzero(mask)
    counts = pd.Series(y[idx]).value_counts()
    stratify = y[idx] if counts.min() >= 2 else None
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=cfg.seed, stratify=stratify)
    imputer, model = fit_random_forest(x[train_idx], y[train_idx], cfg)
    pred = model.predict(imputer.transform(x[test_idx]))
    metrics = classification_metrics(y[test_idx], pred,
                                     labels=np.unique(y[idx]))
    metrics["validation"] = "random_pixel_split_baseline"
    metrics["test_size"] = float(test_size)
    metrics["caveat"] = ("Random pixel splitting is spatially optimistic; "
                         "the spatial block CV result is the primary "
                         "validation.")
    return {"metrics": metrics, "predictions": pred,
            "test_index": test_idx, "model": model}
