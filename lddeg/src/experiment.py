"""Shared experiment preparation (M4).

The M2 and M3 runners performed the same first four stages - load, validate,
gate, engineer features, classify trajectories, lay out spatial folds, draw
the model sample - in duplicated code. Those stages are the experiment's
common ground, so they live here once:

    configuration
      -> data source            (data_source.DataSource)
      -> contract validation    (dataset.validate_dataset)
      -> quality gating         (quality.assess)
      -> feature engineering    (features.build_feature_table)
      -> trajectory classes     (trajectory.classify_trajectories)
      -> spatial fold design    (validation.spatial_block_folds)
      -> model sampling         (seeded, class-stratified)
      -> PreparedExperiment

A runner then adds only what makes it that experiment: M2 adds ablation and
sensitivity, M3 adds the leakage audit, the method matrix, uncertainty and
explainability.

`PreparedExperiment` is a plain dataclass of aligned arrays. Everything in
it is indexed the same way - one row per ANALYSED pixel - so a runner never
has to re-derive an index mapping, which is where the alignment bugs live.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import Config
from .data_source import DataSource, StandardizedDataset, load_dataset
from .features import build_feature_table, feature_names
from .geo import GeoRef
from .quality import QualityReport, assess
from .trajectory import classify_trajectories, trajectory_rules, trajectory_summary
from .validation import block_coordinates, spatial_block_folds

__all__ = ["PreparedExperiment", "prepare_experiment",
           "select_analysis_pixels", "select_model_samples",
           "SYNTHETIC_NOTICE"]

#: Written into every experiment directory. One sentence, no hedging.
SYNTHETIC_NOTICE = (
    "Synthetic data are used for development and pipeline validation. They "
    "are not real observations and must not be interpreted as real-world "
    "research findings. No output in this directory measures any real "
    "location.\n")


@dataclass
class PreparedExperiment:
    """Everything the analysis stages of a runner need, aligned by row.

    Grid-shaped members are (H, W); per-pixel members are (N_analysed,) or
    (N_analysed, ...) unless documented otherwise.
    """
    config: Config
    dataset: StandardizedDataset
    georef: GeoRef
    validation_report: Dict[str, Any]
    quality: QualityReport
    quality_summary: Dict[str, Any]

    #: Boolean mask over ALL grid pixels selecting the analysed ones.
    analysis_mask: np.ndarray
    features: pd.DataFrame                 # (N_analysed, n_features)
    extras: Dict[str, Any]                 # M1 diagnostics, aligned by row
    model_columns: List[str]
    trajectory_labels: np.ndarray          # (N_analysed,) object
    trajectory_summary: Dict[str, Any]

    folds: np.ndarray                      # (N_analysed,) fold id
    fold_grid: np.ndarray                  # (H, W) fold id, for figures/maps
    block_row: np.ndarray                  # (N_analysed,) block row index
    block_col: np.ndarray                  # (N_analysed,) block column index
    block_id_grid: np.ndarray              # (H, W) block id, for audits

    labels: Optional[np.ndarray] = None    # (N_analysed,) reference class
    sample_mask: Optional[np.ndarray] = None   # (N_analysed,) model sample
    #: Column indices into the FLAT grid for the sampled pixels, so a runner
    #: can slice the raw series without recomputing the mapping.
    sample_columns: Optional[np.ndarray] = None
    series: Optional[np.ndarray] = None    # (T, N_analysed) raw NDVI
    rain_series: Optional[np.ndarray] = None
    notes: List[str] = field(default_factory=list)

    # ------------------------------------------------------------- geometry
    @property
    def shape(self) -> tuple:
        return self.dataset.shape

    @property
    def analysis_grid(self) -> np.ndarray:
        """The analysis mask reshaped to the grid, for writing rasters."""
        return self.analysis_mask.reshape(self.shape)

    @property
    def n_analysed(self) -> int:
        return int(self.analysis_mask.sum())

    @property
    def has_labels(self) -> bool:
        return self.labels is not None

    def summary(self) -> dict:
        """Machine-readable description of what was prepared."""
        return {
            "n_time_steps": self.dataset.n_time,
            "grid": list(self.shape),
            "n_pixels": self.dataset.n_pixels,
            "n_analysed": self.n_analysed,
            "n_model_samples": int(self.sample_mask.sum())
            if self.sample_mask is not None else 0,
            "n_features": len(self.model_columns),
            "n_folds": int(np.unique(self.folds).size),
            "block_size": self.config.research.spatial_cv.block_size,
            "has_reference_labels": self.has_labels,
            "notes": list(self.notes),
        }


def select_analysis_pixels(usable: np.ndarray, cfg: Config) -> np.ndarray:
    """Optionally thin the analysed pixels, deterministically.

    `research.max_analysis_pixels` of -1 analyses every pixel that passes
    quality gating. A positive cap draws a seeded subsample, which is what
    the `--quick` smoke configurations use; the choice is recorded in the
    experiment summary so a thinned run is never mistaken for a full one.
    """
    limit = cfg.research.max_analysis_pixels
    if limit is None or limit < 0 or int(usable.sum()) <= limit:
        return usable
    rng = np.random.default_rng(cfg.seed)
    chosen = rng.choice(np.flatnonzero(usable), int(limit), replace=False)
    thinned = np.zeros_like(usable)
    thinned[np.sort(chosen)] = True
    return thinned


def select_model_samples(truth: Optional[np.ndarray], mask: np.ndarray,
                         cfg: Config) -> np.ndarray:
    """Seeded, class-stratified sample of the analysed pixels.

    Stratification keeps rare reference classes represented; the cap keeps
    an experiment tractable. Both are configuration
    (`research.samples_per_class`), and the draw is reproducible from the
    seed alone.
    """
    if truth is None:
        return np.zeros(mask.shape, bool)
    rng = np.random.default_rng(cfg.seed)
    selected = np.zeros(mask.shape, bool)
    per_class = cfg.research.samples_per_class
    for value in np.unique(truth[mask]):
        candidates = np.flatnonzero(mask & (truth == value))
        if candidates.size == 0:
            continue
        take = rng.choice(candidates, min(per_class, candidates.size),
                          replace=False)
        selected[take] = True
    return selected


def prepare_experiment(cfg: Config, *, source: DataSource | None = None,
                       strict_validation: bool = False,
                       logger=None) -> PreparedExperiment:
    """Run the stages every experiment shares and return aligned arrays.

    Raises `DatasetValidationError` if the data violate the contract, and
    `RuntimeError` if quality gating leaves nothing to analyse - neither is
    recoverable by guessing.
    """
    dataset, report = load_dataset(cfg, source, strict=strict_validation)
    notes = []
    if logger is not None:
        logger.info("data source: %s (%s)",
                    dataset.metadata.get("source", "unknown"),
                    dataset.metadata.get("description", ""))
        logger.info("grid %dx%d, %d time steps, CRS=%s", *dataset.shape,
                    dataset.n_time, dataset.georef.crs)
        for warning in report.get("warnings", []):
            logger.warning("data warning: %s", warning)

    ndvi_flat, rain_flat = dataset.flat()
    quality = assess(ndvi_flat, cfg.quality)
    quality_summary = quality.summary()
    analysis_mask = select_analysis_pixels(quality.usable, cfg)
    if not analysis_mask.any():
        raise RuntimeError(
            "no pixels passed the quality gate; loosen QualityConfig "
            "(min_valid_obs, max_missing_fraction) or check the input data")
    if int(analysis_mask.sum()) < int(quality.usable.sum()):
        notes.append(
            f"analysed pixels thinned to {int(analysis_mask.sum())} of "
            f"{int(quality.usable.sum())} usable "
            "(research.max_analysis_pixels); this is a reduced-scale run")
    if logger is not None:
        logger.info("usable %d/%d pixels; analysing %d",
                    quality_summary["n_usable"], quality_summary["n_pixels"],
                    int(analysis_mask.sum()))

    series = ndvi_flat[:, analysis_mask]
    rain_series = rain_flat[:, analysis_mask]
    features, extras = build_feature_table(series, rain_series, cfg)
    model_columns = feature_names(cfg.research.features.groups)
    labels_grid = dataset.flat_truth()

    trajectory_labels = classify_trajectories(features, extras, cfg)
    summary = trajectory_summary(trajectory_labels)
    summary["definitions"] = trajectory_rules(cfg)
    if logger is not None:
        logger.info("%d modelling features over %d pixels",
                    len(model_columns), len(features))
        logger.info("trajectory classes: %s", ", ".join(
            f"{name} {count}" for name, count in summary["counts"].items()))

    height, width = dataset.shape
    block_row_grid, block_col_grid, block_id_grid = block_coordinates(
        height, width, cfg.research.spatial_cv.block_size)
    _, fold_grid = spatial_block_folds(height, width, cfg.research.spatial_cv)

    labels = sample_mask = sample_columns = None
    if labels_grid is not None:
        labels = labels_grid[analysis_mask]
        sample_all = select_model_samples(labels_grid, analysis_mask, cfg)
        index_of = np.full(analysis_mask.size, -1, dtype="int64")
        index_of[analysis_mask] = np.arange(int(analysis_mask.sum()))
        sample_mask = np.zeros(len(features), bool)
        sample_mask[index_of[sample_all]] = True
        sample_columns = np.flatnonzero(analysis_mask)[sample_mask]
        if logger is not None:
            logger.info("%d model samples (%d per class requested) over "
                        "%d spatial folds, block_size=%d",
                        int(sample_mask.sum()), cfg.research.samples_per_class,
                        cfg.research.spatial_cv.n_folds,
                        cfg.research.spatial_cv.block_size)
    elif logger is not None:
        logger.warning("no reference labels available; supervised stages "
                       "will be skipped")

    return PreparedExperiment(
        config=cfg, dataset=dataset, georef=dataset.georef,
        validation_report=report, quality=quality,
        quality_summary=quality_summary, analysis_mask=analysis_mask,
        features=features, extras=extras, model_columns=model_columns,
        trajectory_labels=trajectory_labels, trajectory_summary=summary,
        folds=fold_grid.reshape(-1)[analysis_mask], fold_grid=fold_grid,
        block_row=block_row_grid.reshape(-1)[analysis_mask],
        block_col=block_col_grid.reshape(-1)[analysis_mask],
        block_id_grid=block_id_grid, labels=labels, sample_mask=sample_mask,
        sample_columns=sample_columns, series=series, rain_series=rain_series,
        notes=notes)
