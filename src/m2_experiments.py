"""Backwards-compatible entry point for the M2 research utilities.

The first M2 implementation put feature engineering, trajectory categories,
spatial cross-validation, the Random Forest pipeline, the ablation runner
and the holdout helpers in this single module. M2 split them into focused
modules so each concern can be read and tested on its own:

    features.py          standardized temporal feature engineering
    trajectory.py        analytical vegetation-trajectory classes
    validation.py        spatial block CV + the Random Forest pipeline
    ablation.py          configuration-driven ablation study
    sensitivity.py       methodological-parameter robustness sweeps
    holdout.py           temporal holdout infrastructure
    dataset.py           standardized-data contract and validity checks
    research_figures.py  research figures

This module re-exports the original names so existing callers, notebooks
and `cnn_experiment` keep working unchanged. New code should import from
the focused modules.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .ablation import experiment_id, run_ablation_study
from .config import (AblationConfig, Config, RFExperimentConfig,
                     SpatialCVConfig, TrajectoryConfig)
from .dataset import DatasetValidationError, validate_dataset
from .features import (FEATURE_GROUPS, build_feature_table, feature_dictionary,
                       feature_names)
from .holdout import temporal_holdout_indices
from .trajectory import classify_trajectories, trajectory_summary
from .validation import (classification_metrics, spatial_block_folds,
                         spatial_cv_rf)

__all__ = ["FEATURE_SETS", "SpatialCVConfig", "RFExperimentConfig",
           "build_m2_features", "build_feature_table", "feature_names",
           "feature_dictionary", "FEATURE_GROUPS", "validate_dataset",
           "DatasetValidationError", "trajectory_categories",
           "classify_trajectories", "trajectory_summary",
           "spatial_block_folds", "classification_metrics", "spatial_cv_rf",
           "experiment_id", "run_ablations", "run_ablation_study",
           "temporal_holdout_indices", "sensitivity_configs"]


def _feature_sets() -> dict:
    """{ablation cell name: resolved feature list} for the default design."""
    return {e.name: feature_names(e.groups)
            for e in AblationConfig().experiments}


#: Ablation feature sets A-F, resolved from the configured feature groups.
FEATURE_SETS = _feature_sets()


def build_m2_features(ndvi, rain, cfg: Config | None = None):
    """Deprecated alias for `features.build_feature_table`."""
    return build_feature_table(ndvi, rain, cfg)


def trajectory_categories(features: pd.DataFrame, extras: dict, *,
                          alpha: float = 0.05,
                          cyclicity_threshold: float = 2.0) -> np.ndarray:
    """Deprecated alias for `trajectory.classify_trajectories`.

    Kept so existing callers can pass the two thresholds positionally by
    keyword; new code should build a `TrajectoryConfig` instead.
    """
    cfg = TrajectoryConfig(alpha=alpha,
                           cyclicity_enrichment_threshold=cyclicity_threshold)
    return classify_trajectories(features, extras, cfg)


def run_ablations(features, labels, fold_grid, output_dir,
                  cfg: RFExperimentConfig | None = None, sample_mask=None):
    """Deprecated alias for `ablation.run_ablation_study`."""
    return run_ablation_study(features, labels, fold_grid, output_dir,
                              rf_cfg=cfg or RFExperimentConfig(),
                              sample_mask=sample_mask)


def sensitivity_configs(base_cfg, values: dict) -> Iterable:
    """Deprecated: enumerate (parameter, value) scenarios.

    Superseded by `sensitivity.run_sensitivity_analysis`, which actually
    runs each scenario end-to-end and saves machine-readable results.
    """
    for dotted, candidates in values.items():
        for value in candidates:
            yield dotted, value
