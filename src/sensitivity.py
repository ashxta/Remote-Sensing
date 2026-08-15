"""Sensitivity analysis for methodological parameters (M2 Part 7).

Purpose: establish whether the conclusions drawn from this framework survive
reasonable changes to the parameters that were chosen by convention -
significance level, minimum valid observations, the periodicity band, the
breakpoint gain threshold, the recovery threshold, and so on.

What this is NOT
----------------
This is not hyper-parameter tuning. No parameter is selected by comparing
test-set metrics: the framework reports what each setting produces and
leaves the choice to the documented scientific default in `Config`. Doing
otherwise would tune the method on the evaluation data.

Each scenario re-runs the full analysis (feature engineering, trajectory
categories and, optionally, the spatially validated classifier) from the
raw series under the modified configuration, so the reported spread is the
end-to-end effect of the parameter, not a partial one.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import Config, ParameterSweep, RFExperimentConfig
from .features import build_feature_table, feature_names
from .trajectory import TRAJECTORY_CLASSES, classify_trajectories
from .validation import spatial_cv_rf

__all__ = ["apply_override", "read_parameter", "scenario_table",
           "run_sensitivity_analysis"]


def _resolve(cfg: Config, dotted: str):
    """Return (owner_object, attribute_name) for a dotted config path."""
    parts = dotted.split(".")
    owner = cfg
    for part in parts[:-1]:
        if not hasattr(owner, part):
            raise ValueError(f"unknown configuration section {part!r} in "
                             f"{dotted!r}")
        owner = getattr(owner, part)
    if not hasattr(owner, parts[-1]):
        raise ValueError(f"unknown configuration parameter {dotted!r}")
    return owner, parts[-1]


def read_parameter(cfg: Config, dotted: str):
    owner, name = _resolve(cfg, dotted)
    return getattr(owner, name)


def apply_override(cfg: Config, dotted: str, value) -> Config:
    """Return a deep copy of `cfg` with one parameter changed.

    The input configuration is never mutated, so scenarios cannot leak into
    each other or into the baseline run.
    """
    new = copy.deepcopy(cfg)
    owner, name = _resolve(new, dotted)
    current = getattr(owner, name)
    if isinstance(current, bool):
        value = bool(value)
    elif isinstance(current, int) and not isinstance(value, bool):
        value = int(value)
    elif isinstance(current, float):
        value = float(value)
    setattr(owner, name, value)
    return new


def scenario_table(parameters: Sequence[ParameterSweep],
                   base_cfg: Config) -> pd.DataFrame:
    """Every (parameter, value) scenario the sweep will run, with baselines."""
    rows = []
    for sweep in parameters:
        baseline = read_parameter(base_cfg, sweep.path)
        for value in sweep.values:
            rows.append({"parameter": sweep.path, "value": value,
                         "baseline_value": baseline,
                         "is_baseline": bool(value == baseline),
                         "description": sweep.description})
    return pd.DataFrame(rows)


def _evaluate(ndvi, rain, labels, fold_grid, cfg: Config,
              rf_cfg: RFExperimentConfig | None, sample_mask,
              model_features) -> dict:
    """One scenario: features -> trajectories -> (optional) model metrics."""
    table, extras = build_feature_table(ndvi, rain, cfg)
    categories = classify_trajectories(table, extras, cfg)
    out = {
        "n_pixels": int(len(table)),
        "mean_sen_slope": float(np.nanmean(table["sen"].to_numpy())),
        "significant_trend_fraction": float(
            np.nanmean(table["mk_significant"].to_numpy())),
        "periodic_fraction": float(
            np.nanmean(table["cyclicity_periodic"].to_numpy())),
        "restrend_significant_fraction": float(
            np.nanmean(table["restrend_significant"].to_numpy())),
        "break_fraction": float(np.nanmean(table["has_break"].to_numpy())),
        "disturbance_fraction": float(
            np.nanmean(table["has_disturbance"].to_numpy())),
    }
    for cls in TRAJECTORY_CLASSES:
        key = cls.split(" ")[0].lower()
        out[f"trajectory_{key}_fraction"] = float(np.mean(categories == cls))
    if rf_cfg is not None and labels is not None:
        result = spatial_cv_rf(table, labels, fold_grid,
                               sample_mask=sample_mask,
                               feature_names=model_features, cfg=rf_cfg)
        metrics = result["metrics"]
        out.update({
            "accuracy": metrics["accuracy"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "fold_f1_macro_mean": metrics["fold_summary"].get(
                "f1_macro_mean", float("nan")),
            "fold_f1_macro_std": metrics["fold_summary"].get(
                "f1_macro_std", float("nan")),
        })
    return out


def run_sensitivity_analysis(ndvi, rain, output_dir, cfg: Config | None = None,
                             *, labels=None, fold_grid=None, sample_mask=None,
                             parameters: Sequence[ParameterSweep] | None = None,
                             evaluate_model: bool | None = None,
                             data_status: str | None = None,
                             logger=None) -> pd.DataFrame:
    """Run every configured parameter scenario and save the results.

    Returns a tidy DataFrame with one row per (parameter, value) scenario.
    Writes `sensitivity_results.csv`, `sensitivity_results.json` and
    `sensitivity_scenarios.csv` into `output_dir`.
    """
    cfg = cfg or Config()
    sens = cfg.research.sensitivity
    parameters = list(parameters if parameters is not None
                      else sens.parameters)
    if evaluate_model is None:
        evaluate_model = sens.evaluate_model
    evaluate_model = bool(evaluate_model and labels is not None
                          and fold_grid is not None)

    rf_cfg = None
    if evaluate_model:
        rf_cfg = copy.deepcopy(cfg.research.model)
        rf_cfg.n_estimators = sens.n_estimators
    model_features = feature_names(cfg.research.features.groups)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    scenario_table(parameters, cfg).to_csv(
        root / "sensitivity_scenarios.csv", index=False)

    baseline = _evaluate(ndvi, rain, labels, fold_grid, cfg, rf_cfg,
                         sample_mask, model_features)
    rows = [{"parameter": "(baseline)", "value": None,
             "is_baseline": True, **baseline}]
    for sweep in parameters:
        current = read_parameter(cfg, sweep.path)
        for value in sweep.values:
            scenario_cfg = apply_override(cfg, sweep.path, value)
            result = _evaluate(ndvi, rain, labels, fold_grid, scenario_cfg,
                               rf_cfg, sample_mask, model_features)
            rows.append({"parameter": sweep.path, "value": value,
                         "is_baseline": bool(value == current),
                         "description": sweep.description, **result})
            if logger is not None:
                logger.info("  sensitivity %-38s = %-6s | periodic %.3f | "
                            "degrading %.3f%s", sweep.path, value,
                            result["periodic_fraction"],
                            result["trajectory_degrading_fraction"],
                            f" | macro F1 {result['f1_macro']:.4f}"
                            if "f1_macro" in result else "")

    table = pd.DataFrame(rows)
    table.to_csv(root / "sensitivity_results.csv", index=False)

    numeric = table.select_dtypes(include=[np.number])
    spread = {}
    for sweep in parameters:
        subset = table[table["parameter"] == sweep.path]
        spread[sweep.path] = {
            column: {"min": float(subset[column].min()),
                     "max": float(subset[column].max()),
                     "range": float(subset[column].max()
                                    - subset[column].min())}
            for column in numeric.columns if column in subset
            and subset[column].notna().any()
        }
    (root / "sensitivity_results.json").write_text(json.dumps({
        "baseline": baseline,
        "parameters": [asdict(p) for p in parameters],
        "model_evaluated": bool(evaluate_model),
        "spread_by_parameter": spread,
        "note": ("Scenarios are reported, not optimised: no parameter is "
                 "selected by comparing test-set metrics."),
        # The caller states the provenance. Hard-coding "synthetic" here
        # would stamp that label onto real-data results.
        "data_status": data_status or ("unspecified; the caller did not "
                                       "declare the data provenance"),
    }, indent=2, default=str))
    return table
