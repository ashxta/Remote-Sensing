"""Vegetation-trajectory representation (M2 Part 3).

WHAT THESE CATEGORIES ARE
-------------------------
`Stable`, `Degrading`, `Recovering`, `Cyclic` and `Uncertain / Other` are
ANALYTICAL TRAJECTORY CLASSES derived from the shape of an NDVI time
series. They are NOT verified land-cover classes and they are NOT proven
land-degradation states.

Specifically, and deliberately:

* `Cyclic` means "spectral power concentrated in the configured period
  band". Rotational cultivation, plantation harvest rotation, fire-regrowth
  cycles and multi-year climate oscillation all produce that signature.
  Cyclic does NOT mean jhum.
* `Degrading` means "a statistically significant negative monotonic trend
  under the configured test". Vegetation decline has many causes, including
  unmodelled climate variability. A negative trend does NOT by itself prove
  land degradation.
* `Recovering` means "a detected disturbance whose NDVI drop has since been
  substantially regained". It does not identify the disturbance agent.

Attribution to a land-use practice or to a degradation process requires
independent ground or ancillary evidence, which this framework does not
have.

RULES
-----
Every threshold comes from `config.TrajectoryConfig` and the evaluation
order is configurable (`priority`). The first matching rule wins; a pixel
matching no rule, or whose core statistics are not finite, is
`Uncertain / Other` rather than being forced into a class.
"""
from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import pandas as pd

from .config import Config, TrajectoryConfig

__all__ = ["TRAJECTORY_CLASSES", "TRAJECTORY_CODES", "UNCERTAIN",
           "classify_trajectories", "trajectory_rules", "trajectory_summary",
           "trajectory_codes", "effective_trajectory_config"]

UNCERTAIN = "Uncertain / Other"
TRAJECTORY_CLASSES = ("Stable", "Degrading", "Recovering", "Cyclic",
                      UNCERTAIN)
#: Stable integer codes, used for georeferenced trajectory maps.
TRAJECTORY_CODES: Dict[str, int] = {
    "Stable": 1, "Degrading": 2, "Recovering": 3, "Cyclic": 4, UNCERTAIN: 5,
}


def effective_trajectory_config(cfg: Config | TrajectoryConfig | None
                                ) -> TrajectoryConfig:
    """Resolve the rule thresholds actually applied.

    A full `Config` wins over the mirrored fields in `TrajectoryConfig`, so
    the significance level, the periodicity threshold and the minimum
    disturbance magnitude have exactly one source of truth and a sensitivity
    sweep over the M1 sections reaches the trajectory rules too. A bare
    `TrajectoryConfig` is used as given.
    """
    if isinstance(cfg, TrajectoryConfig):
        return cfg
    cfg = cfg or Config()
    resolved = copy.deepcopy(cfg.research.trajectory)
    resolved.alpha = cfg.trend.alpha
    resolved.cyclicity_enrichment_threshold = \
        cfg.cyclicity.periodicity_threshold
    resolved.min_disturbance_magnitude = cfg.recovery.min_disturbance_magnitude
    return resolved


def _rule_masks(features: pd.DataFrame, extras: dict,
                cfg: TrajectoryConfig) -> Dict[str, np.ndarray]:
    """Boolean mask per candidate class, before priority is applied."""
    sen = features["sen"].to_numpy(dtype="float64")
    mk_p = np.asarray(extras["mk_p"], dtype="float64")
    enrichment = features["cyc_enrichment"].to_numpy(dtype="float64")
    periodic = np.asarray(extras["cyc_periodic"], dtype=bool)
    magnitude = features["disturbance_magnitude"].to_numpy(dtype="float64")
    fraction = features["recovery_fraction"].to_numpy(dtype="float64")
    disturbed = features["has_disturbance"].to_numpy(dtype="float64") > 0

    significant = np.isfinite(mk_p) & (mk_p < cfg.alpha)
    declining = np.isfinite(sen) & (sen < 0)
    degrading = declining & significant if cfg.require_trend_significance \
        else declining

    cyclic = np.isfinite(enrichment) \
        & (enrichment >= cfg.cyclicity_enrichment_threshold)
    if cfg.require_periodic_flag:
        cyclic &= periodic

    recovering = (disturbed
                  & (magnitude >= cfg.min_disturbance_magnitude)
                  & np.isfinite(fraction)
                  & (fraction >= cfg.recovery_fraction_threshold))
    if cfg.require_significant_breakpoint:
        recovering &= np.asarray(extras["break_significant"], dtype=bool)

    increasing = np.isfinite(sen) & (sen > 0) & significant
    stable = ~degrading & ~cyclic & ~recovering & ~increasing
    return {"Degrading": degrading, "Cyclic": cyclic,
            "Recovering": recovering, "Stable": stable}


def classify_trajectories(features: pd.DataFrame, extras: dict,
                          cfg: Config | TrajectoryConfig | None = None
                          ) -> np.ndarray:
    """Assign one analytical trajectory class per pixel.

    Returns an object array of labels from `TRAJECTORY_CLASSES`. Pixels
    whose core statistics (`mean`, `sen`, `mk_z`) are not all finite are
    `Uncertain / Other`: with too few valid observations the framework
    reports uncertainty instead of guessing.
    """
    cfg = effective_trajectory_config(cfg)
    unknown = [c for c in cfg.priority if c not in TRAJECTORY_CLASSES]
    if unknown:
        raise ValueError(f"unknown trajectory class(es) in priority: {unknown}")

    n = len(features)
    labels = np.full(n, UNCERTAIN, dtype=object)
    finite = np.isfinite(
        features[["mean", "sen", "mk_z"]].to_numpy(dtype="float64")
    ).all(axis=1)
    masks = _rule_masks(features, extras, cfg)
    # Lowest priority first, so the highest-priority rule overwrites last.
    for name in reversed(list(cfg.priority)):
        labels[finite & masks.get(name, np.zeros(n, bool))] = name
    return labels


def trajectory_rules(cfg: Config | TrajectoryConfig | None = None) -> dict:
    """Machine-readable description of the rules actually applied."""
    cfg = effective_trajectory_config(cfg)
    trend_rule = "Sen slope < 0" + (
        f" AND Mann-Kendall p < {cfg.alpha}"
        if cfg.require_trend_significance else "")
    cyclic_rule = (f"spectral enrichment >= "
                   f"{cfg.cyclicity_enrichment_threshold}")
    if cfg.require_periodic_flag:
        cyclic_rule += " AND the estimator's periodic flag is set"
    return {
        "priority": list(cfg.priority),
        "classes": list(TRAJECTORY_CLASSES),
        "codes": dict(TRAJECTORY_CODES),
        "rules": {
            "Degrading": trend_rule,
            "Cyclic": cyclic_rule,
            "Recovering": (
                f"a disturbance of at least {cfg.min_disturbance_magnitude} "
                f"NDVI with recovery fraction >= "
                f"{cfg.recovery_fraction_threshold}"
                + (" AND a significant structural break"
                   if cfg.require_significant_breakpoint else "")),
            "Stable": "no degrading, cyclic, recovering or significant "
                      "increasing signal",
            UNCERTAIN: "core statistics not finite, or no rule matched",
        },
        "interpretation_limit": (
            "Analytical trajectory classes derived from NDVI shape only. "
            "They are not verified land-cover classes; cyclic does not mean "
            "jhum and a negative trend does not prove degradation."),
    }


def trajectory_summary(labels) -> dict:
    """Counts and shares per trajectory class, in a stable class order."""
    labels = np.asarray(labels, dtype=object)
    total = int(labels.size)
    counts = {c: int((labels == c).sum()) for c in TRAJECTORY_CLASSES}
    return {"n_pixels": total, "counts": counts,
            "fractions": {c: (v / total if total else 0.0)
                          for c, v in counts.items()}}


def trajectory_codes(labels) -> np.ndarray:
    """Integer codes for writing a georeferenced trajectory map."""
    labels = np.asarray(labels, dtype=object)
    return np.array([TRAJECTORY_CODES.get(v, TRAJECTORY_CODES[UNCERTAIN])
                     for v in labels], dtype="int16")
