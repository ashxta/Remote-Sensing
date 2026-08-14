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
DEGRADING = "Degrading"
RAINFALL_DECLINE = "Rainfall-associated decline"
DISTURBED = "Disturbed"
RECOVERING = "Recovering"
CYCLIC = "Cyclic"
STABLE = "Stable"

#: The full set of analytical trajectory classes.
#:
#: These separate the six behaviours the research question has to tell
#: apart: ordinary variability (Stable), rainfall-associated change
#: (Rainfall-associated decline), cyclic behaviour (Cyclic), a temporary
#: disturbance that has not returned (Disturbed), recovery (Recovering) and
#: persistent decline that survives climate adjustment (Degrading).
TRAJECTORY_CLASSES = (STABLE, DEGRADING, RAINFALL_DECLINE, DISTURBED,
                      RECOVERING, CYCLIC, UNCERTAIN)

#: Stable integer codes, used for georeferenced trajectory maps.
TRAJECTORY_CODES: Dict[str, int] = {
    STABLE: 1, DEGRADING: 2, RAINFALL_DECLINE: 3, DISTURBED: 4,
    RECOVERING: 5, CYCLIC: 6, UNCERTAIN: 7,
}

#: Classes that describe a decline of some kind, for reporting convenience.
DECLINE_CLASSES = (DEGRADING, RAINFALL_DECLINE, DISTURBED)


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
    declining_significant = declining & significant \
        if cfg.require_trend_significance else declining

    # ------------------------------------------------ climate adjustment
    # A raw negative NDVI trend is not by itself evidence of degradation:
    # a drier run of years produces the same signature. RESTREND is what
    # separates the two, so the decline classes are split by it rather
    # than pooled.
    #
    # Where the NDVI~rainfall relationship is itself significant
    # (`restrend_valid`), the climate-adjusted trend is interpretable:
    #   * still significantly negative -> the decline is NOT explained by
    #     rainfall, so it is reported as a persistent decline;
    #   * no longer significant        -> rainfall accounts for it, so it
    #     is reported as a rainfall-associated decline.
    # Where the relationship is NOT significant, the adjustment means
    # nothing (M1's documented limit), so the pixel keeps the uncorrected
    # decline label. `restrend_valid` travels with the outputs so those
    # pixels can always be separated out again.
    restrend_valid = np.asarray(extras["restrend_valid"], dtype=bool)
    restrend_p = np.asarray(extras["restrend_p"], dtype="float64")
    restrend_slope = features["restrend"].to_numpy(dtype="float64")
    adjusted_decline = (np.isfinite(restrend_p) & (restrend_p < cfg.alpha)
                        & np.isfinite(restrend_slope) & (restrend_slope < 0))

    if cfg.require_climate_adjustment:
        rainfall_decline = (declining_significant & restrend_valid
                            & ~adjusted_decline)
        degrading = declining_significant & ~rainfall_decline
    else:
        rainfall_decline = np.zeros(len(features), bool)
        degrading = declining_significant

    cyclic = np.isfinite(enrichment) \
        & (enrichment >= cfg.cyclicity_enrichment_threshold)
    if cfg.require_periodic_flag:
        cyclic &= periodic

    # --------------------------------------------- disturbance vs recovery
    # A pixel that dropped abruptly and has not returned is a disturbance
    # event, not the same thing as a steady decline; without this class it
    # would be called Stable whenever the drop is too abrupt to register as
    # a significant monotonic trend.
    real_disturbance = disturbed & (magnitude >= cfg.min_disturbance_magnitude)
    if cfg.require_significant_breakpoint:
        real_disturbance &= np.asarray(extras["break_significant"], dtype=bool)
    recovering = (real_disturbance & np.isfinite(fraction)
                  & (fraction >= cfg.recovery_fraction_threshold))
    disturbed_unrecovered = real_disturbance & ~recovering

    increasing = np.isfinite(sen) & (sen > 0) & significant
    stable = (~degrading & ~rainfall_decline & ~cyclic & ~recovering
              & ~disturbed_unrecovered & ~increasing)
    return {DEGRADING: degrading, RAINFALL_DECLINE: rainfall_decline,
            CYCLIC: cyclic, RECOVERING: recovering,
            DISTURBED: disturbed_unrecovered, STABLE: stable}


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
    disturbance_rule = (
        f"a disturbance of at least {cfg.min_disturbance_magnitude} NDVI"
        + (" with a significant structural break"
           if cfg.require_significant_breakpoint else ""))
    return {
        "priority": list(cfg.priority),
        "classes": list(TRAJECTORY_CLASSES),
        "codes": dict(TRAJECTORY_CODES),
        "climate_adjustment_applied": bool(cfg.require_climate_adjustment),
        "rules": {
            DEGRADING: (
                trend_rule + (
                    ", AND either the climate-adjusted (RESTREND) trend is "
                    f"also significantly negative at p < {cfg.alpha}, or the "
                    "NDVI~rainfall relationship is too weak for the "
                    "adjustment to be interpretable (restrend_valid = 0), in "
                    "which case the decline is reported UNCORRECTED"
                    if cfg.require_climate_adjustment else
                    " (no climate adjustment applied)")),
            RAINFALL_DECLINE: (
                "a significant raw decline that is NO LONGER significant "
                "after adjusting for rainfall, where that adjustment is "
                "interpretable (restrend_valid = 1); the decline is "
                "associated with rainfall variability"
                if cfg.require_climate_adjustment else
                "not used: climate adjustment is disabled"),
            DISTURBED: (
                disturbance_rule + ", with recovery fraction below "
                f"{cfg.recovery_fraction_threshold}: an abrupt drop that has "
                "not returned, which is not the same as a steady decline"),
            RECOVERING: (
                disturbance_rule + " and recovery fraction >= "
                f"{cfg.recovery_fraction_threshold}"),
            CYCLIC: cyclic_rule,
            STABLE: "no decline, disturbance, cyclic or significant "
                    "increasing signal",
            UNCERTAIN: "core statistics not finite, or no rule matched",
        },
        "interpretation_limit": (
            "Analytical trajectory classes derived from NDVI and rainfall "
            "shape only. They are not verified land-cover classes: cyclic "
            "does not mean jhum, a negative trend does not prove "
            "degradation, and a decline surviving climate adjustment is not "
            "proof of anthropogenic causation - unmodelled climate "
            "variables, fire, pests and land-cover conversion produce the "
            "same signature."),
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
