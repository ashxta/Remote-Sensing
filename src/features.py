"""Standardized temporal feature engineering (M2 Part 2).

One entry point, `build_feature_table`, turns a standardized (T, N) NDVI /
rainfall pair into a documented per-pixel feature table. It consumes the M1
estimators unchanged (`timeseries`, `recovery`, `classify.build_features`)
and adds only descriptive summaries and diagnostics on top of them.

Design rules
------------
* Deterministic: no RNG is touched anywhere in this module.
* Configurable: every threshold comes from `Config`; nothing is hard-coded
  in a signature.
* Documented: every column has a `FeatureSpec` with a group, a description
  and the estimator it came from, exported by `feature_dictionary()` and
  written next to the results of every experiment.
* NaN-aware: missing observations are excluded per pixel. Pixels with too
  few valid observations get NaN features (M1 policy) rather than a
  fabricated value.
* Structural vs unknown missingness: descriptors that are UNDEFINED because
  the underlying event did not happen (no break, no disturbance) are set to
  a neutral 0 next to an explicit indicator (`has_break`,
  `has_disturbance`), exactly as M1 does. That is an explicit encoding of a
  known state, not imputation of an unobserved value.

Feature groups are the unit of configuration for the ablation study
(`config.AblationConfig`): a group is either fully available to a model or
fully withheld.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from . import timeseries as TS
from .classify import FEATURES as M1_FEATURES, build_features
from .config import Config

__all__ = ["FeatureSpec", "FEATURE_SPECS", "FEATURE_GROUPS", "GROUP_ORDER",
           "feature_names", "feature_dictionary", "build_feature_table",
           "DIAGNOSTIC_COLUMNS"]


@dataclass(frozen=True)
class FeatureSpec:
    """Definition and provenance of one engineered feature."""
    name: str
    group: str
    description: str
    source: str


#: Group evaluation order. Also the column order of the design matrix.
GROUP_ORDER = ("vegetation", "trend", "rainfall", "restrend", "cyclicity",
               "disturbance", "recovery", "diagnostic")

FEATURE_SPECS: tuple = (
    # ------------------------------------------------------------ vegetation
    FeatureSpec("mean", "vegetation", "Mean NDVI over valid observations",
                "classify.build_features"),
    FeatureSpec("median_ndvi", "vegetation", "Median NDVI (outlier-robust "
                "central level)", "features.build_feature_table"),
    FeatureSpec("minimum_ndvi", "vegetation", "Minimum observed NDVI",
                "features.build_feature_table"),
    FeatureSpec("maximum_ndvi", "vegetation", "Maximum observed NDVI",
                "features.build_feature_table"),
    FeatureSpec("std", "vegetation", "Standard deviation of NDVI "
                "(temporal variability)", "classify.build_features"),
    FeatureSpec("cv", "vegetation", "Coefficient of variation, std/mean",
                "classify.build_features"),
    FeatureSpec("amp", "vegetation", "Amplitude, 95th minus 5th percentile",
                "classify.build_features"),
    FeatureSpec("sen", "vegetation", "Theil-Sen slope of NDVI per time step",
                "timeseries.sens_slope"),
    FeatureSpec("ols_slope", "vegetation", "Ordinary least-squares temporal "
                "slope of NDVI per time step", "features.build_feature_table"),
    # ----------------------------------------------------------------- trend
    FeatureSpec("mk_z", "trend", "Mann-Kendall Z, Hamed-Rao autocorrelation "
                "adjusted when enabled in TrendConfig",
                "timeseries.mann_kendall + hamed_rao_correction"),
    FeatureSpec("mk_p_value", "trend", "Two-sided Mann-Kendall p-value "
                "(autocorrelation adjusted when enabled)",
                "timeseries.mann_kendall + hamed_rao_correction"),
    FeatureSpec("mk_tau", "trend", "Kendall's tau rank correlation with time",
                "timeseries.mann_kendall"),
    FeatureSpec("mk_significant", "trend", "1 where the Mann-Kendall p-value "
                "is below TrendConfig.alpha", "features.build_feature_table"),
    FeatureSpec("trend_direction", "trend", "Sign of the Theil-Sen slope "
                "(-1, 0, +1)", "features.build_feature_table"),
    # -------------------------------------------------------------- rainfall
    FeatureSpec("rain_mean", "rainfall", "Mean rainfall over valid steps",
                "features.build_feature_table"),
    FeatureSpec("rain_std", "rainfall", "Standard deviation of rainfall",
                "features.build_feature_table"),
    FeatureSpec("rain_cv", "rainfall", "Rainfall coefficient of variation",
                "features.build_feature_table"),
    FeatureSpec("rain_sen", "rainfall", "Theil-Sen slope of rainfall per "
                "time step", "timeseries.sens_slope"),
    FeatureSpec("rain_ndvi_correlation", "rainfall", "Pearson correlation of "
                "NDVI and rainfall over commonly valid steps",
                "features.build_feature_table"),
    # -------------------------------------------------------------- restrend
    FeatureSpec("restrend", "restrend", "Climate-adjusted (residual) NDVI "
                "trend; interpretable only where restrend_significant",
                "timeseries.restrend"),
    FeatureSpec("restrend_p_value", "restrend", "p-value of the "
                "climate-adjusted trend", "timeseries.restrend"),
    FeatureSpec("restrend_r2", "restrend", "Partial r2 of the NDVI~rainfall "
                "relationship controlling for time", "timeseries.restrend"),
    FeatureSpec("restrend_beta", "restrend", "NDVI sensitivity to rainfall",
                "timeseries.restrend"),
    FeatureSpec("restrend_valid", "restrend", "1 where the NDVI~rainfall "
                "relationship is strong and significant enough for the "
                "climate adjustment to be interpretable at all; where 0, a "
                "residual trend is NOT climate-corrected in any meaningful "
                "sense", "timeseries.restrend"),
    FeatureSpec("restrend_significant", "restrend", "1 where the residual "
                "trend is significant AND the rainfall relation is valid",
                "features.build_feature_table"),
    # ------------------------------------------------------------- cyclicity
    FeatureSpec("cyc_score", "cyclicity", "Share of non-DC spectral power "
                "inside the configured period band", "timeseries.cyclicity"),
    FeatureSpec("cyc_enrichment", "cyclicity", "Band power fraction divided "
                "by its flat-noise expectation; 1.0 = white noise",
                "timeseries.cyclicity"),
    FeatureSpec("cyc_period", "cyclicity", "Dominant period in time steps "
                "(0 where undefined)", "timeseries.cyclicity"),
    FeatureSpec("cyc_band_power", "cyclicity", "Absolute spectral power "
                "inside the period band", "timeseries.cyclicity"),
    FeatureSpec("cyc_total_power", "cyclicity", "Absolute non-DC spectral "
                "power of the series", "timeseries.cyclicity"),
    FeatureSpec("cyclicity_periodic", "cyclicity", "1 where enrichment "
                "reaches CyclicityConfig.periodicity_threshold; periodicity "
                "is NOT attribution of any land use", "timeseries.cyclicity"),
    # ----------------------------------------------------------- disturbance
    FeatureSpec("has_break", "disturbance", "1 where an admissible "
                "structural break exists", "timeseries.best_breakpoint"),
    FeatureSpec("break_t", "disturbance", "Break position as a fraction of "
                "the record (0 where none)", "timeseries.best_breakpoint"),
    FeatureSpec("break_dslope", "disturbance", "Post minus pre segment slope "
                "(0 where none)", "timeseries.best_breakpoint"),
    FeatureSpec("break_gain", "disturbance", "Variance explained by the "
                "two-segment fit (0 where none)", "timeseries.best_breakpoint"),
    FeatureSpec("breakpoint_index", "disturbance", "Break time index "
                "(-1 where none)", "timeseries.best_breakpoint"),
    FeatureSpec("breakpoint_significant", "disturbance", "1 where the "
                "selection-adjusted Chow test is significant",
                "timeseries.best_breakpoint"),
    FeatureSpec("pre_breakpoint_level", "disturbance", "Mean NDVI over the "
                "pre-break window (0 where no break)", "recovery.analyze"),
    FeatureSpec("post_breakpoint_minimum", "disturbance", "Minimum NDVI after "
                "the break (0 where no break)", "recovery.analyze"),
    FeatureSpec("disturbance_magnitude", "disturbance", "Pre-break level "
                "minus post-break trough (0 where no disturbance)",
                "recovery.analyze"),
    # -------------------------------------------------------------- recovery
    FeatureSpec("has_disturbance", "recovery", "1 where a disturbance "
                "exceeding RecoveryConfig.min_disturbance_magnitude exists",
                "recovery.analyze"),
    FeatureSpec("recovery_magnitude", "recovery", "Last value minus trough "
                "(0 where no disturbance)", "recovery.analyze"),
    FeatureSpec("recovery_fraction", "recovery", "Recovered share of the "
                "disturbance magnitude (0 where no disturbance)",
                "recovery.analyze"),
    FeatureSpec("recovery_duration", "recovery", "Steps from trough to the "
                "first crossing of the recovery threshold; 0 where the "
                "threshold was never reached (see recovery_status)",
                "recovery.analyze"),
    FeatureSpec("recovery_slope", "recovery", "OLS slope from the trough to "
                "the end of the record (0 where no disturbance)",
                "recovery.analyze"),
    FeatureSpec("recovery_status", "recovery", "Categorical code: 0 none, "
                "1 recovered, 2 recovering, 3 not recovering, "
                "4 insufficient data", "recovery.analyze"),
    # ------------------------------------------------------------ diagnostic
    FeatureSpec("n_valid_ndvi", "diagnostic", "Number of finite NDVI "
                "observations", "features.build_feature_table"),
    FeatureSpec("n_valid_pairs", "diagnostic", "Number of steps where NDVI "
                "and rainfall are both finite", "features.build_feature_table"),
)

#: Columns produced for traceability but withheld from models by default.
DIAGNOSTIC_COLUMNS = tuple(s.name for s in FEATURE_SPECS
                           if s.group == "diagnostic")

FEATURE_GROUPS = {
    group: [s.name for s in FEATURE_SPECS if s.group == group]
    for group in GROUP_ORDER
}


def feature_names(groups: Iterable[str] | None = None) -> List[str]:
    """Ordered feature names for the requested groups.

    `groups=None` returns every modelling group (diagnostics excluded).
    Unknown group names are an error, never a silent empty set.
    """
    if groups is None:
        groups = [g for g in GROUP_ORDER if g != "diagnostic"]
    groups = list(groups)
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"unknown feature group(s) {unknown}; "
                         f"known groups are {list(FEATURE_GROUPS)}")
    ordered = [g for g in GROUP_ORDER if g in groups]
    names: List[str] = []
    for g in ordered:
        names.extend(n for n in FEATURE_GROUPS[g] if n not in names)
    return names


def feature_dictionary(names: Sequence[str] | None = None) -> pd.DataFrame:
    """Machine-readable definition table for the produced features."""
    specs = FEATURE_SPECS if names is None else \
        [s for s in FEATURE_SPECS if s.name in set(names)]
    return pd.DataFrame([{"feature": s.name, "group": s.group,
                          "description": s.description, "source": s.source}
                         for s in specs])


def _ols_slope(x: np.ndarray) -> np.ndarray:
    """NaN-aware least-squares slope against the time index, per column."""
    T = x.shape[0]
    t = np.arange(T, dtype="float64")[:, None]
    good = np.isfinite(x)
    tt = np.where(good, t, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        tm = tt - np.nanmean(tt, axis=0)
        xm = x - np.nanmean(x, axis=0)
        stt = np.nansum(tm ** 2, axis=0)
        slope = np.where(stt > 0, np.nansum(tm * xm, axis=0) / stt, np.nan)
    return np.where(good.sum(axis=0) >= 2, slope, np.nan)


def _pearson(a: np.ndarray, b: np.ndarray, min_pairs: int = 3) -> np.ndarray:
    """NaN-aware Pearson correlation over commonly valid steps, per column."""
    both = np.isfinite(a) & np.isfinite(b)
    n = both.sum(axis=0)
    av = np.where(both, a, np.nan)
    bv = np.where(both, b, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ac = av - np.nanmean(av, axis=0)
        bc = bv - np.nanmean(bv, axis=0)
        denom = np.sqrt(np.nansum(ac ** 2, axis=0) * np.nansum(bc ** 2, axis=0))
        corr = np.where((n >= min_pairs) & (denom > 0),
                        np.nansum(ac * bc, axis=0) / denom, np.nan)
    return corr


def _structural(values, present) -> np.ndarray:
    """M1 encoding: neutral 0 where the descriptor is undefined."""
    v = np.asarray(values, dtype="float64").copy()
    present = np.asarray(present, dtype=bool)
    v[~present] = 0.0
    v[present & ~np.isfinite(v)] = 0.0
    return v


def build_feature_table(ndvi, rain, cfg: Config | None = None):
    """Build the standardized M2 feature table.

    Parameters
    ----------
    ndvi, rain : (T, N) arrays; NaN marks a missing observation.
    cfg : Config; all thresholds are read from it.

    Returns
    -------
    (DataFrame with one row per pixel and every documented column,
     dict of M1 diagnostics as returned by `classify.build_features`)

    The M1 feature block is reproduced column-for-column and unchanged; M2
    columns are appended. Column order follows `GROUP_ORDER`.
    """
    cfg = cfg or Config()
    ndvi = np.asarray(ndvi, dtype="float64")
    rain = np.asarray(rain, dtype="float64")
    if ndvi.ndim == 1:
        ndvi, rain = ndvi[:, None], rain[:, None]
    if ndvi.shape != rain.shape:
        raise ValueError(f"NDVI {ndvi.shape} and rainfall {rain.shape} must "
                         "have identical shapes")

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        # Pixels with no (or too few) valid observations legitimately produce
        # NaN summaries; they are gated upstream by `quality.assess`, so the
        # empty-slice warnings they raise are noise, not a defect.
        warnings.simplefilter("ignore", RuntimeWarning)
        base, ex = build_features(ndvi, rain, cfg)
        rec = ex["recovery"]
        n_valid = np.isfinite(ndvi).sum(axis=0)
        n_pairs = (np.isfinite(ndvi) & np.isfinite(rain)).sum(axis=0)
        rain_mean = np.nanmean(rain, axis=0)
        rain_std = np.nanstd(rain, axis=0)
        extra = {
            # vegetation
            "median_ndvi": np.nanmedian(ndvi, axis=0),
            "minimum_ndvi": np.nanmin(ndvi, axis=0),
            "maximum_ndvi": np.nanmax(ndvi, axis=0),
            "ols_slope": _ols_slope(ndvi),
            # trend
            "mk_p_value": np.asarray(ex["mk_p"], dtype="float64"),
            "mk_tau": np.asarray(ex["mk_tau"], dtype="float64"),
            "mk_significant": (np.asarray(ex["mk_p"], dtype="float64")
                               < cfg.trend.alpha).astype("float64"),
            "trend_direction": np.sign(base["sen"].to_numpy(dtype="float64")),
            # rainfall
            "rain_mean": rain_mean,
            "rain_std": rain_std,
            "rain_cv": np.where(np.abs(rain_mean) > 1e-9,
                                rain_std / rain_mean, np.nan),
            "rain_sen": TS.sens_slope(rain, min_obs=cfg.trend.min_obs),
            "rain_ndvi_correlation": _pearson(ndvi, rain),
            # restrend
            "restrend_p_value": np.asarray(ex["restrend_p"], dtype="float64"),
            "restrend_r2": np.asarray(ex["restrend_r2"], dtype="float64"),
            "restrend_beta": np.asarray(ex["restrend_beta"], dtype="float64"),
            "restrend_valid": np.asarray(ex["restrend_valid"],
                                         dtype=bool).astype("float64"),
            "restrend_significant": (
                (np.asarray(ex["restrend_p"], dtype="float64")
                 < cfg.restrend.alpha)
                & np.asarray(ex["restrend_valid"], dtype=bool)
            ).astype("float64"),
            # cyclicity
            "cyc_band_power": np.asarray(ex["cyc_band_power"],
                                         dtype="float64"),
            "cyc_total_power": np.asarray(ex["cyc_total_power"],
                                          dtype="float64"),
            "cyclicity_periodic": np.asarray(
                ex["cyc_periodic"], dtype=bool).astype("float64"),
            # disturbance
            "breakpoint_index": np.asarray(ex["break_index"],
                                           dtype="float64"),
            "breakpoint_significant": np.asarray(
                ex["break_significant"], dtype=bool).astype("float64"),
            # recovery
            "recovery_status": np.asarray(rec["recovery_status"],
                                          dtype="float64"),
        }

        has_break = base["has_break"].to_numpy(dtype=bool)
        has_dist = base["has_disturbance"].to_numpy(dtype=bool)
        # Pre/post break levels are UNDEFINED without a break; encode the
        # known state with a neutral 0 next to `has_break` (M1 policy).
        extra["pre_breakpoint_level"] = _structural(rec["pre_level"],
                                                    has_break)
        extra["post_breakpoint_minimum"] = _structural(rec["trough_value"],
                                                       has_break)
        extra["recovery_magnitude"] = _structural(rec["recovery_magnitude"],
                                                  has_dist)
        # Duration is censored where the threshold was never reached; 0 with
        # recovery_status != RECOVERED means "not reached", not "immediate".
        extra["recovery_duration"] = _structural(rec["recovery_duration"],
                                                 has_dist)
        extra["n_valid_ndvi"] = n_valid.astype("float64")
        extra["n_valid_pairs"] = n_pairs.astype("float64")

    table = pd.concat([base, pd.DataFrame(extra, index=base.index)], axis=1)
    ordered = feature_names(GROUP_ORDER)
    missing = [c for c in ordered if c not in table.columns]
    if missing:                                    # pragma: no cover - guard
        raise RuntimeError(f"feature specification lists unbuilt columns: "
                           f"{missing}")
    unspecified = [c for c in table.columns if c not in set(ordered)]
    if unspecified:                                # pragma: no cover - guard
        raise RuntimeError(f"undocumented feature columns produced: "
                           f"{unspecified}")
    return table.loc[:, ordered].astype("float32"), ex


# The M1 feature block must remain a strict subset of the M2 table.
assert set(M1_FEATURES).issubset({s.name for s in FEATURE_SPECS}), \
    "M2 feature specification dropped an M1 feature"
