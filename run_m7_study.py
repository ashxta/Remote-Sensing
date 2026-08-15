"""M7: the final real-world research study.

Runs the validated M1-M5 analysis on the REAL Landsat/CHIRPS record acquired
by `run_m7_acquire.py`, and produces the research outputs: data-quality
verification, descriptive analysis, trend, climate-adjusted trend,
cyclicity, disturbance, recovery, integrated trajectories, area statistics,
sensitivity, indicator-disagreement uncertainty, publication maps,
representative temporal profiles, and the integrated results table.

    python run_m7_study.py --config configs/m7_karbi_anglong_final.json

WHAT IS AND IS NOT RUN
----------------------
Every unsupervised and statistical stage runs on real observations.

The SUPERVISED stages (Random Forest, 1D CNN, ablation A-F, supervised
spatial-CV metrics, temporal holdout metrics, error analysis, model
uncertainty) are NOT run and are reported as BLOCKED BY DATA. No independent
reference labels exist for this study area, and the analytical trajectory
classes cannot substitute for them: they are computed from the same features
a classifier would consume, so any accuracy measured against them would
report self-consistency, not detection skill. Fabricating that number is the
single most damaging thing this phase could do, so the pipeline refuses it
and says so in `models/BLOCKED.json`.

INTERPRETATION
--------------
Outputs are evidence of VEGETATION DYNAMICS. A negative trend is not proof
of degradation; periodicity is not proof of shifting cultivation; a residual
decline is not proof of human causation. The wording of every saved finding
observes that distinction.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src import geo
from src import timeseries as TS
from src.config import Config
from src.experiment import prepare_experiment
from src.features import feature_dictionary
from src.real_data import RealRemoteSensingSource
from src.real_report import (MISSINGNESS_CAVEAT, build_quality_report,
                             write_quality_report)
from src.reproducibility import start_experiment
from src.sensitivity import run_sensitivity_analysis
from src.stac_source import CHIRPS, PLANETARY_COMPUTER, SUBSAMPLING_NOTE
from src.study_area import area_statistics, load_study_area, pixel_area_km2
from src.trajectory import (DECLINE_CLASSES, TRAJECTORY_CLASSES,
                            TRAJECTORY_CODES, trajectory_codes)

TREE = ("configuration", "data_quality", "temporal_analysis", "features",
        "models", "validation", "ablation", "sensitivity", "uncertainty",
        "maps", "figures", "tables", "logs", "summary")

SOURCE_LINE = (
    "Data: USGS Landsat Collection 2 Level-2 surface reflectance "
    "(Landsat 5/7/8/9) via Microsoft Planetary Computer; rainfall: CHIRPS "
    "v2.0 annual. Analysis: this repository, M1-M7.")

BLOCKED = {
    "status": "NOT SCIENTIFICALLY VALID / BLOCKED BY DATA",
    "blocked_experiments": [
        "Part 11 - Random Forest", "Part 12 - 1D CNN",
        "Part 13 - supervised spatial cross-validation metrics",
        "Part 14 - temporal generalization metrics",
        "Part 15 - ablation study A-F",
        "Part 17 - supervised baseline comparison",
        "Part 18 - supervised error analysis",
        "Part 19 - model-probability uncertainty"],
    "why": (
        "These experiments require reference labels that are independent of "
        "the features. Satellite imagery does not supply land-degradation "
        "labels, and no field survey, expert interpretation or published "
        "degradation dataset is available for this study area."),
    "why_the_trajectory_classes_cannot_be_used": (
        "The analytical trajectory classes are produced by "
        "trajectory.classify_trajectories from the Mann-Kendall p-value, the "
        "Sen slope, the RESTREND residual trend and its validity flag, the "
        "spectral enrichment, the breakpoint and the recovery fraction - the "
        "same quantities that make up the classifier's feature table. A "
        "model trained on them would be scored on reproducing a "
        "deterministic rule from that rule's own inputs. It would report a "
        "near-perfect accuracy that measures nothing about degradation. "
        "This is label leakage by construction."),
    "what_would_unblock_it": [
        "Field observations of degradation status at located points.",
        "Expert photo-interpretation of high-resolution imagery at a "
        "stratified pixel sample, by an interpreter blind to these outputs.",
        "A published shifting-cultivation or degradation dataset with a "
        "compatible scheme, resolution and vintage.",
        "An authoritative land-cover product for the confounder classes, "
        "with its scheme explicitly mapped onto this study's classes."],
    "what_was_run_instead": (
        "Every unsupervised and statistical stage: quality gating, "
        "Mann-Kendall with Hamed-Rao autocorrelation adjustment, Sen's "
        "slope, RESTREND, spectral cyclicity with AR(1) surrogate "
        "significance, Chow breakpoint detection, recovery descriptors, the "
        "integrated trajectory classification, area statistics, parameter "
        "sensitivity, and an indicator-disagreement uncertainty analysis."),
}


# ---------------------------------------------------------------------------
def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def _grid_of(prepared, values, fill=np.nan) -> np.ndarray:
    grid = np.full(prepared.shape, fill, dtype="float64")
    grid[prepared.analysis_grid] = np.asarray(values, dtype="float64")
    return grid


# ---------------------------------------------------------------------------
def stage_dataset(cfg, area, exp, log):
    """Load, validate and describe the real dataset (Parts 1-2)."""
    source = RealRemoteSensingSource(cfg, study_area=area, logger=log)
    prepared = prepare_experiment(cfg, source=source, logger=log)
    dataset = prepared.dataset
    if dataset.metadata.get("synthetic", False):
        raise SystemExit(
            "REFUSING TO PROCEED: the configured cubes are marked SYNTHETIC. "
            "M7 produces real-world research findings and must not be run on "
            "fixture data. Run run_m7_acquire.py first.")

    root = exp.path("configuration")
    area.save(root / "study_area.geojson")
    prepared.georef.save(root / "georeference.json")
    _write(root / "dataset_metadata.json", prepared.validation_report)
    _write(root / "dataset_description.json", dataset.describe())
    _write(root / "frozen_config.json", cfg.to_dict())
    _write(root / "research_config.json", asdict(cfg.research))
    _write(root / "real_data_config.json", asdict(cfg.real_data))
    acquisition = Path(cfg.real_data.metadata_dir) / "m7_acquisition.json"
    if acquisition.exists():
        _write(root / "acquisition.json", json.loads(acquisition.read_text()))
    return prepared


def stage_quality(prepared, cfg, exp, log, *, area=None):
    """Data-quality verification and report (Parts 2, 14)."""
    dataset = prepared.dataset
    # Restrict every statistic to the study-area polygon. Outside it the
    # cube is NaN by construction, and counting that as missing data would
    # misreport a ~4% record as ~52% missing.
    inside = None
    if area is not None:
        try:
            inside = area.mask(dataset.georef)
        except Exception as error:                       # pragma: no cover
            log.warning("could not build the boundary mask: %s", error)
    report = build_quality_report(dataset, cfg, inside=inside)
    if inside is not None:
        report["_inside_mask"] = inside
    written = write_quality_report(
        {k: v for k, v in report.items() if not k.startswith("_")},
        exp.path("data_quality"))
    _write(exp.path("data_quality") / "quality_gate.json",
           prepared.quality_summary)

    problems = []
    if np.isinf(dataset.ndvi).any() or np.isinf(dataset.rain).any():
        problems.append("infinite values present")
    if report["vegetation"]["min"] is not None and (
            report["vegetation"]["min"] < -1.0
            or report["vegetation"]["max"] > 1.0):
        problems.append("NDVI outside the physical range")
    if report["rainfall"]["min"] is not None and report["rainfall"]["min"] < 0:
        problems.append("negative rainfall")
    if report["temporal"]["steps_with_no_valid_observation"]:
        problems.append("time steps with no valid observation anywhere")
    _write(exp.path("data_quality") / "verification.json",
           {"checks_failed": problems,
            "passed": not problems,
            "temporal_alignment": dataset.metadata["temporal_alignment"],
            "boundary_clipping": dataset.metadata["boundary_clipping"],
            "caveat": MISSINGNESS_CAVEAT})

    log.info("data quality: NDVI %.3f..%.3f mean %.3f, %.1f%% missing; "
             "rainfall %.0f..%.0f mm", report["vegetation"]["min"],
             report["vegetation"]["max"], report["vegetation"]["mean"],
             100 * report["vegetation"]["missing_fraction"],
             report["rainfall"]["min"], report["rainfall"]["max"])
    if problems:
        log.warning("data-quality problems: %s", problems)
    return report, written


def stage_descriptive(prepared, exp, log):
    """Exploratory vegetation description (Part 4). Not degradation evidence."""
    dataset = prepared.dataset
    times = [str(t) for t in dataset.times]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_ndvi = np.nanmean(dataset.ndvi, axis=0)
        std_ndvi = np.nanstd(dataset.ndvi, axis=0)
        annual_mean = np.nanmean(dataset.ndvi.reshape(dataset.n_time, -1),
                                 axis=1)
        annual_p10 = np.nanpercentile(
            dataset.ndvi.reshape(dataset.n_time, -1), 10, axis=1)
        annual_p90 = np.nanpercentile(
            dataset.ndvi.reshape(dataset.n_time, -1), 90, axis=1)
        annual_rain = np.nanmean(dataset.rain.reshape(dataset.n_time, -1),
                                 axis=1)
        valid = np.isfinite(dataset.ndvi).sum(axis=0).astype("float64")

    table = pd.DataFrame({
        "time_step": times,
        "ndvi_mean": annual_mean, "ndvi_p10": annual_p10,
        "ndvi_p90": annual_p90,
        "ndvi_valid_fraction": np.isfinite(
            dataset.ndvi.reshape(dataset.n_time, -1)).mean(axis=1),
        "rainfall_mean_mm": annual_rain})
    table.to_csv(exp.path("tables") / "annual_summary.csv", index=False)

    geo.write_raster(exp.path("temporal_analysis") / "ndvi_mean.tif",
                     mean_ndvi, prepared.georef, dtype="float32",
                     description="Mean NDVI over the record")
    geo.write_raster(exp.path("temporal_analysis") / "ndvi_std.tif",
                     std_ndvi, prepared.georef, dtype="float32",
                     description="Temporal standard deviation of NDVI")
    geo.write_raster(exp.path("temporal_analysis") / "valid_observations.tif",
                     valid, prepared.georef, dtype="float32",
                     description="Count of valid NDVI composites per pixel")

    log.info("descriptive: mean NDVI %.3f (spatial sd %.3f); annual mean "
             "ranges %.3f-%.3f", float(np.nanmean(mean_ndvi)),
             float(np.nanstd(mean_ndvi)), float(np.nanmin(annual_mean)),
             float(np.nanmax(annual_mean)))
    return {"mean_ndvi": mean_ndvi, "std_ndvi": std_ndvi, "valid": valid,
            "annual": table}


def stage_temporal_layers(prepared, exp, log):
    """Write every estimator's output as a georeferenced layer (Parts 5-9)."""
    features = prepared.features
    layers = (
        ("sens_slope", "sen", "Theil-Sen NDVI slope per year"),
        ("mann_kendall_z", "mk_z", "Mann-Kendall Z, Hamed-Rao adjusted"),
        ("mann_kendall_p", "mk_p_value", "Mann-Kendall p-value"),
        ("trend_significant", "mk_significant",
         "1 where the Mann-Kendall test is significant at the configured alpha"),
        ("restrend_slope", "restrend", "Climate-adjusted (residual) NDVI slope"),
        ("restrend_p", "restrend_p_value", "Residual-trend p-value"),
        ("restrend_valid", "restrend_valid",
         "1 where the NDVI~rainfall relation is strong enough to adjust"),
        ("cyclicity_enrichment", "cyc_enrichment",
         "Spectral band-power enrichment; 1.0 = white noise"),
        ("dominant_period", "cyc_period", "Dominant period in years"),
        ("cyclicity_periodic", "cyclicity_periodic",
         "1 where the enrichment threshold is met"),
        ("break_index", "breakpoint_index", "Structural break index (-1 none)"),
        ("break_significant", "breakpoint_significant",
         "1 where the selection-adjusted Chow test is significant"),
        ("disturbance_magnitude", "disturbance_magnitude",
         "Pre-break level minus post-break trough, NDVI units"),
        ("recovery_fraction", "recovery_fraction",
         "Share of the disturbance magnitude regained"),
        ("recovery_slope", "recovery_slope", "Post-trough NDVI slope per year"),
        ("recovery_status", "recovery_status",
         "0 none 1 recovered 2 recovering 3 not-recovering 4 insufficient"),
        ("n_valid_ndvi", "n_valid_ndvi", "Valid observations used per pixel"),
    )
    written = 0
    for name, column, description in layers:
        if column not in features.columns:
            log.warning("feature column %s missing; layer skipped", column)
            continue
        geo.write_layer(exp.path("temporal_analysis") / f"{name}.tif",
                        features[column].to_numpy(), prepared.analysis_grid,
                        prepared.georef, dtype="float32",
                        description=description)
        written += 1
    geo.write_layer(exp.path("validation") / "spatial_cv_fold.tif",
                    prepared.folds, prepared.analysis_grid, prepared.georef,
                    dtype="uint8",
                    description="Deterministic spatial block CV fold layout")
    geo.write_raster(exp.path("data_quality") / "quality_flag.tif",
                     prepared.quality.flag.astype("float64").reshape(
                         prepared.shape),
                     prepared.georef, dtype="uint8",
                     description="0 OK 1 insufficient 2 too-missing "
                                 "3 constant 4 out-of-range")
    log.info("wrote %d georeferenced analysis layers", written)
    return written


def stage_features(prepared, cfg, exp, log):
    """The engineered feature matrix and its quality (Part 20).

    The matrix is written gzip-compressed: 200,000 rows x 48 columns is
    ~100 MB as plain text, and a result directory that large is one nobody
    keeps. Every column is also on disk as a georeferenced layer in
    `temporal_analysis/`, so the CSV is the tabular convenience copy, not the
    only record.
    """
    root = exp.path("features")
    features = prepared.features
    features.to_csv(root / "feature_matrix.csv.gz", index=False,
                    compression="gzip")

    quality = features.describe().T
    quality["missing_fraction"] = features.isna().mean()
    quality["n_finite"] = features.notna().sum()
    quality.to_csv(root / "feature_quality.csv")

    _write(root / "feature_metadata.json", {
        "n_rows": int(len(features)),
        "n_columns": int(features.shape[1]),
        "modelling_columns": list(prepared.model_columns),
        "n_modelling_features": len(prepared.model_columns),
        "feature_groups": list(cfg.research.features.groups),
        "dictionary": feature_dictionary().to_dict(orient="records"),
        "data_status": "REAL remote-sensing observations",
        "note": ("Feature definitions are unchanged from M2-M5. No feature "
                 "was added, removed or re-tuned for the real data."),
    })
    exp.path("tables")  # ensure it exists before the summary write
    log.info("feature matrix: %d rows x %d columns (%d modelling features); "
             "mean missingness %.3f", len(features), features.shape[1],
             len(prepared.model_columns),
             float(features.isna().mean().mean()))
    return {"n_rows": int(len(features)),
            "n_columns": int(features.shape[1]),
            "n_modelling_features": len(prepared.model_columns),
            "mean_missing_fraction": float(features.isna().mean().mean())}


def stage_trend(prepared, cfg, exp, log):
    """Long-term trend results and areas (Part 5)."""
    features = prepared.features
    alpha = cfg.trend.alpha
    slope = features["sen"].to_numpy()
    p_value = features["mk_p_value"].to_numpy()
    significant = np.isfinite(p_value) & (p_value < alpha)
    areas = pixel_area_km2(prepared.georef)[prepared.analysis_grid]

    groups = {
        "significant_increase": significant & (slope > 0),
        "significant_decrease": significant & (slope < 0),
        "no_significant_trend": ~significant,
    }
    rows = [{"class": name, "n_pixels": int(mask.sum()),
             "area_km2": float(areas[mask].sum()),
             "fraction_of_analysed": float(mask.mean()),
             "median_sen_slope_per_year": float(np.nanmedian(slope[mask]))
             if mask.any() else float("nan")}
            for name, mask in groups.items()]
    table = pd.DataFrame(rows)
    table.to_csv(exp.path("tables") / "trend_areas.csv", index=False)

    summary = {
        "alpha": alpha,
        "autocorrelation_correction": cfg.trend.apply_autocorrelation_correction,
        "n_analysed": int(len(slope)),
        "analysed_area_km2": float(areas.sum()),
        "median_sen_slope_per_year": float(np.nanmedian(slope)),
        "mean_sen_slope_per_year": float(np.nanmean(slope)),
        "significant_fraction": float(significant.mean()),
        "areas": rows,
        "interpretation": (
            "A significant negative NDVI trend is a persistent decline in "
            "the vegetation index. It is NOT by itself evidence of land "
            "degradation: rainfall variability, species composition change, "
            "cropping change and sensor effects all produce trends. The "
            "climate-adjusted analysis below separates the part of the "
            "decline the modelled rainfall relationship explains."),
    }
    _write(exp.path("summary") / "trend.json", summary)
    log.info("trend: %.1f%% significant | decline %.0f km2, increase %.0f km2, "
             "no trend %.0f km2", 100 * significant.mean(),
             rows[1]["area_km2"], rows[0]["area_km2"], rows[2]["area_km2"])
    return summary


def stage_restrend(prepared, cfg, exp, log):
    """Observed vs climate-adjusted trend (Part 6)."""
    features = prepared.features
    alpha = cfg.trend.alpha
    slope = features["sen"].to_numpy()
    p_value = features["mk_p_value"].to_numpy()
    declining = np.isfinite(p_value) & (p_value < alpha) & (slope < 0)
    increasing = np.isfinite(p_value) & (p_value < alpha) & (slope > 0)
    valid = features["restrend_valid"].to_numpy(bool)
    residual = features["restrend"].to_numpy()
    residual_p = features["restrend_p_value"].to_numpy()
    residual_declining = (np.isfinite(residual_p) & (residual_p < alpha)
                          & np.isfinite(residual) & (residual < 0))
    areas = pixel_area_km2(prepared.georef)[prepared.analysis_grid]

    categories = {
        "decline_persists_after_climate_adjustment":
            declining & valid & residual_declining,
        "decline_largely_explained_by_rainfall":
            declining & valid & ~residual_declining,
        "decline_not_adjustable_weak_rainfall_relation": declining & ~valid,
        "stable_or_no_significant_trend": ~declining & ~increasing,
        "significant_increase": increasing,
    }
    rows = [{"category": name, "n_pixels": int(mask.sum()),
             "area_km2": float(areas[mask].sum()),
             "fraction_of_analysed": float(mask.mean())}
            for name, mask in categories.items()]
    pd.DataFrame(rows).to_csv(exp.path("tables") / "restrend_categories.csv",
                              index=False)

    explained = int((declining & valid & ~residual_declining).sum())
    total_declining = int(declining.sum())

    # How strong is the NDVI~rainfall relation at all? RESTREND assumes
    # vegetation is moisture-limited. Where it is not, the regression has no
    # explanatory power and the residual trend is simply the raw trend
    # again. Measuring this directly - rather than only counting how often
    # the validity gate fired - is what turns "the gate rejected most
    # pixels" into a statement about the landscape.
    series = prepared.series
    rain_series = prepared.rain_series
    finite = np.isfinite(series) & np.isfinite(rain_series)
    with np.errstate(invalid="ignore", divide="ignore"):
        ndvi_centred = np.where(finite, series, np.nan)
        rain_centred = np.where(finite, rain_series, np.nan)
        ndvi_centred = ndvi_centred - np.nanmean(ndvi_centred, axis=0)
        rain_centred = rain_centred - np.nanmean(rain_centred, axis=0)
        numerator = np.nansum(ndvi_centred * rain_centred, axis=0)
        denominator = np.sqrt(np.nansum(ndvi_centred ** 2, axis=0)
                              * np.nansum(rain_centred ** 2, axis=0))
        correlation = np.where(denominator > 0, numerator / denominator,
                               np.nan)
    positive = float(np.nanmean(correlation > 0))
    applicable = bool(valid.mean() >= 0.25)

    # WHY each excluded pixel was excluded. "Only 2.5% valid" is not a
    # finding until the reader knows which criterion did the rejecting -
    # a weak fit and a wrong-signed sensitivity mean different things.
    r_squared = features["restrend_r2"].to_numpy()
    beta = features["restrend_beta"].to_numpy()
    n_valid_obs = features["n_valid_ndvi"].to_numpy()
    fails_r2 = np.isfinite(r_squared) & (r_squared < cfg.restrend.min_r2)
    fails_beta = (np.isfinite(beta) & (beta <= 0)
                  if cfg.restrend.require_positive_beta
                  else np.zeros(len(beta), bool))
    fails_obs = n_valid_obs < cfg.restrend.min_obs
    not_finite = ~np.isfinite(r_squared)
    exclusions = {
        "insufficient_valid_observations": {
            "n": int((~valid & fails_obs).sum()),
            "criterion": f"n_valid_ndvi < restrend.min_obs "
                         f"({cfg.restrend.min_obs})"},
        "rainfall_relation_too_weak": {
            "n": int((~valid & fails_r2 & ~fails_obs).sum()),
            "criterion": f"partial r2 < restrend.min_r2 "
                         f"({cfg.restrend.min_r2})"},
        "rainfall_sensitivity_not_positive": {
            "n": int((~valid & fails_beta & ~fails_r2 & ~fails_obs).sum()),
            "criterion": "NDVI decreases with rainfall, so the regression "
                         "cannot be read as a moisture-limitation response"},
        "regression_not_estimable": {
            "n": int((~valid & not_finite).sum()),
            "criterion": "the fit produced no finite r2"},
    }
    valid_area = float(areas[valid].sum())
    valid_subset = {
        "n_pixels": int(valid.sum()),
        "area_km2": valid_area,
        "median_r2": float(np.nanmedian(r_squared[valid]))
        if valid.any() else float("nan"),
        "median_rainfall_sensitivity": float(np.nanmedian(beta[valid]))
        if valid.any() else float("nan"),
        "n_with_significant_residual_decline": int(
            (valid & residual_declining).sum()),
        "area_with_significant_residual_decline_km2": float(
            areas[valid & residual_declining].sum()),
        "what_it_indicates": (
            "Within this subset - and only within it - the residual trend is "
            "interpretable as a climate-adjusted trend, because the "
            "NDVI~rainfall regression that defines the adjustment is itself "
            "meaningful there. Everywhere else the residual is numerically "
            "computable but scientifically empty, and is reported as such "
            "rather than mapped."),
    }

    summary = {
        "restrend_valid_fraction": float(valid.mean()),
        "restrend_valid_area_km2": valid_area,
        "restrend_excluded_area_km2": float(areas[~valid].sum()),
        "why_pixels_were_excluded": exclusions,
        "what_the_valid_subset_shows": valid_subset,
        "limits_of_a_rainfall_only_adjustment": (
            "RESTREND removes only the variance a LINEAR ANNUAL RAINFALL "
            "model explains. It does not account for temperature, vapour "
            "pressure deficit, incoming radiation, soil moisture storage, "
            "rainfall timing and intensity within the year, antecedent "
            "conditions beyond the accumulation window, CO2 fertilisation, "
            "or nutrient limitation. A residual trend is therefore 'not "
            "explained by the modelled rainfall relationship', which is a "
            "much weaker statement than 'not explained by climate' and far "
            "weaker than 'human-caused'. In a humid landscape where water is "
            "not the limiting factor, most of the climate signal that "
            "matters is in variables this adjustment never sees."),
        "applicability": {
            "restrend_is_broadly_applicable_here": applicable,
            "median_ndvi_rainfall_correlation": float(
                np.nanmedian(correlation)),
            "fraction_of_pixels_with_positive_correlation": positive,
            "assessment": (
                "RESTREND assumes vegetation productivity is limited by "
                "rainfall, which is why it was developed for drylands. In "
                "this study area the annual NDVI~rainfall correlation is "
                f"centred near zero (median r = "
                f"{float(np.nanmedian(correlation)):+.3f}) and only "
                f"{positive:.1%} of pixels show a positive relation, so the "
                "validity criteria are met on only "
                f"{float(valid.mean()):.1%} of pixels. This is consistent "
                "with a humid, densely vegetated landscape where canopy "
                "greenness is not moisture-limited and NDVI saturates. "
                "It is a property of the landscape, not a failure of the "
                "computation: the validity gate exists precisely to stop a "
                "meaningless residual being reported as a climate-adjusted "
                "trend."
                if not applicable else
                "The NDVI~rainfall relation is strong enough across the "
                "study area for the climate adjustment to be broadly "
                "interpretable."),
            "consequence_for_the_trajectory_classes": (
                "Because the adjustment is unavailable on most pixels, the "
                "'Degrading' class here is dominated by declines that could "
                "NOT be climate-adjusted rather than by declines shown to "
                "survive adjustment. That is a weaker statement and must be "
                "reported as such."),
        },
        "min_r2_required": cfg.restrend.min_r2,
        "require_positive_beta": cfg.restrend.require_positive_beta,
        "n_significant_decline": total_declining,
        "n_decline_with_valid_adjustment": int((declining & valid).sum()),
        "n_decline_explained_by_rainfall": explained,
        "share_of_declines_reclassified_as_rainfall_associated": (
            explained / total_declining if total_declining else float("nan")),
        "categories": rows,
        "interpretation": (
            "Where the NDVI~rainfall regression is strong enough to be "
            "meaningful and the residual trend is no longer significantly "
            "negative, the observed decline is largely consistent with the "
            "modelled rainfall variability. Where the residual decline "
            "persists, the decline is NOT explained by the modelled rainfall "
            "relationship. That is not proof of human causation: "
            "temperature, soil, fire, species change and the limits of a "
            "linear rainfall model can all leave a residual trend."),
        "validity_caveat": (
            "Where restrend_valid is 0 the rainfall relation is too weak for "
            "the residual to be interpreted as climate-adjusted at all. "
            "Those pixels are reported separately rather than pooled with "
            "either outcome."),
    }
    _write(exp.path("summary") / "restrend.json", summary)
    log.info("RESTREND: valid on %.1f%% of pixels | of %d significant "
             "declines, %d (%.1f%%) are largely rainfall-explained",
             100 * valid.mean(), total_declining, explained,
             100 * explained / total_declining if total_declining else 0.0)
    if not applicable:
        log.warning("RESTREND IS LARGELY INAPPLICABLE HERE: median NDVI~"
                    "rainfall r = %+.3f, only %.1f%% of pixels positive. "
                    "The 'Degrading' class is therefore dominated by "
                    "declines that could not be climate-adjusted.",
                    float(np.nanmedian(correlation)), 100 * positive)
    return summary


def stage_cyclicity(prepared, cfg, exp, log):
    """Cyclicity with surrogate significance (Part 7)."""
    features = prepared.features
    enrichment = features["cyc_enrichment"].to_numpy()
    period = features["cyc_period"].to_numpy()
    periodic = features["cyclicity_periodic"].to_numpy(bool)
    areas = pixel_area_km2(prepared.georef)[prepared.analysis_grid]

    rng = np.random.default_rng(cfg.seed)
    n = int(prepared.analysis_mask.sum())
    take = min(n, 4000)
    columns = np.sort(rng.choice(n, take, replace=False))
    surrogate = TS.cyclicity_significance(
        prepared.series[:, columns], min_period=cfg.cyclicity.min_period,
        max_period=cfg.cyclicity.max_period, detrend=cfg.cyclicity.detrend,
        min_obs=cfg.cyclicity.min_obs, n_surrogates=cfg.cyclicity.n_surrogates,
        alpha=cfg.cyclicity.surrogate_alpha, seed=cfg.cyclicity.surrogate_seed)
    pd.DataFrame({
        "p_value": surrogate["p_value"],
        "significant": surrogate["significant"],
        "enrichment": enrichment[columns],
        "dominant_period": period[columns],
        "threshold_rule_flag": periodic[columns],
    }).to_csv(exp.path("tables") / "cyclicity_surrogate.csv", index=False)

    finite_period = period[periodic & np.isfinite(period)]
    summary = {
        "period_band_years": [cfg.cyclicity.min_period,
                              cfg.cyclicity.max_period],
        "enrichment_threshold": cfg.cyclicity.periodicity_threshold,
        "periodic_fraction_threshold_rule": float(periodic.mean()),
        "periodic_area_km2": float(areas[periodic].sum()),
        "median_enrichment": float(np.nanmedian(enrichment)),
        "dominant_period_median_years": float(np.median(finite_period))
        if finite_period.size else float("nan"),
        "surrogate_test": {
            "null_model": surrogate["null"],
            "n_surrogates": surrogate["n_surrogates"],
            "n_pixels_tested": int(take),
            "significant_fraction": float(np.mean(surrogate["significant"])),
            "agreement_with_threshold_rule": float(
                np.mean(surrogate["significant"] == periodic[columns])),
            "median_p_value": float(np.nanmedian(surrogate["p_value"])),
        },
        "interpretation": (
            "A small surrogate p-value means the concentration of spectral "
            "power inside the 4-12 year band is unlikely for a series with "
            "this pixel's own autocorrelation. It identifies RECURRENT "
            "TEMPORAL BEHAVIOUR. It does not identify a cause. Rotational "
            "cultivation, plantation harvest cycles, fire-regrowth cycles "
            "and multi-year climate oscillation all produce periodicity in "
            "this band. PERIODICITY IS NOT JHUM. Attribution to shifting "
            "cultivation requires independent ground or ancillary evidence, "
            "which this study does not have."),
        "detectability_limit": {
            "headline": (
                "THE LOW CYCLIC FRACTION IS NOT EVIDENCE THAT ROTATIONAL "
                "CULTIVATION IS ABSENT. It is consistent with the phenomenon "
                "being below the resolution of this analysis."),
            "spatial": (
                f"One analysis cell is "
                f"{cfg.real_data.target_resolution_m:.0f} m on a side "
                f"({(cfg.real_data.target_resolution_m ** 2) / 1e4:.0f} ha) "
                f"and averages the valid 30 m pixels inside it. Individual "
                f"shifting-cultivation plots in this region are typically a "
                f"small fraction of that area, so a plot on its own cycle is "
                f"averaged together with surrounding land on a different "
                f"phase or none at all. Averaging cancels out-of-phase "
                f"cycles, which suppresses band power - the statistic the "
                f"detection rests on. Resolving field-scale rotation "
                f"requires the native 30 m grid."),
            "temporal": (
                f"With {prepared.dataset.n_time} annual steps, a cycle must "
                f"repeat at least twice to be distinguishable from a trend, "
                f"so periods above about "
                f"{prepared.dataset.n_time / 2:.0f} years are not "
                f"detectable, and the configured upper edge of "
                f"{cfg.cyclicity.max_period:.0f} years sits comfortably "
                f"inside that. The annual step means sub-annual and "
                f"single-season regrowth signals are invisible by "
                f"construction."),
            "observational": (
                f"{int((features['n_valid_ndvi'].to_numpy() < cfg.cyclicity.min_obs).sum()):,} "
                f"analysed pixels have fewer than the "
                f"{cfg.cyclicity.min_obs} valid observations the periodicity "
                f"estimator requires and cannot be tested at all. Gaps also "
                f"broaden the spectrum, which lowers measured enrichment and "
                f"biases this statistic toward FALSE NEGATIVES rather than "
                f"false positives."),
            "what_may_be_concluded": (
                "'Recurrent vegetation dynamics were detected on X km2 at "
                "this scale' is supportable. 'Shifting cultivation was "
                "confirmed' is not, and neither is 'shifting cultivation is "
                "rare here' - this analysis cannot see the scale at which "
                "the practice operates."),
            "thresholds_unchanged": (
                f"The enrichment threshold remains at the M1-M5 default of "
                f"{cfg.cyclicity.periodicity_threshold} and the band at "
                f"{cfg.cyclicity.min_period:.0f}-"
                f"{cfg.cyclicity.max_period:.0f} years. Neither was adjusted "
                f"to increase the detected area; the sensitivity sweep "
                f"reports what other values would have produced."),
        },
    }
    _write(exp.path("summary") / "cyclicity.json", summary)
    log.info("cyclicity: %.1f%% periodic by threshold, %.1f%% significant "
             "under the AR(1) surrogate null, agreement %.1f%%",
             100 * periodic.mean(),
             100 * summary["surrogate_test"]["significant_fraction"],
             100 * summary["surrogate_test"]["agreement_with_threshold_rule"])
    return summary


def stage_disturbance_recovery(prepared, cfg, exp, log):
    """Breakpoint, disturbance and recovery results (Parts 8-9)."""
    features = prepared.features
    dataset = prepared.dataset
    times = [str(t) for t in dataset.times]
    break_index = features["breakpoint_index"].to_numpy()
    significant = features["breakpoint_significant"].to_numpy(bool)
    magnitude = features["disturbance_magnitude"].to_numpy()
    fraction = features["recovery_fraction"].to_numpy()
    status = features["recovery_status"].to_numpy()
    areas = pixel_area_km2(prepared.georef)[prepared.analysis_grid]

    real = (significant
            & np.isfinite(magnitude)
            & (magnitude >= cfg.recovery.min_disturbance_magnitude))
    years = []
    for index in break_index[real]:
        value = int(index)
        years.append(times[value] if 0 <= value < len(times) else "NA")
    year_counts = pd.Series(years).value_counts().sort_index()
    year_counts.to_csv(exp.path("tables") / "breakpoint_years.csv",
                       header=["n_pixels"])

    status_names = {0: "no disturbance", 1: "recovered", 2: "recovering",
                    3: "not recovering", 4: "insufficient post-observations"}
    rows = [{"recovery_status": status_names.get(int(code), str(code)),
             "n_pixels": int((status == code).sum()),
             "area_km2": float(areas[status == code].sum())}
            for code in sorted(np.unique(status[np.isfinite(status)]))]
    pd.DataFrame(rows).to_csv(exp.path("tables") / "recovery_status.csv",
                              index=False)

    summary = {
        "min_disturbance_magnitude": cfg.recovery.min_disturbance_magnitude,
        "recovery_threshold": cfg.recovery.recovery_threshold,
        "chow_test_applied": cfg.breakpoint.apply_significance_test,
        "n_breakpoints_detected": int(np.isfinite(break_index).sum()
                                      - int((break_index < 0).sum())),
        "n_significant_disturbances": int(real.sum()),
        "disturbed_area_km2": float(areas[real].sum()),
        "disturbed_fraction": float(real.mean()),
        "median_disturbance_magnitude_ndvi": float(
            np.nanmedian(magnitude[real])) if real.any() else float("nan"),
        "median_recovery_fraction": float(np.nanmedian(fraction[real]))
        if real.any() else float("nan"),
        "breakpoint_year_counts": {str(k): int(v)
                                   for k, v in year_counts.items()},
        "recovery_status_areas": rows,
        "interpretation": (
            "A breakpoint is an abrupt, statistically supported change in "
            "the NDVI series. Causes include land-use change, fire, extreme "
            "weather, harvest, disease, and data artefacts such as an "
            "unharmonised sensor transition or a change in observation "
            "density. This study harmonises across sensors and reports "
            "observation counts, but cannot attribute an individual "
            "breakpoint without independent evidence."),
    }
    _write(exp.path("summary") / "disturbance_recovery.json", summary)
    log.info("disturbance: %d significant disturbances over %.0f km2; median "
             "magnitude %.3f NDVI, median recovery fraction %.2f",
             int(real.sum()), summary["disturbed_area_km2"],
             summary["median_disturbance_magnitude_ndvi"],
             summary["median_recovery_fraction"])
    return summary


def stage_sensor_confound(prepared, cfg, exp, log):
    """Could the instrument change be manufacturing the trend? (Part 18)

    The record runs Landsat 5 -> 7 -> 8/9, and the OLI instruments enter in
    2013. Roy et al. (2016) harmonisation is applied, but if a residual
    offset survives it appears as a STEP at a known date - and a step inside
    a monotonic trend test reads as a trend. This is the most serious
    systematic threat to a multi-sensor trend result, so it is measured
    rather than assumed away, and the answer is reported next to the trend
    itself.
    """
    dataset = prepared.dataset
    manifest = Path(cfg.real_data.raw_dir) / "scenes.json"
    if not manifest.exists():
        return {"assessed": False,
                "reason": "scene manifest not available in this run"}
    scenes = json.loads(manifest.read_text()).get("scenes", [])
    by_year: dict = {}
    for scene in scenes:
        by_year.setdefault(int(scene["date"][:4]), []).append(scene["sensor"])

    years = [int(str(t)) for t in dataset.times]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        annual = np.nanmean(dataset.ndvi.reshape(dataset.n_time, -1), axis=1)

    oli = {"LANDSAT8_OLI", "LANDSAT9_OLI2"}
    oli_years = sorted(y for y, mix in by_year.items()
                       if oli.intersection(mix))
    if not oli_years:
        return {"assessed": False, "reason": "no OLI scenes in the record"}
    first_oli = int(oli_years[0])

    pre = np.array([annual[i] for i, y in enumerate(years) if y < first_oli])
    post = np.array([annual[i] for i, y in enumerate(years) if y >= first_oli])
    pre = pre[np.isfinite(pre)]
    post = post[np.isfinite(post)]
    if pre.size < 3 or post.size < 3:
        return {"assessed": False, "reason": "too few years either side"}

    step = float(post.mean() - pre.mean())
    error = float(np.sqrt(pre.var(ddof=1) / pre.size
                          + post.var(ddof=1) / post.size))
    t_statistic = float(step / error) if error > 0 else float("nan")
    # If the whole step were mistaken for a linear trend, this is roughly the
    # per-year slope it would contribute across the record.
    equivalent = float(step / (len(years) / 2.0))
    median_slope = float(np.nanmedian(prepared.features["sen"].to_numpy()))
    ratio = (abs(equivalent / median_slope)
             if median_slope not in (0.0,) and np.isfinite(median_slope)
             else float("nan"))

    # The paired cross-sensor measurement, when it exists, is far stronger
    # evidence than this annual-mean step: a step confounds the instrument
    # change with real vegetation change, whereas near-simultaneous pairs
    # isolate the instrument. If the two point in OPPOSITE directions, the
    # sensor cannot be manufacturing the step.
    paired = {}
    for candidate in ("sensor_harmonisation_check.json",
                      "sensor_harmonisation_check_original.json"):
        path = Path(cfg.real_data.metadata_dir) / candidate
        if path.exists():
            paired = json.loads(path.read_text())
            break
    residual = None
    if paired.get("with_harmonisation", {}).get("n_pairs"):
        residual = float(paired["with_harmonisation"][
            "pooled_median_oli_minus_etm"])

    summary = {
        "assessed": True,
        "first_year_with_oli": first_oli,
        "pre_transition_mean_ndvi": float(pre.mean()),
        "post_transition_mean_ndvi": float(post.mean()),
        "step_ndvi": step,
        "welch_t": t_statistic,
        "step_is_statistically_detectable": bool(abs(t_statistic) > 2.0),
        "trend_equivalent_of_the_step_per_year": equivalent,
        "median_sen_slope_per_year": median_slope,
        "step_equivalent_over_median_slope": ratio,
        "harmonisation_applied": (
            "Roy et al. (2016) OLS transform, OLI NDVI -> ETM+ NDVI, applied "
            "per scene before compositing"),
        "paired_cross_sensor_residual_ndvi": residual,
        "paired_measurement": paired.get("verdict"),
        "assessment": (
            f"The annual mean is {step:+.4f} NDVI higher after the OLI "
            f"transition ({first_oli}) than before it (Welch "
            f"t = {t_statistic:.2f}); spread across the record that is "
            f"equivalent to {equivalent:+.5f} NDVI/yr, or {ratio:.1f}x the "
            f"median per-pixel Sen slope of {median_slope:+.5f} NDVI/yr. "
            + ("A step of that size relative to the trend would, on its own, "
               "leave the regional direction of change unresolvable. "
               if np.isfinite(ratio) and ratio >= 0.5 else
               "That is small relative to the observed per-pixel slopes. ")
            + ("HOWEVER, this step confounds the instrument change with real "
               "vegetation change. The direct measurement on "
               f"near-simultaneous Landsat 7 / Landsat 8-9 pairs puts the "
               f"harmonised instrument residual at {residual:+.4f} NDVI"
               + (" - OPPOSITE IN SIGN to the observed step. Including OLI "
                  "scenes therefore DEPRESSES post-transition composites, so "
                  "the instrument cannot be producing the increase; the real "
                  "change is at least as large as measured, and this step is "
                  "not a sensor artefact."
                  if residual is not None and residual < 0 < step else
                  " - the SAME sign as the observed step, so part of the "
                  "step may be instrumental and the observed change must be "
                  "discounted by up to that amount.")
               if residual is not None else
               "No near-simultaneous cross-sensor pairs were available to "
               "separate the instrument from real change, so the step "
               "remains ambiguous and no regional direction-of-change "
               "conclusion should be drawn.")),
        "caveat": (
            "The step statistic is a REGIONAL-MEAN diagnostic and cannot "
            "rule out larger residuals on particular land covers, because "
            "the harmonisation coefficients are global rather than "
            "cover-specific. The paired residual is an UPPER BOUND on the "
            "instrument effect: even scenes days apart differ in "
            "illumination, view geometry and atmosphere."),
    }
    _write(exp.path("summary") / "sensor_confound.json", summary)
    pd.DataFrame({
        "year": years, "mean_ndvi": annual,
        "n_scenes": [len(by_year.get(y, [])) for y in years],
        "sensors": [",".join(sorted(set(by_year.get(y, [])))) for y in years],
    }).to_csv(exp.path("tables") / "annual_ndvi_by_sensor_mix.csv",
              index=False)

    if summary["step_is_statistically_detectable"]:
        log.warning("SENSOR CONFOUND: residual step of %+.4f NDVI at the %d "
                    "OLI transition (t=%.2f), equivalent to %+.5f NDVI/yr "
                    "= %.1fx the median Sen slope", step, first_oli,
                    t_statistic, equivalent, ratio)
    else:
        log.info("sensor confound: step %+.4f NDVI at %d (t=%.2f), not "
                 "statistically detectable", step, first_oli, t_statistic)
    return summary


def stage_trajectories(prepared, cfg, exp, log):
    """Integrated trajectory classification and areas (Parts 10, 20)."""
    labels = prepared.trajectory_labels
    codes = trajectory_codes(labels)
    grid = _grid_of(prepared, codes)
    names = {v: k for k, v in TRAJECTORY_CODES.items()}

    geo.write_layer(exp.path("temporal_analysis") / "trajectory_class.tif",
                    codes, prepared.analysis_grid, prepared.georef,
                    dtype="uint8",
                    description="Integrated trajectory class: " + ", ".join(
                        f"{v}={k}" for k, v in TRAJECTORY_CODES.items()))
    geo.write_class_geojson(
        exp.path("maps") / "trajectory_class.geojson",
        np.nan_to_num(grid, nan=0).astype("int32"), prepared.georef,
        class_names=names,
        description="Integrated vegetation-trajectory classes. Analytical "
                    "signal categories derived from the temporal record; NOT "
                    "verified land cover and NOT a degradation map.")

    table = area_statistics(grid, prepared.georef, class_names=names,
                            valid_mask=prepared.analysis_grid)
    table.to_csv(exp.path("tables") / "trajectory_areas.csv", index=False)
    areas = {row["class_name"]: row["area_km2"] for _, row in table.iterrows()}

    # WHAT IS IN "Uncertain / Other"? On this record it is the largest class,
    # and reporting that as "28% ambiguous" without decomposing it would be
    # misleading. The trajectory scheme was designed around DEGRADATION and
    # has no class for significant IMPROVEMENT, so a significantly greening
    # pixel matches no rule and lands here. That is a property of the class
    # design, not evidence of ambiguity, and it must be said plainly.
    features = prepared.features
    uncertain = labels == "Uncertain / Other"
    significant = (np.isfinite(features["mk_p_value"].to_numpy())
                   & (features["mk_p_value"].to_numpy() < cfg.trend.alpha))
    greening = significant & (features["sen"].to_numpy() > 0)
    composition = {
        "n_uncertain": int(uncertain.sum()),
        "fraction_of_analysed": float(uncertain.mean()),
        "significant_greening_within_uncertain": int((uncertain & greening).sum()),
        "share_of_uncertain_that_is_significant_greening": (
            float((uncertain & greening).sum() / uncertain.sum())
            if uncertain.any() else float("nan")),
        "genuinely_unmatched": int((uncertain & ~greening).sum()),
        "explanation": (
            "The trajectory scheme has classes for stability, decline, "
            "disturbance, recovery and recurrence, but NO class for "
            "significant vegetation INCREASE. A pixel with a significant "
            "positive trend therefore matches no rule and is placed in "
            "'Uncertain / Other'. On this record that is most of the class. "
            "'Uncertain' here should be read as 'not covered by the "
            "degradation-oriented scheme', not as 'the data are ambiguous'."),
        "consequence": (
            "Any statement of the form 'X% of the study area is uncertain' "
            "would misrepresent this result and must not be made."),
    }
    _write(exp.path("summary") / "uncertain_class_composition.json",
           composition)
    log.info("uncertain class: %d pixels, of which %d (%.1f%%) are "
             "significant GREENING with no matching class",
             composition["n_uncertain"],
             composition["significant_greening_within_uncertain"],
             100 * composition[
                 "share_of_uncertain_that_is_significant_greening"])

    summary = {
        "classes": list(TRAJECTORY_CLASSES),
        "counts": {name: int((labels == name).sum())
                   for name in TRAJECTORY_CLASSES},
        "areas_km2": areas,
        "area_method": table.attrs.get("method", ""),
        "total_analysed_area_km2": table.attrs.get(
            "total_analysed_area_km2", float("nan")),
        "rules": {
            "Stable": "no significant Mann-Kendall trend and no other rule met",
            "Degrading": (
                "significant negative trend that SURVIVES climate adjustment, "
                "or whose rainfall relation is too weak to adjust"),
            "Rainfall-associated decline": (
                "significant negative trend, valid NDVI~rainfall relation, "
                "and no significant residual decline"),
            "Disturbed": (
                "significant breakpoint with magnitude >= "
                f"{cfg.recovery.min_disturbance_magnitude} NDVI that has not "
                f"regained {cfg.recovery.recovery_threshold:.0%} of the drop"),
            "Recovering": (
                "significant breakpoint that HAS regained "
                f"{cfg.recovery.recovery_threshold:.0%} of the drop"),
            "Cyclic": (
                f"spectral enrichment >= "
                f"{cfg.cyclicity.periodicity_threshold}x the white-noise "
                f"level inside the {cfg.cyclicity.min_period:.0f}-"
                f"{cfg.cyclicity.max_period:.0f} year band"),
            "Uncertain / Other": "no rule matched, or inputs were not finite",
        },
        "priority": list(cfg.research.trajectory.priority),
        "uncertain_class_composition": composition,
        "interpretation": (
            "These are ANALYTICAL SIGNAL CATEGORIES describing the shape of "
            "each pixel's vegetation record. They are not verified land "
            "cover, not a degradation map, and not evidence of any land-use "
            "practice."),
    }
    _write(exp.path("summary") / "trajectories.json", summary)
    log.info("trajectories: %s", ", ".join(
        f"{k} {v}" for k, v in summary["counts"].items() if v))
    return summary, table, areas


def stage_baseline_comparison(prepared, cfg, exp, log):
    """What integration changes versus a trend-only rule (Part 17).

    Without reference labels no ACCURACY comparison is possible. What IS
    measurable, and is the question the project actually asks, is how many
    pixels a conventional significant-negative-trend rule would call
    degradation that the integrated framework attributes to rainfall,
    recurrence or a recovering disturbance instead. That is a real,
    falsifiable, label-free result.
    """
    features = prepared.features
    labels = prepared.trajectory_labels
    alpha = cfg.trend.alpha
    slope = features["sen"].to_numpy()
    p_value = features["mk_p_value"].to_numpy()
    baseline_flag = np.isfinite(p_value) & (p_value < alpha) & (slope < 0)
    areas = pixel_area_km2(prepared.georef)[prepared.analysis_grid]

    reassigned = {}
    for name in TRAJECTORY_CLASSES:
        mask = baseline_flag & (labels == name)
        reassigned[name] = {"n_pixels": int(mask.sum()),
                            "area_km2": float(areas[mask].sum()),
                            "share_of_baseline_flags": float(
                                mask.sum() / baseline_flag.sum())
                            if baseline_flag.sum() else float("nan")}
    integrated_decline = np.isin(labels, list(DECLINE_CLASSES))
    persistent = labels == "Degrading"

    summary = {
        "baseline_rule": (
            "significant negative Mann-Kendall trend (the conventional "
            "trend-only degradation indicator)"),
        "baseline_flagged_pixels": int(baseline_flag.sum()),
        "baseline_flagged_area_km2": float(areas[baseline_flag].sum()),
        "integrated_persistent_decline_pixels": int(persistent.sum()),
        "integrated_persistent_decline_area_km2": float(
            areas[persistent].sum()),
        "integrated_any_decline_pixels": int(integrated_decline.sum()),
        "reduction_from_baseline_to_persistent": (
            1.0 - persistent.sum() / baseline_flag.sum()
            if baseline_flag.sum() else float("nan")),
        "where_baseline_flags_land_in_the_integrated_scheme": reassigned,
        "what_this_shows": (
            "The share of conventional trend-only degradation flags that the "
            "integrated framework attributes to rainfall variability, "
            "recurrent dynamics or a recovering disturbance instead. It "
            "quantifies what integration CHANGES. It does not establish that "
            "the integrated answer is correct - that would need independent "
            "reference labels, which this study does not have."),
    }
    _write(exp.path("summary") / "baseline_comparison.json", summary)
    pd.DataFrame([{"trajectory_class": k, **v}
                  for k, v in reassigned.items()]).to_csv(
        exp.path("tables") / "baseline_reassignment.csv", index=False)
    log.info("baseline comparison: trend-only flags %d pixels; the integrated "
             "framework calls %d of them persistent decline (%.1f%% "
             "reduction)", int(baseline_flag.sum()), int(persistent.sum()),
             100 * summary["reduction_from_baseline_to_persistent"])
    return summary


def stage_uncertainty(prepared, cfg, exp, log):
    """Indicator-disagreement uncertainty (Part 19).

    No model probabilities exist here, because no model can legitimately be
    fitted. The available uncertainty is DISAGREEMENT BETWEEN INDEPENDENT
    INDICATORS: where the trend test, the climate-adjusted trend, the
    breakpoint test and the periodicity statistic point in different
    directions, the pixel's trajectory is genuinely ambiguous.
    """
    features = prepared.features
    alpha = cfg.trend.alpha
    slope = features["sen"].to_numpy()
    p_value = features["mk_p_value"].to_numpy()
    residual = features["restrend"].to_numpy()
    residual_p = features["restrend_p_value"].to_numpy()
    valid = features["restrend_valid"].to_numpy(bool)
    periodic = features["cyclicity_periodic"].to_numpy(bool)
    disturbed = features["breakpoint_significant"].to_numpy(bool)
    n_valid = features["n_valid_ndvi"].to_numpy()
    areas = pixel_area_km2(prepared.georef)[prepared.analysis_grid]

    decline_indicators = np.vstack([
        (np.isfinite(p_value) & (p_value < alpha) & (slope < 0)),
        (valid & np.isfinite(residual_p) & (residual_p < alpha)
         & (residual < 0)),
        disturbed,
    ]).astype(int)
    agreeing = decline_indicators.sum(axis=0)
    ambiguous = (agreeing == 1) | (periodic & (agreeing >= 1))
    thin = n_valid < (cfg.quality.min_valid_obs + 3)
    labelled_uncertain = prepared.trajectory_labels == "Uncertain / Other"

    grid = _grid_of(prepared, agreeing.astype("float64"))
    geo.write_raster(exp.path("uncertainty") / "decline_indicator_agreement.tif",
                     grid, prepared.georef, dtype="float32",
                     description="Number of independent decline indicators "
                                 "agreeing (0-3)")
    geo.write_layer(exp.path("uncertainty") / "ambiguous_trajectory.tif",
                    ambiguous.astype("float64"), prepared.analysis_grid,
                    prepared.georef, dtype="uint8",
                    description="1 where indicators disagree or a decline "
                                "coincides with recurrent behaviour")
    geo.write_layer(exp.path("uncertainty") / "thin_record.tif",
                    thin.astype("float64"), prepared.analysis_grid,
                    prepared.georef, dtype="uint8",
                    description="1 where the pixel has barely enough valid "
                                "observations to test")

    summary = {
        "indicators": ["significant negative Mann-Kendall trend",
                       "significant negative climate-adjusted residual trend "
                       "(where the rainfall relation is valid)",
                       "significant structural breakpoint"],
        "agreement_counts": {str(k): int((agreeing == k).sum())
                             for k in range(4)},
        "all_three_agree_pixels": int((agreeing == 3).sum()),
        "all_three_agree_area_km2": float(areas[agreeing == 3].sum()),
        "ambiguous_pixels": int(ambiguous.sum()),
        "ambiguous_area_km2": float(areas[ambiguous].sum()),
        "ambiguous_fraction": float(ambiguous.mean()),
        "thin_record_pixels": int(thin.sum()),
        "thin_record_area_km2": float(areas[thin].sum()),
        "classified_uncertain_pixels": int(labelled_uncertain.sum()),
        "classified_uncertain_area_km2": float(areas[labelled_uncertain].sum()),
        "interpretation": (
            "Agreement between independent indicators is the strongest "
            "label-free evidence available here. Where all three decline "
            "indicators agree, the evidence for a persistent decline is as "
            "strong as this data can make it. Where only one fires, or a "
            "decline coincides with recurrent behaviour, the trajectory is "
            "ambiguous and is reported as such rather than forced into a "
            "class. None of these counts is a probability that the ground is "
            "degraded."),
        "ceiling_on_three_way_agreement": {
            "restrend_valid_fraction": float(valid.mean()),
            "note": (
                "The climate-adjusted indicator can only fire where the "
                "NDVI~rainfall relation is valid, which is true on "
                f"{float(valid.mean()):.1%} of pixels in this study area. "
                "Three-way agreement is therefore bounded by that fraction "
                "and a small count is expected here BY CONSTRUCTION. It is "
                "not independent evidence that persistent decline is rare; "
                "it reflects that one of the three indicators is mostly "
                "unavailable. Two-way agreement between the raw trend and "
                "the breakpoint test is the more informative statistic on "
                "this record."),
            "trend_and_breakpoint_agree_pixels": int(
                ((decline_indicators[0] + decline_indicators[2]) == 2).sum()),
            "trend_and_breakpoint_agree_area_km2": float(
                areas[(decline_indicators[0] + decline_indicators[2]) == 2
                      ].sum()),
        },
    }
    _write(exp.path("uncertainty") / "indicator_disagreement.json", summary)
    pd.DataFrame([{"n_indicators_agreeing": k,
                   "n_pixels": int((agreeing == k).sum()),
                   "area_km2": float(areas[agreeing == k].sum())}
                  for k in range(4)]).to_csv(
        exp.path("tables") / "indicator_agreement.csv", index=False)
    log.info("uncertainty: all three decline indicators agree on %d pixels "
             "(%.0f km2); %d pixels ambiguous; %d have a thin record",
             int((agreeing == 3).sum()), summary["all_three_agree_area_km2"],
             int(ambiguous.sum()), int(thin.sum()))
    return summary, agreeing, ambiguous


def stage_sensitivity(prepared, cfg, exp, log):
    """Parameter robustness on real data (Part 16)."""
    rng = np.random.default_rng(cfg.seed)
    n = int(prepared.analysis_mask.sum())
    take = min(n, 3000)
    columns = np.sort(rng.choice(n, take, replace=False))
    table = run_sensitivity_analysis(
        prepared.series[:, columns], prepared.rain_series[:, columns],
        exp.path("sensitivity"), cfg, evaluate_model=False,
        data_status=("REAL Landsat Collection 2 Level-2 and CHIRPS "
                     "observations; no reference labels, so no model metrics "
                     "are included"),
        logger=log)
    baseline = table[table["parameter"] == "(baseline)"].iloc[0]
    spread = {}
    for column in ("trajectory_degrading_fraction", "periodic_fraction",
                   "significant_trend_fraction", "disturbance_fraction"):
        if column in table:
            values = table[column].astype(float)
            spread[column] = {
                "baseline": float(baseline[column]),
                "min": float(values.min()), "max": float(values.max()),
                "range": float(values.max() - values.min())}
    summary = {"n_pixels_tested": int(take), "n_scenarios": int(len(table)),
               "spread": spread,
               "interpretation": (
                   "Scenarios are reported, never optimised. A conclusion is "
                   "robust if its direction survives the whole sweep; the "
                   "ranges below show how far each headline fraction moves "
                   "across all tested parameter values.")}
    _write(exp.path("summary") / "sensitivity.json", summary)
    log.info("sensitivity: %d scenarios; degrading fraction ranges "
             "%.3f-%.3f (baseline %.3f)", len(table),
             spread.get("trajectory_degrading_fraction", {}).get("min",
                                                                 float("nan")),
             spread.get("trajectory_degrading_fraction", {}).get("max",
                                                                 float("nan")),
             spread.get("trajectory_degrading_fraction", {}).get("baseline",
                                                                 float("nan")))
    return summary, table


def stage_blocked(exp, log):
    """Record what could not be run, and why (Parts 11-15, 18)."""
    for directory in ("models", "validation", "ablation"):
        _write(exp.path(directory) / "BLOCKED.json", BLOCKED)
    log.warning("SUPERVISED EXPERIMENTS BLOCKED BY DATA: %s", BLOCKED["why"])
    return BLOCKED


def main(cfg: Config | None = None):
    cfg = cfg or Config.load("configs/m7_karbi_anglong_final.json")
    # The declared time axis must match the real record, or the contract
    # check in `dataset.validate_dataset` compares the cube's band count
    # against an unrelated default and fails for the wrong reason.
    cfg.years = list(range(int(cfg.real_data.start_year),
                           int(cfg.real_data.end_year) + 1))
    area = load_study_area(cfg.study_area)
    exp = start_experiment(cfg, results_root=Path(cfg.paths.results)
                           / "final_real_data", subdirs=TREE)
    log = exp.logger
    log.info("M7 FINAL REAL-DATA STUDY | study area: %s", area.name)
    # State the sampling method the DATA actually used, read from the
    # acquisition record - not a constant, which would keep announcing
    # nearest-neighbour subsampling after the pipeline stopped doing it.
    sampling = SUBSAMPLING_NOTE
    acquisition_path = Path(cfg.real_data.metadata_dir) / "m7_acquisition.json"
    if acquisition_path.exists():
        sampling = json.loads(acquisition_path.read_text()).get(
            "landsat", {}).get("sampling", SUBSAMPLING_NOTE)
    log.info("%s", sampling)

    results = {"study_area": area.describe(),
               "data_status": "REAL remote-sensing observations",
               "sources": {"vegetation": PLANETARY_COMPUTER["provenance"],
                           "rainfall": CHIRPS["citation"]},
               "sampling": sampling}

    log.info("STAGE 1/12  real dataset and contract validation")
    prepared = stage_dataset(cfg, area, exp, log)
    results["dataset"] = prepared.validation_report
    results["experiment"] = prepared.summary()

    log.info("STAGE 2/12  data-quality verification")
    quality, _ = stage_quality(prepared, cfg, exp, log, area=area)
    results["data_quality"] = {
        "ndvi": quality["vegetation"], "rainfall": quality["rainfall"],
        "satellite_quality": quality["satellite_quality"]}

    log.info("STAGE 3/12  descriptive vegetation analysis")
    descriptive = stage_descriptive(prepared, exp, log)

    log.info("STAGE 4/12  georeferenced analysis layers and feature matrix")
    stage_temporal_layers(prepared, exp, log)
    results["features"] = stage_features(prepared, cfg, exp, log)

    log.info("STAGE 5/12  long-term trend")
    results["trend"] = stage_trend(prepared, cfg, exp, log)

    log.info("STAGE 6/12  climate-adjusted trend (RESTREND)")
    results["restrend"] = stage_restrend(prepared, cfg, exp, log)

    log.info("STAGE 7/12  cyclicity and surrogate significance")
    results["cyclicity"] = stage_cyclicity(prepared, cfg, exp, log)

    log.info("STAGE 8/12  disturbance, recovery and the sensor confound")
    results["disturbance_recovery"] = stage_disturbance_recovery(
        prepared, cfg, exp, log)
    results["sensor_confound"] = stage_sensor_confound(prepared, cfg, exp, log)

    log.info("STAGE 9/12  integrated trajectories and areas")
    trajectories, area_table, areas = stage_trajectories(prepared, cfg, exp,
                                                         log)
    results["trajectories"] = trajectories

    log.info("STAGE 10/12  label-free baseline comparison")
    results["baseline_comparison"] = stage_baseline_comparison(
        prepared, cfg, exp, log)

    log.info("STAGE 11/12  uncertainty and sensitivity")
    uncertainty, agreeing, ambiguous = stage_uncertainty(prepared, cfg, exp,
                                                         log)
    results["uncertainty"] = uncertainty
    sensitivity, sensitivity_table = stage_sensitivity(prepared, cfg, exp, log)
    results["sensitivity"] = sensitivity

    log.info("STAGE 12/12  supervised stages (blocked) and outputs")
    results["supervised"] = stage_blocked(exp, log)

    # Figures and the integrated table are produced by a separate module so
    # that a figure change never risks re-running the analysis.
    from src.m7_outputs import (write_findings, write_integrated_table,
                                write_maps, write_profiles,
                                write_reproducibility_package)
    log.info("writing maps and figures")
    maps = write_maps(prepared, descriptive, areas, exp, cfg, log)
    profiles = write_profiles(prepared, exp, cfg, log)
    integrated = write_integrated_table(results, prepared, exp, log)
    results["integrated_table"] = integrated
    findings = write_findings(results, exp, cfg, log)
    package = write_reproducibility_package(cfg, exp, results, log)
    results["outputs"] = {"maps": len(maps), "profiles": len(profiles),
                          "reproducibility_package": str(package)}

    _write(exp.path("summary") / "results.json", results)
    log.info("M7 STUDY COMPLETE -> %s", exp.root)
    return exp, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",
                        default="configs/m7_karbi_anglong_final.json")
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    configuration = Config.load(args.config)
    if args.name:
        configuration.experiment_name = args.name
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main(configuration)


