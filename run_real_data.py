"""M6 real-data runner.

Runs the UNCHANGED M1-M5 research pipeline on real remote-sensing
observations. There is no second analytical pipeline: this script swaps the
data source and calls the same `experiment.prepare_experiment` that the M2
and M3 runners call.

    python run_real_data.py --prepare      # scenes -> composited cubes
    python run_real_data.py                # analysis on the prepared cubes
    python run_real_data.py --smoke        # small subset, end to end
    python run_real_data.py --export-plan  # what a GEE export would request

WHAT RUNS AND WHAT DOES NOT
---------------------------
Everything statistical runs on real data: quality gating, Mann-Kendall,
Sen's slope, RESTREND, cyclicity, breakpoint detection, recovery, the full
engineered feature table, the analytical trajectory classes, georeferenced
exports and area statistics.

The SUPERVISED stages (Random Forest, 1D CNN, ablation, spatial CV
metrics, discrimination) run only when `real_data.reference` names
independent reference labels. Without them this script reports supervised
learning as BLOCKED and says exactly what is missing. It does not substitute
the analytical trajectory classes for ground truth: those are computed from
the same features a classifier would consume, so scoring a model against
them measures self-consistency, not accuracy. See README, "Reference data".

DATA STATUS: outputs of this script are computed from REAL observations and
are labelled as such. The synthetic development pipeline is untouched and
remains the basis of the test suite.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from run_m2_experiments import (write_analysis_layers,
                                write_temporal_diagnostics,
                                write_trajectory_outputs)
from src import geo
from src import research_figures as RF
from src import timeseries as TS
from src.config import Config
from src.data_source import save_requirements
from src.experiment import prepare_experiment
from src.features import feature_dictionary
from src.real_data import (REAL_DATA_NOTICE, SYNTHETIC_FIXTURE_NOTICE,
                           RealDataError, RealRemoteSensingSource,
                           preprocess_real_data, resolve_target_grid)
from src.real_report import (MISSINGNESS_CAVEAT, build_quality_report,
                             plot_quality_report, write_quality_report)
from src.reproducibility import start_experiment
from src.sensors import index_table, sensor_table
from src.study_area import area_statistics, load_study_area
from src.trajectory import TRAJECTORY_CODES

NOTICE = REAL_DATA_NOTICE

#: Why supervised learning cannot be faked into running. Written into the
#: results so the omission is a documented finding, not a silent gap.
BLOCKED_STATEMENT = {
    "status": "BLOCKED",
    "what_is_blocked": ["random_forest", "cnn_1d", "ablation_experiments",
                        "spatial_cv_metrics", "temporal_holdout_metrics",
                        "degradation_discrimination", "model_uncertainty",
                        "permutation_importance"],
    "why": ("Supervised evaluation needs reference labels that are "
            "independent of the features. Satellite imagery does not supply "
            "them, and none is configured (real_data.reference.path)."),
    "why_not_use_the_trajectory_classes": (
        "The analytical trajectory classes are computed by "
        "trajectory.classify_trajectories from the SAME engineered features "
        "the classifier would consume - the Mann-Kendall p-value, the Sen "
        "slope, the RESTREND residual trend, the spectral enrichment, the "
        "breakpoint and the recovery fraction. Training on them and "
        "reporting the resulting accuracy would measure whether a Random "
        "Forest can re-derive a deterministic rule from that rule's own "
        "inputs. It would score near-perfectly and would mean nothing. This "
        "is label leakage by construction and the pipeline refuses it."),
    "what_would_unblock_it": [
        "Field observations of degradation status at located points.",
        "Expert photo-interpretation of high-resolution imagery "
        "(e.g. Google Earth / Planet time series) at a stratified sample of "
        "pixels, by an interpreter blind to the model output.",
        "A published, peer-reviewed shifting-cultivation or land-degradation "
        "dataset whose classification scheme, spatial resolution and vintage "
        "are compatible with this study.",
        "An authoritative land-cover product (e.g. a national land-use/"
        "land-cover map) used for the confounder classes only, with its "
        "scheme mapped explicitly onto this study's classes.",
    ],
    "what_still_runs": (
        "Every unsupervised and statistical stage: quality gating, "
        "Mann-Kendall, Sen's slope, RESTREND, cyclicity with surrogate "
        "significance, breakpoint detection, recovery metrics, the full "
        "feature table, the analytical trajectory classes, georeferenced "
        "exports and area statistics."),
}


def smoke_config(cfg: Config) -> Config:
    """Small spatial subset and short record: Part 17's smoke test."""
    real = cfg.real_data
    real.end_year = min(real.end_year, real.start_year + 11)
    real.target_resolution_m = max(real.target_resolution_m, 300.0)
    cfg.years = list(range(real.start_year, real.end_year + 1))
    cfg.research.max_analysis_pixels = 2000
    cfg.research.samples_per_class = 40
    cfg.research.model.n_estimators = 40
    cfg.research.spatial_cv.n_folds = 3
    cfg.research.model.block_cv.n_folds = 3
    cfg.cyclicity.n_surrogates = 39
    cfg.experiment_name = f"{cfg.experiment_name}_smoke"
    return cfg


def sync_time_axis(cfg: Config) -> Config:
    """`cfg.years` is the declared time axis; keep it and the real data one.

    `dataset.validate` compares the cube's band count with `len(cfg.years)`.
    Deriving the years from the real-data temporal design keeps that check
    meaningful instead of making it fail on an unrelated default.
    """
    real = cfg.real_data
    if str(real.temporal_unit).lower() == "annual":
        cfg.years = list(range(int(real.start_year), int(real.end_year) + 1))
    else:
        from src.compositing import build_windows
        cfg.years = [w.label for w in build_windows(
            real.temporal_unit, real.start_year, real.end_year,
            window_start=real.window_start, window_end=real.window_end)]
    return cfg


def write_area_statistics(prepared, exp, cfg, log) -> dict:
    """Part 26: areas from real pixel geometry, never from a pixel count."""
    from src.trajectory import trajectory_codes

    georef = prepared.georef
    grid = np.full(prepared.shape, np.nan)
    grid[prepared.analysis_grid] = trajectory_codes(prepared.trajectory_labels)
    names = {v: k for k, v in TRAJECTORY_CODES.items()}
    trajectory_area = area_statistics(grid, georef, class_names=names,
                                      valid_mask=prepared.analysis_grid)
    exp.save_table("area_by_trajectory_class",
                   trajectory_area.set_index("class_value"))

    features = prepared.features
    alpha = cfg.trend.alpha
    significant = (features["mk_p_value"].to_numpy() < alpha)
    slope = features["sen"].to_numpy()
    restrend_slope = features["restrend"].to_numpy()
    restrend_p = features["restrend_p_value"].to_numpy()
    restrend_valid = features["restrend_valid"].to_numpy(bool)
    conditions = {
        "significant_decline": significant & (slope < 0),
        "significant_increase": significant & (slope > 0),
        "no_significant_trend": ~significant,
        "climate_adjusted_decline": (restrend_valid & (restrend_p < alpha)
                                     & (restrend_slope < 0)),
        "cyclic_signal": features["cyclicity_periodic"].to_numpy(bool),
        "disturbance_detected": features["has_disturbance"].to_numpy(bool),
        "recovering_after_disturbance":
            features["recovery_status"].to_numpy() == 2,
    }
    condition_grid = np.zeros(prepared.shape, dtype="int32")
    rows = []
    from src.study_area import pixel_area_km2
    areas = pixel_area_km2(georef)
    for i, (name, mask) in enumerate(conditions.items(), start=1):
        condition_grid[:] = 0
        condition_grid[prepared.analysis_grid] = np.where(mask, i, 0)
        selected = condition_grid == i
        rows.append({"condition": name,
                     "n_pixels": int(selected.sum()),
                     "area_km2": float(areas[selected].sum())})
    total = float(areas[prepared.analysis_grid].sum())
    condition_area = pd.DataFrame(rows)
    condition_area["fraction_of_analysed_area"] = \
        condition_area["area_km2"] / total if total else np.nan
    condition_area["method"] = trajectory_area.attrs.get("method", "")
    exp.save_table("area_by_condition", condition_area.set_index("condition"))

    # Area is computed over the ANALYSED pixels. When the run is thinned
    # (research.max_analysis_pixels) that is a fraction of the study area, and
    # a total of "180 km2" would otherwise read as the size of the district.
    # State the relationship rather than leaving it to be inferred.
    study_area_km2 = float(areas.sum())
    caveats = [
        "Areas cover the ANALYSED pixels, not necessarily the whole study "
        f"area. The configured extent is {study_area_km2:,.1f} km2; "
        f"{total:,.1f} km2 passed quality gating and was analysed.",
        "If the boundary is a bounding box rather than an administrative "
        "polygon, these are areas of the rectangle, not of the district.",
    ]
    if prepared.notes:
        caveats.append(
            "THIS RUN IS THINNED: " + "; ".join(prepared.notes)
            + ". Absolute areas are therefore a subsample and must not be "
              "reported as extents; the fractions remain interpretable.")
    summary = {
        "analysed_area_km2": total,
        "study_area_extent_km2": study_area_km2,
        "analysed_fraction_of_extent": (total / study_area_km2
                                        if study_area_km2 else float("nan")),
        "thinned_run": bool(prepared.notes),
        "area_calculation_method": trajectory_area.attrs.get("method", ""),
        "by_trajectory_class": trajectory_area.to_dict(orient="records"),
        "by_condition": condition_area.to_dict(orient="records"),
        "caveats": caveats,
    }
    exp.save_metrics("area_statistics", summary)
    if prepared.notes:
        log.warning("area statistics cover a THINNED subsample (%s); report "
                    "fractions, not absolute extents", "; ".join(prepared.notes))
    log.info("analysed area %.1f km2; significant decline %.1f km2, "
             "climate-adjusted decline %.1f km2", total,
             condition_area.loc[condition_area["condition"]
                                == "significant_decline", "area_km2"].iloc[0],
             condition_area.loc[condition_area["condition"]
                                == "climate_adjusted_decline",
                                "area_km2"].iloc[0])
    return summary


def run_cyclicity_significance(prepared, cfg, exp, log) -> dict:
    """Surrogate significance on a seeded subsample of analysed pixels."""
    rng = np.random.default_rng(cfg.seed)
    n = int(prepared.analysis_mask.sum())
    take = min(n, 4000)
    columns = np.sort(rng.choice(n, take, replace=False))
    surrogate = TS.cyclicity_significance(
        prepared.series[:, columns], min_period=cfg.cyclicity.min_period,
        max_period=cfg.cyclicity.max_period, detrend=cfg.cyclicity.detrend,
        min_obs=cfg.cyclicity.min_obs, n_surrogates=cfg.cyclicity.n_surrogates,
        alpha=cfg.cyclicity.surrogate_alpha, seed=cfg.cyclicity.surrogate_seed)
    threshold_flag = prepared.features["cyclicity_periodic"].to_numpy(
        bool)[columns]
    report = {
        "null_model": surrogate["null"],
        "n_surrogates": surrogate["n_surrogates"],
        "n_pixels_tested": int(take),
        "significant_fraction": float(np.mean(surrogate["significant"])),
        "threshold_flag_fraction": float(np.mean(threshold_flag)),
        "agreement_with_threshold_rule": float(
            np.mean(surrogate["significant"] == threshold_flag)),
        "median_p_value": float(np.nanmedian(surrogate["p_value"])),
        "interpretation": (
            "A small p-value means the concentration of spectral power "
            "inside the period band is unlikely for a series with this "
            "pixel's own autocorrelation. It does NOT identify a cause. "
            "Rotational cultivation, plantation harvest cycles, "
            "fire-regrowth cycles and multi-year climate oscillation all "
            "produce periodicity. Periodicity is not jhum."),
    }
    exp.save_metrics("cyclicity_surrogate_test", report)
    pd.DataFrame({"p_value": surrogate["p_value"],
                  "significant": surrogate["significant"],
                  "enrichment": prepared.features["cyc_enrichment"
                                                  ].to_numpy()[columns],
                  "threshold_rule_flag": threshold_flag}).to_csv(
        exp.path("metrics", "cyclicity_surrogate.csv"), index=False)
    log.info("surrogate test (%s null): %.1f%% significant; threshold rule "
             "%.1f%%; agreement %.1f%%", surrogate["null"],
             100 * report["significant_fraction"],
             100 * report["threshold_flag_fraction"],
             100 * report["agreement_with_threshold_rule"])
    return report


def run_supervised_stages(prepared, cfg, exp, log) -> dict:
    """The M3 supervised evaluation, on real independent reference labels."""
    from src.ablation import run_ablation_study
    from src.experiment_matrix import run_experiment_matrix
    from src.explain import explain_experiment
    from src.uncertainty import (prediction_confidence, uncertainty_summary,
                                 uncertainty_table)
    from src.validation import fit_random_forest, spatial_cv_rf

    features, labels = prepared.features, prepared.labels
    folds, sample_mask = prepared.folds, prepared.sample_mask
    columns = prepared.model_columns

    cv = spatial_cv_rf(features, labels, folds, sample_mask=sample_mask,
                       feature_names=columns, cfg=cfg.research.model,
                       block_row=prepared.block_row,
                       block_col=prepared.block_col)
    metrics = cv["metrics"]
    evaluated = cv["evaluated"]
    metrics["uncertainty"] = uncertainty_summary(
        cv["probabilities"][evaluated], truth=labels[evaluated],
        predictions=cv["predictions"][evaluated],
        cfg=cfg.research.uncertainty)
    exp.save_metrics("spatial_cv_metrics", metrics)
    uncertainty_table(cv["predictions"][evaluated],
                      cv["probabilities"][evaluated], cv["classes"],
                      truth=labels[evaluated], cfg=cfg.research.uncertainty
                      ).assign(fold=folds[evaluated]).to_csv(
        exp.path("predictions", "rf_predictions_uncertainty.csv"), index=False)
    names = cfg.real_data.reference.classes or cfg.classes
    RF.plot_confusion_matrix(
        metrics["confusion_matrix"], metrics["labels"],
        exp.figure("confusion_spatial_cv.png"),
        class_names=[names.get(int(c), str(c)) for c in cv["classes"]],
        title="Random Forest, spatial block CV, REAL reference labels")
    log.info("RF spatial CV on real labels: accuracy %.4f | macro F1 %.4f",
             metrics["accuracy"], metrics["f1_macro"])

    matrix = run_experiment_matrix(
        features, labels, folds, exp.path("metrics", "experiment_matrix"),
        cfg, series=np.stack([prepared.series, prepared.rain_series]),
        channel_names=["ndvi", "rainfall"], sample_mask=sample_mask,
        block_row=prepared.block_row, block_col=prepared.block_col, logger=log)
    exp.save_table("experiment_matrix", matrix.set_index(["method", "task"]))

    ablation = run_ablation_study(features, labels, folds,
                                  exp.path("metrics", "ablations"), cfg,
                                  sample_mask=sample_mask,
                                  rf_cfg=cfg.research.model,
                                  block_row=prepared.block_row,
                                  block_col=prepared.block_col, logger=log)
    exp.save_table("ablation_comparison", ablation.set_index("experiment"))
    RF.plot_ablation_comparison(ablation,
                                exp.figure("ablation_comparison.png"),
                                synthetic=False)

    explanation = explain_experiment(
        features, labels, folds, exp.path("metrics", "explainability"),
        feature_names=columns, impurity=cv["importance"],
        sample_mask=sample_mask, rf_cfg=cfg.research.model,
        cfg=cfg.research.explain, logger=log)

    x_all = features.loc[:, columns].to_numpy(dtype="float64")
    imputer, model = fit_random_forest(x_all[sample_mask], labels[sample_mask],
                                       cfg.research.model)
    probability = model.predict_proba(imputer.transform(x_all))
    measures = prediction_confidence(probability, cfg=cfg.research.uncertainty)
    geo.write_layer(exp.path("predictions", "model_class.tif"),
                    model.predict(imputer.transform(x_all)).astype(float),
                    prepared.analysis_grid, prepared.georef, dtype="uint8",
                    description="Random Forest class from REAL reference "
                                "labels; a model output, not a verified map")
    geo.write_layer(exp.path("predictions", "model_confidence.tif"),
                    measures["confidence"], prepared.analysis_grid,
                    prepared.georef, dtype="float32",
                    description="Maximum class probability (not certainty)")
    geo.write_layer(exp.path("predictions", "model_uncertain_flag.tif"),
                    measures["uncertain"].astype(float),
                    prepared.analysis_grid, prepared.georef, dtype="uint8",
                    description="1 where flagged low-confidence")

    return {"status": "RUN",
            "spatial_block_cv": {k: v for k, v in metrics.items()
                                 if k != "fold_metrics"},
            "experiment_matrix": matrix.to_dict(orient="records"),
            "ablation": ablation.to_dict(orient="records"),
            "explainability": explanation["report"],
            "reference_labels": prepared.dataset.metadata["reference_labels"]}


def main(cfg: Config | None = None, *, prepare: bool = False,
         source=None):
    cfg = cfg or Config(experiment_name="m6_real_data")
    cfg = sync_time_axis(cfg)
    area = load_study_area(cfg.study_area)

    exp = start_experiment(cfg, results_root=cfg.paths.real_results)
    log = exp.logger
    results: dict = {"study_area": area.describe()}
    area.save(exp.path("config", "study_area.geojson"))
    save_requirements(exp.path("config", "real_data_requirements.json"))
    log.info("study area: %s (%s)", area.name, area.source)
    if not area.attributes.get("is_administrative_boundary", True):
        log.warning("the configured boundary is a %s, not an administrative "
                    "polygon; area statistics describe that extent, not the "
                    "named district",
                    area.attributes.get("geometry_kind", "rectangle"))

    # ------------------------------------------------- 0 acquisition
    if prepare:
        log.info("STAGE 0/7  preprocessing real scenes into composited cubes")
        provenance = preprocess_real_data(cfg, area=area, logger=log)
        exp.save_metrics("dataset_provenance", provenance)
        cfg.real_data.ndvi_cube = provenance["outputs"]["ndvi_cube"]
        cfg.real_data.rain_cube = provenance["outputs"]["rain_cube"]
        results["preprocessing"] = provenance["compositing_summary"]

    # ------------------------------------------------- 1 load and validate
    log.info("STAGE 1/7  loading the real dataset and enforcing the contract")
    source = source or RealRemoteSensingSource(cfg, study_area=area, logger=log)
    prepared = prepare_experiment(cfg, source=source, logger=log)
    dataset = prepared.dataset
    # The dataset states its own provenance; the runner does not assume it.
    # Fixture cubes travel through this identical code path, so the labels on
    # every figure and notice below follow the data rather than the script.
    synthetic = bool(dataset.metadata.get("synthetic", False))
    results["data_status"] = dataset.metadata.get("notice", NOTICE).strip()
    results["synthetic"] = synthetic
    exp.path("metrics", "m6_notice.txt").write_text(
        (SYNTHETIC_FIXTURE_NOTICE if synthetic else NOTICE))
    prepared.georef.save(exp.path("config", "georeference.json"))
    exp.save_metrics("dataset_metadata", prepared.validation_report)
    exp.save_metrics("dataset_description", dataset.describe())
    exp.save_metrics("data_quality", prepared.quality_summary)
    exp.save_metrics("trajectory_summary", prepared.trajectory_summary)
    exp.save_metrics("experiment_summary", prepared.summary())
    exp.save_table("feature_dictionary",
                   feature_dictionary().set_index("feature"))
    exp.save_table("sensor_registry", sensor_table().set_index("key"))
    exp.save_table("index_definitions", index_table().set_index("name"))
    results.update({"dataset": prepared.validation_report,
                    "data_quality": prepared.quality_summary,
                    "trajectory": prepared.trajectory_summary,
                    "experiment": prepared.summary()})

    # ------------------------------------------------- 2 quality report
    log.info("STAGE 2/7  real-data quality report")
    quality_dir = exp.root / "data_quality"
    report = build_quality_report(dataset, cfg)
    write_quality_report(report, quality_dir)
    for figure in plot_quality_report(dataset, report, exp.path("figures")):
        log.info("quality figure: %s", figure.name)
    results["data_quality_report"] = {
        "n_time_steps": report["temporal"]["n_time_steps"],
        "ndvi_missing_fraction": report["vegetation"]["missing_fraction"],
        "rainfall_missing_fraction": report["rainfall"]["missing_fraction"],
        "mean_valid_observations_per_pixel":
            report["satellite_quality"]["mean_valid_observations_per_pixel"],
        "caveat": MISSINGNESS_CAVEAT}
    log.info("NDVI %.3f..%.3f (mean %.3f), %.1f%% missing; rainfall "
             "%.1f..%.1f", report["vegetation"]["min"] or float("nan"),
             report["vegetation"]["max"] or float("nan"),
             report["vegetation"]["mean"] or float("nan"),
             100 * report["vegetation"]["missing_fraction"],
             report["rainfall"]["min"] or float("nan"),
             report["rainfall"]["max"] or float("nan"))

    # ------------------------------------------------- 3 temporal analysis
    log.info("STAGE 3/7  georeferenced temporal analysis (M1-M5, unchanged)")
    write_analysis_layers(prepared, exp, log, synthetic=synthetic)
    write_trajectory_outputs(prepared, exp, synthetic=synthetic)
    if prepared.has_labels:
        write_temporal_diagnostics(prepared, exp, cfg, synthetic=synthetic)
    geo.write_layer(exp.path("predictions", "spatial_cv_fold.tif"),
                    prepared.folds, prepared.analysis_grid, prepared.georef,
                    dtype="uint8",
                    description="Deterministic spatial CV fold assignment")
    RF.plot_spatial_folds(prepared.fold_grid,
                          exp.figure("spatial_cv_folds.png"))

    # ------------------------------------------------- 4 cyclicity rigour
    log.info("STAGE 4/7  cyclicity surrogate-data significance")
    results["cyclicity_surrogate_test"] = run_cyclicity_significance(
        prepared, cfg, exp, log)

    # ------------------------------------------------- 5 feature outputs
    log.info("STAGE 5/7  feature matrix and feature quality")
    feature_dir = exp.root / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    prepared.features.to_csv(feature_dir / "feature_matrix.csv", index=False)
    quality = prepared.features.describe().T
    quality["missing_fraction"] = prepared.features.isna().mean()
    quality.to_csv(feature_dir / "feature_quality.csv")
    (feature_dir / "feature_metadata.json").write_text(json.dumps(
        {"n_rows": int(len(prepared.features)),
         "modelling_columns": list(prepared.model_columns),
         "feature_groups": cfg.research.features.groups,
         "dictionary": feature_dictionary().to_dict(orient="records"),
         "data_status": "REAL observations"}, indent=2, default=str))
    log.info("feature matrix: %d rows x %d columns (%d modelling features)",
             len(prepared.features), prepared.features.shape[1],
             len(prepared.model_columns))

    # ------------------------------------------------- 6 area statistics
    log.info("STAGE 6/7  area statistics")
    results["area_statistics"] = write_area_statistics(prepared, exp, cfg, log)

    # ------------------------------------------------- 7 supervised or not
    log.info("STAGE 7/7  supervised evaluation")
    if prepared.has_labels:
        results["supervised"] = run_supervised_stages(prepared, cfg, exp, log)
    else:
        results["supervised"] = dict(BLOCKED_STATEMENT)
        results["supervised"]["reference_label_status"] = \
            dataset.metadata["reference_labels"]
        exp.save_metrics("supervised_blocked", results["supervised"])
        log.warning("SUPERVISED LEARNING BLOCKED: %s", BLOCKED_STATEMENT["why"])
        log.warning("every statistical and unsupervised stage completed on "
                    "real data; see metrics/supervised_blocked.json")

    (exp.root / "config" / "research_config.json").write_text(
        json.dumps(asdict(cfg.research), indent=2, default=str))
    (exp.root / "config" / "real_data_config.json").write_text(
        json.dumps(asdict(cfg.real_data), indent=2, default=str))
    exp.save_metrics("results", results)
    log.info("REAL-DATA EXPERIMENT COMPLETE -> %s", exp.root)
    return exp, results


def _export_plan(cfg: Config) -> int:
    from src.gee_export import build_export_plan, ee_available
    area = load_study_area(cfg.study_area)
    plan = build_export_plan(area, cfg.real_data)
    grid, note = resolve_target_grid(area, cfg.real_data)
    plan["analysis_grid"] = note
    plan["earthengine_api_installed"] = ee_available()
    target = Path(cfg.real_data.metadata_dir) / "gee_export_plan.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan, indent=2, default=str))
    print(json.dumps(plan, indent=2, default=str))
    print(f"\nplan written to {target}")
    if not plan["earthengine_api_installed"]:
        from src.gee_export import AUTH_INSTRUCTIONS
        print("\nearthengine-api is NOT installed; nothing was exported.\n")
        print(AUTH_INSTRUCTIONS)
        return 2
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default=None,
                        help="configuration JSON (must set study_area and "
                             "real_data)")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--boundary", type=str, default=None,
                        help="study-area polygon; overrides the config")
    parser.add_argument("--ndvi-cube", type=str, default=None)
    parser.add_argument("--rain-cube", type=str, default=None)
    parser.add_argument("--raw-dir", type=str, default=None,
                        help="directory holding scenes.json and rainfall.json")
    parser.add_argument("--prepare", action="store_true",
                        help="composite raw scenes into cubes first")
    parser.add_argument("--smoke", action="store_true",
                        help="small subset, short record: the Part 17 test")
    parser.add_argument("--export-plan", action="store_true",
                        help="print what a GEE export would request, and exit")
    parser.add_argument("--no-cnn", action="store_true")
    args = parser.parse_args()

    configuration = Config.load(args.config) if args.config \
        else Config(experiment_name="m6_real_data")
    if args.boundary:
        configuration.study_area.boundary = args.boundary
    if args.ndvi_cube:
        configuration.real_data.ndvi_cube = args.ndvi_cube
    if args.rain_cube:
        configuration.real_data.rain_cube = args.rain_cube
    if args.raw_dir:
        configuration.real_data.raw_dir = args.raw_dir
    if args.name:
        configuration.experiment_name = args.name
    if args.seed is not None:
        configuration.seed = args.seed
    if args.no_cnn:
        configuration.research.cnn.enabled = False
        configuration.research.matrix.run_cnn = False
    if args.smoke:
        configuration = smoke_config(configuration)

    if args.export_plan:
        raise SystemExit(_export_plan(configuration))

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    try:
        main(configuration, prepare=args.prepare)
    except RealDataError as error:
        raise SystemExit(f"\nREAL-DATA ERROR\n{'=' * 60}\n{error}\n")
