"""M2 research experiment runner: features, spatial CV, ablation, sensitivity.

    configuration -> data source -> contract validation -> quality control
    -> temporal analysis -> feature engineering -> trajectory classes
    -> spatial fold design -> model -> validation -> metrics -> predictions
    -> figures -> geospatial exports -> saved configuration

Stages 1-3 are `experiment.prepare_experiment`, shared with the M3 runner;
this file adds only what makes the experiment M2. It holds no analysis logic
of its own, so an experiment is defined by its `Config` alone.

    python run_m2_experiments.py
    python run_m2_experiments.py --seed 7 --name m2_seed7
    python run_m2_experiments.py --quick           # fast smoke run
    python run_m2_experiments.py --no-sensitivity --no-holdout
    python run_m2_experiments.py --config results/<id>/config/config.json

DATA STATUS: the bundled dataset is synthetic. Synthetic data are used for
development and pipeline validation. They are not real observations and must
not be interpreted as real-world research findings.
"""
from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import asdict

import numpy as np
import pandas as pd

from src import geo, maps
from src import research_figures as RF
from src.ablation import run_ablation_study
from src.config import Config
from src.data_source import save_requirements
from src.experiment import (SYNTHETIC_NOTICE, prepare_experiment,
                            select_analysis_pixels, select_model_samples)
from src.features import feature_dictionary
from src.holdout import run_temporal_holdout
from src.sensitivity import run_sensitivity_analysis
from src.reproducibility import start_experiment
from src.trajectory import TRAJECTORY_CODES, trajectory_codes
from src.uncertainty import uncertainty_summary, uncertainty_table
from src.validation import (fit_random_forest, random_split_baseline,
                            spatial_cv_rf)

#: Kept as module-level names because earlier phases imported them from this
#: runner; they now live in `src.experiment` and are re-exported here.
NOTICE = SYNTHETIC_NOTICE
__all__ = ["main", "quick_config", "write_analysis_layers",
           "write_trajectory_outputs", "write_temporal_diagnostics",
           "NOTICE", "select_analysis_pixels", "select_model_samples",
           "prepare_experiment"]

#: Georeferenced layers written from the feature table, as
#: (file stem, feature column, dtype, description).
FEATURE_LAYERS = (
    ("sens_slope", "sen", "float32", "Theil-Sen NDVI slope per time step"),
    ("mann_kendall_z", "mk_z", "float32",
     "Mann-Kendall Z (autocorrelation adjusted)"),
    ("mann_kendall_p", "mk_p_value", "float32", "Mann-Kendall p-value"),
    ("restrend_slope", "restrend", "float32",
     "Climate-adjusted NDVI slope; see restrend_significant"),
    ("cyclicity_enrichment", "cyc_enrichment", "float32",
     "Spectral enrichment; 1.0 = white noise, not land-use attribution"),
    ("dominant_period", "cyc_period", "float32",
     "Dominant period in time steps"),
    ("break_index", "breakpoint_index", "float32",
     "Structural break index (-1 = none)"),
    ("disturbance_magnitude", "disturbance_magnitude", "float32",
     "Pre-break level minus post-break trough"),
    ("recovery_fraction", "recovery_fraction", "float32",
     "Recovered share of the disturbance magnitude"),
    ("recovery_status", "recovery_status", "uint8",
     "0 none 1 recovered 2 recovering 3 not-recovering 4 insufficient"),
)


def quick_config(cfg: Config) -> Config:
    """Shrink an experiment to a fast end-to-end smoke run."""
    research = cfg.research
    research.samples_per_class = 30
    research.max_analysis_pixels = 3000
    research.model.n_estimators = 40
    research.model.block_cv.n_folds = 3
    research.spatial_cv.n_folds = 3
    research.sensitivity.n_estimators = 20
    research.sensitivity.parameters = research.sensitivity.parameters[:2]
    return cfg


def write_analysis_layers(prepared, exp, log, *,
                          synthetic: bool = True) -> int:
    """GeoTIFF + PNG for every temporal-analysis quantity.

    `synthetic` controls the DEVELOPMENT/SYNTHETIC banner on the figures. It
    defaults to True so M2/M3 runs keep it; the M6 real-data runner passes
    False, because labelling real observations as synthetic is as misleading
    as the reverse.
    """
    grid = prepared.analysis_grid
    shape = prepared.shape
    features, georef = prepared.features, prepared.georef
    for name, column, dtype, description in FEATURE_LAYERS:
        geo.write_layer(exp.path("predictions", f"{name}.tif"),
                        features[column].to_numpy(), grid, georef,
                        dtype=dtype, description=description)
    geo.write_layer(exp.path("predictions", "quality_flag.tif"),
                    prepared.quality.flag.astype(float),
                    np.ones(shape, bool), georef, dtype="uint8",
                    description="0 OK 1 insufficient 2 too-missing "
                                "3 constant 4 out-of-range")
    for column, title, filename, kwargs in (
            ("sen", "Theil-Sen NDVI slope", "map_trend_sen.png",
             {"center": 0}),
            ("restrend", "RESTREND adjusted slope", "map_restrend.png",
             {"center": 0}),
            ("cyc_enrichment", "Spectral enrichment (periodicity, not "
             "attribution)", "map_cyclicity.png", {"cmap": "magma"}),
            ("disturbance_magnitude", "Disturbance magnitude",
             "map_disturbance.png", {"cmap": "inferno"}),
            ("recovery_fraction", "Recovery fraction", "map_recovery.png",
             {"cmap": "YlGn"})):
        maps.save_continuous(features[column].to_numpy(), grid, shape,
                             RF.dev_title(title, synthetic),
                             exp.figure(filename), **kwargs)
    RF.plot_quality_map(prepared.quality.flag, shape,
                        exp.figure("map_data_quality.png"),
                        synthetic=synthetic)
    log.info("wrote %d georeferenced layers and 6 maps",
             len(FEATURE_LAYERS) + 1)
    return len(FEATURE_LAYERS) + 1


def write_trajectory_outputs(prepared, exp, *, synthetic: bool = True) -> None:
    """Trajectory classes as GeoTIFF, GeoJSON polygons and figures."""
    codes = trajectory_codes(prepared.trajectory_labels)
    geo.write_layer(exp.path("predictions", "trajectory_category.tif"),
                    codes, prepared.analysis_grid, prepared.georef,
                    dtype="uint8",
                    description="Analytical trajectory class: " + ", ".join(
                        f"{v}={k}" for k, v in TRAJECTORY_CODES.items()))
    grid = np.zeros(prepared.shape, dtype="int32")
    grid[prepared.analysis_grid] = codes
    geo.write_class_geojson(
        exp.path("predictions", "trajectory_category.geojson"), grid,
        prepared.georef,
        class_names={v: k for k, v in TRAJECTORY_CODES.items()},
        description="Analytical vegetation-trajectory classes; analytical "
                    "signal categories, NOT verified land cover")
    RF.plot_trajectory_map(prepared.trajectory_labels, prepared.analysis_grid,
                           prepared.shape,
                           exp.figure("map_trajectory_categories.png"),
                           synthetic=synthetic)
    RF.plot_class_distribution(prepared.trajectory_summary["counts"],
                               exp.figure("trajectory_class_distribution.png"),
                               synthetic=synthetic)


def write_temporal_diagnostics(prepared, exp, cfg: Config, *,
                               synthetic: bool = True) -> list:
    """One four-panel temporal figure per reference class.

    Covers the whole temporal story for a representative pixel: NDVI with
    its Theil-Sen trend line, the detected breakpoint and the recovery
    segment; rainfall; the RESTREND residual series with its residual
    trend; and the power spectrum with the periodicity band shaded.
    """
    if not prepared.has_labels:
        return []
    rng = np.random.default_rng(cfg.seed)
    features = prepared.features
    trough = prepared.extras["recovery"]["trough_index"]
    written = []
    for value in np.unique(prepared.labels[prepared.sample_mask]):
        candidates = np.flatnonzero(prepared.sample_mask
                                    & (prepared.labels == value))
        if candidates.size == 0:
            continue
        row = int(rng.choice(candidates))
        name = cfg.classes.get(int(value), str(value))
        written.append(RF.plot_temporal_diagnostics(
            prepared.series[:, row], prepared.rain_series[:, row],
            exp.figure(f"temporal_diagnostics_class{int(value)}.png"),
            cfg=cfg, title=f"Reference class {int(value)} ({name})",
            sen_slope=features["sen"].iloc[row],
            break_index=features["breakpoint_index"].iloc[row],
            trough_index=None if features["has_disturbance"].iloc[row] == 0
            else trough[row],
            recovery_slope=features["recovery_slope"].iloc[row],
            synthetic=synthetic))
    return written


def main(cfg: Config | None = None, *, source=None):
    cfg = cfg or Config(experiment_name="m2_synthetic_development")
    exp = start_experiment(cfg)
    log = exp.logger
    results: dict = {"data_status": NOTICE.strip()}

    # ------------------------------------ 1-3 shared preparation stages
    log.info("STAGE 1/8  data, quality control, features, trajectories, folds")
    prepared = prepare_experiment(cfg, source=source, logger=log)
    prepared.georef.save(exp.path("config", "georeference.json"))
    save_requirements(exp.path("config", "real_data_requirements.json"))
    exp.save_metrics("dataset_metadata", prepared.validation_report)
    exp.save_metrics("data_quality", prepared.quality_summary)
    exp.save_metrics("trajectory_summary", prepared.trajectory_summary)
    exp.save_metrics("experiment_summary", prepared.summary())
    exp.save_table("feature_dictionary",
                   feature_dictionary().set_index("feature"))
    exp.save_table("feature_summary", prepared.features.describe().T.round(5))
    results.update({"dataset": prepared.validation_report,
                    "data_quality": prepared.quality_summary,
                    "trajectory": prepared.trajectory_summary,
                    "experiment": prepared.summary()})

    # ------------------------------------------- 2 georeferenced analysis
    log.info("STAGE 2/8  georeferenced layers and temporal figures")
    write_analysis_layers(prepared, exp, log)
    write_trajectory_outputs(prepared, exp)
    log.info("wrote %d per-class temporal diagnostic figures",
             len(write_temporal_diagnostics(prepared, exp, cfg)))

    # -------------------------------------------------- 3 spatial folds
    log.info("STAGE 3/8  spatial block cross-validation design")
    geo.write_layer(exp.path("predictions", "spatial_cv_fold.tif"),
                    prepared.folds, prepared.analysis_grid, prepared.georef,
                    dtype="uint8",
                    description="Deterministic spatial CV fold assignment")
    RF.plot_spatial_folds(prepared.fold_grid,
                          exp.figure("spatial_cv_folds.png"))
    log.info("block_size=%d, folds=%d, buffer_blocks=%d",
             cfg.research.spatial_cv.block_size,
             cfg.research.spatial_cv.n_folds,
             cfg.research.model.block_cv.buffer_blocks)

    if not prepared.has_labels:
        log.warning("no reference labels; stopping after the unsupervised "
                    "stages")
        exp.path("metrics", "m2_notice.txt").write_text(NOTICE)
        exp.save_metrics("results", results)
        return exp, results

    features = prepared.features
    labels = prepared.labels
    folds = prepared.folds
    sample_mask = prepared.sample_mask
    columns = prepared.model_columns

    # ------------------------------------------------- 4 model + validation
    log.info("STAGE 4/8  Random Forest with spatial block cross-validation")
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
    results["spatial_block_cv"] = {k: v for k, v in metrics.items()
                                   if k != "fold_metrics"}
    uncertainty_table(cv["predictions"][evaluated],
                      cv["probabilities"][evaluated], cv["classes"],
                      truth=labels[evaluated], cfg=cfg.research.uncertainty
                      ).assign(fold=folds[evaluated]).to_csv(
        exp.path("predictions", "spatial_cv_predictions.csv"), index=False)
    exp.save_table("rf_feature_importance", cv["importance"].to_frame())
    class_names = [cfg.classes.get(int(c), str(c)) for c in cv["classes"]]
    RF.plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels"],
                             exp.figure("confusion_spatial_cv.png"),
                             class_names=class_names,
                             title="Random Forest, spatial block CV")
    RF.plot_feature_importance(cv["importance"],
                               exp.figure("feature_importance.png"))
    log.info("spatial CV: accuracy %.4f | macro F1 %.4f | weighted F1 %.4f",
             metrics["accuracy"], metrics["f1_macro"], metrics["f1_weighted"])
    log.info("fold macro F1 %.4f +/- %.4f over %d folds",
             metrics["fold_summary"]["f1_macro_mean"],
             metrics["fold_summary"]["f1_macro_std"],
             metrics["fold_summary"]["n_folds"])

    comparison = [{"method": "spatial block CV (primary)",
                   "accuracy": metrics["accuracy"],
                   "f1_macro": metrics["f1_macro"],
                   "f1_weighted": metrics["f1_weighted"],
                   "std": metrics["fold_summary"]["f1_macro_std"]}]
    if cfg.research.random_split_baseline:
        baseline = random_split_baseline(
            features, labels, sample_mask=sample_mask, feature_names=columns,
            cfg=cfg.research.model,
            test_size=cfg.research.baseline_test_size)["metrics"]
        exp.save_metrics("random_split_baseline_metrics", baseline)
        results["random_split_baseline"] = baseline
        comparison.append({"method": "random pixel split (baseline)",
                           "accuracy": baseline["accuracy"],
                           "f1_macro": baseline["f1_macro"],
                           "f1_weighted": baseline["f1_weighted"], "std": 0.0})
        log.info("random-split baseline (optimistic): accuracy %.4f | "
                 "macro F1 %.4f", baseline["accuracy"], baseline["f1_macro"])
    comparison_table = pd.DataFrame(comparison)
    exp.save_table("validation_comparison",
                   comparison_table.set_index("method"))
    RF.plot_metric_comparison(comparison_table,
                              exp.figure("validation_comparison.png"),
                              error_column="std",
                              title="Validation strategy comparison")

    # ------------------------------------------------- 5 map-wide products
    log.info("STAGE 5/8  map-wide prediction, confidence and vector exports")
    x_all = features.loc[:, columns].to_numpy(dtype="float64")
    imputer, model = fit_random_forest(x_all[sample_mask],
                                       labels[sample_mask], cfg.research.model)
    transformed = imputer.transform(x_all)
    map_prediction = model.predict(transformed)
    map_probability = model.predict_proba(transformed)
    measures = uncertainty_summary(map_probability,
                                   cfg=cfg.research.uncertainty)
    results["map_uncertainty"] = measures
    geo.write_layer(exp.path("predictions", "model_class.tif"),
                    map_prediction.astype(float), prepared.analysis_grid,
                    prepared.georef, dtype="uint8",
                    description="Random Forest class; model fitted on the "
                                "sampled pixels, not a validated map")
    geo.write_layer(exp.path("predictions", "model_confidence.tif"),
                    map_probability.max(axis=1), prepared.analysis_grid,
                    prepared.georef, dtype="float32",
                    description="Maximum class probability (model confidence "
                                "estimate, not certainty)")
    maps.save_classmap(map_prediction, prepared.analysis_grid, prepared.shape,
                       exp.figure("map_model_class.png"), cfg,
                       title=RF.dev_title("Random Forest predicted class"))
    RF.plot_probability_map(map_probability, prepared.analysis_grid,
                            prepared.shape,
                            exp.figure("map_model_confidence.png"))
    sample_grid = np.zeros(prepared.shape, bool)
    sample_grid.reshape(-1)[prepared.sample_columns] = True
    geo.write_point_geojson(
        exp.path("predictions", "model_samples.geojson"), sample_grid,
        prepared.georef,
        {"reference_class": labels[sample_mask],
         "prediction": cv["predictions"][sample_mask],
         "evaluated": evaluated[sample_mask],
         "fold": folds[sample_mask]},
        description="Model samples with spatially cross-validated "
                    "predictions; synthetic development output")
    try:
        import joblib
        joblib.dump({"model": model, "imputer": imputer,
                     "features": columns},
                    exp.path("models", "m2_random_forest.joblib"))
    except Exception as error:                          # pragma: no cover
        log.warning("model not persisted: %s", error)

    # ---------------------------------------------------------- 6 ablation
    log.info("STAGE 6/8  ablation study A-F")
    table = run_ablation_study(features, labels, folds,
                               exp.path("metrics", "ablations"), cfg,
                               sample_mask=sample_mask,
                               rf_cfg=cfg.research.model,
                               block_row=prepared.block_row,
                               block_col=prepared.block_col, logger=log)
    exp.save_table("ablation_comparison", table.set_index("experiment"))
    RF.plot_ablation_comparison(table, exp.figure("ablation_comparison.png"))
    results["ablation"] = table.to_dict(orient="records")

    # -------------------------------------------------------- 7 sensitivity
    if cfg.research.sensitivity.enabled:
        log.info("STAGE 7/8  sensitivity analysis")
        sensitivity = run_sensitivity_analysis(
            prepared.series[:, sample_mask],
            prepared.rain_series[:, sample_mask],
            exp.path("metrics", "sensitivity"), cfg,
            labels=labels[sample_mask], fold_grid=folds[sample_mask],
            logger=log)
        RF.plot_sensitivity(sensitivity, exp.figure("sensitivity.png"))
        results["sensitivity_rows"] = int(len(sensitivity))
    else:
        log.info("STAGE 7/8  sensitivity analysis disabled by configuration")

    # ---------------------------------------------------- 8 temporal holdout
    if cfg.research.holdout.enabled:
        log.info("STAGE 8/8  temporal holdout")
        holdout = run_temporal_holdout(
            prepared.series[:, sample_mask],
            prepared.rain_series[:, sample_mask], labels[sample_mask],
            exp.path("metrics", "temporal_holdout"), cfg,
            fold_grid=folds[sample_mask], logger=log)
        results["temporal_holdout"] = {
            k: v for k, v in holdout["metrics"].items()
            if k not in ("per_class", "confusion_matrix")}
    else:
        log.info("STAGE 8/8  temporal holdout disabled by configuration")

    exp.path("metrics", "m2_notice.txt").write_text(NOTICE)
    (exp.root / "config" / "research_config.json").write_text(
        json.dumps(asdict(cfg.research), indent=2, default=str))
    exp.save_metrics("results", results)
    log.info("EXPERIMENT COMPLETE -> %s", exp.root)
    return exp, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--config", type=str, default=None,
                        help="saved config.json to reproduce a run")
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--quick", action="store_true",
                        help="fast end-to-end smoke run")
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument("--no-holdout", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")
    args = parser.parse_args()

    configuration = Config.load(args.config) if args.config \
        else Config(experiment_name="m2_synthetic_development")
    if args.seed is not None:
        configuration.seed = args.seed
        configuration.research.model.seed = args.seed
        configuration.research.spatial_cv.seed = args.seed
        configuration.research.model.block_cv.seed = args.seed
    if args.name:
        configuration.experiment_name = args.name
    if args.samples_per_class is not None:
        configuration.research.samples_per_class = args.samples_per_class
    if args.quick:
        configuration = quick_config(configuration)
    if args.no_sensitivity:
        configuration.research.sensitivity.enabled = False
    if args.no_holdout:
        configuration.research.holdout.enabled = False
    if args.no_baseline:
        configuration.research.random_split_baseline = False

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main(configuration)
