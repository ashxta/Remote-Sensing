"""M3 research evaluation runner.

Adds to the M2 flow the parts that make the evaluation research-grade:

    leakage audit  -> executable assertions saved with every run
    cyclicity      -> surrogate-data significance for periodicity
    uncertainty    -> confidence, margin, entropy, low-confidence flags
    explainability -> held-out permutation importance, optional SHAP
    experiment     -> conventional trend rule vs Random Forest vs 1D CNN vs
    matrix            the proposed feature framework, one common test set
    temporal       -> optional historical/later holdout, re-audited

Stages 1-3 are `experiment.prepare_experiment`, shared with the M2 runner.

    python run_m3_experiments.py
    python run_m3_experiments.py --quick        # fast smoke run
    python run_m3_experiments.py --no-cnn --no-shap
    python run_m3_experiments.py --config results/<id>/config/config.json

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

from run_m2_experiments import (write_analysis_layers,
                                write_temporal_diagnostics,
                                write_trajectory_outputs)
from src import geo, maps
from src import research_figures as RF
from src import timeseries as TS
from src.ablation import run_ablation_study
from src.config import Config
from src.data_source import save_requirements
from src.experiment import SYNTHETIC_NOTICE, prepare_experiment
from src.experiment_matrix import run_experiment_matrix
from src.explain import explain_experiment
from src.features import FEATURE_GROUPS, build_feature_table, feature_dictionary, feature_names
from src.holdout import resolve_cutoff, run_temporal_holdout
from src.leakage import (audit_report, check_ablation_feature_groups,
                         check_ablation_isolation, check_block_purity,
                         check_buffer_separation, check_label_leakage,
                         check_no_lookahead, check_preprocessing_isolation,
                         check_spatial_separation, check_temporal_separation)
from src.reproducibility import start_experiment
from src.sensitivity import run_sensitivity_analysis
from src.uncertainty import (CONFIDENCE_DISCLAIMER, prediction_confidence,
                             uncertainty_summary, uncertainty_table)
from src.validation import fit_random_forest, random_split_baseline, spatial_cv_rf

NOTICE = SYNTHETIC_NOTICE


def quick_config(cfg: Config) -> Config:
    """Shrink the whole matrix to a fast end-to-end smoke run."""
    research = cfg.research
    research.samples_per_class = 40
    research.max_analysis_pixels = 3000
    research.model.n_estimators = 40
    research.model.block_cv.n_folds = 4
    research.spatial_cv.n_folds = 4
    research.sensitivity.n_estimators = 20
    research.sensitivity.parameters = research.sensitivity.parameters[:2]
    research.cnn.max_epochs = 6
    research.cnn.patience = 3
    research.cnn.max_folds = 2
    research.explain.permutation_repeats = 2
    research.explain.shap_max_samples = 60
    research.matrix.buffer_sensitivity = [0]
    cfg.cyclicity.n_surrogates = 39
    return cfg


def _historical_builder(cfg: Config):
    """Builder handed to the no-lookahead check: it must slice, not peek."""
    def build(full_ndvi, full_rain, cutoff):
        return build_feature_table(full_ndvi[:cutoff], full_rain[:cutoff], cfg)
    return build


def run_leakage_audit(prepared, cfg: Config, exp, log) -> dict:
    """Every leakage claim, re-evaluated as an assertion on this run."""
    checks = [check_block_purity(prepared.block_id_grid, prepared.fold_grid),
              check_label_leakage(prepared.features, prepared.labels,
                                  prepared.model_columns)]
    for fold in sorted(np.unique(prepared.folds[prepared.sample_mask])):
        test = prepared.sample_mask & (prepared.folds == fold)
        train = prepared.sample_mask & ~test
        checks.append(check_spatial_separation(train, test))
        checks.append(check_preprocessing_isolation(train, test))
        checks.append(check_buffer_separation(
            train, test, prepared.block_row, prepared.block_col,
            cfg.research.model.block_cv.buffer_blocks))

    n_time = prepared.dataset.n_time
    cutoff = resolve_cutoff(n_time, cfg.research.holdout)
    checks.append(check_temporal_separation(np.arange(cutoff),
                                            np.arange(cutoff, n_time),
                                            cutoff, n_time))
    probe = prepared.sample_columns[:40]
    ndvi_flat, rain_flat = prepared.dataset.flat()
    checks.append(check_no_lookahead(_historical_builder(cfg),
                                     ndvi_flat[:, probe], rain_flat[:, probe],
                                     cutoff))
    ablation_sets = {e.name: list(feature_names(e.groups))
                     for e in cfg.research.ablation.experiments}
    checks.append(check_ablation_isolation(ablation_sets, FEATURE_GROUPS))
    checks.append(check_ablation_feature_groups(
        cfg.research.ablation.experiments,
        lambda e: feature_names(e.groups), FEATURE_GROUPS))

    audit = audit_report(checks, exp.path("metrics", "leakage_audit.json"),
                         strict=True,
                         context={"block_size": cfg.research.spatial_cv.block_size,
                                  "n_folds": cfg.research.spatial_cv.n_folds,
                                  "buffer_blocks":
                                      cfg.research.model.block_cv.buffer_blocks,
                                  "seed": cfg.seed})
    log.info("leakage audit: %d checks, all passed", audit["n_checks"])
    return audit


def run_cyclicity_significance(prepared, cfg: Config, exp, log) -> dict:
    """Red-noise surrogate significance for the periodicity statistic."""
    sample = prepared.sample_mask
    surrogate = TS.cyclicity_significance(
        prepared.series[:, sample], min_period=cfg.cyclicity.min_period,
        max_period=cfg.cyclicity.max_period, detrend=cfg.cyclicity.detrend,
        min_obs=cfg.cyclicity.min_obs, n_surrogates=cfg.cyclicity.n_surrogates,
        alpha=cfg.cyclicity.surrogate_alpha, seed=cfg.cyclicity.surrogate_seed)
    enrichment = prepared.features["cyc_enrichment"].to_numpy()[sample]
    threshold_flag = prepared.features["cyclicity_periodic"].to_numpy(
        bool)[sample]
    report = {
        "null_model": surrogate["null"],
        "n_surrogates": surrogate["n_surrogates"],
        "alpha": cfg.cyclicity.surrogate_alpha,
        "n_samples": int(sample.sum()),
        "significant_fraction": float(np.mean(surrogate["significant"])),
        "threshold_flag_fraction": float(np.mean(threshold_flag)),
        "agreement_with_threshold_rule": float(
            np.mean(surrogate["significant"] == threshold_flag)),
        "median_p_value": float(np.nanmedian(surrogate["p_value"])),
        "mean_enrichment": float(np.nanmean(enrichment)),
        "mean_ar1_coefficient": float(
            np.nanmean(surrogate["ar1_coefficient"])),
        "interpretation": (
            "A small p-value means the concentration of power inside the "
            "period band is unlikely for a series with this pixel's own "
            "autocorrelation. It does NOT identify a cause: rotational "
            "cultivation, plantation harvest cycles, fire-regrowth cycles "
            "and multi-year climate oscillation all produce periodicity. "
            "Periodicity is not jhum."),
    }
    exp.save_metrics("cyclicity_surrogate_test", report)
    pd.DataFrame({"p_value": surrogate["p_value"],
                  "significant": surrogate["significant"],
                  "observed_band_fraction": surrogate["observed_fraction"],
                  "surrogate_mean_fraction":
                      surrogate["surrogate_mean_fraction"],
                  "ar1_coefficient": surrogate["ar1_coefficient"],
                  "enrichment": enrichment,
                  "threshold_rule_flag": threshold_flag}
                 ).to_csv(exp.path("metrics", "cyclicity_surrogate.csv"),
                          index=False)
    log.info("surrogate test (%s null): %.1f%% significant at alpha=%.2f "
             "(threshold rule flags %.1f%%); agreement %.1f%%",
             surrogate["null"], 100 * report["significant_fraction"],
             cfg.cyclicity.surrogate_alpha,
             100 * report["threshold_flag_fraction"],
             100 * report["agreement_with_threshold_rule"])
    return report


def main(cfg: Config | None = None, *, source=None):
    cfg = cfg or Config(experiment_name="m3_synthetic_development")
    exp = start_experiment(cfg)
    log = exp.logger
    results: dict = {"data_status": NOTICE.strip(),
                     "confidence_disclaimer": CONFIDENCE_DISCLAIMER}

    # ------------------------------------ 1 shared preparation stages
    log.info("STAGE 1/9  data, quality control, features, trajectories, folds")
    prepared = prepare_experiment(cfg, source=source, logger=log)
    if not prepared.has_labels:
        raise RuntimeError("the M3 evaluation needs reference labels; the "
                           "configured data source provided none")
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

    features = prepared.features
    labels = prepared.labels
    folds = prepared.folds
    sample_mask = prepared.sample_mask
    columns = prepared.model_columns

    # ---------------------------------------- 2 georeferenced analysis
    log.info("STAGE 2/9  georeferenced layers and temporal figures")
    write_analysis_layers(prepared, exp, log)
    write_trajectory_outputs(prepared, exp)
    log.info("wrote %d per-class temporal diagnostic figures",
             len(write_temporal_diagnostics(prepared, exp, cfg)))
    geo.write_layer(exp.path("predictions", "spatial_cv_fold.tif"),
                    folds, prepared.analysis_grid, prepared.georef,
                    dtype="uint8",
                    description="Deterministic spatial CV fold assignment")
    RF.plot_spatial_folds(prepared.fold_grid,
                          exp.figure("spatial_cv_folds.png"))

    # ------------------------------------------------------ 3 leakage audit
    log.info("STAGE 3/9  leakage audit")
    audit = run_leakage_audit(prepared, cfg, exp, log)
    results["leakage_audit"] = {"n_checks": audit["n_checks"],
                                "passed": audit["passed"]}

    # ---------------------------------------- 4 cyclicity surrogate testing
    log.info("STAGE 4/9  cyclicity surrogate-data significance")
    results["cyclicity_surrogate_test"] = run_cyclicity_significance(
        prepared, cfg, exp, log)

    # ------------------------------------- 5 primary model + uncertainty
    log.info("STAGE 5/9  Random Forest, spatial block CV and uncertainty")
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
        exp.path("predictions", "rf_predictions_uncertainty.csv"), index=False)
    class_names = [cfg.classes.get(int(c), str(c)) for c in cv["classes"]]
    RF.plot_confusion_matrix(metrics["confusion_matrix"], metrics["labels"],
                             exp.figure("confusion_spatial_cv.png"),
                             class_names=class_names,
                             title="Random Forest, spatial block CV")
    log.info("RF spatial CV: accuracy %.4f | macro F1 %.4f | fold macro F1 "
             "%.4f +/- %.4f", metrics["accuracy"], metrics["f1_macro"],
             metrics["fold_summary"]["f1_macro_mean"],
             metrics["fold_summary"]["f1_macro_std"])
    unc = metrics["uncertainty"]
    log.info("uncertainty: %.1f%% flagged; accuracy confident %s vs flagged %s",
             100 * unc["uncertain_fraction"],
             unc["accuracy_confident_subset"], unc["accuracy_uncertain_subset"])

    baseline = random_split_baseline(
        features, labels, sample_mask=sample_mask, feature_names=columns,
        cfg=cfg.research.model,
        test_size=cfg.research.baseline_test_size)["metrics"]
    exp.save_metrics("random_split_baseline_metrics", baseline)
    results["random_split_baseline"] = baseline

    buffer_rows = []
    for buffer in cfg.research.matrix.buffer_sensitivity:
        model_cfg = Config.from_dict(cfg.to_dict()).research.model
        model_cfg.block_cv.buffer_blocks = int(buffer)
        result = spatial_cv_rf(features, labels, folds,
                               sample_mask=sample_mask, feature_names=columns,
                               cfg=model_cfg, block_row=prepared.block_row,
                               block_col=prepared.block_col)
        buffer_rows.append({
            "buffer_blocks": int(buffer),
            "accuracy": result["metrics"]["accuracy"],
            "f1_macro": result["metrics"]["f1_macro"],
            "fold_f1_macro_std":
                result["metrics"]["fold_summary"]["f1_macro_std"],
            "n_evaluated": result["metrics"]["n_evaluated"]})
    exp.save_table("spatial_buffer_sensitivity",
                   pd.DataFrame(buffer_rows).set_index("buffer_blocks"))
    results["spatial_buffer_sensitivity"] = buffer_rows
    log.info("buffer sensitivity: %s", ", ".join(
        f"buffer={r['buffer_blocks']} macro F1 {r['f1_macro']:.4f}"
        for r in buffer_rows))

    # -------------------------------------- 6 map-wide products + exports
    log.info("STAGE 6/9  map-wide prediction, confidence and vector exports")
    x_all = features.loc[:, columns].to_numpy(dtype="float64")
    imputer, model = fit_random_forest(x_all[sample_mask], labels[sample_mask],
                                       cfg.research.model)
    transformed = imputer.transform(x_all)
    map_prediction = model.predict(transformed)
    map_probability = model.predict_proba(transformed)
    map_measures = prediction_confidence(map_probability,
                                         cfg=cfg.research.uncertainty)
    results["map_uncertainty"] = uncertainty_summary(
        map_probability, cfg=cfg.research.uncertainty)
    geo.write_layer(exp.path("predictions", "model_class.tif"),
                    map_prediction.astype(float), prepared.analysis_grid,
                    prepared.georef, dtype="uint8",
                    description="Random Forest class; model fitted on the "
                                "sampled pixels, not a validated map")
    geo.write_layer(exp.path("predictions", "model_confidence.tif"),
                    map_measures["confidence"], prepared.analysis_grid,
                    prepared.georef, dtype="float32",
                    description="Maximum class probability (model confidence "
                                "estimate, not certainty)")
    geo.write_layer(exp.path("predictions", "model_uncertain_flag.tif"),
                    map_measures["uncertain"].astype(float),
                    prepared.analysis_grid, prepared.georef, dtype="uint8",
                    description="1 where the prediction is flagged "
                                "low-confidence by the configured thresholds")
    degradation = [i for i, c in enumerate(model.classes_)
                   if int(c) in cfg.research.matrix.degradation_classes]
    if degradation:
        geo.write_layer(
            exp.path("predictions", "degradation_probability.tif"),
            map_probability[:, degradation].sum(axis=1),
            prepared.analysis_grid, prepared.georef, dtype="float32",
            description="Model probability assigned to the reference "
                        "degradation class(es); a model score, not a "
                        "probability that the ground is degraded")
        maps.save_continuous(
            map_probability[:, degradation].sum(axis=1),
            prepared.analysis_grid, prepared.shape,
            RF.dev_title("Model probability of the degradation class\n"
                         "(model score, not certainty)"),
            exp.figure("map_degradation_probability.png"), cmap="inferno")
    maps.save_classmap(map_prediction, prepared.analysis_grid, prepared.shape,
                       exp.figure("map_model_class.png"), cfg,
                       title=RF.dev_title("Random Forest predicted class"))
    RF.plot_probability_map(map_probability, prepared.analysis_grid,
                            prepared.shape,
                            exp.figure("map_model_confidence.png"))
    RF.plot_uncertainty_map(map_measures["uncertain"].astype(float),
                            prepared.analysis_grid, prepared.shape,
                            exp.figure("map_uncertainty_flag.png"))
    sample_grid = np.zeros(prepared.shape, bool)
    sample_grid.reshape(-1)[prepared.sample_columns] = True
    sample_measures = prediction_confidence(
        cv["probabilities"][sample_mask], cfg=cfg.research.uncertainty)
    geo.write_point_geojson(
        exp.path("predictions", "model_samples.geojson"), sample_grid,
        prepared.georef,
        {"reference_class": labels[sample_mask],
         "prediction": cv["predictions"][sample_mask],
         "confidence": sample_measures["confidence"],
         "uncertain": sample_measures["uncertain"],
         "fold": folds[sample_mask],
         "evaluated": evaluated[sample_mask]},
        description="Model samples with spatially cross-validated "
                    "predictions and confidence; synthetic development "
                    "output")
    try:
        import joblib
        joblib.dump({"model": model, "imputer": imputer, "features": columns},
                    exp.path("models", "m3_random_forest.joblib"))
    except Exception as error:                          # pragma: no cover
        log.warning("model not persisted: %s", error)

    # --------------------------------------------------- 7 explainability
    log.info("STAGE 7/9  explainability")
    explanation = explain_experiment(
        features, labels, folds, exp.path("metrics", "explainability"),
        feature_names=columns, impurity=cv["importance"],
        sample_mask=sample_mask, rf_cfg=cfg.research.model,
        cfg=cfg.research.explain, logger=log)
    exp.save_table("rf_feature_importance", cv["importance"].to_frame())
    RF.plot_feature_importance(cv["importance"],
                               exp.figure("feature_importance_impurity.png"))
    RF.plot_feature_importance(
        explanation["permutation"].set_index("feature")["importance_mean"],
        exp.figure("feature_importance_permutation.png"),
        top_n=cfg.research.explain.top_n,
        title="Permutation importance (held-out folds, macro F1 drop)")
    results["explainability"] = explanation["report"]

    # ------------------------------------------------- 8 experiment matrix
    log.info("STAGE 8/9  experiment matrix: baseline vs RF vs CNN vs proposed")
    # The CNN receives BOTH channels: the Random Forest sees rainfall
    # through the engineered features, so an NDVI-only CNN would be judged
    # on strictly less information than its competitor.
    cnn_series = np.stack([prepared.series, prepared.rain_series])
    matrix = run_experiment_matrix(
        features, labels, folds, exp.path("metrics", "experiment_matrix"),
        cfg, series=cnn_series, channel_names=["ndvi", "rainfall"],
        sample_mask=sample_mask, block_row=prepared.block_row,
        block_col=prepared.block_col, logger=log)
    exp.save_table("experiment_matrix", matrix.set_index(["method", "task"]))
    results["experiment_matrix"] = matrix.to_dict(orient="records")
    for task, filename, title in (
            ("binary_degradation", "experiment_matrix_binary.png",
             "Degradation detection: conventional trend rule vs learned "
             "methods"),
            ("multiclass_trajectory", "experiment_matrix_multiclass.png",
             "Multiclass trajectory classification by method")):
        subset = matrix[matrix["task"] == task]
        if not subset.empty:
            RF.plot_metric_comparison(subset, exp.figure(filename),
                                      error_column="fold_f1_macro_std",
                                      title=title)
    history_path = (exp.path("metrics", "experiment_matrix")
                    / "cnn_multiclass_trajectory" / "training_history.csv")
    if history_path.exists():
        RF.plot_learning_curves(pd.read_csv(history_path),
                                exp.figure("cnn_learning_curves.png"))
    # The research question, made visible: which behaviours does each method
    # mistake for degradation?
    confusion_path = (exp.path("metrics", "experiment_matrix")
                      / "discrimination" / "confounder_confusion.csv")
    if confusion_path.exists():
        confounders = pd.read_csv(confusion_path)
        RF.plot_discrimination(confounders,
                               exp.figure("degradation_discrimination.png"))
        summary_path = (exp.path("metrics", "experiment_matrix")
                        / "discrimination" / "discrimination_summary.csv")
        discrimination = pd.read_csv(summary_path)
        exp.save_table("degradation_discrimination",
                       discrimination.set_index("method"))
        results["discrimination"] = discrimination.to_dict(orient="records")

    # ------------------------------------ 9 ablation, sensitivity, holdout
    log.info("STAGE 9/9  ablation, sensitivity and temporal holdout")
    ablation = run_ablation_study(features, labels, folds,
                                  exp.path("metrics", "ablations"), cfg,
                                  sample_mask=sample_mask,
                                  rf_cfg=cfg.research.model,
                                  block_row=prepared.block_row,
                                  block_col=prepared.block_col, logger=log)
    exp.save_table("ablation_comparison", ablation.set_index("experiment"))
    RF.plot_ablation_comparison(ablation, exp.figure("ablation_comparison.png"))
    results["ablation"] = ablation.to_dict(orient="records")

    if cfg.research.sensitivity.enabled:
        sensitivity = run_sensitivity_analysis(
            prepared.series[:, sample_mask],
            prepared.rain_series[:, sample_mask],
            exp.path("metrics", "sensitivity"), cfg,
            labels=labels[sample_mask], fold_grid=folds[sample_mask],
            logger=log)
        RF.plot_sensitivity(sensitivity, exp.figure("sensitivity.png"))
        results["sensitivity_rows"] = int(len(sensitivity))

    if cfg.research.holdout.enabled:
        holdout = run_temporal_holdout(
            prepared.series[:, sample_mask],
            prepared.rain_series[:, sample_mask], labels[sample_mask],
            exp.path("metrics", "temporal_holdout"), cfg,
            fold_grid=folds[sample_mask], logger=log)
        results["temporal_holdout"] = {
            k: v for k, v in holdout["metrics"].items()
            if k not in ("per_class", "confusion_matrix")}
        n_time = prepared.dataset.n_time
        probe = prepared.sample_columns[:40]
        ndvi_flat, rain_flat = prepared.dataset.flat()
        post = audit_report(
            [check_temporal_separation(np.arange(holdout["cutoff"]),
                                       np.arange(holdout["cutoff"], n_time),
                                       holdout["cutoff"], n_time),
             check_no_lookahead(_historical_builder(cfg),
                                ndvi_flat[:, probe], rain_flat[:, probe],
                                holdout["cutoff"])],
            exp.path("metrics", "temporal_holdout") / "leakage_audit.json",
            strict=True, context={"cutoff": holdout["cutoff"]})
        results["temporal_holdout"]["leakage_checks_passed"] = post["passed"]

    exp.path("metrics", "m3_notice.txt").write_text(NOTICE)
    (exp.root / "config" / "research_config.json").write_text(
        json.dumps(asdict(cfg.research), indent=2, default=str))
    exp.save_metrics("results", results)
    log.info("EXPERIMENT COMPLETE -> %s", exp.root)
    return exp, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-cnn", action="store_true")
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--no-sensitivity", action="store_true")
    parser.add_argument("--no-holdout", action="store_true")
    args = parser.parse_args()

    configuration = Config.load(args.config) if args.config \
        else Config(experiment_name="m3_synthetic_development")
    if args.seed is not None:
        configuration.seed = args.seed
        for section in (configuration.research.model,
                        configuration.research.spatial_cv,
                        configuration.research.model.block_cv,
                        configuration.research.cnn):
            section.seed = args.seed
    if args.name:
        configuration.experiment_name = args.name
    if args.samples_per_class is not None:
        configuration.research.samples_per_class = args.samples_per_class
    if args.quick:
        configuration = quick_config(configuration)
    if args.no_cnn:
        configuration.research.cnn.enabled = False
        configuration.research.matrix.run_cnn = False
    if args.no_shap:
        configuration.research.explain.shap = False
    if args.no_sensitivity:
        configuration.research.sensitivity.enabled = False
    if args.no_holdout:
        configuration.research.holdout.enabled = False

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main(configuration)
