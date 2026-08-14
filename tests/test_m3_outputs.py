"""End-to-end M3 experiment test and output contract (Part 11).

Runs the complete research-evaluation flow on the synthetic development
dataset and asserts that everything a reviewer would ask for is on disk:
the leakage audit, per-class metrics, uncertainty columns, explainability,
the method comparison, and the synthetic-data declaration.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def m3_run(tmp_path_factory):
    cfg = Config(experiment_name="pytest_m3_e2e", seed=5)
    if not Path(cfg.paths.ndvi_stack).exists():
        pytest.skip("synthetic stacks not generated")
    import run_m3_experiments

    run_m3_experiments.quick_config(cfg)
    cfg.research.samples_per_class = 30
    cfg.research.max_analysis_pixels = 1200
    cfg.research.model.n_estimators = 12
    cfg.research.cnn.max_epochs = 3
    cfg.research.cnn.max_folds = 2
    cfg.cyclicity.n_surrogates = 19
    cfg.paths.results = str(tmp_path_factory.mktemp("m3_results"))
    return run_m3_experiments.main(cfg)


def test_leakage_audit_runs_and_passes(m3_run):
    experiment, results = m3_run
    audit = json.loads(
        (experiment.root / "metrics" / "leakage_audit.json").read_text())
    assert audit["passed"] is True
    assert audit["n_checks"] >= 10
    names = {check["check"] for check in audit["checks"]}
    for required in ("block_purity", "label_leakage",
                     "spatial_sample_overlap", "preprocessing_isolation",
                     "buffer_separation", "temporal_separation",
                     "no_lookahead", "ablation_isolation",
                     "ablation_feature_groups"):
        assert required in names, f"missing leakage check {required}"
    assert results["leakage_audit"]["passed"] is True


def test_every_leakage_check_carries_evidence(m3_run):
    experiment, _ = m3_run
    audit = json.loads(
        (experiment.root / "metrics" / "leakage_audit.json").read_text())
    for check in audit["checks"]:
        assert check["statement"], check["check"]
        assert isinstance(check["evidence"], dict)


def test_cyclicity_surrogate_results_are_saved_with_their_caveat(m3_run):
    experiment, _ = m3_run
    report = json.loads((experiment.root / "metrics"
                         / "cyclicity_surrogate_test.json").read_text())
    assert report["n_surrogates"] >= 19
    assert 0.0 <= report["significant_fraction"] <= 1.0
    assert "Periodicity is not jhum" in report["interpretation"]
    table = pd.read_csv(experiment.root / "metrics"
                        / "cyclicity_surrogate.csv")
    assert {"p_value", "significant", "enrichment"} <= set(table.columns)
    assert ((table["p_value"] > 0) & (table["p_value"] <= 1)).all()


def test_uncertainty_outputs_are_complete(m3_run):
    experiment, results = m3_run
    frame = pd.read_csv(experiment.root / "predictions"
                        / "rf_predictions_uncertainty.csv")
    for column in ("truth", "prediction", "confidence", "margin", "entropy",
                   "uncertain"):
        assert column in frame.columns
    probability_columns = [c for c in frame.columns
                           if c.startswith("probability_")]
    assert probability_columns
    assert np.allclose(frame[probability_columns].sum(axis=1), 1.0, atol=1e-6)
    summary = results["spatial_block_cv"]["uncertainty"]
    assert "uncertain_fraction" in summary
    assert "not certainty" in summary["disclaimer"]


def test_explainability_outputs_exist_and_are_qualified(m3_run):
    experiment, results = m3_run
    root = experiment.root / "metrics" / "explainability"
    permutation = pd.read_csv(root / "permutation_importance.csv")
    assert {"feature", "importance_mean"} <= set(permutation.columns)
    assert (experiment.root / "metrics" / "explainability"
            / "importance_comparison.csv").exists()
    report = json.loads((root / "explainability.json").read_text())
    assert "not causal influence" in report["disclaimer"]
    assert results["explainability"]["permutation"]["top"]


def test_experiment_matrix_compares_all_configured_methods(m3_run):
    experiment, _ = m3_run
    root = experiment.root / "metrics" / "experiment_matrix"
    matrix = pd.read_csv(root / "experiment_matrix.csv")
    methods = set(matrix["method"])
    assert {"baseline_trend", "rf_basic", "rf_proposed"} <= methods
    binary = matrix[matrix["task"] == "binary_degradation"]
    assert len(binary) >= 3
    saved = json.loads((root / "experiment_matrix.json").read_text())
    assert saved["conclusion"]["available"]
    assert "synthetic" in saved["caveat"]
    for column in ("f1_macro", "precision_macro", "recall_macro",
                   "cohen_kappa", "fold_f1_macro_std"):
        assert column in matrix.columns


def test_cnn_participates_or_records_why_not(m3_run):
    experiment, _ = m3_run
    root = experiment.root / "metrics" / "experiment_matrix"
    saved = json.loads((root / "experiment_matrix.json").read_text())
    methods = {row["method"] for row in saved["rows"]}
    skipped = {entry["method"] for entry in saved["skipped"]}
    assert "cnn_1d" in methods or "cnn_1d" in skipped
    if "cnn_1d" in methods:
        metrics = json.loads(
            (root / "cnn_binary_degradation" / "metrics.json").read_text())
        assert metrics["model"] == "cnn_1d"
        assert "per_class" in metrics and "fold_summary" in metrics
        assert (root / "cnn_binary_degradation"
                / "training_history.csv").exists()
        assert list((root / "cnn_binary_degradation").glob("checkpoint*.pt"))


def test_buffer_sensitivity_is_reported(m3_run):
    experiment, results = m3_run
    table = pd.read_csv(experiment.root / "metrics"
                        / "spatial_buffer_sensitivity.csv")
    assert "f1_macro" in table.columns
    assert len(results["spatial_buffer_sensitivity"]) >= 1


def test_temporal_holdout_is_audited_and_qualified(m3_run):
    experiment, results = m3_run
    root = experiment.root / "metrics" / "temporal_holdout"
    metrics = json.loads((root / "metrics.json").read_text())
    assert metrics["meaningful_on_this_dataset"] is False
    audit = json.loads((root / "leakage_audit.json").read_text())
    assert audit["passed"] is True
    assert results["temporal_holdout"]["leakage_checks_passed"] is True


def test_figures_and_georeferenced_layers_are_written(m3_run):
    import rasterio
    from src import geo

    experiment, _ = m3_run
    figures = {p.name for p in (experiment.root / "figures").glob("*.png")}
    expected = {
        # temporal / spatial analysis
        "map_trajectory_categories.png", "map_trend_sen.png",
        "map_restrend.png", "map_cyclicity.png", "map_disturbance.png",
        "map_recovery.png", "map_data_quality.png",
        "trajectory_class_distribution.png",
        # model
        "spatial_cv_folds.png", "confusion_spatial_cv.png",
        "feature_importance_impurity.png",
        "feature_importance_permutation.png",
        "experiment_matrix_binary.png", "experiment_matrix_multiclass.png",
        "ablation_comparison.png",
        # uncertainty
        "map_model_confidence.png", "map_uncertainty_flag.png",
        "map_degradation_probability.png", "map_model_class.png"}
    assert expected <= figures, f"missing figures: {expected - figures}"
    assert any(f.startswith("temporal_diagnostics_class") for f in figures), \
        "per-class temporal diagnostics are missing"
    for figure in (experiment.root / "figures").glob("*.png"):
        assert figure.stat().st_size > 1000, f"{figure.name} looks empty"

    source = geo.GeoRef.from_raster(Config().paths.ndvi_stack)
    layers = list((experiment.root / "predictions").glob("*.tif"))
    assert len(layers) >= 12
    for layer in layers:
        with rasterio.open(layer) as raster:
            assert raster.crs == source.crs
            assert (raster.height, raster.width) == source.shape
            assert raster.nodata is not None
            assert np.allclose(list(raster.transform)[:6],
                               list(source.transform)[:6])
            assert raster.res == source.resolution


def test_cnn_learning_curves_are_plotted_when_the_cnn_runs(m3_run):
    experiment, _ = m3_run
    history = (experiment.root / "metrics" / "experiment_matrix"
               / "cnn_multiclass_trajectory" / "training_history.csv")
    if not history.exists():
        pytest.skip("CNN did not run in this configuration")
    assert (experiment.root / "figures" / "cnn_learning_curves.png").exists()


def test_vector_exports_are_written_and_valid(m3_run):
    experiment, _ = m3_run
    points = json.loads((experiment.root / "predictions"
                         / "model_samples.geojson").read_text())
    assert points["type"] == "FeatureCollection"
    assert points["features"]
    properties = points["features"][0]["properties"]
    for key in ("reference_class", "prediction", "confidence", "uncertain",
                "fold"):
        assert key in properties
    polygons = json.loads((experiment.root / "predictions"
                           / "trajectory_category.geojson").read_text())
    assert polygons["n_polygons"] >= 1
    assert "class_name" in polygons["features"][0]["properties"]
    assert "NOT verified land cover" in polygons["description"]


def test_uncertainty_layers_are_exported(m3_run):
    import rasterio

    experiment, results = m3_run
    with rasterio.open(experiment.root / "predictions"
                       / "model_uncertain_flag.tif") as raster:
        flags = raster.read(1, masked=True)
    assert set(np.unique(flags.compressed())) <= {0, 1}
    with rasterio.open(experiment.root / "predictions"
                       / "degradation_probability.tif") as raster:
        probability = raster.read(1, masked=True).compressed()
    assert probability.min() >= 0.0 and probability.max() <= 1.0
    assert "uncertain_fraction" in results["map_uncertainty"]


def test_real_data_contract_is_saved_with_the_run(m3_run):
    experiment, _ = m3_run
    contract = json.loads((experiment.root / "config"
                           / "real_data_requirements.json").read_text())
    assert contract["contract"] == "StandardizedDataset"
    assert len(contract["requirements"]) >= 8


def test_experiment_summary_records_the_run_shape(m3_run):
    experiment, results = m3_run
    summary = json.loads((experiment.root / "metrics"
                          / "experiment_summary.json").read_text())
    assert summary["n_analysed"] > 0
    assert summary["n_model_samples"] > 0
    from src.features import feature_names
    assert summary["n_features"] == len(feature_names())
    assert results["experiment"]["n_folds"] >= 2


def test_configuration_is_snapshotted_and_replayable(m3_run):
    experiment, _ = m3_run
    restored = Config.load(experiment.root / "config" / "config.json")
    assert restored.research.cnn.max_epochs == 3
    assert restored.research.uncertainty.confidence_threshold > 0
    assert restored.research.explain.permutation_repeats >= 1
    assert restored.research.matrix.degradation_classes == [4]
    assert (experiment.root / "config" / "research_config.json").exists()


def test_synthetic_status_is_declared(m3_run):
    experiment, results = m3_run
    notice = (experiment.root / "metrics" / "m3_notice.txt").read_text()
    assert "not real observations" in notice
    assert "Synthetic data" in results["data_status"]
    assert "not certainty" in results["confidence_disclaimer"]
