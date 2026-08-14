"""End-to-end M2 experiment smoke test and output contract (Part 12).

Runs the complete research flow

    data -> quality control -> temporal analysis -> feature engineering
    -> model -> validation -> metrics -> figures -> saved outputs

on the synthetic development dataset, then asserts that the experiment
directory is self-describing: configuration, metrics, predictions, figures
and georeferenced layers must all be present, and the geospatial metadata of
the input must survive to the outputs.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Config

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def m2_run(tmp_path_factory):
    """One quick end-to-end M2 experiment, shared by the tests below."""
    cfg = Config(experiment_name="pytest_m2_e2e", seed=3)
    if not Path(cfg.paths.ndvi_stack).exists():
        pytest.skip("synthetic stacks not generated")
    import run_m2_experiments

    run_m2_experiments.quick_config(cfg)
    cfg.research.samples_per_class = 24
    cfg.research.max_analysis_pixels = 1500
    cfg.research.model.n_estimators = 12
    cfg.research.sensitivity.n_estimators = 8
    cfg.research.sensitivity.evaluate_model = False
    cfg.paths.results = str(tmp_path_factory.mktemp("m2_results"))
    experiment, results = run_m2_experiments.main(cfg)
    return experiment, results


def test_experiment_completes_and_saves_configuration(m2_run):
    experiment, _ = m2_run
    saved = json.loads((experiment.root / "config" / "config.json").read_text())
    assert saved["seed"] == 3
    assert saved["research"]["samples_per_class"] == 24
    assert "spatial_cv" in saved["research"]
    assert (experiment.root / "config" / "environment.json").exists()
    assert (experiment.root / "config" / "georeference.json").exists()
    assert (experiment.root / "config" / "research_config.json").exists()


def test_configuration_snapshot_round_trips(m2_run):
    experiment, _ = m2_run
    restored = Config.load(experiment.root / "config" / "config.json")
    assert restored.research.samples_per_class == 24
    assert restored.research.model.n_estimators == 12
    assert len(restored.research.ablation.experiments) == 6
    assert restored.research.sensitivity.parameters[0].path


def test_metrics_are_machine_readable(m2_run):
    experiment, results = m2_run
    metrics = json.loads(
        (experiment.root / "metrics" / "spatial_cv_metrics.json").read_text())
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                "f1_weighted", "per_class", "confusion_matrix", "labels",
                "fold_metrics", "fold_summary"):
        assert key in metrics, f"missing metric {key}"
    assert metrics["validation"] == "spatial_block_cv"
    for per_class in metrics["per_class"].values():
        assert {"precision", "recall", "f1", "support"} <= set(per_class)
    assert "data_quality" in results and "trajectory" in results


def test_spatial_cv_is_the_primary_validation_and_baseline_is_labelled(m2_run):
    experiment, results = m2_run
    baseline = json.loads((experiment.root / "metrics"
                           / "random_split_baseline_metrics.json").read_text())
    assert baseline["validation"] == "random_pixel_split_baseline"
    assert "optimistic" in baseline["caveat"]
    assert results["spatial_block_cv"]["validation"] == "spatial_block_cv"


def test_predictions_and_probabilities_are_saved(m2_run):
    """One tidy per-sample table: truth, prediction, scores, uncertainty."""
    experiment, _ = m2_run
    predictions = pd.read_csv(experiment.root / "predictions"
                              / "spatial_cv_predictions.csv")
    for column in ("truth", "prediction", "correct", "fold", "confidence",
                   "margin", "entropy", "uncertain"):
        assert column in predictions.columns, column
    probability_columns = [c for c in predictions.columns
                           if c.startswith("probability_")]
    assert probability_columns
    rows = predictions[probability_columns].to_numpy()
    assert np.allclose(np.nansum(rows, axis=1), 1.0, atol=1e-6)
    assert (predictions["confidence"] <= 1.0).all()


def test_vector_and_quality_outputs_exist(m2_run):
    experiment, _ = m2_run
    points = json.loads((experiment.root / "predictions"
                         / "model_samples.geojson").read_text())
    assert points["type"] == "FeatureCollection" and points["features"]
    polygons = json.loads((experiment.root / "predictions"
                           / "trajectory_category.geojson").read_text())
    assert polygons["n_polygons"] >= 1
    assert (experiment.root / "figures" / "map_data_quality.png").exists()
    assert (experiment.root / "config"
            / "real_data_requirements.json").exists()


def test_feature_dictionary_and_importance_are_saved(m2_run):
    experiment, _ = m2_run
    dictionary = pd.read_csv(experiment.root / "metrics"
                             / "feature_dictionary.csv")
    assert {"group", "description", "source"} <= set(dictionary.columns)
    importance = pd.read_csv(experiment.root / "metrics"
                             / "rf_feature_importance.csv", index_col=0)
    assert (importance["importance"] >= 0).all()
    assert set(importance.index) <= set(dictionary["feature"])


def test_ablation_outputs_cover_every_experiment(m2_run):
    experiment, _ = m2_run
    table = pd.read_csv(experiment.root / "metrics"
                        / "ablation_comparison.csv")
    assert list(table["experiment"]) == list("ABCDEF")
    assert table["n_features"].is_monotonic_increasing
    for column in ("accuracy", "f1_macro", "f1_weighted",
                   "fold_f1_macro_mean", "fold_f1_macro_std"):
        assert column in table.columns
    cells = list((experiment.root / "metrics" / "ablations").glob("*/"))
    assert len(cells) == 6


def test_sensitivity_outputs_exist(m2_run):
    experiment, _ = m2_run
    root = experiment.root / "metrics" / "sensitivity"
    table = pd.read_csv(root / "sensitivity_results.csv")
    assert "(baseline)" in set(table["parameter"])
    assert len(table) > 1
    saved = json.loads((root / "sensitivity_results.json").read_text())
    assert "not optimised" in saved["note"]


def test_temporal_holdout_outputs_state_the_limitation(m2_run):
    experiment, _ = m2_run
    root = experiment.root / "metrics" / "temporal_holdout"
    metrics = json.loads((root / "metrics.json").read_text())
    assert metrics["meaningful_on_this_dataset"] is False
    assert "M6/M7" in metrics["limitation"]
    assert (root / "LIMITATION.txt").exists()


def test_figures_are_generated(m2_run):
    experiment, _ = m2_run
    figures = {p.name for p in (experiment.root / "figures").glob("*.png")}
    expected = {"map_trajectory_categories.png", "spatial_cv_folds.png",
                "confusion_spatial_cv.png", "feature_importance.png",
                "ablation_comparison.png", "validation_comparison.png",
                "map_trend_sen.png", "map_restrend.png", "map_cyclicity.png",
                "map_disturbance.png", "map_recovery.png",
                "map_model_class.png", "map_model_confidence.png",
                "trajectory_class_distribution.png"}
    assert expected <= figures, f"missing figures: {expected - figures}"
    assert any(f.startswith("temporal_diagnostics_class") for f in figures)
    for figure in (experiment.root / "figures").glob("*.png"):
        assert figure.stat().st_size > 1000, f"{figure.name} looks empty"


def test_geospatial_metadata_is_preserved(m2_run):
    import rasterio
    from src import geo

    experiment, _ = m2_run
    source = geo.GeoRef.from_raster(Config().paths.ndvi_stack)
    layers = list((experiment.root / "predictions").glob("*.tif"))
    assert len(layers) >= 10
    for layer in layers:
        with rasterio.open(layer) as raster:
            assert raster.crs == source.crs, f"{layer.name} lost its CRS"
            assert (raster.height, raster.width) == source.shape
            assert raster.nodata is not None, f"{layer.name} declares no NoData"
            assert np.allclose(list(raster.transform)[:6],
                               list(source.transform)[:6])


def test_synthetic_status_is_declared_in_the_outputs(m2_run):
    experiment, _ = m2_run
    notice = (experiment.root / "metrics" / "m2_notice.txt").read_text()
    assert "Synthetic data are used for development and pipeline validation" \
        in notice
    assert "not real observations" in notice


def test_run_is_reproducible_from_its_own_configuration(m2_run, tmp_path):
    """Re-running the saved configuration reproduces the reported metrics."""
    experiment, results = m2_run
    import run_m2_experiments

    cfg = Config.load(experiment.root / "config" / "config.json")
    cfg.paths.results = str(tmp_path)
    cfg.experiment_name = "pytest_m2_repeat"
    cfg.research.sensitivity.enabled = False
    cfg.research.holdout.enabled = False
    _, repeated = run_m2_experiments.main(cfg)
    for key in ("accuracy", "f1_macro", "f1_weighted"):
        assert repeated["spatial_block_cv"][key] == \
            pytest.approx(results["spatial_block_cv"][key])
