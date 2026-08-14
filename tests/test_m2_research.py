"""Trajectory classes, ablation, sensitivity, temporal holdout, data
contract and output tests (M2 Parts 3, 6, 7, 9, 10 and 12)."""
import json

import numpy as np
import pandas as pd
import pytest

from src.ablation import (experiment_id, feature_sets_for, resolve_feature_set,
                          run_ablation_study)
from src.config import (AblationExperiment, Config, ParameterSweep,
                        RFExperimentConfig, SpatialCVConfig, TrajectoryConfig)
from src.dataset import DatasetValidationError, validate_dataset
from src.features import build_feature_table, feature_names
from src.holdout import (resolve_cutoff, run_temporal_holdout, split_series,
                         temporal_holdout_indices)
from src.sensitivity import (apply_override, read_parameter,
                             run_sensitivity_analysis, scenario_table)
from src.trajectory import (TRAJECTORY_CLASSES, UNCERTAIN,
                            classify_trajectories, effective_trajectory_config,
                            trajectory_codes, trajectory_rules,
                            trajectory_summary)
from src.validation import spatial_block_folds

T = 36
H, W = 12, 12


def archetypes(seed=0):
    """A grid of stable, degrading, recovering and cyclic pixels."""
    rng = np.random.default_rng(seed)
    n = H * W
    t = np.arange(T)
    columns, labels = [], []
    for i in range(n):
        kind = i % 4
        if kind == 0:
            s = 0.75 + rng.normal(0, 0.02, T)
        elif kind == 1:
            s = 0.7 - 0.014 * t + rng.normal(0, 0.02, T)
        elif kind == 2:
            k = 10
            s = np.concatenate([np.full(k, 0.72),
                                np.linspace(0.28, 0.71, T - k)]) \
                + rng.normal(0, 0.02, T)
        else:
            s = 0.5 + 0.2 * np.sin(2 * np.pi * t / 6.0) \
                + rng.normal(0, 0.02, T)
        columns.append(np.clip(s, 0.05, 0.95))
        labels.append(kind + 1)
    ndvi = np.array(columns).T
    rain = rng.normal(1800, 200, size=(T, n))
    return ndvi, rain, np.array(labels)


@pytest.fixture(scope="module")
def built():
    ndvi, rain, labels = archetypes()
    table, extras = build_feature_table(ndvi, rain, Config())
    folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=3, n_folds=4,
                                                      seed=5))[1].reshape(-1)
    return ndvi, rain, labels, table, extras, folds


# ------------------------------------------------------------- trajectories
def test_trajectory_labels_come_from_the_declared_set(built):
    _, _, _, table, extras, _ = built
    labels = classify_trajectories(table, extras, Config())
    assert set(np.unique(labels)).issubset(set(TRAJECTORY_CLASSES))
    assert len(labels) == len(table)


def test_trajectory_classes_identify_planted_archetypes(built):
    _, _, truth, table, extras, _ = built
    labels = classify_trajectories(table, extras, Config())
    degrading = labels[truth == 2]
    cyclic = labels[truth == 4]
    assert (degrading == "Degrading").mean() > 0.8
    assert (cyclic == "Cyclic").mean() > 0.5


def test_trajectory_thresholds_are_configurable(built):
    _, _, _, table, extras, _ = built
    strict = TrajectoryConfig(cyclicity_enrichment_threshold=50.0)
    lenient = TrajectoryConfig(cyclicity_enrichment_threshold=1.05,
                               require_periodic_flag=False)
    assert (classify_trajectories(table, extras, strict) == "Cyclic").sum() < \
           (classify_trajectories(table, extras, lenient) == "Cyclic").sum()


def test_trajectory_priority_is_configurable(built):
    _, _, _, table, extras, _ = built
    cyclic_first = TrajectoryConfig(priority=["Cyclic", "Recovering",
                                              "Degrading", "Stable"])
    degrading_first = TrajectoryConfig(priority=["Degrading", "Cyclic",
                                                 "Recovering", "Stable"])
    a = classify_trajectories(table, extras, cyclic_first)
    b = classify_trajectories(table, extras, degrading_first)
    assert not np.array_equal(a, b)


def test_trajectory_thresholds_are_inherited_from_the_m1_sections(built):
    """One knob per parameter: the M1 sections win over the mirrored fields."""
    _, _, _, table, extras, _ = built
    cfg = Config()
    cfg.trend.alpha = 0.5
    cfg.cyclicity.periodicity_threshold = 100.0
    resolved = effective_trajectory_config(cfg)
    assert resolved.alpha == 0.5
    assert resolved.cyclicity_enrichment_threshold == 100.0
    assert cfg.research.trajectory.alpha == 0.05, "base config not mutated"
    assert (classify_trajectories(table, extras, cfg) == "Cyclic").sum() == 0


def test_recovering_requires_a_significant_breakpoint(built):
    """Ordinary noise in a stable series must not read as a recovery."""
    _, _, _, table, extras, _ = built
    strict = TrajectoryConfig(require_significant_breakpoint=True)
    loose = TrajectoryConfig(require_significant_breakpoint=False)
    n_strict = (classify_trajectories(table, extras, strict) ==
                "Recovering").sum()
    n_loose = (classify_trajectories(table, extras, loose) ==
               "Recovering").sum()
    assert n_strict <= n_loose
    significant = np.asarray(extras["break_significant"], bool)
    labels = classify_trajectories(table, extras, strict)
    assert not (labels[~significant] == "Recovering").any()


def test_unknown_priority_class_is_rejected(built):
    _, _, _, table, extras, _ = built
    with pytest.raises(ValueError, match="unknown trajectory class"):
        classify_trajectories(table, extras,
                              TrajectoryConfig(priority=["Jhum"]))


def test_pixels_without_finite_statistics_are_uncertain():
    ndvi, rain, _ = archetypes(seed=2)
    ndvi = ndvi.copy()
    ndvi[3:, 0] = np.nan                     # below the minimum observations
    table, extras = build_feature_table(ndvi, rain, Config())
    labels = classify_trajectories(table, extras, Config())
    assert labels[0] == UNCERTAIN


def test_trajectory_rules_document_their_limits():
    rules = trajectory_rules(Config())
    assert set(rules["classes"]) == set(TRAJECTORY_CLASSES)
    text = rules["interpretation_limit"].lower()
    assert "not verified land-cover" in text or "not verified land cover" in text
    assert "jhum" in text


def test_trajectory_summary_and_codes_are_consistent(built):
    _, _, _, table, extras, _ = built
    labels = classify_trajectories(table, extras, Config())
    summary = trajectory_summary(labels)
    assert sum(summary["counts"].values()) == len(labels)
    assert pytest.approx(sum(summary["fractions"].values()), abs=1e-9) == 1.0
    codes = trajectory_codes(labels)
    assert codes.shape == labels.shape
    assert codes.min() >= 1


# ----------------------------------------------------------------- ablation
def test_ablation_feature_sets_are_strictly_nested():
    experiments = Config().research.ablation.experiments
    sets = [resolve_feature_set(e) for e in experiments]
    for previous, current in zip(sets, sets[1:]):
        assert set(previous) < set(current), \
            "each ablation cell must strictly contain the previous one"


def test_ablation_cells_have_no_duplicate_features():
    for name, features in feature_sets_for(
            Config().research.ablation.experiments).items():
        assert len(features) == len(set(features)), name


def test_ablation_cells_withhold_the_features_they_claim_to():
    sets = feature_sets_for(Config().research.ablation.experiments)
    assert "mk_z" not in sets["A_basic_vegetation"]
    assert "mk_z" in sets["B_basic_trend"]
    assert "restrend" not in sets["B_basic_trend"]
    assert "restrend" in sets["C_add_restrend"]
    assert "cyc_score" not in sets["C_add_restrend"]
    assert "cyc_score" in sets["D_add_cyclicity"]
    assert "recovery_fraction" not in sets["D_add_cyclicity"]
    assert "recovery_fraction" in sets["E_add_disturbance"]
    assert "rain_mean" not in sets["E_add_disturbance"]
    assert "rain_mean" in sets["F_full"]


def test_full_ablation_cell_uses_every_modelling_feature():
    sets = feature_sets_for(Config().research.ablation.experiments)
    assert set(sets["F_full"]) == set(feature_names())


def test_experiment_id_is_stable_and_sensitive():
    cfg = RFExperimentConfig(n_estimators=10)
    first = experiment_id("A", ["mean"], cfg)
    assert first == experiment_id("A", ["mean"], cfg)
    assert first != experiment_id("A", ["mean", "std"], cfg)
    assert first != experiment_id("A", ["mean"],
                                  RFExperimentConfig(n_estimators=11))


def test_ablation_study_writes_every_required_artifact(built, tmp_path):
    _, _, labels, table, _, folds = built
    cfg = Config()
    cfg.research.model = RFExperimentConfig(n_estimators=15, seed=1)
    cfg.research.ablation.experiments = [
        AblationExperiment("A", "A_basic_vegetation", ["vegetation"]),
        AblationExperiment("B", "B_basic_trend", ["vegetation", "trend"]),
    ]
    comparison = run_ablation_study(table, labels, folds, tmp_path, cfg)
    assert list(comparison["feature_set"]) == ["A_basic_vegetation",
                                               "B_basic_trend"]
    assert (tmp_path / "ablation_comparison.csv").exists()
    assert (tmp_path / "ablation_comparison.json").exists()
    for eid in comparison["experiment_id"]:
        cell = tmp_path / eid
        for artifact in ("configuration.json", "metrics.json",
                         "confusion_matrix.csv", "predictions.csv",
                         "probabilities.csv", "feature_importance.csv",
                         "log.txt"):
            assert (cell / artifact).exists(), f"{eid} missing {artifact}"
        configuration = json.loads((cell / "configuration.json").read_text())
        assert configuration["n_features"] == len(configuration["features"])
        metrics = json.loads((cell / "metrics.json").read_text())
        for key in ("accuracy", "f1_macro", "f1_weighted", "per_class",
                    "confusion_matrix", "fold_summary"):
            assert key in metrics


def test_ablation_importance_covers_only_that_cells_features(built, tmp_path):
    _, _, labels, table, _, folds = built
    cfg = Config()
    cfg.research.model = RFExperimentConfig(n_estimators=15, seed=1)
    cfg.research.ablation.experiments = [
        AblationExperiment("A", "A_basic_vegetation", ["vegetation"])]
    comparison = run_ablation_study(table, labels, folds, tmp_path, cfg)
    cell = tmp_path / comparison["experiment_id"].iloc[0]
    importance = pd.read_csv(cell / "feature_importance.csv", index_col=0)
    assert set(importance.index) == set(feature_names(["vegetation"]))


# -------------------------------------------------------------- sensitivity
def test_parameter_override_does_not_mutate_the_base_configuration():
    cfg = Config()
    modified = apply_override(cfg, "trend.alpha", 0.2)
    assert modified.trend.alpha == 0.2
    assert cfg.trend.alpha == 0.05
    assert read_parameter(modified, "trend.alpha") == 0.2


def test_override_preserves_the_declared_type():
    cfg = Config()
    assert isinstance(apply_override(cfg, "trend.min_obs", 12.0).trend.min_obs,
                      int)
    assert isinstance(
        apply_override(cfg, "cyclicity.min_period", 3).cyclicity.min_period,
        float)


def test_unknown_parameter_is_rejected():
    with pytest.raises(ValueError, match="unknown configuration"):
        apply_override(Config(), "trend.not_a_parameter", 1)


def test_scenario_table_marks_the_default_value():
    cfg = Config()
    sweeps = [ParameterSweep("trend.alpha", [0.01, 0.05, 0.1])]
    table = scenario_table(sweeps, cfg)
    assert table["is_baseline"].sum() == 1
    assert float(table.loc[table["is_baseline"], "value"].iloc[0]) == \
        cfg.trend.alpha


def test_sensitivity_analysis_runs_every_scenario(built, tmp_path):
    ndvi, rain, labels, _, _, folds = built
    cfg = Config()
    cfg.research.sensitivity.parameters = [
        ParameterSweep("trend.alpha", [0.01, 0.05]),
        ParameterSweep("cyclicity.periodicity_threshold", [1.5, 3.0]),
    ]
    table = run_sensitivity_analysis(ndvi, rain, tmp_path, cfg, labels=labels,
                                     fold_grid=folds, evaluate_model=False)
    assert len(table) == 5, "four scenarios plus the baseline"
    assert (tmp_path / "sensitivity_results.csv").exists()
    assert (tmp_path / "sensitivity_scenarios.csv").exists()
    saved = json.loads((tmp_path / "sensitivity_results.json").read_text())
    assert "spread_by_parameter" in saved and "baseline" in saved
    assert "not optimised" in saved["note"]


def test_sweeping_an_m1_threshold_reaches_the_trajectory_rules(built,
                                                               tmp_path):
    """A sensitivity sweep must move every stage the parameter feeds."""
    ndvi, rain, labels, _, _, folds = built
    cfg = Config()
    cfg.research.sensitivity.parameters = [
        ParameterSweep("trend.alpha", [1e-6, 0.5])]
    table = run_sensitivity_analysis(ndvi, rain, tmp_path, cfg, labels=labels,
                                     fold_grid=folds, evaluate_model=False)
    swept = table[table["parameter"] == "trend.alpha"]
    assert swept["trajectory_degrading_fraction"].nunique() > 1
    assert swept["significant_trend_fraction"].nunique() > 1


def test_sensitivity_detects_a_parameter_that_matters(built, tmp_path):
    ndvi, rain, labels, _, _, folds = built
    cfg = Config()
    cfg.research.sensitivity.parameters = [
        ParameterSweep("cyclicity.periodicity_threshold", [1.05, 50.0])]
    table = run_sensitivity_analysis(ndvi, rain, tmp_path, cfg, labels=labels,
                                     fold_grid=folds, evaluate_model=False)
    swept = table[table["parameter"] != "(baseline)"]
    assert swept["periodic_fraction"].nunique() > 1


# ---------------------------------------------------------------- holdout
def test_holdout_indices_are_chronological():
    historical, later = temporal_holdout_indices(12, 8)
    assert historical.max() < later.min()
    assert len(historical) == 8 and len(later) == 4


def test_cutoff_resolution_respects_the_configuration():
    cfg = Config().research.holdout
    cfg.cutoff_fraction = 0.5
    assert resolve_cutoff(36, cfg) == 18
    cfg.cutoff_index = 20
    assert resolve_cutoff(36, cfg) == 20


def test_cutoff_that_starves_a_window_is_rejected():
    cfg = Config().research.holdout
    cfg.cutoff_index = 34
    with pytest.raises(ValueError, match="min_history|min_future"):
        resolve_cutoff(36, cfg)
    with pytest.raises(ValueError, match="at least"):
        resolve_cutoff(8, cfg)


def test_split_windows_contain_no_future_information():
    ndvi, rain, _ = archetypes(seed=4)
    (hist_ndvi, hist_rain), (late_ndvi, late_rain) = split_series(ndvi, rain,
                                                                  20)
    assert hist_ndvi.shape[0] == 20 and late_ndvi.shape[0] == T - 20
    assert np.array_equal(hist_ndvi, ndvi[:20])
    assert np.array_equal(late_rain, rain[20:])
    # the windows are copies: touching the future cannot reach the past
    late_ndvi[:] = 999.0
    assert np.array_equal(hist_ndvi, ndvi[:20])


def test_historical_features_ignore_future_observations():
    """The strongest no-lookahead test: corrupt the future, expect no change."""
    ndvi, rain, _ = archetypes(seed=6)
    cutoff = 22
    (hist_ndvi, hist_rain), _ = split_series(ndvi, rain, cutoff)
    corrupted = ndvi.copy()
    corrupted[cutoff:] = -0.5
    (corrupt_ndvi, corrupt_rain), _ = split_series(corrupted, rain, cutoff)
    a, _ = build_feature_table(hist_ndvi, hist_rain, Config())
    b, _ = build_feature_table(corrupt_ndvi, corrupt_rain, Config())
    assert np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_temporal_holdout_experiment_is_documented_as_limited(built, tmp_path):
    ndvi, rain, labels, _, _, folds = built
    cfg = Config()
    cfg.research.model = RFExperimentConfig(n_estimators=15, seed=2)
    cfg.research.holdout.cutoff_fraction = 0.6
    result = run_temporal_holdout(ndvi, rain, labels, tmp_path, cfg,
                                  fold_grid=folds)
    metrics = result["metrics"]
    assert metrics["validation"] == "temporal_holdout"
    assert metrics["meaningful_on_this_dataset"] is False
    assert metrics["n_historical_steps"] + metrics["n_later_steps"] == T
    assert (tmp_path / "LIMITATION.txt").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "predictions.csv").exists()
    assert result["train_mask"].sum() > 0
    assert not (result["train_mask"] & result["test_mask"]).any()


def test_temporal_holdout_trains_on_historical_features_only(built, tmp_path):
    ndvi, rain, labels, _, _, folds = built
    cfg = Config()
    cfg.research.model = RFExperimentConfig(n_estimators=10, seed=2)
    result = run_temporal_holdout(ndvi, rain, labels, tmp_path, cfg,
                                  fold_grid=folds)
    cutoff = result["cutoff"]
    expected, _ = build_feature_table(ndvi[:cutoff], rain[:cutoff], cfg)
    assert np.array_equal(result["historical_features"].to_numpy(),
                          expected.to_numpy(), equal_nan=True)


# ---------------------------------------------------------- data contract
def test_valid_dataset_is_described_not_modified():
    ndvi, rain, _ = archetypes(seed=8)
    metadata = validate_dataset(ndvi, rain, cfg=Config())
    assert metadata["n_time_steps"] == T
    assert metadata["n_pixels"] == H * W
    assert metadata["ndvi_min"] >= -1 and metadata["ndvi_max"] <= 1


def test_infinities_are_rejected():
    ndvi, rain, _ = archetypes(seed=9)
    ndvi = ndvi.copy()
    ndvi[0, 0] = np.inf
    with pytest.raises(DatasetValidationError, match="infinite"):
        validate_dataset(ndvi, rain)


def test_dimension_mismatch_is_rejected():
    ndvi, rain, _ = archetypes(seed=10)
    with pytest.raises(DatasetValidationError, match="identical"):
        validate_dataset(ndvi, rain[:, :5])


def test_short_time_dimension_is_rejected():
    with pytest.raises(DatasetValidationError, match="at least 3 time steps"):
        validate_dataset(np.zeros((2, 4)), np.zeros((2, 4)))


def test_unexpected_time_dimension_is_rejected():
    ndvi, rain, _ = archetypes(seed=11)
    with pytest.raises(DatasetValidationError, match="expected 40 time steps"):
        validate_dataset(ndvi, rain, expected_time_steps=40)


def test_out_of_range_ndvi_is_rejected_but_reportable():
    ndvi, rain, _ = archetypes(seed=12)
    ndvi = ndvi.copy()
    ndvi[0, 0] = 4.2
    with pytest.raises(DatasetValidationError, match="physical range"):
        validate_dataset(ndvi, rain)
    metadata = validate_dataset(ndvi, rain, strict=False)
    assert metadata["ndvi_out_of_physical_range"] == 1
    assert metadata["warnings"]


def test_negative_rainfall_is_rejected():
    ndvi, rain, _ = archetypes(seed=13)
    rain = rain.copy()
    rain[0, 0] = -12.0
    with pytest.raises(DatasetValidationError, match="below 0"):
        validate_dataset(ndvi, rain)


def test_all_missing_ndvi_is_rejected():
    ndvi, rain, _ = archetypes(seed=14)
    with pytest.raises(DatasetValidationError, match="no valid observations"):
        validate_dataset(np.full_like(ndvi, np.nan), rain)


def test_too_few_observations_everywhere_is_rejected():
    ndvi, rain, _ = archetypes(seed=15)
    ndvi = ndvi.copy()
    ndvi[3:] = np.nan
    with pytest.raises(DatasetValidationError, match="min_valid_obs"):
        validate_dataset(ndvi, rain)


def test_missing_georeference_is_rejected_when_required():
    ndvi, rain, _ = archetypes(seed=16)
    with pytest.raises(DatasetValidationError, match="CRS and a transform"):
        validate_dataset(ndvi, rain, require_georef=True)


def test_degenerate_transform_is_rejected():
    from rasterio.transform import Affine
    ndvi, rain, _ = archetypes(seed=17)
    grid = ndvi.reshape(T, H, W)
    rain_grid = rain.reshape(T, H, W)
    with pytest.raises(DatasetValidationError, match="zero pixel size"):
        validate_dataset(grid, rain_grid,
                         transform=Affine(0.0, 0.0, 92.0, 0.0, -0.01, 26.0))


def test_impossible_geographic_coordinates_are_rejected():
    import rasterio
    from rasterio.transform import from_origin
    ndvi, rain, _ = archetypes(seed=18)
    with pytest.raises(DatasetValidationError, match="longitude|latitude"):
        validate_dataset(ndvi.reshape(T, H, W), rain.reshape(T, H, W),
                         crs=rasterio.crs.CRS.from_epsg(4326),
                         transform=from_origin(400.0, 26.0, 0.01, 0.01))


def test_valid_georeference_is_accepted_for_any_study_area():
    """No region is hard-coded: an arbitrary valid grid must pass."""
    import rasterio
    from rasterio.transform import from_origin
    ndvi, rain, _ = archetypes(seed=19)
    metadata = validate_dataset(
        ndvi.reshape(T, H, W), rain.reshape(T, H, W),
        crs=rasterio.crs.CRS.from_epsg(4326),
        transform=from_origin(-58.0, -12.0, 0.01, 0.01), require_georef=True)
    assert metadata["georeference"]["pixel_width"] == pytest.approx(0.01)
