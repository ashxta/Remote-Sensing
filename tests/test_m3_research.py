"""Cyclicity rigour, explainability and experiment-matrix tests
(M3 Parts 6, 7, 8 and 11)."""
import json

import numpy as np
import pandas as pd
import pytest

from src import timeseries as TS
from src.config import (Config, ExplainConfig, RFExperimentConfig,
                        SpatialCVConfig)
from src.experiment_matrix import (baseline_trend_prediction,
                                   binary_degradation_labels,
                                   run_experiment_matrix)
from src.explain import (IMPORTANCE_DISCLAIMER, explain_experiment,
                         permutation_importance,
                         spatial_permutation_importance)
from src.features import build_feature_table, feature_names
from src.validation import fit_random_forest, spatial_block_folds

T = 36
H, W = 12, 12


def archetypes(seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    columns, labels = [], []
    for i in range(H * W):
        kind = i % 4
        if kind == 0:
            s = 0.75 + rng.normal(0, 0.02, T)
        elif kind == 1:
            s = 0.5 + 0.001 * t + rng.normal(0, 0.03, T)
        elif kind == 2:
            s = 0.5 + 0.2 * np.sin(2 * np.pi * t / 6) + rng.normal(0, 0.02, T)
        else:
            s = 0.72 - 0.016 * t + rng.normal(0, 0.02, T)
        columns.append(np.clip(s, 0.05, 0.95))
        labels.append(kind + 1)
    ndvi = np.array(columns).T
    rain = rng.normal(1800, 200, (T, H * W))
    return ndvi, rain, np.array(labels)


@pytest.fixture(scope="module")
def built():
    ndvi, rain, labels = archetypes()
    table, extras = build_feature_table(ndvi, rain, Config())
    folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=3, n_folds=4,
                                                      seed=5))[1].reshape(-1)
    return ndvi, rain, labels, table, extras, folds


# ------------------------------------------------------- cyclicity rigour
def test_surrogate_test_is_deterministic():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (T, 20))
    a = TS.cyclicity_significance(x, n_surrogates=29, seed=4)
    b = TS.cyclicity_significance(x, n_surrogates=29, seed=4)
    assert np.array_equal(a["p_value"], b["p_value"])


def test_surrogate_test_detects_a_strong_cycle():
    rng = np.random.default_rng(1)
    t = np.arange(T)
    cyclic = np.sin(2 * np.pi * t / 6)[:, None] + rng.normal(0, .3, (T, 30))
    result = TS.cyclicity_significance(cyclic, n_surrogates=99, seed=2)
    assert result["significant"].mean() > 0.8
    assert np.nanmedian(result["p_value"]) < 0.05


def test_surrogate_test_keeps_white_noise_near_the_nominal_rate():
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 1, (T, 200))
    result = TS.cyclicity_significance(noise, n_surrogates=99, seed=3,
                                       alpha=0.05)
    assert result["significant"].mean() < 0.12, \
        "white noise must not be flagged far above the nominal rate"


def test_ar1_null_controls_false_positives_better_than_a_white_null():
    """The reason the red-noise null exists.

    Autocorrelated series concentrate power at low frequencies. Judged
    against white noise they look periodic; judged against their own
    persistence they do not.
    """
    rng = np.random.default_rng(3)
    red = np.zeros((T, 200))
    noise = rng.normal(0, 1, (T, 200))
    for i in range(1, T):
        red[i] = 0.7 * red[i - 1] + noise[i]
    ar1 = TS.cyclicity_significance(red, n_surrogates=99, seed=4, null="ar1")
    white = TS.cyclicity_significance(red, n_surrogates=99, seed=4,
                                      null="shuffle")
    assert ar1["significant"].mean() < white["significant"].mean()
    assert ar1["significant"].mean() < 0.12


def test_surrogate_test_recovers_the_autocorrelation_it_models():
    rng = np.random.default_rng(5)
    red = np.zeros((T, 50))
    noise = rng.normal(0, 1, (T, 50))
    for i in range(1, T):
        red[i] = 0.6 * red[i - 1] + noise[i]
    result = TS.cyclicity_significance(red, n_surrogates=19, seed=6)
    assert 0.3 < np.nanmean(result["ar1_coefficient"]) < 0.8


def test_surrogate_test_handles_missing_values():
    rng = np.random.default_rng(7)
    x = rng.normal(0, 1, (T, 10))
    x[rng.random(x.shape) < 0.1] = np.nan
    result = TS.cyclicity_significance(x, n_surrogates=19, seed=8)
    assert np.isfinite(result["p_value"]).all()


def test_surrogate_test_refuses_a_series_that_is_too_short():
    result = TS.cyclicity_significance(np.zeros((8, 4)), min_obs=12,
                                       n_surrogates=9)
    assert np.isnan(result["p_value"]).all()


def test_surrogate_test_rejects_an_invalid_null():
    with pytest.raises(ValueError, match="must be 'ar1' or 'shuffle'"):
        TS.cyclicity_significance(np.zeros((T, 2)), null="phase")


def test_period_band_is_configurable_in_the_surrogate_test():
    rng = np.random.default_rng(9)
    t = np.arange(T)
    cyclic = np.sin(2 * np.pi * t / 6)[:, None] + rng.normal(0, .3, (T, 30))
    inside = TS.cyclicity_significance(cyclic, min_period=5, max_period=8,
                                       n_surrogates=49, seed=1)
    outside = TS.cyclicity_significance(cyclic, min_period=14, max_period=20,
                                        n_surrogates=49, seed=1)
    assert inside["significant"].mean() > outside["significant"].mean()


def test_cyclicity_reports_how_many_values_it_imputed():
    rng = np.random.default_rng(10)
    x = rng.normal(0.5, 0.1, (T, 5))
    x[0, 0] = np.nan
    result = TS.cyclicity(x)
    assert result["n_missing"] == 1
    assert result["mean_imputed_for_spectrum"] is True


# ------------------------------------------------------------ explainability
def test_permutation_importance_ranks_an_informative_feature_first():
    rng = np.random.default_rng(0)
    n = 200
    signal = rng.normal(size=n)
    x = np.c_[signal, rng.normal(size=n)]
    y = np.where(signal > 0, 1, 2)
    cfg = RFExperimentConfig(n_estimators=30, seed=1)
    imputer, model = fit_random_forest(x, y, cfg)
    frame = permutation_importance(model, imputer, x, y, ["signal", "noise"],
                                   repeats=3, seed=2)
    assert frame.iloc[0]["feature"] == "signal"
    assert frame.iloc[0]["importance_mean"] > frame.iloc[1]["importance_mean"]


def test_permutation_importance_is_deterministic():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(120, 3))
    y = np.where(x[:, 0] > 0, 1, 2)
    cfg = RFExperimentConfig(n_estimators=20, seed=1)
    imputer, model = fit_random_forest(x, y, cfg)
    names = ["a", "b", "c"]
    a = permutation_importance(model, imputer, x, y, names, repeats=2, seed=7)
    b = permutation_importance(model, imputer, x, y, names, repeats=2, seed=7)
    assert np.allclose(a["importance_mean"], b["importance_mean"])


def test_spatial_permutation_importance_uses_held_out_folds(built):
    _, _, labels, table, _, folds = built
    columns = feature_names(["vegetation"])
    frame = spatial_permutation_importance(
        table, labels, folds, feature_names=columns,
        rf_cfg=RFExperimentConfig(n_estimators=20, seed=1),
        cfg=ExplainConfig(permutation_repeats=2))
    assert set(frame["feature"]) == set(columns)
    assert (frame["n_folds"] > 1).all()
    assert frame["importance_mean"].iloc[0] >= frame["importance_mean"].iloc[-1]


def test_explain_experiment_writes_outputs_and_disclaimer(built, tmp_path):
    _, _, labels, table, _, folds = built
    columns = feature_names(["vegetation", "trend"])
    result = explain_experiment(
        table, labels, folds, tmp_path, feature_names=columns,
        impurity=pd.Series(1.0, index=columns),
        rf_cfg=RFExperimentConfig(n_estimators=20, seed=1),
        cfg=ExplainConfig(permutation_repeats=2, shap=False))
    assert (tmp_path / "permutation_importance.csv").exists()
    assert (tmp_path / "importance_comparison.csv").exists()
    report = json.loads((tmp_path / "explainability.json").read_text())
    assert report["disclaimer"] == IMPORTANCE_DISCLAIMER
    assert "not causal" in IMPORTANCE_DISCLAIMER
    assert report["shap"]["available"] is False
    assert report["shap"]["reason"]
    comparison = pd.read_csv(tmp_path / "importance_comparison.csv",
                             index_col=0)
    assert "permutation_macro_f1_drop" in comparison.columns
    assert set(result["permutation"]["feature"]) == set(columns)


def test_shap_is_optional_and_never_breaks_a_run(built, tmp_path):
    """A missing or failing optional dependency must degrade, not crash."""
    _, _, labels, table, _, folds = built
    result = explain_experiment(
        table, labels, folds, tmp_path,
        feature_names=feature_names(["vegetation"]),
        rf_cfg=RFExperimentConfig(n_estimators=15, seed=1),
        cfg=ExplainConfig(permutation_repeats=1, shap=True))
    assert result["report"]["shap"]["available"] in (True, False)


# --------------------------------------------------------- experiment matrix
def test_binary_degradation_labels_follow_the_configuration():
    labels = np.array([1, 2, 4, 5])
    assert binary_degradation_labels(labels, [4]).tolist() == [0, 0, 1, 0]
    assert binary_degradation_labels(labels, [4, 5]).tolist() == [0, 0, 1, 1]


def test_trend_baseline_is_a_fixed_rule_with_no_training(built):
    _, _, truth, table, _, _ = built
    rule = baseline_trend_prediction(table, Config())
    assert set(np.unique(rule["predictions"])) <= {0, 1}
    assert rule["probabilities"].shape == (len(table), 2)
    assert np.allclose(rule["probabilities"].sum(axis=1), 1.0)
    assert "not a calibrated probability" in rule["note"]
    declining = rule["predictions"][truth == 4]
    assert declining.mean() > 0.5, "the rule must find the planted decline"


def test_trend_baseline_respects_the_configured_alpha(built):
    _, _, _, table, _, _ = built
    strict, lenient = Config(), Config()
    strict.trend.alpha, lenient.trend.alpha = 1e-6, 0.5
    assert baseline_trend_prediction(table, strict)["predictions"].sum() <= \
        baseline_trend_prediction(table, lenient)["predictions"].sum()


def test_experiment_matrix_compares_methods_on_one_protocol(built, tmp_path):
    _, _, labels, table, _, folds = built
    cfg = Config()
    cfg.research.model = RFExperimentConfig(n_estimators=15, seed=1)
    cfg.research.matrix.run_cnn = False
    matrix = run_experiment_matrix(table, labels, folds, tmp_path, cfg)
    assert set(matrix["method"]) == {"baseline_trend", "baseline_restrend",
                                     "baseline_integrated", "rf_basic",
                                     "rf_proposed"}
    binary = matrix[matrix["task"] == "binary_degradation"]
    assert len(binary) == 5, "every method must attempt the shared task"
    assert (tmp_path / "experiment_matrix.csv").exists()
    saved = json.loads((tmp_path / "experiment_matrix.json").read_text())
    assert "question" in saved and "conclusion" in saved
    assert "synthetic" in saved["caveat"]


def test_matrix_conclusion_is_qualified_by_the_fold_spread(built, tmp_path):
    _, _, labels, table, _, folds = built
    cfg = Config()
    cfg.research.model = RFExperimentConfig(n_estimators=15, seed=1)
    cfg.research.matrix.run_cnn = False
    run_experiment_matrix(table, labels, folds, tmp_path, cfg)
    conclusion = json.loads(
        (tmp_path / "experiment_matrix.json").read_text())["conclusion"]
    assert conclusion["available"]
    assert "exceeds_fold_spread" in conclusion
    assert "standard deviation" in conclusion["statement"]


def test_matrix_conclusion_reports_the_multiclass_comparison_too(built,
                                                                 tmp_path):
    """The binary task alone understates what the extra features are for."""
    _, _, labels, table, _, folds = built
    cfg = Config()
    cfg.research.model = RFExperimentConfig(n_estimators=15, seed=1)
    cfg.research.matrix.run_cnn = False
    run_experiment_matrix(table, labels, folds, tmp_path, cfg)
    conclusion = json.loads(
        (tmp_path / "experiment_matrix.json").read_text())["conclusion"]
    assert conclusion["multiclass"]["baseline_participates"] is False
    assert "cannot attempt" in conclusion["multiclass"]["note"]
    assert "proposed_minus_basic" in conclusion["multiclass"]


def test_matrix_scores_every_method_on_one_common_test_set(built, tmp_path):
    """Methods must never be compared on different samples."""
    _, _, labels, table, _, folds = built
    cfg = Config()
    cfg.research.model = RFExperimentConfig(n_estimators=12, seed=1)
    cfg.research.matrix.run_cnn = False
    run_experiment_matrix(table, labels, folds, tmp_path, cfg)
    saved = json.loads((tmp_path / "experiment_matrix.json").read_text())
    for task, coverage in saved["coverage"].items():
        assert coverage["n_common_evaluated"] > 0
        for method, count in coverage["per_method_evaluated"].items():
            assert count >= coverage["n_common_evaluated"], method
        assert "identical_coverage" in coverage
    assert "re-scored on the samples EVERY method evaluated" in \
        saved["protocol"]


def test_matrix_reports_reduced_coverage_when_a_method_skips_folds(built,
                                                                   tmp_path):
    """A method evaluating fewer folds must shrink the common set, not be
    silently compared against a larger one."""
    from src.experiment_matrix import _common_evaluation

    n = 40
    target = np.array([0, 1] * (n // 2))
    folds = np.repeat([0, 1, 2, 3], n // 4)
    full = {"predictions": target.copy(), "evaluated": np.ones(n, bool)}
    partial_mask = folds != 3
    partial = {"predictions": target.copy(), "evaluated": partial_mask}
    rescored, common = _common_evaluation(
        {"full": full, "partial": partial}, target, np.array([0, 1]), folds)
    assert common.sum() == int(partial_mask.sum())
    for name, metrics in rescored.items():
        assert metrics["n_evaluated"] == int(common.sum())
    assert rescored["full"]["n_evaluated_by_this_method"] == n
    assert rescored["partial"]["n_evaluated_by_this_method"] == \
        int(partial_mask.sum())


def test_matrix_records_a_skipped_method_with_its_reason(built, tmp_path):
    _, _, labels, table, _, folds = built
    cfg = Config()
    cfg.research.model = RFExperimentConfig(n_estimators=10, seed=1)
    cfg.research.matrix.run_cnn = True
    run_experiment_matrix(table, labels, folds, tmp_path, cfg, series=None)
    saved = json.loads((tmp_path / "experiment_matrix.json").read_text())
    assert saved["skipped"], "a skipped method must be recorded, not dropped"
    assert saved["skipped"][0]["method"] == "cnn_1d"


def test_matrix_baseline_is_not_scored_on_the_multiclass_task(built, tmp_path):
    """The conventional rule cannot attempt five classes; it must not be
    silently given a multiclass score."""
    _, _, labels, table, _, folds = built
    cfg = Config()
    cfg.research.model = RFExperimentConfig(n_estimators=10, seed=1)
    cfg.research.matrix.run_cnn = False
    matrix = run_experiment_matrix(table, labels, folds, tmp_path, cfg)
    baseline = matrix[matrix["method"] == "baseline_trend"]
    assert set(baseline["task"]) == {"binary_degradation"}
