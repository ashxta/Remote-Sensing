"""Tests for the research question itself (M5).

    Can multi-temporal analysis distinguish persistent land degradation from
    cyclic or environmentally driven vegetation dynamics?

These tests check that the framework (a) makes that distinction in its own
class definitions, (b) measures it per confounder rather than on average,
and (c) does not let a model receive information its competitor is denied.
"""
import json

import numpy as np
import pytest

from src.config import Config, SpatialCVConfig
from src.discrimination import (DISCRIMINATION_LIMIT, confounder_confusion,
                                discrimination_table,
                                run_discrimination_analysis)
from src.experiment_matrix import (baseline_integrated_prediction,
                                   baseline_restrend_prediction,
                                   baseline_trend_prediction)
from src.features import build_feature_table
from src.trajectory import (DEGRADING, DISTURBED, RAINFALL_DECLINE,
                            TRAJECTORY_CLASSES, TRAJECTORY_CODES,
                            classify_trajectories, trajectory_rules)
from src.validation import spatial_block_folds

T = 36
H, W = 12, 12


def confounded_series(seed=0):
    """Four behaviours a degradation detector must tell apart.

    class 1  stable
    class 2  RAINFALL-DRIVEN decline: NDVI tracks a declining rainfall run,
             so the raw trend is negative but climate explains it
    class 3  cyclic
    class 4  PERSISTENT decline that is not explained by rainfall
    """
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    # A rainfall record with a genuine downward run.
    rain_base = 1800 - 14.0 * t + rng.normal(0, 60, T)
    columns, rains, labels = [], [], []
    for i in range(H * W):
        kind = i % 4
        rain = rain_base + rng.normal(0, 40, T)
        if kind == 0:
            ndvi = 0.72 + 0.00005 * (rain - rain.mean()) \
                + rng.normal(0, 0.015, T)
        elif kind == 1:
            # Strong rainfall coupling; the decline comes from the climate.
            ndvi = 0.30 + 0.00022 * rain + rng.normal(0, 0.015, T)
        elif kind == 2:
            ndvi = 0.55 + 0.20 * np.sin(2 * np.pi * t / 6) \
                + rng.normal(0, 0.02, T)
        else:
            # Declines regardless of rainfall.
            ndvi = 0.75 - 0.014 * t + 0.00002 * (rain - rain.mean()) \
                + rng.normal(0, 0.015, T)
        columns.append(np.clip(ndvi, 0.05, 0.95))
        rains.append(rain)
        labels.append(kind + 1)
    return (np.array(columns).T, np.array(rains).T, np.array(labels))


@pytest.fixture(scope="module")
def built():
    ndvi, rain, labels = confounded_series()
    table, extras = build_feature_table(ndvi, rain, Config())
    folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=3, n_folds=4,
                                                      seed=5))[1].reshape(-1)
    return ndvi, rain, labels, table, extras, folds


# ------------------------------------------------- the degradation definition
def test_class_set_separates_the_six_behaviours():
    """Part 7: the classes must not pool distinct behaviours."""
    for required in ("Stable", DEGRADING, RAINFALL_DECLINE, DISTURBED,
                     "Recovering", "Cyclic"):
        assert required in TRAJECTORY_CLASSES
    assert len(set(TRAJECTORY_CODES.values())) == len(TRAJECTORY_CLASSES)


def test_rainfall_driven_decline_is_not_called_degradation(built):
    """The central scientific fix: climate-explained decline is separated."""
    _, _, truth, table, extras, _ = built
    labels = classify_trajectories(table, extras, Config())
    rainfall_driven = labels[truth == 2]
    # Whatever else they are called, they must not be called Degrading.
    assert (rainfall_driven == DEGRADING).mean() < 0.10, \
        "rainfall-driven decline is being reported as degradation"
    assert (rainfall_driven == RAINFALL_DECLINE).mean() > 0.5


def test_persistent_decline_is_still_detected(built):
    """The fix must not be achieved by simply flagging less."""
    _, _, truth, table, extras, _ = built
    labels = classify_trajectories(table, extras, Config())
    assert (labels[truth == 4] == DEGRADING).mean() > 0.8


def test_climate_adjustment_can_be_disabled_for_comparison(built):
    """The pre-adjustment behaviour must remain reachable for ablation."""
    _, _, truth, table, extras, _ = built
    cfg = Config()
    cfg.research.trajectory.require_climate_adjustment = False
    pooled = classify_trajectories(table, extras, cfg)
    adjusted = classify_trajectories(table, extras, Config())
    assert (pooled == DEGRADING).sum() > (adjusted == DEGRADING).sum()
    assert (pooled == RAINFALL_DECLINE).sum() == 0


def test_uninterpretable_adjustment_keeps_the_uncorrected_label():
    """Where RESTREND means nothing, the decline must not be discarded."""
    rng = np.random.default_rng(3)
    t = np.arange(T)
    ndvi = np.clip((0.75 - 0.015 * t)[:, None]
                   + rng.normal(0, 0.015, (T, 40)), 0.05, 0.95)
    rain = rng.normal(1800, 200, (T, ndvi.shape[1]))   # unrelated to NDVI
    table, extras = build_feature_table(ndvi, rain, Config())
    # A few pixels will pass the rainfall-relation test by chance at
    # alpha=0.05; the claim is about the ones where it genuinely fails.
    uninterpretable = ~np.asarray(extras["restrend_valid"], dtype=bool)
    assert uninterpretable.mean() > 0.8, "fixture is not behaving as intended"
    labels = classify_trajectories(table, extras, Config())
    assert (labels[uninterpretable] == DEGRADING).all(), \
        "an uncorrectable decline must keep its uncorrected label"


def test_disturbance_without_recovery_is_not_called_stable():
    """An abrupt drop that never returns must not be reported as Stable."""
    rng = np.random.default_rng(4)
    step = np.concatenate([np.full(18, 0.75), np.full(T - 18, 0.45)])
    ndvi = np.clip(step[:, None] + rng.normal(0, 0.012, (T, 12)), 0.05, 0.95)
    rain = rng.normal(1800, 150, (T, 12))
    table, extras = build_feature_table(ndvi, rain, Config())
    labels = classify_trajectories(table, extras, Config())
    assert (labels == "Stable").mean() < 0.2
    assert set(np.unique(labels)) <= set(TRAJECTORY_CLASSES)


def test_rules_document_the_climate_adjustment():
    rules = trajectory_rules(Config())
    assert rules["climate_adjustment_applied"] is True
    assert "RESTREND" in rules["rules"][DEGRADING]
    assert "rainfall" in rules["rules"][RAINFALL_DECLINE].lower()
    assert "not proof of anthropogenic" in rules["interpretation_limit"]


# --------------------------------------------------- discrimination measures
def test_confounder_confusion_separates_target_from_confounders():
    reference = np.array([1, 1, 2, 2, 4, 4])
    flagged = np.array([0, 1, 0, 0, 1, 1])
    table = confounder_confusion(reference, flagged, degradation_classes=[4],
                                 class_names={1: "Stable", 2: "Cyclic",
                                              4: "Declining"})
    target = table[table["role"] == "degradation target"].iloc[0]
    assert target["flagged_rate"] == pytest.approx(1.0)
    stable = table[table["class_name"] == "Stable"].iloc[0]
    assert stable["role"] == "confounder"
    assert stable["flagged_rate"] == pytest.approx(0.5)


def test_discrimination_margin_is_pessimistic():
    """A method must be judged by the confounder it handles WORST."""
    reference = np.array([1] * 10 + [2] * 10 + [4] * 10)
    # Perfect on class 1, terrible on class 2, perfect recall.
    flagged = np.array([0] * 10 + [1] * 8 + [0] * 2 + [1] * 10)
    _, summary = discrimination_table(
        reference, {"m": flagged}, degradation_classes=[4],
        class_names={1: "Stable", 2: "Cyclic", 4: "Declining"})
    row = summary.iloc[0]
    assert row["recall_on_degradation"] == pytest.approx(1.0)
    assert row["worst_confounder"] == "Cyclic"
    assert row["worst_confounder_false_positive_rate"] == pytest.approx(0.8)
    assert row["discrimination_margin"] == pytest.approx(0.2)
    # An averaged false-positive rate would have hidden it.
    assert row["false_positive_rate_overall"] == pytest.approx(0.4)


def test_a_method_that_flags_everything_scores_badly():
    reference = np.array([1] * 10 + [4] * 10)
    _, summary = discrimination_table(
        reference, {"flag_all": np.ones(20, int)}, degradation_classes=[4],
        class_names={1: "Stable", 4: "Declining"})
    assert summary.iloc[0]["recall_on_degradation"] == pytest.approx(1.0)
    assert summary.iloc[0]["discrimination_margin"] == pytest.approx(0.0)


def test_discrimination_mismatched_inputs_are_rejected():
    with pytest.raises(ValueError, match="same samples"):
        confounder_confusion(np.zeros(5), np.zeros(4), degradation_classes=[1])


def test_discrimination_analysis_is_saved_with_its_limit(tmp_path):
    reference = np.array([1] * 8 + [2] * 8 + [4] * 8)
    predictions = {"good": np.r_[np.zeros(16), np.ones(8)].astype(int),
                   "bad": np.ones(24, int)}
    result = run_discrimination_analysis(
        reference, predictions, tmp_path, degradation_classes=[4],
        class_names={1: "Stable", 2: "Cyclic", 4: "Declining"})
    assert (tmp_path / "confounder_confusion.csv").exists()
    assert (tmp_path / "discrimination_summary.csv").exists()
    report = json.loads((tmp_path / "discrimination.json").read_text())
    assert report["best_by_margin"]["method"] == "good"
    assert report["interpretation_limit"] == DISCRIMINATION_LIMIT
    assert "generator" in DISCRIMINATION_LIMIT


def test_discrimination_reports_every_confounder_separately(tmp_path):
    reference = np.array([1] * 6 + [2] * 6 + [3] * 6 + [4] * 6)
    result = run_discrimination_analysis(
        reference, {"m": np.r_[np.zeros(12), np.ones(12)].astype(int)},
        tmp_path, degradation_classes=[4],
        class_names={1: "Stable", 2: "Cropland", 3: "Cyclic", 4: "Declining"})
    row = result["summary"].iloc[0]
    assert row["false_positive_rate_cyclic"] == pytest.approx(1.0)
    assert row["false_positive_rate_stable"] == pytest.approx(0.0)


# ------------------------------------------------- statistical baselines
def test_restrend_baseline_flags_fewer_rainfall_driven_pixels(built):
    """Experiment 2 must improve on Experiment 1 where climate explains it."""
    _, _, truth, table, _, _ = built
    cfg = Config()
    trend = baseline_trend_prediction(table, cfg)["predictions"].astype(bool)
    restrend = baseline_restrend_prediction(
        table, cfg)["predictions"].astype(bool)
    rainfall_driven = truth == 2
    assert restrend[rainfall_driven].mean() < trend[rainfall_driven].mean()
    # and it must not lose the genuine decline
    persistent = truth == 4
    assert restrend[persistent].mean() > 0.7


def test_restrend_baseline_uses_validity_not_significance(built):
    """Regression guard for a subtle bug.

    Gating on `restrend_significant` instead of `restrend_valid` keeps every
    climate-explained decline flagged, silently making Experiment 2
    identical to Experiment 1.
    """
    _, _, truth, table, _, _ = built
    cfg = Config()
    trend = baseline_trend_prediction(table, cfg)["predictions"]
    restrend = baseline_restrend_prediction(table, cfg)["predictions"]
    assert not np.array_equal(trend, restrend), \
        "the climate-adjusted rule must differ from the raw trend rule"
    assert "restrend_valid" in table.columns


def test_integrated_rule_excludes_cyclic_pixels(built):
    """Experiments 3-4 as a rule: cyclic behaviour must be excluded."""
    _, _, truth, table, _, _ = built
    cfg = Config()
    restrend = baseline_restrend_prediction(
        table, cfg)["predictions"].astype(bool)
    integrated = baseline_integrated_prediction(
        table, cfg)["predictions"].astype(bool)
    assert integrated.sum() <= restrend.sum()
    assert integrated[truth == 4].mean() > 0.7


def test_baselines_expose_their_scores_as_uncalibrated(built):
    _, _, _, table, _, _ = built
    for rule in (baseline_trend_prediction, baseline_restrend_prediction,
                 baseline_integrated_prediction):
        result = rule(table, Config())
        assert result["probabilities"].shape == (len(table), 2)
        assert np.allclose(result["probabilities"].sum(axis=1), 1.0)
        assert "not a calibrated probability" in result["note"] \
            or "no calibrated probability" in result["note"]


def test_baselines_respect_the_configured_alpha(built):
    _, _, _, table, _, _ = built
    strict, lenient = Config(), Config()
    strict.trend.alpha, lenient.trend.alpha = 1e-6, 0.5
    for rule in (baseline_trend_prediction, baseline_restrend_prediction,
                 baseline_integrated_prediction):
        assert rule(table, strict)["predictions"].sum() <= \
            rule(table, lenient)["predictions"].sum()
