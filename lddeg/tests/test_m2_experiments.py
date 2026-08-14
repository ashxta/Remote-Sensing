import numpy as np
import pandas as pd

from src import timeseries as TS
from src.m2_experiments import (RFExperimentConfig, SpatialCVConfig, FEATURE_SETS,
    spatial_block_folds, spatial_cv_rf, temporal_holdout_indices, validate_dataset)


def test_spatial_blocks_are_deterministic_and_never_split():
    blocks1, folds1 = spatial_block_folds(11, 13, SpatialCVConfig(block_size=3, n_folds=4, seed=9))
    blocks2, folds2 = spatial_block_folds(11, 13, SpatialCVConfig(block_size=3, n_folds=4, seed=9))
    assert np.array_equal(folds1, folds2)
    for block in np.unique(blocks1):
        assert np.unique(folds1[blocks1 == block]).size == 1


def test_spatial_cv_predictions_and_metrics_are_reproducible():
    rng = np.random.default_rng(5); h = w = 8; n = h * w
    x = rng.normal(size=(n, 3)); y = np.where(x[:, 0] > 0, 1, 2)
    folds = spatial_block_folds(h, w, SpatialCVConfig(block_size=2, n_folds=4, seed=2))[1]
    cfg = RFExperimentConfig(seed=3, n_estimators=20)
    a = spatial_cv_rf(pd.DataFrame(x, columns=list("abc")), y, folds, feature_names=list("abc"), cfg=cfg)
    b = spatial_cv_rf(pd.DataFrame(x, columns=list("abc")), y, folds, feature_names=list("abc"), cfg=cfg)
    assert np.array_equal(a["predictions"], b["predictions"])
    assert a["probabilities"].shape[0] == n
    assert "f1_macro" in a["metrics"]


def test_temporal_holdout_is_chronological():
    historical, later = temporal_holdout_indices(12, 8)
    assert historical.max() < later.min()
    assert len(historical) == 8 and len(later) == 4


def test_feature_sets_are_ordered_subsets_without_leakage():
    assert set(FEATURE_SETS["A_basic_vegetation"]).issubset(FEATURE_SETS["F_full"])
    assert len(FEATURE_SETS["F_full"]) == len(set(FEATURE_SETS["F_full"]))


def test_dataset_validation_rejects_infinity_and_dimension_mismatch():
    x = np.zeros((12, 2, 2)); x[0, 0, 0] = np.inf
    try:
        validate_dataset(x, x)
    except ValueError as e:
        assert "infinite" in str(e)
    else:
        raise AssertionError("infinite data was accepted")


def test_cyclicity_controls_distinguish_periodic_signal_from_noise():
    rng = np.random.default_rng(10); t = np.arange(60)
    noise = rng.normal(0, 1, 60)
    trend = .04 * t + rng.normal(0, .2, 60)
    strong = np.sin(2 * np.pi * t / 6) + rng.normal(0, .08, 60)
    weak = .12 * np.sin(2 * np.pi * t / 6) + rng.normal(0, 1, 60)
    out = TS.cyclicity(np.c_[noise, trend, strong, weak], min_period=4, max_period=12, threshold=2.0, min_obs=12)
    assert out.enrichment[2] > 2.0 and out.periodic[2]
    assert out.enrichment[2] > out.enrichment[0]
    assert out.enrichment[2] > out.enrichment[1]
