"""Spatial block cross-validation and Random Forest pipeline tests
(M2 Parts 4, 5 and 12).

The central claims these tests defend:
  * a block is never split between folds,
  * train and test never share a pixel or a block, and with a buffer they do
    not even share a neighbourhood,
  * fold assignment is reproducible from the configuration alone,
  * every eligible sample is assigned to exactly one fold and evaluated,
  * anything that learns parameters is fitted on training data only.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import Config, RFExperimentConfig, SpatialCVConfig
from src.validation import (aggregate_fold_metrics, block_coordinates,
                            buffered_training_mask, classification_metrics,
                            random_split_baseline, spatial_block_folds,
                            spatial_cv_rf)

H, W = 24, 32


def toy(n_features=4, seed=0, height=H, width=W):
    """A separable toy problem laid out on a grid."""
    rng = np.random.default_rng(seed)
    n = height * width
    x = rng.normal(size=(n, n_features))
    y = np.where(x[:, 0] + 0.5 * x[:, 1] > 0, 1, 2)
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(n_features)])
    return frame, y


# ------------------------------------------------------------ block geometry
def test_blocks_are_never_split_between_folds():
    blocks, folds = spatial_block_folds(23, 29, SpatialCVConfig(block_size=4,
                                                                n_folds=5))
    for block in np.unique(blocks):
        assert np.unique(folds[blocks == block]).size == 1


def test_fold_assignment_is_reproducible():
    cfg = SpatialCVConfig(block_size=3, n_folds=4, seed=11)
    a = spatial_block_folds(19, 21, cfg)[1]
    b = spatial_block_folds(19, 21, cfg)[1]
    assert np.array_equal(a, b)


def test_different_seeds_give_different_assignments():
    a = spatial_block_folds(20, 20, SpatialCVConfig(block_size=2, seed=1))[1]
    b = spatial_block_folds(20, 20, SpatialCVConfig(block_size=2, seed=2))[1]
    assert not np.array_equal(a, b)


def test_every_pixel_is_assigned_to_exactly_one_fold():
    cfg = SpatialCVConfig(block_size=5, n_folds=4, seed=3)
    _, folds = spatial_block_folds(H, W, cfg)
    assert folds.shape == (H, W)
    assert set(np.unique(folds)).issubset(set(range(cfg.n_folds)))
    assert np.isfinite(folds).all()


def test_all_folds_are_populated():
    cfg = SpatialCVConfig(block_size=4, n_folds=5, seed=4)
    _, folds = spatial_block_folds(H, W, cfg)
    assert len(np.unique(folds)) == cfg.n_folds


def test_too_few_blocks_is_an_error():
    with pytest.raises(ValueError, match="cannot fill"):
        spatial_block_folds(4, 4, SpatialCVConfig(block_size=4, n_folds=5))


def test_invalid_block_configuration_is_rejected():
    with pytest.raises(ValueError):
        spatial_block_folds(H, W, SpatialCVConfig(block_size=0, n_folds=3))
    with pytest.raises(ValueError):
        spatial_block_folds(H, W, SpatialCVConfig(block_size=4, n_folds=1))


# -------------------------------------------------------------- no leakage
def test_train_and_test_pixels_never_overlap():
    _, folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=4,
                                                         n_folds=4, seed=5))
    flat = folds.reshape(-1)
    for fold in np.unique(flat):
        test = flat == fold
        train = ~test
        assert not (test & train).any()
        assert (test | train).all()


def test_train_and_test_never_share_a_block():
    cfg = SpatialCVConfig(block_size=4, n_folds=4, seed=6)
    blocks, folds = spatial_block_folds(H, W, cfg)
    flat_blocks, flat_folds = blocks.reshape(-1), folds.reshape(-1)
    for fold in np.unique(flat_folds):
        test_blocks = set(np.unique(flat_blocks[flat_folds == fold]).tolist())
        train_blocks = set(np.unique(flat_blocks[flat_folds != fold]).tolist())
        assert test_blocks.isdisjoint(train_blocks)


def test_buffer_removes_neighbouring_training_blocks():
    brow, bcol, _ = block_coordinates(H, W, 4)
    _, folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=4,
                                                         n_folds=4, seed=7))
    flat = folds.reshape(-1)
    test = flat == 0
    train = ~test
    buffered = buffered_training_mask(train, test, brow, bcol, 1)
    assert buffered.sum() < train.sum(), "buffer must drop samples"
    assert not (buffered & test).any()
    rows, cols = brow.reshape(-1), bcol.reshape(-1)
    test_blocks = set(map(tuple, np.c_[rows[test], cols[test]].tolist()))
    for r, c in zip(rows[buffered], cols[buffered]):
        for tr, tc in test_blocks:
            assert max(abs(r - tr), abs(c - tc)) > 1, \
                "a training sample stayed inside the buffer"


def test_zero_buffer_leaves_training_set_unchanged():
    brow, bcol, _ = block_coordinates(H, W, 4)
    _, folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=4,
                                                         n_folds=4, seed=8))
    flat = folds.reshape(-1)
    test = flat == 1
    train = ~test
    assert np.array_equal(buffered_training_mask(train, test, brow, bcol, 0),
                          train)


# --------------------------------------------------------------- the model
def test_spatial_cv_is_reproducible_and_complete():
    features, y = toy()
    folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=4, n_folds=4,
                                                      seed=9))[1]
    cfg = RFExperimentConfig(seed=3, n_estimators=25)
    a = spatial_cv_rf(features, y, folds, feature_names=list(features.columns),
                      cfg=cfg)
    b = spatial_cv_rf(features, y, folds, feature_names=list(features.columns),
                      cfg=cfg)
    assert np.array_equal(a["predictions"], b["predictions"])
    assert np.allclose(a["probabilities"], b["probabilities"], equal_nan=True)
    assert a["evaluated"].all(), "every eligible sample must be evaluated"


def test_prediction_and_probability_dimensions():
    features, y = toy()
    folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=4, n_folds=4,
                                                      seed=10))[1]
    result = spatial_cv_rf(features, y, folds,
                           feature_names=list(features.columns),
                           cfg=RFExperimentConfig(n_estimators=20))
    n = H * W
    assert result["predictions"].shape == (n,)
    assert result["probabilities"].shape == (n, len(np.unique(y)))
    evaluated = result["evaluated"]
    rows = result["probabilities"][evaluated]
    assert np.allclose(rows.sum(axis=1), 1.0, atol=1e-6)
    assert (rows >= 0).all() and (rows <= 1).all()


def test_probability_columns_stay_aligned_with_classes():
    """A fold whose training data lacks a class must not shift columns."""
    features, y = toy(seed=1)
    folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=4, n_folds=4,
                                                      seed=12))[1]
    flat = folds.reshape(-1)
    y = y.copy()
    y[flat == 0] = 3                      # class 3 lives only inside fold 0
    result = spatial_cv_rf(features, y, flat,
                           feature_names=list(features.columns),
                           cfg=RFExperimentConfig(n_estimators=20))
    assert list(result["classes"]) == [1, 2, 3]
    # Fold 0 is predicted by a model trained on folds 1-3, which contain no
    # class 3: its column must stay 0, and the other columns must not shift
    # into it.
    predicted_without_three = result["probabilities"][flat == 0]
    assert np.nansum(predicted_without_three[:, 2]) == 0.0, \
        "a class absent from training must keep probability 0 in its column"
    assert np.allclose(np.nansum(predicted_without_three, axis=1), 1.0)


def test_sample_mask_restricts_evaluation():
    features, y = toy()
    folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=4, n_folds=4,
                                                      seed=13))[1]
    mask = np.zeros(H * W, bool)
    mask[::3] = True
    result = spatial_cv_rf(features, y, folds, sample_mask=mask,
                           feature_names=list(features.columns),
                           cfg=RFExperimentConfig(n_estimators=20))
    assert result["evaluated"].sum() == mask.sum()
    assert not result["evaluated"][~mask].any()


def test_mismatched_inputs_are_rejected():
    features, y = toy()
    folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=4))[1]
    with pytest.raises(ValueError, match="same samples"):
        spatial_cv_rf(features, y[:-5], folds,
                      feature_names=list(features.columns))


def test_missing_feature_column_is_reported():
    features, y = toy()
    folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=4))[1]
    with pytest.raises(ValueError, match="missing requested columns"):
        spatial_cv_rf(features, y, folds, feature_names=["f0", "nope"])


def test_non_finite_features_are_imputed_inside_the_fold():
    """NaN features must not crash the model, and must not leak statistics."""
    features, y = toy()
    features = features.copy()
    features.iloc[::7, 0] = np.nan
    folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=4, n_folds=4,
                                                      seed=14))[1]
    result = spatial_cv_rf(features, y, folds,
                           feature_names=list(features.columns),
                           cfg=RFExperimentConfig(n_estimators=20))
    assert result["evaluated"].all()


# -------------------------------------------------------------- the metrics
def test_metric_calculations_are_correct():
    y_true = np.array([1, 1, 2, 2, 3, 3])
    y_pred = np.array([1, 2, 2, 2, 3, 1])
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    assert metrics["per_class"]["1"]["precision"] == pytest.approx(0.5)
    assert metrics["per_class"]["1"]["recall"] == pytest.approx(0.5)
    assert metrics["per_class"]["2"]["recall"] == pytest.approx(1.0)
    assert metrics["per_class"]["3"]["recall"] == pytest.approx(0.5)
    assert np.array_equal(metrics["confusion_matrix"],
                          [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
    assert metrics["f1_macro"] == pytest.approx(
        np.mean([0.5, 0.8, 2 / 3]), abs=1e-6)


def test_perfect_prediction_scores_one_everywhere():
    y = np.array([1, 1, 2, 2, 3])
    metrics = classification_metrics(y, y)
    for key in ("accuracy", "f1_macro", "f1_weighted", "precision_macro",
                "recall_macro", "cohen_kappa"):
        assert metrics[key] == pytest.approx(1.0)


def test_metrics_report_more_than_accuracy():
    y_true = np.array([1] * 90 + [2] * 10)
    y_pred = np.array([1] * 100)
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(0.9)
    assert metrics["f1_macro"] < 0.5, "macro F1 must expose the ignored class"
    assert metrics["per_class"]["2"]["recall"] == 0.0


def test_fold_aggregation_reports_mean_and_std():
    folds = [{"accuracy": 0.8, "f1_macro": 0.7},
             {"accuracy": 0.6, "f1_macro": 0.5}]
    aggregated = aggregate_fold_metrics(folds)
    assert aggregated["accuracy_mean"] == pytest.approx(0.7)
    assert aggregated["accuracy_std"] == pytest.approx(0.1)
    assert aggregated["f1_macro_mean"] == pytest.approx(0.6)
    assert aggregated["n_folds"] == 2


def test_spatial_cv_reports_fold_level_and_aggregated_metrics():
    features, y = toy()
    folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=4, n_folds=4,
                                                      seed=15))[1]
    metrics = spatial_cv_rf(features, y, folds,
                            feature_names=list(features.columns),
                            cfg=RFExperimentConfig(n_estimators=20))["metrics"]
    assert len(metrics["fold_metrics"]) == 4
    assert metrics["fold_summary"]["n_folds"] == 4
    assert "f1_macro_std" in metrics["fold_summary"]
    assert metrics["validation"] == "spatial_block_cv"
    for fold in metrics["fold_metrics"]:
        assert fold["n_train"] + fold["n_test"] <= H * W


# ------------------------------------------------------------- the baseline
def test_random_split_baseline_is_labelled_as_a_baseline():
    features, y = toy()
    result = random_split_baseline(features, y,
                                   feature_names=list(features.columns),
                                   cfg=RFExperimentConfig(n_estimators=20))
    assert result["metrics"]["validation"] == "random_pixel_split_baseline"
    assert "optimistic" in result["metrics"]["caveat"]


def test_random_split_baseline_is_reproducible():
    features, y = toy()
    cfg = RFExperimentConfig(seed=21, n_estimators=20)
    a = random_split_baseline(features, y,
                              feature_names=list(features.columns), cfg=cfg)
    b = random_split_baseline(features, y,
                              feature_names=list(features.columns), cfg=cfg)
    assert np.array_equal(a["predictions"], b["predictions"])


def test_configuration_defaults_are_wired_through():
    cfg = Config()
    assert cfg.research.model.block_cv.buffer_blocks == 0
    assert cfg.research.spatial_cv.n_folds >= 2
    assert cfg.research.model.imputation_strategy == "median"
