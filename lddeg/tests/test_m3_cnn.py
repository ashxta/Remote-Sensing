"""1D CNN research-experiment tests (M3 Parts 3 and 11).

Tests that need the optional `torch` dependency skip cleanly when it is
absent; the split, normalisation and confidence helpers are tested either
way, because those are where leakage would actually occur.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.cnn_experiment import (CNNConfig, CNNExperimentConfig,
                                confidence_outputs, run_spatial_cnn,
                                spatial_train_validation_test,
                                torch_available, training_normalizer,
                                transform_series)
from src.config import SpatialCVConfig
from src.validation import spatial_block_folds

torch_required = pytest.mark.skipif(not torch_available(),
                                    reason="optional dependency 'torch' "
                                           "is not installed")
H, W = 12, 12
T = 30


def sequences(seed=0):
    """Four separable temporal archetypes laid out on a grid."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    columns, labels = [], []
    for i in range(H * W):
        kind = i % 4
        if kind == 0:
            s = 0.75 + rng.normal(0, 0.02, T)
        elif kind == 1:
            s = 0.7 - 0.015 * t + rng.normal(0, 0.02, T)
        elif kind == 2:
            s = np.concatenate([np.full(10, 0.7),
                                np.linspace(0.3, 0.7, T - 10)]) \
                + rng.normal(0, 0.02, T)
        else:
            s = 0.5 + 0.2 * np.sin(2 * np.pi * t / 6) + rng.normal(0, 0.02, T)
        columns.append(s)
        labels.append(kind + 1)
    return np.array(columns).T, np.array(labels)


# -------------------------------------------------------------------- split
def test_split_is_disjoint_and_uses_whole_folds():
    folds = np.repeat([0, 1, 2, 3], 5)
    train, valid, test = spatial_train_validation_test(
        folds, CNNExperimentConfig(test_fold=0, validation_fold=1))
    assert not np.any(train & valid)
    assert not np.any(train & test)
    assert not np.any(valid & test)
    assert np.all(test[folds == 0]) and np.all(valid[folds == 1])
    assert np.all(train[folds >= 2])


def test_split_covers_every_sample_exactly_once():
    folds = np.repeat([0, 1, 2, 3], 5)
    train, valid, test = spatial_train_validation_test(
        folds, CNNExperimentConfig(test_fold=0, validation_fold=1))
    assert np.all(train.astype(int) + valid.astype(int) + test.astype(int) == 1)


def test_split_from_cnn_config_reserves_validation_folds():
    folds = np.repeat([0, 1, 2, 3, 4], 4)
    train, valid, test = spatial_train_validation_test(
        folds, CNNConfig(validation_folds=2))
    assert valid.sum() == 8 and test.sum() == 4
    assert not np.any(train & (valid | test))


def test_split_requires_three_folds():
    with pytest.raises(ValueError, match="three spatial folds"):
        spatial_train_validation_test(np.repeat([0, 1], 5), CNNConfig())


# ------------------------------------------------------------ normalisation
def test_normalization_is_fit_only_on_training_samples():
    series = np.array([[1., 3., 100.], [2., 4., 200.]])
    normalizer = training_normalizer(series, np.array([True, True, False]))
    assert np.allclose(normalizer[0], [2., 3.])
    transformed = transform_series(series, normalizer)
    assert transformed.shape == (3, 1, 2)


def test_test_samples_cannot_change_the_normalizer():
    """The leakage test: mutating held-out columns must not move the scale."""
    rng = np.random.default_rng(1)
    series = rng.normal(0.5, 0.1, (T, 20))
    train = np.zeros(20, bool)
    train[:10] = True
    before = training_normalizer(series, train)
    contaminated = series.copy()
    contaminated[:, 10:] = 999.0
    after = training_normalizer(contaminated, train)
    for a, b in zip(before, after):
        assert np.allclose(a, b)


def test_missing_values_are_filled_from_training_medians():
    series = np.array([[1.0, 3.0, np.nan], [2.0, 4.0, np.nan]])
    normalizer = training_normalizer(series, np.array([True, True, False]))
    transformed = transform_series(series, normalizer)
    assert np.isfinite(transformed).all()


# ---------------------------------------------------------------- confidence
def test_probability_confidence_and_uncertainty_flags():
    confidence, uncertain = confidence_outputs([[.8, .2], [.51, .49]],
                                               threshold=.6)
    assert np.allclose(confidence, [.8, .51])
    assert uncertain.tolist() == [False, True]
    with pytest.raises(ValueError):
        confidence_outputs([[.2, .2]])


# --------------------------------------------------------------- experiment
@torch_required
def test_cnn_experiment_is_reproducible(tmp_path):
    series, labels = sequences()
    folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=3, n_folds=4,
                                                      seed=1))[1].reshape(-1)
    cfg = CNNConfig(max_epochs=4, patience=2, batch_size=16, max_folds=2,
                    seed=11)
    first = run_spatial_cnn(series, labels, folds, tmp_path / "a", cfg)
    second = run_spatial_cnn(series, labels, folds, tmp_path / "b", cfg)
    assert np.array_equal(first["predictions"], second["predictions"])
    assert np.allclose(first["probabilities"], second["probabilities"],
                       equal_nan=True)
    assert first["metrics"]["f1_macro"] == second["metrics"]["f1_macro"]


@torch_required
def test_cnn_outputs_have_the_expected_shapes(tmp_path):
    series, labels = sequences()
    folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=3, n_folds=4,
                                                      seed=2))[1].reshape(-1)
    result = run_spatial_cnn(series, labels, folds, tmp_path,
                             CNNConfig(max_epochs=3, patience=2, max_folds=2))
    n = H * W
    assert result["predictions"].shape == (n,)
    assert result["probabilities"].shape == (n, len(np.unique(labels)))
    evaluated = result["evaluated"]
    rows = result["probabilities"][evaluated]
    assert np.allclose(rows.sum(axis=1), 1.0, atol=1e-5)
    assert (rows >= 0).all() and (rows <= 1).all()


@torch_required
def test_cnn_reports_per_class_metrics_and_curves(tmp_path):
    series, labels = sequences()
    folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=3, n_folds=4,
                                                      seed=3))[1].reshape(-1)
    result = run_spatial_cnn(series, labels, folds, tmp_path,
                             CNNConfig(max_epochs=4, patience=2, max_folds=2))
    metrics = result["metrics"]
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro",
                "f1_weighted", "per_class", "confusion_matrix",
                "fold_summary", "uncertainty"):
        assert key in metrics
    for per_class in metrics["per_class"].values():
        assert {"precision", "recall", "f1", "support"} <= set(per_class)
    history = pd.read_csv(tmp_path / "training_history.csv")
    assert {"fold", "epoch", "train_loss", "validation_loss"} <= set(history)
    assert len(history) > 0


@torch_required
def test_cnn_saves_checkpoints_and_artifacts(tmp_path):
    series, labels = sequences()
    folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=3, n_folds=4,
                                                      seed=4))[1].reshape(-1)
    run_spatial_cnn(series, labels, folds, tmp_path,
                    CNNConfig(max_epochs=3, patience=2, max_folds=1))
    for artifact in ("metrics.json", "configuration.json", "predictions.csv",
                     "probabilities.csv", "confusion_matrix.csv",
                     "training_history.csv"):
        assert (tmp_path / artifact).exists(), artifact
    assert list(tmp_path.glob("checkpoint_fold*.pt")), "no checkpoint saved"
    configuration = json.loads((tmp_path / "configuration.json").read_text())
    assert configuration["folds"][0]["best_epoch"] >= 1


@torch_required
def test_cnn_early_stopping_respects_patience(tmp_path):
    series, labels = sequences()
    folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=3, n_folds=4,
                                                      seed=5))[1].reshape(-1)
    result = run_spatial_cnn(series, labels, folds, tmp_path,
                             CNNConfig(max_epochs=50, patience=2, max_folds=1))
    fold = result["metrics"]["fold_metrics"][0]
    assert fold["epochs_run"] <= 50
    assert fold["best_epoch"] <= fold["epochs_run"]


@torch_required
def test_cnn_test_fold_is_never_used_for_training_or_early_stopping(tmp_path):
    """Structural leakage check on the reported split sizes."""
    series, labels = sequences()
    folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=3, n_folds=4,
                                                      seed=6))[1].reshape(-1)
    result = run_spatial_cnn(series, labels, folds, tmp_path,
                             CNNConfig(max_epochs=3, patience=2))
    for fold in result["metrics"]["fold_metrics"]:
        assert fold["fold"] not in fold["validation_folds"]
        total = fold["n_train"] + fold["n_validation"] + fold["n_test"]
        assert total == len(labels)


@torch_required
def test_cnn_uses_every_fold_under_spatial_cv(tmp_path):
    series, labels = sequences()
    folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=3, n_folds=4,
                                                      seed=7))[1].reshape(-1)
    result = run_spatial_cnn(series, labels, folds, tmp_path,
                             CNNConfig(max_epochs=2, patience=1))
    assert result["evaluated"].all()
    assert len(result["metrics"]["fold_metrics"]) == 4
    assert result["metrics"]["validation"] == "spatial_block_cv"


@torch_required
def test_cnn_learns_something_on_separable_archetypes(tmp_path):
    """A sanity floor: the model must beat guessing on easy synthetic data."""
    series, labels = sequences()
    folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=3, n_folds=4,
                                                      seed=8))[1].reshape(-1)
    result = run_spatial_cnn(series, labels, folds, tmp_path,
                             CNNConfig(max_epochs=40, patience=8, seed=3))
    assert result["metrics"]["f1_macro"] > 0.30, "worse than a coin flip"
