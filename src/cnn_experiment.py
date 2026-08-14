"""1D CNN research experiment on raw vegetation sequences (M3 Part 3).

RESEARCH QUESTION
-----------------
Can a model that learns temporal patterns directly from the NDVI sequence
match or beat the engineered temporal features (trend, RESTREND, cyclicity,
breakpoint, recovery) that M1/M2 compute explicitly?

The comparison is only meaningful if both sides are evaluated the same way,
so the CNN is validated with the SAME spatial block cross-validation as the
Random Forest: every fold is held out in turn, whole blocks at a time, and
the reported number is the pooled out-of-fold result plus the across-fold
spread.

Within each CV iteration there are three disjoint spatial groups:

    test fold        the held-out fold; never seen during training,
                     normalisation or early stopping
    validation folds `CNNConfig.validation_folds` folds taken FROM THE
                     TRAINING SIDE, used for early stopping and checkpoint
                     selection only
    training folds   everything else

The architecture is deliberately small and fixed across the comparison: the
question is representation (learned vs engineered), not capacity. Making the
network bigger would confound the two.

Torch is an optional dependency. Importing this module is always safe; only
running an experiment requires it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CNNConfig, Config, UncertaintyConfig
from .reproducibility import set_seed
from .uncertainty import prediction_confidence, uncertainty_summary
from .validation import aggregate_fold_metrics, classification_metrics

__all__ = ["CNNExperimentConfig", "CNNConfig", "spatial_train_validation_test",
           "training_normalizer", "transform_series", "confidence_outputs",
           "run_spatial_cnn", "torch_available"]


@dataclass
class CNNExperimentConfig:
    """Backwards-compatible view of the M2-era CNN configuration.

    New code should configure the CNN through `Config.research.cnn`
    (`CNNConfig`); this dataclass is kept because the first M3 draft and its
    tests refer to it, and it still describes one train/validation/test
    split.
    """
    seed: int = 42
    channels: tuple = (32, 64)
    kernel_size: int = 5
    dropout: float = .3
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 80
    patience: int = 10
    confidence_threshold: float = .60
    test_fold: int = 0
    validation_fold: int = 1

    def to_cnn_config(self) -> CNNConfig:
        return CNNConfig(seed=self.seed, channels=list(self.channels),
                         kernel_size=self.kernel_size, dropout=self.dropout,
                         learning_rate=self.learning_rate,
                         batch_size=self.batch_size,
                         max_epochs=self.max_epochs, patience=self.patience,
                         validation_folds=1, spatial_cv=False)


def torch_available() -> bool:
    """True when the optional deep-learning dependency is installed."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as error:                       # pragma: no cover
        raise RuntimeError(
            "the 1D CNN experiment requires the optional dependency 'torch'; "
            "install it with `pip install torch` (see requirements.txt)"
        ) from error
    return torch, nn


# ------------------------------------------------------------------- splits
def spatial_train_validation_test(folds, cfg=None):
    """Disjoint spatial train/validation/test masks for one split.

    Whole folds are allocated; this is deliberately not a random-pixel split.
    Accepts either the legacy `CNNExperimentConfig` (explicit test and
    validation fold ids) or a `CNNConfig` with `validation_folds`.
    """
    folds = np.asarray(folds).reshape(-1)
    cfg = cfg or CNNExperimentConfig()
    available = sorted(int(f) for f in np.unique(folds))
    if isinstance(cfg, CNNExperimentConfig):
        if cfg.test_fold == cfg.validation_fold:
            raise ValueError("test and validation folds must differ")
        if not {cfg.test_fold, cfg.validation_fold}.issubset(set(available)):
            raise ValueError("configured test/validation fold is absent")
        test = folds == cfg.test_fold
        validation = folds == cfg.validation_fold
    else:
        if len(available) < 3:
            raise ValueError("at least three spatial folds are required for a "
                             "CNN train/validation/test split")
        test = folds == available[0]
        validation = np.isin(folds, available[1:1 + max(cfg.validation_folds,
                                                        1)])
    train = ~(test | validation)
    if not train.any():
        raise ValueError("at least three spatial folds are required for CNN "
                         "train/validation/test")
    return train, validation, test


def _split_for_fold(folds: np.ndarray, mask: np.ndarray, test_fold: int,
                    n_validation: int, seed: int):
    """Test fold, validation folds carved out of the training side, rest."""
    test = mask & (folds == test_fold)
    remaining = sorted(int(f) for f in np.unique(folds[mask & ~test]))
    if len(remaining) < 2:
        raise ValueError("a CNN fold needs at least one validation fold and "
                         "one training fold besides the test fold")
    n_validation = min(max(n_validation, 1), len(remaining) - 1)
    # Deterministic, and different per test fold so validation is not always
    # the same corner of the map.
    order = np.random.default_rng(seed + test_fold).permutation(remaining)
    validation_folds = set(int(f) for f in order[:n_validation])
    validation = mask & np.isin(folds, list(validation_folds)) & ~test
    train = mask & ~test & ~validation
    return train, validation, test, sorted(validation_folds)


# ----------------------------------------------------------- normalisation
def training_normalizer(series, train_mask):
    """Fit per-time-step median and scale on TRAINING samples only.

    The median fills missing observations and the mean/std standardise the
    sequence. Every statistic comes from training columns, so no test sample
    influences the scale its own features are expressed in.
    """
    x = np.asarray(series, dtype="float64")          # (T, N)
    train = x[:, np.asarray(train_mask, bool)]
    if train.size == 0:
        raise ValueError("normalizer needs at least one training sample")
    median = np.nanmedian(train, axis=1)
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isfinite(train), train, median[:, None])
    mean, std = filled.mean(axis=1), filled.std(axis=1)
    return median, mean, np.where(std > 1e-8, std, 1.0)


def transform_series(series, normalizer):
    """Apply a fitted normalizer; returns (N, 1, T) float32 for torch."""
    median, mean, std = normalizer
    x = np.asarray(series, dtype="float64")
    x = np.where(np.isfinite(x), x, median[:, None])
    return ((x - mean[:, None]) / std[:, None]).T[:, None, :].astype("float32")


def confidence_outputs(probabilities, threshold=.60):
    """Model confidence and an explicit low-confidence flag.

    Thin wrapper over `uncertainty.prediction_confidence`, kept for callers
    written against the first M3 draft.
    """
    measures = prediction_confidence(
        probabilities, cfg=UncertaintyConfig(confidence_threshold=threshold,
                                             margin_threshold=0.0))
    return measures["confidence"], measures["confidence"] < threshold


# ------------------------------------------------------------------- model
def _build_network(nn, n_classes: int, cfg: CNNConfig):
    """Small, fixed 1D CNN. Capacity is held constant across experiments."""
    layers, in_channels = [], 1
    for out_channels in cfg.channels:
        layers += [nn.Conv1d(in_channels, out_channels, cfg.kernel_size,
                             padding=cfg.kernel_size // 2),
                   nn.BatchNorm1d(out_channels), nn.ReLU()]
        in_channels = out_channels
    layers += [nn.AdaptiveAvgPool1d(1), nn.Flatten(),
               nn.Dropout(cfg.dropout), nn.Linear(in_channels, n_classes)]
    return nn.Sequential(*layers)


def _train_one(x, y_index, train, validation, cfg: CNNConfig, n_classes: int,
               checkpoint: Path, seed: int):
    """Train with early stopping on validation loss; return best state."""
    torch, nn = _require_torch()
    set_seed(seed, deterministic=cfg.deterministic)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_network(nn, n_classes, cfg).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                 weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    def tensors(mask):
        return (torch.tensor(x[mask], device=device),
                torch.tensor(y_index[mask], device=device))

    x_train, y_train = tensors(train)
    x_validation, y_validation = tensors(validation)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train),
        batch_size=cfg.batch_size, shuffle=True, generator=generator)

    history, best, stale, best_epoch = [], float("inf"), 0, 0
    for epoch in range(cfg.max_epochs):
        model.train()
        total, seen = 0.0, 0
        for batch_x, batch_y in loader:
            optimiser.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimiser.step()
            total += float(loss.detach()) * len(batch_y)
            seen += len(batch_y)
        model.eval()
        with torch.no_grad():
            logits = model(x_validation)
            validation_loss = float(loss_fn(logits, y_validation))
            validation_accuracy = float(
                (logits.argmax(1) == y_validation).float().mean())
        history.append({"epoch": epoch + 1,
                        "train_loss": total / max(seen, 1),
                        "validation_loss": validation_loss,
                        "validation_accuracy": validation_accuracy})
        if validation_loss < best - 1e-7:
            best, stale, best_epoch = validation_loss, 0, epoch + 1
            torch.save({"state_dict": model.state_dict(),
                        "n_classes": n_classes, "config": asdict(cfg),
                        "epoch": best_epoch}, checkpoint)
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, history, best_epoch, device


def _predict(model, x, mask, device):
    """Deterministic inference: eval mode, no grad, softmax probabilities."""
    torch, _ = _require_torch()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x[mask], device=device))
        return torch.softmax(logits, dim=1).cpu().numpy()


def run_spatial_cnn(series, labels, fold_grid, output_dir, cfg=None, *,
                    sample_mask=None, uncertainty_cfg=None, logger=None):
    """Spatially cross-validated 1D CNN on the raw NDVI sequences.

    `series` is (time, samples); `fold_grid` may be flat or a 2-D grid with
    the same number of entries. With `cfg.spatial_cv` every fold is held out
    in turn; otherwise a single train/validation/test split is run.

    Test samples never influence normalisation, early stopping, checkpoint
    selection or optimisation - only the final metrics.

    Writes per-fold checkpoints, training history, predictions,
    probabilities, confidence and uncertainty flags, per-class metrics and a
    configuration snapshot to `output_dir`.
    """
    if isinstance(cfg, CNNExperimentConfig):
        cfg = cfg.to_cnn_config()
    elif isinstance(cfg, Config):
        cfg = cfg.research.cnn
    cfg = cfg or CNNConfig()
    uncertainty_cfg = uncertainty_cfg or UncertaintyConfig()
    _require_torch()

    x_raw = np.asarray(series, dtype="float64")
    y = np.asarray(labels)
    folds = np.asarray(fold_grid).reshape(-1)
    if x_raw.shape[1] != len(y) or len(y) != len(folds):
        raise ValueError("series, labels and folds must describe identical "
                         "samples")
    mask = np.ones(len(y), bool) if sample_mask is None \
        else np.asarray(sample_mask, bool)
    if not mask.any():
        raise ValueError("sample_mask selects no samples")

    classes = np.unique(y[mask])
    class_index = {c: i for i, c in enumerate(classes)}
    y_index = np.array([class_index.get(v, 0) for v in y])
    available = sorted(int(f) for f in np.unique(folds[mask]))
    if len(available) < 3:
        raise ValueError("spatial CNN validation needs at least three folds "
                         "(train, validation and test)")
    test_folds = available if cfg.spatial_cv else available[:1]
    if cfg.max_folds and cfg.max_folds > 0:
        test_folds = test_folds[:cfg.max_folds]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    predictions = np.zeros_like(y)
    probabilities = np.full((len(y), len(classes)), np.nan)
    evaluated = np.zeros(len(y), bool)
    fold_metrics, histories, fold_records = [], [], []

    for test_fold in test_folds:
        train, validation, test, validation_folds = _split_for_fold(
            folds, mask, test_fold, cfg.validation_folds, cfg.seed)
        if not train.any() or not validation.any() or not test.any():
            continue
        if np.unique(y[train]).size < 2:
            continue
        normalizer = training_normalizer(x_raw, train)
        x = transform_series(x_raw, normalizer)
        checkpoint = out / f"checkpoint_fold{test_fold}.pt"
        model, history, best_epoch, device = _train_one(
            x, y_index, train, validation, cfg, len(classes), checkpoint,
            cfg.seed + test_fold)
        fold_probabilities = _predict(model, x, test, device)
        probabilities[test] = fold_probabilities
        predictions[test] = classes[fold_probabilities.argmax(axis=1)]
        evaluated |= test

        metrics = classification_metrics(y[test], predictions[test],
                                         labels=classes)
        metrics.update({"fold": int(test_fold), "n_train": int(train.sum()),
                        "n_validation": int(validation.sum()),
                        "n_test": int(test.sum()),
                        "validation_folds": validation_folds,
                        "best_epoch": int(best_epoch),
                        "epochs_run": len(history)})
        fold_metrics.append(metrics)
        for row in history:
            histories.append({"fold": int(test_fold), **row})
        fold_records.append({"fold": int(test_fold),
                             "checkpoint": checkpoint.name,
                             "validation_folds": validation_folds,
                             "best_epoch": int(best_epoch)})
        if logger is not None:
            logger.info("  cnn fold %d: %d train / %d validation / %d test | "
                        "best epoch %d | macro F1 %.4f", test_fold,
                        int(train.sum()), int(validation.sum()),
                        int(test.sum()), best_epoch, metrics["f1_macro"])

    if not evaluated.any():
        raise ValueError("the CNN experiment produced no evaluable folds")

    summary = classification_metrics(y[evaluated], predictions[evaluated],
                                     labels=classes)
    summary["validation"] = "spatial_block_cv" if cfg.spatial_cv \
        else "single_spatial_split"
    summary["model"] = "cnn_1d"
    summary["fold_metrics"] = fold_metrics
    summary["fold_summary"] = aggregate_fold_metrics(fold_metrics)
    summary["n_evaluated"] = int(evaluated.sum())

    measures = prediction_confidence(probabilities[evaluated],
                                     cfg=uncertainty_cfg)
    summary["uncertainty"] = uncertainty_summary(
        probabilities[evaluated], truth=y[evaluated],
        predictions=predictions[evaluated], cfg=uncertainty_cfg)

    pd.DataFrame(histories).to_csv(out / "training_history.csv", index=False)
    pd.DataFrame({
        "truth": y[evaluated], "prediction": predictions[evaluated],
        "fold": folds[evaluated], "confidence": measures["confidence"],
        "margin": measures["margin"], "entropy": measures["entropy"],
        "uncertain": measures["uncertain"],
    }).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(probabilities[evaluated],
                 columns=[f"probability_{c}" for c in classes]
                 ).to_csv(out / "probabilities.csv", index=False)
    pd.DataFrame(summary["confusion_matrix"],
                 index=[f"true_{c}" for c in classes],
                 columns=[f"pred_{c}" for c in classes]
                 ).to_csv(out / "confusion_matrix.csv")
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    (out / "configuration.json").write_text(json.dumps(
        {"model": asdict(cfg), "uncertainty": asdict(uncertainty_cfg),
         "classes": classes.tolist(), "folds": fold_records}, indent=2))

    return {"metrics": summary, "predictions": predictions,
            "probabilities": probabilities, "evaluated": evaluated,
            "classes": classes, "history": pd.DataFrame(histories),
            "folds": fold_records}
