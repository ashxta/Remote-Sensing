"""Research-quality figures for the M2 experiments (M2 Part 8).

Every figure is written to an explicit path supplied by the caller (the
experiment directory) and is reproducible: the same inputs and
configuration produce the same figure, and no RNG is used except where a
caller passes a seeded generator's selection in.

Figures produced from the bundled synthetic dataset are labelled
"DEVELOPMENT / SYNTHETIC" in their titles. That label is not decoration: it
prevents a development figure being mistaken for a measurement of a real
landscape. Nothing in this module attaches geographic interpretation to
synthetic output.
"""
from __future__ import annotations

from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

from .config import Config
from .trajectory import TRAJECTORY_CODES

__all__ = ["dev_title", "plot_temporal_diagnostics", "plot_spectrum",
           "plot_confusion_matrix", "plot_metric_comparison",
           "plot_ablation_comparison", "plot_feature_importance",
           "plot_spatial_folds", "plot_trajectory_map",
           "plot_probability_map", "plot_class_distribution",
           "plot_sensitivity", "plot_learning_curves", "plot_quality_map",
           "plot_uncertainty_map"]

DEV_PREFIX = "DEVELOPMENT / SYNTHETIC"


def dev_title(title: str, synthetic: bool = True) -> str:
    """Prefix a title with the development-data label when appropriate."""
    return f"{DEV_PREFIX}: {title}" if synthetic else title


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _grid(values, mask, shape, fill=np.nan):
    out = np.full(shape, fill, dtype="float64")
    out[np.asarray(mask, bool)] = np.asarray(values, dtype="float64").reshape(-1)
    return out


def _sen_line(series, slope):
    """Median-anchored Theil-Sen line, the standard way to draw one."""
    t = np.arange(len(series), dtype="float64")
    good = np.isfinite(series)
    if not good.any() or not np.isfinite(slope):
        return None
    intercept = np.median(series[good] - slope * t[good])
    return intercept + slope * t


# ------------------------------------------------------------------ temporal
def plot_temporal_diagnostics(ndvi_series, rain_series, path, *,
                              cfg: Config | None = None, title="",
                              sen_slope=None, break_index=None,
                              trough_index=None, recovery_slope=None,
                              synthetic: bool = True):
    """Four-panel temporal diagnostic for one pixel.

    NDVI with its Theil-Sen trend line, breakpoint and recovery segment;
    rainfall; the RESTREND residual series with its residual trend; and the
    power spectrum with the configured periodicity band shaded.
    """
    cfg = cfg or Config()
    ndvi = np.asarray(ndvi_series, dtype="float64").reshape(-1)
    rain = np.asarray(rain_series, dtype="float64").reshape(-1)
    t = np.arange(ndvi.size, dtype="float64")

    fig, axes = plt.subplots(4, 1, figsize=(9.5, 11), sharex=False)
    ax = axes[0]
    ax.plot(t, ndvi, "o-", color="#1a7a3a", ms=3, lw=1.4, label="NDVI")
    if sen_slope is not None:
        line = _sen_line(ndvi, float(sen_slope))
        if line is not None:
            ax.plot(t, line, "--", color="#333333", lw=1.6,
                    label=f"Theil-Sen slope {float(sen_slope):+.4f}/step")
    if break_index is not None and np.isfinite(break_index) \
            and break_index >= 0:
        ax.axvline(float(break_index), color="#c0242b", lw=1.6,
                   label=f"breakpoint (step {int(break_index)})")
    if trough_index is not None and np.isfinite(trough_index):
        k = int(trough_index)
        ax.plot([k], [ndvi[k]], "v", color="#c0242b", ms=9, label="trough")
        if recovery_slope is not None and np.isfinite(recovery_slope):
            tail = t[k:]
            ax.plot(tail, ndvi[k] + float(recovery_slope) * (tail - tail[0]),
                    ":", color="#3a86c8", lw=2,
                    label=f"recovery slope {float(recovery_slope):+.4f}/step")
    ax.set_ylabel("NDVI")
    ax.set_title(dev_title(title or "Pixel temporal diagnostics", synthetic))
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t, rain, "o-", color="#3a86c8", ms=3, lw=1.2)
    ax.set_ylabel("Rainfall")
    ax.set_title("Rainfall trajectory (climate driver)", fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[2]
    both = np.isfinite(ndvi) & np.isfinite(rain)
    if both.sum() >= 4:
        design = np.c_[rain[both], t[both], np.ones(both.sum())]
        coeff, *_ = np.linalg.lstsq(design, ndvi[both], rcond=None)
        residual = ndvi[both] - (coeff[0] * rain[both] + coeff[2])
        ax.plot(t[both], residual, "o-", color="#8b5e3c", ms=3, lw=1.2,
                label="NDVI with the rainfall effect removed")
        ax.plot(t[both], coeff[1] * t[both] + np.median(
            residual - coeff[1] * t[both]), "--", color="#333333", lw=1.6,
            label=f"residual trend {coeff[1]:+.4f}/step")
        ax.legend(fontsize=8, loc="best")
    else:
        ax.text(0.5, 0.5, "too few paired observations for RESTREND",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel("Residual NDVI")
    ax.set_title("RESTREND residual series (climate-adjusted)", fontsize=10)
    ax.grid(alpha=0.3)

    plot_spectrum(ndvi, None, cfg=cfg, ax=axes[3], synthetic=synthetic)
    axes[3].set_xlabel("Period (time steps)")
    return _save(fig, path)


def plot_spectrum(series, path, *, cfg: Config | None = None, ax=None,
                  title="Power spectrum and periodicity band",
                  synthetic: bool = True):
    """Detrended power spectrum with the configured period band shaded."""
    cfg = cfg or Config()
    x = np.asarray(series, dtype="float64").reshape(-1)
    filled = np.where(np.isfinite(x), x, np.nanmean(x))
    t = np.arange(filled.size, dtype="float64")
    tm = t - t.mean()
    detrended = filled - filled.mean()
    detrended = detrended - (tm * detrended).sum() / (tm ** 2).sum() * tm

    power = np.abs(np.fft.rfft(detrended)) ** 2
    freqs = np.fft.rfftfreq(filled.size, d=1.0)
    keep = freqs > 0
    periods = 1.0 / freqs[keep]

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(periods, power[keep], "-o", color="#e07b39", ms=3)
    ax.axvspan(cfg.cyclicity.min_period, cfg.cyclicity.max_period,
               color="#e07b39", alpha=0.15,
               label=f"band {cfg.cyclicity.min_period:g}-"
                     f"{cfg.cyclicity.max_period:g} steps")
    ax.set_xlabel("Period (time steps)")
    ax.set_ylabel("Power")
    ax.set_title(dev_title(title, synthetic) if own else title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, path) if own else ax


# --------------------------------------------------------------------- model
def plot_confusion_matrix(matrix, labels, path, *, title="Confusion matrix",
                          class_names=None, normalize=False,
                          synthetic: bool = True):
    """Confusion matrix with counts (or row-normalised shares) annotated."""
    cm = np.asarray(matrix, dtype="float64")
    shown = cm
    if normalize:
        with np.errstate(invalid="ignore", divide="ignore"):
            shown = np.where(cm.sum(axis=1, keepdims=True) > 0,
                             cm / cm.sum(axis=1, keepdims=True), 0.0)
    names = [str(c) for c in (class_names if class_names is not None
                              else labels)]
    fig, ax = plt.subplots(figsize=(1.35 * len(names) + 4.2,
                                    1.05 * len(names) + 3.4))
    im = ax.imshow(shown, cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.8,
                 label="share of true class" if normalize else "samples")
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticklabels(names)
    threshold = shown.max() / 2 if shown.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{shown[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9,
                    color="white" if shown[i, j] > threshold else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(dev_title(title, synthetic))
    return _save(fig, path)


def plot_metric_comparison(frame: pd.DataFrame, path, *,
                           label_column="method",
                           metrics: Sequence[str] = ("accuracy", "f1_macro",
                                                     "f1_weighted"),
                           title="Model performance comparison",
                           error_column=None, synthetic: bool = True):
    """Grouped bars comparing several methods over several metrics."""
    frame = frame.reset_index(drop=True)
    metrics = [m for m in metrics if m in frame.columns]
    positions = np.arange(len(frame))
    width = 0.8 / max(len(metrics), 1)
    fig, ax = plt.subplots(figsize=(1.9 * len(frame) + 4.0, 4.8))
    for i, metric in enumerate(metrics):
        err = frame[error_column] if (error_column
                                      and error_column in frame) else None
        ax.bar(positions + i * width, frame[metric], width, label=metric,
               yerr=err if metric == metrics[0] else None, capsize=3)
    ax.set_xticks(positions + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(frame[label_column].astype(str), rotation=20,
                       ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(dev_title(title, synthetic))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def plot_ablation_comparison(table: pd.DataFrame, path, *,
                             title="Ablation study: feature contribution",
                             synthetic: bool = True):
    """Macro F1 per ablation cell with the across-fold standard deviation."""
    fig, ax = plt.subplots(figsize=(max(7.5, 1.5 * len(table)), 5.0))
    positions = np.arange(len(table))
    yerr = table["fold_f1_macro_std"] if "fold_f1_macro_std" in table else None
    ax.bar(positions, table["f1_macro"], 0.6, color="#3a86c8",
           yerr=yerr, capsize=4)
    for x, (value, n) in enumerate(zip(table["f1_macro"],
                                       table["n_features"])):
        ax.text(x, value + 0.015, f"{value:.3f}\n({n} feat.)", ha="center",
                fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(table["feature_set"].astype(str), rotation=20,
                       ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("macro F1 (spatial block CV)")
    ax.set_title(dev_title(title, synthetic))
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def plot_feature_importance(importance: pd.Series, path, *, top_n=20,
                            title="Random Forest feature importance",
                            synthetic: bool = True):
    """Impurity importance: predictive association, not causal influence."""
    top = importance.sort_values(ascending=False).head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 0.32 * len(top) + 2.2))
    ax.barh(np.arange(len(top)), top.to_numpy(), color="#1a7a3a")
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels(top.index)
    ax.set_xlabel("mean impurity decrease (association, not causation)")
    ax.set_title(dev_title(title, synthetic))
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)


def plot_sensitivity(table: pd.DataFrame, path, *,
                     metric="trajectory_degrading_fraction",
                     title="Sensitivity of conclusions to parameters",
                     synthetic: bool = True):
    """One line per swept parameter, showing how a reported quantity moves."""
    sweeps = [p for p in table["parameter"].unique() if p != "(baseline)"]
    if not sweeps or metric not in table.columns:
        return None
    columns = min(3, len(sweeps))
    rows = int(np.ceil(len(sweeps) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns,
                                                     3.1 * rows),
                             squeeze=False)
    baseline = table[table["parameter"] == "(baseline)"][metric]
    for ax, parameter in zip(axes.ravel(), sweeps):
        subset = table[table["parameter"] == parameter]
        ax.plot(subset["value"].astype(float), subset[metric], "o-",
                color="#3a86c8")
        if len(baseline):
            ax.axhline(float(baseline.iloc[0]), ls="--", color="#c0242b",
                       lw=1.2, label="default configuration")
            ax.legend(fontsize=7)
        ax.set_title(parameter, fontsize=9)
        ax.set_xlabel("value")
        ax.set_ylabel(metric.replace("_", " "), fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(sweeps):]:
        ax.set_axis_off()
    fig.suptitle(dev_title(title, synthetic))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_class_distribution(counts: dict, path, *,
                            title="Trajectory class distribution",
                            synthetic: bool = True):
    """Bar chart of analytical trajectory classes (not land-cover classes)."""
    names = list(counts)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.5 * len(names)), 4.2))
    ax.bar(np.arange(len(names)), [counts[n] for n in names], 0.6,
           color="#8b5e3c")
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("pixels")
    ax.set_title(dev_title(title, synthetic))
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def plot_learning_curves(history: pd.DataFrame, path, *,
                         title="1D CNN training and validation loss",
                         synthetic: bool = True):
    """Per-fold training/validation loss, with the early-stopping point.

    Divergence between the two curves is the honest picture of overfitting;
    the marker shows the epoch whose checkpoint was actually restored.
    """
    if history is None or len(history) == 0:
        return None
    history = pd.DataFrame(history)
    folds = sorted(history["fold"].unique()) if "fold" in history \
        else [None]
    columns = min(3, len(folds))
    rows = int(np.ceil(len(folds) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns,
                                                     3.4 * rows),
                             squeeze=False, sharey=True)
    for ax, fold in zip(axes.ravel(), folds):
        subset = history if fold is None else history[history["fold"] == fold]
        ax.plot(subset["epoch"], subset["train_loss"], "-", color="#3a86c8",
                lw=1.6, label="training")
        ax.plot(subset["epoch"], subset["validation_loss"], "-",
                color="#c0242b", lw=1.6, label="validation")
        if len(subset):
            best = subset.loc[subset["validation_loss"].idxmin()]
            ax.axvline(float(best["epoch"]), ls="--", color="#333333", lw=1.1)
            ax.plot([float(best["epoch"])], [float(best["validation_loss"])],
                    "o", color="#333333", ms=5,
                    label=f"restored epoch {int(best['epoch'])}")
        ax.set_title("all folds" if fold is None else f"test fold {int(fold)}",
                     fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy loss")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(folds):]:
        ax.set_axis_off()
    fig.suptitle(dev_title(title, synthetic))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ------------------------------------------------------------------- spatial
def plot_quality_map(flags, shape, path, *, flag_names=None,
                     title="Per-pixel data-quality flags",
                     synthetic: bool = True):
    """Map of the quality gate: which pixels were excluded, and why.

    Missingness is a covariate, not a nuisance: where data are excluded
    constrains where any conclusion can be drawn, so the exclusion pattern
    is reported as a first-class output.
    """
    from .quality import FLAG_NAMES

    names = flag_names or FLAG_NAMES
    grid = np.asarray(flags, dtype="float64").reshape(shape)
    codes = sorted(names)
    colors = ["#c7e9c0", "#fdd0a2", "#fdae6b", "#9ecae1", "#c0242b",
              "#9e9e9e"][:len(codes)]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    image = ax.imshow(grid, cmap=ListedColormap(colors),
                      norm=BoundaryNorm([c - 0.5 for c in codes]
                                        + [codes[-1] + 0.5], len(codes)),
                      interpolation="nearest")
    cb = fig.colorbar(image, ax=ax, ticks=codes, shrink=0.8)
    cb.ax.set_yticklabels([names[c] for c in codes])
    ax.set_title(dev_title(title, synthetic))
    ax.set_axis_off()
    return _save(fig, path)


def plot_uncertainty_map(values, mask, shape, path, *,
                         title="Flagged low-confidence predictions",
                         synthetic: bool = True):
    """Binary map of predictions the model itself found marginal."""
    grid = _grid(np.asarray(values, dtype="float64"), mask, shape)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    image = ax.imshow(grid, cmap=ListedColormap(["#c7e9c0", "#c0242b"]),
                      norm=BoundaryNorm([-0.5, 0.5, 1.5], 2),
                      interpolation="nearest")
    cb = fig.colorbar(image, ax=ax, ticks=[0, 1], shrink=0.8)
    cb.ax.set_yticklabels(["confident", "flagged uncertain"])
    ax.set_title(dev_title(title + "\n(model confidence, not certainty)",
                           synthetic), fontsize=11)
    ax.set_axis_off()
    return _save(fig, path)


def plot_spatial_folds(fold_grid, path, *, title="Spatial cross-validation "
                       "folds", synthetic: bool = True):
    """Map of the block-to-fold assignment used for spatial validation."""
    grid = np.asarray(fold_grid, dtype="float64")
    folds = np.unique(grid[np.isfinite(grid)])
    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(grid, cmap="tab10", interpolation="nearest")
    cb = fig.colorbar(image, ax=ax, shrink=0.8, ticks=folds)
    cb.set_label("fold")
    ax.set_title(dev_title(title, synthetic))
    ax.set_axis_off()
    return _save(fig, path)


def plot_trajectory_map(labels, mask, shape, path, *,
                        title="Analytical vegetation-trajectory classes",
                        synthetic: bool = True):
    """Map of trajectory classes.

    These are analytical signal categories, stated in the title so the
    figure cannot be read as a verified land-cover map.
    """
    codes = np.array([TRAJECTORY_CODES.get(v, TRAJECTORY_CODES[
        "Uncertain / Other"]) for v in np.asarray(labels, dtype=object)],
        dtype="float64")
    grid = _grid(codes, mask, shape, fill=0.0)
    names = ["(no data)"] + list(TRAJECTORY_CODES)
    colors = ["#dddddd", "#8fbf7f", "#c0242b", "#3a86c8", "#e07b39",
              "#9e9e9e"]
    cmap = ListedColormap(colors[:len(names)])
    bounds = [-0.5] + [i + 0.5 for i in range(len(names))]
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(grid, cmap=cmap, norm=BoundaryNorm(bounds, len(names)),
                      interpolation="nearest")
    cb = fig.colorbar(image, ax=ax, ticks=list(range(len(names))), shrink=0.8)
    cb.ax.set_yticklabels(names)
    ax.set_title(dev_title(title + "\n(analytical categories, not verified "
                           "land cover)", synthetic), fontsize=11)
    ax.set_axis_off()
    return _save(fig, path)


def plot_probability_map(probabilities, mask, shape, path, *,
                         title="Model confidence (maximum class probability)",
                         synthetic: bool = True):
    """Map of the model's maximum class probability.

    This is a model confidence estimate conditional on the training data and
    feature set, NOT a probability that the ground is degraded.
    """
    values = np.asarray(probabilities, dtype="float64")
    if values.ndim == 2:
        with np.errstate(invalid="ignore"):
            values = np.nanmax(values, axis=1)
    grid = _grid(values, mask, shape)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    image = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1,
                      interpolation="nearest")
    fig.colorbar(image, ax=ax, shrink=0.8, label="model confidence estimate")
    ax.set_title(dev_title(title + "\n(model confidence, not certainty)",
                           synthetic), fontsize=11)
    ax.set_axis_off()
    return _save(fig, path)
