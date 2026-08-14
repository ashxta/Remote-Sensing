"""Figure rendering.

Figures are written to explicit paths supplied by the caller (the
experiment directory), never to a hard-coded global output folder, so that
every figure belongs unambiguously to one experiment.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm
import numpy as np

from .config import Config


def _grid(vec, mask, shape, fill=np.nan):
    out = np.full(shape, fill, "float64")
    out[mask] = np.asarray(vec).reshape(-1)
    return out


def save_continuous(vec, mask, shape, title, path, cmap="RdYlGn",
                    center=None, label=""):
    g = _grid(vec, mask, shape)
    fig, ax = plt.subplots(figsize=(9, 6))
    norm = None
    if center == 0 and np.isfinite(g).any():
        lo, hi = np.nanmin(g), np.nanmax(g)
        if lo < 0 < hi:
            norm = TwoSlopeNorm(vmin=lo, vcenter=0, vmax=hi)
    im = ax.imshow(g, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax, shrink=0.75, label=label)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_classmap(class_vec, mask, shape, path, cfg: Config | None = None,
                  title="Vegetation trajectory classes"):
    cfg = cfg or Config()
    g = _grid(class_vec, mask, shape, fill=0)
    g = np.nan_to_num(g, nan=0.0)
    ks = sorted(cfg.classes)
    cmap = ListedColormap(["#dddddd"] + [cfg.class_colors[k] for k in ks])
    norm = BoundaryNorm([-0.5] + [k + 0.5 for k in [0] + ks], len(ks) + 1)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    im = ax.imshow(g, cmap=cmap, norm=norm)
    cb = fig.colorbar(im, ax=ax, ticks=[0] + ks, shrink=0.8)
    cb.ax.set_yticklabels(["(no data)"] + [cfg.classes[k] for k in ks])
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def sdg_productivity(sen, p, mask, shape, path, cfg: Config | None = None):
    """Three-class productivity trajectory in the style of the UNCCD
    SDG 15.3.1 sub-indicator: significant decline / stable / significant
    increase.

    This is a map IN THE STYLE OF the productivity trajectory
    sub-indicator. It is NOT an official SDG 15.3.1 assessment and must not
    be reported as one. A complete 15.3.1 assessment additionally requires
    land-cover change and soil organic carbon, uses the UNCCD's own
    baseline and reporting periods, and is computed on real observations.
    None of those conditions is met here.
    """
    cfg = cfg or Config()
    a = cfg.trend.alpha
    valid = np.isfinite(p) & np.isfinite(sen)
    cls = np.where(valid & (p < a) & (sen < 0), 1,
                   np.where(valid & (p < a) & (sen > 0), 3,
                            np.where(valid, 2, 0)))
    g = _grid(cls, mask, shape, fill=0)
    cmap = ListedColormap(["#dddddd", "#c0242b", "#e8e4cf", "#1a7a3a"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], 4)
    fig, ax = plt.subplots(figsize=(9.5, 6))
    im = ax.imshow(g, cmap=cmap, norm=norm)
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], shrink=0.8)
    cb.ax.set_yticklabels(["(no data)", "Declining", "Stable", "Increasing"])
    ax.set_title("Land productivity trajectory, in the style of the SDG "
                 "15.3.1 productivity sub-indicator\n(not an official SDG "
                 "15.3.1 assessment)", fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    n = max(int(valid.sum()), 1)
    return {"pct_declining": float(100 * (cls == 1).sum() / n),
            "pct_stable": float(100 * (cls == 2).sum() / n),
            "pct_increasing": float(100 * (cls == 3).sum() / n),
            "n_assessed": int(valid.sum()),
            "caveat": ("Productivity trajectory in the style of the SDG "
                       "15.3.1 sub-indicator. Not an official SDG 15.3.1 "
                       "assessment: land-cover change and soil organic "
                       "carbon are out of scope and the input is synthetic.")}


def save_confusion(cm, path, title, cfg: Config | None = None):
    cfg = cfg or Config()
    ks = sorted(cfg.classes)
    fig, ax = plt.subplots(figsize=(7.4, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(ks)))
    ax.set_yticks(range(len(ks)))
    ax.set_xticklabels([cfg.classes[k] for k in ks], rotation=35, ha="right")
    ax.set_yticklabels([cfg.classes[k] for k in ks])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def example_trajectories(ndvi, truth_vec, path, cfg: Config | None = None):
    cfg = cfg or Config()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    rng = np.random.default_rng(cfg.seed)
    steps = np.arange(ndvi.shape[0])
    labels = cfg.years if len(cfg.years) == ndvi.shape[0] else steps
    for k in sorted(cfg.classes):
        idx = np.where(truth_vec == k)[0]
        if idx.size == 0:
            continue
        ax.plot(labels, ndvi[:, rng.choice(idx)], label=cfg.classes[k],
                color=cfg.class_colors[k], lw=2)
    ax.set_xlabel("Time step")
    ax.set_ylabel("NDVI")
    ax.set_title("Example pixel trajectories by class")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
