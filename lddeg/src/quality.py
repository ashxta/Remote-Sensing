"""Data-quality assessment for NDVI time-series stacks.

Prepares the pipeline for real satellite data, where missing observations
are the norm (cloud, shadow, SLC-off gaps, sensor outages) rather than the
exception.

Policy (Part 5): missing values are QUANTIFIED and FLAGGED. They are not
silently interpolated. Interpolation is available but opt-in, capped to a
maximum gap length, and always records how many values it touched.

Quality flags
-------------
0  OK                 passes all checks
1  INSUFFICIENT_OBS   fewer than min_valid_obs valid observations
2  TOO_MANY_MISSING   missing fraction exceeds max_missing_fraction
3  CONSTANT           variance below min_variance (no signal to analyse)
4  OUT_OF_RANGE       all valid values fall outside the physical NDVI range
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from .config import QualityConfig

FLAG_OK = 0
FLAG_INSUFFICIENT_OBS = 1
FLAG_TOO_MANY_MISSING = 2
FLAG_CONSTANT = 3
FLAG_OUT_OF_RANGE = 4

FLAG_NAMES = {
    FLAG_OK: "OK",
    FLAG_INSUFFICIENT_OBS: "INSUFFICIENT_OBS",
    FLAG_TOO_MANY_MISSING: "TOO_MANY_MISSING",
    FLAG_CONSTANT: "CONSTANT",
    FLAG_OUT_OF_RANGE: "OUT_OF_RANGE",
}


@dataclass
class QualityReport:
    """Per-pixel quality metadata for an (T, N) stack."""
    valid_count: np.ndarray      # (N,) int   number of finite observations
    missing_fraction: np.ndarray  # (N,) float fraction missing
    variance: np.ndarray         # (N,) float variance of valid values
    flag: np.ndarray             # (N,) int   one of FLAG_*
    n_obs: int                   # T

    @property
    def usable(self) -> np.ndarray:
        """Boolean mask of pixels safe to run statistics on."""
        return self.flag == FLAG_OK

    def summary(self) -> dict:
        total = int(self.flag.size)
        counts = {FLAG_NAMES[f]: int((self.flag == f).sum())
                  for f in sorted(FLAG_NAMES)}
        return {
            "n_pixels": total,
            "n_obs_per_pixel": int(self.n_obs),
            "n_usable": int(self.usable.sum()),
            "usable_fraction": float(self.usable.mean()) if total else 0.0,
            "mean_valid_observations": float(np.mean(self.valid_count))
            if total else 0.0,
            "mean_missing_fraction": float(np.mean(self.missing_fraction))
            if total else 0.0,
            "flag_counts": counts,
        }


def sanitize(stack: np.ndarray, cfg: QualityConfig) -> np.ndarray:
    """Convert physically impossible values to NaN so they count as missing.

    Real NDVI is bounded to [-1, 1]. Values outside that range indicate
    scaling or masking errors and must not enter a regression.
    """
    x = np.asarray(stack, dtype="float64").copy()
    if x.ndim != 2:
        raise ValueError(f"expected (T, N) stack, got shape {x.shape}")
    bad = (x < cfg.ndvi_min) | (x > cfg.ndvi_max)
    x[bad] = np.nan
    return x


def assess(stack: np.ndarray, cfg: QualityConfig,
           sanitize_range: bool = True) -> QualityReport:
    """Compute per-pixel quality metadata for an (T, N) stack."""
    x = sanitize(stack, cfg) if sanitize_range else \
        np.asarray(stack, dtype="float64")
    T, N = x.shape

    finite = np.isfinite(x)
    valid_count = finite.sum(axis=0).astype(int)
    missing_fraction = 1.0 - valid_count / float(T)

    with np.errstate(invalid="ignore", divide="ignore"), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        variance = np.nanvar(x, axis=0)
    variance = np.where(np.isfinite(variance), variance, 0.0)

    flag = np.full(N, FLAG_OK, dtype="int8")
    # Order matters: the most fundamental failure wins.
    flag = np.where(variance < cfg.min_variance, FLAG_CONSTANT, flag)
    flag = np.where(missing_fraction > cfg.max_missing_fraction,
                    FLAG_TOO_MANY_MISSING, flag)
    flag = np.where(valid_count < cfg.min_valid_obs,
                    FLAG_INSUFFICIENT_OBS, flag)
    flag = np.where(valid_count == 0, FLAG_OUT_OF_RANGE, flag)

    return QualityReport(valid_count=valid_count,
                         missing_fraction=missing_fraction,
                         variance=variance, flag=flag, n_obs=T)


def interpolate_gaps(stack: np.ndarray, cfg: QualityConfig) -> tuple:
    """Optional, opt-in temporal linear interpolation of SHORT gaps only.

    Returns (filled_stack, n_filled). Disabled unless
    cfg.allow_interpolation is True, in which case only interior gaps of at
    most cfg.max_interpolation_gap consecutive steps are filled. Leading and
    trailing gaps are never extrapolated.

    Rationale: interpolating long gaps manufactures data and biases trend
    tests toward significance. Short interior gaps are filled only because
    many estimators cannot accept NaN at all.
    """
    x = np.asarray(stack, dtype="float64").copy()
    if not cfg.allow_interpolation:
        return x, 0

    T, N = x.shape
    t = np.arange(T, dtype="float64")
    n_filled = 0
    for j in range(N):
        col = x[:, j]
        good = np.isfinite(col)
        if good.sum() < 2 or good.all():
            continue
        first, last = np.argmax(good), T - 1 - np.argmax(good[::-1])
        idx = np.where(~good)[0]
        idx = idx[(idx > first) & (idx < last)]   # interior only
        if idx.size == 0:
            continue
        # split interior gaps into runs, fill only short ones
        runs, start = [], idx[0]
        for a, b in zip(idx[:-1], idx[1:]):
            if b != a + 1:
                runs.append((start, a))
                start = b
        runs.append((start, idx[-1]))
        fillable = np.concatenate(
            [np.arange(a, b + 1) for a, b in runs
             if (b - a + 1) <= cfg.max_interpolation_gap]
        ) if any((b - a + 1) <= cfg.max_interpolation_gap
                 for a, b in runs) else np.array([], dtype=int)
        if fillable.size:
            col[fillable] = np.interp(t[fillable], t[good], col[good])
            x[:, j] = col
            n_filled += int(fillable.size)
    return x, n_filled
