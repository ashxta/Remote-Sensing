"""Post-disturbance recovery analysis (M1 Part 4).

Given an NDVI stack and a per-pixel disturbance index (from
`timeseries.best_breakpoint`, or from LandTrendr in the real-data
workflow), derive the standard recovery descriptors used in the
disturbance-recovery literature.

Metrics per pixel
-----------------
disturbance_index   time index of the detected break (-1 if none)
pre_level           mean NDVI over `pre_window` steps before the break
trough_value        minimum NDVI after the break
trough_index        time index of that minimum
disturbance_mag     pre_level - trough_value  (positive = a real drop)
recovery_magnitude  last_value - trough_value
recovery_fraction   recovery_magnitude / disturbance_mag, clipped >= 0
recovery_duration   steps from trough to first crossing of the recovery
                    threshold; NaN if never reached within the record
recovery_slope      OLS slope of NDVI from the trough to the end
recovery_status     categorical, see STATUS_*

Status codes
------------
0 NO_DISTURBANCE     no admissible break, or drop below the minimum
1 RECOVERED          crossed recovery_threshold * pre_level
2 RECOVERING         positive recovery slope, threshold not yet reached
3 NOT_RECOVERING     no meaningful upward movement after the trough
4 INSUFFICIENT_DATA  too few post-trough observations to judge

All thresholds come from `config.RecoveryConfig` and none are hard-coded.
"""
from __future__ import annotations

import numpy as np

from .config import RecoveryConfig

STATUS_NO_DISTURBANCE = 0
STATUS_RECOVERED = 1
STATUS_RECOVERING = 2
STATUS_NOT_RECOVERING = 3
STATUS_INSUFFICIENT_DATA = 4

STATUS_NAMES = {
    STATUS_NO_DISTURBANCE: "NO_DISTURBANCE",
    STATUS_RECOVERED: "RECOVERED",
    STATUS_RECOVERING: "RECOVERING",
    STATUS_NOT_RECOVERING: "NOT_RECOVERING",
    STATUS_INSUFFICIENT_DATA: "INSUFFICIENT_DATA",
}


def _ols_slope(t: np.ndarray, y: np.ndarray) -> float:
    good = np.isfinite(y)
    if good.sum() < 2:
        return np.nan
    tt, yy = t[good], y[good]
    tm = tt - tt.mean()
    denom = float((tm ** 2).sum())
    if denom <= 0:
        return np.nan
    return float((tm * (yy - yy.mean())).sum() / denom)


def analyze(ndvi: np.ndarray, break_index: np.ndarray,
            cfg: RecoveryConfig) -> dict:
    """Compute recovery descriptors for an (T, N) stack.

    `break_index` is (N,) int; -1 means no disturbance detected.
    Returns a dict of (N,) arrays.
    """
    x = np.asarray(ndvi, dtype="float64")
    if x.ndim == 1:
        x = x[:, None]
    T, N = x.shape
    bidx = np.asarray(break_index).astype(int).reshape(-1)
    if bidx.size != N:
        raise ValueError(f"break_index size {bidx.size} != n_pixels {N}")

    t = np.arange(T, dtype="float64")
    out = {
        "disturbance_index": bidx.astype("float64").copy(),
        "pre_level": np.full(N, np.nan),
        "trough_value": np.full(N, np.nan),
        "trough_index": np.full(N, np.nan),
        "disturbance_magnitude": np.full(N, np.nan),
        "recovery_magnitude": np.full(N, np.nan),
        "recovery_fraction": np.full(N, np.nan),
        "recovery_duration": np.full(N, np.nan),
        "recovery_slope": np.full(N, np.nan),
        "recovery_status": np.full(N, STATUS_NO_DISTURBANCE, dtype="int8"),
    }

    for j in range(N):
        k = bidx[j]
        if k < 0 or k >= T:
            out["disturbance_index"][j] = np.nan
            continue
        col = x[:, j]

        lo = max(0, k - cfg.pre_window)
        pre = col[lo:k]
        pre = pre[np.isfinite(pre)]
        if pre.size == 0:
            out["recovery_status"][j] = STATUS_INSUFFICIENT_DATA
            continue
        pre_level = float(pre.mean())
        out["pre_level"][j] = pre_level

        post = col[k:]
        post_ok = np.isfinite(post)
        if post_ok.sum() < 2:
            out["recovery_status"][j] = STATUS_INSUFFICIENT_DATA
            continue

        rel_trough = int(np.nanargmin(post))
        trough_idx = k + rel_trough
        trough_val = float(post[rel_trough])
        out["trough_index"][j] = trough_idx
        out["trough_value"][j] = trough_val

        dist_mag = pre_level - trough_val
        out["disturbance_magnitude"][j] = dist_mag
        if dist_mag < cfg.min_disturbance_magnitude:
            out["recovery_status"][j] = STATUS_NO_DISTURBANCE
            continue

        tail = col[trough_idx:]
        tail_ok = np.isfinite(tail)
        if tail_ok.sum() < cfg.min_post_obs:
            out["recovery_status"][j] = STATUS_INSUFFICIENT_DATA
            continue

        last_val = float(tail[tail_ok][-1])
        rec_mag = last_val - trough_val
        out["recovery_magnitude"][j] = rec_mag
        out["recovery_fraction"][j] = max(rec_mag / dist_mag, 0.0) \
            if dist_mag > 0 else np.nan

        slope = _ols_slope(t[trough_idx:], tail)
        out["recovery_slope"][j] = slope

        target = cfg.recovery_threshold * pre_level
        reached = np.flatnonzero(tail_ok & (tail >= target))
        if reached.size:
            out["recovery_duration"][j] = float(reached[0])
            out["recovery_status"][j] = STATUS_RECOVERED
        elif np.isfinite(slope) and slope > 0:
            out["recovery_status"][j] = STATUS_RECOVERING
        else:
            out["recovery_status"][j] = STATUS_NOT_RECOVERING

    return out


def summary(rec: dict) -> dict:
    """Aggregate recovery status counts for reporting."""
    st = rec["recovery_status"]
    total = int(st.size)
    counts = {STATUS_NAMES[s]: int((st == s).sum())
              for s in sorted(STATUS_NAMES)}
    disturbed = st != STATUS_NO_DISTURBANCE
    with np.errstate(invalid="ignore"):
        med_dur = float(np.nanmedian(rec["recovery_duration"])) \
            if np.isfinite(rec["recovery_duration"]).any() else float("nan")
        med_frac = float(np.nanmedian(rec["recovery_fraction"])) \
            if np.isfinite(rec["recovery_fraction"]).any() else float("nan")
    return {
        "n_pixels": total,
        "status_counts": counts,
        "n_disturbed": int(disturbed.sum()),
        "median_recovery_duration": med_dur,
        "median_recovery_fraction": med_frac,
    }
