"""Temporal compositing of irregular observations (M6 Parts 5, 8, 12, 13).

Satellites observe on their own schedule; the M1-M5 estimators require a
regular time axis. This module is the bridge, and it is where most of the
opportunities to quietly corrupt a long record live.

THE TEMPORAL UNIT
-----------------
The project asks whether multi-temporal analysis can separate persistent
degradation from cyclic or climate-driven dynamics. That question constrains
the temporal unit from both directions:

* Cyclicity is detected over periods of 4-12 time steps
  (`CyclicityConfig.min_period`/`max_period`). Rotational cultivation cycles
  in the literature run roughly 5-12 YEARS. An ANNUAL step therefore places
  the cycles of interest squarely inside the detection band.
* A monthly step would put those same cycles at 60-144 steps - far outside
  the configured band - and would flood the spectrum with the annual
  phenological cycle, whose power at period 12 sits inside the band and
  would dominate every pixel. Monthly data do not merely cost compute; they
  would make the cyclicity statistic measure phenology instead of land use.
* A seasonal step (4/yr) has the same defect in weaker form and quadruples
  the missing-data burden in a monsoon region.

So: ONE COMPOSITE PER YEAR, from a fixed seasonal window. The window is
configuration, not code. Both remain configurable
(`RealDataConfig.temporal_unit`) and the choice is recorded with every run,
because "we used annual composites" is a methodological claim that a reader
must be able to check.

THE COMPOSITING STATISTIC
-------------------------
`median` is the default and `max` is deliberately NOT. Maximum-value
compositing was designed for coarse AVHRR data where the cloud mask was
unreliable and the maximum was the best available cloud filter. With a
per-pixel QA mask already applied, the maximum instead selects the extreme
of the within-window distribution, which biases the composite upward, and -
because the bias depends on how many observations survived - biases it MORE
in cloudy years than in clear ones. Missingness varies systematically over a
36-year record (Landsat 7's SLC failure, the 2012-2013 gap between missions,
the improving revisit rate after 2021), so that bias becomes a spurious
TREND. The median is insensitive to the residual outliers a QA mask misses
and does not move with the number of observations.

WHAT IS NEVER DONE
------------------
No gap is filled here. A window with no usable observation produces NaN and
is counted. Interpolation is a separate, opt-in, gap-length-limited step
(`quality.interpolate_gaps`) that records every value it touches.
"""
from __future__ import annotations

import datetime as _dt
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = ["CompositeWindow", "CompositeResult", "CompositingError",
           "annual_windows", "seasonal_windows", "monthly_windows",
           "build_windows", "composite_observations", "COMPOSITE_STATISTICS",
           "describe_temporal_design", "as_date"]


class CompositingError(ValueError):
    """Raised when a compositing request cannot be satisfied honestly."""


#: Supported aggregation statistics. `max` is available because a reviewer
#: may ask for the comparison, not because it is recommended; see the module
#: docstring.
COMPOSITE_STATISTICS = ("median", "mean", "max", "min", "percentile")


@dataclass(frozen=True)
class CompositeWindow:
    """One output time step: a label and the date range feeding it."""
    label: str
    start: _dt.date
    end: _dt.date                    # inclusive
    year: Optional[int] = None

    def contains(self, when: _dt.date) -> bool:
        return self.start <= when <= self.end

    def describe(self) -> dict:
        return {"label": self.label, "start": self.start.isoformat(),
                "end": self.end.isoformat(), "year": self.year,
                "n_days": (self.end - self.start).days + 1}


def as_date(value) -> _dt.date:
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, str):
        return _dt.date.fromisoformat(value[:10])
    if isinstance(value, np.datetime64):
        return _dt.date.fromisoformat(str(value)[:10])
    raise CompositingError(f"cannot interpret {value!r} as a date")


# ---------------------------------------------------------------------------
# Window construction
# ---------------------------------------------------------------------------
def annual_windows(start_year: int, end_year: int, *,
                   window_start: str = "10-15",
                   window_end: str = "12-31") -> List[CompositeWindow]:
    """One window per year, over a fixed month-day season.

    A window that ends before it starts (e.g. 11-15 to 02-15) is treated as
    crossing the new year and closes in the FOLLOWING calendar year; the
    window is still labelled by the year it opened in, so the time axis stays
    one step per year.

    The default post-monsoon window (15 Oct - 31 Dec) is the repository's
    existing choice, retained deliberately: in the monsoon-dominated
    northeast it is both the least cloudy season and the period of peak
    standing biomass, so it maximises usable observations while sampling the
    same phenological stage every year. Sampling a DIFFERENT stage in
    different years is the classic way to manufacture a vegetation trend, and
    a fixed window is the defence against it.
    """
    if end_year < start_year:
        raise CompositingError(
            f"end_year {end_year} precedes start_year {start_year}")
    windows = []
    for year in range(int(start_year), int(end_year) + 1):
        start = _dt.date.fromisoformat(f"{year}-{window_start}")
        end = _dt.date.fromisoformat(f"{year}-{window_end}")
        if end < start:
            end = _dt.date.fromisoformat(f"{year + 1}-{window_end}")
        windows.append(CompositeWindow(label=str(year), start=start, end=end,
                                       year=year))
    return windows


def seasonal_windows(start_year: int, end_year: int, *,
                     seasons: Sequence[tuple] = None
                     ) -> List[CompositeWindow]:
    """Four windows per year. Available, but see the module docstring."""
    seasons = seasons or (("DJF", "01-01", "02-28"), ("MAM", "03-01", "05-31"),
                          ("JJA", "06-01", "08-31"), ("SON", "09-01", "11-30"))
    windows = []
    for year in range(int(start_year), int(end_year) + 1):
        for name, first, last in seasons:
            windows.append(CompositeWindow(
                label=f"{year}-{name}",
                start=_dt.date.fromisoformat(f"{year}-{first}"),
                end=_dt.date.fromisoformat(f"{year}-{last}"), year=year))
    return windows


def monthly_windows(start_year: int, end_year: int) -> List[CompositeWindow]:
    """Twelve windows per year. Available, but see the module docstring."""
    windows = []
    for year in range(int(start_year), int(end_year) + 1):
        for month in range(1, 13):
            start = _dt.date(year, month, 1)
            end = (_dt.date(year + (month == 12), month % 12 + 1, 1)
                   - _dt.timedelta(days=1))
            windows.append(CompositeWindow(label=f"{year}-{month:02d}",
                                           start=start, end=end, year=year))
    return windows


def build_windows(unit: str, start_year: int, end_year: int, *,
                  window_start: str = "10-15",
                  window_end: str = "12-31") -> List[CompositeWindow]:
    """Dispatch on the configured temporal unit."""
    key = str(unit).lower()
    if key == "annual":
        return annual_windows(start_year, end_year, window_start=window_start,
                              window_end=window_end)
    if key == "seasonal":
        return seasonal_windows(start_year, end_year)
    if key == "monthly":
        return monthly_windows(start_year, end_year)
    raise CompositingError(
        f"unknown temporal unit {unit!r}; expected 'annual', 'seasonal' or "
        "'monthly'. Annual is the project default; see compositing.py for "
        "why a finer unit breaks the cyclicity analysis.")


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------
@dataclass
class CompositeResult:
    """Composited cube plus the quality metadata that explains it."""
    values: np.ndarray               # (T, H, W) float64, NaN = no observation
    n_valid: np.ndarray              # (T, H, W) int, usable obs per window
    n_masked: np.ndarray             # (T, H, W) int, obs rejected by QA
    n_scenes: np.ndarray             # (T,) int, scenes intersecting the window
    windows: List[CompositeWindow]
    statistic: str
    metadata: Dict[str, Any]

    @property
    def times(self) -> List[str]:
        return [w.label for w in self.windows]

    def summary(self) -> dict:
        total = int(self.values.size)
        missing = int(np.isnan(self.values).sum())
        per_step = np.isnan(self.values).reshape(len(self.windows), -1
                                                 ).mean(axis=1)
        empty = [w.label for w, frac in zip(self.windows, per_step)
                 if frac >= 1.0]
        return {
            "statistic": self.statistic,
            "n_time_steps": len(self.windows),
            "temporal_coverage": [self.windows[0].label,
                                  self.windows[-1].label] if self.windows
            else [],
            "n_cells": total,
            "n_missing_cells": missing,
            "missing_fraction": missing / total if total else float("nan"),
            "mean_valid_observations_per_composite":
                float(np.mean(self.n_valid)),
            "mean_masked_observations_per_composite":
                float(np.mean(self.n_masked)),
            "scenes_per_window": {w.label: int(n) for w, n
                                  in zip(self.windows, self.n_scenes)},
            "windows_with_no_valid_observation_anywhere": empty,
            "missing_fraction_per_step": {w.label: float(f) for w, f
                                          in zip(self.windows, per_step)},
        }


def composite_observations(observations: Sequence[np.ndarray],
                           dates: Sequence[Any],
                           windows: Sequence[CompositeWindow], *,
                           statistic: str = "median",
                           percentile: float = 90.0,
                           masked_counts: Optional[Sequence[np.ndarray]] = None,
                           min_observations: int = 1,
                           metadata: Optional[dict] = None) -> CompositeResult:
    """Aggregate QA-masked observations onto a regular time axis.

    `observations` are index arrays (H, W) with NaN wherever the pixel was
    masked or never seen; `dates` gives each one's acquisition date.
    `masked_counts` optionally supplies, per observation, a 0/1 grid marking
    pixels that were rejected by the quality mask, so the report can
    distinguish "cloudy" from "never overflown".

    `min_observations` sets how many usable observations a window needs
    before it yields a value. It defaults to 1: a single clear observation IS
    an observation, and raising the bar discards real data. Where a stricter
    rule is wanted it belongs in configuration, and the composites it
    rejects appear as NaN and are counted, not hidden.
    """
    if statistic not in COMPOSITE_STATISTICS:
        raise CompositingError(
            f"unknown compositing statistic {statistic!r}; expected one of "
            f"{COMPOSITE_STATISTICS}")
    if len(observations) != len(dates):
        raise CompositingError(
            f"{len(observations)} observations but {len(dates)} dates")
    if not windows:
        raise CompositingError("no composite windows were requested")

    stack = [np.asarray(o, dtype="float64") for o in observations]
    shapes = {o.shape for o in stack}
    if len(shapes) > 1:
        raise CompositingError(
            f"observations have inconsistent grids: {sorted(shapes)}; a "
            "source must deliver every scene on one grid")
    grid = stack[0].shape if stack else (0, 0)
    if not stack:
        raise CompositingError("no observations were supplied")
    when = [as_date(d) for d in dates]

    n_windows = len(windows)
    values = np.full((n_windows, *grid), np.nan)
    n_valid = np.zeros((n_windows, *grid), dtype="int32")
    n_masked = np.zeros((n_windows, *grid), dtype="int32")
    n_scenes = np.zeros(n_windows, dtype="int32")

    for w, window in enumerate(windows):
        members = [i for i, day in enumerate(when) if window.contains(day)]
        n_scenes[w] = len(members)
        if not members:
            continue
        block = np.stack([stack[i] for i in members])           # (k, H, W)
        finite = np.isfinite(block)
        n_valid[w] = finite.sum(axis=0)
        if masked_counts is not None:
            n_masked[w] = np.stack(
                [np.asarray(masked_counts[i]) for i in members]
            ).astype("int32").sum(axis=0)
        else:
            n_masked[w] = len(members) - n_valid[w]

        # An all-NaN pixel-window is normal (persistent cloud, scene edge,
        # SLC-off gap) and numpy warns about it; the result is NaN, which is
        # exactly what should be recorded, so the warning is noise.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            if statistic == "median":
                aggregate = np.nanmedian(block, axis=0)
            elif statistic == "mean":
                aggregate = np.nanmean(block, axis=0)
            elif statistic == "max":
                aggregate = np.nanmax(block, axis=0)
            elif statistic == "min":
                aggregate = np.nanmin(block, axis=0)
            else:
                aggregate = np.nanpercentile(block, percentile, axis=0)
        values[w] = np.where(n_valid[w] >= max(int(min_observations), 1),
                             aggregate, np.nan)

    record = {
        "statistic": statistic,
        "percentile": percentile if statistic == "percentile" else None,
        "min_observations_per_composite": int(max(int(min_observations), 1)),
        "n_input_observations": len(stack),
        "windows": [w.describe() for w in windows],
        "gap_filling": ("none; a window with no usable observation is NaN "
                        "and is counted as missing"),
    }
    record.update(metadata or {})
    return CompositeResult(values=values, n_valid=n_valid, n_masked=n_masked,
                           n_scenes=n_scenes, windows=list(windows),
                           statistic=statistic, metadata=record)


def describe_temporal_design(cfg) -> dict:
    """The Part-5 record: what temporal design a run actually used."""
    return {
        "start_year": int(cfg.start_year),
        "end_year": int(cfg.end_year),
        "temporal_unit": str(cfg.temporal_unit),
        "composite_window": [str(cfg.window_start), str(cfg.window_end)],
        "n_time_steps": len(build_windows(
            cfg.temporal_unit, cfg.start_year, cfg.end_year,
            window_start=cfg.window_start, window_end=cfg.window_end)),
        "compositing_statistic": str(cfg.composite_statistic),
        "min_observations_per_composite": int(cfg.min_observations_per_composite),
        "missing_data_handling": (
            "A composite window with fewer than the required usable "
            "observations is NaN. NaN is the project's missing-data "
            "convention throughout; per-pixel gating in quality.assess "
            "decides which pixels have enough valid steps to analyse. Gaps "
            "are never zero-filled or forward-filled."),
        "interpolation": (
            "off by default (QualityConfig.allow_interpolation); when "
            "enabled, only interior gaps of at most "
            f"{getattr(cfg, 'max_interpolation_gap', 2)} steps are filled, "
            "and an interpolation mask records every value touched"),
        "rationale": (
            "Annual steps place multi-year cultivation cycles inside the "
            "configured 4-12 step cyclicity band; a monthly step would move "
            "them outside it and fill the band with the annual phenological "
            "cycle instead. A fixed seasonal window keeps the same "
            "phenological stage sampled every year."),
    }
