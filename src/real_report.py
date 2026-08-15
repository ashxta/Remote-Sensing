"""Real-data quality report (M6 Part 14).

Written BEFORE the research pipeline runs, and read before its results are
believed. On synthetic data the quality report is a formality - the
generator produced clean, complete cubes. On real data it is the single most
informative artefact in the run, because missingness is not random:

* Landsat 7's Scan Line Corrector failed in May 2003, removing ~22% of every
  subsequent ETM+ scene in a systematic geometric pattern;
* Landsat 5 ended in 2012 and Landsat 8 began in 2013, so the middle of the
  record is thinner than either end;
* cloud cover in a monsoon region is seasonal and interannually variable.

Each of those makes the number of valid observations vary SYSTEMATICALLY
over time and space. A trend test has more power where it has more
observations, so a map of "significant decline" partly maps observation
density. That is a confound, it is measurable, and this report measures it -
which is why `missingness_vs_time` is computed and plotted rather than
summarised into a single percentage.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

__all__ = ["build_quality_report", "write_quality_report",
           "plot_quality_report", "MISSINGNESS_CAVEAT"]

MISSINGNESS_CAVEAT = (
    "Valid-observation counts vary systematically across the record "
    "(SLC-off gaps after 2003, the 2012-2013 mission gap, seasonal cloud). "
    "Statistical power therefore varies in space and time, so any map of "
    "trend significance partly reflects observation density. Report the "
    "valid-observation map alongside every significance map.")


def _describe(values: np.ndarray, name: str) -> Dict[str, Any]:
    finite = values[np.isfinite(values)]
    total = int(values.size)
    missing = total - int(finite.size)
    if finite.size == 0:
        return {"variable": name, "n_cells": total, "n_missing": missing,
                "missing_fraction": 1.0, "min": None, "max": None,
                "mean": None, "median": None, "std": None,
                "p05": None, "p95": None}
    return {
        "variable": name,
        "n_cells": total,
        "n_missing": missing,
        "missing_fraction": missing / total if total else float("nan"),
        "min": float(np.min(finite)), "max": float(np.max(finite)),
        "mean": float(np.mean(finite)), "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "p05": float(np.percentile(finite, 5)),
        "p95": float(np.percentile(finite, 95)),
    }


def build_quality_report(dataset, cfg, *, valid_counts: Optional[np.ndarray] = None,
                         inside: Optional[np.ndarray] = None
                         ) -> Dict[str, Any]:
    """Assemble the spatial, temporal, vegetation and rainfall sections.

    `inside` is the boundary mask. With an irregular study area, roughly half
    the pixels of the enclosing raster lie OUTSIDE the polygon and are NaN by
    construction. Averaging over the whole grid would report that as
    "missing data" - on this study area it turns a genuine ~4% missingness
    into a meaningless 52% - and would make the record look far worse than it
    is. Every statistic below is therefore computed over the pixels inside
    the boundary, and the outside count is reported separately as geometry
    rather than as a data-quality problem.
    """
    ndvi, rain = dataset.ndvi, dataset.rain
    georef = dataset.georef
    times = [str(t) for t in dataset.times] or [str(i) for i
                                                in range(dataset.n_time)]

    if inside is None:
        inside = np.ones(dataset.shape, dtype=bool)
    inside = np.asarray(inside, dtype=bool)
    flat_inside = inside.reshape(-1)
    n_inside = int(flat_inside.sum())

    ndvi_in = ndvi.reshape(dataset.n_time, -1)[:, flat_inside]
    rain_in = rain.reshape(dataset.n_time, -1)[:, flat_inside]

    per_step_missing = np.isnan(ndvi_in).mean(axis=1)
    per_step_rain_missing = np.isnan(rain_in).mean(axis=1)
    per_pixel_valid = np.isfinite(ndvi_in).sum(axis=0)

    from .study_area import pixel_area_km2
    areas = pixel_area_km2(georef)
    observed_anywhere = np.isfinite(ndvi).any(axis=0) & inside

    # The dataset declares its own provenance; the report repeats it rather
    # than assuming that anything reaching this module is real.
    synthetic = bool(dataset.metadata.get("synthetic", False))
    report: Dict[str, Any] = {
        "data_status": ("SYNTHETIC FIXTURE data (development/testing)"
                        if synthetic else "REAL remote-sensing observations"),
        "synthetic": synthetic,
        "notice": dataset.metadata.get("notice", ""),
        "spatial": {
            "study_area": dataset.metadata.get("study_area", {}),
            "crs": str(georef.crs),
            "transform": list(georef.transform)[:6],
            "raster_dimensions": {"height": georef.height,
                                  "width": georef.width},
            "resolution": {"x": georef.resolution[0],
                           "y": georef.resolution[1]},
            "extent": georef.to_dict()["bounds"],
            "n_pixels_in_raster": int(dataset.n_pixels),
            "n_pixels_inside_boundary": n_inside,
            "n_pixels_outside_boundary": int(dataset.n_pixels - n_inside),
            "outside_boundary_note": (
                "Pixels outside the study-area polygon are NaN by "
                "construction. They are geometry, not missing data, and are "
                "excluded from every statistic in this report."),
            "study_area_km2": float(areas[inside].sum()),
            "analysed_area_km2": float(areas[observed_anywhere].sum()),
            "pixel_area_km2": {"min": float(areas.min()),
                               "max": float(areas.max())},
        },
        "temporal": {
            "start": times[0], "end": times[-1],
            "n_time_steps": int(dataset.n_time),
            "labels": times,
            "temporal_unit": getattr(cfg.real_data, "temporal_unit", None),
            "composite_window": [getattr(cfg.real_data, "window_start", None),
                                 getattr(cfg.real_data, "window_end", None)],
            "compositing_statistic": getattr(cfg.real_data,
                                             "composite_statistic", None),
            "steps_with_no_valid_observation": [
                label for label, frac in zip(times, per_step_missing)
                if frac >= 1.0],
            "missing_fraction_per_step": {
                label: float(frac) for label, frac
                in zip(times, per_step_missing)},
            "alignment": dataset.metadata.get("temporal_alignment", {}),
        },
        "vegetation": _describe(ndvi_in, "ndvi"),
        "rainfall": _describe(rain_in, "rainfall"),
        "satellite_quality": {
            "mean_valid_observations_per_pixel": float(per_pixel_valid.mean()),
            "median_valid_observations_per_pixel":
                float(np.median(per_pixel_valid)),
            "min_valid_observations_per_pixel": int(per_pixel_valid.min()),
            "max_valid_observations_per_pixel": int(per_pixel_valid.max()),
            "pixels_below_min_valid_obs": int(
                (per_pixel_valid < cfg.quality.min_valid_obs).sum()),
            "min_valid_obs_required": int(cfg.quality.min_valid_obs),
            "pixels_above_max_missing_fraction": int(
                ((1.0 - per_pixel_valid / dataset.n_time)
                 > cfg.quality.max_missing_fraction).sum()),
            "compositing": dataset.metadata.get("ndvi_cube_tags", {}
                                                ).get("compositing", {}),
        },
        "rainfall_processing": dataset.metadata.get("rain_cube_tags", {}
                                                    ).get("rainfall", {}),
        "reference_labels": dataset.metadata.get("reference_labels", {}),
        "interpolation": dataset.metadata.get("interpolation", {}),
        "boundary_clipping": dataset.metadata.get("boundary_clipping", {}),
        "caveats": [MISSINGNESS_CAVEAT],
    }
    if valid_counts is not None:
        counts = np.asarray(valid_counts, dtype="float64")
        report["satellite_quality"]["mean_usable_scenes_per_composite"] = \
            float(np.nanmean(counts))
        report["satellite_quality"]["usable_scenes_per_step"] = {
            label: float(np.nanmean(counts[i]))
            for i, label in enumerate(times[:counts.shape[0]])}

    # The confound, stated as a number rather than as a worry: does
    # observation density itself trend over the record?
    steps = np.arange(dataset.n_time, dtype="float64")
    availability = 1.0 - per_step_missing
    if np.isfinite(availability).sum() >= 3 and np.std(availability) > 0:
        slope, intercept = np.polyfit(steps, availability, 1)
        correlation = float(np.corrcoef(steps, availability)[0, 1])
        report["temporal"]["observation_availability_trend"] = {
            "slope_per_step": float(slope),
            "intercept": float(intercept),
            "correlation_with_time": correlation,
            "interpretation": (
                "A non-zero slope means the record's observation density "
                "changes over time. Because trend tests have more power "
                "where they have more observations, this must be reported "
                "alongside any temporal trend result."),
        }

    report["missingness_table"] = [
        {"time_step": label,
         "ndvi_missing_fraction": float(a),
         "rainfall_missing_fraction": float(b)}
        for label, a, b in zip(times, per_step_missing, per_step_rain_missing)]
    return report


def write_quality_report(report: Dict[str, Any], output_dir) -> Dict[str, Path]:
    """Save the machine-readable artefacts Part 14 specifies."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    written = {}

    summary = {k: v for k, v in report.items() if k != "missingness_table"}
    path = root / "dataset_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str))
    written["dataset_summary"] = path

    missingness = pd.DataFrame(report.get("missingness_table", []))
    path = root / "missingness.csv"
    missingness.to_csv(path, index=False)
    written["missingness"] = path

    rows = []
    for section in ("vegetation", "rainfall"):
        block = report.get(section)
        if isinstance(block, dict):
            rows.append(block)
    quality = report.get("satellite_quality", {})
    rows.append({"variable": "valid_observations_per_pixel",
                 "mean": quality.get("mean_valid_observations_per_pixel"),
                 "median": quality.get("median_valid_observations_per_pixel"),
                 "min": quality.get("min_valid_observations_per_pixel"),
                 "max": quality.get("max_valid_observations_per_pixel")})
    path = root / "quality_report.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    written["quality_report"] = path
    return written


def plot_quality_report(dataset, report: Dict[str, Any], figure_dir) -> list:
    """Figures a reviewer needs to judge whether the record is analysable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(figure_dir)
    root.mkdir(parents=True, exist_ok=True)
    written = []
    times = [str(t) for t in dataset.times] or [str(i) for i
                                                in range(dataset.n_time)]
    steps = np.arange(dataset.n_time)
    banner = ("SYNTHETIC FIXTURE" if report.get("synthetic") else "REAL")

    # 1. availability over time -------------------------------------------
    table = pd.DataFrame(report["missingness_table"])
    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(steps, 100 * (1 - table["ndvi_missing_fraction"]),
              marker="o", markersize=3, label="NDVI")
    axis.plot(steps, 100 * (1 - table["rainfall_missing_fraction"]),
              marker="s", markersize=3, label="rainfall")
    axis.set_ylim(-2, 102)
    axis.set_xticks(steps[::max(len(steps) // 12, 1)])
    axis.set_xticklabels([times[i] for i in steps[::max(len(steps) // 12, 1)]],
                         rotation=45, ha="right")
    axis.set_ylabel("cells with a valid value (%)")
    axis.set_title(f"Data availability over the record\n{banner} "
                   f"observations; varying density affects statistical power",
                   fontsize=10)
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    path = root / "real_data_availability.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    written.append(path)

    # 2. valid observations per pixel -------------------------------------
    inside = report.get("_inside_mask")
    counts = np.isfinite(dataset.ndvi).sum(axis=0).astype("float64")
    if inside is not None:
        counts = np.where(np.asarray(inside, bool), counts, np.nan)
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    image = left.imshow(np.ma.masked_invalid(counts), cmap="viridis")
    left.set_title("Valid NDVI observations per pixel", fontsize=10)
    left.set_xticks([])
    left.set_yticks([])
    figure.colorbar(image, ax=left, shrink=0.85,
                    label=f"of {dataset.n_time} time steps")
    finite_counts = counts[np.isfinite(counts)]
    right.hist(finite_counts, bins=min(dataset.n_time + 1, 60),
               color="#3a86c8")
    right.axvline(report["satellite_quality"]["min_valid_obs_required"],
                  color="#c0242b", linestyle="--",
                  label="minimum required to analyse")
    right.set_xlabel("valid observations")
    right.set_ylabel("pixels")
    right.legend(fontsize=8)
    figure.suptitle(f"{banner} data quality: where the record can support a "
                    f"trend test", fontsize=11)
    figure.tight_layout()
    path = root / "real_data_valid_observations.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    written.append(path)

    # 3. NDVI and rainfall distributions ----------------------------------
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
    # Distributions describe the study area, so they use the same
    # inside-boundary restriction as the tables.
    keep = (np.ones(dataset.shape, bool) if inside is None
            else np.asarray(inside, bool))
    ndvi_cube = dataset.ndvi[:, keep]
    rain_cube = dataset.rain[:, keep]
    ndvi = ndvi_cube[np.isfinite(ndvi_cube)]
    rain = rain_cube[np.isfinite(rain_cube)]
    if ndvi.size:
        left.hist(ndvi, bins=60, color="#1a7a3a")
    left.set_xlabel("NDVI")
    left.set_ylabel("cells")
    left.set_title(f"NDVI distribution (n={ndvi.size:,})", fontsize=10)
    if rain.size:
        right.hist(rain, bins=60, color="#3a86c8")
    right.set_xlabel(f"rainfall ({report.get('rainfall_processing', {}).get('units', 'mm')})")
    right.set_title(f"Rainfall distribution (n={rain.size:,})", fontsize=10)
    figure.suptitle(f"{banner} observation distributions", fontsize=11)
    figure.tight_layout()
    path = root / "real_data_distributions.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    written.append(path)

    # 4. spatial mean NDVI ------------------------------------------------
    # Pixels outside the study-area boundary are NaN at every step, which is
    # exactly what should be plotted as blank; numpy warns about it anyway.
    import warnings
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_ndvi = np.nanmean(dataset.ndvi, axis=0)
    figure, axis = plt.subplots(figsize=(7, 5.5))
    image = axis.imshow(mean_ndvi, cmap="YlGn", vmin=-0.1, vmax=1.0)
    axis.set_title(f"Mean NDVI over the record ({banner} observations)",
                   fontsize=10)
    axis.set_xticks([])
    axis.set_yticks([])
    figure.colorbar(image, ax=axis, shrink=0.85, label="NDVI")
    figure.tight_layout()
    path = root / "real_data_mean_ndvi.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    written.append(path)
    return written
