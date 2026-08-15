"""Redraw an M7 study's figures from its saved layers.

Figures and analysis are separated on purpose: a cartographic fix should
never require re-running a six-minute study, and re-running one to change a
colour would risk the numbers and the pictures drifting apart. This reads the
GeoTIFFs and tables an M7 run already wrote and redraws from them, so the
figures cannot disagree with the results they illustrate.

    python tools/regenerate_m7_figures.py results/final_real_data/<run-id>
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import m7_figures as MF                              # noqa: E402
from src.geo import GeoRef                                    # noqa: E402
from src.m7_outputs import SOURCE                             # noqa: E402
from src.trajectory import TRAJECTORY_CODES                   # noqa: E402


def read(path: Path) -> tuple:
    with rasterio.open(path) as source:
        data = source.read(1).astype("float64")
        nodata = source.nodata
        georef = GeoRef(source.crs, source.transform, source.height,
                        source.width)
    if nodata is not None and np.isfinite(nodata):
        data[data == nodata] = np.nan
    return data, georef


def main(run: Path) -> int:
    analysis = run / "temporal_analysis"
    figures = run / "figures"
    if not analysis.exists():
        raise SystemExit(f"no temporal_analysis/ under {run}")

    summary_path = run / "summary" / "trajectories.json"
    areas = {}
    if summary_path.exists():
        areas = json.loads(summary_path.read_text()).get("areas_km2", {})
    label = "Karbi_Anglong_bbox"
    study = run / "configuration" / "study_area.geojson"
    if study.exists():
        label = json.loads(study.read_text()).get("name", label)

    codes, georef = read(analysis / "trajectory_class.tif")
    codes = np.where(codes == 0, np.nan, codes)
    names = {v: k for k, v in TRAJECTORY_CODES.items()}

    written = []
    written.append(MF.categorical_map(
        codes, georef, figures / "12_trajectory_classes.png",
        class_names=names, areas=areas,
        title="Integrated vegetation trajectories, 1990-2024",
        subtitle="analytical signal categories combining trend, climate "
                 "adjustment, recurrence, disturbance and recovery",
        source=SOURCE, study_area_label=label))
    written.append(MF.trajectory_facets(
        codes, georef, figures / "13_trajectory_facets.png",
        class_names=names,
        title="Trajectory classes, one panel per class", source=SOURCE))

    mean_path = analysis / "ndvi_mean.tif"
    if mean_path.exists():
        mean_ndvi, ref = read(mean_path)
        written.append(MF.map_panel(
            mean_ndvi, ref, figures / "02_mean_ndvi.png",
            title="Mean NDVI, 1990-2024",
            subtitle="post-monsoon annual composites (15 Oct - 31 Dec), "
                     "median",
            label="mean NDVI", kind="sequential",
            cmap=MF.SEQUENTIAL["vegetation"], vmin=0.0, vmax=1.0,
            source=SOURCE, study_area_label=label))

    slope_path = analysis / "sens_slope.tif"
    if slope_path.exists():
        slope, ref = read(slope_path)
        written.append(MF.map_panel(
            slope, ref, figures / "04_trend_sen_slope.png",
            title="Theil-Sen NDVI trend, 1990-2024",
            subtitle="a trend is a change in the vegetation index, not proof "
                     "of degradation",
            label="NDVI change per year", kind="diverging",
            cmap=MF.DIVERGING["trend"], source=SOURCE,
            study_area_label=label))

    written.extend(_profiles(run, figures, label))
    written.extend(_documents(run))
    for path in written:
        print("redrew", path.name)
    return 0


def _documents(run: Path) -> list:
    """Rebuild the findings document from the run's own saved results.

    The findings text is generated FROM the numbers, so regenerating it
    cannot change any result - only the wording that describes them. This
    exists so a wording correction does not require a six-minute re-run that
    would produce byte-identical numbers.
    """
    from src.config import Config
    from src.m7_outputs import write_findings

    results_path = run / "summary" / "results.json"
    config_path = run / "configuration" / "config.json"
    if not results_path.exists() or not config_path.exists():
        return []
    results = json.loads(results_path.read_text())
    if "baseline_comparison" not in results:
        return []
    cfg = Config.load(config_path)

    class Handle:
        experiment_id = run.name

        def path(self, subdir, filename=""):
            target = run / subdir
            target.mkdir(parents=True, exist_ok=True)
            return target / filename if filename else target

    class Quiet:
        def info(self, *args, **kwargs):
            pass

    return [write_findings(results, Handle(), cfg, Quiet())]


def _profiles(run: Path, figures: Path, label: str) -> list:
    """Redraw the representative temporal profiles.

    Everything needed is already on disk: the chosen pixel's grid position
    is in `summary/representative_pixels.json`, its series come from the
    composited cubes, and its statistics come from the layers this same run
    wrote. Nothing is recomputed, so a redrawn profile cannot disagree with
    the numbers the study reported.
    """
    chosen_path = run / "summary" / "representative_pixels.json"
    config_path = run / "configuration" / "real_data_config.json"
    if not chosen_path.exists() or not config_path.exists():
        return []
    chosen = json.loads(chosen_path.read_text())
    real = json.loads(config_path.read_text())
    ndvi_path, rain_path = Path(real["ndvi_cube"]), Path(real["rain_cube"])
    if not ndvi_path.exists() or not rain_path.exists():
        print("cubes unavailable; profiles not redrawn")
        return []

    with rasterio.open(ndvi_path) as source:
        ndvi = source.read().astype("float64")
        nodata = source.nodata
        times = [d for d in source.descriptions]
    ndvi[ndvi == nodata] = np.nan
    with rasterio.open(rain_path) as source:
        rain = source.read().astype("float64")
        rain_nodata = source.nodata
    rain[rain == rain_nodata] = np.nan

    analysis = run / "temporal_analysis"
    layers = {}
    for name in ("sens_slope", "mann_kendall_p", "restrend_slope",
                 "restrend_p", "restrend_valid", "break_index",
                 "dominant_period", "cyclicity_enrichment"):
        path = analysis / f"{name}.tif"
        if path.exists():
            layers[name] = read(path)[0]

    written = []
    for index, (name, record) in enumerate(sorted(chosen.items()), start=1):
        row, col = record["grid_row"], record["grid_col"]

        def at(layer, default=np.nan):
            values = layers.get(layer)
            return float(values[row, col]) if values is not None else default

        safe = name.replace(" / ", "_").replace(" ", "_").replace("-", "_")
        break_index = at("break_index", -1.0)
        written.append(MF.temporal_profile(
            times, ndvi[:, row, col], rain[:, row, col],
            figures / f"20_profile_{index:02d}_{safe}.png",
            title=f"Representative pixel — {name}",
            subtitle=(f"row {row}, col {col}; "
                      f"{record['n_valid_observations']} valid composites; "
                      f"most typical of {record['n_candidates']} pixels in "
                      f"this class"),
            source=SOURCE,
            sen_slope=at("sens_slope"), mk_p=at("mann_kendall_p"),
            restrend_slope=at("restrend_slope"), restrend_p=at("restrend_p"),
            restrend_valid=bool(at("restrend_valid", 0.0) > 0.5),
            break_index=int(break_index) if np.isfinite(break_index) else -1,
            period=at("dominant_period"),
            enrichment=at("cyclicity_enrichment")))
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="an M7 results directory")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    raise SystemExit(main(Path(args.run)))
