"""Is native 30 m feasible, and what does 300 m subsampling actually cost?

M7 read scenes onto a 300 m grid by NEAREST-NEIGHBOUR subsampling: each
analysis cell took the value of one 30 m pixel and ignored the other ~99.
That was a defensible choice about MASKING (a QA bitmask cannot be averaged,
and averaging reflectance before masking blends cloud into clear), but it is
not a 300 m spatial representation, and this benchmark measures what it
costs and what the alternatives cost.

Three representations are compared on identical ground:

  A  native 30 m                       - no resampling at all
  B  aggregate-after-masking to 300 m  - mask each 30 m pixel, then average
                                         the survivors within each cell
  C  nearest-neighbour 300 m           - what M7 did

B is the scientifically correct coarsening: masking happens at the
resolution the mask was made for, and the cell value then summarises every
valid observation inside it. The benchmark also records how many valid 30 m
pixels contributed to each cell, which is the mixed-pixel information C
throws away.

The script additionally establishes, empirically, what resampling the
archive's COG overviews use - because if the reflectance overviews are
averaged, then reading an overview has ALREADY blended cloudy and clear
pixels before any mask could be applied, and no amount of care afterwards
recovers it.

    python tools/benchmark_resolution.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geo import GeoRef                                    # noqa: E402
from src.sensors import (apply_scale_factors, compute_index,  # noqa: E402
                         get_sensor, landsat_qa_mask)
from src.stac_source import (GDAL_HTTP_ENV, _choose_overview,  # noqa: E402
                             search_landsat, sign_href)
from src.study_area import StudyArea                          # noqa: E402


def read_at(href: str, grid: GeoRef, level):
    """Read an asset onto `grid`; `level` None means full resolution."""
    url = f"/vsicurl/{sign_href(href)}"
    with rasterio.Env(**GDAL_HTTP_ENV):
        opened = rasterio.open(url, overview_level=level) if level is not None \
            else rasterio.open(url)
        with opened as source:
            with WarpedVRT(source, crs=grid.crs, transform=grid.transform,
                           width=grid.width, height=grid.height,
                           resampling=Resampling.nearest,
                           src_nodata=0, nodata=0) as vrt:
                return vrt.read(1)


def _choose_overview_for(href: str, target_resolution: float):
    with rasterio.Env(**GDAL_HTTP_ENV):
        with rasterio.open(f"/vsicurl/{sign_href(href)}") as source:
            return _choose_overview(source, target_resolution)


def aggregate(fine: np.ndarray, factor: int, how: str = "mean"):
    """Block-reduce a fine grid by an integer factor, ignoring NaN."""
    rows = (fine.shape[0] // factor) * factor
    cols = (fine.shape[1] // factor) * factor
    blocks = fine[:rows, :cols].reshape(rows // factor, factor,
                                        cols // factor, factor)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if how == "mean":
            return np.nanmean(blocks, axis=(1, 3))
        if how == "median":
            return np.nanmedian(blocks, axis=(1, 3))
        if how == "count":
            return np.isfinite(blocks).sum(axis=(1, 3))
    raise ValueError(how)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary",
                        default="data/boundaries/karbi_anglong.geojson")
    parser.add_argument("--year", type=int, default=2015)
    parser.add_argument("--window-km", type=float, default=18.0,
                        help="edge of the square test window")
    parser.add_argument("--out", default="data/metadata/resolution_benchmark.json")
    args = parser.parse_args()

    area = StudyArea.from_geojson(args.boundary)
    items = search_landsat(area, start_year=args.year, end_year=args.year,
                           per_year=1)
    if not items:
        raise SystemExit("no scene found for the benchmark")
    item = items[0]
    sensor = get_sensor(item.sensor)
    print(f"benchmark scene: {item.item_id} ({item.sensor}, "
          f"cloud {item.cloud_cover}%)")

    # Place the test window where THIS SCENE actually has data. A Landsat
    # footprint covers only part of the district, so a window at the
    # district centre can miss the scene entirely - which is exactly what
    # happened on the first attempt and produced an empty comparison.
    full = area.grid(30.0, crs="EPSG:32646")
    side = int(args.window_km * 1000 / 30.0)
    side -= side % 10                       # divisible by the 300/30 factor
    from rasterio.transform import Affine

    scout_grid = area.grid(300.0, crs="EPSG:32646")
    scout = read_at(item.assets["red"], scout_grid,
                    _choose_overview_for(item.assets["red"], 300.0))
    covered = scout > 0
    if not covered.any():
        raise SystemExit(f"{item.item_id} covers none of the study area")
    # Densest coverage: the coarse cell whose neighbourhood is fullest.
    block = side // 10
    best, row0, col0 = -1.0, 0, 0
    step = max(block // 4, 1)
    for row in range(0, max(covered.shape[0] - block, 1), step):
        for col in range(0, max(covered.shape[1] - block, 1), step):
            share = covered[row:row + block, col:col + block].mean()
            if share > best:
                best, row0, col0 = share, row * 10, col * 10
    print(f"window placed at native row {row0}, col {col0} "
          f"({best:.0%} scene coverage)")
    fine = GeoRef(full.crs, full.transform * Affine.translation(col0, row0),
                  side, side)
    coarse = GeoRef(full.crs,
                    full.transform * Affine.translation(col0, row0)
                    * Affine.scale(10, 10), side // 10, side // 10)
    print(f"test window: {side}x{side} at 30 m "
          f"({side * 30 / 1000:.1f} km square) -> {side // 10}x{side // 10} "
          f"at 300 m")

    report = {"scene": item.describe(), "window_px_30m": side,
              "window_km": side * 30 / 1000}

    # ---- native 30 m ---------------------------------------------------
    tracemalloc.start()
    start = time.time()
    red_n = read_at(item.assets["red"], fine, None)
    nir_n = read_at(item.assets["nir"], fine, None)
    qa_n = read_at(item.assets["qa"], fine, None)
    native_seconds = time.time() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"native 30 m read : {native_seconds:.1f}s, peak "
          f"{peak / 1e6:.0f} MB for 3 bands")
    report["native_read_seconds_per_scene_window"] = native_seconds
    report["native_peak_mb_window"] = peak / 1e6

    # ---- overview semantics --------------------------------------------
    with rasterio.Env(**GDAL_HTTP_ENV):
        with rasterio.open(f"/vsicurl/{sign_href(item.assets['red'])}") as src:
            factors = src.overviews(1)
    report["overview_factors"] = factors
    print(f"overview factors : {factors}")

    semantics = {}
    for level, factor in enumerate(factors[:3]):
        target = GeoRef(fine.crs, fine.transform * Affine.scale(factor, factor),
                        side // factor, side // factor)
        over_red = read_at(item.assets["red"], target, level).astype("float64")
        over_qa = read_at(item.assets["qa"], target, level).astype("float64")
        native_red = np.where(red_n == 0, np.nan, red_n).astype("float64")
        mean_block = aggregate(native_red, factor, "mean")
        near_block = native_red[::factor, ::factor][:target.height,
                                                    :target.width]
        over = np.where(over_red == 0, np.nan, over_red)
        both = np.isfinite(over) & np.isfinite(mean_block) & np.isfinite(near_block)
        if both.sum() < 50:
            continue
        d_mean = float(np.nanmedian(np.abs(over[both] - mean_block[both])))
        d_near = float(np.nanmedian(np.abs(over[both] - near_block[both])))
        # QA is a bitmask: if its overview matched an average it would be
        # meaningless, so check whether the values are even valid QA words.
        qa_near = qa_n[::factor, ::factor][:target.height, :target.width]
        qa_match = float(np.mean(over_qa[both] == qa_near[both]))
        semantics[f"{factor}x"] = {
            "median_abs_diff_vs_block_mean_DN": d_mean,
            "median_abs_diff_vs_nearest_DN": d_near,
            "reflectance_overview_looks_like":
                "average" if d_mean < d_near else "nearest",
            "qa_overview_matches_nearest_fraction": qa_match,
        }
        print(f"  {factor}x overview: |over-mean|={d_mean:.0f} DN, "
              f"|over-nearest|={d_near:.0f} DN -> "
              f"{semantics[f'{factor}x']['reflectance_overview_looks_like']}"
              f"; QA==nearest on {qa_match:.1%}")
    report["overview_semantics"] = semantics

    # ---- the three representations --------------------------------------
    red = apply_scale_factors(red_n, sensor.scale, sensor.offset)
    nir = apply_scale_factors(nir_n, sensor.scale, sensor.offset)
    usable = landsat_qa_mask(np.where(qa_n == 0, 1, qa_n))
    ndvi_native = compute_index("ndvi", {"red": np.where(usable, red, np.nan),
                                         "nir": np.where(usable, nir, np.nan)})

    # B: mask at 30 m, then average the survivors into each 300 m cell.
    ndvi_aggregated = aggregate(ndvi_native, 10, "mean")
    valid_count = aggregate(ndvi_native, 10, "count")

    # C: what M7 did - one 30 m pixel per 300 m cell, chosen by position.
    ndvi_nearest = ndvi_native[::10, ::10][:coarse.height, :coarse.width]

    common = np.isfinite(ndvi_aggregated) & np.isfinite(ndvi_nearest)
    if common.sum() < 20:
        raise SystemExit(
            f"only {int(common.sum())} cells have both representations; the "
            "test window does not overlap enough usable scene data")
    difference = ndvi_nearest[common] - ndvi_aggregated[common]
    report["representation_comparison"] = {
        "n_cells": int(common.sum()),
        "mean_valid_30m_pixels_per_cell": float(np.nanmean(valid_count)),
        "cells_with_no_valid_30m_pixel": int((valid_count == 0).sum()),
        "nearest_minus_aggregated": {
            "mean": float(np.mean(difference)),
            "median": float(np.median(difference)),
            "std": float(np.std(difference)),
            "p05": float(np.percentile(difference, 5)),
            "p95": float(np.percentile(difference, 95)),
            "mean_abs": float(np.mean(np.abs(difference))),
        },
        "within_cell_ndvi_spread": {
            "median_std_across_cells": float(np.nanmedian(
                aggregate(ndvi_native, 10, "mean") * 0 +
                _within_cell_std(ndvi_native, 10))),
        },
    }
    stats = report["representation_comparison"]["nearest_minus_aggregated"]
    print(f"\nnearest vs aggregated NDVI over {int(common.sum())} cells:")
    print(f"  mean difference {stats['mean']:+.4f}, "
          f"sd {stats['std']:.4f}, mean|diff| {stats['mean_abs']:.4f}, "
          f"p05..p95 {stats['p05']:+.4f}..{stats['p95']:+.4f}")
    print(f"  valid 30 m pixels per 300 m cell: "
          f"{report['representation_comparison']['mean_valid_30m_pixels_per_cell']:.1f}"
          f" of 100")
    print(f"  median within-cell NDVI sd: "
          f"{report['representation_comparison']['within_cell_ndvi_spread']['median_std_across_cells']:.4f}")

    # ---- extrapolate the cost of a full native run ----------------------
    district = area.grid(30.0, crs="EPSG:32646")
    inside = area.mask(district)
    n_native = int(inside.sum())
    window_px = side * side
    per_scene = native_seconds * (n_native / window_px)
    report["full_native_extrapolation"] = {
        "district_pixels_at_30m": n_native,
        "district_pixels_at_300m": int(area.mask(area.grid(300.0,
                                                           crs="EPSG:32646")).sum()),
        "seconds_per_scene_estimated": per_scene,
        "scenes": 264,
        "acquisition_hours_estimated": per_scene * 264 / 3600,
        "feature_engineering_hours_estimated": (
            n_native / 200493) * (3.6 / 60),
        "note": ("Feature engineering was measured at ~3.6 minutes for "
                 "200,493 pixels over 35 steps in the M7 run; the estimate "
                 "scales that linearly."),
    }
    extrapolation = report["full_native_extrapolation"]
    print(f"\nfull-district native 30 m estimate:")
    print(f"  pixels     : {n_native:,} (vs "
          f"{extrapolation['district_pixels_at_300m']:,} at 300 m)")
    print(f"  acquisition: ~{extrapolation['acquisition_hours_estimated']:.1f} h")
    print(f"  features   : ~{extrapolation['feature_engineering_hours_estimated']:.1f} h")

    target_path = Path(args.out)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {target_path}")
    return 0


def _within_cell_std(fine: np.ndarray, factor: int) -> np.ndarray:
    rows = (fine.shape[0] // factor) * factor
    cols = (fine.shape[1] // factor) * factor
    blocks = fine[:rows, :cols].reshape(rows // factor, factor,
                                        cols // factor, factor)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(blocks, axis=(1, 3))


if __name__ == "__main__":
    raise SystemExit(main())
