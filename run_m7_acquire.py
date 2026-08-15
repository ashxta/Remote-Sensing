"""M7 acquisition: fetch the real study record from public archives.

Downloads real Landsat Collection 2 Level-2 scenes and CHIRPS annual
rainfall for the configured study area, caches them locally, and writes the
M6 manifests. Nothing here analyses anything; the analysis is
`run_real_data.py`, unchanged.

    python run_m7_acquire.py --config configs/m7_karbi_anglong_final.json

No credential is used or required. The Landsat pixels are the USGS
Collection 2 Level-2 product; Microsoft Planetary Computer redistributes
them as Cloud-Optimized GeoTIFFs with anonymous read access.

Re-running is cheap: cached scenes are skipped, so an interrupted download
resumes. `--overwrite` forces a refetch.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from src.config import Config
from src.real_data import resolve_target_grid
from src.stac_source import (CHIRPS, PLANETARY_COMPUTER, SUBSAMPLING_NOTE,
                             build_scene_cache, fetch_chirps_annual,
                             search_landsat)
from src.study_area import load_study_area


def build_logger() -> logging.Logger:
    logger = logging.getLogger("m7.acquire")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    return logger


def main(cfg: Config, *, per_year: int = 8, workers: int = 8,
         overwrite: bool = False, logger=None) -> dict:
    logger = logger or build_logger()
    real = cfg.real_data
    area = load_study_area(cfg.study_area)
    grid, grid_note = resolve_target_grid(area, real)

    logger.info("study area: %s", area.name)
    logger.info("analysis grid: %dx%d at %s, CRS %s", grid.height, grid.width,
                grid_note["grid_resolution"], grid.crs)
    logger.info("period: %d-%d, window %s to %s", real.start_year,
                real.end_year, real.window_start, real.window_end)

    started = time.time()
    logger.info("searching %s for real Landsat scenes...",
                PLANETARY_COMPUTER["name"])
    items = search_landsat(
        area, start_year=real.start_year, end_year=real.end_year,
        window_start=real.window_start, window_end=real.window_end,
        max_cloud=real.max_scene_cloud_cover, platforms=real.sensors,
        per_year=per_year, logger=logger)
    logger.info("selected %d scene(s) in %.1fs", len(items),
                time.time() - started)

    by_sensor: dict = {}
    for item in items:
        by_sensor[item.sensor] = by_sensor.get(item.sensor, 0) + 1
    logger.info("by sensor: %s", by_sensor)

    raw_dir = Path(real.raw_dir)
    logger.info("fetching scenes onto the analysis grid (%d workers)...",
                workers)
    started = time.time()
    cache = build_scene_cache(items, grid, raw_dir, workers=workers,
                             overwrite=overwrite, logger=logger)
    logger.info("scenes cached in %.1fs", time.time() - started)

    logger.info("fetching CHIRPS annual rainfall...")
    rainfall = fetch_chirps_annual(area, raw_dir, start_year=real.start_year,
                                   end_year=real.end_year, logger=logger)

    record = {
        "study_area": area.describe(),
        "analysis_grid": grid_note,
        "period": [real.start_year, real.end_year],
        "composite_window": [real.window_start, real.window_end],
        "landsat": {
            "archive": PLANETARY_COMPUTER["name"],
            "collection": "landsat-c2-l2",
            "provenance": PLANETARY_COMPUTER["provenance"],
            "licence": PLANETARY_COMPUTER["licence"],
            "documentation": PLANETARY_COMPUTER["documentation"],
            "authentication": "none (anonymous)",
            "scenes_selected": len(items),
            "scenes_per_year_cap": per_year,
            "scenes_cached": cache["n_cached"],
            "scenes_outside_study_area": cache["n_empty"],
            "scenes_failed": cache["n_failed"],
            "failures": cache["failures"],
            "by_sensor": by_sensor,
            "sampling": SUBSAMPLING_NOTE,
            "manifest": str(cache["manifest"]),
        },
        "rainfall": {
            "product": CHIRPS["name"] + " " + CHIRPS["product"],
            "citation": CHIRPS["citation"],
            "licence": CHIRPS["licence"],
            "years_retrieved": len(rainfall["dates"]),
            "years_unavailable": rainfall["years_unavailable"],
            "manifest": str(rainfall["manifest"]),
        },
    }
    target = Path(real.metadata_dir) / "m7_acquisition.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2, default=str))
    logger.info("acquisition record -> %s", target)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--per-year", type=int, default=8,
                        help="cap on scenes contributed by each year")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    main(Config.load(args.config), per_year=args.per_year,
         workers=args.workers, overwrite=args.overwrite)
