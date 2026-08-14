"""FABRICATED raw satellite scenes, for testing the real-data ingestion path.

WHAT THIS IS
------------
`demo/make_synthetic_data.py` produces finished NDVI/rainfall stacks - it
starts where the real-data pipeline ends, so it cannot test any of M6. This
script instead fabricates the INPUTS of the real-data path:

* per-scene Collection-2-style surface-reflectance bands, as unsigned 16-bit
  integers with the real scale factor and fill value;
* per-scene QA_PIXEL bitmasks with genuine Collection-2 bit positions for
  fill, dilated cloud, cloud, cloud shadow, cirrus and snow;
* QA_RADSAT saturation bands;
* Landsat 5/7/8/9 acquisitions on irregular dates, with the correct
  per-sensor band naming (SR_B3/SR_B4 for TM/ETM+, SR_B4/SR_B5 for OLI),
  the real mission date ranges, and Landsat 7 SLC-off wedge gaps after
  May 2003;
* a CHIRPS-like daily precipitation raster on a COARSE geographic grid in a
  different CRS from the scenes, so that reprojection and alignment are
  genuinely exercised rather than trivially satisfied.

Running the ingestion over these files exercises band mapping, scale
factors, QA bit masking, index computation, cross-sensor NDVI
harmonisation, temporal compositing, rainfall accumulation, CRS
reprojection, grid alignment and the standardized-data contract - the whole
of M6 - offline, with no credentials and no network.

WHAT THIS IS NOT
----------------
These are NOT observations. They are not a simulation of Karbi Anglong or
of anywhere else; the spatial pattern is a schematic invented to give the
estimators something structured to find. The manifest it writes carries
`"synthetic": true`, that flag is written into the resulting cubes' GeoTIFF
tags, and `real_data.RealRemoteSensingSource` reads it back and labels the
dataset SYNTHETIC FIXTURE. Nothing computed from these files may be
reported as a research finding.

    python demo/make_scene_fixture.py --out data/raw/fixture
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sensors import LANDSAT_QA_BITS, get_sensor          # noqa: E402
from src.study_area import StudyArea                          # noqa: E402

FIXTURE_NOTICE = (
    "FABRICATED scene fixture for testing the M6 real-data ingestion path. "
    "Not observations of any location. See demo/make_scene_fixture.py.")

#: Mission operating periods, so a scene is never dated outside its sensor's
#: real archive. Getting this wrong would make the fixture test a scenario
#: that cannot occur.
MISSIONS = {
    "LANDSAT5_TM": (dt.date(1984, 3, 1), dt.date(2012, 5, 5)),
    "LANDSAT7_ETM": (dt.date(1999, 5, 28), dt.date(2024, 1, 19)),
    "LANDSAT8_OLI": (dt.date(2013, 3, 18), dt.date(2100, 1, 1)),
    "LANDSAT9_OLI2": (dt.date(2021, 10, 31), dt.date(2100, 1, 1)),
}
SLC_FAILURE = dt.date(2003, 5, 31)


def landscape(height: int, width: int, rng) -> dict:
    """A schematic of vegetation archetypes. Invented, not observed."""
    yy, xx = np.mgrid[0:height, 0:width] / np.array([[height], [width]]
                                                    ).reshape(2, 1, 1)
    hill = np.exp(-(((yy - 0.55) ** 2) / 0.09 + ((xx - 0.60) ** 2) / 0.10))
    archetype = np.full((height, width), 2, dtype="int16")   # cropland
    archetype[hill > 0.55] = 1                               # stable forest
    archetype[(hill > 0.25) & (hill <= 0.55)] = 3            # cyclic
    for code, count, radius in ((4, 6, 0.05), (5, 4, 0.045)):
        for _ in range(count):
            cy, cx = rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)
            blob = ((yy - cy) ** 2 + ((xx - cx) * 1.4) ** 2) < radius ** 2
            archetype[blob] = code
    return {"archetype": archetype, "hill": hill}


def ndvi_for(archetype: np.ndarray, year_fraction: float, rain_anomaly: float,
             rng) -> np.ndarray:
    """Per-archetype NDVI at one moment. A schematic, not a growth model."""
    shape = archetype.shape
    value = np.full(shape, 0.45, dtype="float64")
    noise = rng.normal(0, 0.03, shape)

    stable = archetype == 1
    value[stable] = 0.78 + 0.02 * rain_anomaly
    crop = archetype == 2
    value[crop] = 0.46 + 0.07 * rain_anomaly + 0.05 * year_fraction
    cyclic = archetype == 3
    phase = (year_fraction * 36.0 / 7.0) % 1.0
    value[cyclic] = 0.30 + 0.34 * phase + 0.03 * rain_anomaly
    declining = archetype == 4
    value[declining] = 0.72 - 0.45 * year_fraction + 0.03 * rain_anomaly
    recovering = archetype == 5
    value[recovering] = 0.28 + 0.40 * year_fraction + 0.03 * rain_anomaly
    return np.clip(value + noise, 0.02, 0.95)


def reflectance_from_ndvi(ndvi: np.ndarray, sensor, rng) -> tuple:
    """Invert NDVI into a plausible (red, nir) reflectance pair.

    NDVI fixes only the ratio, so NIR is drawn around a vegetation-like
    level and red follows. The result is then encoded exactly as
    Collection-2 does - (reflectance - offset) / scale, clipped to uint16,
    with 0 reserved for fill - so `sensors.apply_scale_factors` has real
    integers to decode.
    """
    nir = np.clip(0.30 + 0.12 * ndvi + rng.normal(0, 0.01, ndvi.shape),
                  0.05, 0.6)
    red = np.clip(nir * (1.0 - ndvi) / (1.0 + ndvi), 0.001, 0.5)

    def encode(values):
        raw = np.round((values - sensor.offset) / sensor.scale)
        return np.clip(raw, 1, 65455).astype("uint16")

    return encode(red), encode(nir)


def qa_pixel(shape, date, sensor_key, rng) -> np.ndarray:
    """A Collection-2 QA_PIXEL band with real bit positions.

    Cloud is drawn as coherent blobs rather than salt-and-pepper, because a
    per-pixel random mask would average away in compositing and would not
    test anything. Landsat 7 acquisitions after the 2003 SLC failure get
    wedge-shaped fill gaps.
    """
    height, width = shape
    qa = np.zeros(shape, dtype="uint16")
    qa |= 1 << LANDSAT_QA_BITS["clear"]

    yy, xx = np.mgrid[0:height, 0:width]
    cloud = np.zeros(shape, bool)
    for _ in range(rng.integers(0, 4)):
        cy, cx = rng.uniform(0, height), rng.uniform(0, width)
        radius = rng.uniform(0.05, 0.25) * max(height, width)
        cloud |= ((yy - cy) ** 2 + (xx - cx) ** 2) < radius ** 2
    shadow = np.roll(np.roll(cloud, int(0.03 * height), axis=0),
                     int(0.03 * width), axis=1) & ~cloud
    dilated = np.roll(cloud, 1, axis=0) & ~cloud & ~shadow

    qa[cloud] |= 1 << LANDSAT_QA_BITS["cloud"]
    qa[cloud] &= ~np.uint16(1 << LANDSAT_QA_BITS["clear"])
    qa[shadow] |= 1 << LANDSAT_QA_BITS["cloud_shadow"]
    qa[shadow] &= ~np.uint16(1 << LANDSAT_QA_BITS["clear"])
    qa[dilated] |= 1 << LANDSAT_QA_BITS["dilated_cloud"]

    if sensor_key == "LANDSAT7_ETM" and date > SLC_FAILURE:
        stripe = ((xx + 3 * yy) // max(width // 22, 1)) % 3 == 0
        edge = (xx < 0.12 * width) | (xx > 0.88 * width)
        gap = stripe & edge
        qa[gap] |= 1 << LANDSAT_QA_BITS["fill"]
        qa[gap] &= ~np.uint16(1 << LANDSAT_QA_BITS["clear"])
    return qa


def write_single(path: Path, array, crs, transform, dtype, nodata=None,
                 description="") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "height": array.shape[0],
               "width": array.shape[1], "count": 1, "dtype": dtype,
               "crs": crs, "transform": transform, "compress": "deflate"}
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as target:
        target.write(array.astype(dtype), 1)
        if description:
            target.update_tags(DESCRIPTION=description, FIXTURE=FIXTURE_NOTICE)
    return path


def build(out_dir: Path, *, start_year: int, end_year: int, height: int,
          width: int, scenes_per_year: int, seed: int,
          bounds=(92.30, 25.55, 93.85, 26.60)) -> dict:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    scene_dir = out_dir / "scenes"

    west, south, east, north = bounds
    area = StudyArea.from_bounds(west, south, east, north,
                                 name="fixture_extent")
    # Scenes are written in UTM (metres), the rainfall product in WGS84
    # (degrees), so the ingestion has to reproject one onto the other for
    # real rather than trivially agreeing.
    scene_crs_name = "EPSG:32646"
    utm_west, utm_south, utm_east, utm_north = area.bounds_in(scene_crs_name)
    resolution_m = max((utm_east - utm_west) / width,
                       (utm_north - utm_south) / height)
    scene_grid = area.grid(resolution_m, crs=scene_crs_name)
    scene_crs, scene_transform = scene_grid.crs, scene_grid.transform
    height, width = scene_grid.shape

    scape = landscape(height, width, rng)
    archetype = scape["archetype"]

    years = list(range(start_year, end_year + 1))
    rainfall_by_year = {y: 1800 + 260 * np.sin(2 * np.pi * i / 7.3)
                        + rng.normal(0, 170)
                        for i, y in enumerate(years)}
    for drought in (years[len(years) // 3], years[2 * len(years) // 3]):
        rainfall_by_year[drought] -= 430
    anomalies = np.array([rainfall_by_year[y] for y in years])
    anomalies = (anomalies - anomalies.mean()) / (anomalies.std() + 1e-9)

    records = []
    for index, year in enumerate(years):
        fraction = index / max(len(years) - 1, 1)
        available = [k for k, (first, last) in MISSIONS.items()
                     if first.year <= year <= last.year]
        for _ in range(scenes_per_year):
            key = str(rng.choice(available))
            first, last = MISSIONS[key]
            day = dt.date(year, 10, 15) + dt.timedelta(
                days=int(rng.integers(0, 78)))
            if not (first <= day <= last):
                continue
            sensor = get_sensor(key)
            ndvi = ndvi_for(archetype, fraction, float(anomalies[index]), rng)
            red, nir = reflectance_from_ndvi(ndvi, sensor, rng)
            qa = qa_pixel((height, width), day, key, rng)
            # Fill in QA means no observation, so the reflectance must be
            # fill too - that is what a real scene looks like.
            fill = (qa & (1 << LANDSAT_QA_BITS["fill"])) != 0
            red[fill] = 0
            nir[fill] = 0
            saturation = np.zeros((height, width), dtype="uint16")
            saturation[rng.random((height, width)) < 0.0005] = 1

            scene_id = f"{key}_{day.isoformat()}_{rng.integers(1000, 9999)}"
            base = scene_dir / scene_id
            red_path = write_single(
                base.with_name(f"{scene_id}_{sensor.band('red')}.tif"), red,
                scene_crs, scene_transform, "uint16",
                description=f"fixture {sensor.band('red')} DN")
            nir_path = write_single(
                base.with_name(f"{scene_id}_{sensor.band('nir')}.tif"), nir,
                scene_crs, scene_transform, "uint16",
                description=f"fixture {sensor.band('nir')} DN")
            qa_path = write_single(
                base.with_name(f"{scene_id}_QA_PIXEL.tif"), qa, scene_crs,
                scene_transform, "uint16", description="fixture QA_PIXEL")
            sat_path = write_single(
                base.with_name(f"{scene_id}_QA_RADSAT.tif"), saturation,
                scene_crs, scene_transform, "uint16",
                description="fixture QA_RADSAT")

            records.append({
                "date": day.isoformat(), "sensor": key, "scene_id": scene_id,
                "bands": {"red": str(red_path), "nir": str(nir_path)},
                "qa": str(qa_path), "saturation": str(sat_path),
                "scene_cloud_cover": float(
                    100 * ((qa & (1 << LANDSAT_QA_BITS["cloud"])) != 0).mean()),
            })

    scenes_manifest = out_dir / "scenes.json"
    scenes_manifest.write_text(json.dumps(
        {"scenes": records,
         "metadata": {
             "synthetic": True,
             "generator": "demo/make_scene_fixture.py",
             "notice": FIXTURE_NOTICE,
             "seed": seed,
             "years": [start_year, end_year],
             "n_scenes": len(records),
             "grid": {"crs": str(scene_crs), "height": height, "width": width},
         }}, indent=2), encoding="utf-8")

    # --- rainfall: coarse, geographic, monthly ---------------------------
    rain_res = 0.05                                   # CHIRPS-like
    rain_grid = area.grid(rain_res, crs="EPSG:4326")
    rain_h, rain_w = rain_grid.shape
    yy = np.linspace(0, 1, rain_h)[:, None] * np.ones((1, rain_w))
    rain_dates, bands = [], []
    for index, year in enumerate(years):
        annual = rainfall_by_year[year]
        for month in range(1, 13):
            # Monsoon-shaped seasonality, so an annual accumulation is not
            # just twelve equal slices.
            weight = 0.02 + 0.20 * np.exp(-((month - 7) ** 2) / 6.0)
            field = (annual * weight * (1 + 0.25 * (yy - 0.5))
                     + rng.normal(0, annual * weight * 0.08, (rain_h, rain_w)))
            bands.append(np.clip(field, 0, None))
            rain_dates.append(dt.date(year, month, 15).isoformat())

    rain_path = out_dir / "rainfall_monthly.tif"
    with rasterio.open(rain_path, "w", driver="GTiff", height=rain_h,
                       width=rain_w, count=len(bands), dtype="float32",
                       crs=rain_grid.crs, transform=rain_grid.transform,
                       nodata=-9999.0, compress="deflate") as target:
        for i, band in enumerate(bands, start=1):
            target.write(band.astype("float32"), i)
            target.set_band_description(i, rain_dates[i - 1])
        target.update_tags(FIXTURE=FIXTURE_NOTICE)

    rain_manifest = out_dir / "rainfall.json"
    rain_manifest.write_text(json.dumps(
        {"file": str(rain_path), "dates": rain_dates,
         "metadata": {"synthetic": True, "notice": FIXTURE_NOTICE,
                      "product": "CHIRPS-like fixture, monthly totals",
                      "units": "mm", "resolution_deg": rain_res,
                      "crs": "EPSG:4326"}}, indent=2), encoding="utf-8")

    # --- archetype grid, saved but NOT as reference labels ---------------
    # It is written so a test can check that the ingestion preserves spatial
    # structure. It is deliberately NOT wired to real_data.reference: these
    # are generator archetypes, and calling them ground truth is exactly the
    # fabrication M6 forbids.
    truth_path = out_dir / "generator_archetypes.tif"
    write_single(truth_path, archetype, scene_crs, scene_transform, "int16",
                 nodata=0,
                 description=("archetypes PLANTED by the fixture generator; "
                              "NOT reference labels and NOT ground truth"))

    summary = {"scenes_manifest": str(scenes_manifest),
               "rainfall_manifest": str(rain_manifest),
               "generator_archetypes": str(truth_path),
               "n_scenes": len(records), "grid": [height, width],
               "scene_crs": str(scene_crs), "rain_crs": "EPSG:4326",
               "years": [start_year, end_year], "notice": FIXTURE_NOTICE}
    (out_dir / "fixture_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw/fixture")
    parser.add_argument("--start-year", type=int, default=1990)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--width", type=int, default=120)
    parser.add_argument("--scenes-per-year", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = build(Path(args.out), start_year=args.start_year,
                   end_year=args.end_year, height=args.height,
                   width=args.width, scenes_per_year=args.scenes_per_year,
                   seed=args.seed)
    print(json.dumps(result, indent=2))
    print("\n" + FIXTURE_NOTICE)
