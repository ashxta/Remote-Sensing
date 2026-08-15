"""Real Landsat acquisition over STAC (M7).

M6 built the preprocessing but could not reach any satellite archive: Earth
Engine needs a credential and USGS EarthExplorer needs a login. This module
supplies the missing acquisition path using a service that needs NEITHER.

    Microsoft Planetary Computer, collection `landsat-c2-l2`

holds the USGS Landsat Collection 2 Level-2 archive as Cloud-Optimized
GeoTIFFs, with an anonymous STAC API and anonymously-issued read tokens.
The PIXELS ARE THE USGS PRODUCT - the same Collection 2 Level-2 surface
reflectance that EarthExplorer serves, with the same scene identifiers, the
same scale factors and the same QA_PIXEL band. Only the delivery differs.

    STAC search  ->  sign asset URLs  ->  windowed COG read
                 ->  cached scene GeoTIFFs + scenes.json (the M6 manifest)
                 ->  real_data.preprocess_real_data   [ unchanged ]
                 ->  StandardizedDataset              [ unchanged ]
                 ->  the M1-M5 analysis               [ unchanged ]

Nothing downstream knows the data arrived over STAC. This module writes the
manifest format `real_data.load_manifest` already reads, so the whole of M6's
preprocessing - band mapping, scale factors, QA bit decoding, NDVI,
harmonisation, compositing - runs on it without modification.

RESOLUTION AND SUBSAMPLING - READ THIS BEFORE QUOTING A RESULT
--------------------------------------------------------------
Landsat is 30 m. A 36-year, district-wide record at 30 m is several hundred
gigabytes, which is not retrievable in a single session. This module
therefore reads each scene onto a COARSER analysis grid, and it does so by
NEAREST-NEIGHBOUR SUBSAMPLING, not by averaging.

That choice is deliberate and it is not the obvious one:

* Averaging reflectance to 300 m BEFORE the cloud mask is applied would
  blend clear and cloudy native pixels into a single contaminated value that
  no later masking can separate. The QA band cannot be averaged at all - it
  is a bitmask, and the mean of two bit patterns is a third, meaningless
  pattern.
* Nearest-neighbour subsampling keeps every analysis cell a GENUINE single
  30 m observation, carrying its own exact QA flags. The mask stays exact
  and NDVI stays a real measured value.

The cost is stated plainly: the result is a **spatial subsample of the 30 m
record, not a 300 m aggregate of it**. Each analysis cell reports one 30 m
pixel and says nothing about the other ~99 within its footprint. Area
statistics computed from it are estimates whose precision is bounded by that
sampling, and a small feature that falls between sample points is invisible.
Every run records `sampling: "nearest-neighbour subsample"` in its metadata,
and the M7 report repeats it.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from .geo import GeoRef
from .study_area import StudyArea

__all__ = ["StacError", "StacItem", "search_landsat", "sign_href",
           "fetch_scene", "build_scene_cache", "PLANETARY_COMPUTER",
           "SUBSAMPLING_NOTE", "PLATFORM_TO_SENSOR", "CHIRPS",
           "fetch_chirps_annual"]

PLANETARY_COMPUTER = {
    "search": "https://planetarycomputer.microsoft.com/api/stac/v1/search",
    "collections": ("https://planetarycomputer.microsoft.com/api/stac/v1"
                    "/collections"),
    "sas": "https://planetarycomputer.microsoft.com/api/sas/v1/token",
    "name": "Microsoft Planetary Computer",
    "documentation": "https://planetarycomputer.microsoft.com/docs/",
    "licence": ("Landsat Collection 2 data are US Government public domain "
                "(USGS). Planetary Computer provides hosting and anonymous "
                "read access under its terms of use."),
    "provenance": ("USGS Landsat Collection 2 Level-2 Science Products, "
                   "redistributed unmodified as Cloud-Optimized GeoTIFFs."),
}

SUBSAMPLING_NOTE = (
    "Scenes were read onto the analysis grid by NEAREST-NEIGHBOUR "
    "subsampling of the native 30 m pixels, not by spatial averaging. Each "
    "analysis cell is one genuine 30 m observation with its own exact QA "
    "flags; it does not summarise the other native pixels in its footprint. "
    "Results are a spatial subsample of the 30 m record.")

#: The corrected coarsening, used when `aggregate_factor > 1`.
AGGREGATION_NOTE = (
    "Scenes were read at NATIVE 30 m resolution, the per-pixel QA mask was "
    "applied at that resolution, and only the surviving 30 m pixels were "
    "averaged into each analysis cell. Each cell therefore summarises every "
    "valid observation inside its footprint, and the count of contributing "
    "30 m pixels is retained as a mixed-pixel diagnostic. Reflectance is "
    "averaged before the index is formed; because the sensor scale factor "
    "is affine, the mean of the stored integers is exactly the mean of the "
    "reflectances.")

#: Why reading a COG overview instead is NOT an acceptable shortcut here.
OVERVIEW_WARNING = (
    "Measured on this archive (tools/benchmark_resolution.py): the 4x and 8x "
    "reflectance overviews are AVERAGES of the underlying native pixels, "
    "while the QA_PIXEL overviews are point samples. Reading an overview "
    "therefore averages cloudy and clear reflectance together before any "
    "mask can be applied, and pairs the result with a quality flag that "
    "describes only one of the contributing pixels. Overviews are used only "
    "for scouting coverage, never for the analysed values.")

#: STAC `platform` value -> this project's sensor key.
PLATFORM_TO_SENSOR = {
    "landsat-4": "LANDSAT5_TM",      # TM; identical band layout to Landsat 5
    "landsat-5": "LANDSAT5_TM",
    "landsat-7": "LANDSAT7_ETM",
    "landsat-8": "LANDSAT8_OLI",
    "landsat-9": "LANDSAT9_OLI2",
}

#: Physical band -> Planetary Computer asset key. The archive names assets by
#: wavelength rather than by the sensor's own band number, so this is the one
#: place the two naming schemes meet.
ASSETS_BY_SENSOR = {
    "LANDSAT5_TM": {"red": "red", "nir": "nir08"},
    "LANDSAT7_ETM": {"red": "red", "nir": "nir08"},
    "LANDSAT8_OLI": {"red": "red", "nir": "nir08"},
    "LANDSAT9_OLI2": {"red": "red", "nir": "nir08"},
}
QA_ASSETS = {"qa": "qa_pixel", "saturation": "qa_radsat"}


class StacError(RuntimeError):
    """Raised when the archive cannot be searched or read."""


@dataclass
class StacItem:
    """One archive scene, reduced to what the acquisition needs."""
    item_id: str
    datetime: str
    platform: str
    sensor: str
    cloud_cover: float
    assets: Dict[str, str] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def date(self) -> str:
        return self.datetime[:10]

    def describe(self) -> dict:
        return {"item_id": self.item_id, "date": self.date,
                "platform": self.platform, "sensor": self.sensor,
                "cloud_cover": self.cloud_cover,
                "wrs_path": self.properties.get("landsat:wrs_path"),
                "wrs_row": self.properties.get("landsat:wrs_row")}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def _request(url: str, payload: Optional[dict] = None, *, retries: int = 4,
             timeout: int = 90) -> dict:
    """GET or POST JSON, with backoff. A transient failure is not a result."""
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode() if payload is not None
                else None,
                headers={"Content-Type": "application/json",
                         "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except Exception as error:                       # pragma: no cover
            last = error
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    raise StacError(f"could not reach {url}: {last}")


_TOKENS: Dict[str, Dict[str, Any]] = {}


def sign_href(href: str) -> str:
    """Attach an anonymously-issued read token to a Planetary Computer URL.

    Tokens are per storage container and expire, so they are cached and
    refreshed a minute before expiry rather than requested per asset - a
    scene needs four assets and a full run needs thousands.
    """
    if "blob.core.windows.net" not in href:
        return href
    try:
        remainder = href.split("blob.core.windows.net/", 1)
        account = remainder[0].split("//", 1)[1].rstrip(".")
        container = remainder[1].split("/", 1)[0]
    except (IndexError, ValueError):                     # pragma: no cover
        return href
    key = f"{account}/{container}"

    entry = _TOKENS.get(key)
    if entry is None or time.time() > entry["expires_at"] - 60:
        payload = _request(f"{PLANETARY_COMPUTER['sas']}/{key}")
        expiry = payload.get("msft:expiry", "")
        try:
            import datetime as dt
            expires_at = dt.datetime.fromisoformat(
                expiry.replace("Z", "+00:00")).timestamp()
        except Exception:                                # pragma: no cover
            expires_at = time.time() + 1800
        entry = {"token": payload["token"], "expires_at": expires_at}
        _TOKENS[key] = entry
    separator = "&" if "?" in href else "?"
    return f"{href}{separator}{entry['token']}"


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def search_landsat(area: StudyArea, *, start_year: int, end_year: int,
                   window_start: str = "10-15", window_end: str = "12-31",
                   max_cloud: float = 80.0,
                   platforms: Optional[Sequence[str]] = None,
                   per_year: int = 0, logger=None) -> List[StacItem]:
    """Find real scenes intersecting the study area, one search per year.

    Searching per year rather than once keeps every year's result set below
    the API page limit, so a busy year cannot crowd out a sparse one - which
    would silently bias the record toward the well-observed recent decade.

    `per_year` caps how many scenes each year contributes, selecting the
    LEAST CLOUDY first. The cap exists because a full-record download is not
    feasible in one session; it is applied identically to every year, and the
    number actually taken is recorded per year so the sampling is auditable.
    """
    west, south, east, north = area.bounds_in("EPSG:4326")
    wanted = set(platforms) if platforms else None
    items: List[StacItem] = []
    per_year_counts: Dict[int, int] = {}

    for year in range(int(start_year), int(end_year) + 1):
        payload = {
            "collections": ["landsat-c2-l2"],
            "bbox": [west, south, east, north],
            "datetime": (f"{year}-{window_start}T00:00:00Z/"
                         f"{year}-{window_end}T23:59:59Z"),
            "query": {"eo:cloud_cover": {"lt": float(max_cloud)}},
            "limit": 250,
        }
        result = _request(PLANETARY_COMPUTER["search"], payload)
        found = []
        for feature in result.get("features", []):
            properties = feature.get("properties", {})
            platform = properties.get("platform", "")
            sensor = PLATFORM_TO_SENSOR.get(platform)
            if sensor is None:
                continue
            if wanted and sensor not in wanted:
                continue
            assets = feature.get("assets", {})
            needed = dict(ASSETS_BY_SENSOR[sensor])
            hrefs = {}
            missing = False
            for physical, asset_key in needed.items():
                if asset_key not in assets:
                    missing = True
                    break
                hrefs[physical] = assets[asset_key]["href"]
            for role, asset_key in QA_ASSETS.items():
                if asset_key in assets:
                    hrefs[role] = assets[asset_key]["href"]
            if missing or "qa" not in hrefs:
                # A scene without a usable cloud mask is not usable data.
                continue
            found.append(StacItem(
                item_id=feature["id"],
                datetime=properties.get("datetime", f"{year}-11-15T00:00:00Z"),
                platform=platform, sensor=sensor,
                cloud_cover=float(properties.get("eo:cloud_cover", float("nan"))),
                assets=hrefs, properties=properties))

        found.sort(key=lambda item: (item.cloud_cover
                                     if np.isfinite(item.cloud_cover)
                                     else 999.0, item.item_id))
        if per_year > 0:
            found = found[:int(per_year)]
        per_year_counts[year] = len(found)
        items.extend(found)
        if logger is not None:
            logger.info("  %d: %d scene(s) selected", year, len(found))

    if logger is not None:
        empty = [y for y, n in per_year_counts.items() if n == 0]
        if empty:
            logger.warning("years with NO usable scene: %s", empty)
    return items


# ---------------------------------------------------------------------------
# Windowed reads
# ---------------------------------------------------------------------------
#: GDAL settings for reading COGs over HTTP. Without these, GDAL lists the
#: remote directory on every open and re-fetches headers per read, which
#: dominates the cost of a many-scene run.
GDAL_HTTP_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
    "GDAL_HTTP_MULTIPLEX": "YES",
    "GDAL_HTTP_VERSION": "2",
    "VSI_CACHE": "TRUE",
    "VSI_CACHE_SIZE": str(64 * 1024 * 1024),
    # rasterio's config setter requires this one as an int, not a string.
    "GDAL_CACHEMAX": 512,
    "GDAL_NUM_THREADS": "ALL_CPUS",
}


def _choose_overview(source, target_resolution: float) -> Optional[int]:
    """Index of the coarsest overview still finer than the analysis grid.

    This is the single most important performance decision in M7. A Landsat
    scene is ~7700x7600 at 30 m; warping it to a 300 m grid without selecting
    an overview makes GDAL fetch every full-resolution tile it touches -
    measured at ~155 s per scene, which is 33 hours for the record. Reading
    the 8x overview instead fetches ~1/64 of the bytes.

    "Still finer than the target" matters: picking an overview COARSER than
    the analysis grid would upsample, inventing detail the read never
    retrieved. Returning None (full resolution) is the safe fallback for a
    file without suitable overviews.
    """
    factors = source.overviews(1)
    if not factors:
        return None
    native = abs(source.res[0])
    if native <= 0:
        return None
    # `overviews(1)` returns decimation factors; overview_level is their
    # 0-based index in that list.
    usable = [(index, factor) for index, factor in enumerate(factors)
              if native * factor <= target_resolution]
    if not usable:
        return None
    return max(usable, key=lambda pair: pair[1])[0]


def _read_onto_grid(href: str, grid: GeoRef, *, mark_zero_as_fill: bool,
                    dtype: str = "uint16",
                    native: bool = False) -> np.ndarray:
    """Read one COG asset onto the analysis grid by nearest-neighbour.

    Reflectance and QA are both read with nearest neighbour; the module
    docstring explains why averaging reflectance before masking would be
    wrong and why a bitmask cannot be interpolated at all.

    `mark_zero_as_fill` applies to QA_PIXEL ONLY. Outside the scene footprint
    the warp fills 0, and a zero QA_PIXEL word has neither the fill bit nor
    any reject bit set, so `landsat_qa_mask` would read it as a clear
    observation. Setting bit 0 there makes it fill.

    It must NOT be applied to QA_RADSAT, whose convention is the opposite:
    zero means "no band saturated", i.e. GOOD. Setting it to 1 marks every
    pixel as saturated, which masks the entire record. That is not
    hypothetical - it is the defect this parameter was split out to fix.
    """
    url = f"/vsicurl/{sign_href(href)}"
    target_resolution = max(abs(grid.transform.a), abs(grid.transform.e))
    with rasterio.Env(**GDAL_HTTP_ENV):
        # `native` forces full resolution. It is required whenever the values
        # will be masked and aggregated, because the archive's overviews
        # average reflectance while point-sampling QA - see OVERVIEW_WARNING.
        level = None
        if not native:
            with rasterio.open(url) as probe:
                level = _choose_overview(probe, target_resolution)
        opened = rasterio.open(url, overview_level=level) if level is not None \
            else rasterio.open(url)
        with opened as source:
            with WarpedVRT(source, crs=grid.crs, transform=grid.transform,
                           width=grid.width, height=grid.height,
                           resampling=Resampling.nearest,
                           src_nodata=0, nodata=0) as vrt:
                values = vrt.read(1).astype(dtype)
    if mark_zero_as_fill:
        values[values == 0] = 1
    return values


def _aggregate_valid(values: np.ndarray, usable: np.ndarray,
                     factor: int) -> tuple:
    """Mean of the VALID fine pixels in each coarse cell, and their count.

    Masking happens before averaging, which is the whole point: a mean taken
    across cloudy and clear pixels is not a measurement of anything, and no
    later masking can undo it.
    """
    valid = usable & (values > 0)
    rows = (values.shape[0] // factor) * factor
    cols = (values.shape[1] // factor) * factor
    shape = (rows // factor, factor, cols // factor, factor)
    contribution = np.where(valid[:rows, :cols],
                            values[:rows, :cols].astype("float64"), 0.0)
    total = contribution.reshape(shape).sum(axis=(1, 3))
    count = valid[:rows, :cols].reshape(shape).sum(axis=(1, 3))
    # Sum/count rather than nanmean: an empty cell is expected wherever the
    # scene footprint does not reach, and `nanmean` warns on every one of
    # them. `warnings.catch_warnings` would not help - the filter is global
    # and these run on a thread pool, so one worker can clear another's.
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return mean, count.astype("int32")


def fetch_scene(item: StacItem, grid: GeoRef, cache_dir: Path, *,
                overwrite: bool = False,
                aggregate_factor: int = 1) -> Optional[dict]:
    """Read one scene onto the analysis grid and cache it locally.

    With `aggregate_factor > 1` the scene is read at NATIVE resolution on a
    grid that many times finer, masked there, and averaged down - see
    `AGGREGATION_NOTE` and `OVERVIEW_WARNING`. With a factor of 1 the older
    single-sample behaviour is kept, for reproducing the earlier run.

    Returns a manifest entry in `real_data.SceneRecord` form, or None when
    the scene contributes no usable pixel to the study area (a scene can
    intersect the search box and still miss the boundary, or be entirely
    cloud there).
    """
    if aggregate_factor > 1:
        return _fetch_scene_aggregated(item, grid, Path(cache_dir),
                                       factor=int(aggregate_factor),
                                       overwrite=overwrite)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: cache_dir / f"{item.item_id}_{name}.tif"
             for name in ("red", "nir", "qa", "saturation")}

    if not overwrite and paths["red"].exists() and paths["nir"].exists() \
            and paths["qa"].exists():
        record = {"date": item.date, "sensor": item.sensor,
                  "scene_id": item.item_id,
                  "bands": {"red": str(paths["red"]),
                            "nir": str(paths["nir"])},
                  "qa": str(paths["qa"]),
                  "scene_cloud_cover": item.cloud_cover}
        if paths["saturation"].exists():
            record["saturation"] = str(paths["saturation"])
        return record

    arrays: Dict[str, np.ndarray] = {}
    for name, physical in (("red", "red"), ("nir", "nir"), ("qa", "qa"),
                           ("saturation", "saturation")):
        href = item.assets.get(physical)
        if href is None:
            continue
        try:
            # QA_PIXEL only: see `_read_onto_grid`. QA_RADSAT's zero means
            # "not saturated", so it must be left alone.
            arrays[name] = _read_onto_grid(href, grid,
                                           mark_zero_as_fill=name == "qa")
        except Exception as error:
            if name in ("red", "nir", "qa"):
                raise StacError(
                    f"{item.item_id}: could not read {name}: {error}") from error

    # A scene whose footprint misses the study area comes back all-fill.
    if not np.any(arrays["red"] > 0) or not np.any(arrays["nir"] > 0):
        return None

    profile = {"driver": "GTiff", "height": grid.height, "width": grid.width,
               "count": 1, "dtype": "uint16", "crs": grid.crs,
               "transform": grid.transform, "nodata": 0,
               "compress": "deflate"}
    for name, values in arrays.items():
        with rasterio.open(paths[name], "w", **profile) as target:
            target.write(values, 1)
            target.update_tags(
                SCENE_ID=item.item_id, PLATFORM=item.platform,
                DATETIME=item.datetime, ASSET=name,
                SOURCE=PLANETARY_COMPUTER["provenance"],
                SAMPLING=SUBSAMPLING_NOTE)

    record = {"date": item.date, "sensor": item.sensor,
              "scene_id": item.item_id,
              "bands": {"red": str(paths["red"]), "nir": str(paths["nir"])},
              "qa": str(paths["qa"]),
              "scene_cloud_cover": item.cloud_cover}
    if "saturation" in arrays:
        record["saturation"] = str(paths["saturation"])
    return record


def _fetch_scene_aggregated(item: StacItem, grid: GeoRef, cache_dir: Path, *,
                            factor: int, overwrite: bool) -> Optional[dict]:
    """Native-resolution read, mask at native, then average into cells."""
    from rasterio.transform import Affine

    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: cache_dir / f"{item.item_id}_{name}.tif"
             for name in ("red", "nir", "qa", "valid_count")}
    if not overwrite and all(p.exists() for p in
                             (paths["red"], paths["nir"], paths["qa"])):
        return _manifest_entry(item, paths, aggregated=True)

    fine = GeoRef(grid.crs,
                  grid.transform * Affine.scale(1.0 / factor, 1.0 / factor),
                  grid.height * factor, grid.width * factor)

    try:
        red = _read_onto_grid(item.assets["red"], fine, mark_zero_as_fill=False,
                              native=True)
        nir = _read_onto_grid(item.assets["nir"], fine, mark_zero_as_fill=False,
                              native=True)
        qa = _read_onto_grid(item.assets["qa"], fine, mark_zero_as_fill=True,
                             native=True)
    except Exception as error:
        raise StacError(f"{item.item_id}: native read failed: {error}") from error

    saturation = None
    if item.assets.get("saturation"):
        try:
            saturation = _read_onto_grid(item.assets["saturation"], fine,
                                         mark_zero_as_fill=False, native=True)
        except Exception:                                # pragma: no cover
            saturation = None

    from .sensors import landsat_qa_mask
    usable = landsat_qa_mask(qa, saturation=saturation)
    red_mean, count = _aggregate_valid(red, usable, factor)
    nir_mean, _ = _aggregate_valid(nir, usable, factor)
    if not np.isfinite(red_mean).any() or not np.isfinite(nir_mean).any():
        return None

    profile = {"driver": "GTiff", "height": grid.height, "width": grid.width,
               "count": 1, "dtype": "uint16", "crs": grid.crs,
               "transform": grid.transform, "nodata": 0,
               "compress": "deflate"}
    tags = {"SCENE_ID": item.item_id, "PLATFORM": item.platform,
            "DATETIME": item.datetime,
            "SOURCE": PLANETARY_COMPUTER["provenance"],
            "SAMPLING": AGGREGATION_NOTE,
            "AGGREGATE_FACTOR": str(factor)}

    for name, values in (("red", red_mean), ("nir", nir_mean)):
        stored = np.where(np.isfinite(values), np.rint(values), 0)
        with rasterio.open(paths[name], "w", **profile) as target:
            target.write(np.clip(stored, 0, 65535).astype("uint16"), 1)
            target.update_tags(ASSET=name, **tags)

    # A cell is "clear" when at least one valid native pixel reached it, and
    # "fill" otherwise. The real quality information - how many pixels - is
    # written alongside rather than folded into a flag.
    from .sensors import LANDSAT_QA_BITS
    quality = np.where(count > 0, 1 << LANDSAT_QA_BITS["clear"],
                       1 << LANDSAT_QA_BITS["fill"]).astype("uint16")
    with rasterio.open(paths["qa"], "w", **profile) as target:
        target.write(quality, 1)
        target.update_tags(ASSET="qa", **tags)
    with rasterio.open(paths["valid_count"], "w",
                       **{**profile, "dtype": "uint16"}) as target:
        target.write(np.clip(count, 0, 65535).astype("uint16"), 1)
        target.update_tags(ASSET="valid_count",
                           DESCRIPTION=f"valid native pixels of "
                                       f"{factor * factor} per cell", **tags)
    return _manifest_entry(item, paths, aggregated=True)


def _manifest_entry(item: StacItem, paths: dict, *, aggregated: bool) -> dict:
    record = {"date": item.date, "sensor": item.sensor,
              "scene_id": item.item_id,
              "bands": {"red": str(paths["red"]), "nir": str(paths["nir"])},
              "qa": str(paths["qa"]),
              "scene_cloud_cover": item.cloud_cover}
    if not aggregated and paths.get("saturation") \
            and Path(paths["saturation"]).exists():
        record["saturation"] = str(paths["saturation"])
    return record


def build_scene_cache(items: Sequence[StacItem], grid: GeoRef, raw_dir,
                      *, workers: int = 6, overwrite: bool = False,
                      aggregate_factor: int = 1, logger=None) -> dict:
    """Fetch every scene and write the M6 manifest.

    Failures are recorded, not raised: a 36-year archive read over a public
    service will drop occasional requests, and losing the whole record to one
    timeout would be worse than losing one scene. The count of failures is
    part of the run's provenance, so a run that quietly lost half its scenes
    cannot be mistaken for a complete one.
    """
    raw_dir = Path(raw_dir)
    cache_dir = raw_dir / "scenes"
    raw_dir.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []
    failures: List[dict] = []
    empty: List[str] = []
    done = 0

    with ThreadPoolExecutor(max_workers=max(int(workers), 1)) as pool:
        futures = {pool.submit(fetch_scene, item, grid, cache_dir,
                               overwrite=overwrite,
                               aggregate_factor=aggregate_factor): item
                   for item in items}
        for future in as_completed(futures):
            item = futures[future]
            done += 1
            try:
                record = future.result()
            except Exception as error:
                failures.append({"scene": item.item_id, "error": str(error)})
                if logger is not None:
                    logger.warning("  [%d/%d] FAILED %s: %s", done, len(items),
                                   item.item_id, error)
                continue
            if record is None:
                empty.append(item.item_id)
            else:
                records.append(record)
            if logger is not None and done % 25 == 0:
                logger.info("  [%d/%d] cached %d, empty %d, failed %d",
                            done, len(items), len(records), len(empty),
                            len(failures))

    records.sort(key=lambda r: (r["date"], r["scene_id"]))
    manifest = raw_dir / "scenes.json"
    manifest.write_text(json.dumps({
        "scenes": records,
        "metadata": {
            "synthetic": False,
            "archive": PLANETARY_COMPUTER["name"],
            "collection": "landsat-c2-l2",
            "provenance": PLANETARY_COMPUTER["provenance"],
            "licence": PLANETARY_COMPUTER["licence"],
            "documentation": PLANETARY_COMPUTER["documentation"],
            "sampling": SUBSAMPLING_NOTE,
            "analysis_grid": grid.to_dict(),
            "n_requested": len(items),
            "n_cached": len(records),
            "n_outside_study_area": len(empty),
            "n_failed": len(failures),
            "failures": failures,
            "scenes_outside_study_area": empty,
        }}, indent=2), encoding="utf-8")

    if logger is not None:
        logger.info("scene cache: %d usable, %d outside the area, %d failed",
                    len(records), len(empty), len(failures))
    return {"manifest": manifest, "n_cached": len(records),
            "n_failed": len(failures), "n_empty": len(empty),
            "failures": failures, "records": records}


# ---------------------------------------------------------------------------
# Rainfall
# ---------------------------------------------------------------------------
CHIRPS = {
    "name": "CHIRPS v2.0",
    "product": "global annual precipitation total",
    "base": ("https://data.chc.ucsb.edu/products/CHIRPS-2.0/"
             "global_annual/tifs"),
    "resolution_deg": 0.05,
    "crs": "EPSG:4326",
    "units": "mm",
    "nodata": -9999.0,
    "coverage": "1981 to 2024 (annual files)",
    "documentation": "https://www.chc.ucsb.edu/data/chirps",
    "citation": ("Funk, C. et al. (2015). The climate hazards infrared "
                 "precipitation with stations - a new environmental record "
                 "for monitoring extremes. Scientific Data 2, 150066."),
    "licence": "Public, free to use with attribution.",
}

#: WHY ANNUAL RATHER THAN MONTHLY. CHIRPS monthly files are served gzipped
#: (`.tif.gz`), which defeats HTTP range requests: obtaining one month's
#: 31x21-cell window would require downloading and decompressing the entire
#: global grid, ~6 MB, 420 times over. The annual files are plain GeoTIFFs, so
#: a windowed read fetches only the strips the study area touches (~1.4 s per
#: year). The cost of that choice is that the accumulation period becomes the
#: CALENDAR year rather than the configured hydrological year - see
#: `chirps_accumulation_caveat`.
CHIRPS_ANNUAL_RATIONALE = (
    "CHIRPS monthly products are gzipped, which prevents windowed HTTP reads; "
    "the annual products are plain GeoTIFFs and support them. Annual files "
    "were therefore used, which fixes the rainfall accumulation to the "
    "CALENDAR year.")

CHIRPS_ACCUMULATION_CAVEAT = (
    "Rainfall is the CALENDAR-year total (1 Jan - 31 Dec). The NDVI "
    "composite is the post-monsoon window (15 Oct - 31 Dec) of the same "
    "year, so the great majority of each year's rainfall - the June-"
    "September monsoon - precedes the vegetation observation, which is the "
    "ordering RESTREND assumes. The Oct-Dec portion is contemporaneous with "
    "the composite rather than antecedent. No rainfall falling AFTER the "
    "composite window enters the total, so this is not look-ahead; it is a "
    "slightly wider accumulation than the configured hydrological year.")


def fetch_chirps_annual(area: StudyArea, out_dir, *, start_year: int,
                        end_year: int, pad_cells: int = 3,
                        logger=None) -> dict:
    """Windowed CHIRPS annual totals over the study area, as one cube.

    Writes a multi-band GeoTIFF on the CHIRPS native grid (one band per year)
    plus the rainfall manifest `real_data.preprocess_real_data` expects. The
    reprojection onto the analysis grid is left to M6's
    `alignment.align_to_reference`, so the resampling is checked and recorded
    there rather than done silently here.
    """
    from rasterio.windows import from_bounds

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    west, south, east, north = area.bounds_in("EPSG:4326")
    pad = pad_cells * CHIRPS["resolution_deg"]
    bounds = (west - pad, south - pad, east + pad, north + pad)

    bands: List[np.ndarray] = []
    dates: List[str] = []
    missing: List[int] = []
    transform = None
    profile: Dict[str, Any] = {}

    with rasterio.Env(**GDAL_HTTP_ENV):
        for year in range(int(start_year), int(end_year) + 1):
            url = f"/vsicurl/{CHIRPS['base']}/chirps-v2.0.{year}.tif"
            try:
                with rasterio.open(url) as source:
                    window = from_bounds(*bounds, transform=source.transform)
                    values = source.read(1, window=window).astype("float64")
                    if transform is None:
                        transform = source.window_transform(window)
                        profile = {"crs": source.crs}
            except Exception as error:
                missing.append(year)
                if logger is not None:
                    logger.warning("  CHIRPS %d unavailable: %s", year, error)
                continue
            # CHIRPS marks absent data with -9999 and does not always declare
            # it as NoData in the file header, so it is removed explicitly.
            values[values <= CHIRPS["nodata"] + 1] = np.nan
            if np.isfinite(values).sum() == 0:
                missing.append(year)
                continue
            bands.append(values)
            # Mid-year date: it must fall inside the calendar-year window
            # that `real_data.rainfall_accumulation_windows` builds.
            dates.append(f"{year}-07-01")
            if logger is not None:
                logger.info("  CHIRPS %d: mean %.0f mm", year,
                            float(np.nanmean(values)))

    if not bands:
        raise StacError(
            f"no CHIRPS annual data could be retrieved for "
            f"{start_year}-{end_year}. The product covers "
            f"{CHIRPS['coverage']}.")

    cube = np.stack(bands)
    raster = out_dir / f"chirps_annual_{dates[0][:4]}_{dates[-1][:4]}.tif"
    with rasterio.open(raster, "w", driver="GTiff", height=cube.shape[1],
                       width=cube.shape[2], count=cube.shape[0],
                       dtype="float32", crs=profile["crs"],
                       transform=transform, nodata=CHIRPS["nodata"],
                       compress="deflate") as target:
        for index, (band, date) in enumerate(zip(cube, dates), start=1):
            target.write(np.nan_to_num(band, nan=CHIRPS["nodata"]
                                       ).astype("float32"), index)
            target.set_band_description(index, date)
        target.update_tags(SOURCE=CHIRPS["citation"],
                           PRODUCT=CHIRPS["product"],
                           UNITS=CHIRPS["units"],
                           ACCUMULATION=CHIRPS_ACCUMULATION_CAVEAT)

    manifest = out_dir / "rainfall.json"
    manifest.write_text(json.dumps({
        "file": str(raster),
        "dates": dates,
        "metadata": {
            "synthetic": False,
            "product": CHIRPS["name"] + " " + CHIRPS["product"],
            "units": CHIRPS["units"],
            "native_resolution_deg": CHIRPS["resolution_deg"],
            "crs": CHIRPS["crs"],
            "citation": CHIRPS["citation"],
            "documentation": CHIRPS["documentation"],
            "licence": CHIRPS["licence"],
            "why_annual": CHIRPS_ANNUAL_RATIONALE,
            "accumulation_caveat": CHIRPS_ACCUMULATION_CAVEAT,
            "years_retrieved": [int(d[:4]) for d in dates],
            "years_unavailable": missing,
        }}, indent=2), encoding="utf-8")

    if logger is not None:
        logger.info("CHIRPS: %d year(s) retrieved, %d unavailable -> %s",
                    len(dates), len(missing), raster.name)
    return {"raster": raster, "manifest": manifest, "dates": dates,
            "years_unavailable": missing, "grid": cube.shape}
