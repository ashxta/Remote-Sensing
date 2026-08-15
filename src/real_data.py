"""Real remote-sensing ingestion (M6 Parts 6-13).

This module is the whole of M6's scientific surface. It converts raw
satellite scenes and a precipitation product into exactly the
`StandardizedDataset` that M1-M5 already consume, and it does so without the
analysis knowing anything changed:

    study-area boundary        study_area.StudyArea
      -> scene manifest        SceneRecord (date, sensor, band paths)
      -> reflectance           sensors.apply_scale_factors
      -> quality masking       sensors.landsat_qa_mask
      -> spectral index        sensors.compute_index
      -> cross-sensor scale    sensors.harmonise_ndvi
      -> regular time axis     compositing.composite_observations
      -> rainfall accumulation this module
      -> one grid              alignment.align_to_reference
      -> StandardizedDataset   data_source.StandardizedDataset
      -> [ unchanged M1-M5 pipeline ]

WHAT THIS MODULE REFUSES TO DO
------------------------------
* It does not fabricate observations. A window with no usable scene is NaN
  and is counted; nothing is forward-filled, zero-filled or borrowed from a
  neighbouring year.
* It does not fabricate labels. `truth` is populated only from a reference
  raster that configuration names AND whose provenance is declared
  independent of the NDVI series. Otherwise it stays None and the runner
  reports supervised learning as blocked.
* It does not silently reconcile grids. Rainfall is resampled through its
  geometry, the result is re-checked, and the check raises on failure.
* It does not mix sensors without a harmonisation coefficient.

TWO WAYS IN
-----------
`backend="local"` reads scenes already on disk - the USGS EarthExplorer /
Copernicus download path, which needs no cloud account. `backend="gee"`
drives an Earth Engine export first (see `gee_export.py`) and then reads the
result through exactly this code. The preprocessing is shared, so the
science does not depend on where the pixels came from.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import rasterio

from .alignment import (align_to_reference, check_temporal_alignment,
                        require_alignment)
from .compositing import (CompositeResult, CompositeWindow, as_date,
                          composite_observations, build_windows,
                          describe_temporal_design)
from .config import Config
from .data_source import DataSource, StandardizedDataset
from .geo import GeoRef, write_raster
from .sensors import (apply_scale_factors, compute_index, get_sensor,
                      harmonise_ndvi, landsat_qa_mask, sentinel2_scl_mask)
from .study_area import StudyArea, load_study_area

__all__ = ["RealDataError", "SceneRecord", "RealRemoteSensingSource",
           "resolve_target_grid", "load_manifest", "save_manifest",
           "read_scene_index", "build_index_cube", "rainfall_accumulation_windows",
           "accumulate_rainfall", "preprocess_real_data", "write_cube",
           "manifest_metadata", "REAL_DATA_NOTICE",
           "SYNTHETIC_FIXTURE_NOTICE", "CIRCULAR_LABEL_PROVENANCE"]


#: Written into every real-data experiment directory. The counterpart of
#: `experiment.SYNTHETIC_NOTICE`, and just as blunt.
REAL_DATA_NOTICE = (
    "This directory contains results computed from REAL remote-sensing "
    "observations. Every number traces to the satellite and precipitation "
    "products recorded in metadata/dataset_provenance.json. No synthetic "
    "value was substituted for a missing observation: gaps are NaN and are "
    "counted in the data-quality report.\n")

#: The counterpart, for cubes built from fixture scenes. The ingestion code
#: is identical for real and fixture input - that is the point of a fixture -
#: so the ONLY thing preventing a mislabelling is this marker, which travels
#: with the data from the manifest into the GeoTIFF tags and out again.
SYNTHETIC_FIXTURE_NOTICE = (
    "SYNTHETIC FIXTURE DATA. These cubes were built by the real-data "
    "ingestion path from FABRICATED scenes, to exercise and test that path. "
    "They are not observations of anywhere. No number derived from them "
    "describes any real location, and none may be reported as a research "
    "finding.\n")

#: Label provenances that would make supervised evaluation circular.
CIRCULAR_LABEL_PROVENANCE = (
    "trajectory", "algorithmic", "derived", "pipeline", "pseudo",
    "self", "ndvi", "model")


class RealDataError(ValueError):
    """Raised when real data cannot be ingested honestly."""


# ---------------------------------------------------------------------------
# Target grid
# ---------------------------------------------------------------------------
def _utm_epsg(longitude: float, latitude: float) -> str:
    """EPSG code of the UTM zone containing a point."""
    zone = int(math.floor((longitude + 180.0) / 6.0) % 60) + 1
    return f"EPSG:{(32600 if latitude >= 0 else 32700) + zone}"


def resolve_target_grid(area: StudyArea, cfg) -> tuple:
    """The analysis grid, derived from the boundary and the configuration.

    Returns (georef, description). With `target_crs="auto"` the appropriate
    UTM zone for the study area's centre is chosen, so that a pixel is
    (approximately) square on the ground and the spatial-CV block size, which
    is expressed in pixels, corresponds to a constant ground distance. On a
    geographic grid it would not, and blocks would be wider at the equator
    than at the poles.
    """
    west, south, east, north = area.bounds_in("EPSG:4326")
    requested = str(getattr(cfg, "target_crs", "auto") or "auto")
    if requested.lower() == "auto":
        crs = _utm_epsg((west + east) / 2.0, (south + north) / 2.0)
        reason = ("automatic UTM zone for the study-area centroid; a "
                  "projected grid keeps pixels equal-sized on the ground so "
                  "that spatial-CV blocks are a constant ground distance")
    else:
        crs = requested
        reason = "explicitly configured analysis CRS"

    resolution_m = float(getattr(cfg, "target_resolution_m", 30.0))
    try:
        from rasterio.crs import CRS
        geographic = CRS.from_user_input(crs).is_geographic
    except Exception:                                   # pragma: no cover
        geographic = str(crs).endswith("4326")
    if geographic:
        # Convert metres to degrees at the study area's latitude. Recorded,
        # because a degree grid has latitude-dependent pixel areas and every
        # area statistic downstream must account for it (study_area.
        # pixel_area_km2 does).
        resolution = resolution_m / 111_320.0
        reason += ("; resolution converted from metres to degrees at the "
                   "study-area centroid, so ground pixel size varies with "
                   "latitude")
    else:
        resolution = resolution_m

    georef = area.grid(resolution, crs=crs)
    return georef, {
        "crs": str(crs), "reason": reason,
        "requested_resolution_m": resolution_m,
        "grid_resolution": resolution,
        "is_geographic": bool(geographic),
        "shape": list(georef.shape),
        "bounds": georef.to_dict()["bounds"],
    }


# ---------------------------------------------------------------------------
# Scene manifest
# ---------------------------------------------------------------------------
@dataclass
class SceneRecord:
    """One satellite acquisition, as described by a manifest.

    Bands may be separate files (`bands`) or bands of one file
    (`band_index`), which covers both the USGS per-band GeoTIFF layout and a
    stacked export.
    """
    date: str
    sensor: str
    bands: Dict[str, str] = field(default_factory=dict)
    file: str = ""
    band_index: Dict[str, int] = field(default_factory=dict)
    qa: str = ""
    qa_band_index: int = 0
    saturation: str = ""
    saturation_band_index: int = 0
    scene_cloud_cover: float = float("nan")
    scene_id: str = ""

    def describe(self) -> dict:
        return {"date": self.date, "sensor": self.sensor,
                "scene_id": self.scene_id,
                "scene_cloud_cover": self.scene_cloud_cover}


def manifest_metadata(path) -> dict:
    """The manifest's own `metadata` block.

    This carries provenance that must survive into the finished cubes -
    above all the `synthetic` flag. A test fixture that imitates raw scenes
    exercises exactly the same ingestion code as a genuine download, so the
    ONLY thing that can keep its outputs from being mistaken for real
    observations is a marker that travels with the data. It is read here,
    written into the cube's GeoTIFF tags by `preprocess_real_data`, and read
    back by `RealRemoteSensingSource`, which labels the dataset accordingly.
    """
    target = Path(path)
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    return dict(payload.get("metadata") or {}) if isinstance(payload, dict) \
        else {}


def load_manifest(path) -> List[SceneRecord]:
    """Read a scene manifest (JSON list, or {"scenes": [...]})."""
    target = Path(path)
    if not target.exists():
        raise RealDataError(
            f"scene manifest not found: {target}. Acquisition writes one; "
            "see docs/REAL_DATA_SETUP.md for the format and for how to "
            "produce it from a USGS download or an Earth Engine export.")
    payload = json.loads(target.read_text(encoding="utf-8"))
    records = payload.get("scenes", payload) if isinstance(payload, dict) \
        else payload
    if not isinstance(records, list) or not records:
        raise RealDataError(f"{target} lists no scenes")
    known = {f.name for f in SceneRecord.__dataclass_fields__.values()}
    out = []
    for entry in records:
        unknown = set(entry) - known
        if unknown:
            raise RealDataError(
                f"scene manifest entry has unknown key(s) {sorted(unknown)}; "
                f"valid keys are {sorted(known)}")
        out.append(SceneRecord(**entry))
    return out


def save_manifest(path, records: Sequence[SceneRecord], *,
                  metadata: Optional[dict] = None) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(
        {"scenes": [{k: v for k, v in vars(r).items() if v not in ("", {}, 0)}
                    for r in records],
         "metadata": metadata or {}}, indent=2, default=str))
    return target


# ---------------------------------------------------------------------------
# Scene -> index
# ---------------------------------------------------------------------------
def _read_band(record: SceneRecord, name: str) -> tuple:
    """Read one band of a scene, returning (array, GeoRef)."""
    if name in record.bands:
        path, index = record.bands[name], 1
    elif name in record.band_index and record.file:
        path, index = record.file, int(record.band_index[name])
    else:
        raise RealDataError(
            f"scene {record.scene_id or record.date} provides no {name!r} "
            f"band; it declares files for {sorted(record.bands)} and band "
            f"indices for {sorted(record.band_index)}")
    with rasterio.open(path) as source:
        return source.read(index), GeoRef(source.crs, source.transform,
                                          source.height, source.width)


def _read_optional(path: str, index: int) -> tuple:
    if not path:
        return None, None
    with rasterio.open(path) as source:
        return source.read(max(int(index), 1)), GeoRef(
            source.crs, source.transform, source.height, source.width)


def read_scene_index(record: SceneRecord, cfg, target: GeoRef, *,
                     index_name: Optional[str] = None) -> dict:
    """Reflectance -> QA mask -> index -> harmonised -> target grid.

    Returns the index on the target grid plus the counts needed for the
    quality report. Every step is explicit and every one of them can drop a
    pixel to NaN; nothing repairs a dropped pixel.
    """
    sensor = get_sensor(record.sensor)
    index_name = str(index_name or getattr(cfg, "index", "ndvi")).lower()
    from .sensors import INDEX_DEFINITIONS
    if index_name not in INDEX_DEFINITIONS:
        raise RealDataError(f"unknown index {index_name!r}")
    needed = INDEX_DEFINITIONS[index_name].bands

    reflectance, source_ref = {}, None
    for physical in needed:
        raw, ref = _read_band(record, physical)
        reflectance[physical] = apply_scale_factors(raw, sensor.scale,
                                                    sensor.offset)
        if source_ref is None:
            source_ref = ref
        else:
            require_alignment(source_ref, ref,
                              what=f"{record.scene_id or record.date} band "
                                   f"{physical}")

    # --- quality mask -----------------------------------------------------
    qa_path = record.qa or (record.file if sensor.quality_band
                            in record.band_index else "")
    qa_index = record.qa_band_index or record.band_index.get(
        sensor.quality_band, 0)
    qa, qa_ref = _read_optional(qa_path, qa_index)
    if qa is None:
        raise RealDataError(
            f"scene {record.scene_id or record.date} has no {sensor.quality_band} "
            "band. Compositing without a cloud mask would treat cloud tops "
            "as vegetation; supply the QA band or drop the scene.")
    require_alignment(source_ref, qa_ref,
                      what=f"{record.scene_id or record.date} QA band")

    if sensor.quality_scheme == "landsat_qa_pixel":
        saturation, sat_ref = _read_optional(
            record.saturation, record.saturation_band_index)
        if saturation is not None:
            require_alignment(source_ref, sat_ref,
                              what=f"{record.scene_id or record.date} QA_RADSAT")
        usable = landsat_qa_mask(
            qa, exclude=list(getattr(cfg, "mask_bits", None) or []) or None,
            saturation=saturation if getattr(cfg, "mask_saturated", True)
            else None,
            strict_confidence=bool(getattr(cfg, "strict_cloud_confidence",
                                           False)))
    elif sensor.quality_scheme == "sentinel2_scl":
        usable = sentinel2_scl_mask(qa)
    else:                                               # pragma: no cover
        raise RealDataError(f"unknown quality scheme "
                            f"{sensor.quality_scheme!r} for {sensor.key}")

    masked_by_qa = int((~usable).sum())
    for physical in needed:
        reflectance[physical] = np.where(usable, reflectance[physical], np.nan)

    # --- index and harmonisation -----------------------------------------
    index = compute_index(index_name, reflectance)
    harmonised = index
    if index_name == "ndvi":
        override = (getattr(cfg, "harmonisation_overrides", {}) or {}
                    ).get(sensor.key)
        harmonised = harmonise_ndvi(index, sensor, override=override)
        harmonised = np.where(np.isfinite(harmonised)
                              & (harmonised >= -1.0) & (harmonised <= 1.0),
                              harmonised, np.nan)

    # --- onto the analysis grid ------------------------------------------
    aligned, alignment = align_to_reference(harmonised, source_ref, target,
                                            kind="index")
    # The QA-rejection count must land on the same grid as the values it
    # explains, or the report would attribute a scene's cloud to the wrong
    # pixels.
    rejected_grid, _ = align_to_reference(
        (~usable).astype("float64"), source_ref, target, kind="categorical")

    return {
        "index": aligned,
        "masked": np.nan_to_num(rejected_grid, nan=0.0) > 0.5,
        "n_masked_native": masked_by_qa,
        "n_valid_native": int(np.isfinite(harmonised).sum()),
        "sensor": sensor.key,
        "date": record.date,
        "alignment": alignment,
        "harmonisation": dict(sensor.harmonisation) if sensor.harmonisation
        else None,
    }


def build_index_cube(records: Sequence[SceneRecord], cfg, target: GeoRef,
                     windows: Sequence[CompositeWindow], *,
                     index_name: Optional[str] = None,
                     logger=None) -> CompositeResult:
    """Read every scene, composite onto the regular time axis."""
    if not records:
        raise RealDataError("no scenes to composite")
    limit = float(getattr(cfg, "max_scene_cloud_cover", 100.0))
    kept, skipped = [], []
    for record in records:
        cover = record.scene_cloud_cover
        if np.isfinite(cover) and cover > limit:
            skipped.append({"scene": record.scene_id or record.date,
                            "scene_cloud_cover": float(cover)})
            continue
        kept.append(record)
    if not kept:
        raise RealDataError(
            f"every scene exceeds max_scene_cloud_cover={limit}%; loosen the "
            "prefilter or widen the composite window")

    observations, dates, masked, per_scene = [], [], [], []
    sensors_used: Dict[str, int] = {}
    harmonisation: Dict[str, Any] = {}
    for record in kept:
        result = read_scene_index(record, cfg, target, index_name=index_name)
        observations.append(result["index"])
        masked.append(result["masked"].astype("int32"))
        dates.append(record.date)
        sensors_used[result["sensor"]] = sensors_used.get(result["sensor"], 0) + 1
        harmonisation[result["sensor"]] = result["harmonisation"]
        per_scene.append({"date": record.date, "sensor": result["sensor"],
                          "scene_id": record.scene_id,
                          "n_valid_native": result["n_valid_native"],
                          "n_masked_native": result["n_masked_native"],
                          "resampled": result["alignment"]["resampled"]})
        if logger is not None:
            logger.info("  scene %s %-13s valid %8d masked %8d",
                        record.date, result["sensor"],
                        result["n_valid_native"], result["n_masked_native"])

    composite = composite_observations(
        observations, dates, windows,
        statistic=str(getattr(cfg, "composite_statistic", "median")),
        percentile=float(getattr(cfg, "composite_percentile", 90.0)),
        masked_counts=masked,
        min_observations=int(getattr(cfg, "min_observations_per_composite", 1)),
        metadata={"index": index_name or getattr(cfg, "index", "ndvi"),
                  "sensors": sensors_used,
                  "ndvi_harmonisation": harmonisation,
                  "harmonisation_reference":
                      getattr(cfg, "harmonisation_reference", ""),
                  "scenes_skipped_by_cloud_prefilter": skipped,
                  "max_scene_cloud_cover": limit,
                  "quality_mask": {
                      "bits_excluded": list(getattr(cfg, "mask_bits", []) or []),
                      "saturated_excluded": bool(getattr(cfg, "mask_saturated",
                                                         True)),
                      "strict_cloud_confidence":
                          bool(getattr(cfg, "strict_cloud_confidence", False)),
                      "scheme": "Landsat Collection 2 QA_PIXEL bit flags "
                                "(LSDS-1619)"},
                  "per_scene": per_scene})
    return composite


# ---------------------------------------------------------------------------
# Rainfall (Part 9)
# ---------------------------------------------------------------------------
def rainfall_accumulation_windows(windows: Sequence[CompositeWindow],
                                  accumulation: str = "hydrological_year"
                                  ) -> List[tuple]:
    """The date range whose rainfall is paired with each NDVI composite.

    `hydrological_year` accumulates the 12 months ENDING with the composite
    window. That is the rain that actually grew the vegetation the composite
    observes, and it is what RESTREND's NDVI~rainfall regression assumes.

    `calendar_year` accumulates 1 January to 31 December of the composite's
    year. It reproduces the repository's earlier convention and is retained
    so the two can be compared, but for a post-monsoon composite it includes
    rain that fell AFTER the vegetation was observed, which is a mild form of
    look-ahead in the covariate.

    `window` accumulates only the composite window itself - a short interval
    that carries little of the growing season's water supply.
    """
    import datetime as dt
    key = str(accumulation).lower()
    ranges = []
    for window in windows:
        if key == "hydrological_year":
            start = window.end - dt.timedelta(days=364)
            ranges.append((start, window.end))
        elif key == "calendar_year":
            year = window.year or window.end.year
            ranges.append((dt.date(year, 1, 1), dt.date(year, 12, 31)))
        elif key == "window":
            ranges.append((window.start, window.end))
        else:
            raise RealDataError(
                f"unknown rainfall accumulation {accumulation!r}; expected "
                "'hydrological_year', 'calendar_year' or 'window'")
    return ranges


def accumulate_rainfall(cube: np.ndarray, dates: Sequence[Any],
                        windows: Sequence[CompositeWindow], *,
                        accumulation: str = "hydrological_year",
                        min_coverage: float = 0.9) -> tuple:
    """Sum a precipitation series into one total per composite window.

    Returns (totals, report). A window whose contributing observations cover
    less than `min_coverage` of the expected period yields NaN: a partial
    total is not a smaller total, it is an unknown one, and feeding it to
    RESTREND as if it were a real annual rainfall would bias the regression.
    """
    import datetime as dt

    values = np.asarray(cube, dtype="float64")
    if values.ndim != 3:
        raise RealDataError(f"expected a (T, H, W) rainfall cube, got "
                            f"{values.shape}")
    if values.shape[0] != len(dates):
        raise RealDataError(f"{values.shape[0]} rainfall bands but "
                            f"{len(dates)} dates")
    if np.nanmin(values) < 0:
        raise RealDataError(
            "the precipitation product contains negative values; a rainfall "
            "total cannot be below zero, so this is a NoData or scaling error")

    observed = [as_date(d) for d in dates]
    ranges = rainfall_accumulation_windows(windows, accumulation)
    # Observation spacing determines what "full coverage" means: a daily
    # product contributes ~365 records per year, a monthly product 12.
    spacing = 1
    if len(observed) > 1:
        gaps = [(b - a).days for a, b in zip(observed[:-1], observed[1:])
                if (b - a).days > 0]
        spacing = int(np.median(gaps)) if gaps else 1

    totals = np.full((len(windows), *values.shape[1:]), np.nan)
    coverage, contributing = [], []
    for w, (start, end) in enumerate(ranges):
        members = [i for i, day in enumerate(observed) if start <= day <= end]
        expected = max(((end - start).days + 1) / max(spacing, 1), 1.0)
        share = len(members) / expected
        coverage.append(float(min(share, 1.0)))
        contributing.append(len(members))
        if not members or share < float(min_coverage):
            continue
        block = values[members]
        # A NaN in a precipitation record is a missing observation, not zero
        # rain. If any contributing record is missing at a pixel, its total
        # is unknown.
        totals[w] = np.where(np.isfinite(block).all(axis=0),
                             np.nansum(block, axis=0), np.nan)

    report = {
        "accumulation": accumulation,
        "accumulation_ranges": [{"window": w.label,
                                 "from": s.isoformat(), "to": e.isoformat()}
                                for w, (s, e) in zip(windows, ranges)],
        "observation_spacing_days": spacing,
        "min_coverage_required": float(min_coverage),
        "coverage_per_window": {w.label: c for w, c
                                in zip(windows, coverage)},
        "n_contributing_observations": {w.label: n for w, n
                                        in zip(windows, contributing)},
        "windows_below_coverage": [w.label for w, c in zip(windows, coverage)
                                   if c < float(min_coverage)],
        "missing_policy": ("a window with insufficient temporal coverage, or "
                           "a pixel with any missing contributing record, is "
                           "NaN; partial totals are never reported as totals"),
        "generated": dt.datetime.now().isoformat(timespec="seconds"),
    }
    return totals, report


# ---------------------------------------------------------------------------
# Cube I/O
# ---------------------------------------------------------------------------
def write_cube(path, cube: np.ndarray, georef: GeoRef, *,
               band_names: Sequence[str], description: str = "",
               tags: Optional[dict] = None) -> Path:
    """Write a (T, H, W) cube with its time axis in the band descriptions."""
    written = write_raster(path, cube, georef, dtype="float32",
                           band_names=list(band_names),
                           description=description)
    if tags:
        with rasterio.open(written, "r+") as target:
            target.update_tags(**{k: json.dumps(v, default=str)
                                  for k, v in tags.items()})
    return written


# ---------------------------------------------------------------------------
# The DataSource
# ---------------------------------------------------------------------------
class RealRemoteSensingSource(DataSource):
    """Prepared real cubes -> `StandardizedDataset`.

    Reads the NDVI and rainfall cubes produced by `preprocess_real_data`,
    clips them to the study-area boundary, proves that they share one grid
    and one time axis, attaches reference labels only when they are declared
    independent, and hands the analysis exactly the contract it already
    understands.
    """

    name = "real_remote_sensing"

    def __init__(self, cfg: Config, *, study_area: Optional[StudyArea] = None,
                 ndvi_path: Optional[str] = None,
                 rain_path: Optional[str] = None,
                 metadata: Optional[dict] = None, logger=None):
        self.cfg = cfg
        self.real = cfg.real_data
        self.area = study_area or load_study_area(cfg.study_area)
        # An unset path must stay None. `Path("")` is `Path(".")`, which
        # exists, so an emptiness check written against it would pass and the
        # actionable "run --prepare first" message would never fire.
        self.ndvi_path = _optional_path(ndvi_path or self.real.ndvi_cube)
        self.rain_path = _optional_path(rain_path or self.real.rain_cube)
        self.extra_metadata = dict(metadata or {})
        self.logger = logger

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _read_cube(path: Optional[Path], what: str) -> tuple:
        if path is None or not path.is_file():
            raise RealDataError(
                f"{what} cube not found: "
                f"{path if path is not None else '<not configured>'}. Run "
                "the acquisition/preprocessing step first "
                "(`python run_real_data.py --prepare`), or point "
                f"real_data.{what}_cube at an existing cube.")
        with rasterio.open(path) as source:
            data = source.read().astype("float64")
            nodata = source.nodata
            names = [d for d in (source.descriptions or []) if d]
            georef = GeoRef(source.crs, source.transform, source.height,
                            source.width)
            tags = dict(source.tags())
        if nodata is not None and np.isfinite(nodata):
            data[data == nodata] = np.nan
        return data, georef, names, tags

    def _load_reference_labels(self, georef: GeoRef) -> tuple:
        """Reference labels, or an explicit statement of why there are none."""
        reference = self.real.reference
        if not reference.path:
            return None, {
                "available": False,
                "reason": ("no reference-label raster is configured "
                           "(real_data.reference.path is empty)"),
                "consequence": ("supervised Random Forest and CNN experiments "
                                "are BLOCKED on real data; the statistical "
                                "and unsupervised analysis runs in full"),
                "requirement": (
                    "Independent labels are needed: field observations, "
                    "expert photo-interpretation of high-resolution imagery, "
                    "or a published land-cover / shifting-cultivation "
                    "dataset whose classification scheme, spatial resolution "
                    "and vintage are compatible with this study."),
            }
        provenance = str(reference.provenance or "").lower()
        if not provenance:
            raise RealDataError(
                "real_data.reference.path is set but "
                "real_data.reference.provenance is empty. State where the "
                "labels came from ('field', 'expert_interpretation', "
                "'published_dataset', 'ancillary_landcover'); labels of "
                "unknown origin cannot be shown to be independent of the "
                "features.")
        if any(bad in provenance for bad in CIRCULAR_LABEL_PROVENANCE):
            raise RealDataError(
                f"reference provenance {reference.provenance!r} indicates "
                "labels derived from the vegetation series or from this "
                "pipeline's own output. Scoring a classifier against them "
                "measures whether it can reproduce its own inputs, not "
                "whether it detects degradation. Supply independent labels "
                "or leave reference.path empty and accept that supervised "
                "learning is blocked.")

        path = Path(reference.path)
        if not path.exists():
            raise RealDataError(f"reference-label raster not found: {path}")
        with rasterio.open(path) as source:
            labels = source.read(1)
            label_ref = GeoRef(source.crs, source.transform, source.height,
                               source.width)
            nodata = source.nodata
        from .alignment import check_grid_alignment, reproject_to_grid
        check = check_grid_alignment(georef, label_ref)
        if not check.aligned:
            resampled = reproject_to_grid(
                labels.astype("float64"), label_ref, georef,
                resampling=reference.resampling or "nearest")
            labels = np.where(np.isfinite(resampled), resampled, -1
                              ).astype("int32")
        else:
            labels = labels.astype("int32")
        if nodata is not None:
            labels = np.where(labels == int(nodata), -1, labels)
        present = sorted(int(v) for v in np.unique(labels) if v >= 0)
        return labels, {
            "available": True, "path": str(path),
            "source": reference.source, "provenance": reference.provenance,
            "classes_configured": dict(reference.classes),
            "classes_present": present,
            "degradation_classes": list(reference.degradation_classes),
            "resampling": reference.resampling,
            "independent_validation_set": bool(reference.validation_path),
            "alignment": check.summary(),
            "notes": reference.notes,
        }

    # ---------------------------------------------------------------- load
    def load(self) -> StandardizedDataset:
        ndvi, ndvi_ref, ndvi_bands, ndvi_tags = self._read_cube(
            self.ndvi_path, "ndvi")
        rain, rain_ref, rain_bands, rain_tags = self._read_cube(
            self.rain_path, "rain")

        # 1. one grid --------------------------------------------------
        require_alignment(ndvi_ref, rain_ref, what="rainfall cube")
        # 2. one time axis ---------------------------------------------
        times = ndvi_bands if len(ndvi_bands) == ndvi.shape[0] \
            else [str(i) for i in range(ndvi.shape[0])]
        rain_times = rain_bands if len(rain_bands) == rain.shape[0] \
            else times
        temporal = check_temporal_alignment(times, rain_times)
        if rain.shape != ndvi.shape:
            raise RealDataError(
                f"NDVI cube {ndvi.shape} and rainfall cube {rain.shape} "
                "differ in shape after alignment; this indicates a "
                "preprocessing fault, not a recoverable mismatch")

        # 3. clip to the boundary --------------------------------------
        ndvi, georef, inside = self.area.clip(
            ndvi, ndvi_ref, all_touched=self.cfg.study_area.all_touched)
        rain, _, _ = self.area.clip(
            rain, rain_ref, all_touched=self.cfg.study_area.all_touched)

        truth, label_status = self._load_reference_labels(georef)
        if truth is not None:
            truth = np.where(inside, truth, -1).astype("int32")

        interpolation = {"applied": False,
                         "reason": "real_data.allow_interpolation is False"}
        if self.real.allow_interpolation:
            ndvi, interpolation = self._interpolate(ndvi)

        # Missingness INSIDE the boundary. Over the whole raster it would be
        # dominated by the pixels the polygon excludes, which are geometry
        # rather than missing observations, and would report a good record
        # as a bad one.
        missing = (float(np.isnan(ndvi[:, inside]).mean()) if inside.any()
                   else float("nan"))
        # Provenance is not asserted, it is READ. A cube built from fixture
        # scenes carries a `synthetic` marker in its GeoTIFF tags, and it
        # wins: the loader has no other way to tell the two apart, because
        # the ingestion path is deliberately identical.
        ndvi_provenance = _decode_tags(ndvi_tags).get("provenance", {}) or {}
        synthetic = bool(ndvi_provenance.get("synthetic", False))
        if self.logger is not None:
            self.logger.info(
                "%s dataset: %d steps, grid %dx%d, CRS=%s, %d pixels inside "
                "the boundary, %.1f%% of their NDVI cells missing",
                "SYNTHETIC FIXTURE" if synthetic else "real",
                ndvi.shape[0], *georef.shape, georef.crs, int(inside.sum()),
                100 * missing)
            if synthetic:
                self.logger.warning(
                    "these cubes are SYNTHETIC FIXTURE data; every output of "
                    "this run is development evidence, not a research finding")

        metadata = {
            "source": self.name,
            "description": (
                f"SYNTHETIC FIXTURE cubes over {self.area.name}" if synthetic
                else f"real remote-sensing observations over {self.area.name}"),
            "synthetic": synthetic,
            "study_area": self.area.describe(),
            "ndvi_path": str(self.ndvi_path),
            "rain_path": str(self.rain_path),
            "boundary_clipping": {
                "all_touched": bool(self.cfg.study_area.all_touched),
                "pixels_inside": int(inside.sum()),
                "pixels_outside_set_to_nan": int((~inside).sum())},
            "temporal_alignment": temporal,
            "reference_labels": label_status,
            "interpolation": interpolation,
            "ndvi_cube_tags": _decode_tags(ndvi_tags),
            "rain_cube_tags": _decode_tags(rain_tags),
            "notice": (SYNTHETIC_FIXTURE_NOTICE.strip() if synthetic
                       else REAL_DATA_NOTICE.strip()),
        }
        metadata.update(self.extra_metadata)
        return StandardizedDataset(ndvi=ndvi, rain=rain, georef=georef,
                                   times=list(times), truth=truth,
                                   metadata=metadata)

    def _interpolate(self, cube: np.ndarray) -> tuple:
        """Opt-in short-gap filling, with every filled cell recorded."""
        from .quality import interpolate_gaps
        from .config import QualityConfig
        quality = QualityConfig(
            allow_interpolation=True,
            max_interpolation_gap=int(self.real.max_interpolation_gap))
        shape = cube.shape
        flat = cube.reshape(shape[0], -1)
        before = np.isnan(flat)
        filled, n_filled = interpolate_gaps(flat, quality)
        mask = before & ~np.isnan(filled)
        return filled.reshape(shape), {
            "applied": True,
            "method": "temporal linear interpolation of interior gaps only",
            "max_gap_filled": int(self.real.max_interpolation_gap),
            "n_values_filled": int(n_filled),
            "fraction_of_cells_filled": float(mask.mean()),
            "leading_trailing_gaps": "never extrapolated",
            "caveat": ("interpolated values are not observations; they "
                       "reduce the apparent missingness that quality gating "
                       "would otherwise act on"),
        }


def _optional_path(value) -> Optional[Path]:
    """A configured path, or None when the setting is blank."""
    text = str(value or "").strip()
    return Path(text) if text else None


def _decode_tags(tags: dict) -> dict:
    out = {}
    for key, value in (tags or {}).items():
        try:
            out[key] = json.loads(value)
        except Exception:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Preprocessing driver
# ---------------------------------------------------------------------------
def preprocess_real_data(cfg: Config, *, scene_manifest=None,
                         rainfall_manifest=None, area: Optional[StudyArea] = None,
                         logger=None) -> dict:
    """Scenes + precipitation -> analysis-ready cubes on disk.

    Idempotent: with `reuse_cache` the cubes are rebuilt only when they are
    missing, so a full study area is processed once and every subsequent
    experiment reads the cache. Provenance is written next to the cubes.
    """
    real = cfg.real_data
    area = area or load_study_area(cfg.study_area)
    target, grid_note = resolve_target_grid(area, real)
    windows = build_windows(real.temporal_unit, real.start_year, real.end_year,
                            window_start=real.window_start,
                            window_end=real.window_end)

    composite_dir = Path(real.composite_dir)
    composite_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{area.name}_{real.index}_{real.temporal_unit}_" \
           f"{real.start_year}_{real.end_year}"
    ndvi_path = composite_dir / f"{stem}_ndvi.tif"
    rain_path = composite_dir / f"{stem}_rain.tif"
    counts_path = composite_dir / f"{stem}_valid_observations.tif"
    provenance_path = Path(real.metadata_dir) / f"{stem}_provenance.json"

    if real.reuse_cache and ndvi_path.exists() and rain_path.exists() \
            and provenance_path.exists():
        if logger is not None:
            logger.info("reusing cached cubes: %s", ndvi_path.name)
        record = json.loads(provenance_path.read_text())
        record["reused_cache"] = True
        return record

    scene_manifest = scene_manifest or (Path(real.raw_dir) / "scenes.json")
    rainfall_manifest = rainfall_manifest or (Path(real.raw_dir)
                                              / "rainfall.json")
    records = load_manifest(scene_manifest)
    origin = manifest_metadata(scene_manifest)
    synthetic = bool(origin.get("synthetic", False))
    if synthetic and logger is not None:
        logger.warning("the scene manifest declares SYNTHETIC input; the "
                       "cubes will be marked synthetic and every downstream "
                       "output will be labelled development data")
    if logger is not None:
        logger.info("compositing %d scenes onto %d %s windows, grid %dx%d %s",
                    len(records), len(windows), real.temporal_unit,
                    *target.shape, target.crs)

    composite = build_index_cube(records, real, target, windows,
                                 index_name=real.index, logger=logger)

    rain_cube, rain_dates, rain_ref = _read_rainfall(rainfall_manifest)
    aligned_rain, rain_alignment = align_to_reference(
        rain_cube, rain_ref, target, kind="rainfall",
        resampling=real.rainfall_resampling)
    totals, rain_report = accumulate_rainfall(
        aligned_rain, rain_dates, windows,
        accumulation=real.rainfall_accumulation)

    labels = [w.label for w in windows]
    provenance_tag = {
        "synthetic": synthetic,
        "input_manifest": str(scene_manifest),
        "manifest_metadata": origin,
        "notice": (SYNTHETIC_FIXTURE_NOTICE.strip() if synthetic
                   else REAL_DATA_NOTICE.strip())}
    write_cube(ndvi_path, composite.values, target, band_names=labels,
               description=(("SYNTHETIC FIXTURE - " if synthetic else "")
                            + f"{real.index.upper()} {real.temporal_unit} "
                              f"composites, {real.composite_statistic}"),
               tags={"compositing": composite.metadata,
                     "study_area": area.describe(),
                     "grid": grid_note, "provenance": provenance_tag})
    write_cube(rain_path, totals, target, band_names=labels,
               description=(("SYNTHETIC FIXTURE - " if synthetic else "")
                            + f"{real.rainfall_product} accumulated per "
                              f"{real.rainfall_accumulation}, "
                              f"{real.rainfall_units}"),
               tags={"rainfall": rain_report, "alignment": rain_alignment,
                     "provenance": provenance_tag})
    write_cube(counts_path, composite.n_valid.astype("float64"), target,
               band_names=labels,
               description="usable observations per composite window")

    provenance = {
        "generated_from": ("SYNTHETIC fixture scenes (development/testing)"
                           if synthetic
                           else "real remote-sensing observations"),
        "synthetic": synthetic,
        "input_manifest": str(scene_manifest),
        "manifest_metadata": origin,
        "study_area": area.describe(),
        "analysis_grid": grid_note,
        "temporal_design": describe_temporal_design(real),
        "sensors": [get_sensor(k).describe() for k in real.sensors],
        "index": real.index,
        "compositing": composite.metadata,
        "compositing_summary": composite.summary(),
        "rainfall": {"product": real.rainfall_product,
                     "variable": real.rainfall_variable,
                     "units": real.rainfall_units,
                     **rain_report},
        "rainfall_alignment": rain_alignment,
        "outputs": {"ndvi_cube": str(ndvi_path), "rain_cube": str(rain_path),
                    "valid_observation_cube": str(counts_path)},
        "notice": (SYNTHETIC_FIXTURE_NOTICE.strip() if synthetic
                   else REAL_DATA_NOTICE.strip()),
    }
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(json.dumps(provenance, indent=2, default=str))
    provenance["reused_cache"] = False
    return provenance


def _read_rainfall(manifest) -> tuple:
    """Read a precipitation series described by a manifest.

    The manifest names one multi-band raster plus the observation date of
    every band, which is the layout both a CHIRPS download and an Earth
    Engine export can be arranged into.
    """
    path = Path(manifest)
    if not path.exists():
        raise RealDataError(
            f"rainfall manifest not found: {path}. It must name the "
            "precipitation raster and one date per band; see "
            "docs/REAL_DATA_SETUP.md.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raster = payload.get("file")
    dates = payload.get("dates")
    if not raster or not dates:
        raise RealDataError(f"{path} must supply 'file' and 'dates'")
    with rasterio.open(raster) as source:
        cube = source.read().astype("float64")
        nodata = source.nodata
        georef = GeoRef(source.crs, source.transform, source.height,
                        source.width)
    if nodata is not None and np.isfinite(nodata):
        cube[cube == nodata] = np.nan
    if cube.shape[0] != len(dates):
        raise RealDataError(
            f"{raster} has {cube.shape[0]} bands but the manifest lists "
            f"{len(dates)} dates")
    return cube, dates, georef
