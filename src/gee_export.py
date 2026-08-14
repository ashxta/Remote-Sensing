"""Google Earth Engine acquisition (M6 Part 4).

WHY EARTH ENGINE
----------------
A 36-year, 30 m record over a district is roughly 1500 Landsat scenes and
several hundred gigabytes if downloaded raw. Earth Engine applies the QA
mask, computes the index and composites server-side, so what crosses the
network is the finished cube. It is also the more REPRODUCIBLE path: the
collection ids below pin exact archive versions, so a reader with an account
re-runs this script and obtains the same pixels. The alternative -
"we downloaded some scenes" - is not reproducible.

The local path (`backend="local"`) remains fully supported for users without
an Earth Engine account: download Collection-2 Level-2 scenes from USGS
EarthExplorer, write a manifest, and `real_data.preprocess_real_data` runs
exactly the same preprocessing on them. THE SCIENCE IS IDENTICAL; only the
place the arithmetic happens differs.

CREDENTIALS
-----------
Nothing here reads, writes, embeds or logs a credential. Authentication is
the user's, performed once with `earthengine authenticate`, stored by the
Earth Engine client outside this repository, and the storage locations are
in `.gitignore` as a second line of defence. `project` is a Cloud project
id, which is not a secret.

STATUS IN THIS REPOSITORY
-------------------------
`earthengine-api` is NOT installed in the development environment and no
credential exists here, so THIS MODULE HAS NOT BEEN EXECUTED AGAINST THE
LIVE SERVICE. Its request-building logic is unit-tested against a recording
stub (`tests/test_m6_gee.py`), which proves the collections, masks, bands,
harmonisation and date windows it would request are the intended ones - but
that is not the same as a successful export, and this file does not claim
otherwise. See docs/REAL_DATA_SETUP.md.
"""
from __future__ import annotations

from .compositing import build_windows
from .sensors import LANDSAT_QA_BITS, get_sensor
from .study_area import StudyArea

__all__ = ["GEEError", "initialize", "ee_available", "build_export_plan",
           "masked_index_collection", "export_composites", "AUTH_INSTRUCTIONS"]

AUTH_INSTRUCTIONS = """\
Earth Engine access is not configured. One-time setup:

  1. Create a (free, non-commercial) Earth Engine account and register a
     Cloud project:            https://code.earthengine.google.com/register
  2. pip install earthengine-api
  3. earthengine authenticate
  4. Set the project id in configuration:  real_data.gee_project

The credential is stored by the Earth Engine client in your user profile
(~/.config/earthengine on Linux/macOS, %USERPROFILE%\\.config\\earthengine on
Windows). It must never be copied into this repository; those paths are in
.gitignore.

No credential is needed for the local backend: download Collection-2
Level-2 scenes from USGS EarthExplorer and set real_data.backend="local".
"""


class GEEError(RuntimeError):
    """Raised when Earth Engine is unavailable or a request is invalid."""


def ee_available() -> bool:
    try:
        import ee                                        # noqa: F401
        return True
    except ImportError:
        return False


def initialize(project: str = "", *, ee_module=None):
    """Import and initialise Earth Engine, with an actionable failure.

    `ee_module` exists so the request-building logic can be tested against a
    stub without the package or a network.
    """
    module = ee_module
    if module is None:
        try:
            import ee as module                          # type: ignore
        except ImportError as error:
            raise GEEError(
                "earthengine-api is not installed.\n\n" + AUTH_INSTRUCTIONS
            ) from error
    try:
        module.Initialize(project=project) if project else module.Initialize()
    except Exception as error:
        raise GEEError(
            f"Earth Engine could not be initialised: {error}\n\n"
            + AUTH_INSTRUCTIONS) from error
    return module


# ---------------------------------------------------------------------------
def build_export_plan(area: StudyArea, cfg) -> dict:
    """Everything an export would request, as inspectable data.

    Separating the PLAN from the CALL means the parameters can be reviewed,
    saved with a run and unit-tested without an account. A reader can check
    the study area, dates, collections, masks and scale that produced a cube
    without reading Earth Engine code.
    """
    windows = build_windows(cfg.temporal_unit, cfg.start_year, cfg.end_year,
                            window_start=cfg.window_start,
                            window_end=cfg.window_end)
    sensors = [get_sensor(key) for key in cfg.sensors]
    unavailable = [s.key for s in sensors
                   if s.harmonisation is None
                   and s.key not in (cfg.harmonisation_overrides or {})]
    if len(sensors) > 1 and unavailable:
        raise GEEError(
            f"sensors {unavailable} have no NDVI harmonisation coefficients "
            "and would be stacked with sensors that do. Supply "
            "real_data.harmonisation_overrides with a published reference, "
            "or remove them from real_data.sensors.")
    return {
        "study_area": area.describe(),
        "region_geojson": area.to_crs("EPSG:4326").geometry,
        "collections": [
            {"sensor": s.key, "collection": s.collection,
             "red": s.band("red"), "nir": s.band("nir"),
             "quality_band": s.quality_band,
             "scale": s.scale, "offset": s.offset,
             "harmonisation": (cfg.harmonisation_overrides or {}).get(
                 s.key, s.harmonisation)}
            for s in sensors],
        "rainfall_collection": cfg.rainfall_product,
        "rainfall_variable": cfg.rainfall_variable,
        "rainfall_accumulation": cfg.rainfall_accumulation,
        "index": cfg.index,
        "windows": [w.describe() for w in windows],
        "composite_statistic": cfg.composite_statistic,
        "quality_mask": {
            "scheme": "Landsat Collection 2 QA_PIXEL",
            "bits_excluded": list(cfg.mask_bits),
            "bit_positions": {b: LANDSAT_QA_BITS[b] for b in cfg.mask_bits
                              if b in LANDSAT_QA_BITS},
            "saturation_band_excluded": bool(cfg.mask_saturated)},
        "scale_m": float(cfg.target_resolution_m),
        "target_crs": cfg.target_crs,
        "export": {"target": cfg.export_target, "folder": cfg.export_folder,
                   "max_pixels": cfg.max_export_pixels},
        "credentials": "user Earth Engine credential; none stored in this repo",
    }


def masked_index_collection(ee, sensor_key: str, region, cfg):
    """One sensor's masked, harmonised index collection.

    Mirrors `sensors.landsat_qa_mask` and `sensors.compute_index` bit for bit
    - the same QA bits, the same scale factors, the same Roy et al. NDVI
    transform - so the server-side and local paths agree.
    """
    sensor = get_sensor(sensor_key)
    if sensor.quality_scheme != "landsat_qa_pixel":
        raise GEEError(
            f"{sensor_key} uses the {sensor.quality_scheme} quality scheme, "
            "which this export path does not implement; use the local "
            "backend for it.")
    bits = [LANDSAT_QA_BITS[b] for b in cfg.mask_bits if b in LANDSAT_QA_BITS]
    coefficients = (cfg.harmonisation_overrides or {}).get(
        sensor.key, sensor.harmonisation) or {"gain": 1.0, "bias": 0.0}
    red_band, nir_band = sensor.band("red"), sensor.band("nir")
    scale, offset = sensor.scale, sensor.offset
    mask_saturated = bool(cfg.mask_saturated)

    def prepare(image):
        qa = image.select(sensor.quality_band)
        keep = None
        for bit in bits:
            flag = qa.bitwiseAnd(1 << bit).eq(0)
            keep = flag if keep is None else keep.And(flag)
        if mask_saturated and sensor.saturation_band:
            keep = keep.And(image.select(sensor.saturation_band).eq(0))
        image = image.updateMask(keep) if keep is not None else image
        red = image.select(red_band).multiply(scale).add(offset)
        nir = image.select(nir_band).multiply(scale).add(offset)
        index = (nir.subtract(red).divide(nir.add(red))
                 .multiply(float(coefficients["gain"]))
                 .add(float(coefficients["bias"])))
        return (index.rename(cfg.index)
                .updateMask(index.gte(-1).And(index.lte(1)))
                .copyProperties(image, ["system:time_start"]))

    return (ee.ImageCollection(sensor.collection)
            .filterBounds(region)
            .filter(ee.Filter.lte("CLOUD_COVER",
                                  float(cfg.max_scene_cloud_cover)))
            .map(prepare))


def export_composites(area: StudyArea, cfg, *, ee_module=None,
                      dry_run: bool = False, logger=None) -> dict:
    """Start the Earth Engine exports for the configured study.

    With `dry_run` the plan is built and validated but nothing is submitted,
    which is how the configuration is checked before spending export quota.
    """
    plan = build_export_plan(area, cfg)
    if dry_run:
        plan["submitted"] = False
        plan["note"] = "dry run: the plan was validated, nothing was exported"
        return plan

    ee = initialize(cfg.gee_project, ee_module=ee_module)
    region = ee.Geometry(plan["region_geojson"])
    windows = build_windows(cfg.temporal_unit, cfg.start_year, cfg.end_year,
                            window_start=cfg.window_start,
                            window_end=cfg.window_end)

    merged = None
    for key in cfg.sensors:
        collection = masked_index_collection(ee, key, region, cfg)
        merged = collection if merged is None else merged.merge(collection)

    reducer = _reducer(ee, cfg.composite_statistic, cfg.composite_percentile)
    index_bands = [
        (merged.filterDate(str(w.start), str(w.end + _one_day()))
         .reduce(reducer).rename(f"{w.label}"))
        for w in windows]
    index_stack = ee.Image.cat(index_bands).clip(region).toFloat()

    rainfall = ee.ImageCollection(cfg.rainfall_product).filterBounds(region)
    from .real_data import rainfall_accumulation_windows
    ranges = rainfall_accumulation_windows(windows, cfg.rainfall_accumulation)
    rain_bands = [
        (rainfall.select(cfg.rainfall_variable)
         .filterDate(str(start), str(end + _one_day()))
         .sum().rename(f"{w.label}"))
        for w, (start, end) in zip(windows, ranges)]
    rain_stack = ee.Image.cat(rain_bands).clip(region).toFloat()

    tasks = {}
    for name, image in (("ndvi_cube", index_stack), ("rain_cube", rain_stack)):
        task = ee.batch.Export.image.toDrive(
            image=image, description=f"{area.name}_{name}",
            folder=cfg.export_folder, region=region,
            scale=float(cfg.target_resolution_m),
            crs=(cfg.target_crs if cfg.target_crs != "auto" else None),
            maxPixels=float(cfg.max_export_pixels))
        task.start()
        tasks[name] = getattr(task, "id", str(task))
        if logger is not None:
            logger.info("Earth Engine export started: %s -> %s", name,
                        tasks[name])

    plan["submitted"] = True
    plan["tasks"] = tasks
    plan["next_step"] = (
        "When the exports finish, place the GeoTIFFs where "
        "real_data.ndvi_cube / real_data.rain_cube point, then run "
        "`python run_real_data.py`. Band descriptions carry the time axis.")
    return plan


def _one_day():
    import datetime as dt
    return dt.timedelta(days=1)


def _reducer(ee, statistic: str, percentile: float):
    name = str(statistic).lower()
    if name == "median":
        return ee.Reducer.median()
    if name == "mean":
        return ee.Reducer.mean()
    if name == "max":
        return ee.Reducer.max()
    if name == "min":
        return ee.Reducer.min()
    if name == "percentile":
        return ee.Reducer.percentile([float(percentile)])
    raise GEEError(f"unsupported compositing statistic {statistic!r}")
