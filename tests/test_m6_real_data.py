"""Real-data ingestion and the standardized-data contract (M6 Parts 11-16).

The tests that matter most here are the ones about what the loader REFUSES
to do: invent observations, invent labels, accept labels of unknown origin,
accept labels derived from the pipeline's own output, or quietly reconcile
two grids. Each of those is a way to produce confident nonsense from real
data, so each has a test that would fail if the guard were removed.
"""
import datetime as dt
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.compositing import annual_windows
from src.config import Config
from src.dataset import DatasetValidationError
from src.geo import GeoRef
from src.real_data import (CIRCULAR_LABEL_PROVENANCE, RealDataError,
                           RealRemoteSensingSource, SceneRecord,
                           accumulate_rainfall, build_index_cube,
                           load_manifest, manifest_metadata,
                           rainfall_accumulation_windows, read_scene_index,
                           resolve_target_grid, save_manifest, write_cube)
from src.sensors import LANDSAT_QA_BITS, get_sensor
from src.study_area import StudyArea

WEST, SOUTH, EAST, NORTH = 92.0, 25.0, 92.4, 25.4
AREA = StudyArea.from_bounds(WEST, SOUTH, EAST, NORTH, name="test_extent")


# --------------------------------------------------------------- helpers
def scene_grid():
    return GeoRef(rasterio.crs.CRS.from_epsg(4326),
                  from_origin(WEST, NORTH, 0.02, 0.02), 20, 20)


def write_band(path, array, dtype="uint16", georef=None, nodata=None):
    georef = georef or scene_grid()
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "height": georef.height,
               "width": georef.width, "count": 1, "dtype": dtype,
               "crs": georef.crs, "transform": georef.transform}
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as target:
        target.write(np.asarray(array).astype(dtype), 1)
    return path


def encode(reflectance, sensor):
    raw = np.round((np.asarray(reflectance) - sensor.offset) / sensor.scale)
    return np.clip(raw, 1, 65455).astype("uint16")


def make_scene(tmp_path, date, sensor_key="LANDSAT5_TM", ndvi=0.6,
               cloud_rows=(), fill_rows=()):
    """A scene whose NDVI is known exactly, so the result can be checked."""
    sensor = get_sensor(sensor_key)
    nir = np.full((20, 20), 0.4)
    red = nir * (1 - ndvi) / (1 + ndvi)
    qa = np.full((20, 20), 1 << LANDSAT_QA_BITS["clear"], dtype="uint16")
    for row in cloud_rows:
        qa[row] = 1 << LANDSAT_QA_BITS["cloud"]
    for row in fill_rows:
        qa[row] = 1 << LANDSAT_QA_BITS["fill"]

    stem = tmp_path / "scenes" / f"{sensor_key}_{date}"
    record = SceneRecord(
        date=date, sensor=sensor_key, scene_id=f"{sensor_key}_{date}",
        bands={"red": str(write_band(stem.with_name(f"{stem.name}_red.tif"),
                                     encode(red, sensor))),
               "nir": str(write_band(stem.with_name(f"{stem.name}_nir.tif"),
                                     encode(nir, sensor)))},
        qa=str(write_band(stem.with_name(f"{stem.name}_qa.tif"), qa)))
    return record


# -------------------------------------------------------------- manifests
def test_manifest_round_trips(tmp_path):
    records = [make_scene(tmp_path, "2000-11-01")]
    path = save_manifest(tmp_path / "scenes.json", records,
                         metadata={"synthetic": True})
    restored = load_manifest(path)
    assert len(restored) == 1
    assert restored[0].sensor == "LANDSAT5_TM"
    assert manifest_metadata(path)["synthetic"] is True


def test_a_manifest_typo_is_reported_not_ignored(tmp_path):
    path = tmp_path / "scenes.json"
    path.write_text(json.dumps({"scenes": [
        {"date": "2000-11-01", "sensor": "LANDSAT5_TM", "sensr": "typo"}]}))
    with pytest.raises(RealDataError, match="unknown key"):
        load_manifest(path)


def test_a_missing_manifest_says_where_the_format_is_documented(tmp_path):
    with pytest.raises(RealDataError, match="REAL_DATA_SETUP"):
        load_manifest(tmp_path / "absent.json")


def test_an_empty_manifest_is_refused(tmp_path):
    path = tmp_path / "scenes.json"
    path.write_text(json.dumps({"scenes": []}))
    with pytest.raises(RealDataError, match="lists no scenes"):
        load_manifest(path)


# ------------------------------------------------------------ scene -> NDVI
def test_a_scene_yields_the_ndvi_that_was_encoded_into_it(tmp_path):
    record = make_scene(tmp_path, "2000-11-01", ndvi=0.6)
    result = read_scene_index(record, Config().real_data, scene_grid())
    valid = result["index"][np.isfinite(result["index"])]
    assert np.allclose(valid, 0.6, atol=2e-3)


def test_cloudy_pixels_come_back_nan_not_as_cloud_top_reflectance(tmp_path):
    record = make_scene(tmp_path, "2000-11-01", cloud_rows=(0, 1))
    result = read_scene_index(record, Config().real_data, scene_grid())
    assert np.isnan(result["index"][0]).all()
    assert np.isfinite(result["index"][5]).all()
    assert result["n_masked_native"] == 40


def test_oli_scenes_are_harmonised_onto_the_etm_scale(tmp_path):
    """Otherwise the 2013 instrument change is a step the breakpoint
    detector reports as a real disturbance."""
    cfg = Config().real_data
    etm = read_scene_index(
        make_scene(tmp_path / "a", "2005-11-01", "LANDSAT7_ETM", ndvi=0.6),
        cfg, scene_grid())["index"]
    oli = read_scene_index(
        make_scene(tmp_path / "b", "2015-11-01", "LANDSAT8_OLI", ndvi=0.6),
        cfg, scene_grid())["index"]
    assert np.allclose(np.nanmean(etm), 0.6, atol=2e-3)
    assert np.isclose(np.nanmean(oli), 0.9589 * 0.6 + 0.0029, atol=2e-3)
    assert result_differs(np.nanmean(etm), np.nanmean(oli))


def result_differs(a, b, tolerance=1e-4):
    return abs(a - b) > tolerance


def test_a_scene_without_a_quality_band_is_refused(tmp_path):
    record = make_scene(tmp_path, "2000-11-01")
    record.qa = ""
    with pytest.raises(RealDataError, match="cloud mask"):
        read_scene_index(record, Config().real_data, scene_grid())


def test_a_scene_missing_a_required_band_says_which(tmp_path):
    record = make_scene(tmp_path, "2000-11-01")
    del record.bands["nir"]
    with pytest.raises(RealDataError, match="'nir'"):
        read_scene_index(record, Config().real_data, scene_grid())


def test_bands_on_different_grids_are_refused(tmp_path):
    from src.alignment import AlignmentError
    record = make_scene(tmp_path, "2000-11-01")
    other = GeoRef(rasterio.crs.CRS.from_epsg(4326),
                   from_origin(WEST, NORTH, 0.04, 0.04), 10, 10)
    record.bands["nir"] = str(write_band(
        tmp_path / "odd_nir.tif", np.full((10, 10), 20000), georef=other))
    with pytest.raises(AlignmentError):
        read_scene_index(record, Config().real_data, scene_grid())


# ---------------------------------------------------------- index cubes
def test_compositing_places_scenes_in_the_right_year(tmp_path):
    cfg = Config().real_data
    records = [make_scene(tmp_path / "a", "2000-11-01", ndvi=0.3),
               make_scene(tmp_path / "b", "2001-11-01", ndvi=0.7)]
    cube = build_index_cube(records, cfg, scene_grid(),
                            annual_windows(2000, 2001))
    assert np.isclose(np.nanmean(cube.values[0]), 0.3, atol=2e-3)
    assert np.isclose(np.nanmean(cube.values[1]), 0.7, atol=2e-3)


def test_the_cloud_prefilter_drops_scenes_and_records_which(tmp_path):
    cfg = Config().real_data
    cfg.max_scene_cloud_cover = 50.0
    clear = make_scene(tmp_path / "a", "2000-11-01")
    clear.scene_cloud_cover = 10.0
    cloudy = make_scene(tmp_path / "b", "2000-11-10")
    cloudy.scene_cloud_cover = 90.0
    cube = build_index_cube([clear, cloudy], cfg, scene_grid(),
                            annual_windows(2000, 2000))
    skipped = cube.metadata["scenes_skipped_by_cloud_prefilter"]
    assert len(skipped) == 1 and skipped[0]["scene_cloud_cover"] == 90.0


def test_dropping_every_scene_is_an_error_not_an_empty_cube(tmp_path):
    cfg = Config().real_data
    cfg.max_scene_cloud_cover = 5.0
    record = make_scene(tmp_path, "2000-11-01")
    record.scene_cloud_cover = 90.0
    with pytest.raises(RealDataError, match="max_scene_cloud_cover"):
        build_index_cube([record], cfg, scene_grid(),
                         annual_windows(2000, 2000))


def test_the_cube_records_which_sensors_and_masks_produced_it(tmp_path):
    cfg = Config().real_data
    records = [make_scene(tmp_path / "a", "2000-11-01", "LANDSAT5_TM"),
               make_scene(tmp_path / "b", "2015-11-01", "LANDSAT8_OLI")]
    cube = build_index_cube(records, cfg, scene_grid(),
                            annual_windows(2000, 2015))
    assert cube.metadata["sensors"] == {"LANDSAT5_TM": 1, "LANDSAT8_OLI": 1}
    assert cube.metadata["ndvi_harmonisation"]["LANDSAT8_OLI"]["gain"] == 0.9589
    assert "cloud" in cube.metadata["quality_mask"]["bits_excluded"]
    assert len(cube.metadata["per_scene"]) == 2


# ------------------------------------------------------------- rainfall
def test_hydrological_accumulation_ends_at_the_composite_window():
    windows = annual_windows(2000, 2000)
    (start, end), = rainfall_accumulation_windows(windows,
                                                  "hydrological_year")
    assert end == windows[0].end == dt.date(2000, 12, 31)
    assert start == dt.date(2000, 1, 2)          # 12 months ending at `end`


def test_calendar_accumulation_reproduces_the_earlier_convention():
    (start, end), = rainfall_accumulation_windows(annual_windows(2000, 2000),
                                                  "calendar_year")
    assert (start, end) == (dt.date(2000, 1, 1), dt.date(2000, 12, 31))


def test_an_unknown_accumulation_is_refused():
    with pytest.raises(RealDataError, match="hydrological_year"):
        rainfall_accumulation_windows(annual_windows(2000, 2000), "decade")


def monthly_rain(years, value=100.0, shape=(4, 4)):
    dates, bands = [], []
    for year in years:
        for month in range(1, 13):
            dates.append(dt.date(year, month, 15).isoformat())
            bands.append(np.full(shape, value))
    return np.stack(bands), dates


def test_rainfall_totals_are_the_sum_over_the_accumulation_period():
    cube, dates = monthly_rain([1999, 2000])
    totals, report = accumulate_rainfall(cube, dates,
                                         annual_windows(2000, 2000))
    assert np.allclose(totals[0], 12 * 100.0)
    assert report["observation_spacing_days"] in (28, 29, 30, 31)


def test_a_window_with_too_few_records_is_nan_not_a_partial_total():
    """A partial total is not a smaller total; it is an unknown one."""
    cube, dates = monthly_rain([2000])
    keep = [i for i, d in enumerate(dates) if int(d[5:7]) <= 4]
    totals, report = accumulate_rainfall(cube[keep], [dates[i] for i in keep],
                                         annual_windows(2000, 2000))
    assert np.isnan(totals[0]).all()
    assert report["windows_below_coverage"] == ["2000"]


def test_a_missing_pixel_in_any_contributing_record_makes_the_total_unknown():
    cube, dates = monthly_rain([1999, 2000])
    inside = dates.index("2000-06-15")      # a record the window does use
    cube[inside, 0, 0] = np.nan
    totals, _ = accumulate_rainfall(cube, dates, annual_windows(2000, 2000))
    assert np.isnan(totals[0, 0, 0])
    assert np.isfinite(totals[0, 1, 1])


def test_negative_precipitation_is_a_nodata_error_not_a_value():
    cube, dates = monthly_rain([2000])
    cube[0, 0, 0] = -9999.0
    with pytest.raises(RealDataError, match="cannot be below zero"):
        accumulate_rainfall(cube, dates, annual_windows(2000, 2000))


def test_the_accumulation_ranges_are_recorded_per_window():
    cube, dates = monthly_rain([1999, 2000, 2001])
    _, report = accumulate_rainfall(cube, dates, annual_windows(2000, 2001))
    assert len(report["accumulation_ranges"]) == 2
    assert report["accumulation_ranges"][0]["window"] == "2000"
    assert "never reported as totals" in report["missing_policy"]


# ---------------------------------------------------------- target grid
def test_auto_crs_selects_the_utm_zone_for_the_study_area():
    cfg = Config().real_data
    grid, note = resolve_target_grid(AREA, cfg)
    assert note["crs"] == "EPSG:32646"          # zone 46N for ~92 E, 25 N
    assert note["is_geographic"] is False
    assert np.allclose(grid.resolution, (30.0, 30.0))
    assert "spatial-CV blocks" in note["reason"]


def test_a_southern_study_area_gets_a_southern_utm_zone():
    southern = StudyArea.from_bounds(20.0, -25.0, 20.4, -24.6, name="south")
    _, note = resolve_target_grid(southern, Config().real_data)
    assert note["crs"].startswith("EPSG:327")


def test_an_explicit_geographic_crs_converts_the_resolution_and_says_so():
    cfg = Config().real_data
    cfg.target_crs = "EPSG:4326"
    grid, note = resolve_target_grid(AREA, cfg)
    assert note["is_geographic"] is True
    assert grid.resolution[0] < 0.001
    assert "varies with latitude" in note["reason"]


# ------------------------------------------------- the DataSource contract
def build_cubes(tmp_path, *, years=range(2000, 2016), synthetic=True,
                shift_rain_grid=False, rain_labels=None):
    """Write NDVI/rainfall cubes on the analysis grid."""
    grid = AREA.grid(0.02, crs="EPSG:4326")
    n = len(list(years))
    labels = [str(y) for y in years]
    rng = np.random.default_rng(3)
    ndvi = np.clip(0.6 - 0.01 * np.arange(n)[:, None, None]
                   + rng.normal(0, 0.02, (n, *grid.shape)), 0, 1)
    rain = 1500 + rng.normal(0, 120, (n, *grid.shape))
    provenance = {"synthetic": synthetic, "notice": "test fixture"}

    ndvi_path = write_cube(tmp_path / "ndvi.tif", ndvi, grid,
                           band_names=labels, tags={"provenance": provenance})
    rain_grid = grid
    if shift_rain_grid:
        rain_grid = GeoRef(grid.crs,
                           from_origin(WEST + 0.01, NORTH, 0.02, 0.02),
                           grid.height, grid.width)
    rain_path = write_cube(tmp_path / "rain.tif", rain, rain_grid,
                           band_names=rain_labels or labels,
                           tags={"provenance": provenance})
    return ndvi_path, rain_path, labels, grid


def make_config(tmp_path, labels, **real_kwargs):
    cfg = Config(experiment_name="m6_test")
    cfg.years = [int(v) for v in labels]
    cfg.study_area.name = "test_extent"
    cfg.study_area.bounds = [WEST, SOUTH, EAST, NORTH]
    for key, value in real_kwargs.items():
        setattr(cfg.real_data, key, value)
    return cfg


def test_the_source_produces_a_valid_standardized_dataset(tmp_path):
    ndvi, rain, labels, _ = build_cubes(tmp_path)
    cfg = make_config(tmp_path, labels, ndvi_cube=str(ndvi),
                      rain_cube=str(rain))
    dataset, report = RealRemoteSensingSource(cfg, study_area=AREA
                                              ).load_validated(cfg)
    assert dataset.ndvi.shape == dataset.rain.shape
    assert dataset.ndvi.dtype == np.float64
    assert dataset.times == labels
    assert dataset.georef.crs is not None
    assert report["n_time_steps"] == len(labels)
    assert -1.0 <= report["ndvi_min"] and report["ndvi_max"] <= 1.0
    assert report["rain_min"] >= 0


def test_a_rainfall_cube_on_a_different_grid_is_refused(tmp_path):
    """The failure that would pair a pixel with its neighbour's rainfall."""
    from src.alignment import AlignmentError
    ndvi, rain, labels, _ = build_cubes(tmp_path, shift_rain_grid=True)
    cfg = make_config(tmp_path, labels, ndvi_cube=str(ndvi),
                      rain_cube=str(rain))
    with pytest.raises(AlignmentError, match="rainfall cube"):
        RealRemoteSensingSource(cfg, study_area=AREA).load()


def test_a_shifted_rainfall_time_axis_is_refused(tmp_path):
    from src.alignment import AlignmentError
    years = list(range(2000, 2016))
    ndvi, rain, labels, _ = build_cubes(
        tmp_path, years=years, rain_labels=[str(y - 1) for y in years])
    cfg = make_config(tmp_path, labels, ndvi_cube=str(ndvi),
                      rain_cube=str(rain))
    with pytest.raises(AlignmentError, match="lags vegetation"):
        RealRemoteSensingSource(cfg, study_area=AREA).load()


def test_the_boundary_clips_the_cube_and_masks_the_outside(tmp_path):
    ndvi, rain, labels, _ = build_cubes(tmp_path)
    cfg = make_config(tmp_path, labels, ndvi_cube=str(ndvi),
                      rain_cube=str(rain))
    smaller = StudyArea.from_bounds(WEST + 0.1, SOUTH + 0.1, EAST - 0.1,
                                    NORTH - 0.1, name="inner")
    dataset = RealRemoteSensingSource(cfg, study_area=smaller).load()
    clipping = dataset.metadata["boundary_clipping"]
    assert clipping["pixels_inside"] > 0
    assert dataset.shape[0] * dataset.shape[1] < 20 * 20
    assert np.isnan(dataset.ndvi).any()


def test_a_missing_cube_says_how_to_produce_one(tmp_path):
    cfg = make_config(tmp_path, ["2000"], ndvi_cube=str(tmp_path / "no.tif"))
    with pytest.raises(RealDataError, match="--prepare"):
        RealRemoteSensingSource(cfg, study_area=AREA).load()


def test_a_cube_with_the_wrong_number_of_bands_fails_the_contract(tmp_path):
    ndvi, rain, labels, _ = build_cubes(tmp_path)
    cfg = make_config(tmp_path, labels + ["2016"], ndvi_cube=str(ndvi),
                      rain_cube=str(rain))
    source = RealRemoteSensingSource(cfg, study_area=AREA)
    with pytest.raises(DatasetValidationError, match="time steps"):
        source.load_validated(cfg, expected_time_steps=len(cfg.years))


# ------------------------------------------ reference labels (Parts 15-16)
def test_without_configured_labels_supervised_learning_is_declared_blocked(
        tmp_path):
    ndvi, rain, labels, _ = build_cubes(tmp_path)
    cfg = make_config(tmp_path, labels, ndvi_cube=str(ndvi),
                      rain_cube=str(rain))
    dataset = RealRemoteSensingSource(cfg, study_area=AREA).load()
    status = dataset.metadata["reference_labels"]
    assert dataset.truth is None
    assert status["available"] is False
    assert "BLOCKED" in status["consequence"]
    assert "field observations" in status["requirement"]


def write_labels(tmp_path, grid, values=None):
    values = np.ones(grid.shape, dtype="int16") if values is None else values
    path = tmp_path / "labels.tif"
    with rasterio.open(path, "w", driver="GTiff", height=grid.height,
                       width=grid.width, count=1, dtype="int16",
                       crs=grid.crs, transform=grid.transform,
                       nodata=-1) as target:
        target.write(values.astype("int16"), 1)
    return path


@pytest.mark.parametrize("provenance", ["trajectory_classes",
                                        "derived_from_ndvi",
                                        "pipeline_pseudo_labels",
                                        "model_output"])
def test_labels_derived_from_the_pipeline_are_rejected_as_circular(
        tmp_path, provenance):
    """The central Part-16 guard: labels made from the features are not a
    target, they are the features restated."""
    ndvi, rain, labels, grid = build_cubes(tmp_path)
    cfg = make_config(tmp_path, labels, ndvi_cube=str(ndvi),
                      rain_cube=str(rain))
    cfg.real_data.reference.path = str(write_labels(tmp_path, grid))
    cfg.real_data.reference.provenance = provenance
    with pytest.raises(RealDataError, match="reproduce its own inputs"):
        RealRemoteSensingSource(cfg, study_area=AREA).load()
    assert any(bad in provenance for bad in CIRCULAR_LABEL_PROVENANCE)


def test_labels_of_unstated_origin_are_rejected(tmp_path):
    ndvi, rain, labels, grid = build_cubes(tmp_path)
    cfg = make_config(tmp_path, labels, ndvi_cube=str(ndvi),
                      rain_cube=str(rain))
    cfg.real_data.reference.path = str(write_labels(tmp_path, grid))
    cfg.real_data.reference.provenance = ""
    with pytest.raises(RealDataError, match="provenance is empty"):
        RealRemoteSensingSource(cfg, study_area=AREA).load()


def test_independent_labels_are_accepted_and_their_provenance_recorded(
        tmp_path):
    ndvi, rain, labels, grid = build_cubes(tmp_path)
    values = np.ones(grid.shape, dtype="int16")
    values[:10] = 4
    cfg = make_config(tmp_path, labels, ndvi_cube=str(ndvi),
                      rain_cube=str(rain))
    reference = cfg.real_data.reference
    reference.path = str(write_labels(tmp_path, grid, values))
    # Honest provenance for a TEST fixture: it says what it is.
    reference.provenance = "fixture_reference_for_testing"
    reference.source = "synthetic test fixture"
    reference.classes = {1: "stable", 4: "declining"}
    reference.degradation_classes = [4]

    dataset = RealRemoteSensingSource(cfg, study_area=AREA).load()
    assert dataset.truth is not None
    assert dataset.has_labels()
    status = dataset.metadata["reference_labels"]
    assert status["available"] is True
    assert status["provenance"] == "fixture_reference_for_testing"
    assert set(status["classes_present"]) == {1, 4}
    assert status["degradation_classes"] == [4]


def test_a_missing_label_raster_is_an_error(tmp_path):
    ndvi, rain, labels, _ = build_cubes(tmp_path)
    cfg = make_config(tmp_path, labels, ndvi_cube=str(ndvi),
                      rain_cube=str(rain))
    cfg.real_data.reference.path = str(tmp_path / "absent.tif")
    cfg.real_data.reference.provenance = "field"
    with pytest.raises(RealDataError, match="reference-label raster not found"):
        RealRemoteSensingSource(cfg, study_area=AREA).load()


# ------------------------------------------------ provenance and labelling
def test_a_fixture_cube_is_labelled_synthetic_all_the_way_through(tmp_path):
    """The ingestion path is identical for real and fixture input, so the
    marker travelling with the data is the only thing preventing a
    mislabelling."""
    ndvi, rain, labels, _ = build_cubes(tmp_path, synthetic=True)
    cfg = make_config(tmp_path, labels, ndvi_cube=str(ndvi),
                      rain_cube=str(rain))
    dataset = RealRemoteSensingSource(cfg, study_area=AREA).load()
    assert dataset.metadata["synthetic"] is True
    assert "SYNTHETIC FIXTURE" in dataset.metadata["notice"]
    assert "SYNTHETIC FIXTURE" in dataset.metadata["description"]


def test_a_cube_marked_real_is_labelled_real(tmp_path):
    ndvi, rain, labels, _ = build_cubes(tmp_path, synthetic=False)
    cfg = make_config(tmp_path, labels, ndvi_cube=str(ndvi),
                      rain_cube=str(rain))
    dataset = RealRemoteSensingSource(cfg, study_area=AREA).load()
    assert dataset.metadata["synthetic"] is False
    assert "REAL remote-sensing" in dataset.metadata["notice"]


def test_the_dataset_carries_the_study_area_and_the_time_alignment(tmp_path):
    ndvi, rain, labels, _ = build_cubes(tmp_path)
    cfg = make_config(tmp_path, labels, ndvi_cube=str(ndvi),
                      rain_cube=str(rain))
    dataset = RealRemoteSensingSource(cfg, study_area=AREA).load()
    assert dataset.metadata["study_area"]["name"] == "test_extent"
    assert dataset.metadata["temporal_alignment"]["aligned"] is True
    assert dataset.describe()["georeference"]["crs"]


# ------------------------------------------------------- interpolation
def test_interpolation_is_off_by_default(tmp_path):
    ndvi, rain, labels, _ = build_cubes(tmp_path)
    cfg = make_config(tmp_path, labels, ndvi_cube=str(ndvi),
                      rain_cube=str(rain))
    dataset = RealRemoteSensingSource(cfg, study_area=AREA).load()
    assert dataset.metadata["interpolation"]["applied"] is False


def test_enabled_interpolation_records_every_value_it_touched(tmp_path):
    grid = AREA.grid(0.02, crs="EPSG:4326")
    labels = [str(y) for y in range(2000, 2016)]
    ndvi = np.full((len(labels), *grid.shape), 0.5)
    ndvi[5] = np.nan                       # a one-step interior gap
    rain = np.full((len(labels), *grid.shape), 1500.0)
    ndvi_path = write_cube(tmp_path / "ndvi.tif", ndvi, grid,
                           band_names=labels,
                           tags={"provenance": {"synthetic": True}})
    rain_path = write_cube(tmp_path / "rain.tif", rain, grid,
                           band_names=labels)
    cfg = make_config(tmp_path, labels, ndvi_cube=str(ndvi_path),
                      rain_cube=str(rain_path), allow_interpolation=True,
                      max_interpolation_gap=2)
    dataset = RealRemoteSensingSource(cfg, study_area=AREA).load()
    record = dataset.metadata["interpolation"]
    assert record["applied"] is True
    assert record["n_values_filled"] > 0
    assert record["leading_trailing_gaps"] == "never extrapolated"
    assert "not observations" in record["caveat"]
    assert np.isfinite(dataset.ndvi[5]).any()
