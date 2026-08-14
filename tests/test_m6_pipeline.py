"""End-to-end real-data smoke test and synthetic regression (M6 Parts 17, 30).

Two things are proved here.

PART 17 - the real-data path works end to end. Fabricated raw scenes
(`demo/make_scene_fixture.py`) are pushed through the entire ingestion
chain - band mapping, scale factors, QA bitmask decoding, index
computation, cross-sensor harmonisation, temporal compositing, rainfall
accumulation, cross-CRS reprojection, grid alignment, boundary clipping,
contract validation - and then through the UNCHANGED M1-M5 analysis. The
scenes are fabricated; the code path is the real one. No credential and no
network are involved.

PART 30 - the synthetic development pipeline still works. M6 added a second
DataSource; it must not have disturbed the first.

The fixture scenes carry `"synthetic": true`, and one of the tests below
checks that the flag survives all the way to the finished dataset. If it
ever stopped surviving, fixture output could be mistaken for observations,
which is the single most damaging mistake this phase could make.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

from src.config import Config
from src.data_source import RasterStackSource, load_dataset
from src.experiment import prepare_experiment
from src.real_data import RealRemoteSensingSource, preprocess_real_data

pytestmark = pytest.mark.slow

BOUNDARY = Path(Config().paths.boundaries) / "karbi_anglong_bbox.geojson"


@pytest.fixture(scope="module")
def fixture_scenes(tmp_path_factory):
    """Fabricated raw scenes: the INPUT of the real-data path."""
    from demo.make_scene_fixture import build

    root = tmp_path_factory.mktemp("m6_scenes")
    summary = build(root, start_year=2000, end_year=2015, height=28,
                    width=36, scenes_per_year=2, seed=11)
    return root, summary


@pytest.fixture(scope="module")
def prepared_cubes(fixture_scenes, tmp_path_factory):
    """Run the ingestion once; every test below reads its output."""
    root, _ = fixture_scenes
    out = tmp_path_factory.mktemp("m6_cubes")
    cfg = Config(experiment_name="m6_pipeline_test")
    cfg.study_area.name = "fixture_extent"
    cfg.study_area.boundary = str(BOUNDARY)
    real = cfg.real_data
    real.start_year, real.end_year = 2000, 2015
    real.target_resolution_m = 4000.0
    real.raw_dir = str(root)
    real.composite_dir = str(out / "composites")
    real.metadata_dir = str(out / "metadata")
    real.reuse_cache = False

    provenance = preprocess_real_data(cfg, area=None)
    real.ndvi_cube = provenance["outputs"]["ndvi_cube"]
    real.rain_cube = provenance["outputs"]["rain_cube"]
    real.reuse_cache = True
    cfg.years = list(range(real.start_year, real.end_year + 1))
    cfg.quality.min_valid_obs = 8
    cfg.trend.min_obs = 8
    cfg.cyclicity.min_obs = 8
    return cfg, provenance


# ------------------------------------------------- 1 acquisition/preprocess
def test_the_fixture_supplies_several_sensors_on_irregular_dates(
        fixture_scenes):
    root, summary = fixture_scenes
    scenes = json.loads((root / "scenes.json").read_text())["scenes"]
    assert len(scenes) == summary["n_scenes"] > 20
    sensors = {s["sensor"] for s in scenes}
    assert len(sensors) >= 3, f"only {sensors} - the sensor change is untested"
    dates = {s["date"] for s in scenes}
    assert len(dates) > 20                      # genuinely irregular


def test_no_scene_is_dated_outside_its_mission(fixture_scenes):
    """A Landsat 8 scene in 1995 would test a scenario that cannot occur."""
    import datetime as dt
    from demo.make_scene_fixture import MISSIONS

    root, _ = fixture_scenes
    for scene in json.loads((root / "scenes.json").read_text())["scenes"]:
        first, last = MISSIONS[scene["sensor"]]
        assert first <= dt.date.fromisoformat(scene["date"]) <= last


def test_the_scene_and_rainfall_grids_genuinely_differ(fixture_scenes):
    """If they matched, reprojection and alignment would be untested."""
    root, summary = fixture_scenes
    assert summary["scene_crs"] == "EPSG:32646"
    assert summary["rain_crs"] == "EPSG:4326"


def test_preprocessing_produces_cubes_on_one_grid_and_one_time_axis(
        prepared_cubes):
    cfg, provenance = prepared_cubes
    with rasterio.open(provenance["outputs"]["ndvi_cube"]) as ndvi, \
            rasterio.open(provenance["outputs"]["rain_cube"]) as rain:
        assert ndvi.crs == rain.crs
        assert ndvi.transform == rain.transform
        assert (ndvi.height, ndvi.width) == (rain.height, rain.width)
        assert ndvi.count == rain.count == 16
        assert list(ndvi.descriptions) == list(rain.descriptions)
        assert ndvi.descriptions[0] == "2000"


def test_the_rainfall_was_actually_resampled_onto_the_ndvi_grid(
        prepared_cubes):
    _, provenance = prepared_cubes
    alignment = provenance["rainfall_alignment"]
    assert alignment["resampled"] is True
    assert alignment["method"] == "bilinear"
    assert alignment["alignment_after"]["aligned"] is True
    assert "does not add spatial detail" in \
        alignment["native_resolution_retained"]


def test_provenance_records_the_sensors_masks_and_temporal_design(
        prepared_cubes):
    _, provenance = prepared_cubes
    assert provenance["temporal_design"]["temporal_unit"] == "annual"
    assert provenance["temporal_design"]["compositing_statistic"] == "median"
    assert len(provenance["sensors"]) == 4
    mask = provenance["compositing"]["quality_mask"]
    assert "cloud_shadow" in mask["bits_excluded"]
    assert provenance["compositing"]["sensors"]        # counts per sensor


def test_missing_composites_are_counted_not_filled(prepared_cubes):
    _, provenance = prepared_cubes
    summary = provenance["compositing_summary"]
    assert 0.0 <= summary["missing_fraction"] < 1.0
    assert "gap_filling" in provenance["compositing"]
    assert provenance["compositing"]["gap_filling"].startswith("none")
    assert set(summary["missing_fraction_per_step"]) == \
        {str(y) for y in range(2000, 2016)}


def test_the_cache_is_reused_rather_than_recomputed(prepared_cubes):
    cfg, _ = prepared_cubes
    again = preprocess_real_data(cfg)
    assert again["reused_cache"] is True


# ----------------------------------------------------- 2 the dataset
@pytest.fixture(scope="module")
def real_dataset(prepared_cubes):
    cfg, _ = prepared_cubes
    source = RealRemoteSensingSource(cfg)
    return source.load_validated(cfg, expected_time_steps=len(cfg.years))


def test_the_dataset_satisfies_the_m1_m5_contract(real_dataset):
    dataset, report = real_dataset
    assert dataset.ndvi.shape == dataset.rain.shape
    assert dataset.ndvi.ndim == 3
    assert dataset.ndvi.dtype == np.float64
    assert report["ndvi_out_of_physical_range"] == 0
    assert report["rain_negative_values"] == 0
    assert report["crs_present"] and report["transform_present"]
    assert report["pixels_with_sufficient_observations"] > 0
    assert len(dataset.times) == dataset.n_time


def test_missing_observations_are_nan_and_never_a_sentinel(real_dataset):
    dataset, _ = real_dataset
    assert not (dataset.ndvi == -9999).any()
    assert not np.isinf(dataset.ndvi).any()
    finite = dataset.ndvi[np.isfinite(dataset.ndvi)]
    assert finite.min() >= -1.0 and finite.max() <= 1.0


def test_the_synthetic_marker_survives_the_whole_ingestion(real_dataset):
    """The one guarantee that keeps fixture output from being mislabelled."""
    dataset, _ = real_dataset
    assert dataset.metadata["synthetic"] is True
    assert "SYNTHETIC FIXTURE" in dataset.metadata["notice"]
    assert "not observations of anywhere" in \
        dataset.metadata["notice"].lower()


def test_supervised_learning_is_blocked_because_there_are_no_real_labels(
        real_dataset):
    dataset, _ = real_dataset
    assert dataset.truth is None
    status = dataset.metadata["reference_labels"]
    assert status["available"] is False
    assert "BLOCKED" in status["consequence"]


def test_the_generator_archetypes_are_not_wired_in_as_ground_truth(
        fixture_scenes, real_dataset):
    """They exist on disk; using them as labels would be fabrication."""
    root, summary = fixture_scenes
    assert Path(summary["generator_archetypes"]).exists()
    dataset, _ = real_dataset
    assert dataset.truth is None


# ------------------------------------------- 3 the unchanged M1-M5 pipeline
@pytest.fixture(scope="module")
def real_prepared(prepared_cubes):
    cfg, _ = prepared_cubes
    cfg.research.max_analysis_pixels = 600
    return prepare_experiment(cfg, source=RealRemoteSensingSource(cfg))


def test_the_existing_analysis_runs_unmodified_on_the_real_path(
        real_prepared):
    from src.features import feature_names

    prepared = real_prepared
    assert prepared.n_analysed > 0
    assert len(prepared.features) == prepared.n_analysed
    assert set(feature_names()) <= set(prepared.features.columns)
    assert len(prepared.model_columns) == len(feature_names())


def test_every_estimator_produced_finite_values_somewhere(real_prepared):
    """Real-shaped input must not silently degenerate to all-NaN columns."""
    features = real_prepared.features
    for column in ("sen", "mk_z", "mk_p_value", "restrend",
                   "cyc_enrichment", "disturbance_magnitude",
                   "recovery_fraction"):
        assert np.isfinite(features[column].to_numpy()).any(), \
            f"{column} is entirely non-finite on real-path data"


def test_trajectory_classification_produces_the_configured_classes(
        real_prepared):
    from src.trajectory import TRAJECTORY_CLASSES

    counts = real_prepared.trajectory_summary["counts"]
    assert set(counts) <= set(TRAJECTORY_CLASSES)
    assert sum(counts.values()) == real_prepared.n_analysed


def test_spatial_folds_are_laid_out_on_the_real_grid(real_prepared):
    folds = real_prepared.folds
    assert folds.shape == (real_prepared.n_analysed,)
    assert np.unique(folds).size >= 2


def test_exports_inherit_the_real_georeference(real_prepared, tmp_path):
    from src import geo

    prepared = real_prepared
    path = geo.write_layer(tmp_path / "sen.tif",
                           prepared.features["sen"].to_numpy(),
                           prepared.analysis_grid, prepared.georef,
                           dtype="float32")
    with rasterio.open(path) as raster:
        assert raster.crs == prepared.georef.crs
        assert (raster.height, raster.width) == prepared.georef.shape
        assert raster.nodata is not None
        assert np.allclose(list(raster.transform)[:6],
                           list(prepared.georef.transform)[:6])


def test_the_quality_report_is_complete_and_labelled(real_prepared, tmp_path):
    from src.real_report import (build_quality_report, plot_quality_report,
                                 write_quality_report)

    prepared = real_prepared
    report = build_quality_report(prepared.dataset, prepared.config)
    assert report["synthetic"] is True
    assert "SYNTHETIC FIXTURE" in report["data_status"]
    for section in ("spatial", "temporal", "vegetation", "rainfall",
                    "satellite_quality"):
        assert report[section]
    assert report["spatial"]["analysed_area_km2"] > 0
    assert report["temporal"]["n_time_steps"] == 16
    assert report["vegetation"]["min"] is not None

    written = write_quality_report(report, tmp_path / "dq")
    for name in ("dataset_summary", "missingness", "quality_report"):
        assert written[name].exists() and written[name].stat().st_size > 0
    figures = plot_quality_report(prepared.dataset, report, tmp_path / "fig")
    assert len(figures) == 4
    for figure in figures:
        assert figure.stat().st_size > 1000


def test_area_statistics_use_pixel_geometry_not_a_pixel_count(real_prepared):
    from src.study_area import area_statistics
    from src.trajectory import TRAJECTORY_CODES, trajectory_codes

    prepared = real_prepared
    grid = np.full(prepared.shape, np.nan)
    grid[prepared.analysis_grid] = trajectory_codes(prepared.trajectory_labels)
    table = area_statistics(grid, prepared.georef,
                            class_names={v: k for k, v
                                         in TRAJECTORY_CODES.items()},
                            valid_mask=prepared.analysis_grid)
    assert not table.empty
    assert (table["area_km2"] > 0).all()
    assert np.isclose(table["fraction_of_analysed_area"].sum(), 1.0)
    assert "projected" in table["method"].iloc[0]


# ------------------------------------------------ 4 the M6 runner, end to end
def test_the_runner_completes_and_reports_the_block(prepared_cubes, tmp_path):
    import run_real_data

    cfg, _ = prepared_cubes
    cfg = Config.from_dict(cfg.to_dict())
    cfg.experiment_name = "m6_runner_test"
    cfg.paths.real_results = str(tmp_path / "results")
    cfg.research.max_analysis_pixels = 500
    cfg.cyclicity.n_surrogates = 19

    exp, results = run_real_data.main(cfg)

    assert results["synthetic"] is True
    assert results["supervised"]["status"] == "BLOCKED"
    assert "reference labels" in results["supervised"]["why"]
    circularity = results["supervised"]["why_not_use_the_trajectory_classes"]
    assert "re-derive a deterministic rule from that rule's own inputs" in \
        circularity
    assert "label leakage by construction" in circularity
    assert results["supervised"]["what_would_unblock_it"]
    areas = results["area_statistics"]
    assert areas["analysed_area_km2"] > 0
    # A thinned run must say so: an absolute area from a subsample would
    # otherwise read as the size of the study area.
    assert areas["thinned_run"] is True
    assert areas["analysed_area_km2"] < areas["study_area_extent_km2"]
    assert any("THINNED" in c for c in areas["caveats"])
    assert results["cyclicity_surrogate_test"]["n_surrogates"] == 19
    assert "Periodicity is not jhum" in \
        results["cyclicity_surrogate_test"]["interpretation"]

    # The artefacts a reviewer would ask for.
    for relative in ("metrics/m6_notice.txt", "metrics/supervised_blocked.json",
                     "metrics/area_statistics.json",
                     "metrics/dataset_metadata.json",
                     "metrics/sensor_registry.csv",
                     "data_quality/dataset_summary.json",
                     "data_quality/missingness.csv",
                     "features/feature_matrix.csv",
                     "features/feature_metadata.json",
                     "config/study_area.geojson",
                     "config/real_data_config.json",
                     "predictions/trajectory_category.tif",
                     "predictions/sens_slope.tif",
                     "predictions/restrend_slope.tif"):
        assert (exp.root / relative).exists(), f"missing {relative}"

    notice = (exp.root / "metrics" / "m6_notice.txt").read_text()
    assert "SYNTHETIC FIXTURE" in notice


def test_the_runner_writes_georeferenced_layers_that_open_in_a_gis(
        prepared_cubes, tmp_path):
    import run_real_data

    cfg, _ = prepared_cubes
    cfg = Config.from_dict(cfg.to_dict())
    cfg.experiment_name = "m6_georef_test"
    cfg.paths.real_results = str(tmp_path / "results")
    cfg.research.max_analysis_pixels = 400
    cfg.cyclicity.n_surrogates = 19
    exp, _ = run_real_data.main(cfg)

    with rasterio.open(cfg.real_data.ndvi_cube) as cube:
        expected_crs = cube.crs

    layers = list((exp.root / "predictions").glob("*.tif"))
    assert len(layers) >= 10
    for layer in layers:
        with rasterio.open(layer) as raster:
            assert raster.crs == expected_crs
            assert raster.nodata is not None
            assert raster.transform.b == 0 and raster.transform.d == 0


# ------------------------------------- 5 the synthetic pipeline is intact
def test_the_synthetic_source_still_produces_a_valid_dataset():
    """Part 30: M6 must not have disturbed the development pipeline."""
    cfg = Config()
    if not Path(cfg.paths.ndvi_stack).exists():
        pytest.skip("synthetic stacks not generated")
    dataset, report = load_dataset(cfg)
    assert dataset.metadata["synthetic"] is True
    assert dataset.n_time == len(cfg.years)
    assert dataset.has_labels()
    assert report["pixels_with_sufficient_observations"] > 0


def test_the_synthetic_source_still_reaches_the_analysis_unchanged():
    cfg = Config(experiment_name="m6_synthetic_regression")
    if not Path(cfg.paths.ndvi_stack).exists():
        pytest.skip("synthetic stacks not generated")
    cfg.research.max_analysis_pixels = 800
    cfg.research.samples_per_class = 20
    prepared = prepare_experiment(cfg, source=RasterStackSource.from_config(cfg))
    assert prepared.has_labels
    assert prepared.n_analysed > 0
    assert prepared.sample_mask.sum() > 0
    assert prepared.dataset.metadata["synthetic"] is True


def test_both_sources_satisfy_the_same_contract(real_dataset):
    """One pipeline, two sources: the whole point of the M4 interface."""
    cfg = Config()
    if not Path(cfg.paths.ndvi_stack).exists():
        pytest.skip("synthetic stacks not generated")
    synthetic, _ = load_dataset(cfg)
    real, _ = real_dataset
    for dataset in (synthetic, real):
        assert dataset.ndvi.ndim == 3
        assert dataset.ndvi.shape == dataset.rain.shape
        assert dataset.georef.crs is not None
        assert isinstance(dataset.describe(), dict)
        assert "synthetic" in dataset.metadata
    # ... and they are genuinely different datasets, not the same file twice.
    assert synthetic.georef.crs != real.georef.crs
