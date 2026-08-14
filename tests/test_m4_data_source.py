"""Data-contract and experiment-preparation tests (M4 Parts 2, 11, 14).

The contract is the seam where the synthetic generator will be replaced by a
real Landsat/Sentinel/CHIRPS loader in M6. These tests pin the seam: they
assert what a source must deliver and what the pipeline refuses to accept,
so a real loader can be written against a specification rather than against
whatever the synthetic path happened to do.
"""
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.config import Config
from src.data_source import (REAL_DATA_REQUIREMENTS, DataSource,
                             InMemorySource, RasterStackSource,
                             StandardizedDataset, load_dataset,
                             requirements_table, save_requirements)
from src.dataset import DatasetValidationError
from src.experiment import (PreparedExperiment, prepare_experiment,
                            select_analysis_pixels, select_model_samples)
from src.geo import GeoRef

T, H, W = 36, 10, 12
CRS = "EPSG:4326"
TRANSFORM = from_origin(92.0, 26.0, 0.01, 0.01)


def cube(seed=0, n_classes=4):
    """A small synthetic cube with planted archetypes and labels."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    ndvi = np.zeros((T, H, W))
    truth = np.zeros((H, W), dtype="int16")
    for row in range(H):
        for col in range(W):
            kind = (row * W + col) % n_classes
            if kind == 0:
                series = 0.75 + rng.normal(0, 0.02, T)
            elif kind == 1:
                series = 0.7 - 0.014 * t + rng.normal(0, 0.02, T)
            elif kind == 2:
                series = 0.5 + 0.2 * np.sin(2 * np.pi * t / 6) \
                    + rng.normal(0, 0.02, T)
            else:
                series = np.concatenate([np.full(12, 0.7),
                                         np.linspace(0.3, 0.7, T - 12)]) \
                    + rng.normal(0, 0.02, T)
            ndvi[:, row, col] = np.clip(series, 0.05, 0.95)
            truth[row, col] = kind + 1
    rain = rng.normal(1800, 200, (T, H, W))
    return ndvi, rain, truth


@pytest.fixture
def georef():
    return GeoRef(rasterio.crs.CRS.from_string(CRS), TRANSFORM, H, W)


@pytest.fixture
def dataset(georef):
    ndvi, rain, truth = cube()
    return StandardizedDataset(ndvi=ndvi, rain=rain, georef=georef,
                               times=list(range(1990, 1990 + T)), truth=truth,
                               metadata={"source": "test", "synthetic": True})


@pytest.fixture
def stacks(tmp_path):
    """Real GeoTIFF stacks on disk, as a source must consume them."""
    ndvi, rain, truth = cube(seed=1)
    profile = dict(driver="GTiff", height=H, width=W, count=T,
                   dtype="float32", crs=CRS, transform=TRANSFORM,
                   nodata=-9999.0)
    paths = {}
    for name, data in (("ndvi", ndvi), ("rain", rain)):
        path = tmp_path / f"{name}.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data.astype("float32"))
            for band in range(T):
                dst.set_band_description(band + 1, f"{name}_{1990 + band}")
        paths[name] = path
    truth_path = tmp_path / "truth.tif"
    with rasterio.open(truth_path, "w", **{**profile, "count": 1,
                                           "dtype": "int16",
                                           "nodata": 0}) as dst:
        dst.write(truth, 1)
    paths["truth"] = truth_path
    return paths


# ---------------------------------------------------------------- contract
def test_standardized_dataset_exposes_the_shapes_the_pipeline_needs(dataset):
    assert dataset.n_time == T
    assert dataset.shape == (H, W)
    assert dataset.n_pixels == H * W
    ndvi_flat, rain_flat = dataset.flat()
    assert ndvi_flat.shape == (T, H * W) == rain_flat.shape
    assert dataset.flat_truth().shape == (H * W,)
    assert dataset.has_labels()


def test_valid_dataset_passes_the_contract(dataset):
    report = dataset.validate(Config(), expected_time_steps=T)
    assert report["n_time_steps"] == T
    assert report["has_reference_labels"] is True
    assert report["n_time_labels"] == T


def test_contract_rejects_a_mismatched_time_axis(dataset):
    dataset.times = list(range(5))
    with pytest.raises(DatasetValidationError, match="time axis"):
        dataset.validate(Config())


def test_contract_rejects_labels_on_the_wrong_grid(dataset):
    dataset.truth = np.zeros((H + 1, W), dtype="int16")
    with pytest.raises(DatasetValidationError, match="do not match the grid"):
        dataset.validate(Config())


def test_contract_rejects_a_sentinel_instead_of_nan(dataset):
    """The classic real-data mistake: -9999 left in the array."""
    dataset.ndvi[0, 0, 0] = -9999.0
    with pytest.raises(DatasetValidationError, match="physical range"):
        dataset.validate(Config(), strict=True)


def test_contract_rejects_unscaled_integer_ndvi(dataset):
    """NDVI x 10000 passes every shape check and must still be caught."""
    dataset.ndvi = dataset.ndvi * 10000.0
    with pytest.raises(DatasetValidationError, match="physical range"):
        dataset.validate(Config(), strict=True)


def test_dataset_describe_records_provenance(dataset):
    description = dataset.describe()
    assert description["metadata"]["source"] == "test"
    assert description["georeference"]["crs"] == CRS
    assert len(description["times"]) == T


# ------------------------------------------------------------------ sources
def test_raster_source_reads_stacks_and_honours_nodata(stacks):
    source = RasterStackSource(stacks["ndvi"], stacks["rain"], stacks["truth"])
    dataset = source.load()
    assert dataset.ndvi.shape == (T, H, W)
    assert dataset.has_labels()
    assert dataset.metadata["ndvi_nodata"] == -9999.0
    assert str(dataset.georef.crs) == CRS


def test_raster_source_reads_band_descriptions_as_the_time_axis(stacks):
    source = RasterStackSource(stacks["ndvi"], stacks["rain"])
    dataset = source.load()
    assert len(dataset.times) == T
    assert str(dataset.times[0]).endswith("1990")


def test_raster_source_converts_nodata_to_nan(tmp_path):
    ndvi, rain, _ = cube(seed=2)
    ndvi[0, 0, 0] = -9999.0
    profile = dict(driver="GTiff", height=H, width=W, count=T,
                   dtype="float32", crs=CRS, transform=TRANSFORM,
                   nodata=-9999.0)
    paths = []
    for name, data in (("a", ndvi), ("b", rain)):
        path = tmp_path / f"{name}.tif"
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data.astype("float32"))
        paths.append(path)
    dataset = RasterStackSource(*paths).load()
    assert np.isnan(dataset.ndvi[0, 0, 0]), "NoData must become NaN"


def test_missing_stack_gives_an_actionable_error(tmp_path):
    source = RasterStackSource(tmp_path / "absent.tif", tmp_path / "b.tif")
    with pytest.raises(FileNotFoundError, match="make_synthetic_data"):
        source.load()


def test_source_without_labels_is_supported(georef):
    ndvi, rain, _ = cube(seed=3)
    dataset = InMemorySource(ndvi, rain, georef).load()
    assert not dataset.has_labels()
    assert dataset.flat_truth() is None


def test_load_validated_enforces_the_contract(georef):
    ndvi, rain, _ = cube(seed=4)
    ndvi[0, 0, 0] = np.inf
    source = InMemorySource(ndvi, rain, georef)
    with pytest.raises(DatasetValidationError, match="infinite"):
        source.load_validated(Config())


def test_a_new_source_needs_only_the_load_method(georef):
    """The M6 contract: one subclass, no pipeline changes."""
    ndvi, rain, truth = cube(seed=5)

    class FakeSatelliteSource(DataSource):
        name = "fake_satellite"

        def load(self):
            return StandardizedDataset(
                ndvi=ndvi, rain=rain, georef=georef,
                times=list(range(T)), truth=truth,
                metadata={"source": self.name, "synthetic": True})

    dataset, report = FakeSatelliteSource().load_validated(Config())
    assert dataset.metadata["source"] == "fake_satellite"
    assert report["n_pixels"] == H * W


def test_load_dataset_uses_the_configured_paths(stacks):
    cfg = Config()
    cfg.paths.ndvi_stack = str(stacks["ndvi"])
    cfg.paths.rain_stack = str(stacks["rain"])
    cfg.paths.truth = str(stacks["truth"])
    cfg.years = list(range(T))
    dataset, report = load_dataset(cfg)
    assert dataset.n_time == T
    assert report["n_time_steps"] == T


# ------------------------------------------------------- real-data contract
def test_real_data_requirements_are_documented_and_saveable(tmp_path):
    assert len(REAL_DATA_REQUIREMENTS) >= 8
    for item in REAL_DATA_REQUIREMENTS:
        assert {"id", "requirement", "why"} == set(item)
        assert item["requirement"] and item["why"]
    path = save_requirements(tmp_path / "contract.json")
    saved = json.loads(path.read_text())
    assert saved["contract"] == "StandardizedDataset"
    assert len(saved["requirements"]) == len(requirements_table())


def test_requirements_cover_the_known_real_data_traps():
    ids = {item["id"] for item in REAL_DATA_REQUIREMENTS}
    for required in ("nan_for_missing", "physical_units",
                     "sensor_harmonisation", "study_area_boundary",
                     "independent_reference_labels", "rainfall_alignment"):
        assert required in ids


# ------------------------------------------------------------- preparation
def test_prepare_experiment_returns_aligned_arrays(georef):
    ndvi, rain, truth = cube(seed=6)
    cfg = Config(seed=3)
    cfg.research.samples_per_class = 5
    cfg.research.spatial_cv.block_size = 3
    cfg.research.spatial_cv.n_folds = 3
    cfg.years = list(range(T))
    prepared = prepare_experiment(
        cfg, source=InMemorySource(ndvi, rain, georef, truth=truth,
                                   times=list(range(T))))
    assert isinstance(prepared, PreparedExperiment)
    n = prepared.n_analysed
    for array in (prepared.labels, prepared.folds, prepared.block_row,
                  prepared.block_col, prepared.sample_mask,
                  prepared.trajectory_labels):
        assert len(array) == n, "every per-pixel array must align"
    assert prepared.features.shape[0] == n
    assert prepared.series.shape == (T, n)
    assert prepared.analysis_grid.shape == (H, W)
    assert prepared.sample_columns.size == int(prepared.sample_mask.sum())


def test_prepared_sample_columns_index_the_original_grid(georef):
    ndvi, rain, truth = cube(seed=7)
    cfg = Config(seed=1)
    cfg.research.samples_per_class = 4
    cfg.research.spatial_cv.block_size = 3
    cfg.research.spatial_cv.n_folds = 3
    cfg.years = list(range(T))
    prepared = prepare_experiment(
        cfg, source=InMemorySource(ndvi, rain, georef, truth=truth))
    flat_ndvi, _ = prepared.dataset.flat()
    assert np.allclose(flat_ndvi[:, prepared.sample_columns],
                       prepared.series[:, prepared.sample_mask])
    assert np.array_equal(prepared.dataset.flat_truth()[prepared.sample_columns],
                          prepared.labels[prepared.sample_mask])


def test_prepare_experiment_is_deterministic(georef):
    ndvi, rain, truth = cube(seed=8)
    cfg = Config(seed=11)
    cfg.research.samples_per_class = 4
    cfg.research.spatial_cv.block_size = 3
    cfg.research.spatial_cv.n_folds = 3
    cfg.years = list(range(T))
    source = InMemorySource(ndvi, rain, georef, truth=truth)
    first = prepare_experiment(cfg, source=source)
    second = prepare_experiment(cfg, source=source)
    assert np.array_equal(first.sample_mask, second.sample_mask)
    assert np.array_equal(first.folds, second.folds)
    assert np.array_equal(first.features.to_numpy(),
                          second.features.to_numpy(), equal_nan=True)


def test_prepare_experiment_summary_records_the_run_shape(georef):
    ndvi, rain, truth = cube(seed=9)
    cfg = Config(seed=2)
    cfg.research.samples_per_class = 3
    cfg.research.spatial_cv.block_size = 3
    cfg.research.spatial_cv.n_folds = 3
    cfg.years = list(range(T))
    prepared = prepare_experiment(
        cfg, source=InMemorySource(ndvi, rain, georef, truth=truth))
    summary = prepared.summary()
    assert summary["n_analysed"] == prepared.n_analysed
    assert summary["n_features"] == len(prepared.model_columns)
    assert summary["has_reference_labels"] is True
    assert summary["n_folds"] >= 2


def test_thinned_runs_are_recorded_as_reduced_scale(georef):
    ndvi, rain, truth = cube(seed=10)
    cfg = Config(seed=4)
    cfg.research.samples_per_class = 3
    cfg.research.max_analysis_pixels = 40
    cfg.research.spatial_cv.block_size = 3
    cfg.research.spatial_cv.n_folds = 3
    cfg.years = list(range(T))
    prepared = prepare_experiment(
        cfg, source=InMemorySource(ndvi, rain, georef, truth=truth))
    assert prepared.n_analysed == 40
    assert any("reduced-scale" in note for note in prepared.notes)


def test_analysis_thinning_is_seeded_and_reproducible():
    usable = np.ones(500, bool)
    cfg = Config(seed=17)
    cfg.research.max_analysis_pixels = 100
    first = select_analysis_pixels(usable, cfg)
    second = select_analysis_pixels(usable, cfg)
    assert first.sum() == 100
    assert np.array_equal(first, second)


def test_model_sampling_is_stratified_and_seeded():
    truth = np.repeat([1, 2, 3], 40)
    mask = np.ones(120, bool)
    cfg = Config(seed=9)
    cfg.research.samples_per_class = 10
    selected = select_model_samples(truth, mask, cfg)
    assert selected.sum() == 30
    for value in (1, 2, 3):
        assert (selected & (truth == value)).sum() == 10
    assert np.array_equal(selected, select_model_samples(truth, mask, cfg))


def test_empty_quality_gate_fails_loudly(georef):
    """Constant series pass the data contract but carry no signal to analyse."""
    ndvi, rain, truth = cube(seed=11)
    ndvi[:] = 0.5                        # valid NDVI, zero variance
    cfg = Config()
    cfg.years = list(range(T))
    with pytest.raises(RuntimeError, match="no pixels passed the quality gate"):
        prepare_experiment(cfg, source=InMemorySource(ndvi, rain, georef,
                                                      truth=truth))


def test_too_few_observations_is_caught_by_the_contract_first(georef):
    """A cube with no analysable pixel must fail before feature building."""
    ndvi, rain, truth = cube(seed=12)
    ndvi[3:] = np.nan                    # 3 valid steps, below min_valid_obs
    cfg = Config()
    cfg.years = list(range(T))
    with pytest.raises(DatasetValidationError, match="min_valid_obs"):
        prepare_experiment(cfg, source=InMemorySource(ndvi, rain, georef,
                                                      truth=truth))
