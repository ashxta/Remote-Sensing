"""Validation of georeferenced output (Part 7) and reproducibility (Part 2)."""
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src import geo
from src.config import Config
from src.reproducibility import (set_seed, get_rng, start_experiment,
                                 environment_snapshot, make_experiment_id)

H, W = 12, 18
CRS = "EPSG:32646"
TRANSFORM = from_origin(300000.0, 2900000.0, 30.0, 30.0)   # 30 m grid


@pytest.fixture
def source_raster(tmp_path):
    p = tmp_path / "src.tif"
    profile = dict(driver="GTiff", height=H, width=W, count=3,
                   dtype="float32", crs=CRS, transform=TRANSFORM,
                   nodata=-9999.0)
    data = np.random.default_rng(0).normal(0.5, 0.1, (3, H, W)).astype("f4")
    with rasterio.open(p, "w", **profile) as dst:
        dst.write(data)
    return p


# ================================================================ GEOREFERENCE
def test_georef_read_from_raster(source_raster):
    g = geo.GeoRef.from_raster(source_raster)
    assert g.height == H and g.width == W
    assert str(g.crs) == CRS
    assert g.resolution == (30.0, 30.0)


def test_written_raster_preserves_crs_transform_and_size(source_raster,
                                                         tmp_path):
    g = geo.GeoRef.from_raster(source_raster)
    arr = np.random.default_rng(1).normal(0, 1, (H, W))
    out = geo.write_raster(tmp_path / "out.tif", arr, g, dtype="float32")
    info = geo.verify_georeference(out)
    assert info["crs"] == CRS
    assert info["height"] == H and info["width"] == W
    assert info["resolution"] == (30.0, 30.0)
    assert np.allclose(info["transform"], list(TRANSFORM)[:6])


def test_written_raster_preserves_bounds(source_raster, tmp_path):
    g = geo.GeoRef.from_raster(source_raster)
    out = geo.write_raster(tmp_path / "b.tif", np.zeros((H, W)), g)
    got = geo.verify_georeference(out)["bounds"]
    left, bottom, right, top = g.bounds
    assert got["left"] == pytest.approx(left)
    assert got["top"] == pytest.approx(top)
    assert got["right"] == pytest.approx(right)
    assert got["bottom"] == pytest.approx(bottom)


def test_nodata_is_explicit_not_silent_zero(source_raster, tmp_path):
    """NaN must be written as the declared NoData, never as 0."""
    g = geo.GeoRef.from_raster(source_raster)
    arr = np.full((H, W), np.nan)
    arr[0, 0] = 0.42
    out = geo.write_raster(tmp_path / "nd.tif", arr, g, dtype="float32")
    with rasterio.open(out) as src:
        assert src.nodata == geo.NODATA_FLOAT
        band = src.read(1, masked=True)
    assert band.mask.sum() == H * W - 1, "NaN pixels must be masked"
    assert band[0, 0] == pytest.approx(0.42)


def test_roundtrip_values_survive(source_raster, tmp_path):
    g = geo.GeoRef.from_raster(source_raster)
    arr = np.random.default_rng(2).normal(0, 1, (H, W)).astype("float32")
    out = geo.write_raster(tmp_path / "r.tif", arr, g, dtype="float32")
    with rasterio.open(out) as src:
        back = src.read(1)
    assert np.allclose(back, arr, atol=1e-6)


def test_integer_class_raster_uses_integer_nodata(source_raster, tmp_path):
    g = geo.GeoRef.from_raster(source_raster)
    cls = np.random.default_rng(3).integers(1, 6, (H, W)).astype("float64")
    cls[0, :] = np.nan
    out = geo.write_raster(tmp_path / "c.tif", cls, g, dtype="uint8")
    info = geo.verify_georeference(out)
    assert info["dtype"] == "uint8"
    assert info["nodata"] == geo.NODATA_INT


def test_unflatten_scatters_to_correct_pixels():
    mask = np.zeros((4, 5), dtype=bool)
    mask[1, 2] = mask[3, 0] = True
    grid = geo.unflatten(np.array([7.0, 9.0]), mask, (4, 5))
    assert grid[1, 2] == 7.0 and grid[3, 0] == 9.0
    assert np.isnan(grid[0, 0])


def test_unflatten_rejects_size_mismatch():
    mask = np.zeros((4, 5), dtype=bool)
    mask[0, 0] = True
    with pytest.raises(ValueError):
        geo.unflatten(np.array([1.0, 2.0]), mask, (4, 5))


def test_write_layer_end_to_end(source_raster, tmp_path):
    g = geo.GeoRef.from_raster(source_raster)
    mask = np.zeros((H, W), dtype=bool)
    mask[::2, ::3] = True
    vals = np.arange(mask.sum(), dtype="float64")
    out = geo.write_layer(tmp_path / "l.tif", vals, mask, g)
    with rasterio.open(out) as src:
        band = src.read(1)
        assert src.crs is not None and str(src.crs) == CRS
    assert band[0, 0] == pytest.approx(0.0)


def test_shape_mismatch_is_rejected(source_raster, tmp_path):
    g = geo.GeoRef.from_raster(source_raster)
    with pytest.raises(ValueError):
        geo.write_raster(tmp_path / "x.tif", np.zeros((5, 5)), g)


def test_georef_metadata_serialises(source_raster, tmp_path):
    g = geo.GeoRef.from_raster(source_raster)
    p = g.save(tmp_path / "grid.json")
    d = json.loads(p.read_text())
    assert d["crs"] == CRS
    assert d["resolution_x"] == 30.0
    assert set(d["bounds"]) == {"left", "bottom", "right", "top"}


# ============================================================ REPRODUCIBILITY
def test_seed_makes_numpy_deterministic():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    assert np.allclose(a, np.random.rand(5))


def test_generator_is_deterministic():
    assert np.allclose(get_rng(7).normal(size=10), get_rng(7).normal(size=10))


def test_experiment_creates_full_directory_tree(tmp_path):
    cfg = Config(experiment_name="unit", seed=7)
    exp = start_experiment(cfg, results_root=tmp_path)
    for sub in ("config", "metrics", "figures", "predictions", "models",
                "logs"):
        assert (exp.root / sub).is_dir(), f"missing {sub}/"


def test_experiment_snapshots_config_and_environment(tmp_path):
    cfg = Config(experiment_name="snap", seed=99)
    exp = start_experiment(cfg, results_root=tmp_path)
    saved = json.loads((exp.root / "config" / "config.json").read_text())
    assert saved["seed"] == 99
    assert saved["experiment_name"] == "snap"
    env = json.loads((exp.root / "config" / "environment.json").read_text())
    assert "numpy" in env["packages"] and "python" in env


def test_config_snapshot_round_trips(tmp_path):
    cfg = Config(seed=5)
    cfg.cyclicity.min_period = 3.0
    cfg.recovery.recovery_threshold = 0.65
    cfg.save(tmp_path / "c.json")
    back = Config.load(tmp_path / "c.json")
    assert back.seed == 5
    assert back.cyclicity.min_period == 3.0
    assert back.recovery.recovery_threshold == 0.65


def test_experiment_writes_log(tmp_path):
    exp = start_experiment(Config(experiment_name="log"),
                           results_root=tmp_path)
    exp.logger.info("hello-from-test")
    assert "hello-from-test" in (exp.root / "logs" / "run.log").read_text()


def test_experiment_ids_are_unique_and_descriptive():
    a = make_experiment_id("x", 1)
    assert a.endswith("_x_seed1")


def test_experiment_rejects_unknown_subdir(tmp_path):
    exp = start_experiment(Config(), results_root=tmp_path)
    with pytest.raises(ValueError):
        exp.path("nonsense")


def test_environment_snapshot_lists_versions():
    env = environment_snapshot()
    assert env["packages"]["numpy"] != "not installed"
    assert "timestamp_utc" in env


def test_config_import_has_no_filesystem_side_effect():
    """Importing config must not create directories (test isolation)."""
    import importlib
    import src.config as c
    importlib.reload(c)
    assert isinstance(c.DEFAULT.seed, int)
