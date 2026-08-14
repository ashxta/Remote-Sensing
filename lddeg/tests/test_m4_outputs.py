"""Geospatial export, figure and experiment-isolation tests
(M4 Parts 5, 9, 10, 11)."""
import json

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from src import geo
from src import research_figures as RF
from src.config import Config, SpatialCVConfig
from src.quality import FLAG_NAMES
from src.reproducibility import start_experiment
from src.validation import spatial_block_folds

H, W = 8, 10
CRS = "EPSG:4326"
TRANSFORM = from_origin(92.0, 26.0, 0.01, 0.01)
UTM = "EPSG:32646"
UTM_TRANSFORM = from_origin(300000.0, 2900000.0, 30.0, 30.0)


@pytest.fixture
def georef():
    return geo.GeoRef(rasterio.crs.CRS.from_string(CRS), TRANSFORM, H, W)


# --------------------------------------------------------------- GeoJSON
def test_point_geojson_is_valid_and_carries_properties(georef, tmp_path):
    mask = np.zeros((H, W), bool)
    mask[1, 2] = mask[3, 4] = mask[5, 6] = True
    path = geo.write_point_geojson(
        tmp_path / "points.geojson", mask, georef,
        {"prediction": np.array([1, 2, 3]),
         "confidence": np.array([0.9, 0.5, np.nan])},
        description="test points")
    payload = json.loads(path.read_text())
    assert payload["type"] == "FeatureCollection"
    assert len(payload["features"]) == 3
    first = payload["features"][0]
    assert first["geometry"]["type"] == "Point"
    assert len(first["geometry"]["coordinates"]) == 2
    assert first["properties"]["prediction"] == 1
    assert first["properties"]["row"] == 1 and first["properties"]["col"] == 2
    assert payload["features"][2]["properties"]["confidence"] is None, \
        "NaN must be written as null, not as invalid JSON"


def test_point_geojson_coordinates_fall_inside_the_raster_bounds(georef,
                                                                 tmp_path):
    mask = np.zeros((H, W), bool)
    mask[0, 0] = mask[H - 1, W - 1] = True
    payload = json.loads(geo.write_point_geojson(
        tmp_path / "p.geojson", mask, georef, {"v": np.array([1, 2])}
    ).read_text())
    left, bottom, right, top = georef.bounds
    for feature in payload["features"]:
        x, y = feature["geometry"]["coordinates"]
        assert left <= x <= right
        assert bottom <= y <= top


def test_point_geojson_reprojects_a_projected_grid(tmp_path):
    """RFC 7946 requires WGS84 lon/lat, whatever the source CRS is."""
    projected = geo.GeoRef(rasterio.crs.CRS.from_string(UTM), UTM_TRANSFORM,
                           H, W)
    mask = np.zeros((H, W), bool)
    mask[0, 0] = True
    payload = json.loads(geo.write_point_geojson(
        tmp_path / "utm.geojson", mask, projected, {"v": np.array([1])}
    ).read_text())
    assert payload["crs_of_coordinates"] == "EPSG:4326"
    assert payload["source_crs"] == UTM
    x, y = payload["features"][0]["geometry"]["coordinates"]
    assert -180 <= x <= 180 and -90 <= y <= 90


def test_point_geojson_rejects_mismatched_properties(georef, tmp_path):
    mask = np.zeros((H, W), bool)
    mask[0, 0] = mask[1, 1] = True
    with pytest.raises(ValueError, match="selects 2 pixels"):
        geo.write_point_geojson(tmp_path / "bad.geojson", mask, georef,
                                {"v": np.array([1])})


def test_point_geojson_rejects_a_mask_of_the_wrong_shape(georef, tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        geo.write_point_geojson(tmp_path / "bad.geojson",
                                np.ones((H + 1, W), bool), georef, {})


def test_class_geojson_vectorises_contiguous_regions(georef, tmp_path):
    grid = np.zeros((H, W), dtype="int32")
    grid[0:3, 0:3] = 1
    grid[5:7, 5:8] = 2
    payload = json.loads(geo.write_class_geojson(
        tmp_path / "classes.geojson", grid, georef,
        class_names={1: "Stable", 2: "Degrading"}).read_text())
    assert payload["n_polygons"] == 2
    values = {f["properties"]["class_value"] for f in payload["features"]}
    assert values == {1, 2}
    names = {f["properties"]["class_name"] for f in payload["features"]}
    assert names == {"Stable", "Degrading"}
    for feature in payload["features"]:
        assert feature["geometry"]["type"] == "Polygon"
        assert len(feature["geometry"]["coordinates"][0]) >= 4


def test_class_geojson_skips_the_ignored_background(georef, tmp_path):
    grid = np.zeros((H, W), dtype="int32")
    grid[0, 0] = 3
    payload = json.loads(geo.write_class_geojson(
        tmp_path / "c.geojson", grid, georef).read_text())
    assert payload["n_polygons"] == 1
    assert payload["features"][0]["properties"]["class_value"] == 3


def test_class_geojson_caps_output_and_records_the_cap(georef, tmp_path):
    rng = np.random.default_rng(0)
    grid = rng.integers(1, 5, (H, W)).astype("int32")
    payload = json.loads(geo.write_class_geojson(
        tmp_path / "capped.geojson", grid, georef, max_features=3).read_text())
    assert payload["n_polygons"] == 3
    assert payload["n_polygons_dropped_by_cap"] > 0


def test_class_geojson_rejects_a_grid_of_the_wrong_shape(georef, tmp_path):
    with pytest.raises(ValueError, match="does not match"):
        geo.write_class_geojson(tmp_path / "bad.geojson",
                                np.zeros((H + 2, W), "int32"), georef)


# ---------------------------------------------------------------- figures
def test_learning_curves_render_per_fold(tmp_path):
    history = pd.DataFrame([
        {"fold": f, "epoch": e, "train_loss": 1.0 / e,
         "validation_loss": 1.0 / e + 0.05 * f}
        for f in (0, 1) for e in range(1, 8)])
    path = RF.plot_learning_curves(history, tmp_path / "curves.png")
    assert path.exists() and path.stat().st_size > 1000


def test_learning_curves_handle_an_empty_history(tmp_path):
    assert RF.plot_learning_curves(pd.DataFrame(), tmp_path / "e.png") is None


def test_quality_map_renders_every_flag(tmp_path):
    flags = np.arange(H * W) % len(FLAG_NAMES)
    path = RF.plot_quality_map(flags, (H, W), tmp_path / "quality.png")
    assert path.exists() and path.stat().st_size > 1000


def test_uncertainty_map_renders(tmp_path):
    mask = np.ones((H, W), bool)
    values = (np.arange(H * W) % 2).astype(float)
    path = RF.plot_uncertainty_map(values, mask, (H, W),
                                   tmp_path / "uncertain.png")
    assert path.exists() and path.stat().st_size > 1000


def test_spatial_fold_figure_renders(tmp_path):
    _, folds = spatial_block_folds(16, 16, SpatialCVConfig(block_size=4,
                                                           n_folds=4))
    path = RF.plot_spatial_folds(folds, tmp_path / "folds.png")
    assert path.exists() and path.stat().st_size > 1000


def test_figures_declare_synthetic_status():
    assert RF.dev_title("x").startswith("DEVELOPMENT / SYNTHETIC")
    assert RF.dev_title("x", synthetic=False) == "x"


# ------------------------------------------------------ experiment isolation
def test_two_runs_never_share_a_directory(tmp_path):
    """Timestamps have one-second resolution; runs must still be isolated."""
    cfg = Config(experiment_name="isolation", seed=1)
    first = start_experiment(cfg, results_root=tmp_path)
    second = start_experiment(cfg, results_root=tmp_path)
    assert first.root != second.root
    assert first.experiment_id != second.experiment_id
    (first.root / "metrics" / "a.txt").write_text("first")
    (second.root / "metrics" / "a.txt").write_text("second")
    assert (first.root / "metrics" / "a.txt").read_text() == "first"


def test_experiment_directories_carry_the_full_tree(tmp_path):
    exp = start_experiment(Config(experiment_name="tree"),
                           results_root=tmp_path)
    for sub in ("config", "metrics", "figures", "predictions", "models",
                "logs"):
        assert (exp.root / sub).is_dir()


def test_georeferenced_layer_round_trips_all_metadata(georef, tmp_path):
    mask = np.zeros((H, W), bool)
    mask[::2, ::2] = True
    values = np.arange(int(mask.sum()), dtype="float64")
    path = geo.write_layer(tmp_path / "layer.tif", values, mask, georef,
                           dtype="float32", description="test layer")
    info = geo.verify_georeference(path)
    assert info["crs"] == CRS
    assert (info["height"], info["width"]) == (H, W)
    assert info["nodata"] == geo.NODATA_FLOAT
    assert np.allclose(info["transform"], list(TRANSFORM)[:6])
    assert info["resolution"] == pytest.approx((0.01, 0.01))
    with rasterio.open(path) as raster:
        assert raster.tags().get("DESCRIPTION") == "test layer"
