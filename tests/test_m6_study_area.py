"""Configurable study-area boundaries (M6 Part 2).

The point of these tests is that the boundary is DATA. If any of them could
only pass for one particular region, the abstraction would have failed, so
every geometry here is arbitrary and none of them is Assam.
"""
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.config import StudyAreaConfig
from src.geo import GeoRef
from src.study_area import (StudyArea, StudyAreaError, area_statistics,
                            geometry_bounds, load_study_area, pixel_area_km2)

SQUARE = {"type": "Polygon",
          "coordinates": [[[10.0, 40.0], [11.0, 40.0], [11.0, 41.0],
                           [10.0, 41.0], [10.0, 40.0]]]}


def write_geojson(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------- loading
def test_bare_geometry_feature_and_collection_all_load(tmp_path):
    """A boundary file may legitimately be any of the three GeoJSON forms."""
    forms = {
        "geometry.geojson": SQUARE,
        "feature.geojson": {"type": "Feature", "geometry": SQUARE,
                            "properties": {"district": "Somewhere"}},
        "collection.geojson": {"type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": SQUARE,
             "properties": {"district": "Somewhere"}}]},
    }
    for name, payload in forms.items():
        area = StudyArea.from_geojson(write_geojson(tmp_path / name, payload))
        assert area.geometry["type"] == "Polygon"
        assert area.bounds == (10.0, 40.0, 11.0, 41.0)


def test_multi_feature_file_is_filtered_by_property(tmp_path):
    other = {"type": "Polygon",
             "coordinates": [[[20.0, 40.0], [21.0, 40.0], [21.0, 41.0],
                              [20.0, 41.0], [20.0, 40.0]]]}
    path = write_geojson(tmp_path / "districts.geojson", {
        "type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": SQUARE,
             "properties": {"NAME": "Alpha"}},
            {"type": "Feature", "geometry": other,
             "properties": {"NAME": "Beta"}}]})
    area = StudyArea.from_geojson(path, select={"NAME": "Beta"},
                                  name_property="NAME")
    assert area.name == "Beta"
    assert area.bounds == (20.0, 40.0, 21.0, 41.0)
    with pytest.raises(StudyAreaError, match="matches"):
        StudyArea.from_geojson(path, select={"NAME": "Gamma"})


def test_several_matching_features_merge_into_one_multipolygon(tmp_path):
    second = {"type": "Polygon",
              "coordinates": [[[11.0, 40.0], [12.0, 40.0], [12.0, 41.0],
                               [11.0, 41.0], [11.0, 40.0]]]}
    path = write_geojson(tmp_path / "blocks.geojson", {
        "type": "FeatureCollection", "features": [
            {"type": "Feature", "geometry": SQUARE, "properties": {"D": "x"}},
            {"type": "Feature", "geometry": second, "properties": {"D": "x"}}]})
    area = StudyArea.from_geojson(path, select={"D": "x"})
    assert area.geometry["type"] == "MultiPolygon"
    assert area.bounds == (10.0, 40.0, 12.0, 41.0)
    assert area.attributes["n_features_merged"] == 2


def test_missing_file_says_what_to_configure(tmp_path):
    with pytest.raises(StudyAreaError, match="study_area.boundary"):
        StudyArea.from_geojson(tmp_path / "absent.geojson")


def test_malformed_json_is_reported_as_such(tmp_path):
    path = tmp_path / "broken.geojson"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(StudyAreaError, match="not valid JSON"):
        StudyArea.from_geojson(path)


# ------------------------------------------------------------- validation
@pytest.mark.parametrize("geometry,message", [
    ({"type": "Point", "coordinates": [10.0, 40.0]}, "Polygon"),
    ({"type": "Polygon", "coordinates": [[[10.0, 40.0], [11.0, 40.0]]]},
     "at least 4 positions"),
    ({"type": "Polygon",
      "coordinates": [[[10.0, 40.0], [10.0, 40.0], [10.0, 40.0],
                       [10.0, 40.0]]]}, "empty extent"),
    ({"type": "Polygon",
      "coordinates": [[[10.0, 40.0], [float("nan"), 40.0], [11.0, 41.0],
                       [10.0, 41.0], [10.0, 40.0]]]}, "non-finite"),
])
def test_unusable_geometries_are_rejected(geometry, message):
    with pytest.raises(StudyAreaError, match=message):
        StudyArea(name="test", geometry=geometry)


def test_projected_coordinates_labelled_as_wgs84_are_caught():
    """A UTM polygon tagged EPSG:4326 is the classic silent boundary bug."""
    utm = {"type": "Polygon",
           "coordinates": [[[440000.0, 2840000.0], [500000.0, 2840000.0],
                            [500000.0, 2900000.0], [440000.0, 2900000.0],
                            [440000.0, 2840000.0]]]}
    with pytest.raises(StudyAreaError, match=r"outside \[-180, 180\]"):
        StudyArea(name="mislabelled", geometry=utm, crs="EPSG:4326")


def test_a_boundary_needs_a_name():
    with pytest.raises(StudyAreaError, match="name"):
        StudyArea(name="  ", geometry=SQUARE)


# --------------------------------------------------------------- geometry
def test_reprojection_moves_coordinates_and_records_the_source():
    area = StudyArea(name="test", geometry=SQUARE)
    moved = area.to_crs("EPSG:32632")
    assert moved.crs == "EPSG:32632"
    assert moved.attributes["reprojected_from"] == "EPSG:4326"
    west, south, east, north = moved.bounds
    assert west > 1000 and north > 1000          # metres, not degrees
    # Reprojecting back recovers the original to within a metre of ground.
    back = moved.to_crs("EPSG:4326")
    assert np.allclose(back.bounds, area.bounds, atol=1e-4)


def test_reprojection_to_the_same_crs_is_a_no_op():
    area = StudyArea(name="test", geometry=SQUARE)
    assert area.to_crs("EPSG:4326") is area


def test_geometry_bounds_walks_nested_multipolygons():
    multi = {"type": "MultiPolygon",
             "coordinates": [SQUARE["coordinates"],
                             [[[30.0, 50.0], [31.0, 50.0], [31.0, 51.0],
                               [30.0, 51.0], [30.0, 50.0]]]]}
    assert geometry_bounds(multi) == (10.0, 40.0, 31.0, 51.0)


# ---------------------------------------------------------------- rasters
def make_georef(crs="EPSG:4326"):
    return GeoRef(rasterio.crs.CRS.from_user_input(crs),
                  from_origin(9.5, 41.5, 0.1, 0.1), 20, 20)


def test_mask_selects_the_interior_of_the_boundary():
    area = StudyArea(name="test", geometry=SQUARE)
    inside = area.mask(make_georef())
    assert inside.shape == (20, 20)
    assert inside.any() and not inside.all()
    # The 1x1 degree square at 0.1 degree pixels covers ~100 cells.
    assert 80 <= int(inside.sum()) <= 121


def test_mask_reprojects_the_boundary_onto_a_projected_grid():
    """A WGS84 boundary must still mask a UTM raster correctly."""
    area = StudyArea(name="test", geometry=SQUARE)
    utm = area.to_crs("EPSG:32632")
    west, south, east, north = utm.bounds
    georef = GeoRef(rasterio.crs.CRS.from_epsg(32632),
                    from_origin(west - 5000, north + 5000, 2000, 2000),
                    int((north - south) / 2000) + 5,
                    int((east - west) / 2000) + 5)
    inside = area.mask(georef)
    assert inside.any() and not inside.all()


def test_mask_without_a_crs_is_an_error_not_a_guess():
    area = StudyArea(name="test", geometry=SQUARE)
    with pytest.raises(StudyAreaError, match="no CRS"):
        area.mask(GeoRef(None, from_origin(0, 0, 1, 1), 5, 5))


def test_clip_crops_masks_outside_and_keeps_the_georeference():
    area = StudyArea(name="test", geometry=SQUARE)
    georef = make_georef()
    cube = np.ones((4, 20, 20))
    clipped, out_ref, inside = area.clip(cube, georef)

    assert clipped.shape[0] == 4
    assert clipped.shape[1:] == out_ref.shape
    assert clipped.shape[1] < 20 and clipped.shape[2] < 20   # actually cropped
    assert np.isnan(clipped[:, ~inside]).all()               # outside is NaN
    assert np.isfinite(clipped[:, inside]).all()             # inside kept
    # The cropped georeference still describes the same ground.
    left, _, _, top = out_ref.bounds
    assert left >= georef.bounds[0] and top <= georef.bounds[3]
    assert out_ref.resolution == georef.resolution


def test_clip_promotes_an_integer_grid_rather_than_inventing_a_sentinel():
    area = StudyArea(name="test", geometry=SQUARE)
    labels = np.ones((20, 20), dtype="int16")
    clipped, _, inside = area.clip(labels, make_georef())
    assert np.issubdtype(clipped.dtype, np.floating)
    assert np.isnan(clipped[~inside]).all()


def test_a_boundary_that_misses_the_raster_raises():
    far = StudyArea.from_bounds(150.0, -30.0, 151.0, -29.0, name="elsewhere")
    with pytest.raises(StudyAreaError, match="does not overlap"):
        far.window(make_georef())


def test_grid_covers_the_boundary_at_the_requested_resolution():
    area = StudyArea(name="test", geometry=SQUARE)
    grid = area.grid(0.05)
    assert np.allclose(grid.resolution, (0.05, 0.05))
    west, south, east, north = grid.bounds
    assert west <= 10.0 and east >= 11.0
    assert south <= 40.0 and north >= 41.0
    with pytest.raises(StudyAreaError, match="positive"):
        area.grid(0.0)


# ------------------------------------------------------- area statistics
def test_geographic_pixel_area_shrinks_towards_the_pole():
    """The reason area statistics cannot be a pixel count on a lat/lon grid."""
    georef = GeoRef(rasterio.crs.CRS.from_epsg(4326),
                    from_origin(0.0, 60.0, 0.1, 0.1), 300, 5)
    areas = pixel_area_km2(georef)
    assert areas.shape == (300, 5)
    # Every row is constant; rows shrink monotonically going north (row 0 is
    # the top = highest latitude).
    assert np.allclose(areas.std(axis=1), 0)
    column = areas[:, 0]
    assert (np.diff(column) > 0).all()
    assert column[-1] / column[0] > 1.5


def test_projected_pixel_area_is_constant_and_correct():
    georef = GeoRef(rasterio.crs.CRS.from_epsg(32646),
                    from_origin(400000, 2900000, 30, 30), 10, 10)
    areas = pixel_area_km2(georef)
    assert np.allclose(areas, (30 * 30) / 1e6)


def test_area_statistics_sum_to_the_analysed_area():
    georef = GeoRef(rasterio.crs.CRS.from_epsg(32646),
                    from_origin(400000, 2900000, 100, 100), 10, 10)
    grid = np.zeros((10, 10))
    grid[:4] = 1
    grid[4:7] = 2
    grid[7:] = 3
    table = area_statistics(grid, georef,
                            class_names={1: "a", 2: "b", 3: "c"})
    assert list(table["class_name"]) == ["a", "b", "c"]
    assert np.isclose(table["area_km2"].sum(), 100 * 0.01)
    assert np.isclose(table["fraction_of_analysed_area"].sum(), 1.0)
    assert "projected" in table["method"].iloc[0]


def test_area_statistics_honour_the_analysis_mask():
    georef = GeoRef(rasterio.crs.CRS.from_epsg(32646),
                    from_origin(400000, 2900000, 100, 100), 10, 10)
    grid = np.ones((10, 10))
    mask = np.zeros((10, 10), bool)
    mask[:5] = True
    table = area_statistics(grid, georef, valid_mask=mask)
    assert int(table["n_pixels"].iloc[0]) == 50


# ------------------------------------------------------------ config path
def test_load_study_area_prefers_a_boundary_file(tmp_path):
    path = write_geojson(tmp_path / "b.geojson", SQUARE)
    cfg = StudyAreaConfig(name="Named", boundary=str(path),
                          bounds=[0, 0, 1, 1])
    area = load_study_area(cfg)
    assert area.bounds == (10.0, 40.0, 11.0, 41.0)   # the file, not the bbox


def test_load_study_area_falls_back_to_bounds():
    area = load_study_area(StudyAreaConfig(name="Box",
                                           bounds=[10.0, 40.0, 11.0, 41.0]))
    assert area.bounds == (10.0, 40.0, 11.0, 41.0)
    assert area.attributes["is_administrative_boundary"] is False


def test_no_study_area_configured_is_an_error_not_a_default():
    """The pipeline must never invent an extent."""
    with pytest.raises(StudyAreaError, match="will not invent"):
        load_study_area(StudyAreaConfig())


# -------------------------------------------------------------- provenance
def test_bounding_boxes_declare_that_they_are_not_administrative():
    area = StudyArea.from_bounds(10.0, 40.0, 11.0, 41.0, name="Box")
    described = area.describe()
    assert described["attributes"]["is_administrative_boundary"] is False
    assert described["attributes"]["geometry_kind"] == "bounding box"


def test_the_repository_boundary_declares_itself_a_bounding_box():
    """The shipped bounding box must not be mistaken for a district."""
    from src.config import Config
    from pathlib import Path

    path = Path(Config().paths.boundaries) / "karbi_anglong_bbox.geojson"
    if not path.exists():
        pytest.skip("repository boundary not present")
    area = StudyArea.from_geojson(path)
    assert area.attributes.get("is_administrative_boundary") is False
    assert "NOT the administrative district boundary" in \
        area.attributes["description"]


def test_the_authoritative_boundary_is_a_real_administrative_polygon():
    """The boundary the final study uses must be the district, and must
    carry the provenance a published map has to cite."""
    from pathlib import Path

    from src.config import Config
    from src.study_area import pixel_area_km2

    path = Path(Config().paths.boundaries) / "karbi_anglong.geojson"
    if not path.exists():
        pytest.skip("authoritative boundary not fetched "
                    "(tools/fetch_study_area_boundary.py)")
    area = StudyArea.from_geojson(path)
    properties = area.attributes.get("properties", area.attributes)

    assert properties.get("is_administrative_boundary") is True
    assert properties.get("geometry_kind") == "administrative polygon"
    for field in ("source", "licence", "vintage", "citation"):
        assert properties.get(field), f"missing provenance: {field}"
    assert properties.get("modifications") == \
        "none; vertices are exactly as published"

    # The 2016 split spans the study period, so both successors belong.
    merged = properties.get("districts_merged", [])
    assert len(merged) == 2, f"expected both successor districts, got {merged}"
    assert properties.get("merge_rationale")

    # Independent sanity check: the undivided district is ~10,434 km2.
    grid = area.grid(0.004, crs="EPSG:4326")
    km2 = float(pixel_area_km2(grid)[area.mask(grid)].sum())
    assert 9_500 < km2 < 11_500, (
        f"polygon area {km2:,.0f} km2 is not close to the published "
        "~10,434 km2 for undivided Karbi Anglong")
    assert area.geometry["type"] in ("Polygon", "MultiPolygon")


def test_save_round_trips_the_geometry(tmp_path):
    area = StudyArea.from_bounds(10.0, 40.0, 11.0, 41.0, name="Box",
                                 attributes={"vintage": "2011"})
    saved = area.save(tmp_path / "out.geojson")
    restored = StudyArea.from_geojson(saved)
    assert restored.bounds == area.bounds
    assert restored.attributes["properties"]["vintage"] == "2011"


def test_describe_states_that_the_analysis_is_boundary_independent():
    area = StudyArea.from_bounds(10.0, 40.0, 11.0, 41.0, name="Box")
    note = area.describe()["note"]
    assert "independent of this" in note
    assert "only valid for the area actually processed" in note
