"""Spatial and temporal alignment (M6 Parts 9-10).

Misalignment is the quietest way to invalidate this project: RESTREND would
regress one place's NDVI on another place's rainfall and return a confident
number. Every check below is written so that it FAILS on a deliberately
broken input - a check that cannot fail proves nothing.
"""
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine, from_origin

from src.alignment import (RESAMPLING_FOR, AlignmentError, align_to_reference,
                           check_grid_alignment, check_temporal_alignment,
                           reproject_cube, reproject_to_grid,
                           require_alignment)
from src.geo import GeoRef

CRS_WGS84 = rasterio.crs.CRS.from_epsg(4326)
CRS_UTM = rasterio.crs.CRS.from_epsg(32646)


def grid(crs=CRS_WGS84, west=92.0, north=26.0, resolution=0.01,
         height=20, width=20):
    return GeoRef(crs, from_origin(west, north, resolution, resolution),
                  height, width)


# ------------------------------------------------------- alignment checks
def test_identical_grids_are_aligned():
    report = check_grid_alignment(grid(), grid())
    assert report.aligned
    assert report.mismatches == []


def test_a_crs_difference_is_detected():
    report = check_grid_alignment(grid(), grid(crs=CRS_UTM))
    assert not report.aligned
    assert any("CRS differs" in m for m in report.mismatches)


def test_a_resolution_difference_is_detected():
    report = check_grid_alignment(grid(), grid(resolution=0.02))
    assert any("resolution differs" in m for m in report.mismatches)


def test_a_dimension_difference_is_detected():
    report = check_grid_alignment(grid(), grid(height=21))
    assert any("dimensions differ" in m for m in report.mismatches)


def test_an_origin_shift_is_detected():
    report = check_grid_alignment(grid(), grid(west=92.5))
    assert any("origin differs" in m for m in report.mismatches)


def test_a_sub_pixel_offset_is_called_out_specifically():
    """Same CRS, same resolution, same size - and every pixel is wrong."""
    report = check_grid_alignment(grid(), grid(west=92.0 + 0.005))
    assert not report.aligned
    assert any("FRACTION of a pixel" in m for m in report.mismatches)


def test_a_whole_pixel_offset_is_not_reported_as_a_fractional_one():
    report = check_grid_alignment(grid(), grid(west=92.0 + 0.03))
    assert not report.aligned
    assert not any("FRACTION of a pixel" in m for m in report.mismatches)


def test_a_rotated_transform_is_rejected():
    rotated = GeoRef(CRS_WGS84,
                     Affine(0.01, 0.002, 92.0, 0.002, -0.01, 26.0), 20, 20)
    report = check_grid_alignment(grid(), rotated)
    assert any("rotated or sheared" in m for m in report.mismatches)


def test_require_alignment_raises_and_says_what_to_do():
    with pytest.raises(AlignmentError, match="reproject_to_grid"):
        require_alignment(grid(), grid(crs=CRS_UTM), what="rainfall cube")


def test_require_alignment_names_the_layer_that_failed():
    with pytest.raises(AlignmentError, match="rainfall cube"):
        require_alignment(grid(), grid(resolution=0.05), what="rainfall cube")


def test_the_report_is_serialisable_for_the_run_record():
    summary = check_grid_alignment(grid(), grid(crs=CRS_UTM)).summary()
    assert summary["aligned"] is False
    assert summary["reference_grid"]["crs"]
    assert isinstance(summary["mismatches"], list)


# ------------------------------------------------------------ resampling
def test_reprojection_lands_values_on_the_target_grid():
    coarse = grid(resolution=0.04, height=5, width=5)
    fine = grid(resolution=0.01, height=20, width=20)
    values = np.arange(25, dtype="float64").reshape(5, 5)
    result = reproject_to_grid(values, coarse, fine, resampling="nearest")
    assert result.shape == (20, 20)
    # Nearest neighbour must reproduce the source values exactly.
    assert set(np.unique(result[np.isfinite(result)])) <= set(np.unique(values))


def test_reprojection_across_a_crs_change_preserves_the_field():
    """Going WGS84 -> UTM must move data, not merely reshape it."""
    source = grid(resolution=0.02, height=20, width=20)
    values = np.full((20, 20), 1500.0)
    from src.study_area import StudyArea
    area = StudyArea.from_bounds(*source.bounds, name="t")
    target = area.grid(2000.0, crs="EPSG:32646")
    result = reproject_to_grid(values, source, target, resampling="bilinear")
    interior = result[np.isfinite(result)]
    assert interior.size > 0
    assert np.allclose(interior, 1500.0, atol=1e-6)


def test_nodata_is_preserved_and_never_becomes_zero():
    coarse = grid(resolution=0.04, height=5, width=5)
    fine = grid(resolution=0.01, height=20, width=20)
    values = np.full((5, 5), 100.0)
    values[0, 0] = np.nan
    result = reproject_to_grid(values, coarse, fine, resampling="nearest")
    assert np.isnan(result).any()
    assert not (result == 0).any()


def test_uncovered_target_cells_come_back_nan():
    source = grid(resolution=0.01, height=5, width=5)          # small
    target = grid(resolution=0.01, height=40, width=40)        # much larger
    result = reproject_to_grid(np.ones((5, 5)), source, target)
    assert np.isnan(result).any()
    assert np.isfinite(result).any()


def test_a_cube_is_reprojected_band_by_band():
    coarse = grid(resolution=0.04, height=5, width=5)
    fine = grid(resolution=0.01, height=20, width=20)
    cube = np.stack([np.full((5, 5), float(i)) for i in range(4)])
    result = reproject_cube(cube, coarse, fine, resampling="nearest")
    assert result.shape == (4, 20, 20)
    for i in range(4):
        band = result[i][np.isfinite(result[i])]
        assert np.allclose(band, float(i))


def test_an_array_that_lies_about_its_source_grid_is_refused():
    with pytest.raises(AlignmentError, match="declared source grid"):
        reproject_to_grid(np.ones((4, 4)), grid(), grid())


def test_reprojection_needs_a_crs_on_both_sides():
    ungeoreferenced = GeoRef(None, from_origin(0, 0, 1, 1), 5, 5)
    with pytest.raises(AlignmentError, match="ungeoreferenced"):
        reproject_to_grid(np.ones((5, 5)), ungeoreferenced, grid())


def test_an_unknown_resampling_method_lists_the_real_ones():
    coarse = grid(resolution=0.04, height=5, width=5)
    with pytest.raises(AlignmentError, match="unknown resampling"):
        reproject_to_grid(np.ones((5, 5)), coarse, grid(),
                          resampling="telepathy")


# --------------------------------------------------- align_to_reference
def test_an_already_aligned_layer_is_not_resampled():
    aligned, report = align_to_reference(np.ones((20, 20)), grid(), grid())
    assert report["resampled"] is False
    assert np.allclose(aligned, 1.0)


def test_resampling_records_the_reason_and_the_justification():
    coarse = grid(resolution=0.04, height=5, width=5)
    _, report = align_to_reference(np.ones((5, 5)), coarse, grid(),
                                   kind="rainfall")
    assert report["resampled"] is True
    assert report["method"] == "bilinear"
    assert report["reason"]
    assert "autocorrelated" in report["justification"]
    assert report["alignment_after"]["aligned"] is True
    assert "does not add spatial detail" in report["native_resolution_retained"]


def test_categorical_layers_default_to_nearest_neighbour():
    """An interpolated class code would be a class that does not exist."""
    assert RESAMPLING_FOR["labels"] == "nearest"
    assert RESAMPLING_FOR["categorical"] == "nearest"
    coarse = grid(resolution=0.04, height=5, width=5)
    codes = np.array([[1, 2, 3, 4, 5]] * 5, dtype="float64")
    result, report = align_to_reference(codes, coarse, grid(),
                                        kind="categorical")
    assert report["method"] == "nearest"
    present = np.unique(result[np.isfinite(result)])
    assert set(present) <= {1.0, 2.0, 3.0, 4.0, 5.0}


def test_continuous_layers_default_to_bilinear():
    assert RESAMPLING_FOR["rainfall"] == "bilinear"
    assert RESAMPLING_FOR["index"] == "bilinear"


# ------------------------------------------------------ temporal alignment
def test_identical_time_axes_pass():
    report = check_temporal_alignment(["2000", "2001"], ["2000", "2001"])
    assert report["aligned"] and report["n_time_steps"] == 2


def test_different_lengths_are_caught():
    with pytest.raises(AlignmentError, match="but rainfall has"):
        check_temporal_alignment(["2000", "2001"], ["2000"])


def test_a_one_step_lag_is_identified_by_name():
    """The failure that would regress NDVI on the PREVIOUS year's rain."""
    years = [str(y) for y in range(2000, 2010)]
    with pytest.raises(AlignmentError, match="rainfall lags vegetation"):
        check_temporal_alignment(years, [str(y - 1) for y in
                                         range(2000, 2010)])


def test_a_one_step_lead_is_identified_by_name():
    years = [str(y) for y in range(2000, 2010)]
    with pytest.raises(AlignmentError, match="rainfall leads vegetation"):
        check_temporal_alignment(years, [str(y + 1) for y in
                                         range(2000, 2010)])


def test_unordered_time_labels_are_caught():
    with pytest.raises(AlignmentError, match="ascending order"):
        check_temporal_alignment(["2001", "2000"], ["2001", "2000"])


def test_the_layer_being_compared_is_named_in_the_message():
    with pytest.raises(AlignmentError, match="evapotranspiration"):
        check_temporal_alignment(["2000"], ["2000", "2001"],
                                 what="evapotranspiration")
