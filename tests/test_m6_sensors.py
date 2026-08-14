"""Sensor handling, quality masking and spectral indices (M6 Parts 6-7).

These are the transformations that decide what a pixel's NDVI actually is.
Each is tested against small integer arrays with values chosen so the
expected answer can be computed by hand - no reference to any scene, no
network, no Earth Engine.
"""
import numpy as np
import pytest

from src.sensors import (DEFAULT_LANDSAT_MASK_BITS, INDEX_DEFINITIONS,
                         LANDSAT_QA_BITS, SENSORS, SensorError,
                         apply_scale_factors, compute_index, get_sensor,
                         harmonise_ndvi, index_table, landsat_qa_mask,
                         sensor_table, sentinel2_scl_mask)


def qa_with(*names, base_clear=True):
    """Build a QA_PIXEL value with the named flags set."""
    value = (1 << LANDSAT_QA_BITS["clear"]) if base_clear else 0
    for name in names:
        value |= 1 << LANDSAT_QA_BITS[name]
    return np.uint16(value)


# ------------------------------------------------------------ band mapping
def test_tm_and_oli_use_different_bands_for_the_same_wavelength():
    """The whole reason sensor handling cannot be a single code path."""
    tm = get_sensor("LANDSAT5_TM")
    oli = get_sensor("LANDSAT8_OLI")
    assert (tm.band("red"), tm.band("nir")) == ("SR_B3", "SR_B4")
    assert (oli.band("red"), oli.band("nir")) == ("SR_B4", "SR_B5")
    assert tm.band("red") != oli.band("red")


def test_landsat_7_shares_the_tm_band_layout():
    assert get_sensor("LANDSAT7_ETM").bands == get_sensor("LANDSAT5_TM").bands


def test_landsat_9_shares_the_oli_band_layout():
    assert get_sensor("LANDSAT9_OLI2").bands == get_sensor("LANDSAT8_OLI").bands


def test_asking_for_a_band_a_sensor_lacks_names_what_it_has():
    with pytest.raises(SensorError, match="rededge"):
        get_sensor("LANDSAT5_TM").band("rededge")


def test_unknown_sensor_lists_the_configured_ones():
    with pytest.raises(SensorError, match="LANDSAT8_OLI"):
        get_sensor("MODIS")


def test_every_sensor_records_the_part_3_metadata():
    for key, sensor in SENSORS.items():
        described = sensor.describe()
        for field in ("platform", "instrument", "collection",
                      "processing_level", "spatial_resolution_m",
                      "temporal_resolution_days", "temporal_coverage",
                      "quality_band", "documentation", "licence"):
            assert described[field], f"{key} is missing {field}"
    assert len(sensor_table()) == len(SENSORS)


# ------------------------------------------------------------- reflectance
def test_scale_factors_convert_stored_integers_to_reflectance():
    sensor = get_sensor("LANDSAT8_OLI")
    # 0.3 reflectance stored as (0.3 + 0.2) / 0.0000275
    stored = np.array([[round((0.3 + 0.2) / 2.75e-05)]])
    assert np.isclose(sensor.to_reflectance(stored)[0, 0], 0.3, atol=1e-4)


def test_fill_becomes_nan_and_not_minus_zero_point_two():
    """Fill is 0; scaling it blindly yields -0.2, a plausible reflectance."""
    scaled = apply_scale_factors(np.array([[0, 20000]]), 2.75e-05, -0.2)
    assert np.isnan(scaled[0, 0])
    assert np.isfinite(scaled[0, 1])


def test_impossible_reflectance_becomes_nan():
    scaled = apply_scale_factors(np.array([[65000, 15000]]), 2.75e-05, -0.2,
                                 valid_min=-0.2, valid_max=1.0)
    assert np.isnan(scaled[0, 0])          # above the physical range
    assert np.isfinite(scaled[0, 1])


# -------------------------------------------------------- quality masking
@pytest.mark.parametrize("flag", ["fill", "dilated_cloud", "cirrus", "cloud",
                                  "cloud_shadow", "snow"])
def test_each_default_masked_flag_rejects_the_pixel(flag):
    usable = landsat_qa_mask(np.array([[qa_with(), qa_with(flag)]]))
    assert usable[0, 0] and not usable[0, 1]


def test_water_is_kept_by_default_because_that_is_a_study_decision():
    assert "water" not in DEFAULT_LANDSAT_MASK_BITS
    assert landsat_qa_mask(np.array([[qa_with("water")]]))[0, 0]


def test_the_excluded_bit_set_is_configurable():
    qa = np.array([[qa_with("snow")]])
    assert not landsat_qa_mask(qa)[0, 0]
    assert landsat_qa_mask(qa, exclude=["cloud", "fill"])[0, 0]


def test_an_unknown_bit_name_is_rejected_rather_than_ignored():
    with pytest.raises(SensorError, match="unknown QA_PIXEL bit"):
        landsat_qa_mask(np.array([[qa_with()]]), exclude=["haze"])


def test_saturated_pixels_are_excluded_when_qa_radsat_is_supplied():
    qa = np.array([[qa_with(), qa_with()]])
    saturation = np.array([[0, 8]])
    usable = landsat_qa_mask(qa, saturation=saturation)
    assert usable[0, 0] and not usable[0, 1]


def test_strict_cloud_confidence_rejects_high_confidence_clear_pixels():
    """Bits 8-9 carry cloud confidence; 3 = high even with the cloud bit off."""
    qa = np.array([[qa_with() | np.uint16(0b11 << 8)]])
    assert landsat_qa_mask(qa)[0, 0]
    assert not landsat_qa_mask(qa, strict_confidence=True,
                               cloud_confidence_max=2)[0, 0]


def test_qa_mask_returns_usable_not_masked():
    """The sense of the mask is part of the contract; callers depend on it."""
    assert landsat_qa_mask(np.array([[qa_with()]])).dtype == bool
    assert landsat_qa_mask(np.array([[qa_with()]]))[0, 0] is np.True_


def test_non_integer_qa_is_refused():
    with pytest.raises(SensorError, match="bitmask"):
        landsat_qa_mask(np.array([[np.nan]]))


def test_sentinel2_scene_classification_excludes_cloud_and_shadow():
    scl = np.array([[4, 5, 3, 8, 9, 10, 11, 0]])
    usable = sentinel2_scl_mask(scl)
    assert list(usable[0]) == [True, True, False, False, False, False,
                               False, False]


# -------------------------------------------------------- spectral indices
def test_ndvi_matches_the_hand_computed_value():
    value = compute_index("ndvi", {"nir": np.array([[0.4]]),
                                   "red": np.array([[0.1]])})
    assert np.isclose(value[0, 0], 0.3 / 0.5)


def test_a_vanishing_denominator_yields_nan_not_a_huge_number():
    value = compute_index("ndvi", {"nir": np.array([[0.0]]),
                                   "red": np.array([[0.0]])})
    assert np.isnan(value[0, 0])


def test_index_values_outside_the_physical_range_become_nan():
    # EVI's denominator can drive the result far outside [-1, 1].
    value = compute_index("evi", {"nir": np.array([[0.5]]),
                                  "red": np.array([[0.02]]),
                                  "blue": np.array([[0.2]])})
    assert np.isnan(value[0, 0]) or -1.0 <= value[0, 0] <= 1.0


def test_nan_reflectance_propagates_to_nan_index():
    value = compute_index("ndvi", {"nir": np.array([[np.nan]]),
                                   "red": np.array([[0.1]])})
    assert np.isnan(value[0, 0])


@pytest.mark.parametrize("name,bands,expected", [
    ("nbr", {"nir": 0.4, "swir2": 0.1}, 0.3 / 0.5),
    ("ndmi", {"nir": 0.4, "swir1": 0.2}, 0.2 / 0.6),
])
def test_optional_indices_use_their_documented_bands(name, bands, expected):
    arrays = {k: np.array([[v]]) for k, v in bands.items()}
    assert np.isclose(compute_index(name, arrays)[0, 0], expected)


def test_a_missing_band_names_what_the_index_needs():
    with pytest.raises(SensorError, match="blue"):
        compute_index("evi", {"nir": np.array([[0.4]]),
                              "red": np.array([[0.1]])})


def test_unknown_index_lists_the_available_ones():
    with pytest.raises(SensorError, match="ndvi"):
        compute_index("savi", {})


def test_every_index_records_its_formula_bands_and_range():
    for name, definition in INDEX_DEFINITIONS.items():
        described = definition.describe()
        assert described["formula"] and described["bands"]
        assert described["valid_range"] == [-1.0, 1.0]
        assert described["reference"], f"{name} has no citation"
    assert len(index_table()) == len(INDEX_DEFINITIONS)


# ---------------------------------------------------------- harmonisation
def test_the_reference_sensors_are_left_unchanged():
    values = np.array([[0.2, 0.7]])
    for key in ("LANDSAT5_TM", "LANDSAT7_ETM"):
        assert np.allclose(harmonise_ndvi(values, get_sensor(key)), values)


def test_oli_ndvi_is_transformed_onto_the_etm_scale():
    """Roy et al. (2016) OLS: ETM+ = 0.0029 + 0.9589 x OLI."""
    values = np.array([[0.5]])
    result = harmonise_ndvi(values, get_sensor("LANDSAT8_OLI"))
    assert np.isclose(result[0, 0], 0.9589 * 0.5 + 0.0029)
    assert not np.isclose(result[0, 0], 0.5)      # it actually does something


def test_landsat_9_uses_the_same_transform_as_landsat_8():
    values = np.array([[0.5]])
    assert np.allclose(harmonise_ndvi(values, get_sensor("LANDSAT8_OLI")),
                       harmonise_ndvi(values, get_sensor("LANDSAT9_OLI2")))


def test_a_sensor_without_coefficients_refuses_to_assume_identity():
    """The failure mode is a step change read as a real disturbance."""
    with pytest.raises(SensorError, match="breakpoint detection"):
        harmonise_ndvi(np.array([[0.5]]), get_sensor("SENTINEL2_MSI"))


def test_an_override_supplies_the_missing_transform():
    result = harmonise_ndvi(np.array([[0.5]]), get_sensor("SENTINEL2_MSI"),
                            override={"gain": 0.98, "bias": 0.01})
    assert np.isclose(result[0, 0], 0.98 * 0.5 + 0.01)


def test_every_landsat_sensor_carries_a_citation_for_its_transform():
    for key in ("LANDSAT5_TM", "LANDSAT7_ETM", "LANDSAT8_OLI",
                "LANDSAT9_OLI2"):
        assert get_sensor(key).harmonisation["reference"]


def test_sentinel2_documents_why_it_is_not_a_default():
    notes = get_sensor("SENTINEL2_MSI").notes
    assert "2017" in notes and "default" in notes
