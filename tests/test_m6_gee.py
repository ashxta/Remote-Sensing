"""Earth Engine request construction (M6 Part 4).

`earthengine-api` is not installed here and no credential exists, so these
tests do NOT prove that an export succeeds. They prove something weaker and
still worth having: that the request the module would build asks for the
intended collections, band names, QA bits, scale factors, harmonisation
coefficients and date windows - checked against a recording stub that
captures the calls instead of contacting Google.

The distinction is stated in `gee_export`'s docstring and repeated in the M6
report. A green test here is not a successful export.
"""
import pytest

from src.config import Config
from src.gee_export import (AUTH_INSTRUCTIONS, GEEError, build_export_plan,
                            ee_available, initialize,
                            masked_index_collection)
from src.sensors import LANDSAT_QA_BITS
from src.study_area import StudyArea

AREA = StudyArea.from_bounds(92.0, 25.0, 92.4, 25.4, name="test_extent")


# --------------------------------------------------------------- the plan
def test_the_plan_names_the_pinned_archive_collections():
    plan = build_export_plan(AREA, Config().real_data)
    collections = {c["sensor"]: c["collection"] for c in plan["collections"]}
    assert collections["LANDSAT5_TM"] == "LANDSAT/LT05/C02/T1_L2"
    assert collections["LANDSAT7_ETM"] == "LANDSAT/LE07/C02/T1_L2"
    assert collections["LANDSAT8_OLI"] == "LANDSAT/LC08/C02/T1_L2"
    assert collections["LANDSAT9_OLI2"] == "LANDSAT/LC09/C02/T1_L2"


def test_the_plan_uses_the_right_bands_for_each_generation():
    plan = build_export_plan(AREA, Config().real_data)
    bands = {c["sensor"]: (c["red"], c["nir"]) for c in plan["collections"]}
    assert bands["LANDSAT5_TM"] == ("SR_B3", "SR_B4")
    assert bands["LANDSAT8_OLI"] == ("SR_B4", "SR_B5")


def test_the_plan_carries_the_harmonisation_coefficients():
    plan = build_export_plan(AREA, Config().real_data)
    oli = next(c for c in plan["collections"] if c["sensor"] == "LANDSAT8_OLI")
    assert oli["harmonisation"]["gain"] == 0.9589
    assert "Roy" in oli["harmonisation"]["reference"]


def test_the_plan_records_the_qa_bits_it_would_mask():
    plan = build_export_plan(AREA, Config().real_data)
    mask = plan["quality_mask"]
    assert set(mask["bits_excluded"]) >= {"cloud", "cloud_shadow", "fill"}
    assert mask["bit_positions"]["cloud"] == LANDSAT_QA_BITS["cloud"]


def test_the_plan_has_one_window_per_year_of_the_configured_record():
    cfg = Config().real_data
    cfg.start_year, cfg.end_year = 1990, 2025
    plan = build_export_plan(AREA, cfg)
    assert len(plan["windows"]) == 36
    assert plan["windows"][0]["start"] == "1990-10-15"


def test_the_plan_is_in_wgs84_as_geojson_requires():
    plan = build_export_plan(AREA, Config().real_data)
    assert plan["region_geojson"]["type"] == "Polygon"
    west = plan["region_geojson"]["coordinates"][0][0][0]
    assert -180 <= west <= 180


def test_the_plan_states_that_no_credential_is_stored():
    plan = build_export_plan(AREA, Config().real_data)
    assert "none stored in this repo" in plan["credentials"]


def test_mixing_an_unharmonised_sensor_is_refused_before_any_export():
    cfg = Config().real_data
    cfg.sensors = cfg.sensors + ["SENTINEL2_MSI"]
    with pytest.raises(GEEError, match="harmonisation"):
        build_export_plan(AREA, cfg)


def test_an_override_permits_the_mixed_sensor_set():
    cfg = Config().real_data
    cfg.sensors = cfg.sensors + ["SENTINEL2_MSI"]
    cfg.harmonisation_overrides = {
        "SENTINEL2_MSI": {"gain": 0.98, "bias": 0.01,
                          "reference": "supplied by the user"}}
    plan = build_export_plan(AREA, cfg)
    s2 = next(c for c in plan["collections"]
              if c["sensor"] == "SENTINEL2_MSI")
    assert s2["harmonisation"]["gain"] == 0.98


# ------------------------------------------------------------ the stub
class Recorder:
    """Captures the chained calls a real `ee` object would receive."""

    def __init__(self, log, name="ee"):
        self.log = log
        self.name = name

    def __getattr__(self, attribute):
        return Recorder(self.log, f"{self.name}.{attribute}")

    def __call__(self, *args, **kwargs):
        self.log.append((self.name, args, kwargs))
        return Recorder(self.log, f"{self.name}()")


def test_the_masked_collection_requests_the_configured_collection():
    log = []
    masked_index_collection(Recorder(log), "LANDSAT5_TM", object(),
                            Config().real_data)
    collections = [args[0] for name, args, _ in log
                   if name == "ee.ImageCollection" and args]
    assert "LANDSAT/LT05/C02/T1_L2" in collections


def test_the_masked_collection_applies_the_scene_cloud_prefilter():
    log = []
    cfg = Config().real_data
    cfg.max_scene_cloud_cover = 40.0
    masked_index_collection(Recorder(log), "LANDSAT8_OLI", object(), cfg)
    filters = [(args, kwargs) for name, args, kwargs in log
               if name == "ee.Filter.lte"]
    assert filters and filters[0][0] == ("CLOUD_COVER", 40.0)


def test_sentinel2_is_refused_by_this_export_path_rather_than_mishandled():
    cfg = Config().real_data
    cfg.harmonisation_overrides = {"SENTINEL2_MSI": {"gain": 1.0, "bias": 0.0}}
    with pytest.raises(GEEError, match="local backend"):
        masked_index_collection(Recorder([]), "SENTINEL2_MSI", object(), cfg)


def test_a_dry_run_validates_without_submitting_anything():
    plan = build_export_plan(AREA, Config().real_data)
    from src.gee_export import export_composites
    result = export_composites(AREA, Config().real_data, dry_run=True)
    assert result["submitted"] is False
    assert "nothing was exported" in result["note"]
    assert result["collections"] == plan["collections"]


# -------------------------------------------------------- availability
def test_the_module_reports_honestly_whether_ee_is_installed():
    assert isinstance(ee_available(), bool)


def test_a_missing_package_produces_setup_instructions_not_a_traceback():
    if ee_available():
        pytest.skip("earthengine-api is installed in this environment")
    with pytest.raises(GEEError) as error:
        initialize()
    message = str(error.value)
    assert "earthengine authenticate" in message
    assert "pip install earthengine-api" in message


def test_the_setup_instructions_never_ask_for_a_credential_in_the_repo():
    assert "must never be copied into this repository" in AUTH_INSTRUCTIONS
    assert ".gitignore" in AUTH_INSTRUCTIONS


def test_an_initialisation_failure_is_wrapped_with_the_instructions():
    class Failing:
        @staticmethod
        def Initialize(*args, **kwargs):
            raise RuntimeError("not authenticated")

    with pytest.raises(GEEError, match="earthengine authenticate"):
        initialize("some-project", ee_module=Failing)


def test_initialisation_passes_the_project_when_one_is_configured():
    seen = {}

    class Fake:
        @staticmethod
        def Initialize(project=None):
            seen["project"] = project

    initialize("my-project", ee_module=Fake)
    assert seen["project"] == "my-project"
