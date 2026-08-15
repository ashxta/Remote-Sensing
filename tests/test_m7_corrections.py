"""The targeted-correction tooling (boundary, resolution, comparison).

These cover the parts of the correction that produce numbers a reader will
quote: the merged administrative geometry, and the before/after table. The
network-dependent halves (downloading the boundary, reading scenes) are not
exercised here - they are driven by hand and their outputs are checked in
`test_m6_study_area.py` and `test_m7_acquisition.py`.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def load_tool(name: str):
    """Import a script from tools/ without making tools/ a package."""
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------- boundary merging
def test_two_districts_merge_into_one_multipolygon():
    """The 2016 split spans the study period, so both successors belong."""
    tool = load_tool("fetch_study_area_boundary")
    east = {"geometry": {"type": "Polygon",
                         "coordinates": [[[93.0, 25.6], [93.8, 25.6],
                                          [93.8, 26.5], [93.0, 26.5],
                                          [93.0, 25.6]]]}}
    west = {"geometry": {"type": "MultiPolygon",
                         "coordinates": [[[[92.2, 25.6], [93.0, 25.6],
                                           [93.0, 26.1], [92.2, 26.1],
                                           [92.2, 25.6]]]]}}
    merged = tool.merge([east, west])
    assert merged["type"] == "MultiPolygon"
    assert len(merged["coordinates"]) == 2


def test_a_single_district_stays_a_polygon():
    tool = load_tool("fetch_study_area_boundary")
    one = {"geometry": {"type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1],
                                         [0, 0]]]}}
    assert tool.merge([one])["type"] == "Polygon"


def test_a_non_polygonal_geometry_is_refused():
    tool = load_tool("fetch_study_area_boundary")
    with pytest.raises(SystemExit, match="unexpected geometry"):
        tool.merge([{"geometry": {"type": "LineString",
                                  "coordinates": [[0, 0], [1, 1]]}}])


def test_merging_moves_no_vertex():
    """A topological union would move vertices; concatenation must not."""
    tool = load_tool("fetch_study_area_boundary")
    a = [[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]]
    b = [[[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 0.0]]]
    merged = tool.merge([{"geometry": {"type": "Polygon", "coordinates": a}},
                         {"geometry": {"type": "Polygon", "coordinates": b}}])
    assert merged["coordinates"] == [a, b]


def test_the_default_selection_is_both_successor_districts():
    tool = load_tool("fetch_study_area_boundary")
    assert set(tool.DEFAULT_DISTRICTS) == {"Karbi Anglong East",
                                           "Karbi Anglong West"}


def test_the_source_record_carries_what_a_citation_needs():
    tool = load_tool("fetch_study_area_boundary")
    for field in ("name", "url", "licence", "vintage", "citation"):
        assert tool.SOURCE[field]
    assert "NOT the Survey of India" in tool.SOURCE["authority_note"]


# ------------------------------------------------------ run comparison
def _results(**overrides):
    base = {
        "study_area": {"name": "Area",
                       "attributes": {"geometry_kind": "bounding box"}},
        "experiment": {"grid": [100, 200], "n_analysed": 5000,
                       "n_time_steps": 35},
        "trend": {"analysed_area_km2": 450.0,
                  "areas": [
                      {"class": "significant_increase", "area_km2": 100.0},
                      {"class": "significant_decrease", "area_km2": 50.0},
                      {"class": "no_significant_trend", "area_km2": 300.0}]},
        "restrend": {"restrend_valid_fraction": 0.025,
                     "restrend_valid_area_km2": 11.0,
                     "categories": [
                         {"category": "decline_persists_after_climate_adjustment",
                          "area_km2": 8.0}]},
        "cyclicity": {"periodic_area_km2": 3.0},
        "disturbance_recovery": {"disturbed_area_km2": 40.0},
        "trajectories": {"areas_km2": {"Stable": 300.0, "Degrading": 40.0,
                                        "Recovering": 30.0,
                                        "Uncertain / Other": 80.0}},
        "baseline_comparison": {"baseline_flagged_pixels": 600,
                                "integrated_persistent_decline_pixels": 550,
                                "reduction_from_baseline_to_persistent": 0.083},
        "sensor_confound": {"step_ndvi": 0.02,
                            "paired_cross_sensor_residual_ndvi": -0.018},
    }
    base.update(overrides)
    return base


def test_the_comparison_extracts_every_quantity_the_report_quotes():
    tool = load_tool("compare_m7_runs")
    row = tool.rows(_results())
    assert row["analysed area (km2)"] == 450.0
    assert row["significant increase (km2)"] == 100.0
    assert row["significant decrease (km2)"] == 50.0
    assert row["RESTREND-valid (km2)"] == 11.0
    assert row["cyclic (km2)"] == 3.0
    assert row["trajectory Degrading (km2)"] == 40.0
    assert row["paired cross-sensor residual (NDVI)"] == -0.018
    assert row["boundary kind"] == "bounding box"


def test_missing_quantities_become_nan_rather_than_a_guess():
    tool = load_tool("compare_m7_runs")
    row = tool.rows({"trend": {}, "trajectories": {}})
    value = row["significant increase (km2)"]
    assert value != value, "absent metrics must be NaN, never invented"


def test_the_comparison_writes_both_a_table_and_a_record(tmp_path):
    tool = load_tool("compare_m7_runs")
    before = tmp_path / "before"
    after = tmp_path / "after"
    for run, results in ((before, _results()),
                         (after, _results(trend={
                             "analysed_area_km2": 500.0,
                             "areas": [{"class": "significant_increase",
                                        "area_km2": 120.0}]}))):
        (run / "summary").mkdir(parents=True)
        (run / "summary" / "results.json").write_text(json.dumps(results))

    sys.argv = ["compare", "--before", str(before), "--after", str(after)]
    assert tool.main() == 0
    payload = json.loads(
        (after / "summary" / "comparison_with_previous.json").read_text())
    assert payload["before_run"].endswith("before")
    assert any(r["quantity"] == "analysed area (km2)" for r in payload["rows"])
    assert "combined effect" in payload["caveat"]
    assert (after / "summary" / "comparison_with_previous.txt").exists()


def test_the_comparison_warns_that_two_changes_are_confounded(tmp_path):
    """The boundary and the coarsening changed together, so no single
    difference can be attributed to one of them. The saved record must say
    so, because the table on its own invites exactly that inference."""
    tool = load_tool("compare_m7_runs")
    assert "two things at once" in tool.__doc__

    before, after = tmp_path / "b", tmp_path / "a"
    for run in (before, after):
        (run / "summary").mkdir(parents=True)
        (run / "summary" / "results.json").write_text(json.dumps(_results()))
    sys.argv = ["compare", "--before", str(before), "--after", str(after)]
    tool.main()

    caveat = json.loads((after / "summary"
                         / "comparison_with_previous.json").read_text())["caveat"]
    assert "combined effect" in caveat.lower()
    assert "compare the shares" in caveat.lower()


# -------------------------------------------------- resolution benchmark
def test_block_aggregation_ignores_missing_values():
    tool = load_tool("benchmark_resolution")
    import numpy as np

    values = np.array([[1.0, 2.0], [np.nan, 4.0]])
    assert tool.aggregate(values, 2, "mean")[0, 0] == pytest.approx(7 / 3)
    assert tool.aggregate(values, 2, "count")[0, 0] == 3


def test_block_aggregation_handles_a_ragged_edge():
    tool = load_tool("benchmark_resolution")
    import numpy as np

    # 5x5 with factor 2 -> the last row and column are dropped, not padded.
    result = tool.aggregate(np.ones((5, 5)), 2, "mean")
    assert result.shape == (2, 2)
