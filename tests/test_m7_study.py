"""M7 study runner: the guards that protect the final research claims.

The runner itself is far too slow to execute in a test (35 years over
200,000 real pixels). What is tested here is the small set of decisions that
determine whether a number leaving this phase is honest: that synthetic data
cannot be published as a real finding, that the blocked experiments are
recorded rather than quietly skipped, and that the sensor-confound
diagnostic reaches the right verdict on inputs whose answer is known.
"""
import json

import numpy as np
import pandas as pd
import pytest

import run_m7_study as M7


# ------------------------------------------------------- the blocked record
def test_the_blocked_record_names_every_supervised_experiment():
    blocked = M7.BLOCKED
    assert blocked["status"].startswith("NOT SCIENTIFICALLY VALID")
    text = " ".join(blocked["blocked_experiments"])
    for expected in ("Random Forest", "CNN", "ablation",
                     "spatial cross-validation", "temporal generalization"):
        assert expected.lower() in text.lower(), f"{expected} not declared"


def test_the_blocked_record_explains_the_circularity_rather_than_hand_waving():
    reason = M7.BLOCKED["why_the_trajectory_classes_cannot_be_used"]
    for expected in ("Mann-Kendall", "Sen slope", "RESTREND",
                     "label leakage"):
        assert expected in reason
    assert "near-perfect accuracy that measures nothing" in reason


def test_the_blocked_record_says_what_would_unblock_it():
    routes = M7.BLOCKED["what_would_unblock_it"]
    assert len(routes) >= 3
    joined = " ".join(routes).lower()
    assert "field" in joined and "interpretation" in joined


def test_the_blocked_record_states_what_did_run():
    ran = M7.BLOCKED["what_was_run_instead"].lower()
    for expected in ("mann-kendall", "restrend", "cyclicity", "recovery",
                     "sensitivity"):
        assert expected in ran


def test_the_output_tree_matches_the_documented_structure():
    for directory in ("configuration", "data_quality", "temporal_analysis",
                      "features", "models", "validation", "ablation",
                      "sensitivity", "uncertainty", "maps", "figures",
                      "tables", "logs", "summary"):
        assert directory in M7.TREE


def test_blocked_notices_are_written_where_a_reader_looks_for_results(tmp_path):
    class FakeExperiment:
        def path(self, subdir, filename=""):
            target = tmp_path / subdir
            target.mkdir(parents=True, exist_ok=True)
            return target / filename if filename else target

    class FakeLog:
        def warning(self, *args, **kwargs):
            pass

    M7.stage_blocked(FakeExperiment(), FakeLog())
    for directory in ("models", "validation", "ablation"):
        path = tmp_path / directory / "BLOCKED.json"
        assert path.exists(), f"{directory} has no BLOCKED.json"
        payload = json.loads(path.read_text())
        assert payload["status"].startswith("NOT SCIENTIFICALLY VALID")


# ------------------------------------------------- the synthetic-data guard
class FakeDataset:
    def __init__(self, synthetic):
        self.metadata = {"synthetic": synthetic}


class FakePrepared:
    def __init__(self, synthetic):
        self.dataset = FakeDataset(synthetic)


def test_the_study_refuses_to_run_on_synthetic_cubes(monkeypatch, tmp_path):
    """The single guard preventing fixture output becoming a 'finding'."""
    monkeypatch.setattr(M7, "RealRemoteSensingSource",
                        lambda *args, **kwargs: object())
    monkeypatch.setattr(M7, "prepare_experiment",
                        lambda *args, **kwargs: FakePrepared(True))

    class FakeExperiment:
        def path(self, subdir, filename=""):
            target = tmp_path / subdir
            target.mkdir(parents=True, exist_ok=True)
            return target / filename if filename else target

    class FakeLog:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

    from src.config import Config
    with pytest.raises(SystemExit, match="SYNTHETIC"):
        M7.stage_dataset(Config(), object(), FakeExperiment(), FakeLog())


# ------------------------------------------------- the sensor confound
class ConfoundPrepared:
    """Minimal stand-in exposing what `stage_sensor_confound` reads."""

    def __init__(self, annual_means, slope, years):
        rows, cols = 4, 5
        cube = np.empty((len(annual_means), rows, cols))
        for index, value in enumerate(annual_means):
            cube[index] = value
        self.dataset = type("D", (), {
            "ndvi": cube, "n_time": len(annual_means),
            "times": [str(y) for y in years]})()
        self.features = pd.DataFrame({"sen": np.full(rows * cols, slope)})


def _confound_setup(tmp_path, annual_means, slope, oli_from=2013,
                    start=1990):
    years = list(range(start, start + len(annual_means)))
    scenes = []
    for year in years:
        sensor = ("LANDSAT8_OLI" if year >= oli_from else "LANDSAT5_TM")
        scenes.append({"date": f"{year}-11-01", "sensor": sensor,
                       "scene_id": f"s{year}"})
    raw = tmp_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "scenes.json").write_text(json.dumps({"scenes": scenes}))

    from src.config import Config
    cfg = Config()
    cfg.real_data.raw_dir = str(raw)

    class FakeExperiment:
        def path(self, subdir, filename=""):
            target = tmp_path / subdir
            target.mkdir(parents=True, exist_ok=True)
            return target / filename if filename else target

    messages = []

    class FakeLog:
        def info(self, *args, **kwargs):
            messages.append(("info", args))

        def warning(self, *args, **kwargs):
            messages.append(("warning", args))

    return (ConfoundPrepared(annual_means, slope, years), cfg,
            FakeExperiment(), FakeLog(), messages)


def test_a_clean_record_reports_no_detectable_step(tmp_path):
    rng = np.random.default_rng(0)
    means = list(0.65 + rng.normal(0, 0.002, 35))
    prepared, cfg, exp, log, _ = _confound_setup(tmp_path, means, 0.002)
    result = M7.stage_sensor_confound(prepared, cfg, exp, log)
    assert result["assessed"] is True
    assert abs(result["step_ndvi"]) < 0.01
    assert result["step_is_statistically_detectable"] is False
    assert "unlikely to be driving" in result["assessment"]


def test_a_planted_step_is_detected_and_compared_to_the_trend(tmp_path):
    """A step at the instrument change is the failure mode that matters."""
    means = [0.60] * 23 + [0.70] * 12          # a clean +0.10 step at 2013
    prepared, cfg, exp, log, messages = _confound_setup(tmp_path, means,
                                                        0.0001)
    result = M7.stage_sensor_confound(prepared, cfg, exp, log)
    assert result["step_ndvi"] == pytest.approx(0.10, abs=1e-9)
    assert result["step_is_statistically_detectable"] is True
    assert result["first_year_with_oli"] == 2013
    # The step dwarfs the per-pixel slope, so the verdict must be the strong one.
    assert result["step_equivalent_over_median_slope"] > 1
    assert "CANNOT BE SEPARATED" in result["assessment"]
    assert any(kind == "warning" for kind, _ in messages)


def test_a_step_small_against_the_trend_is_not_over_claimed(tmp_path):
    means = [0.60] * 23 + [0.602] * 12         # tiny step
    prepared, cfg, exp, log, _ = _confound_setup(tmp_path, means, 0.01)
    result = M7.stage_sensor_confound(prepared, cfg, exp, log)
    assert result["step_equivalent_over_median_slope"] < 0.5
    assert "unlikely to be driving" in result["assessment"]


def test_the_confound_is_skipped_cleanly_without_a_manifest(tmp_path):
    prepared, cfg, exp, log, _ = _confound_setup(tmp_path, [0.6] * 35, 0.001)
    cfg.real_data.raw_dir = str(tmp_path / "absent")
    result = M7.stage_sensor_confound(prepared, cfg, exp, log)
    assert result["assessed"] is False
    assert "manifest" in result["reason"]


def test_a_record_without_oli_scenes_is_reported_as_unassessed(tmp_path):
    prepared, cfg, exp, log, _ = _confound_setup(
        tmp_path, [0.6] * 20, 0.001, oli_from=9999, start=1990)
    result = M7.stage_sensor_confound(prepared, cfg, exp, log)
    assert result["assessed"] is False
    assert "OLI" in result["reason"]


def test_the_confound_record_states_its_own_limit(tmp_path):
    means = [0.60] * 23 + [0.65] * 12
    prepared, cfg, exp, log, _ = _confound_setup(tmp_path, means, 0.001)
    result = M7.stage_sensor_confound(prepared, cfg, exp, log)
    assert "REGIONAL-MEAN" in result["caveat"]
    assert "cover-specific" in result["caveat"]
    assert "Roy" in result["harmonisation_applied"]
