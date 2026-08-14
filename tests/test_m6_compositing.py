"""Temporal compositing (M6 Parts 5, 8, 12, 13).

The compositing step converts irregular acquisitions into the regular time
axis every M1-M5 estimator assumes. These tests pin down the two things that
would silently corrupt a long record: which observations land in which
window, and what happens when a window has none.
"""
import datetime as dt

import numpy as np
import pytest

from src.compositing import (COMPOSITE_STATISTICS, CompositeWindow,
                             CompositingError, annual_windows, as_date,
                             build_windows, composite_observations,
                             describe_temporal_design, monthly_windows,
                             seasonal_windows)
from src.config import RealDataConfig


# ----------------------------------------------------------------- windows
def test_annual_windows_give_one_step_per_year():
    windows = annual_windows(1990, 2025)
    assert len(windows) == 36
    assert [w.label for w in windows][:3] == ["1990", "1991", "1992"]
    assert windows[0].start == dt.date(1990, 10, 15)
    assert windows[0].end == dt.date(1990, 12, 31)


def test_a_window_crossing_new_year_closes_in_the_following_year():
    """A Nov-Feb season must still produce ONE step per year, not two."""
    windows = annual_windows(2000, 2002, window_start="11-15",
                             window_end="02-15")
    assert len(windows) == 3
    assert windows[0].start == dt.date(2000, 11, 15)
    assert windows[0].end == dt.date(2001, 2, 15)
    assert windows[0].label == "2000"


def test_the_composite_window_is_the_same_season_every_year():
    """Sampling a different phenological stage each year fakes a trend."""
    windows = annual_windows(1990, 2025)
    starts = {(w.start.month, w.start.day) for w in windows}
    ends = {(w.end.month, w.end.day) for w in windows}
    assert len(starts) == 1 and len(ends) == 1


def test_seasonal_and_monthly_units_are_available_and_correctly_sized():
    assert len(seasonal_windows(2000, 2002)) == 12
    assert len(monthly_windows(2000, 2002)) == 36
    assert monthly_windows(2000, 2000)[-1].end == dt.date(2000, 12, 31)


def test_build_windows_dispatches_and_explains_an_unknown_unit():
    assert len(build_windows("annual", 2000, 2004)) == 5
    with pytest.raises(CompositingError, match="cyclicity"):
        build_windows("weekly", 2000, 2004)


def test_a_reversed_year_range_is_rejected():
    with pytest.raises(CompositingError, match="precedes"):
        annual_windows(2020, 2010)


def test_window_membership_is_inclusive_of_both_ends():
    window = CompositeWindow("2000", dt.date(2000, 10, 15),
                             dt.date(2000, 12, 31), 2000)
    assert window.contains(dt.date(2000, 10, 15))
    assert window.contains(dt.date(2000, 12, 31))
    assert not window.contains(dt.date(2000, 10, 14))
    assert not window.contains(dt.date(2001, 1, 1))


def test_dates_are_accepted_in_the_forms_a_manifest_may_use():
    expected = dt.date(2005, 6, 1)
    for value in ("2005-06-01", "2005-06-01T10:30:00", expected,
                  dt.datetime(2005, 6, 1, 10, 30),
                  np.datetime64("2005-06-01")):
        assert as_date(value) == expected
    with pytest.raises(CompositingError, match="cannot interpret"):
        as_date(12345)


# ------------------------------------------------------------- compositing
def three_windows():
    return annual_windows(2000, 2002)


def test_observations_land_in_the_window_containing_their_date():
    observations = [np.full((2, 2), 0.2), np.full((2, 2), 0.8),
                    np.full((2, 2), 0.5)]
    dates = ["2000-11-01", "2001-11-01", "2002-11-01"]
    result = composite_observations(observations, dates, three_windows())
    assert np.allclose(result.values[:, 0, 0], [0.2, 0.8, 0.5])
    assert list(result.n_scenes) == [1, 1, 1]


def test_observations_outside_every_window_are_simply_not_used():
    """A July acquisition must not leak into a post-monsoon composite."""
    observations = [np.full((2, 2), 0.9), np.full((2, 2), 0.2)]
    dates = ["2000-07-01", "2000-11-01"]
    result = composite_observations(observations, dates, three_windows())
    assert np.allclose(result.values[0], 0.2)
    assert result.n_scenes[0] == 1


def test_the_median_ignores_a_residual_outlier():
    observations = [np.full((1, 1), 0.70), np.full((1, 1), 0.72),
                    np.full((1, 1), 0.05)]
    dates = ["2000-11-01", "2000-11-10", "2000-11-20"]
    result = composite_observations(observations, dates, three_windows())
    assert np.isclose(result.values[0, 0, 0], 0.70)


@pytest.mark.parametrize("statistic,expected", [
    ("median", 0.70), ("mean", (0.70 + 0.72 + 0.05) / 3),
    ("max", 0.72), ("min", 0.05)])
def test_each_statistic_aggregates_as_documented(statistic, expected):
    observations = [np.full((1, 1), 0.70), np.full((1, 1), 0.72),
                    np.full((1, 1), 0.05)]
    dates = ["2000-11-01", "2000-11-10", "2000-11-20"]
    result = composite_observations(observations, dates, three_windows(),
                                    statistic=statistic)
    assert np.isclose(result.values[0, 0, 0], expected)
    assert result.statistic == statistic


def test_percentile_compositing_is_supported():
    observations = [np.full((1, 1), float(v)) for v in range(10)]
    dates = [f"2000-11-{d:02d}" for d in range(1, 11)]
    result = composite_observations(observations, dates, three_windows(),
                                    statistic="percentile", percentile=90.0)
    assert result.values[0, 0, 0] > 7.5
    assert result.metadata["percentile"] == 90.0


def test_an_unknown_statistic_is_refused():
    with pytest.raises(CompositingError, match="unknown compositing"):
        composite_observations([np.zeros((1, 1))], ["2000-11-01"],
                               three_windows(), statistic="mode")
    assert "median" in COMPOSITE_STATISTICS


def test_a_window_with_no_observation_is_nan_and_is_counted():
    """The core missing-data rule: gaps are recorded, never filled."""
    observations = [np.full((2, 2), 0.5)]
    result = composite_observations(observations, ["2000-11-01"],
                                    three_windows())
    assert np.isfinite(result.values[0]).all()
    assert np.isnan(result.values[1]).all()
    assert np.isnan(result.values[2]).all()
    assert list(result.n_scenes) == [1, 0, 0]
    summary = result.summary()
    assert summary["windows_with_no_valid_observation_anywhere"] == ["2001",
                                                                    "2002"]
    assert "never" not in summary["statistic"]


def test_masked_pixels_do_not_contribute_and_are_counted_separately():
    first = np.array([[0.5, np.nan]])
    second = np.array([[0.5, 0.9]])
    result = composite_observations([first, second],
                                    ["2000-11-01", "2000-11-10"],
                                    three_windows())
    assert np.isclose(result.values[0, 0, 0], 0.5)
    assert np.isclose(result.values[0, 0, 1], 0.9)     # only the valid one
    assert result.n_valid[0, 0, 0] == 2
    assert result.n_valid[0, 0, 1] == 1
    assert result.n_masked[0, 0, 1] == 1


def test_supplied_mask_counts_are_preserved_verbatim():
    """Distinguishes "cloudy" from "never overflown" in the report."""
    result = composite_observations(
        [np.array([[0.5]])], ["2000-11-01"], three_windows(),
        masked_counts=[np.array([[7]])])
    assert result.n_masked[0, 0, 0] == 7


def test_min_observations_rejects_thin_composites_visibly():
    observations = [np.full((1, 1), 0.5)]
    result = composite_observations(observations, ["2000-11-01"],
                                    three_windows(), min_observations=2)
    assert np.isnan(result.values[0, 0, 0])
    assert result.n_valid[0, 0, 0] == 1           # the count still records it
    assert result.metadata["min_observations_per_composite"] == 2


def test_the_default_keeps_a_single_clear_observation():
    result = composite_observations([np.full((1, 1), 0.5)], ["2000-11-01"],
                                    three_windows())
    assert np.isfinite(result.values[0, 0, 0])


def test_inconsistent_scene_grids_are_refused():
    with pytest.raises(CompositingError, match="inconsistent grids"):
        composite_observations([np.zeros((2, 2)), np.zeros((3, 3))],
                               ["2000-11-01", "2000-11-02"], three_windows())


def test_mismatched_dates_and_observations_are_refused():
    with pytest.raises(CompositingError, match="but 2 dates"):
        composite_observations([np.zeros((2, 2))],
                               ["2000-11-01", "2000-11-02"], three_windows())


def test_no_windows_is_an_error():
    with pytest.raises(CompositingError, match="no composite windows"):
        composite_observations([np.zeros((2, 2))], ["2000-11-01"], [])


def test_the_time_axis_is_the_window_labels_in_order():
    result = composite_observations([np.zeros((1, 1))], ["2000-11-01"],
                                    three_windows())
    assert result.times == ["2000", "2001", "2002"]


def test_the_summary_reports_missingness_per_step():
    result = composite_observations([np.array([[0.5, np.nan]])],
                                    ["2000-11-01"], three_windows())
    summary = result.summary()
    assert np.isclose(summary["missing_fraction_per_step"]["2000"], 0.5)
    assert summary["missing_fraction_per_step"]["2001"] == 1.0
    assert summary["n_time_steps"] == 3


# ------------------------------------------------------- temporal design
def test_the_temporal_design_record_states_the_gap_policy():
    design = describe_temporal_design(RealDataConfig())
    assert design["temporal_unit"] == "annual"
    assert design["n_time_steps"] == 36
    assert design["compositing_statistic"] == "median"
    assert "never zero-filled or forward-filled" in \
        design["missing_data_handling"]
    assert "cyclicity band" in design["rationale"]


def test_the_temporal_design_follows_the_configuration():
    cfg = RealDataConfig(start_year=2000, end_year=2004,
                         temporal_unit="monthly")
    assert describe_temporal_design(cfg)["n_time_steps"] == 60
