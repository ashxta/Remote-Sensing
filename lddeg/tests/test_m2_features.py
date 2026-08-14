"""Feature-engineering contract tests (M2 Part 12).

The pipeline must produce documented columns, in a stable order, from the
same inputs every time, and must degrade to NaN rather than to a fabricated
value when observations are missing or too few.
"""
import numpy as np
import pytest

from src.classify import FEATURES as M1_FEATURES
from src.config import Config
from src.features import (DIAGNOSTIC_COLUMNS, FEATURE_GROUPS, FEATURE_SPECS,
                          GROUP_ORDER, build_feature_table,
                          feature_dictionary, feature_names)

T = 36


def synth(n=40, seed=7, holes=0.0):
    """Small stack with stable, trending, periodic and disturbed pixels."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    columns = []
    for i in range(n):
        kind = i % 4
        if kind == 0:
            s = 0.75 + rng.normal(0, 0.02, T)
        elif kind == 1:
            s = 0.6 - 0.012 * t + rng.normal(0, 0.02, T)
        elif kind == 2:
            s = 0.5 + 0.18 * np.sin(2 * np.pi * t / 7.0) \
                + rng.normal(0, 0.02, T)
        else:
            k = 14
            s = np.concatenate([np.full(k, 0.7),
                                np.linspace(0.3, 0.68, T - k)]) \
                + rng.normal(0, 0.02, T)
        columns.append(s)
    ndvi = np.clip(np.array(columns).T, 0.02, 0.95)
    rain = rng.normal(1800, 200, size=(T, n))
    if holes:
        ndvi = ndvi.copy()
        ndvi[rng.random(ndvi.shape) < holes] = np.nan
    return ndvi, rain


# ------------------------------------------------------------------ contract
def test_every_feature_is_documented_and_unique():
    names = [s.name for s in FEATURE_SPECS]
    assert len(names) == len(set(names)), "duplicate feature specification"
    for spec in FEATURE_SPECS:
        assert spec.group in GROUP_ORDER
        assert spec.description and spec.source


def test_m1_feature_block_is_preserved():
    """M2 must not drop or rename any M1 feature."""
    produced = set(feature_names(GROUP_ORDER))
    assert set(M1_FEATURES).issubset(produced)


def test_expected_feature_names_and_dimensions():
    ndvi, rain = synth(n=25)
    table, extras = build_feature_table(ndvi, rain, Config())
    assert list(table.columns) == feature_names(GROUP_ORDER)
    assert table.shape == (25, len(feature_names(GROUP_ORDER)))
    assert "recovery" in extras and "mk_p" in extras


def test_feature_groups_partition_the_table():
    covered = [name for group in GROUP_ORDER for name in FEATURE_GROUPS[group]]
    assert sorted(covered) == sorted(set(covered))
    assert set(covered) == {s.name for s in FEATURE_SPECS}


def test_requested_groups_control_the_design_matrix():
    only_vegetation = feature_names(["vegetation"])
    assert set(only_vegetation) == set(FEATURE_GROUPS["vegetation"])
    assert "mk_z" not in only_vegetation
    assert "cyc_score" not in only_vegetation


def test_default_feature_names_exclude_diagnostics():
    default = feature_names()
    for column in DIAGNOSTIC_COLUMNS:
        assert column not in default


def test_unknown_group_is_rejected():
    with pytest.raises(ValueError, match="unknown feature group"):
        feature_names(["vegetation", "not_a_group"])


def test_feature_dictionary_documents_every_column():
    dictionary = feature_dictionary()
    assert set(dictionary["feature"]) == {s.name for s in FEATURE_SPECS}
    assert dictionary["description"].str.len().min() > 0


# --------------------------------------------------------------- determinism
def test_feature_generation_is_deterministic():
    ndvi, rain = synth()
    a, _ = build_feature_table(ndvi, rain, Config())
    b, _ = build_feature_table(ndvi, rain, Config())
    assert np.array_equal(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_features_are_traceable_to_configuration():
    """Changing a configured threshold must change the output."""
    ndvi, rain = synth()
    strict, lenient = Config(), Config()
    strict.trend.alpha, lenient.trend.alpha = 0.001, 0.5
    a, _ = build_feature_table(ndvi, rain, strict)
    b, _ = build_feature_table(ndvi, rain, lenient)
    assert not np.array_equal(a["mk_significant"].to_numpy(),
                              b["mk_significant"].to_numpy())


def test_pixel_order_does_not_change_features():
    ndvi, rain = synth(n=20)
    order = np.arange(20)[::-1]
    a, _ = build_feature_table(ndvi, rain, Config())
    b, _ = build_feature_table(ndvi[:, order], rain[:, order], Config())
    assert np.allclose(a.to_numpy()[order], b.to_numpy(), equal_nan=True,
                       atol=1e-6)


# ---------------------------------------------------------------- robustness
def test_missing_observations_do_not_crash_or_fabricate():
    ndvi, rain = synth(holes=0.15)
    table, _ = build_feature_table(ndvi, rain, Config())
    assert len(table) == ndvi.shape[1]
    assert table["n_valid_ndvi"].max() <= T


def test_insufficient_observations_produce_nan_not_a_guess():
    """A pixel below TrendConfig.min_obs must not receive a trend value."""
    ndvi, rain = synth(n=4)
    ndvi = ndvi.copy()
    ndvi[5:, 0] = np.nan                       # 5 valid observations left
    table, _ = build_feature_table(ndvi, rain, Config())
    assert table["n_valid_ndvi"].iloc[0] == 5
    assert np.isnan(table["sen"].iloc[0])
    assert np.isnan(table["mk_z"].iloc[0])
    assert np.isfinite(table["sen"].iloc[1]), "other pixels stay unaffected"


def test_all_nan_pixel_does_not_raise():
    ndvi, rain = synth(n=3)
    ndvi = ndvi.copy()
    ndvi[:, 0] = np.nan
    table, _ = build_feature_table(ndvi, rain, Config())
    assert np.isnan(table["mean"].iloc[0])
    assert table["n_valid_ndvi"].iloc[0] == 0


def test_constant_series_does_not_produce_infinities():
    ndvi = np.full((T, 2), 0.5)
    rain = np.full((T, 2), 1500.0)
    table, _ = build_feature_table(ndvi, rain, Config())
    assert not np.isinf(table.to_numpy()).any()


def test_structural_encoding_is_consistent():
    """Descriptors undefined without an event are 0 next to their flag."""
    ndvi, rain = synth(n=48)
    table, _ = build_feature_table(ndvi, rain, Config())
    no_break = table["has_break"].to_numpy() == 0
    if no_break.any():
        assert np.all(table["pre_breakpoint_level"].to_numpy()[no_break] == 0)
        assert np.all(table["break_t"].to_numpy()[no_break] == 0)
    no_disturbance = table["has_disturbance"].to_numpy() == 0
    if no_disturbance.any():
        assert np.all(
            table["recovery_magnitude"].to_numpy()[no_disturbance] == 0)


def test_shape_mismatch_is_rejected():
    ndvi, rain = synth(n=6)
    with pytest.raises(ValueError, match="identical shapes"):
        build_feature_table(ndvi, rain[:, :3], Config())


# ------------------------------------------------------------ feature values
def test_descriptive_features_match_their_definition():
    ndvi, rain = synth(n=12)
    table, _ = build_feature_table(ndvi, rain, Config())
    assert np.allclose(table["median_ndvi"], np.nanmedian(ndvi, axis=0),
                       atol=1e-5)
    assert np.allclose(table["minimum_ndvi"], np.nanmin(ndvi, axis=0),
                       atol=1e-5)
    assert np.allclose(table["maximum_ndvi"], np.nanmax(ndvi, axis=0),
                       atol=1e-5)
    assert np.allclose(table["rain_mean"], np.nanmean(rain, axis=0),
                       rtol=1e-4)


def test_ols_slope_recovers_a_known_ramp():
    t = np.arange(T, dtype=float)
    ndvi = np.c_[0.4 + 0.01 * t, 0.6 - 0.005 * t]
    rain = np.full((T, 2), 1500.0)
    table, _ = build_feature_table(ndvi, rain, Config())
    assert table["ols_slope"].iloc[0] == pytest.approx(0.01, abs=1e-5)
    assert table["ols_slope"].iloc[1] == pytest.approx(-0.005, abs=1e-5)


def test_rainfall_correlation_detects_a_planted_relationship():
    rng = np.random.default_rng(3)
    rain = rng.normal(1800, 200, size=(T, 2))
    ndvi = np.c_[0.4 + 0.0002 * rain[:, 0], rng.normal(0.5, 0.02, T)]
    table, _ = build_feature_table(ndvi, rain, Config())
    assert table["rain_ndvi_correlation"].iloc[0] > 0.95
    assert abs(table["rain_ndvi_correlation"].iloc[1]) < 0.7


def test_significance_flags_follow_the_configured_alpha():
    ndvi, rain = synth(n=16)
    cfg = Config()
    table, extras = build_feature_table(ndvi, rain, cfg)
    expected = np.asarray(extras["mk_p"]) < cfg.trend.alpha
    assert np.array_equal(table["mk_significant"].to_numpy().astype(bool),
                          expected)
