"""Scientific validation of RESTREND, cyclicity and breakpoint detection."""
import numpy as np
import pytest

from src import timeseries as TS

T = 36
SEED = 42


def col(v):
    return np.asarray(v, dtype="float64").reshape(-1, 1)


# ================================================================= RESTREND
def test_restrend_flags_rainfall_driven_change_as_valid_and_flat():
    """NDVI driven purely by rainfall: strong beta, no residual trend."""
    rng = np.random.default_rng(SEED)
    rain = rng.normal(1800, 250, size=(T, 1))
    ndvi = 0.3 + 0.0002 * (rain - 1800) + rng.normal(0, 0.005, (T, 1))
    r = TS.restrend(ndvi, rain)
    assert r.valid[0], "a genuine NDVI~rain relationship must be valid"
    assert r.r2[0] > 0.8
    assert r.beta[0] > 0
    assert r.p[0] > 0.05, "no residual trend should remain"


def test_restrend_detects_decline_independent_of_rainfall():
    """Rainfall-explained NDVI plus an imposed decline: residual trend < 0."""
    rng = np.random.default_rng(SEED)
    rain = rng.normal(1800, 250, size=(T, 1))
    decline = -0.008 * np.arange(T)[:, None]
    ndvi = 0.6 + 0.0002 * (rain - 1800) + decline + \
        rng.normal(0, 0.005, (T, 1))
    r = TS.restrend(ndvi, rain)
    assert r.valid[0]
    assert r.slope[0] < 0
    assert r.p[0] < 0.01


def test_restrend_invalid_when_rainfall_relationship_is_absent():
    """No NDVI~rain link => RESTREND is not interpretable as climate-adjusted."""
    rng = np.random.default_rng(SEED)
    rain = rng.normal(1800, 250, size=(T, 1))
    ndvi = 0.5 - 0.008 * np.arange(T)[:, None] + rng.normal(0, 0.01, (T, 1))
    r = TS.restrend(ndvi, rain)
    assert not r.valid[0], (
        "must refuse to call this climate-adjusted: rainfall explains nothing")


def test_restrend_invalid_for_negative_beta_under_water_limited_assumption():
    rng = np.random.default_rng(SEED)
    rain = rng.normal(1800, 250, size=(T, 1))
    ndvi = 0.5 - 0.0003 * (rain - 1800) + rng.normal(0, 0.005, (T, 1))
    r = TS.restrend(ndvi, rain, require_positive_beta=True)
    assert r.beta[0] < 0
    assert not r.valid[0]


def test_restrend_constant_rainfall_is_not_valid():
    """Degenerate predictor must not silently produce a beta."""
    rng = np.random.default_rng(SEED)
    rain = np.full((T, 1), 1800.0)
    ndvi = 0.5 + rng.normal(0, 0.01, (T, 1))
    r = TS.restrend(ndvi, rain)
    assert not r.valid[0]


def test_restrend_shape_mismatch_raises():
    with pytest.raises(ValueError):
        TS.restrend(np.zeros((T, 2)), np.zeros((T, 3)))


def test_restrend_handles_missing_observations():
    rng = np.random.default_rng(SEED)
    rain = rng.normal(1800, 250, size=(T, 1))
    ndvi = 0.5 + 0.0002 * (rain - 1800) + rng.normal(0, 0.005, (T, 1))
    ndvi[[3, 9, 20]] = np.nan
    r = TS.restrend(ndvi, rain)
    assert r.n_obs[0] == T - 3
    assert np.isfinite(r.slope[0])


def test_restrend_insufficient_observations_returns_nan():
    ndvi = np.full((T, 1), np.nan)
    rain = np.full((T, 1), np.nan)
    ndvi[:5, 0] = np.linspace(0.3, 0.5, 5)
    rain[:5, 0] = np.linspace(1500, 2000, 5)
    r = TS.restrend(ndvi, rain, min_obs=10)
    assert np.isnan(r.slope[0])
    assert not r.valid[0]


# =============================================================== CYCLICITY
def test_known_period_is_recovered():
    for period in (5.0, 6.0, 9.0):
        t = np.arange(T)
        x = col(0.5 + 0.2 * np.sin(2 * np.pi * t / period))
        c = TS.cyclicity(x, min_period=4, max_period=12)
        assert c.periodic[0], f"period {period} should be detected"
        assert c.dominant_period[0] == pytest.approx(period, abs=1.2)
        assert c.score[0] > 0.5


def test_non_periodic_signal_scores_low():
    rng = np.random.default_rng(SEED)
    x = rng.normal(0.5, 0.05, size=(T, 200))
    c = TS.cyclicity(x, min_period=4, max_period=12)
    assert np.nanmean(c.periodic) < 0.35, \
        "white noise must rarely be called periodic"


def test_pure_trend_is_not_periodic_after_detrending():
    x = col(0.3 + 0.01 * np.arange(T))
    c = TS.cyclicity(x, min_period=4, max_period=12, detrend=True)
    assert not c.periodic[0], "a linear ramp is not a cycle"


def test_period_outside_band_is_not_flagged():
    """A 2.5-step cycle must not register in a 4-12 step band.

    (A period of exactly 2 is the Nyquist limit and aliases to zero at
    integer sampling, so 2.5 is used as a genuine out-of-band signal.)
    """
    t = np.arange(T)
    x = col(0.5 + 0.2 * np.sin(2 * np.pi * t / 2.5))
    c = TS.cyclicity(x, min_period=4, max_period=12)
    assert not c.periodic[0]
    assert c.enrichment[0] < 1.0


def test_white_noise_enrichment_is_about_one():
    """Enrichment is normalised so the white-noise null sits at 1.0."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(0.5, 0.05, size=(T, 400))
    c = TS.cyclicity(x, min_period=4, max_period=12)
    assert 0.85 < np.nanmean(c.enrichment) < 1.15
    assert c.expected_fraction == pytest.approx(7 / 18, abs=1e-9)


def test_cyclicity_band_is_configurable():
    t = np.arange(T)
    x = col(0.5 + 0.2 * np.sin(2 * np.pi * t / 3.0))
    narrow = TS.cyclicity(x, min_period=4, max_period=12)
    wide = TS.cyclicity(x, min_period=2, max_period=12)
    assert wide.score[0] > narrow.score[0]
    assert wide.dominant_period[0] == pytest.approx(3.0, abs=0.6)


def test_cyclicity_exposes_power_components():
    t = np.arange(T)
    x = col(0.5 + 0.2 * np.sin(2 * np.pi * t / 6.0))
    c = TS.cyclicity(x)
    assert c.band_power[0] > 0 and c.total_power[0] > 0
    assert c.band_power[0] <= c.total_power[0] * 1.0000001
    assert c.score[0] == pytest.approx(c.band_power[0] / c.total_power[0])


def test_cyclicity_rejects_invalid_band():
    with pytest.raises(ValueError):
        TS.cyclicity(np.zeros((T, 1)), min_period=12, max_period=4)
    with pytest.raises(ValueError):
        TS.cyclicity(np.zeros((T, 1)), min_period=0.5, max_period=4)


def test_cyclicity_refuses_series_shorter_than_max_period():
    x = np.random.default_rng(SEED).normal(0.5, 0.05, size=(8, 1))
    c = TS.cyclicity(x, min_period=4, max_period=12)
    assert np.isnan(c.score[0]), "cannot assess a 12-yr cycle in 8 steps"


def test_cyclicity_is_deterministic():
    rng = np.random.default_rng(SEED)
    x = rng.normal(0.5, 0.05, size=(T, 20))
    a = TS.cyclicity(x)
    b = TS.cyclicity(x)
    assert np.allclose(a.score, b.score, equal_nan=True)


# ============================================================== BREAKPOINT
def test_known_breakpoint_is_located():
    true_k = 20
    x = np.concatenate([np.full(true_k, 0.7),
                        0.7 - 0.03 * np.arange(T - true_k)])
    b = TS.best_breakpoint(col(x), min_segment=4)
    assert abs(int(b.index[0]) - true_k) <= 2
    assert b.significant[0]
    assert b.delta_slope[0] < 0, "slope must turn downward at the break"


def test_upward_break_has_positive_delta_slope():
    true_k = 15
    x = np.concatenate([np.full(true_k, 0.3),
                        0.3 + 0.02 * np.arange(T - true_k)])
    b = TS.best_breakpoint(col(x), min_segment=4)
    assert b.delta_slope[0] > 0
    assert b.significant[0]


def test_no_break_in_a_straight_line():
    x = col(0.3 + 0.01 * np.arange(T))
    b = TS.best_breakpoint(x, min_segment=4, min_gain=0.05)
    assert not b.significant[0], "a straight line has no structural break"


def test_no_break_in_pure_noise_usually():
    rng = np.random.default_rng(SEED)
    x = rng.normal(0.5, 0.05, size=(T, 200))
    b = TS.best_breakpoint(x, min_segment=4, min_gain=0.05, alpha=0.05)
    assert np.mean(b.significant) < 0.25


def test_minimum_segment_length_is_enforced():
    x = np.random.default_rng(SEED).normal(0.5, 0.05, size=(T, 30))
    m = 8
    b = TS.best_breakpoint(x, min_segment=m)
    found = b.index[b.index >= 0]
    assert np.all(found >= m)
    assert np.all(found <= T - m)


def test_series_too_short_returns_minus_one_not_zero():
    """The original code returned index 0 for unresolvable pixels."""
    x = np.random.default_rng(SEED).normal(0.5, 0.05, size=(7, 3))
    b = TS.best_breakpoint(x, min_segment=4)
    assert np.all(b.index == -1)
    assert not b.significant.any()


def test_breakpoint_rejects_tiny_min_segment():
    with pytest.raises(ValueError):
        TS.best_breakpoint(np.zeros((T, 1)), min_segment=2)


def test_breakpoint_handles_nan():
    true_k = 18
    x = np.concatenate([np.full(true_k, 0.7),
                        0.7 - 0.03 * np.arange(T - true_k)])
    x[[2, 25]] = np.nan
    b = TS.best_breakpoint(col(x), min_segment=4)
    assert abs(int(b.index[0]) - true_k) <= 3
