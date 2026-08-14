"""Scientific validation of the trend estimators.

These tests assert STATISTICAL BEHAVIOUR on signals whose correct answer is
known analytically, not merely that the functions execute.
"""
import numpy as np
import pytest

from src import timeseries as TS

T = 36
SEED = 42


def col(v):
    return np.asarray(v, dtype="float64").reshape(-1, 1)


# ---------------------------------------------------------------- monotonic
def test_monotonic_increasing_is_significant_and_positive():
    x = col(0.3 + 0.01 * np.arange(T))
    S, var, z, p, tau = TS.mann_kendall(x)
    assert S[0] > 0
    assert z[0] > 0
    assert p[0] < 0.01, "a perfect increasing ramp must be significant"
    assert tau[0] == pytest.approx(1.0), "perfect monotonic => tau = 1"


def test_monotonic_decreasing_is_significant_and_negative():
    x = col(0.8 - 0.01 * np.arange(T))
    S, var, z, p, tau = TS.mann_kendall(x)
    assert S[0] < 0 and z[0] < 0
    assert p[0] < 0.01
    assert tau[0] == pytest.approx(-1.0)


def test_sens_slope_recovers_known_slope():
    true_slope = -0.013
    x = col(0.75 + true_slope * np.arange(T))
    assert TS.sens_slope(x)[0] == pytest.approx(true_slope, rel=1e-6)


def test_sens_slope_robust_to_outliers():
    """Theil-Sen must resist gross outliers far better than OLS."""
    base = 0.5 + 0.01 * np.arange(T)
    contaminated = base.copy()
    contaminated[[5, 17]] = [5.0, -5.0]
    sen = TS.sens_slope(col(contaminated))[0]
    t = np.arange(T)
    ols = np.polyfit(t, contaminated, 1)[0]
    assert abs(sen - 0.01) < abs(ols - 0.01)
    assert sen == pytest.approx(0.01, abs=2e-3)


# ---------------------------------------------------------------- constant
def test_constant_series_gives_no_trend():
    x = col(np.full(T, 0.5))
    S, var, z, p, tau = TS.mann_kendall(x)
    assert S[0] == 0
    assert z[0] == 0.0
    assert p[0] == pytest.approx(1.0), "constant series must not be a trend"
    assert TS.sens_slope(x)[0] == pytest.approx(0.0)


def test_constant_series_tie_correction_zeroes_variance():
    """All values tied => the tie term cancels the variance entirely."""
    x = col(np.full(T, 0.5))
    _, var, _, _, _ = TS.mann_kendall(x, apply_ties=True)
    assert not np.isfinite(var[0]), "fully tied series has no valid variance"


# ---------------------------------------------------------------- noise
def test_pure_noise_is_usually_not_significant():
    """False-positive rate on white noise must be near the nominal alpha."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(0.5, 0.05, size=(T, 300))
    _, _, _, p, _ = TS.mann_kendall(x)
    false_pos = np.nanmean(p < 0.05)
    assert false_pos < 0.12, f"false-positive rate {false_pos:.3f} too high"


def test_trend_detected_through_noise():
    rng = np.random.default_rng(SEED)
    x = (0.6 - 0.012 * np.arange(T))[:, None] + \
        rng.normal(0, 0.02, size=(T, 50))
    _, _, z, p, _ = TS.mann_kendall(x)
    assert np.nanmean(p < 0.05) > 0.9
    assert np.all(z < 0)


# ------------------------------------------------- autocorrelation / Hamed-Rao
def test_hamed_rao_inflates_variance_for_autocorrelated_noise():
    """AR(1) noise inflates MK significance; the correction must reduce it."""
    rng = np.random.default_rng(SEED)
    n_pix, rho = 400, 0.75
    x = np.zeros((T, n_pix))
    x[0] = rng.normal(0, 0.05, n_pix)
    for i in range(1, T):
        x[i] = rho * x[i - 1] + rng.normal(0, 0.05, n_pix)
    x += 0.5

    S, var, z, p, _ = TS.mann_kendall(x)
    _, p_corr = TS.hamed_rao_correction(x, S, var)
    raw_rate = np.nanmean(p < 0.05)
    corr_rate = np.nanmean(p_corr < 0.05)
    assert corr_rate < raw_rate, (
        f"correction did not reduce false positives "
        f"({raw_rate:.3f} -> {corr_rate:.3f})")


def test_hamed_rao_leaves_white_noise_roughly_unchanged():
    rng = np.random.default_rng(SEED)
    x = rng.normal(0.5, 0.05, size=(T, 200))
    S, var, z, p, _ = TS.mann_kendall(x)
    z_corr, _ = TS.hamed_rao_correction(x, S, var)
    assert np.nanmean(np.abs(z_corr - z)) < 0.5


# ---------------------------------------------------------------- missing data
def test_nan_aware_trend_matches_complete_series():
    full = 0.3 + 0.01 * np.arange(T)
    gapped = full.copy()
    gapped[[4, 11, 23]] = np.nan
    s_full = TS.sens_slope(col(full))[0]
    s_gap = TS.sens_slope(col(gapped))[0]
    assert s_gap == pytest.approx(s_full, abs=1e-6)
    _, _, _, p, _ = TS.mann_kendall(col(gapped))
    assert p[0] < 0.01


def test_insufficient_observations_returns_nan_not_a_number():
    x = np.full((T, 1), np.nan)
    x[:5, 0] = [0.1, 0.2, 0.3, 0.4, 0.5]
    _, _, z, p, tau = TS.mann_kendall(x, min_obs=10)
    assert np.isnan(z[0]) and np.isnan(p[0]) and np.isnan(tau[0]), \
        "must refuse to report a trend from 5 observations"


def test_all_nan_pixel_is_nan():
    x = np.full((T, 1), np.nan)
    _, _, z, p, _ = TS.mann_kendall(x)
    assert np.isnan(z[0]) and np.isnan(p[0])
    assert np.isnan(TS.sens_slope(x)[0])


# ---------------------------------------------------------------- input guards
def test_rejects_too_short_series():
    with pytest.raises(ValueError):
        TS.mann_kendall(np.array([[0.1], [0.2]]))


def test_accepts_1d_input():
    S, _, z, p, _ = TS.mann_kendall(0.3 + 0.01 * np.arange(T))
    assert z.shape == (1,) and p[0] < 0.01


def test_vectorization_matches_per_pixel_loop():
    """The vectorized path must equal column-by-column evaluation."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(0.5, 0.1, size=(T, 12))
    _, _, z_vec, _, _ = TS.mann_kendall(x)
    z_loop = np.array([TS.mann_kendall(x[:, [j]])[2][0]
                       for j in range(x.shape[1])])
    assert np.allclose(z_vec, z_loop, equal_nan=True)


def test_matches_pymannkendall_reference():
    """Cross-validate against the independent reference implementation."""
    pmk = pytest.importorskip("pymannkendall")
    rng = np.random.default_rng(SEED)
    x = rng.normal(0.5, 0.1, size=(T, 8))
    _, _, z, p, tau = TS.mann_kendall(x)
    for j in range(x.shape[1]):
        ref = pmk.original_test(x[:, j])
        assert z[j] == pytest.approx(ref.z, abs=1e-6)
        assert tau[j] == pytest.approx(ref.Tau, abs=1e-6)
        assert p[j] == pytest.approx(ref.p, abs=1e-6)
