"""Validation of data-quality gating (Part 5) and recovery metrics (Part 4)."""
import numpy as np
import pytest

from src import quality as Q
from src import recovery as R
from src.config import QualityConfig, RecoveryConfig

T = 36
SEED = 42


# ================================================================== QUALITY
def test_complete_series_is_ok():
    x = np.random.default_rng(SEED).normal(0.5, 0.05, size=(T, 5))
    rep = Q.assess(x, QualityConfig())
    assert np.all(rep.flag == Q.FLAG_OK)
    assert np.all(rep.valid_count == T)
    assert np.all(rep.missing_fraction == 0.0)
    assert rep.usable.all()


def test_missing_observations_are_counted_not_hidden():
    x = np.random.default_rng(SEED).normal(0.5, 0.05, size=(T, 1))
    x[[1, 4, 9], 0] = np.nan
    rep = Q.assess(x, QualityConfig())
    assert rep.valid_count[0] == T - 3
    assert rep.missing_fraction[0] == pytest.approx(3 / T)


def test_insufficient_observations_flagged():
    x = np.full((T, 1), np.nan)
    x[:6, 0] = np.linspace(0.3, 0.6, 6)
    rep = Q.assess(x, QualityConfig(min_valid_obs=10))
    assert rep.flag[0] == Q.FLAG_INSUFFICIENT_OBS
    assert not rep.usable[0]


def test_too_many_missing_flagged():
    x = np.full((T, 1), np.nan)
    x[:15, 0] = np.linspace(0.3, 0.6, 15)     # 58% missing
    rep = Q.assess(x, QualityConfig(min_valid_obs=10,
                                    max_missing_fraction=0.4))
    assert rep.flag[0] == Q.FLAG_TOO_MANY_MISSING


def test_constant_series_flagged():
    x = np.full((T, 1), 0.5)
    rep = Q.assess(x, QualityConfig())
    assert rep.flag[0] == Q.FLAG_CONSTANT
    assert not rep.usable[0]


def test_out_of_physical_range_becomes_missing():
    """NDVI outside [-1, 1] is a scaling error and must not enter analysis."""
    x = np.random.default_rng(SEED).normal(0.5, 0.05, size=(T, 1))
    x[[2, 3], 0] = [5.0, -9999.0]
    rep = Q.assess(x, QualityConfig())
    assert rep.valid_count[0] == T - 2


def test_all_invalid_pixel_flagged():
    x = np.full((T, 1), -9999.0)
    rep = Q.assess(x, QualityConfig())
    assert not rep.usable[0]
    assert rep.valid_count[0] == 0


def test_interpolation_is_off_by_default():
    x = np.random.default_rng(SEED).normal(0.5, 0.05, size=(T, 1))
    x[5, 0] = np.nan
    filled, n = Q.interpolate_gaps(x, QualityConfig())
    assert n == 0
    assert np.isnan(filled[5, 0]), "must not interpolate unless asked"


def test_interpolation_fills_only_short_interior_gaps():
    x = np.linspace(0.3, 0.7, T).reshape(T, 1).copy()
    x[5, 0] = np.nan                 # 1-step gap  -> fill
    x[10:15, 0] = np.nan             # 5-step gap  -> leave
    x[0, 0] = np.nan                 # leading     -> never extrapolate
    cfg = QualityConfig(allow_interpolation=True, max_interpolation_gap=2)
    filled, n = Q.interpolate_gaps(x, cfg)
    assert np.isfinite(filled[5, 0]) and n == 1
    assert np.all(np.isnan(filled[10:15, 0])), "long gap must stay missing"
    assert np.isnan(filled[0, 0]), "leading gap must not be extrapolated"


def test_quality_summary_reports_counts():
    x = np.random.default_rng(SEED).normal(0.5, 0.05, size=(T, 10))
    x[:, 0] = 0.5                                  # constant
    x[:, 1] = np.nan                               # empty
    s = Q.assess(x, QualityConfig()).summary()
    assert s["n_pixels"] == 10
    assert s["n_usable"] == 8
    assert s["flag_counts"]["CONSTANT"] == 1


def test_assess_rejects_wrong_dimensions():
    with pytest.raises(ValueError):
        Q.assess(np.zeros((3, 4, 5)), QualityConfig())


# ================================================================= RECOVERY
def build_disturbance(break_at=15, pre=0.75, drop=0.45, rate=0.03, n=T):
    """Flat pre-level, sharp drop at break_at, then linear regrowth."""
    x = np.full(n, pre)
    x[break_at] = pre - drop
    for i in range(break_at + 1, n):
        x[i] = min(pre, (pre - drop) + rate * (i - break_at))
    return x.reshape(-1, 1)


def test_recovery_detects_pre_level_and_trough():
    x = build_disturbance(break_at=15, pre=0.75, drop=0.45)
    rec = R.analyze(x, np.array([15]), RecoveryConfig())
    assert rec["pre_level"][0] == pytest.approx(0.75, abs=1e-6)
    assert rec["trough_value"][0] == pytest.approx(0.30, abs=1e-6)
    assert rec["trough_index"][0] == 15
    assert rec["disturbance_magnitude"][0] == pytest.approx(0.45, abs=1e-6)


def test_full_recovery_is_labelled_recovered():
    x = build_disturbance(break_at=12, pre=0.75, drop=0.45, rate=0.05)
    rec = R.analyze(x, np.array([12]), RecoveryConfig(recovery_threshold=0.8))
    assert rec["recovery_status"][0] == R.STATUS_RECOVERED
    assert rec["recovery_slope"][0] > 0
    assert np.isfinite(rec["recovery_duration"][0])
    assert rec["recovery_fraction"][0] > 0.8


def test_slow_recovery_is_labelled_recovering():
    x = build_disturbance(break_at=28, pre=0.75, drop=0.45, rate=0.01)
    rec = R.analyze(x, np.array([28]), RecoveryConfig(recovery_threshold=0.8))
    assert rec["recovery_status"][0] == R.STATUS_RECOVERING
    assert rec["recovery_slope"][0] > 0
    assert np.isnan(rec["recovery_duration"][0]), \
        "threshold never reached within the record"


def test_no_recovery_is_labelled_not_recovering():
    x = np.full((T, 1), 0.75)
    x[15:, 0] = 0.30
    rec = R.analyze(x, np.array([15]), RecoveryConfig())
    assert rec["recovery_status"][0] == R.STATUS_NOT_RECOVERING
    assert rec["recovery_magnitude"][0] == pytest.approx(0.0, abs=1e-9)


def test_small_dip_below_threshold_is_not_a_disturbance():
    x = np.full((T, 1), 0.75)
    x[15, 0] = 0.73                    # 0.02 drop
    rec = R.analyze(x, np.array([15]),
                    RecoveryConfig(min_disturbance_magnitude=0.05))
    assert rec["recovery_status"][0] == R.STATUS_NO_DISTURBANCE


def test_no_break_index_yields_no_disturbance():
    x = np.random.default_rng(SEED).normal(0.6, 0.02, size=(T, 1))
    rec = R.analyze(x, np.array([-1]), RecoveryConfig())
    assert rec["recovery_status"][0] == R.STATUS_NO_DISTURBANCE
    assert np.isnan(rec["disturbance_index"][0])


def test_break_near_end_gives_insufficient_data():
    x = build_disturbance(break_at=T - 2, pre=0.75, drop=0.45)
    rec = R.analyze(x, np.array([T - 2]), RecoveryConfig(min_post_obs=3))
    assert rec["recovery_status"][0] == R.STATUS_INSUFFICIENT_DATA


def test_recovery_thresholds_are_configurable():
    x = build_disturbance(break_at=12, pre=0.75, drop=0.45, rate=0.02)
    strict = R.analyze(x, np.array([12]),
                       RecoveryConfig(recovery_threshold=0.99))
    lenient = R.analyze(x, np.array([12]),
                        RecoveryConfig(recovery_threshold=0.50))
    assert lenient["recovery_duration"][0] <= strict["recovery_duration"][0] \
        or np.isnan(strict["recovery_duration"][0])


def test_recovery_handles_missing_values():
    x = build_disturbance(break_at=12, pre=0.75, drop=0.45, rate=0.05)
    x[[20, 21], 0] = np.nan
    rec = R.analyze(x, np.array([12]), RecoveryConfig())
    assert np.isfinite(rec["recovery_slope"][0])
    assert rec["recovery_status"][0] in (R.STATUS_RECOVERED,
                                         R.STATUS_RECOVERING)


def test_recovery_summary_aggregates():
    xs = np.hstack([build_disturbance(12, rate=0.05),
                    build_disturbance(28, rate=0.01),
                    np.full((T, 1), 0.6)])
    rec = R.analyze(xs, np.array([12, 28, -1]), RecoveryConfig())
    s = R.summary(rec)
    assert s["n_pixels"] == 3
    assert s["status_counts"]["RECOVERED"] >= 1
    assert s["n_disturbed"] >= 2


def test_recovery_rejects_mismatched_break_index():
    with pytest.raises(ValueError):
        R.analyze(np.zeros((T, 3)), np.array([1, 2]), RecoveryConfig())
