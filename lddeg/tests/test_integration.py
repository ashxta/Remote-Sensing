"""Integration tests: feature contracts and end-to-end pipeline behaviour."""
import numpy as np
import pytest

from src.classify import build_features, FEATURES
from src.config import Config

T = 36
SEED = 42


def synth(n=60, seed=SEED):
    """A small stack containing all trajectory archetypes."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    cols, labels = [], []
    for i in range(n):
        kind = i % 5
        if kind == 0:                                   # stable high
            s = 0.78 + rng.normal(0, 0.02, T)
        elif kind == 1:                                 # cropland
            s = 0.45 + 0.0008 * t + rng.normal(0, 0.03, T)
        elif kind == 2:                                 # periodic
            s = 0.5 + 0.18 * np.sin(2 * np.pi * t / 7.0) \
                + rng.normal(0, 0.02, T)
        elif kind == 3:                                 # declining
            k = 14
            s = np.concatenate([np.full(k, 0.7),
                                0.7 - 0.02 * np.arange(T - k)])
            s = s + rng.normal(0, 0.02, T)
        else:                                           # recovering
            k = 12
            s = np.concatenate([np.full(k, 0.3),
                                0.3 + 0.02 * np.arange(T - k)])
            s = s + rng.normal(0, 0.02, T)
        cols.append(s)
        labels.append(kind + 1)
    ndvi = np.clip(np.array(cols).T, 0.02, 0.95)
    rain = rng.normal(1800, 220, size=(T, n))
    return ndvi, rain, np.array(labels)


# ---------------------------------------------------------------- contracts
def test_features_have_expected_columns_and_order():
    ndvi, rain, _ = synth()
    F, _ = build_features(ndvi, rain, Config())
    assert list(F.columns) == FEATURES
    assert F.shape == (ndvi.shape[1], len(FEATURES))


def test_features_are_finite_for_clean_input():
    """The whole map must be predictable: no NaN features on good pixels.

    Regression guard. Recovery and break descriptors are undefined for
    undisturbed pixels; if they leak through as NaN, roughly half the map
    becomes unclassifiable and map-wide accuracy silently collapses even
    though held-out test accuracy stays high.
    """
    ndvi, rain, _ = synth(n=120)
    F, _ = build_features(ndvi, rain, Config())
    bad = ~np.isfinite(F[FEATURES].values).all(axis=1)
    assert bad.sum() == 0, (
        f"{bad.sum()} pixels have non-finite features: "
        f"{F.columns[~np.isfinite(F.values).all(axis=0)].tolist()}")


def test_structural_indicators_are_consistent():
    """Where has_disturbance is 0, the dependent descriptors must be 0."""
    ndvi, rain, _ = synth(n=100)
    F, _ = build_features(ndvi, rain, Config())
    none = F.has_disturbance.values == 0
    if none.any():
        assert np.all(F.recovery_fraction.values[none] == 0)
        assert np.all(F.recovery_slope.values[none] == 0)
    nob = F.has_break.values == 0
    if nob.any():
        assert np.all(F.break_t.values[nob] == 0)


def test_features_are_deterministic():
    ndvi, rain, _ = synth()
    a, _ = build_features(ndvi, rain, Config())
    b, _ = build_features(ndvi, rain, Config())
    assert np.allclose(a.values, b.values, equal_nan=True)


def test_declining_pixels_have_negative_sen_slope():
    ndvi, rain, labels = synth(n=100)
    F, _ = build_features(ndvi, rain, Config())
    assert np.nanmedian(F.sen.values[labels == 4]) < 0
    assert np.nanmedian(F.sen.values[labels == 5]) > 0


def test_periodic_pixels_score_higher_enrichment():
    ndvi, rain, labels = synth(n=150)
    F, _ = build_features(ndvi, rain, Config())
    periodic = np.nanmedian(F.cyc_enrichment.values[labels == 3])
    stable = np.nanmedian(F.cyc_enrichment.values[labels == 1])
    assert periodic > stable


def test_config_thresholds_change_feature_output():
    """Thresholds must actually be wired through, not hard-coded."""
    ndvi, rain, _ = synth()
    narrow = Config()
    narrow.cyclicity.min_period, narrow.cyclicity.max_period = 6.0, 8.0
    wide = Config()
    wide.cyclicity.min_period, wide.cyclicity.max_period = 3.0, 15.0
    a, _ = build_features(ndvi, rain, narrow)
    b, _ = build_features(ndvi, rain, wide)
    assert not np.allclose(a.cyc_score.values, b.cyc_score.values,
                           equal_nan=True)


def test_missing_observations_do_not_crash_feature_build():
    ndvi, rain, _ = synth()
    rng = np.random.default_rng(1)
    holes = rng.random(ndvi.shape) < 0.10
    ndvi = ndvi.copy()
    ndvi[holes] = np.nan
    F, extras = build_features(ndvi, rain, Config())
    assert F.shape[0] == ndvi.shape[1]
    assert "restrend_valid" in extras


def test_extras_expose_validity_flags():
    ndvi, rain, _ = synth()
    _, ex = build_features(ndvi, rain, Config())
    for key in ("restrend_valid", "cyc_periodic", "break_significant",
                "break_p_adjusted", "recovery"):
        assert key in ex, f"missing diagnostic {key}"
    assert ex["restrend_valid"].dtype == bool


# ------------------------------------------------------------- pipeline e2e
@pytest.mark.slow
def test_pipeline_produces_georeferenced_outputs(tmp_path):
    """End-to-end smoke test on the synthetic stacks, if present."""
    import rasterio
    from pathlib import Path
    from src import geo

    cfg = Config(experiment_name="pytest_e2e", seed=1)
    if not Path(cfg.paths.ndvi_stack).exists():
        pytest.skip("synthetic stacks not generated")
    cfg.model.use_cnn = False
    cfg.model.samples_per_class = 60
    cfg.model.rf_trees = 30
    cfg.paths.results = str(tmp_path)

    import run_pipeline
    exp, results = run_pipeline.main(cfg)

    assert (exp.root / "config" / "config.json").exists()
    assert (exp.root / "logs" / "run.log").exists()
    assert "data_quality" in results and "recovery" in results

    src_ref = geo.GeoRef.from_raster(cfg.paths.ndvi_stack)
    tifs = list((exp.root / "predictions").glob("*.tif"))
    assert len(tifs) >= 10, "expected georeferenced prediction layers"
    for t in tifs:
        with rasterio.open(t) as s:
            assert s.crs == src_ref.crs, f"{t.name} lost its CRS"
            assert (s.height, s.width) == src_ref.shape
            assert s.nodata is not None, f"{t.name} has no NoData declared"
