"""Leakage-audit tests (M3 Parts 9 and 11).

Each test states a leakage scenario and requires the audit to catch it. A
check that cannot fail is worthless, so every check is exercised on both a
clean and a contaminated input.
"""
import json

import numpy as np
import pytest

from src.config import Config
from src.features import FEATURE_GROUPS, build_feature_table, feature_names
from src.leakage import (LeakageError, audit_report,
                         check_ablation_feature_groups,
                         check_ablation_isolation, check_block_purity,
                         check_buffer_separation, check_label_leakage,
                         check_no_lookahead, check_preprocessing_isolation,
                         check_spatial_separation, check_temporal_separation)
from src.validation import block_coordinates, spatial_block_folds
from src.config import SpatialCVConfig

H, W = 16, 16
T = 36


def series(n=32, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    ndvi = np.clip(0.5 + 0.15 * np.sin(2 * np.pi * t / 7)[:, None]
                   + rng.normal(0, 0.03, (T, n)), 0.05, 0.95)
    rain = rng.normal(1800, 200, (T, n))
    return ndvi, rain


# ------------------------------------------------------------------ spatial
def test_spatial_overlap_is_detected():
    train = np.array([True, True, False, False])
    clean = np.array([False, False, True, True])
    dirty = np.array([False, True, True, True])
    assert check_spatial_separation(train, clean)["passed"]
    result = check_spatial_separation(train, dirty)
    assert not result["passed"]
    assert result["evidence"]["overlap"] == 1


def test_block_purity_detects_a_split_block():
    blocks = np.array([[0, 0], [1, 1]])
    assert check_block_purity(blocks, np.array([[0, 0], [1, 1]]))["passed"]
    broken = check_block_purity(blocks, np.array([[0, 1], [1, 1]]))
    assert not broken["passed"]
    assert 0 in broken["evidence"]["split_blocks"]


def test_real_fold_assignment_keeps_blocks_pure():
    cfg = SpatialCVConfig(block_size=4, n_folds=4, seed=3)
    _, _, block_id = block_coordinates(H, W, cfg.block_size)
    _, folds = spatial_block_folds(H, W, cfg)
    assert check_block_purity(block_id, folds)["passed"]


def test_buffer_check_reports_touching_blocks_without_a_buffer():
    brow, bcol, _ = block_coordinates(H, W, 4)
    _, folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=4,
                                                         n_folds=4, seed=2))
    flat = folds.reshape(-1)
    test = flat == 0
    result = check_buffer_separation(~test, test, brow, bcol, 0)
    assert result["passed"], "sharing no block is the requirement at buffer 0"
    assert result["evidence"]["minimum_block_distance"] == 1
    assert "adjacent blocks still touch" in result["statement"]


def test_buffer_check_fails_when_the_buffer_is_not_applied():
    brow, bcol, _ = block_coordinates(H, W, 4)
    _, folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=4,
                                                         n_folds=4, seed=2))
    flat = folds.reshape(-1)
    test = flat == 0
    assert not check_buffer_separation(~test, test, brow, bcol, 1)["passed"]


def test_buffer_check_passes_once_the_buffer_is_applied():
    from src.validation import buffered_training_mask

    brow, bcol, _ = block_coordinates(H, W, 4)
    _, folds = spatial_block_folds(H, W, SpatialCVConfig(block_size=4,
                                                         n_folds=4, seed=2))
    flat = folds.reshape(-1)
    test = flat == 0
    train = buffered_training_mask(~test, test, brow, bcol, 1)
    assert check_buffer_separation(train, test, brow, bcol, 1)["passed"]


# ----------------------------------------------------------------- temporal
def test_temporal_separation_accepts_a_clean_split():
    assert check_temporal_separation(np.arange(20), np.arange(20, 36), 20,
                                     36)["passed"]


def test_temporal_separation_rejects_an_overlapping_split():
    assert not check_temporal_separation(np.arange(22), np.arange(20, 36), 20,
                                         36)["passed"]


def test_no_lookahead_passes_for_window_restricted_features():
    ndvi, rain = series()
    cfg = Config()

    def correct(full_ndvi, full_rain, cutoff):
        return build_feature_table(full_ndvi[:cutoff], full_rain[:cutoff], cfg)

    assert check_no_lookahead(correct, ndvi, rain, 22)["passed"]


def test_no_lookahead_catches_a_builder_that_peeks():
    """The realistic mistake: build over the whole record, subset later."""
    ndvi, rain = series()
    cfg = Config()

    def leaky(full_ndvi, full_rain, cutoff):
        # Ignores the cutoff: every statistic sees the future.
        return build_feature_table(full_ndvi, full_rain, cfg)

    result = check_no_lookahead(leaky, ndvi, rain, 22)
    assert not result["passed"]
    assert "reading past the cutoff" in result["statement"]


def test_no_lookahead_catches_a_single_leaked_statistic():
    """Even one future-derived column must fail the check."""
    ndvi, rain = series()
    cfg = Config()

    def leaky(full_ndvi, full_rain, cutoff):
        table, extras = build_feature_table(full_ndvi[:cutoff],
                                            full_rain[:cutoff], cfg)
        table = table.copy()
        table["mean"] = np.nanmean(full_ndvi, axis=0)   # whole-record mean
        return table, extras

    assert not check_no_lookahead(leaky, ndvi, rain, 22)["passed"]


# -------------------------------------------------------------------- label
def test_label_leakage_passes_for_the_real_feature_table():
    ndvi, rain = series(n=40)
    cfg = Config()
    table, _ = build_feature_table(ndvi, rain, cfg)
    labels = np.tile([1, 2, 3, 4], 10)
    result = check_label_leakage(table, labels, feature_names())
    assert result["passed"], result["statement"]


def test_label_leakage_detects_a_target_derived_column():
    ndvi, rain = series(n=40)
    cfg = Config()
    table, _ = build_feature_table(ndvi, rain, cfg)
    labels = np.tile([1, 2, 3, 4], 10)
    table = table.copy()
    table["sneaky"] = labels.astype(float)
    result = check_label_leakage(table, labels,
                                 list(feature_names()) + ["sneaky"])
    assert not result["passed"]
    assert result["evidence"]["suspicious_features"][0]["feature"] == "sneaky"


def test_label_leakage_detects_a_forbidden_name():
    ndvi, rain = series(n=20)
    table, _ = build_feature_table(ndvi, rain, Config())
    table = table.copy()
    table["trajectory_class"] = 1.0
    labels = np.tile([1, 2], 10)
    result = check_label_leakage(table, labels, ["mean", "trajectory_class"])
    assert not result["passed"]
    assert "trajectory_class" in result["evidence"]["forbidden_names"]


# ------------------------------------------------------------ preprocessing
def test_preprocessing_isolation_detects_a_contaminated_fit():
    train = np.array([True, True, False])
    test = np.array([False, False, True])
    assert check_preprocessing_isolation(train, test)["passed"]
    assert not check_preprocessing_isolation(np.ones(3, bool), test)["passed"]


# --------------------------------------------------------------- experiment
def test_ablation_cells_contain_only_documented_features():
    cfg = Config()
    sets = {e.name: feature_names(e.groups)
            for e in cfg.research.ablation.experiments}
    assert check_ablation_isolation(sets, FEATURE_GROUPS)["passed"]


def test_ablation_isolation_detects_a_stray_feature():
    sets = {"A": ["mean", "not_a_feature"]}
    result = check_ablation_isolation(sets, FEATURE_GROUPS)
    assert not result["passed"]
    assert "not_a_feature" in result["evidence"]["violations"]["A"]["unknown"]


def test_ablation_cells_match_their_declared_groups():
    cfg = Config()
    result = check_ablation_feature_groups(
        cfg.research.ablation.experiments,
        lambda e: feature_names(e.groups), FEATURE_GROUPS)
    assert result["passed"]


def test_ablation_group_check_detects_an_extra_feature():
    from src.config import AblationExperiment

    result = check_ablation_feature_groups(
        [AblationExperiment("A", "A", ["vegetation"])],
        lambda e: feature_names(e.groups) + ["mk_z"], FEATURE_GROUPS)
    assert not result["passed"]
    assert "mk_z" in result["evidence"]["violations"]["A"]["unexpected"]


# ------------------------------------------------------------------ report
def test_audit_report_saves_and_passes(tmp_path):
    checks = [check_spatial_separation(np.array([True, False]),
                                       np.array([False, True]))]
    report = audit_report(checks, tmp_path / "audit.json", strict=True)
    assert report["passed"] and report["n_failed"] == 0
    saved = json.loads((tmp_path / "audit.json").read_text())
    assert saved["n_checks"] == 1
    assert "checks" in saved


def test_audit_report_raises_on_failure(tmp_path):
    checks = [check_spatial_separation(np.ones(3, bool), np.ones(3, bool))]
    with pytest.raises(LeakageError, match="leakage audit failed"):
        audit_report(checks, tmp_path / "audit.json", strict=True)
    saved = json.loads((tmp_path / "audit.json").read_text())
    assert saved["passed"] is False, "a failed audit must still be recorded"


def test_audit_report_can_collect_without_raising(tmp_path):
    checks = [check_spatial_separation(np.ones(2, bool), np.ones(2, bool))]
    report = audit_report(checks, tmp_path / "audit.json", strict=False)
    assert report["passed"] is False
