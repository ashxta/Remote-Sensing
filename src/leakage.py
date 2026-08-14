"""Programmatic leakage audit (M3 Part 9).

A leakage claim in a report is worth nothing unless something checks it on
every run. This module turns each claim into an executable assertion whose
result is saved next to the metrics it qualifies.

Five kinds of leakage are audited:

spatial        do train and test share pixels, or blocks, or (when a buffer
               is configured) neighbouring blocks?
temporal       were historical features built from observations at or after
               the cutoff?
label          is any model input derived from the target - the reference
               labels, or a quantity computed from them?
preprocessing  was anything that learns parameters fitted on data that
               includes the test fold?
experiment     did an ablation cell receive a feature that its declared
               feature groups exclude?

Each check returns `passed`, a human-readable statement, and the evidence it
used. `audit_report` raises on failure when `strict=True`, so a leaking
experiment stops instead of publishing a number.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

__all__ = ["LeakageError", "check_spatial_separation", "check_block_purity",
           "check_buffer_separation", "check_temporal_separation",
           "check_label_leakage", "check_preprocessing_isolation",
           "check_ablation_isolation", "audit_report"]


class LeakageError(AssertionError):
    """Raised when an experiment fails a leakage check."""


def _result(name: str, passed: bool, statement: str, **evidence) -> dict:
    return {"check": name, "passed": bool(passed), "statement": statement,
            "evidence": {k: v for k, v in evidence.items()}}


# ------------------------------------------------------------------ spatial
def check_spatial_separation(train_mask, test_mask) -> dict:
    """Train and test must not share a single sample."""
    train = np.asarray(train_mask, bool)
    test = np.asarray(test_mask, bool)
    overlap = int((train & test).sum())
    return _result(
        "spatial_sample_overlap", overlap == 0,
        f"{overlap} samples appear in both the training and the test set",
        n_train=int(train.sum()), n_test=int(test.sum()), overlap=overlap)


def check_block_purity(block_ids, fold_grid) -> dict:
    """A spatial block must belong to exactly one fold."""
    blocks = np.asarray(block_ids).reshape(-1)
    folds = np.asarray(fold_grid).reshape(-1)
    split = [int(b) for b in np.unique(blocks)
             if np.unique(folds[blocks == b]).size > 1]
    return _result(
        "block_purity", not split,
        f"{len(split)} spatial blocks are split across more than one fold",
        n_blocks=int(np.unique(blocks).size), split_blocks=split[:20])


def check_buffer_separation(train_mask, test_mask, block_row, block_col,
                            buffer_blocks: int) -> dict:
    """With a buffer configured, no training block may sit inside it.

    With `buffer_blocks == 0` the check states the honest position: train and
    test share no block, but blocks that touch across a boundary are still
    spatially autocorrelated. That is a documented property of plain block
    CV, not a silent failure.
    """
    train = np.asarray(train_mask, bool)
    test = np.asarray(test_mask, bool)
    rows = np.asarray(block_row).reshape(-1)
    cols = np.asarray(block_col).reshape(-1)
    if not train.any() or not test.any():
        return _result("buffer_separation", True,
                       "no train/test pair to check", buffer_blocks=0)
    test_blocks = np.unique(np.c_[rows[test], cols[test]], axis=0)
    train_blocks = np.unique(np.c_[rows[train], cols[train]], axis=0)
    distances = np.abs(train_blocks[:, None, :] - test_blocks[None, :, :]
                       ).max(axis=2)
    minimum = int(distances.min()) if distances.size else 0
    if buffer_blocks <= 0:
        return _result(
            "buffer_separation", minimum >= 1,
            "plain block CV: train and test share no block "
            f"(minimum block distance {minimum}); adjacent blocks still "
            "touch, which is a known limitation of block CV without a buffer",
            buffer_blocks=0, minimum_block_distance=minimum)
    return _result(
        "buffer_separation", minimum > buffer_blocks,
        f"minimum train-to-test block distance is {minimum}, required "
        f"greater than the configured buffer of {buffer_blocks}",
        buffer_blocks=int(buffer_blocks), minimum_block_distance=minimum)


# ----------------------------------------------------------------- temporal
def check_temporal_separation(historical_window, later_window, cutoff: int,
                              n_time: int) -> dict:
    """Historical indices must all precede the cutoff, later ones follow it."""
    historical = np.asarray(historical_window).reshape(-1)
    later = np.asarray(later_window).reshape(-1)
    ok = (historical.size and later.size
          and int(historical.max()) < int(cutoff) <= int(later.min())
          and historical.size + later.size == n_time)
    return _result(
        "temporal_separation", bool(ok),
        f"historical indices end at {int(historical.max()) if historical.size else None} "
        f"and later indices start at {int(later.min()) if later.size else None} "
        f"around cutoff {cutoff}",
        cutoff=int(cutoff), n_historical=int(historical.size),
        n_later=int(later.size), n_time=int(n_time))


def check_no_lookahead(build_historical, ndvi, rain, cutoff: int,
                       corrupt_value: float = -0.5) -> dict:
    """Empirical no-lookahead test: corrupt the future, expect no change.

    `build_historical(ndvi, rain, cutoff)` is handed the COMPLETE record and
    the cutoff, and is responsible for restricting itself to observations
    before the cutoff. That is deliberate: handing it a pre-sliced window
    would make the check unfalsifiable, because a builder that never sees the
    future cannot possibly use it. Giving it the whole array reproduces the
    realistic mistake - computing features over the full record and
    subsetting rows afterwards - and catches it.

    The check runs the builder twice, once normally and once with every
    observation from the cutoff onwards replaced by a constant, and requires
    the historical features to be bit-identical.
    """
    ndvi = np.asarray(ndvi, dtype="float64")
    rain = np.asarray(rain, dtype="float64")

    def features_of(array):
        built = build_historical(array, rain, cutoff)
        table = built[0] if isinstance(built, tuple) else built
        return np.asarray(table)

    clean = features_of(ndvi)
    corrupted_ndvi = ndvi.copy()
    corrupted_ndvi[cutoff:] = corrupt_value
    corrupted = features_of(corrupted_ndvi)
    identical = (clean.shape == corrupted.shape
                 and np.array_equal(clean, corrupted, equal_nan=True))
    return _result(
        "no_lookahead", identical,
        "historical features are unchanged when every observation from the "
        "cutoff onwards is corrupted" if identical else
        "historical features changed when future observations were corrupted, "
        "so the builder is reading past the cutoff",
        cutoff=int(cutoff), n_features=int(clean.shape[1]))


# -------------------------------------------------------------------- label
def check_label_leakage(features: pd.DataFrame, labels,
                        feature_names: Sequence[str], *,
                        forbidden: Sequence[str] = (),
                        perfect_threshold: float = 0.999) -> dict:
    """No model input may be derived from the target.

    Two tests. First, a name check against columns known to be derived from
    labels or from the model's own outputs (trajectory categories, reference
    class, predictions). Second, an empirical check: a feature that
    reproduces the label almost exactly is treated as suspicious and
    reported, because that is what a leaked target looks like.
    """
    names = list(feature_names)
    forbidden = {f.lower() for f in (
        tuple(forbidden) + ("truth", "label", "class", "target",
                            "trajectory_category", "trajectory_class",
                            "prediction", "predicted", "reference_class"))}
    by_name = [n for n in names
               if any(token == n.lower() or token in n.lower().split("_")
                      for token in forbidden)]

    y = pd.Series(np.asarray(labels))
    codes = y.astype("category").cat.codes.to_numpy(dtype="float64")
    suspicious = []
    for name in names:
        column = features[name].to_numpy(dtype="float64")
        good = np.isfinite(column)
        if good.sum() < 3 or np.unique(column[good]).size < 2:
            continue
        with np.errstate(invalid="ignore"):
            correlation = np.corrcoef(column[good], codes[good])[0, 1]
        if np.isfinite(correlation) and abs(correlation) >= perfect_threshold:
            suspicious.append({"feature": name,
                               "abs_correlation_with_label":
                                   float(abs(correlation))})
    passed = not by_name and not suspicious
    return _result(
        "label_leakage", passed,
        "no model input is named after or numerically identical to the "
        "target" if passed else
        f"{len(by_name)} feature name(s) and {len(suspicious)} feature "
        "value(s) look target-derived",
        forbidden_names=by_name, suspicious_features=suspicious,
        n_features=len(names))


# ------------------------------------------------------------ preprocessing
def check_preprocessing_isolation(fitted_on_mask, test_mask,
                                  component: str = "imputer") -> dict:
    """Anything that learns parameters must not have seen the test fold."""
    fitted = np.asarray(fitted_on_mask, bool)
    test = np.asarray(test_mask, bool)
    contaminated = int((fitted & test).sum())
    return _result(
        "preprocessing_isolation", contaminated == 0,
        f"the {component} was fitted on {contaminated} test samples",
        component=component, n_fitted_on=int(fitted.sum()),
        contaminated=contaminated)


# --------------------------------------------------------------- experiment
def check_ablation_isolation(feature_sets: dict,
                             group_members: dict) -> dict:
    """No ablation cell may contain an undocumented or duplicated feature.

    Per-cell group membership is checked separately by
    `check_ablation_feature_groups`; this check catches a feature that is not
    part of any declared group at all, which is how a stray column would
    reach one experiment but not the others.
    """
    allowed = {name for members in group_members.values() for name in members}
    violations = {}
    for name, features in feature_sets.items():
        duplicates = [f for f in features if features.count(f) > 1]
        unknown = [f for f in features if f not in allowed]
        if duplicates or unknown:
            violations[name] = {"duplicates": sorted(set(duplicates)),
                                "unknown": sorted(set(unknown))}
    return _result(
        "ablation_isolation", not violations,
        "every ablation cell contains only documented, unique features"
        if not violations else
        f"{len(violations)} ablation cell(s) contain undocumented or "
        "duplicated features",
        violations=violations, n_cells=len(feature_sets))


def check_ablation_feature_groups(experiments, resolver, group_members) -> dict:
    """A cell must contain exactly the union of its declared groups."""
    violations = {}
    for experiment in experiments:
        declared = set()
        for group in experiment.groups:
            declared |= set(group_members[group])
        resolved = set(resolver(experiment))
        if resolved != declared:
            violations[experiment.name] = {
                "unexpected": sorted(resolved - declared),
                "missing": sorted(declared - resolved)}
    return _result(
        "ablation_feature_groups", not violations,
        "each ablation cell resolves to exactly its declared feature groups"
        if not violations else
        f"{len(violations)} cell(s) do not match their declared groups",
        violations=violations)


# ------------------------------------------------------------------- report
def audit_report(checks: Sequence[dict], output_path=None, *,
                 strict: bool = True, context: dict | None = None) -> dict:
    """Collect checks, save them, and refuse to continue on a failure."""
    checks = list(checks)
    failed = [c for c in checks if not c["passed"]]
    report = {
        "n_checks": len(checks),
        "n_failed": len(failed),
        "passed": not failed,
        "context": context or {},
        "checks": checks,
        "note": ("Executable leakage assertions, re-evaluated on every run. "
                 "A passing audit constrains the experiment design; it does "
                 "not by itself make a result generalise."),
    }
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, default=str))
    if failed and strict:
        raise LeakageError("leakage audit failed: " + "; ".join(
            f"{c['check']}: {c['statement']}" for c in failed))
    return report
