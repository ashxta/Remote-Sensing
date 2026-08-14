"""The research experiment matrix (M3 Part 8).

QUESTION
--------
    Does multi-temporal trajectory analysis improve detection and
    interpretation of land degradation compared with conventional trend
    analysis?

To answer that, four methods are run under one validation protocol:

    baseline_trend   Mann-Kendall + Sen's slope decision rule (no learning)
    rf_basic         Random Forest on basic vegetation features only
    cnn_1d           1D CNN on the raw NDVI sequence (learned representation)
    rf_proposed      Random Forest on the full proposed framework: trend,
                     rainfall-adjusted (RESTREND), cyclicity, breakpoint and
                     recovery features

FAIRNESS OF THE COMPARISON
--------------------------
The conventional baseline can only answer one question - "is there a
significant negative monotonic trend?" - so it cannot be scored on a
five-class problem. Comparing its multiclass accuracy against a classifier's
would be rigged. The matrix therefore evaluates every method on a BINARY
degradation-detection task (reference degradation class vs everything else),
which the baseline can genuinely attempt, and reports the multiclass result
separately for the methods that support it.

All methods share the same spatial block folds, the same samples and the
same metric code, so the comparison isolates the method rather than the
protocol.

The comparison answers whether the extra features and the learned
representation help on THIS dataset under THIS protocol. On synthetic data
that is a statement about the generator, not about any landscape.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .discrimination import run_discrimination_analysis
from .features import feature_names
from .uncertainty import uncertainty_summary
from .validation import (aggregate_fold_metrics, classification_metrics,
                         spatial_cv_rf)

__all__ = ["baseline_trend_prediction", "baseline_restrend_prediction",
           "baseline_integrated_prediction", "binary_degradation_labels",
           "run_experiment_matrix", "MATRIX_METHODS"]

MATRIX_METHODS = ("baseline_trend", "baseline_restrend",
                  "baseline_integrated", "rf_basic", "cnn_1d", "rf_proposed")


def binary_degradation_labels(labels, degradation_classes) -> np.ndarray:
    """1 where the reference class is counted as degradation, else 0."""
    labels = np.asarray(labels)
    return np.isin(labels, list(degradation_classes)).astype(int)


def baseline_trend_prediction(features: pd.DataFrame, cfg: Config) -> dict:
    """Conventional practice: significant negative Mann-Kendall + Sen slope.

    This is a fixed decision rule, not a fitted model: it has no parameters
    learned from labels, so it needs no training fold and cannot leak. It is
    evaluated on exactly the same samples as the learned methods.

    A pseudo-probability is derived from the p-value as 1 - p for flagged
    pixels and p/2 otherwise, purely so the method has a comparable
    confidence column. It is a monotone transform of the test statistic, NOT
    a calibrated probability, and is labelled as such in the outputs.
    """
    alpha = cfg.trend.alpha
    p = features["mk_p_value"].to_numpy(dtype="float64")
    slope = features["sen"].to_numpy(dtype="float64")
    declining = np.isfinite(p) & (p < alpha) & np.isfinite(slope) & (slope < 0)
    prediction = declining.astype(int)
    with np.errstate(invalid="ignore"):
        score = np.where(declining, 1.0 - p, np.clip(p / 2.0, 0.0, 0.5))
    score = np.where(np.isfinite(score), score, 0.5)
    probabilities = np.c_[1.0 - score, score]
    return {"predictions": prediction, "probabilities": probabilities,
            "classes": np.array([0, 1]),
            "note": ("Fixed decision rule (Mann-Kendall p < alpha and "
                     "negative Sen slope). The probability column is a "
                     "monotone transform of the p-value, not a calibrated "
                     "probability.")}


def baseline_restrend_prediction(features: pd.DataFrame, cfg: Config) -> dict:
    """Experiment 2: conventional trend detection PLUS rainfall correction.

    A pixel is flagged only if the decline survives climate adjustment, or
    if the NDVI~rainfall relationship is too weak for the adjustment to
    mean anything (M1's `restrend_valid` gate), in which case the raw
    decline stands uncorrected rather than being silently dropped.

    This is the rule-based ancestor of the proposed framework: it isolates
    what the rainfall correction alone contributes over Experiment 1.
    """
    alpha = cfg.trend.alpha
    mk_p = features["mk_p_value"].to_numpy(dtype="float64")
    sen = features["sen"].to_numpy(dtype="float64")
    restrend_p = features["restrend_p_value"].to_numpy(dtype="float64")
    restrend = features["restrend"].to_numpy(dtype="float64")
    # `restrend_valid` marks where the NDVI~rainfall relation is strong
    # enough for the adjustment to mean anything. It is NOT the same as
    # `restrend_significant`, which additionally requires the ADJUSTED TREND
    # to be significant - using that here would keep every climate-explained
    # decline flagged and make this rule identical to Experiment 1.
    interpretable = features["restrend_valid"].to_numpy(dtype="float64") > 0
    raw_decline = (np.isfinite(mk_p) & (mk_p < alpha) & np.isfinite(sen)
                   & (sen < 0))
    adjusted_decline = (np.isfinite(restrend_p) & (restrend_p < alpha)
                        & np.isfinite(restrend) & (restrend < 0))
    # Uninterpretable adjustment -> keep the uncorrected decline.
    flagged = raw_decline & (adjusted_decline | ~interpretable)
    with np.errstate(invalid="ignore"):
        score = np.where(flagged, 1.0 - np.nan_to_num(restrend_p, nan=0.5),
                         np.clip(np.nan_to_num(mk_p, nan=1.0) / 2.0, 0.0, 0.5))
    return {"predictions": flagged.astype(int),
            "probabilities": np.c_[1.0 - score, score],
            "classes": np.array([0, 1]),
            "note": ("Mann-Kendall + Sen decline that survives RESTREND "
                     "climate adjustment, or whose adjustment is not "
                     "interpretable. The score is a monotone transform of "
                     "the p-value, not a calibrated probability.")}


def baseline_integrated_prediction(features: pd.DataFrame, cfg: Config) -> dict:
    """Experiments 3-4 as a RULE: trend + rainfall correction + cyclicity
    + disturbance evidence, with no learning involved.

    This is the honest statistical counterpart of the proposed framework.
    It matters because a Random Forest improvement over Experiment 1 could
    come either from the extra temporal evidence or merely from having a
    flexible learner; comparing against this rule separates the two.

    A pixel is flagged as persistent degradation when the decline
      * is significant,
      * survives climate adjustment (or the adjustment is uninterpretable),
      * is NOT explained by a detected cycle, and
      * is NOT a single disturbance that has already largely recovered.
    """
    restrend_rule = baseline_restrend_prediction(features, cfg)
    flagged = restrend_rule["predictions"].astype(bool)
    trajectory = cfg.research.trajectory
    periodic = features["cyclicity_periodic"].to_numpy(dtype="float64") > 0
    enrichment = features["cyc_enrichment"].to_numpy(dtype="float64")
    cyclic = periodic & np.isfinite(enrichment) & (
        enrichment >= cfg.cyclicity.periodicity_threshold)
    recovered = (features["has_disturbance"].to_numpy(dtype="float64") > 0) \
        & (features["recovery_fraction"].to_numpy(dtype="float64")
           >= trajectory.recovery_fraction_threshold)
    flagged = flagged & ~cyclic & ~recovered
    score = restrend_rule["probabilities"][:, 1]
    score = np.where(flagged, score, np.minimum(score, 0.5))
    return {"predictions": flagged.astype(int),
            "probabilities": np.c_[1.0 - score, score],
            "classes": np.array([0, 1]),
            "note": ("Rule-based integrated framework: significant decline, "
                     "surviving climate adjustment, not attributable to a "
                     "detected cycle, and not an already-recovered "
                     "disturbance. No learning; no calibrated probability.")}


def _fold_metrics_for_rule(y_true, y_pred, folds, mask, labels) -> list:
    """Per-fold metrics for a rule that has no training step."""
    out = []
    for fold in sorted(np.unique(folds[mask])):
        test = mask & (folds == fold)
        if not test.any():
            continue
        metrics = classification_metrics(y_true[test], y_pred[test],
                                         labels=labels)
        metrics["fold"] = int(fold)
        metrics["n_test"] = int(test.sum())
        out.append(metrics)
    return out


def _record(name: str, task: str, metrics: dict, *, n_features=None,
            representation="", validation="spatial_block_cv") -> dict:
    summary = metrics.get("fold_summary", {})
    return {
        "method": name, "task": task, "representation": representation,
        "validation": validation, "n_features": n_features,
        "accuracy": metrics["accuracy"], "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "cohen_kappa": metrics["cohen_kappa"],
        "fold_f1_macro_mean": summary.get("f1_macro_mean", float("nan")),
        "fold_f1_macro_std": summary.get("f1_macro_std", float("nan")),
        "n_evaluated": metrics.get("n_evaluated", metrics["n_samples"]),
    }


def _common_evaluation(runs: dict, target: np.ndarray, labels_present,
                       folds: np.ndarray) -> tuple:
    """Re-score every method on the samples ALL of them evaluated.

    Methods can legitimately cover different samples - a CNN run with
    `max_folds` set evaluates fewer folds than the Random Forest - and
    comparing their headline numbers directly would compare different test
    sets. The comparison is therefore computed on the intersection of the
    evaluated masks, so every reported difference is measured on identical
    samples. Each method's own coverage is preserved alongside it.
    """
    if not runs:
        return {}, np.zeros(len(target), bool)
    common = np.ones(len(target), bool)
    for run in runs.values():
        common &= np.asarray(run["evaluated"], bool)
    rescored = {}
    for name, run in runs.items():
        if not common.any():
            continue
        metrics = classification_metrics(target[common],
                                         np.asarray(run["predictions"])[common],
                                         labels=labels_present)
        metrics["fold_metrics"] = _fold_metrics_for_rule(
            target, np.asarray(run["predictions"]), folds, common,
            labels_present)
        metrics["fold_summary"] = aggregate_fold_metrics(
            metrics["fold_metrics"])
        metrics["n_evaluated"] = int(common.sum())
        metrics["n_evaluated_by_this_method"] = int(
            np.asarray(run["evaluated"], bool).sum())
        metrics["scored_on"] = "samples evaluated by every compared method"
        rescored[name] = metrics
    return rescored, common


def run_experiment_matrix(features: pd.DataFrame, labels, fold_grid,
                          output_dir, cfg: Config | None = None, *,
                          series=None, sample_mask=None, block_row=None,
                          block_col=None, channel_names=None,
                          logger=None) -> pd.DataFrame:
    """Run the method comparison and save machine-readable results.

    `series` is the raw sequence stack for the CNN, either (time, samples)
    or (channels, time, samples). Pass BOTH NDVI and rainfall: the Random
    Forest sees rainfall through the engineered features, so an NDVI-only
    CNN would be judged on strictly less information than its competitor.
    The channels actually used are recorded in the outputs.

    When `series` is absent, or torch is not installed, the CNN row is
    recorded as skipped with the reason - never silently dropped.
    """
    cfg = cfg or Config()
    matrix_cfg = cfg.research.matrix
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    y = np.asarray(labels)
    folds = np.asarray(fold_grid).reshape(-1)
    mask = np.ones(len(y), bool) if sample_mask is None \
        else np.asarray(sample_mask, bool)
    binary = binary_degradation_labels(y, matrix_cfg.degradation_classes)
    binary_labels = np.array([0, 1])

    details, skipped = {}, []
    # Per-task record of every method's predictions and coverage, so the
    # comparison can be re-scored on one common test set.
    runs = {"binary_degradation": {}, "multiclass_trajectory": {}}
    meta = {}

    # ---------------------------------------------- statistical baselines
    # Experiment 1: trend only. Experiment 2: + rainfall correction.
    # Experiments 3-4 as a rule: + cyclicity and disturbance evidence.
    # Running all three isolates what each added component contributes
    # BEFORE any machine learning is involved.
    statistical_rules = []
    if matrix_cfg.run_baseline:
        statistical_rules = [
            ("baseline_trend", baseline_trend_prediction, 2,
             "Mann-Kendall p + Sen slope sign"),
            ("baseline_restrend", baseline_restrend_prediction, 4,
             "trend + RESTREND climate adjustment"),
            ("baseline_integrated", baseline_integrated_prediction, 7,
             "trend + climate adjustment + cyclicity + recovery (rule)"),
        ]
    for name, rule_fn, n_inputs, representation in statistical_rules:
        rule = rule_fn(features, cfg)
        predictions = rule["predictions"]
        metrics = classification_metrics(binary[mask], predictions[mask],
                                         labels=binary_labels)
        metrics["fold_metrics"] = _fold_metrics_for_rule(
            binary, predictions, folds, mask, binary_labels)
        metrics["fold_summary"] = aggregate_fold_metrics(
            metrics["fold_metrics"])
        metrics["validation"] = "decision rule; evaluated on all samples"
        metrics["note"] = rule["note"]
        metrics["uncertainty"] = uncertainty_summary(
            rule["probabilities"][mask], truth=binary[mask],
            predictions=predictions[mask], cfg=cfg.research.uncertainty)
        details[name] = metrics
        runs["binary_degradation"][name] = {
            "predictions": predictions, "evaluated": mask.copy()}
        meta[name] = {"n_features": n_inputs,
                      "representation": representation,
                      "validation": "fixed rule (no training)"}
        if logger is not None:
            logger.info("  matrix %-20s macro F1 %.4f | recall on "
                        "degradation %.4f | precision %.4f", name,
                        metrics["f1_macro"], metrics["per_class"]["1"]["recall"],
                        metrics["per_class"]["1"]["precision"])

    # ----------------------------------------------------- Random Forest(s)
    forest_variants = []
    if matrix_cfg.run_random_forest:
        forest_variants.append(("rf_basic", feature_names(["vegetation"]),
                                "basic vegetation features"))
    if matrix_cfg.run_proposed:
        forest_variants.append(("rf_proposed",
                                feature_names(cfg.research.features.groups),
                                "proposed framework: trend + RESTREND + "
                                "cyclicity + breakpoint/recovery"))
    for name, columns, representation in forest_variants:
        meta[name] = {"n_features": len(columns),
                      "representation": representation,
                      "validation": "spatial_block_cv"}
        for task, target in (("binary_degradation", binary),
                             ("multiclass_trajectory", y)):
            result = spatial_cv_rf(features, target, folds,
                                   sample_mask=mask, feature_names=columns,
                                   cfg=cfg.research.model,
                                   block_row=block_row, block_col=block_col)
            metrics = result["metrics"]
            metrics["uncertainty"] = uncertainty_summary(
                result["probabilities"][result["evaluated"]],
                truth=target[result["evaluated"]],
                predictions=result["predictions"][result["evaluated"]],
                cfg=cfg.research.uncertainty)
            details[f"{name}__{task}"] = metrics
            runs[task][name] = {"predictions": result["predictions"],
                                "evaluated": result["evaluated"]}
            pd.DataFrame({
                "truth": target, "prediction": result["predictions"],
                "fold": folds, "evaluated": result["evaluated"],
            }).to_csv(root / f"{name}_{task}_predictions.csv", index=False)
        if logger is not None:
            logger.info("  matrix %s: binary macro F1 %.4f | multiclass "
                        "macro F1 %.4f", name,
                        details[f"{name}__binary_degradation"]["f1_macro"],
                        details[f"{name}__multiclass_trajectory"]["f1_macro"])

    # -------------------------------------------------------------- the CNN
    if matrix_cfg.run_cnn:
        from .cnn_experiment import run_spatial_cnn, torch_available
        if series is None:
            skipped.append({"method": "cnn_1d",
                            "reason": "raw NDVI series not supplied"})
        elif not torch_available():
            skipped.append({"method": "cnn_1d",
                            "reason": "optional dependency 'torch' is not "
                                      "installed"})
        else:
            stack = np.asarray(series, dtype="float64")
            n_channels = 1 if stack.ndim == 2 else int(stack.shape[0])
            n_steps = stack.shape[0] if stack.ndim == 2 else stack.shape[1]
            names = list(channel_names) if channel_names is not None else (
                ["ndvi"] if n_channels == 1 else
                [f"channel_{i}" for i in range(n_channels)])
            meta["cnn_1d"] = {
                "n_features": int(n_steps * n_channels),
                "representation": "learned from the raw "
                                  + " + ".join(names) + " sequence",
                "validation": "spatial_block_cv"}
            for task, target in (("binary_degradation", binary),
                                 ("multiclass_trajectory", y)):
                result = run_spatial_cnn(
                    series, target, folds, root / f"cnn_{task}", cfg.research.cnn,
                    sample_mask=mask, channel_names=names,
                    uncertainty_cfg=cfg.research.uncertainty, logger=logger)
                metrics = result["metrics"]
                details[f"cnn_1d__{task}"] = metrics
                runs[task]["cnn_1d"] = {"predictions": result["predictions"],
                                        "evaluated": result["evaluated"]}
                meta["cnn_1d"]["validation"] = metrics["validation"]
            if logger is not None:
                logger.info("  matrix cnn_1d: binary macro F1 %.4f | "
                            "multiclass macro F1 %.4f",
                            details["cnn_1d__binary_degradation"]["f1_macro"],
                            details["cnn_1d__multiclass_trajectory"]["f1_macro"])

    # ------------------------------- re-score everything on one common set
    rows, coverage = [], {}
    for task, target, task_labels in (
            ("binary_degradation", binary, binary_labels),
            ("multiclass_trajectory", y, np.unique(y[mask]))):
        rescored, common = _common_evaluation(runs[task], target, task_labels,
                                              folds)
        coverage[task] = {
            "n_common_evaluated": int(common.sum()),
            "n_in_sample_set": int(mask.sum()),
            "per_method_evaluated": {
                name: int(np.asarray(run["evaluated"], bool).sum())
                for name, run in runs[task].items()},
        }
        coverage[task]["identical_coverage"] = bool(
            set(coverage[task]["per_method_evaluated"].values()) == {
                coverage[task]["n_common_evaluated"]}) if runs[task] else True
        for name, metrics in rescored.items():
            details[f"{name}__{task}__common"] = metrics
            rows.append(_record(name, task, metrics, **meta[name]))
        if logger is not None and runs[task] and \
                not coverage[task]["identical_coverage"]:
            logger.warning("  matrix %s: methods covered different samples; "
                           "comparison re-scored on the %d samples every "
                           "method evaluated", task,
                           coverage[task]["n_common_evaluated"])

    table = pd.DataFrame(rows)
    table.to_csv(root / "experiment_matrix.csv", index=False)

    # ---------------------------------- the research question, measured
    # Every method that produced a degradation call is scored on how well
    # it separates degradation from EACH confounder, not just on average.
    discrimination = None
    binary_runs = runs["binary_degradation"]
    if binary_runs:
        discrimination = run_discrimination_analysis(
            y, {name: np.asarray(run["predictions"]).astype(bool)
                for name, run in binary_runs.items()},
            root / "discrimination",
            degradation_classes=matrix_cfg.degradation_classes,
            class_names=cfg.classes, sample_mask=mask, logger=logger)

    conclusion = _conclusion(table)
    if discrimination is not None:
        conclusion["discrimination"] = discrimination["report"]["best_by_margin"]
    (root / "experiment_matrix.json").write_text(json.dumps({
        "question": ("Does multi-temporal trajectory analysis improve "
                     "detection of land degradation compared with "
                     "conventional trend analysis?"),
        "protocol": ("All methods share the same spatial block folds, "
                     "samples and metric code, and the reported comparison "
                     "is re-scored on the samples EVERY method evaluated, so "
                     "no two methods are compared on different test sets. "
                     "The binary degradation task is the like-for-like "
                     "comparison, because the conventional trend rule cannot "
                     "attempt a multiclass problem."),
        "coverage": coverage,
        "rows": rows, "skipped": skipped, "conclusion": conclusion,
        "caveat": ("Development/testing outputs on synthetic data: this "
                   "compares methods on a generator, not on a landscape."),
    }, indent=2, default=str))
    (root / "method_metrics.json").write_text(
        json.dumps(details, indent=2, default=str))
    if logger is not None and skipped:
        for entry in skipped:
            logger.warning("  matrix %s skipped: %s", entry["method"],
                           entry["reason"])
    return table


def _conclusion(table: pd.DataFrame) -> dict:
    """State what the numbers support, in the binary comparison only."""
    binary = table[table["task"] == "binary_degradation"]
    if binary.empty:
        return {"available": False}
    ranked = binary.sort_values("f1_macro", ascending=False)
    best = ranked.iloc[0]
    baseline = binary[binary["method"] == "baseline_trend"]
    result = {
        "available": True,
        "task": "binary_degradation",
        "best_method": str(best["method"]),
        "best_f1_macro": float(best["f1_macro"]),
        "ranking": ranked[["method", "f1_macro"]].to_dict(orient="records"),
    }
    if not baseline.empty:
        difference = float(best["f1_macro"] - baseline.iloc[0]["f1_macro"])
        spread = max(float(best["fold_f1_macro_std"])
                     if np.isfinite(best["fold_f1_macro_std"]) else 0.0,
                     float(baseline.iloc[0]["fold_f1_macro_std"])
                     if np.isfinite(baseline.iloc[0]["fold_f1_macro_std"])
                     else 0.0)
        result.update({
            "baseline_f1_macro": float(baseline.iloc[0]["f1_macro"]),
            "improvement_over_baseline": difference,
            "fold_spread_used": spread,
            "exceeds_fold_spread": bool(abs(difference) > spread),
            "statement": (
                "The best learned method exceeds the conventional "
                "Mann-Kendall/Sen rule by "
                f"{difference:.4f} macro F1 on the binary degradation task, "
                + ("which is larger than the largest across-fold standard "
                   f"deviation involved ({spread:.4f})."
                   if abs(difference) > spread else
                   "which is within the across-fold standard deviation of "
                   f"{spread:.4f} and should not be read as an improvement.")),
        })

    # The binary task alone understates the case for the framework: the
    # conventional rule cannot attempt the multiclass problem at all, so the
    # multiclass comparison is reported alongside it rather than buried.
    multiclass = table[table["task"] == "multiclass_trajectory"]
    if not multiclass.empty:
        ranked_multi = multiclass.sort_values("f1_macro", ascending=False)
        result["multiclass"] = {
            "ranking": ranked_multi[["method", "f1_macro"]].to_dict(
                orient="records"),
            "baseline_participates": False,
            "note": ("The conventional trend rule produces only a "
                     "significant-decline flag, so it cannot attempt the "
                     "multiclass trajectory task. Where the extra features "
                     "matter is exactly here: distinguishing cyclic, "
                     "recovering and stable trajectories, which a monotonic "
                     "trend test cannot separate by construction."),
        }
        binary_by_method = binary.set_index("method")["f1_macro"].to_dict()
        multi_by_method = ranked_multi.set_index("method")["f1_macro"].to_dict()
        if "rf_basic" in multi_by_method and "rf_proposed" in multi_by_method:
            result["multiclass"]["proposed_minus_basic"] = float(
                multi_by_method["rf_proposed"] - multi_by_method["rf_basic"])
            if "rf_basic" in binary_by_method \
                    and "rf_proposed" in binary_by_method:
                result["multiclass"]["binary_proposed_minus_basic"] = float(
                    binary_by_method["rf_proposed"]
                    - binary_by_method["rf_basic"])
    return result
