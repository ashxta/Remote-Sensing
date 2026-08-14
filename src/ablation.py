"""Configuration-driven ablation framework (M2 Part 6).

One reusable experiment mechanism runs every ablation cell; there are no
duplicated per-experiment scripts. An ablation cell is fully described by
`config.AblationExperiment` (id, name, feature groups), so adding or
reordering experiments is a configuration change, not a code change.

The question the study answers is narrow and explicit: does each additional
temporal-analysis component (trend, RESTREND, cyclicity,
disturbance/recovery, rainfall context) carry information the classifier can
actually use, over and above the simpler feature sets?

Every cell writes, under `<output_dir>/<experiment_id>/`:

    configuration.json    id, name, feature groups, resolved feature list,
                          model configuration, validation configuration
    metrics.json          pooled and per-fold metrics, confusion matrix
    confusion_matrix.csv  the same matrix, machine-readable
    predictions.csv       per-sample truth, prediction, fold, evaluated flag
    probabilities.csv     per-class model confidence, class-labelled columns
    feature_importance.csv mean impurity importance across folds
    log.txt               what this cell did

plus a comparison table (CSV and JSON) for the whole study.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import AblationExperiment, Config, RFExperimentConfig
from .features import feature_names
from .validation import spatial_cv_rf

__all__ = ["experiment_id", "resolve_feature_set", "run_ablation_study",
           "feature_sets_for"]


def experiment_id(name: str, features: Sequence[str],
                  cfg: RFExperimentConfig) -> str:
    """Stable id: same name + features + model configuration -> same id."""
    payload = json.dumps({"name": name, "features": list(features),
                          "config": asdict(cfg)}, sort_keys=True)
    return f"{name}_{hashlib.sha256(payload.encode()).hexdigest()[:10]}"


def resolve_feature_set(experiment: AblationExperiment) -> list:
    """Feature columns an ablation cell may use, in canonical order."""
    return feature_names(experiment.groups)


def feature_sets_for(experiments: Sequence[AblationExperiment]) -> dict:
    """{experiment name: resolved feature list} for the whole design."""
    return {e.name: resolve_feature_set(e) for e in experiments}


def _write_cell(out: Path, *, spec, features, rf_cfg, result, labels,
                folds, mask, spatial_cfg) -> None:
    out.mkdir(parents=True, exist_ok=True)
    metrics = result["metrics"]
    (out / "configuration.json").write_text(json.dumps({
        "experiment": spec.id,
        "name": spec.name,
        "feature_groups": list(spec.groups),
        "features": list(features),
        "n_features": len(features),
        "model": asdict(rf_cfg),
        "validation": asdict(spatial_cfg),
    }, indent=2))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(metrics["confusion_matrix"],
                 index=[f"true_{c}" for c in metrics["labels"]],
                 columns=[f"pred_{c}" for c in metrics["labels"]]
                 ).to_csv(out / "confusion_matrix.csv")
    pd.DataFrame({
        "truth": labels,
        "prediction": result["predictions"],
        "fold": np.asarray(folds).reshape(-1),
        "in_sample_set": np.asarray(mask, bool),
        "evaluated": result["evaluated"],
    }).to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(result["probabilities"],
                 columns=[f"probability_{c}" for c in result["classes"]]
                 ).to_csv(out / "probabilities.csv", index=False)
    result["importance"].to_csv(out / "feature_importance.csv")
    fold_summary = metrics["fold_summary"]
    (out / "log.txt").write_text(
        f"experiment {spec.id} ({spec.name})\n"
        f"feature groups: {', '.join(spec.groups)}\n"
        f"features ({len(features)}): {', '.join(features)}\n"
        f"validation: spatial block CV, {fold_summary['n_folds']} folds "
        f"evaluated, block_size={spatial_cfg.block_size}, "
        f"buffer_blocks={spatial_cfg.buffer_blocks}\n"
        f"samples evaluated: {metrics['n_evaluated']}\n"
        f"accuracy {metrics['accuracy']:.4f} | macro F1 "
        f"{metrics['f1_macro']:.4f} | weighted F1 "
        f"{metrics['f1_weighted']:.4f}\n"
        f"fold macro F1 mean+/-std: "
        f"{fold_summary.get('f1_macro_mean', float('nan')):.4f} +/- "
        f"{fold_summary.get('f1_macro_std', float('nan')):.4f}\n"
        "Probabilities are model confidence estimates, not certainty.\n")


def run_ablation_study(features: pd.DataFrame, labels, fold_grid,
                       output_dir, cfg: Config | None = None, *,
                       sample_mask=None, rf_cfg: RFExperimentConfig | None = None,
                       experiments: Sequence[AblationExperiment] | None = None,
                       block_row=None, block_col=None,
                       logger=None) -> pd.DataFrame:
    """Run every ablation cell through the same reusable pipeline.

    Returns the comparison table and writes it as CSV and JSON next to the
    per-experiment directories.
    """
    cfg = cfg or Config()
    rf_cfg = rf_cfg or cfg.research.model
    experiments = list(experiments if experiments is not None
                       else cfg.research.ablation.experiments)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in experiments:
        cell_features = resolve_feature_set(spec)
        eid = experiment_id(spec.name, cell_features, rf_cfg)
        result = spatial_cv_rf(features, labels, fold_grid,
                               sample_mask=sample_mask,
                               feature_names=cell_features, cfg=rf_cfg,
                               block_row=block_row, block_col=block_col)
        _write_cell(root / eid, spec=spec, features=cell_features,
                    rf_cfg=rf_cfg, result=result, labels=labels,
                    folds=fold_grid, mask=sample_mask
                    if sample_mask is not None
                    else np.ones(len(features), bool),
                    spatial_cfg=rf_cfg.block_cv)
        metrics = result["metrics"]
        summary = metrics["fold_summary"]
        rows.append({
            "experiment": spec.id,
            "experiment_id": eid,
            "feature_set": spec.name,
            "feature_groups": "+".join(spec.groups),
            "n_features": len(cell_features),
            "n_evaluated": metrics["n_evaluated"],
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "cohen_kappa": metrics["cohen_kappa"],
            "fold_f1_macro_mean": summary.get("f1_macro_mean", float("nan")),
            "fold_f1_macro_std": summary.get("f1_macro_std", float("nan")),
            "fold_accuracy_mean": summary.get("accuracy_mean", float("nan")),
            "fold_accuracy_std": summary.get("accuracy_std", float("nan")),
        })
        if logger is not None:
            logger.info("  ablation %-20s %2d features | macro F1 %.4f "
                        "(fold mean %.4f +/- %.4f)", spec.name,
                        len(cell_features), metrics["f1_macro"],
                        rows[-1]["fold_f1_macro_mean"],
                        rows[-1]["fold_f1_macro_std"])

    table = pd.DataFrame(rows)
    table.to_csv(root / "ablation_comparison.csv", index=False)
    (root / "ablation_comparison.json").write_text(
        json.dumps({"experiments": rows,
                    "feature_sets": feature_sets_for(experiments),
                    "note": "Development/testing outputs on synthetic data."},
                   indent=2))
    return table
