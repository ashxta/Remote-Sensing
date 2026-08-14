"""Explainability for the trajectory classifiers (M3 Part 6).

INTERPRETATION LIMIT - READ FIRST
--------------------------------
Everything in this module measures PREDICTIVE ASSOCIATION inside a fitted
model. None of it measures causal influence on the land surface. If
`restrend` ranks highest, the correct statement is "the model relies on the
climate-adjusted trend to separate these classes", not "climate-adjusted
decline causes degradation". Correlated features also share credit
arbitrarily: NDVI mean and NDVI median carry nearly the same information, so
whichever the trees happen to split on first absorbs the importance of both.

Three views are provided because each fails differently:

impurity importance    cheap, but biased towards high-cardinality and
                       high-variance features, and computed on TRAINING
                       data, so it partly reflects overfitting.
permutation importance measured on HELD-OUT fold data, so it answers "how
                       much does out-of-fold performance drop when this
                       feature is scrambled" - the honest version, and the
                       one to quote.
SHAP (optional)        per-sample additive attributions, which additionally
                       show WHICH class a feature pushes towards. Requires
                       the optional `shap` package; skipped cleanly when it
                       is missing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .config import ExplainConfig, RFExperimentConfig
from .validation import fit_random_forest

__all__ = ["IMPORTANCE_DISCLAIMER", "shap_available", "permutation_importance",
           "spatial_permutation_importance", "shap_importance",
           "explain_experiment"]

IMPORTANCE_DISCLAIMER = (
    "Feature importance is predictive association inside a fitted model, not "
    "causal influence on the land surface. Correlated features share credit "
    "arbitrarily.")


def shap_available() -> bool:
    try:
        import shap  # noqa: F401
    except Exception:
        return False
    return True


def _score(y_true, y_pred) -> float:
    """Macro F1, so a collapsed class is penalised, not hidden by accuracy."""
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def permutation_importance(model, imputer, x, y, feature_names: Sequence[str],
                           *, repeats: int = 5, seed: int = 42) -> pd.DataFrame:
    """Drop in macro F1 when each feature is scrambled, on held-out data.

    The permutation is applied to the RAW feature matrix before imputation,
    so the measured effect includes how the fitted imputer treats the
    feature - which is what actually happens at prediction time.
    """
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y)
    feature_names = list(feature_names)
    if x.shape[1] != len(feature_names):
        raise ValueError("feature matrix and names disagree")
    rng = np.random.default_rng(seed)
    baseline = _score(y, model.predict(imputer.transform(x)))

    rows = []
    for column, name in enumerate(feature_names):
        drops = []
        for _ in range(repeats):
            shuffled = x.copy()
            shuffled[:, column] = shuffled[rng.permutation(len(x)), column]
            drops.append(baseline - _score(
                y, model.predict(imputer.transform(shuffled))))
        rows.append({"feature": name,
                     "importance_mean": float(np.mean(drops)),
                     "importance_std": float(np.std(drops)),
                     "baseline_macro_f1": baseline})
    return pd.DataFrame(rows).sort_values("importance_mean",
                                          ascending=False
                                          ).reset_index(drop=True)


def spatial_permutation_importance(features: pd.DataFrame, labels, fold_grid,
                                   *, feature_names: Sequence[str],
                                   sample_mask=None,
                                   rf_cfg: RFExperimentConfig | None = None,
                                   cfg: ExplainConfig | None = None
                                   ) -> pd.DataFrame:
    """Permutation importance averaged over the spatial CV folds.

    For each fold the model is refitted on the training folds and the
    permutation is measured on the held-out fold, so importance is never
    read off data the model was fitted on.
    """
    rf_cfg = rf_cfg or RFExperimentConfig()
    cfg = cfg or ExplainConfig()
    feature_names = list(feature_names)
    x = features.loc[:, feature_names].to_numpy(dtype="float64")
    y = np.asarray(labels)
    folds = np.asarray(fold_grid).reshape(-1)
    mask = np.ones(len(y), bool) if sample_mask is None \
        else np.asarray(sample_mask, bool)

    frames = []
    for fold in sorted(np.unique(folds[mask])):
        test = mask & (folds == fold)
        train = mask & ~test
        if not test.any() or np.unique(y[train]).size < 2:
            continue
        imputer, model = fit_random_forest(x[train], y[train], rf_cfg)
        frame = permutation_importance(
            model, imputer, x[test], y[test], feature_names,
            repeats=cfg.permutation_repeats,
            seed=cfg.permutation_seed + int(fold))
        frame["fold"] = int(fold)
        frames.append(frame)
    if not frames:
        raise ValueError("permutation importance produced no evaluable folds")

    combined = pd.concat(frames, ignore_index=True)
    summary = combined.groupby("feature").agg(
        importance_mean=("importance_mean", "mean"),
        importance_std=("importance_mean", "std"),
        n_folds=("fold", "nunique")).reset_index()
    summary["importance_std"] = summary["importance_std"].fillna(0.0)
    return summary.sort_values("importance_mean", ascending=False
                               ).reset_index(drop=True)


def shap_importance(model, x, feature_names: Sequence[str], *,
                    classes=None, cfg: ExplainConfig | None = None):
    """Mean |SHAP value| per feature, and per feature-and-class.

    Returns None when the optional `shap` package is unavailable or its tree
    explainer cannot handle the fitted model - an optional analysis must
    never break an experiment run.
    """
    cfg = cfg or ExplainConfig()
    if not cfg.shap or not shap_available():
        return None
    try:
        import shap

        x = np.asarray(x, dtype="float64")
        if len(x) > cfg.shap_max_samples:
            rng = np.random.default_rng(cfg.shap_seed)
            x = x[rng.choice(len(x), cfg.shap_max_samples, replace=False)]
        values = shap.TreeExplainer(model).shap_values(x, check_additivity=False)
        stacked = np.asarray(values)
        # shap returns (samples, features, classes) for modern versions and
        # a list of (samples, features) for older ones.
        if stacked.ndim == 3 and stacked.shape[0] == len(x):
            per_class = np.abs(stacked).mean(axis=0).T      # (classes, feat)
        elif stacked.ndim == 3:
            per_class = np.abs(stacked).mean(axis=1)        # (classes, feat)
        else:
            per_class = np.abs(stacked).mean(axis=0)[None, :]
        overall = pd.DataFrame({
            "feature": list(feature_names),
            "mean_abs_shap": per_class.mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        labels = [str(c) for c in (classes if classes is not None
                                   else range(per_class.shape[0]))]
        by_class = pd.DataFrame(per_class.T, index=list(feature_names),
                                columns=labels[:per_class.shape[0]])
        return {"overall": overall, "by_class": by_class,
                "n_samples": int(len(x))}
    except Exception:                                   # pragma: no cover
        return None


def explain_experiment(features: pd.DataFrame, labels, fold_grid, output_dir,
                       *, feature_names: Sequence[str], impurity=None,
                       sample_mask=None,
                       rf_cfg: RFExperimentConfig | None = None,
                       cfg: ExplainConfig | None = None, logger=None) -> dict:
    """Run every available explanation and save machine-readable outputs."""
    rf_cfg = rf_cfg or RFExperimentConfig()
    cfg = cfg or ExplainConfig()
    feature_names = list(feature_names)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    permutation = spatial_permutation_importance(
        features, labels, fold_grid, feature_names=feature_names,
        sample_mask=sample_mask, rf_cfg=rf_cfg, cfg=cfg)
    permutation.to_csv(out / "permutation_importance.csv", index=False)

    comparison = permutation.set_index("feature")[["importance_mean"]].rename(
        columns={"importance_mean": "permutation_macro_f1_drop"})
    if impurity is not None:
        comparison["impurity_importance"] = pd.Series(impurity)
        comparison["rank_impurity"] = comparison["impurity_importance"].rank(
            ascending=False)
    comparison["rank_permutation"] = comparison[
        "permutation_macro_f1_drop"].rank(ascending=False)
    comparison.sort_values("permutation_macro_f1_drop", ascending=False
                           ).to_csv(out / "importance_comparison.csv")

    y = np.asarray(labels)
    mask = np.ones(len(y), bool) if sample_mask is None \
        else np.asarray(sample_mask, bool)
    x = features.loc[:, feature_names].to_numpy(dtype="float64")
    imputer, model = fit_random_forest(x[mask], y[mask], rf_cfg)
    shap_result = shap_importance(model, imputer.transform(x[mask]),
                                  feature_names, classes=model.classes_,
                                  cfg=cfg)
    if shap_result is not None:
        shap_result["overall"].to_csv(out / "shap_importance.csv", index=False)
        shap_result["by_class"].to_csv(out / "shap_importance_by_class.csv")

    report = {
        "disclaimer": IMPORTANCE_DISCLAIMER,
        "permutation": {
            "method": "macro F1 drop when a feature is scrambled, measured "
                      "on held-out spatial folds",
            "repeats": cfg.permutation_repeats,
            "top": permutation.head(cfg.top_n).to_dict(orient="records"),
        },
        "shap": {
            "available": shap_result is not None,
            "reason": None if shap_result is not None else (
                "the optional 'shap' package is not installed or its tree "
                "explainer could not handle this model"),
            "n_samples": shap_result["n_samples"] if shap_result else None,
            "top": shap_result["overall"].head(cfg.top_n).to_dict(
                orient="records") if shap_result else None,
        },
    }
    (out / "explainability.json").write_text(json.dumps(report, indent=2))
    if logger is not None:
        top = ", ".join(permutation["feature"].head(5))
        logger.info("  permutation importance (held-out folds), top 5: %s",
                    top)
        logger.info("  SHAP: %s", "computed" if shap_result is not None
                    else "unavailable (optional dependency)")
    return {"permutation": permutation, "comparison": comparison,
            "shap": shap_result, "report": report}
