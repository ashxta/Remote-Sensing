"""End-to-end M1 pipeline.

Stages
  1 load stacks + capture georeference
  2 data-quality assessment (Part 5)
  3 trajectory statistics + features (Part 3)
  4 recovery analysis (Part 4)
  5 trajectory classification (RF, optional 1D-CNN)
  6 method-comparison analysis
  7 georeferenced outputs + figures (Part 7)

Every run creates results/<experiment_id>/ containing the exact config,
environment snapshot, metrics, figures, georeferenced predictions and log.

    python run_pipeline.py                  full run
    python run_pipeline.py --no-cnn         skip the 1D-CNN
    python run_pipeline.py --seed 7         different seed
    python run_pipeline.py --name my_exp    label the experiment
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from src import geo
from src import maps as M
from src import quality as Q
from src import recovery as RC
from src.classify import build_features, train_rf, train_cnn, FEATURES
from src.config import Config
from src.data_source import RasterStackSource
from src.reproducibility import start_experiment


# --------------------------------------------------------------------------
def load_stacks(cfg: Config):
    """Load NDVI/rain/truth and capture the spatial reference.

    Thin wrapper over `data_source.RasterStackSource`, which owns the data
    contract (NoData -> NaN, band descriptions as the time axis, provenance
    metadata). Kept as a four-tuple because M1 callers and tests use that
    shape; new code should take a `StandardizedDataset` instead.
    """
    dataset = RasterStackSource.from_config(cfg).load()
    return dataset.ndvi, dataset.rain, dataset.truth, dataset.georef


def sanity_check_mk(ndvi, logger):
    """Cross-validate the vectorized MK against pymannkendall."""
    try:
        import pymannkendall as pmk
    except ImportError:
        logger.warning("pymannkendall not installed; MK cross-check skipped")
        return None
    from src import timeseries as TS
    rng = np.random.default_rng(0)
    cols = [j for j in range(ndvi.shape[1])
            if np.isfinite(ndvi[:, j]).all()]
    idx = rng.choice(cols, min(5, len(cols)), replace=False)
    _, _, z, _, _ = TS.mann_kendall(ndvi[:, idx])
    worst = max(abs(pmk.original_test(ndvi[:, i]).z - z[k])
                for k, i in enumerate(idx))
    ok = worst < 1e-6
    logger.info("MK cross-check vs pymannkendall: %s (max |dz|=%.2e)",
                "PASS" if ok else "FAIL", worst)
    return ok


# --------------------------------------------------------------------------
def main(cfg: Config):
    exp = start_experiment(cfg)
    log = exp.logger
    log.info("STAGE 1/7  loading stacks")
    ndvi_grid, rain_grid, truth_grid, georef = load_stacks(cfg)
    T, H, W = ndvi_grid.shape
    georef.save(exp.path("config", "georeference.json"))
    log.info("grid %dx%d, %d time steps, CRS=%s, res=%s",
             H, W, T, georef.crs, georef.resolution)

    ndvi_flat = ndvi_grid.reshape(T, -1)
    rain_flat = rain_grid.reshape(T, -1)

    log.info("STAGE 2/7  data-quality assessment")
    report = Q.assess(ndvi_flat, cfg.quality)
    qsum = report.summary()
    exp.save_metrics("data_quality", qsum)
    log.info("usable pixels: %d/%d (%.1f%%)", qsum["n_usable"],
             qsum["n_pixels"], 100 * qsum["usable_fraction"])
    for name, n in qsum["flag_counts"].items():
        if n and name != "OK":
            log.info("  excluded %-18s %d", name, n)

    usable_mask = report.usable.reshape(H, W)
    ndvi = ndvi_flat[:, report.usable]
    rain = rain_flat[:, report.usable]
    if ndvi.shape[1] == 0:
        raise RuntimeError("no pixels passed the quality gate")

    log.info("STAGE 3/7  trajectory statistics")
    sanity_check_mk(ndvi, log)
    F, extras = build_features(ndvi, rain, cfg)
    log.info("features: %s", ", ".join(FEATURES))

    n_restrend_valid = int(np.nansum(extras["restrend_valid"]))
    log.info("RESTREND interpretable (rainfall relation significant) in "
             "%d/%d pixels (%.1f%%)", n_restrend_valid, ndvi.shape[1],
             100 * n_restrend_valid / ndvi.shape[1])
    log.info("periodicity detected in %d pixels (NOT proof of any specific "
             "land-use practice)", int(np.nansum(extras["cyc_periodic"])))

    log.info("STAGE 4/7  recovery analysis")
    rec = extras["recovery"]
    rsum = RC.summary(rec)
    exp.save_metrics("recovery", rsum)
    for k, v in rsum["status_counts"].items():
        log.info("  %-18s %d", k, v)

    results = {"data_quality": qsum, "recovery": rsum}

    log.info("STAGE 5/7  trajectory classification")
    truth = truth_grid.reshape(-1)[report.usable] \
        if truth_grid is not None else None
    traj_pred = None
    if truth is not None:
        rng = np.random.default_rng(cfg.seed)
        take = []
        for k in sorted(cfg.classes):
            ki = np.where(truth == k)[0]
            if ki.size:
                take.append(rng.choice(
                    ki, min(cfg.model.samples_per_class, ki.size),
                    replace=False))
        samp = np.concatenate(take)
        Fs = F.iloc[samp].reset_index(drop=True)
        ys = truth[samp]
        # Models cannot consume NaN; drop incomplete samples explicitly and
        # record how many, rather than imputing them.
        keep = np.isfinite(Fs[FEATURES].values).all(axis=1)
        n_drop = int((~keep).sum())
        if n_drop:
            log.info("dropped %d training samples with incomplete features",
                     n_drop)
        Fs, ys = Fs[keep].reset_index(drop=True), ys[keep]

        rf, rf_mets, cm, report_txt, _ = train_rf(Fs, ys, cfg)
        log.info("RF  %s", {k: round(v, 4) for k, v in rf_mets.items()})
        log.info("     (random pixel split: spatially OPTIMISTIC; the "
                 "spatially validated result comes from run_m2/m3_experiments)")
        rf_mets = dict(rf_mets)
        rf_mets["validation"] = "random_pixel_split"
        rf_mets["caveat"] = (
            "Random pixel splitting puts spatially autocorrelated neighbours "
            "on both sides of the split, so this figure is optimistic. The "
            "spatial block cross-validated result in the M2/M3 runners is "
            "the one to quote.")
        results["random_forest"] = rf_mets
        exp.path("metrics", "rf_classification_report.txt").write_text(
            report_txt)
        M.save_confusion(cm, exp.figure("confusion_rf.png"),
                         "Random Forest trajectory classification", cfg)
        imp = pd.Series(rf.feature_importances_, index=FEATURES) \
            .sort_values(ascending=False)
        exp.save_table("rf_feature_importance", imp.to_frame("importance"))
        log.info("top features: %s", ", ".join(imp.index[:4]))

        try:
            import joblib
            joblib.dump(rf, exp.path("models", "random_forest.joblib"))
        except Exception as e:                       # pragma: no cover
            log.warning("model not persisted: %s", e)

        if cfg.model.use_cnn:
            try:
                _, cnn_mets = train_cnn(ndvi[:, samp][:, keep], ys, cfg)
                log.info("CNN %s", {k: round(v, 4)
                                    for k, v in cnn_mets.items()})
                results["cnn_1d"] = cnn_mets
            except Exception as e:
                log.warning("CNN stage skipped: %s", e)

        Fall = F[FEATURES].values
        complete = np.isfinite(Fall).all(axis=1)
        traj_pred = np.full(F.shape[0], np.nan)
        if complete.any():
            traj_pred[complete] = rf.predict(F[FEATURES][complete])

        log.info("STAGE 6/7  method comparison")
        mk_sig_neg = (extras["mk_p"] < cfg.trend.alpha) & (F.sen.values < 0)
        rs_sig_neg = ((extras["restrend_p"] < cfg.restrend.alpha)
                      & (F.restrend.values < 0) & extras["restrend_valid"])
        comp = {}
        for cid, cname in cfg.classes.items():
            m = truth == cid
            if not m.any():
                continue
            comp[cname] = {
                "n_pixels": int(m.sum()),
                "pct_flagged_declining_by_mann_kendall":
                    float(100 * np.nanmean(mk_sig_neg[m])),
                "pct_flagged_declining_by_restrend":
                    float(100 * np.nanmean(rs_sig_neg[m])),
                "pct_classified_correctly_by_rf":
                    float(100 * np.nanmean(traj_pred[m] == cid)),
            }
        results["method_comparison"] = comp
        for cname, d in comp.items():
            log.info("  %-26s MK %5.1f%% | RESTREND %5.1f%% | RF %5.1f%%",
                     cname, d["pct_flagged_declining_by_mann_kendall"],
                     d["pct_flagged_declining_by_restrend"],
                     d["pct_classified_correctly_by_rf"])
        periodic_median = np.nanmedian(
            F.cyc_period.values[extras["cyc_periodic"]]) \
            if extras["cyc_periodic"].any() else float("nan")
        results["median_detected_period_steps"] = float(periodic_median)
        log.info("median detected period: %.1f time steps", periodic_median)

    log.info("STAGE 7/7  georeferenced outputs")
    layers = [
        ("sens_slope", F.sen.values, "float32",
         "Theil-Sen slope of NDVI per time step"),
        ("mann_kendall_z", F.mk_z.values, "float32",
         "Mann-Kendall Z, Hamed-Rao autocorrelation corrected"),
        ("mann_kendall_p", extras["mk_p"], "float32", "MK two-sided p-value"),
        ("restrend_slope", F.restrend.values, "float32",
         "Climate-adjusted (RESTREND) slope; see restrend_valid"),
        ("restrend_valid", extras["restrend_valid"].astype(float), "uint8",
         "1 where the NDVI~rainfall relation is significant"),
        ("cyclicity_enrichment", F.cyc_enrichment.values, "float32",
         "Spectral enrichment in the configured period band; 1.0 = noise"),
        ("dominant_period", F.cyc_period.values, "float32",
         "Dominant period in time steps (periodicity, not attribution)"),
        ("break_index", extras["break_index"].astype(float), "float32",
         "Detected structural break index (-1 = none)"),
        ("recovery_status", rec["recovery_status"].astype(float), "uint8",
         "0 none 1 recovered 2 recovering 3 not-recovering 4 insufficient"),
        ("recovery_fraction", rec["recovery_fraction"], "float32",
         "Recovered share of the disturbance magnitude"),
        ("quality_flag", report.flag.astype(float), "uint8",
         "0 OK 1 insufficient 2 too-missing 3 constant 4 out-of-range"),
    ]
    all_mask = np.ones((H, W), dtype=bool)
    for name, vals, dtype, desc in layers:
        target = all_mask if name == "quality_flag" else usable_mask
        geo.write_layer(exp.path("predictions", f"{name}.tif"),
                        vals, target, georef, dtype=dtype, description=desc)
    if traj_pred is not None:
        geo.write_layer(exp.path("predictions", "trajectory_class.tif"),
                        traj_pred, usable_mask, georef, dtype="uint8",
                        description="RF trajectory class")
        M.save_classmap(traj_pred, usable_mask, (H, W),
                        exp.figure("trajectory_map.png"), cfg)
        M.example_trajectories(ndvi, truth, exp.figure(
            "example_trajectories.png"), cfg)
    log.info("wrote %d georeferenced layers", len(layers))

    M.save_continuous(F.sen.values, usable_mask, (H, W),
                      "Theil-Sen slope of NDVI (per time step)",
                      exp.figure("trend_sen_map.png"), center=0)
    M.save_continuous(F.restrend.values, usable_mask, (H, W),
                      "RESTREND climate-adjusted slope",
                      exp.figure("restrend_map.png"), center=0)
    sdg_stats = M.sdg_productivity(F.sen.values, extras["mk_p"], usable_mask,
                                   (H, W), exp.figure(
                                       "productivity_trajectory_map.png"), cfg)
    results["productivity_trajectory"] = sdg_stats

    exp.save_metrics("results", results)
    exp.save_table("feature_summary", F.describe().T.round(5))
    log.info("EXPERIMENT COMPLETE -> %s", exp.root)
    return exp, results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cnn", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--config", type=str, default=None,
                    help="path to a saved config.json to reproduce a run")
    a = ap.parse_args()

    cfg = Config.load(a.config) if a.config else Config()
    if a.seed is not None:
        cfg.seed = a.seed
    if a.name:
        cfg.experiment_name = a.name
    if a.no_cnn:
        cfg.model.use_cnn = False

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main(cfg)
