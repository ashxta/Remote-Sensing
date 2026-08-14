"""Trajectory classification: engineered features -> Random Forest, and a
1D-CNN on the raw NDVI series for comparison."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score,
                             confusion_matrix, classification_report)

from . import config as C
from . import timeseries as TS
from . import recovery as RC
from .config import Config

FEATURES = ["mean", "std", "cv", "amp", "sen", "mk_z", "restrend",
            "cyc_score", "cyc_enrichment", "cyc_period",
            "has_break", "break_t", "break_dslope", "break_gain",
            "has_disturbance", "recovery_fraction", "recovery_slope",
            "disturbance_magnitude"]


def build_features(ndvi, rain, cfg: Config | None = None):
    """Per-pixel trajectory features.

    Parameters
    ----------
    ndvi, rain : (T, N) arrays, NaN allowed for missing observations.
    cfg : Config; all thresholds are read from it (no hard-coded values).

    Returns
    -------
    (DataFrame of features, dict of diagnostics)

    NaN policy: features are computed NaN-aware. Pixels that fail the
    quality gate produce NaN features and must be excluded upstream by the
    caller rather than silently imputed here.
    """
    cfg = cfg or Config()
    T = ndvi.shape[0]

    mean = np.nanmean(ndvi, axis=0)
    std = np.nanstd(ndvi, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.where(np.abs(mean) > 1e-9, std / mean, np.nan)
    amp = (np.nanpercentile(ndvi, 95, axis=0)
           - np.nanpercentile(ndvi, 5, axis=0))

    sen = TS.sens_slope(ndvi, min_obs=cfg.trend.min_obs)
    S, var, z_raw, p_raw, tau = TS.mann_kendall(
        ndvi, min_obs=cfg.trend.min_obs,
        apply_ties=cfg.trend.apply_ties_correction)
    if cfg.trend.apply_autocorrelation_correction:
        zc, pc = TS.hamed_rao_correction(
            ndvi, S, var, alpha=cfg.trend.hr_significance_level,
            max_lag_fraction=cfg.trend.hr_max_lag_fraction,
            min_obs=cfg.trend.min_obs)
    else:
        zc, pc = z_raw, p_raw

    rt = TS.restrend(ndvi, rain, min_r2=cfg.restrend.min_r2,
                     alpha=cfg.restrend.alpha,
                     require_positive_beta=cfg.restrend.require_positive_beta,
                     min_obs=cfg.restrend.min_obs)
    cyc = TS.cyclicity(ndvi, min_period=cfg.cyclicity.min_period,
                       max_period=cfg.cyclicity.max_period,
                       detrend=cfg.cyclicity.detrend,
                       threshold=cfg.cyclicity.periodicity_threshold,
                       min_obs=cfg.cyclicity.min_obs)
    bk = TS.best_breakpoint(ndvi, min_segment=cfg.breakpoint.min_segment_length,
                            min_gain=cfg.breakpoint.min_variance_gain,
                            alpha=cfg.breakpoint.significance_alpha,
                            apply_test=cfg.breakpoint.apply_significance_test)
    rec = RC.analyze(ndvi, bk["index"], cfg.recovery)

    # Structural vs unknown missingness.
    # Break and recovery descriptors are UNDEFINED (not unknown) for pixels
    # with no detected break or no disturbance. Leaving them NaN would make
    # ~half the map unpredictable. We therefore encode the condition
    # explicitly with a binary indicator and set the dependent descriptors
    # to a neutral 0. This is an explicit encoding of a known state, not
    # imputation of an unobserved value.
    has_break = np.isfinite(bk["index"]) & (bk["index"] >= 0)
    has_dist = rec["recovery_status"] != RC.STATUS_NO_DISTURBANCE

    def structural(vals, present):
        v = np.asarray(vals, dtype="float64").copy()
        v[~present] = 0.0
        v[present & ~np.isfinite(v)] = 0.0
        return v

    df = pd.DataFrame({
        "mean": mean, "std": std, "cv": cv, "amp": amp,
        "sen": sen, "mk_z": zc, "restrend": rt["slope"],
        "cyc_score": cyc["score"], "cyc_enrichment": cyc["enrichment"],
        "cyc_period": np.where(np.isfinite(cyc["dominant_period"]),
                               cyc["dominant_period"], 0.0),
        "has_break": has_break.astype("float64"),
        "break_t": structural(np.where(has_break, bk["index"] / float(T), 0.0),
                              has_break),
        "break_dslope": structural(bk["delta_slope"], has_break),
        "break_gain": structural(bk["gain"], has_break),
        "has_disturbance": has_dist.astype("float64"),
        "recovery_fraction": structural(rec["recovery_fraction"], has_dist),
        "recovery_slope": structural(rec["recovery_slope"], has_dist),
        "disturbance_magnitude": structural(rec["disturbance_magnitude"],
                                            has_dist),
    })
    extras = dict(mk_p=pc, mk_tau=tau, mk_z_raw=z_raw, mk_p_raw=p_raw,
                  restrend_p=rt["p"],
                  restrend_r2=rt["r2"], restrend_valid=rt["valid"],
                  restrend_beta=rt["beta"], restrend_beta_p=rt["beta_p"],
                  cyc_periodic=cyc["periodic"],
                  cyc_band_power=cyc["band_power"],
                  cyc_total_power=cyc["total_power"],
                  cyc_expected_fraction=cyc["expected_fraction"],
                  break_index=bk["index"], break_significant=bk["significant"],
                  break_p_adjusted=bk["p_adjusted"],
                  recovery=rec)
    return df.astype("float32"), extras


def train_rf(F, y, cfg: Config | None = None):
    cfg = cfg or Config()
    Xtr, Xte, ytr, yte = train_test_split(
        F[FEATURES], y, test_size=cfg.model.test_size, stratify=y,
        random_state=cfg.seed)
    rf = RandomForestClassifier(n_estimators=cfg.model.rf_trees, n_jobs=-1,
                                random_state=cfg.seed,
                                min_samples_leaf=cfg.model.rf_min_samples_leaf)
    rf.fit(Xtr, ytr)
    pred = rf.predict(Xte)
    mets = dict(
        Accuracy=accuracy_score(yte, pred),
        Kappa=cohen_kappa_score(yte, pred),
        F1_macro=f1_score(yte, pred, average="macro"))
    cm = confusion_matrix(yte, pred, labels=sorted(C.CLASSES))
    report = classification_report(
        yte, pred, labels=sorted(C.CLASSES),
        target_names=[C.CLASSES[k] for k in sorted(C.CLASSES)])
    return rf, mets, cm, report, (Xte.index.values, yte, pred)


def train_cnn(ndvi_series, y, cfg=None, verbose=True):
    cfg = cfg or Config()
    epochs = cfg.model.cnn_epochs
    """1D-CNN on the raw normalised NDVI series (T,) per sample."""
    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X = ndvi_series.T[:, None, :]                    # (Nsamp, 1, T)
    classes = sorted(cfg.classes)
    ymap = {c: i for i, c in enumerate(classes)}
    yi = np.array([ymap[v] for v in y])
    Xtr, Xte, ytr, yte = train_test_split(
        X, yi, test_size=cfg.model.test_size, stratify=yi, random_state=cfg.seed)
    # Normalisation is fitted on the TRAINING split only. Standardising the
    # whole array before splitting (as this did originally) lets the test
    # samples influence the scaling every training sample is expressed in,
    # which is preprocessing leakage even though the labels are untouched.
    # `cnn_experiment.run_spatial_cnn` is the leakage-audited research
    # version of this comparison; this function remains the M1 quick look.
    mu, sigma = Xtr.mean(), Xtr.std()
    Xtr = (Xtr - mu) / (sigma + 1e-6)
    Xte = (Xte - mu) / (sigma + 1e-6)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                nn.Dropout(0.3), nn.Linear(64, len(classes)))
        def forward(self, x):
            return self.net(x)

    model = Net().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.model.cnn_lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(
        torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr))
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg.model.cnn_batch, shuffle=True)
    Xte_t = torch.tensor(Xte, dtype=torch.float32).to(dev)

    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
        if verbose and (ep + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                acc = (model(Xte_t).argmax(1).cpu().numpy() == yte).mean()
            print(f"  cnn epoch {ep+1}: test acc {acc:.3f}")

    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).cpu().numpy()
    inv = {i: c for c, i in ymap.items()}
    pred_c = np.array([inv[p] for p in pred])
    yte_c = np.array([inv[p] for p in yte])
    mets = dict(Accuracy=accuracy_score(yte_c, pred_c),
                Kappa=cohen_kappa_score(yte_c, pred_c),
                F1_macro=f1_score(yte_c, pred_c, average="macro"))
    return model, mets
