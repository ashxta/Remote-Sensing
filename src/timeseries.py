"""Vectorized time-series statistics for NDVI raster stacks.

All estimators take x of shape (T, N): T time steps, N pixels. They are
NaN-aware: missing observations are excluded per pixel, and pixels with
fewer than the configured minimum number of valid observations return NaN
rather than a fabricated value.

M1 corrections applied to the original implementation:
  * Mann-Kendall now uses a PER-PIXEL variance with the tie correction and
    the true per-pixel valid-observation count (previously a single scalar
    no-ties variance was shared by every pixel).
  * Hamed-Rao now follows the published formula: significant
    autocorrelations of the DETRENDED RANKS across multiple lags
    (previously a mangled lag-1-only expression on raw values).
  * Sen's slope uses nanmedian over valid pairs.
  * RESTREND exposes an explicit validity flag; degenerate rainfall no
    longer hides behind an epsilon.
  * Breakpoint detection enforces minimum segment length and adds a Chow
    test; unresolvable pixels return -1 rather than index 0.
  * Cyclicity returns a documented score, dominant period and band power,
    and refuses to run when the series is too short for the period band.

References
----------
Mann (1945); Kendall (1975); Sen (1968);
Hamed & Rao (1998) J. Hydrology 204:182-196;
Evans & Geerken (2004) / Wessels et al. (2012) for RESTREND practice;
Chow (1960) for the structural-break F test.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm, f as fdist
from scipy.stats import t as tdist

__all__ = ["mann_kendall", "hamed_rao_correction", "sens_slope", "restrend",
           "best_breakpoint", "cyclicity", "cyclicity_significance",
           "RestrendResult", "BreakResult", "CyclicityResult", "MKResult"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as2d(x) -> np.ndarray:
    a = np.asarray(x, dtype="float64")
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"expected (T,) or (T, N) array, got {a.shape}")
    if a.shape[0] < 3:
        raise ValueError(f"need at least 3 time steps, got {a.shape[0]}")
    return a


def _tie_correction(x: np.ndarray) -> np.ndarray:
    """Sum over tied groups of t(t-1)(2t+5), per pixel, ignoring NaN."""
    T, N = x.shape
    xs = np.sort(x, axis=0)                    # NaN sorted to the end
    out = np.zeros(N)
    for j in range(N):
        col = xs[:, j]
        col = col[np.isfinite(col)]
        if col.size < 2:
            continue
        # run lengths of equal values
        brk = np.flatnonzero(np.diff(col)) + 1
        counts = np.diff(np.concatenate(([0], brk, [col.size])))
        tied = counts[counts > 1]
        if tied.size:
            out[j] = np.sum(tied * (tied - 1) * (2 * tied + 5))
    return out


class MKResult(dict):
    """dict with attribute access, so both styles work in callers."""
    __getattr__ = dict.__getitem__


# ---------------------------------------------------------------------------
# Mann-Kendall
# ---------------------------------------------------------------------------
def mann_kendall(x, min_obs: int = 10, apply_ties: bool = True):
    """NaN-aware vectorized Mann-Kendall test.

    Returns (S, var, z, p, tau) for backwards compatibility. Pixels with
    fewer than `min_obs` valid observations return NaN for z, p and tau.

    Variance uses the per-pixel valid count n:
        Var(S) = [n(n-1)(2n+5) - sum_ties] / 18
    """
    x = _as2d(x)
    T, N = x.shape
    finite = np.isfinite(x)
    n = finite.sum(axis=0).astype("float64")

    S = np.zeros(N)
    for i in range(T - 1):
        d = x[i + 1:] - x[i]
        # np.sign propagates NaN; nansum ignores invalid pairs
        S += np.nansum(np.sign(d), axis=0)

    ties = _tie_correction(x) if apply_ties else np.zeros(N)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (n * (n - 1) * (2 * n + 5) - ties) / 18.0
        var = np.where(var > 0, var, np.nan)
        z = np.where(S > 0, (S - 1) / np.sqrt(var),
                     np.where(S < 0, (S + 1) / np.sqrt(var), 0.0))
        p = 2 * (1 - norm.cdf(np.abs(z)))
        denom = 0.5 * n * (n - 1)
        tau = np.where(denom > 0, S / denom, np.nan)

    bad = n < min_obs
    z = np.where(bad, np.nan, z)
    p = np.where(bad, np.nan, p)
    tau = np.where(bad, np.nan, tau)
    var = np.where(bad, np.nan, var)
    return S, var, z, p, tau


def hamed_rao_correction(x, S, var, alpha: float = 0.05,
                         max_lag_fraction: float = 0.5, min_obs: int = 10):
    """Hamed & Rao (1998) autocorrelation correction.

    Correct procedure, as published:
      1. detrend the series with the Theil-Sen slope,
      2. rank the detrended series,
      3. compute autocorrelation of those ranks at lags 1..L,
      4. keep only autocorrelations significant at `alpha`,
      5. inflate Var(S) by
             n/n* = 1 + 2/(n(n-1)(n-2)) * SUM (n-i)(n-i-1)(n-i-2) * rho_i

    Returns (z_corrected, p_corrected).
    """
    x = _as2d(x)
    T, N = x.shape
    finite = np.isfinite(x)
    n = finite.sum(axis=0).astype("float64")

    # 1-2: detrend with Sen slope, then rank
    slope = sens_slope(x, min_obs=3)
    t = np.arange(T, dtype="float64")[:, None]
    det = x - slope[None, :] * t
    ranks = np.full_like(det, np.nan)
    for j in range(N):
        col = det[:, j]
        good = np.isfinite(col)
        if good.sum() >= 3:
            order = np.argsort(np.argsort(col[good]))
            ranks[good, j] = order + 1.0

    rm = ranks - np.nanmean(ranks, axis=0)
    denom = np.nansum(rm ** 2, axis=0)
    max_lag = max(1, int(np.floor(T * max_lag_fraction)))
    zcrit = norm.ppf(1 - alpha / 2.0)

    factor = np.zeros(N)
    with np.errstate(invalid="ignore", divide="ignore"):
        for lag in range(1, max_lag + 1):
            num = np.nansum(rm[lag:] * rm[:-lag], axis=0)
            rho = np.where(denom > 0, num / denom, 0.0)
            # significance bound for autocorrelation at this lag
            m = n - lag
            bound = np.where(m > 2, zcrit / np.sqrt(np.maximum(m, 1)), np.inf)
            sig = np.abs(rho) > bound
            w = (n - lag) * (n - lag - 1) * (n - lag - 2)
            factor += np.where(sig & (w > 0), w * rho, 0.0)

        nns = 1.0 + np.where(n > 2,
                             2.0 / (n * (n - 1) * (n - 2)) * factor, 0.0)
        # An effective sample size must stay physically sensible.
        nns = np.clip(nns, 0.05, 20.0)
        var_c = var * nns
        var_c = np.where(var_c > 0, var_c, np.nan)
        z = np.where(S > 0, (S - 1) / np.sqrt(var_c),
                     np.where(S < 0, (S + 1) / np.sqrt(var_c), 0.0))
        p = 2 * (1 - norm.cdf(np.abs(z)))

    bad = (n < min_obs) | ~np.isfinite(var)
    return np.where(bad, np.nan, z), np.where(bad, np.nan, p)


def sens_slope(x, chunk: int = 50000, min_obs: int = 10):
    """NaN-aware vectorized Theil-Sen slope (median of pairwise slopes)."""
    x = _as2d(x)
    T, N = x.shape
    n = np.isfinite(x).sum(axis=0)
    pairs = [(i, j) for i in range(T - 1) for j in range(i + 1, T)]
    out = np.empty(N, dtype="float64")
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        sl = np.empty((len(pairs), e - s), dtype="float64")
        for k, (i, j) in enumerate(pairs):
            sl[k] = (x[j, s:e] - x[i, s:e]) / float(j - i)
        with np.errstate(invalid="ignore"), \
                warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            allnan = np.all(~np.isfinite(sl), axis=0)
            med = np.nanmedian(np.where(allnan, 0.0, sl), axis=0)
            out[s:e] = np.where(allnan, np.nan, med)
    return np.where(n < min_obs, np.nan, out)


# ---------------------------------------------------------------------------
# RESTREND
# ---------------------------------------------------------------------------
class RestrendResult(dict):
    __getattr__ = dict.__getitem__


def restrend(ndvi, rain, min_r2: float = 0.10, alpha: float = 0.05,
             require_positive_beta: bool = True, min_obs: int = 10,
             method: str = "joint"):
    """Residual Trend Analysis (RESTREND).

    Two estimation paths:

    method="joint" (default, recommended)
        Fit NDVI ~ b*rain + c*time jointly. The climate-adjusted trend is c;
        the rainfall relationship is b, assessed CONTROLLING FOR TIME.

    method="two_stage" (classic Evans & Geerken formulation)
        Regress NDVI on rainfall, then trend the residuals.

    Why the default changed in M1: in the classic two-stage form the
    NDVI~rain r2 is DILUTED whenever NDVI carries a strong trend, because
    the trend inflates the total variance the rainfall term is judged
    against. A pixel with a perfectly real rainfall response plus a real
    decline can therefore be rejected as "no rainfall relationship". The
    joint model removes that artefact; both estimate the same quantity when
    no trend is present.

    INTERPRETATION LIMITS (enforced through `valid`):
      * A significant negative adjusted trend means vegetation declined
        RELATIVE TO WHAT RAINFALL PREDICTS. It does NOT prove anthropogenic
        causation. Unmodelled climate variables (temperature, radiation,
        rainfall timing, antecedent soil moisture), fire, pests, disease and
        land-cover conversion produce the same signature. Attribution
        requires independent evidence.
      * Where the rainfall relationship is weak (partial r2 < min_r2), not
        significant, or has the wrong sign for a water-limited system, the
        adjusted trend is NOT climate-corrected in any meaningful sense.
        Those pixels are returned with valid=False and must be reported as
        uncorrected trends.

    Returns RestrendResult with keys:
        slope, p, r2, beta, beta_p, valid, n_obs, method
    """
    ndvi, rain = _as2d(ndvi), _as2d(rain)
    if ndvi.shape != rain.shape:
        raise ValueError(f"shape mismatch: ndvi {ndvi.shape} vs "
                         f"rain {rain.shape}")
    if method not in ("joint", "two_stage"):
        raise ValueError("method must be 'joint' or 'two_stage'")
    T, N = ndvi.shape

    both = np.isfinite(ndvi) & np.isfinite(rain)
    n = both.sum(axis=0).astype("float64")
    nd = np.where(both, ndvi, np.nan)
    rn = np.where(both, rain, np.nan)
    t = np.arange(T, dtype="float64")[:, None]
    tt = np.where(both, t, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        nm = nd - np.nanmean(nd, axis=0)
        rm = rn - np.nanmean(rn, axis=0)
        tm = tt - np.nanmean(tt, axis=0)

        srr = np.nansum(rm ** 2, axis=0)
        stt = np.nansum(tm ** 2, axis=0)
        srt = np.nansum(rm * tm, axis=0)
        srn = np.nansum(rm * nm, axis=0)
        stn = np.nansum(tm * nm, axis=0)
        snn = np.nansum(nm ** 2, axis=0)

        if method == "joint":
            # 2x2 normal equations for [beta_rain, slope_time]
            det = srr * stt - srt ** 2
            ok = det > 0
            beta = np.where(ok, (stt * srn - srt * stn) / det, np.nan)
            slope = np.where(ok, (srr * stn - srt * srn) / det, np.nan)
            resid = nm - beta[None, :] * rm - slope[None, :] * tm
            sse = np.nansum(resid ** 2, axis=0)
            df = np.maximum(n - 3, 1)
            sigma2 = sse / df
            # standard errors from the inverse normal matrix
            se_beta = np.where(ok, np.sqrt(sigma2 * stt / det), np.nan)
            se_slope = np.where(ok, np.sqrt(sigma2 * srr / det), np.nan)
            # partial r2 of rainfall, controlling for time
            sse_time_only = snn - np.where(stt > 0, stn ** 2 / stt, 0.0)
            r2 = np.where(sse_time_only > 0,
                          1.0 - sse / sse_time_only, np.nan)
        else:
            beta = np.where(srr > 0, srn / srr, np.nan)
            res1 = nm - beta[None, :] * rm
            sse1 = np.nansum(res1 ** 2, axis=0)
            r2 = np.where(snn > 0, 1.0 - sse1 / snn, np.nan)
            se_beta = np.where((n > 2) & (srr > 0),
                               np.sqrt(sse1 / (n - 2) / srr), np.nan)
            slope = np.where(stt > 0,
                             np.nansum(tm * res1, axis=0) / stt, np.nan)
            res2 = res1 - slope[None, :] * tm
            se_slope = np.where((n > 2) & (stt > 0),
                                np.sqrt(np.nansum(res2 ** 2, axis=0)
                                        / (n - 2) / stt), np.nan)
            df = np.maximum(n - 2, 1)

        beta_p = 2 * (1 - tdist.cdf(np.abs(beta / se_beta), df=df))
        p = 2 * (1 - tdist.cdf(np.abs(slope / se_slope), df=df))

    valid = (n >= min_obs) & np.isfinite(r2) & (r2 >= min_r2) \
        & np.isfinite(beta_p) & (beta_p < alpha)
    if require_positive_beta:
        valid &= (beta > 0)

    bad = n < min_obs
    return RestrendResult(
        slope=np.where(bad, np.nan, slope), p=np.where(bad, np.nan, p),
        r2=np.where(bad, np.nan, r2), beta=np.where(bad, np.nan, beta),
        beta_p=np.where(bad, np.nan, beta_p),
        valid=valid & ~bad, n_obs=n.astype(int), method=method)


# ---------------------------------------------------------------------------
# Breakpoint
# ---------------------------------------------------------------------------
class BreakResult(dict):
    __getattr__ = dict.__getitem__


def best_breakpoint(x, min_segment: int = 4, min_gain: float = 0.05,
                    alpha: float = 0.05, apply_test: bool = True):
    """Single structural break by exhaustive two-segment least squares.

    A local stand-in for LandTrendr, used so the pipeline is testable
    offline; the real-data workflow uses GEE's native LandTrendr.

    Guarantees:
      * each segment keeps at least `min_segment` observations,
      * pixels where no admissible break exists return index -1 (not 0),
      * a Chow F-test judges whether the two-segment model is justified,
      * the break position is SELECTED by maximising fit, so the naive Chow
        p-value is optimistic. We report a Bonferroni-adjusted p over the
        number of candidate positions, which is conservative but honest.
        (The exact remedy is a sup-F / Quandt likelihood-ratio test with
        Andrews (1993) critical values; M2 did not implement it, so it
        remains outstanding and is carried to M3.)
      * series that are already an almost perfect straight line are not
        given a break: the residual sum of squares is compared against the
        total variance, so numerically-zero residuals cannot manufacture an
        arbitrarily large variance-explained ratio.

    Returns BreakResult with keys:
        index, delta_slope, gain, f_stat, p, p_adjusted, significant
    """
    x = _as2d(x)
    T, N = x.shape
    if min_segment < 3:
        raise ValueError("min_segment must be >= 3 for a stable segment fit")
    lo, hi = min_segment, T - min_segment
    idx = np.full(N, -1, dtype="int32")
    dslope = np.full(N, np.nan)
    gain = np.full(N, np.nan)
    fstat = np.full(N, np.nan)
    pval = np.full(N, np.nan)

    if lo >= hi:                      # series too short for any split
        return BreakResult(index=idx, delta_slope=dslope, gain=gain,
                           f_stat=fstat, p=pval, p_adjusted=pval.copy(),
                           significant=np.zeros(N, dtype=bool),
                           n_candidates=0)

    t = np.arange(T, dtype="float64")

    def seg_fit(ts, xs):
        """SSE and slope of an OLS line, NaN-aware."""
        good = np.isfinite(xs)
        tt = np.where(good, ts[:, None], np.nan)
        tm = tt - np.nanmean(tt, axis=0)
        xm = xs - np.nanmean(xs, axis=0)
        stt = np.nansum(tm ** 2, axis=0)
        b = np.where(stt > 0, np.nansum(tm * xm, axis=0) / stt, 0.0)
        r = xm - b[None, :] * tm
        return np.nansum(r ** 2, axis=0), b, good.sum(axis=0)

    sse1, _, n_tot = seg_fit(t, x)
    best = np.full(N, np.inf)
    for k in range(lo, hi):
        sa, ba, na = seg_fit(t[:k], x[:k])
        sb, bb, nb = seg_fit(t[k:], x[k:])
        ok = (na >= min_segment) & (nb >= min_segment)
        tot = np.where(ok, sa + sb, np.inf)
        better = tot < best
        best = np.where(better, tot, best)
        idx = np.where(better, k, idx)
        dslope = np.where(better, bb - ba, dslope)

    found = idx >= 0
    n_candidates = max(hi - lo, 1)
    # Scale guard: a single-line fit whose residuals are negligible relative
    # to the series variance leaves no structure for a break to explain.
    with np.errstate(invalid="ignore", divide="ignore"):
        var_scale = np.nansum((x - np.nanmean(x, axis=0)) ** 2, axis=0)
        degenerate = ~(sse1 > 1e-10 * np.maximum(var_scale, 1e-30))

        gain = np.where(found & ~degenerate & (sse1 > 0),
                        1.0 - best / sse1, np.nan)
        df2 = n_tot - 4
        fstat = np.where(found & ~degenerate & (best > 0) & (df2 > 0),
                         ((sse1 - best) / 2.0) / (best / np.maximum(df2, 1)),
                         np.nan)
        pval = np.where(np.isfinite(fstat),
                        1 - fdist.cdf(fstat, 2, np.maximum(df2, 1)), np.nan)
        p_adj = np.minimum(pval * n_candidates, 1.0)

    significant = found & ~degenerate & np.isfinite(gain) & (gain >= min_gain)
    if apply_test:
        significant &= np.isfinite(p_adj) & (p_adj < alpha)

    return BreakResult(index=idx, delta_slope=dslope, gain=gain,
                       f_stat=fstat, p=pval, p_adjusted=p_adj,
                       significant=significant, n_candidates=n_candidates)


# ---------------------------------------------------------------------------
# Cyclicity
# ---------------------------------------------------------------------------
class CyclicityResult(dict):
    __getattr__ = dict.__getitem__


def cyclicity(x, min_period: float = 4.0, max_period: float = 12.0,
              detrend: bool = True, threshold: float = 2.0,
              min_obs: int = 12, min_total_power: float = 1e-12):
    """Spectral periodicity of an NDVI series within a period band.

    SCIENTIFIC FRAMING (do not overstate in the report): this measures
    PERIODICITY. A high score means the series oscillates with a period in
    [min_period, max_period]. That is CONSISTENT WITH cyclical land use
    such as rotational or shifting cultivation, but it is NOT proof of it:
    multi-year climate oscillations, plantation harvest rotations and
    fire-regrowth cycles produce the same spectral signature. Attribution
    requires independent ground or ancillary evidence.

    Scoring (corrected in M1)
    -------------------------
    The raw band-power fraction CANNOT be thresholded directly. Under white
    noise the expected fraction equals the band's share of the spectrum
    (7/18 = 0.389 for a 4-12 step band over 36 steps), so any fixed
    threshold near that value flags noise as periodic. We therefore report
    ENRICHMENT = observed fraction / expected fraction, which is 1.0 under
    white noise regardless of band width or record length, and threshold on
    that instead.

    Returns CyclicityResult with keys:
        score             band power / total non-DC power, in [0, 1]
        expected_fraction band share of the spectrum under a flat null
        enrichment        score / expected_fraction (1.0 = no periodicity)
        dominant_period   period (time steps) of the strongest band peak
        band_power        absolute spectral power inside the band
        total_power       absolute non-DC spectral power
        periodic          boolean, enrichment >= threshold
    """
    x = _as2d(x)
    T, N = x.shape
    if min_period <= 1:
        raise ValueError("min_period must exceed 1 (Nyquist limit)")
    if max_period <= min_period:
        raise ValueError("max_period must exceed min_period")

    nan_out = np.full(N, np.nan)
    # need at least one full cycle of the longest period requested
    if T < max_period or T < min_obs:
        return CyclicityResult(score=nan_out.copy(),
                               expected_fraction=float("nan"),
                               enrichment=nan_out.copy(),
                               dominant_period=nan_out.copy(),
                               band_power=nan_out.copy(),
                               total_power=nan_out.copy(),
                               periodic=np.zeros(N, dtype=bool),
                               n_missing=int((~np.isfinite(x)).sum()),
                               mean_imputed_for_spectrum=False)

    valid_n = np.isfinite(x).sum(axis=0)
    # FFT cannot accept NaN; mean-impute ONLY for the spectral estimate and
    # mark short-record pixels invalid afterwards.
    col_mean = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    xf = np.where(np.isfinite(x), x, col_mean[None, :])

    if detrend:
        t = np.arange(T, dtype="float64")
        tm = t - t.mean()
        xm = xf - xf.mean(axis=0)
        b = (tm[:, None] * xm).sum(axis=0) / (tm ** 2).sum()
        sig = xm - b[None, :] * tm[:, None]
    else:
        sig = xf - xf.mean(axis=0)

    power = np.abs(np.fft.rfft(sig, axis=0)) ** 2
    freqs = np.fft.rfftfreq(T, d=1.0)
    band = (freqs >= 1.0 / max_period) & (freqs <= 1.0 / min_period)

    if not band.any():
        return CyclicityResult(score=nan_out.copy(),
                               expected_fraction=float("nan"),
                               enrichment=nan_out.copy(),
                               dominant_period=nan_out.copy(),
                               band_power=nan_out.copy(),
                               total_power=nan_out.copy(),
                               periodic=np.zeros(N, dtype=bool),
                               n_missing=int((~np.isfinite(x)).sum()),
                               mean_imputed_for_spectrum=False)

    band_power = power[band].sum(axis=0)
    total_power = power[1:].sum(axis=0)
    n_band = int(band.sum())
    n_total = int(power.shape[0] - 1)
    expected_fraction = n_band / n_total if n_total > 0 else float("nan")

    with np.errstate(invalid="ignore", divide="ignore"):
        degenerate = total_power <= min_total_power
        score = np.where(degenerate, np.nan, band_power / total_power)
        enrichment = score / expected_fraction if expected_fraction > 0 \
            else np.full(N, np.nan)

    band_idx = np.flatnonzero(band)
    peak = band_idx[np.argmax(power[band], axis=0)]
    dominant = np.where(freqs[peak] > 0, 1.0 / freqs[peak], np.nan)
    n_missing = int((~np.isfinite(x)).sum())

    bad = (valid_n < min_obs) | degenerate
    score = np.where(bad, np.nan, score)
    enrichment = np.where(bad, np.nan, enrichment)
    dominant = np.where(bad, np.nan, dominant)
    periodic = np.isfinite(enrichment) & (enrichment >= threshold)
    return CyclicityResult(score=score, expected_fraction=expected_fraction,
                           enrichment=enrichment, dominant_period=dominant,
                           band_power=np.where(bad, np.nan, band_power),
                           total_power=np.where(bad, np.nan, total_power),
                           periodic=periodic, n_missing=n_missing,
                           mean_imputed_for_spectrum=n_missing > 0)


def _band_fraction(sig: np.ndarray, band: np.ndarray) -> np.ndarray:
    """Share of non-DC power inside the period band; the test statistic."""
    power = np.abs(np.fft.rfft(sig, axis=0)) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        total = power[1:].sum(axis=0)
        return np.where(total > 0, power[band].sum(axis=0) / total, np.nan)


def _detrend_columns(sig: np.ndarray) -> np.ndarray:
    T = sig.shape[0]
    t = np.arange(T, dtype="float64")
    tm = t - t.mean()
    centred = sig - sig.mean(axis=0)
    slope = (tm[:, None] * centred).sum(axis=0) / (tm ** 2).sum()
    return centred - slope[None, :] * tm[:, None]


def cyclicity_significance(x, min_period: float = 4.0, max_period: float = 12.0,
                           detrend: bool = True, min_obs: int = 12,
                           n_surrogates: int = 199, alpha: float = 0.05,
                           seed: int = 42, null: str = "ar1",
                           chunk: int = 500):
    """Surrogate-data significance test for spectral band concentration.

    WHY THIS EXISTS
    ---------------
    `cyclicity` reports ENRICHMENT against a flat-spectrum (white-noise)
    expectation. Vegetation series are not white: they are autocorrelated,
    and autocorrelation alone pushes power towards low frequencies. A fixed
    enrichment threshold therefore has no stated false-positive rate on real
    series. This test attaches an explicit null distribution to the
    statistic.

    THE NULL MODELS
    ---------------
    null="ar1" (default, red noise)
        A first-order autoregressive process is fitted per pixel - lag-1
        autocorrelation and residual variance estimated from the (optionally
        detrended) series - and `n_surrogates` realisations are simulated.
        Surrogates share the pixel's persistence but contain no cycle, so a
        small p-value means the band concentration is more than this pixel's
        own autocorrelation can explain. This is the standard red-noise
        significance test used for spectral peaks (Mann & Lees 1996).

    null="shuffle" (white noise)
        The observed values are randomly permuted, destroying all temporal
        order. This tests against the same flat-spectrum assumption the
        enrichment statistic already makes, and is provided so the two nulls
        can be compared.

    WHY NOT PHASE RANDOMISATION
    ---------------------------
    Phase-randomised (Theiler) surrogates are the usual choice for testing
    nonlinearity, but they preserve the amplitude spectrum EXACTLY. Any
    statistic computed from the power spectrum - including this band
    fraction - is therefore identical for every surrogate, and the test
    returns p = 1 by construction. It is the wrong tool for this statistic
    and is deliberately not offered.

    The empirical p-value is the conservative rank estimator

        p = (1 + #{surrogate statistic >= observed}) / (1 + n_surrogates)

    which is never zero. With the default 199 surrogates the resolution is
    0.005.

    WHAT A SMALL p MEANS - AND DOES NOT MEAN
    ----------------------------------------
    It means the concentration of power inside the period band is unlikely
    under the stated null for that pixel. It says NOTHING about the cause.
    Rotational cultivation, plantation harvest cycles, fire-regrowth cycles
    and multi-year climate oscillation all produce periodicity. Periodicity
    is not attribution: linking it to any land-use practice requires
    independent ground evidence.

    Determinism: surrogates come from a seeded Generator, so repeated calls
    with the same seed return identical p-values.

    Returns a dict with keys: p_value, significant, observed_fraction,
    surrogate_mean_fraction, ar1_coefficient, n_surrogates, null.
    """
    x = _as2d(x)
    T, N = x.shape
    if n_surrogates < 1:
        raise ValueError("n_surrogates must be >= 1")
    if null not in ("ar1", "shuffle"):
        raise ValueError("null must be 'ar1' or 'shuffle'")

    nan_out = np.full(N, np.nan)
    empty = {"p_value": nan_out.copy(), "significant": np.zeros(N, bool),
             "observed_fraction": nan_out.copy(),
             "surrogate_mean_fraction": nan_out.copy(),
             "ar1_coefficient": nan_out.copy(),
             "n_surrogates": int(n_surrogates), "null": null}
    valid_n = np.isfinite(x).sum(axis=0)
    if T < max_period or T < min_obs:
        return empty

    # Prepared exactly as in `cyclicity`, so the two agree by construction.
    col_mean = np.nanmean(np.where(np.isfinite(x), x, np.nan), axis=0)
    col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
    xf = np.where(np.isfinite(x), x, col_mean[None, :])
    sig = _detrend_columns(xf) if detrend else xf - xf.mean(axis=0)

    freqs = np.fft.rfftfreq(T, d=1.0)
    band = (freqs >= 1.0 / max_period) & (freqs <= 1.0 / min_period)
    if not band.any():
        return empty
    observed = _band_fraction(sig, band)

    # AR(1) parameters from the prepared series.
    with np.errstate(invalid="ignore", divide="ignore"):
        denominator = (sig ** 2).sum(axis=0)
        phi = np.where(denominator > 0,
                       (sig[1:] * sig[:-1]).sum(axis=0) / denominator, 0.0)
        phi = np.clip(np.where(np.isfinite(phi), phi, 0.0), -0.99, 0.99)
        variance = sig.var(axis=0)
        innovation = np.sqrt(np.maximum(variance * (1.0 - phi ** 2), 1e-30))

    rng = np.random.default_rng(seed)
    exceed = np.zeros(N, dtype="int64")
    surrogate_mean = np.zeros(N)

    for start in range(0, N, chunk):
        stop = min(start + chunk, N)
        width = stop - start
        for _ in range(n_surrogates):
            if null == "ar1":
                noise = rng.standard_normal((T, width))
                surrogate = np.empty((T, width))
                sd = np.sqrt(np.maximum(variance[start:stop], 1e-30))
                surrogate[0] = noise[0] * sd
                for step in range(1, T):
                    surrogate[step] = (phi[start:stop] * surrogate[step - 1]
                                       + noise[step] * innovation[start:stop])
            else:
                order = np.argsort(rng.random((T, width)), axis=0)
                surrogate = np.take_along_axis(sig[:, start:stop], order,
                                               axis=0)
            prepared = _detrend_columns(surrogate) if detrend \
                else surrogate - surrogate.mean(axis=0)
            fraction = _band_fraction(prepared, band)
            exceed[start:stop] += (fraction >= observed[start:stop])
            surrogate_mean[start:stop] += np.nan_to_num(fraction)

    p_value = (1.0 + exceed) / (1.0 + n_surrogates)
    bad = (valid_n < min_obs) | ~np.isfinite(observed)
    p_value = np.where(bad, np.nan, p_value)
    return {"p_value": p_value,
            "significant": np.isfinite(p_value) & (p_value < alpha),
            "observed_fraction": np.where(bad, np.nan, observed),
            "surrogate_mean_fraction": np.where(bad, np.nan,
                                                surrogate_mean / n_surrogates),
            "ar1_coefficient": np.where(bad, np.nan, phi),
            "n_surrogates": int(n_surrogates), "null": null}
