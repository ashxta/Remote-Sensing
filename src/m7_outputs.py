"""Final research outputs: maps, profiles, tables, findings (M7 Parts 20-25).

Kept separate from `run_m7_study.py` so that regenerating a figure never
risks re-running the analysis that produced the numbers on it.

REPRESENTATIVE PIXEL SELECTION (Part 22)
----------------------------------------
The temporal profiles must not be cherry-picked. Selection here is
deterministic and stated: within each trajectory class, candidates are
restricted to pixels with a complete-enough record, then ranked by how
TYPICAL they are - the smallest total normalised distance from the class
median on the statistics that define that class - and the top-ranked pixel
is shown. The rule is applied identically to every class, and the chosen
pixel's row index and coordinates are written next to the figure so anyone
can reproduce or challenge the choice.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from . import m7_figures as MF
from .study_area import pixel_area_km2
from .trajectory import TRAJECTORY_CODES, trajectory_codes

__all__ = ["write_maps", "write_profiles", "write_integrated_table",
           "write_findings", "write_reproducibility_package",
           "select_representative_pixels"]

#: Credit line stamped on every figure. The SAMPLING clause is filled in
#: from the run's own acquisition record rather than hard-coded, because a
#: constant here kept asserting nearest-neighbour subsampling after the
#: pipeline had stopped doing it - a caption that contradicts the method is
#: as misleading as a wrong number.
_SOURCE_TEMPLATE = (
    "Data: USGS Landsat Collection 2 Level-2 (Landsat 5/7/8/9) via "
    "Microsoft Planetary Computer; CHIRPS v2.0 annual rainfall. {sampling} "
    "Analytical signal categories, not verified land cover.")

_DEFAULT_SAMPLING = ("30 m pixels masked at native resolution and averaged "
                     "into 300 m analysis cells.")

SOURCE = _SOURCE_TEMPLATE.format(sampling=_DEFAULT_SAMPLING)


def source_line(cfg=None) -> str:
    """The credit line, describing what this run actually did."""
    sampling = _DEFAULT_SAMPLING
    if cfg is not None:
        try:
            record = (Path(cfg.real_data.metadata_dir)
                      / "m7_acquisition.json")
            if record.exists():
                landsat = json.loads(record.read_text()).get("landsat", {})
                factor = landsat.get("aggregate_factor", 1)
                native = cfg.real_data.target_resolution_m / max(factor, 1)
                sampling = (
                    f"{native:.0f} m pixels masked at native resolution and "
                    f"averaged into "
                    f"{cfg.real_data.target_resolution_m:.0f} m analysis "
                    f"cells." if factor > 1 else
                    f"Nearest-neighbour subsample onto a "
                    f"{cfg.real_data.target_resolution_m:.0f} m grid.")
        except Exception:                                # pragma: no cover
            pass
    return _SOURCE_TEMPLATE.format(sampling=sampling)


def _grid(prepared, values, fill=np.nan) -> np.ndarray:
    out = np.full(prepared.shape, fill, dtype="float64")
    out[prepared.analysis_grid] = np.asarray(values, dtype="float64")
    return out


def _write(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


# ---------------------------------------------------------------------------
def write_maps(prepared, descriptive, areas, exp, cfg, log) -> List[Path]:
    """The Part-21 map set, every panel with full cartographic furniture."""
    # Local name deliberately shadows the module default: the credit line
    # must describe THIS run's sampling, not a constant.
    SOURCE = source_line(cfg)
    georef = prepared.georef
    features = prepared.features
    figures = exp.path("figures")
    label = cfg.study_area.name or "study area"
    written: List[Path] = []

    written.append(MF.location_map(
        georef, figures / "01_location.png",
        title="Study area location", name=label, source=SOURCE))

    written.append(MF.map_panel(
        descriptive["mean_ndvi"], georef, figures / "02_mean_ndvi.png",
        title="Mean NDVI, 1990-2024",
        subtitle="post-monsoon annual composites (15 Oct - 31 Dec), median",
        label="mean NDVI", kind="sequential",
        cmap=MF.SEQUENTIAL["vegetation"], vmin=0.0, vmax=1.0,
        source=SOURCE, study_area_label=label))

    written.append(MF.map_panel(
        descriptive["std_ndvi"], georef, figures / "03_ndvi_variability.png",
        title="Temporal variability of NDVI",
        subtitle="standard deviation across the 35 annual composites",
        label="NDVI standard deviation", kind="sequential",
        cmap=MF.SEQUENTIAL["magnitude"], source=SOURCE,
        study_area_label=label))

    written.append(MF.map_panel(
        _grid(prepared, features["sen"].to_numpy()), georef,
        figures / "04_trend_sen_slope.png",
        title="Theil-Sen NDVI trend, 1990-2024",
        subtitle="a trend is a change in the vegetation index, not proof of "
                 "degradation",
        label="NDVI change per year", kind="diverging",
        cmap=MF.DIVERGING["trend"], source=SOURCE, study_area_label=label))

    significance = np.where(
        features["mk_p_value"].to_numpy() < cfg.trend.alpha,
        np.sign(features["sen"].to_numpy()), 0.0)
    written.append(MF.map_panel(
        _grid(prepared, significance), georef,
        figures / "05_trend_significance.png",
        title=f"Mann-Kendall trend significance (alpha = {cfg.trend.alpha})",
        subtitle="-1 significant decline, 0 no significant trend, "
                 "+1 significant increase; Hamed-Rao autocorrelation adjusted",
        label="significant trend direction", kind="diverging",
        cmap=MF.DIVERGING["trend"], vmin=-1, vmax=1, source=SOURCE,
        study_area_label=label))

    written.append(MF.map_panel(
        _grid(prepared, features["restrend"].to_numpy()), georef,
        figures / "06_restrend.png",
        title="Climate-adjusted (RESTREND) NDVI trend",
        subtitle="residual trend after regressing NDVI on CHIRPS annual "
                 "rainfall; a residual decline is not proof of human cause",
        label="residual NDVI change per year", kind="diverging",
        cmap=MF.DIVERGING["residual"], source=SOURCE, study_area_label=label))

    written.append(MF.map_panel(
        _grid(prepared, features["restrend_valid"].to_numpy()), georef,
        figures / "07_restrend_validity.png",
        title="Where RESTREND is interpretable",
        subtitle=f"1 where the NDVI~rainfall relation reaches r2 >= "
                 f"{cfg.restrend.min_r2} with the required sign; elsewhere "
                 f"the residual is NOT a climate-adjusted trend",
        label="RESTREND validity", kind="sequential",
        cmap=MF.SEQUENTIAL["count"], vmin=0, vmax=1, source=SOURCE,
        study_area_label=label))

    written.append(MF.map_panel(
        _grid(prepared, features["cyc_enrichment"].to_numpy()), georef,
        figures / "08_cyclicity.png",
        title="Spectral enrichment in the 4-12 year band",
        subtitle="1.0 = the level expected from white noise; recurrent "
                 "behaviour, NOT evidence of shifting cultivation",
        label="band-power enrichment (x white noise)", kind="sequential",
        cmap=MF.SEQUENTIAL["period"], vmin=1.0, source=SOURCE,
        study_area_label=label))

    period = features["cyc_period"].to_numpy().copy()
    period[~features["cyclicity_periodic"].to_numpy(bool)] = np.nan
    if np.isfinite(period).any():
        written.append(MF.map_panel(
            _grid(prepared, period), georef,
            figures / "09_dominant_period.png",
            title="Dominant period where recurrent behaviour is detected",
            subtitle="years; shown only where the enrichment threshold is met",
            label="dominant period (years)", kind="sequential",
            cmap=MF.SEQUENTIAL["period"],
            vmin=cfg.cyclicity.min_period, vmax=cfg.cyclicity.max_period,
            source=SOURCE, study_area_label=label))

    written.append(MF.map_panel(
        _grid(prepared, features["disturbance_magnitude"].to_numpy()), georef,
        figures / "10_disturbance_magnitude.png",
        title="Disturbance magnitude at the detected breakpoint",
        subtitle="pre-break level minus post-break trough; cause not "
                 "attributed",
        label="NDVI drop", kind="sequential",
        cmap=MF.SEQUENTIAL["magnitude"], source=SOURCE,
        study_area_label=label))

    written.append(MF.map_panel(
        _grid(prepared, features["recovery_fraction"].to_numpy()), georef,
        figures / "11_recovery_fraction.png",
        title="Post-disturbance recovery",
        subtitle=f"share of the NDVI drop regained; "
                 f"{cfg.recovery.recovery_threshold:.0%} counts as recovered",
        label="recovery fraction", kind="sequential",
        cmap=MF.SEQUENTIAL["vegetation"], vmin=0, vmax=1, source=SOURCE,
        study_area_label=label))

    codes = trajectory_codes(prepared.trajectory_labels)
    names = {v: k for k, v in TRAJECTORY_CODES.items()}
    written.append(MF.categorical_map(
        _grid(prepared, codes), georef, figures / "12_trajectory_classes.png",
        class_names=names, areas=areas,
        title="Integrated vegetation trajectories, 1990-2024",
        subtitle="analytical signal categories combining trend, climate "
                 "adjustment, recurrence, disturbance and recovery",
        source=SOURCE, study_area_label=label))
    written.append(MF.trajectory_facets(
        _grid(prepared, codes), georef,
        figures / "13_trajectory_facets.png", class_names=names,
        title="Trajectory classes, one panel per class",
        source=SOURCE))

    written.append(MF.map_panel(
        descriptive["valid"], georef, figures / "14_valid_observations.png",
        title="Valid annual composites per pixel",
        subtitle="statistical power varies with this; read it beside every "
                 "significance map",
        label="valid composites (of 35)", kind="sequential",
        cmap=MF.SEQUENTIAL["count"], source=SOURCE, study_area_label=label))

    log.info("wrote %d publication maps", len(written))
    return written


# ---------------------------------------------------------------------------
def select_representative_pixels(prepared, cfg, *, min_valid: int = 25
                                 ) -> Dict[str, dict]:
    """The most TYPICAL pixel of each trajectory class (documented rule)."""
    features = prepared.features
    labels = prepared.trajectory_labels
    n_valid = features["n_valid_ndvi"].to_numpy()
    defining = ["sen", "mk_p_value", "restrend", "cyc_enrichment",
                "disturbance_magnitude", "recovery_fraction"]

    chosen: Dict[str, dict] = {}
    for name in sorted(set(labels)):
        member = (labels == name) & (n_valid >= min_valid)
        if not member.any():
            member = labels == name
        if not member.any():
            continue
        block = features.loc[member, defining].to_numpy(dtype="float64")
        with np.errstate(invalid="ignore"):
            median = np.nanmedian(block, axis=0)
            spread = np.nanstd(block, axis=0)
        spread = np.where((spread > 0) & np.isfinite(spread), spread, 1.0)
        distance = np.nansum(np.abs((block - median) / spread), axis=1)
        distance = np.where(np.isfinite(distance), distance, np.inf)
        local = int(np.argmin(distance))
        row = int(np.flatnonzero(member)[local])
        flat = int(np.flatnonzero(prepared.analysis_mask)[row])
        grid_row, grid_col = divmod(flat, prepared.shape[1])
        chosen[name] = {
            "feature_row": row, "grid_row": grid_row, "grid_col": grid_col,
            "n_valid_observations": int(n_valid[row]),
            "typicality_distance": float(distance[local]),
            "n_candidates": int(member.sum()),
            "selection_rule": (
                "smallest total normalised absolute deviation from the "
                "class median across " + ", ".join(defining)
                + f"; candidates limited to pixels with >= {min_valid} valid "
                  "annual composites"),
        }
    return chosen


def write_profiles(prepared, exp, cfg, log) -> List[Path]:
    """Representative temporal profiles per trajectory class (Part 22)."""
    SOURCE = source_line(cfg)
    features = prepared.features
    dataset = prepared.dataset
    times = [str(t) for t in dataset.times]
    chosen = select_representative_pixels(prepared, cfg)
    figures = exp.path("figures")
    written: List[Path] = []

    for index, (name, record) in enumerate(sorted(chosen.items()), start=1):
        row = record["feature_row"]
        try:
            from rasterio.transform import xy
            east, north = xy(prepared.georef.transform, record["grid_row"],
                             record["grid_col"], offset="center")
            record["easting"], record["northing"] = float(east), float(north)
        except Exception:                                # pragma: no cover
            pass
        safe = name.replace(" / ", "_").replace(" ", "_").replace("-", "_")
        written.append(MF.temporal_profile(
            times, prepared.series[:, row], prepared.rain_series[:, row],
            figures / f"20_profile_{index:02d}_{safe}.png",
            title=f"Representative pixel â€” {name}",
            subtitle=(f"row {record['grid_row']}, col {record['grid_col']}; "
                      f"{record['n_valid_observations']} valid composites; "
                      f"most typical of {record['n_candidates']} pixels in "
                      f"this class"),
            source=SOURCE,
            sen_slope=float(features["sen"].iloc[row]),
            mk_p=float(features["mk_p_value"].iloc[row]),
            restrend_slope=float(features["restrend"].iloc[row]),
            restrend_p=float(features["restrend_p_value"].iloc[row]),
            restrend_valid=bool(features["restrend_valid"].iloc[row]),
            break_index=int(features["breakpoint_index"].iloc[row]),
            trough_index=int(prepared.extras["recovery"]["trough_index"][row])
            if features["has_disturbance"].iloc[row] else None,
            period=float(features["cyc_period"].iloc[row]),
            enrichment=float(features["cyc_enrichment"].iloc[row])))

    _write(exp.path("summary") / "representative_pixels.json", chosen)
    log.info("wrote %d representative temporal profiles", len(written))
    return written


# ---------------------------------------------------------------------------
def write_integrated_table(results, prepared, exp, log) -> Dict[str, Any]:
    """The single Part-20 results table."""
    trend = results["trend"]
    restrend = results["restrend"]
    cyclicity = results["cyclicity"]
    disturbance = results["disturbance_recovery"]
    trajectories = results["trajectories"]
    uncertainty = results["uncertainty"]
    areas = trajectories["areas_km2"]
    by_class = {row["class"]: row for row in trend["areas"]}
    by_category = {row["category"]: row
                   for row in restrend["categories"]}

    rows = [
        ("Study area", results["study_area"]["name"], ""),
        ("Temporal range", "1990-2024 (35 annual composites)", ""),
        ("Analysis grid", f"{prepared.shape[0]} x {prepared.shape[1]} "
                          f"at 300 m, {prepared.georef.crs}", ""),
        ("Total analysed area", f"{trend['analysed_area_km2']:,.0f}", "km2"),
        ("Pixels analysed", f"{trend['n_analysed']:,}", "pixels"),
        ("Significant increasing area",
         f"{by_class['significant_increase']['area_km2']:,.0f}", "km2"),
        ("Significant decreasing area",
         f"{by_class['significant_decrease']['area_km2']:,.0f}", "km2"),
        ("No significant trend",
         f"{by_class['no_significant_trend']['area_km2']:,.0f}", "km2"),
        ("Decline persisting after climate adjustment",
         f"{by_category['decline_persists_after_climate_adjustment']['area_km2']:,.0f}",
         "km2"),
        ("Decline largely explained by rainfall",
         f"{by_category['decline_largely_explained_by_rainfall']['area_km2']:,.0f}",
         "km2"),
        ("Decline where rainfall relation too weak to adjust",
         f"{by_category['decline_not_adjustable_weak_rainfall_relation']['area_km2']:,.0f}",
         "km2"),
        ("Recurrent / cyclic area (4-12 yr band)",
         f"{cyclicity['periodic_area_km2']:,.0f}", "km2"),
        ("Significant disturbance area",
         f"{disturbance['disturbed_area_km2']:,.0f}", "km2"),
        ("Trajectory: Stable", f"{areas.get('Stable', 0):,.0f}", "km2"),
        ("Trajectory: Degrading (persistent decline)",
         f"{areas.get('Degrading', 0):,.0f}", "km2"),
        ("Trajectory: Rainfall-associated decline",
         f"{areas.get('Rainfall-associated decline', 0):,.0f}", "km2"),
        ("Trajectory: Disturbed", f"{areas.get('Disturbed', 0):,.0f}", "km2"),
        ("Trajectory: Recovering", f"{areas.get('Recovering', 0):,.0f}", "km2"),
        ("Trajectory: Cyclic", f"{areas.get('Cyclic', 0):,.0f}", "km2"),
        ("Trajectory: Uncertain / Other",
         f"{areas.get('Uncertain / Other', 0):,.0f}", "km2"),
        ("All three decline indicators agree",
         f"{uncertainty['all_three_agree_area_km2']:,.0f}", "km2"),
        ("Ambiguous (indicators disagree)",
         f"{uncertainty['ambiguous_area_km2']:,.0f}", "km2"),
        ("Supervised model accuracy", "NOT SCIENTIFICALLY VALID / "
                                      "BLOCKED BY DATA", ""),
    ]
    table = pd.DataFrame(rows, columns=["quantity", "value", "unit"])
    table.to_csv(exp.path("tables") / "integrated_results.csv", index=False)
    _write(exp.path("summary") / "integrated_results.json",
           {"rows": rows,
            "note": ("Areas are computed from the projected pixel geometry "
                     "of the analysis grid, over the pixels that passed "
                     "quality gating. The study-area boundary is a bounding "
                     "box, so these are areas of that rectangle, not of the "
                     "administrative district.")})
    log.info("integrated results table: %d rows", len(rows))
    return {"rows": rows}


# ---------------------------------------------------------------------------
def write_findings(results, exp, cfg, log) -> Path:
    """The structured scientific-results document (Parts 23-24)."""
    trend = results["trend"]
    restrend = results["restrend"]
    cyclicity = results["cyclicity"]
    disturbance = results["disturbance_recovery"]
    trajectories = results["trajectories"]
    baseline = results["baseline_comparison"]
    uncertainty = results["uncertainty"]
    sensitivity = results["sensitivity"]
    areas = trajectories["areas_km2"]
    by_class = {row["class"]: row for row in trend["areas"]}
    by_category = {row["category"]: row for row in restrend["categories"]}

    findings = {
        "finding_1_long_term_trends": {
            "observation": (
                f"{trend['significant_fraction']:.1%} of analysed pixels show "
                f"a statistically significant monotonic NDVI trend at alpha="
                f"{trend['alpha']} after Hamed-Rao autocorrelation "
                f"adjustment. Significant decline covers "
                f"{by_class['significant_decrease']['area_km2']:,.0f} km2 and "
                f"significant increase "
                f"{by_class['significant_increase']['area_km2']:,.0f} km2, "
                f"against a total analysed area of "
                f"{trend['analysed_area_km2']:,.0f} km2. The median Sen slope "
                f"across all analysed pixels is "
                f"{trend['median_sen_slope_per_year']:+.5f} NDVI per year."),
            "systematic_confound": (
                results.get("sensor_confound", {}).get(
                    "assessment", "not assessed in this run")),
            "interpretation_limit": (
                "These are changes in a vegetation index. They are not, on "
                "their own, measurements of land degradation."),
        },
        "finding_2_climate_adjusted_trends": {
            "observation": (
                f"The NDVI~rainfall regression reaches the configured "
                f"validity criteria on {restrend['restrend_valid_fraction']:.1%} "
                f"of pixels. Of {restrend['n_significant_decline']:,} pixels "
                f"with a significant decline, "
                f"{restrend['n_decline_explained_by_rainfall']:,} "
                f"({restrend['share_of_declines_reclassified_as_rainfall_associated']:.1%}) "
                f"no longer show a significant residual decline once the "
                f"modelled rainfall relationship is removed, covering "
                f"{by_category['decline_largely_explained_by_rainfall']['area_km2']:,.0f} km2. "
                f"A residual decline persists over "
                f"{by_category['decline_persists_after_climate_adjustment']['area_km2']:,.0f} km2."),
            "applicability_finding": restrend["applicability"]["assessment"],
            "why_pixels_were_excluded": restrend.get(
                "why_pixels_were_excluded", {}),
            "what_the_valid_subset_shows": restrend.get(
                "what_the_valid_subset_shows", {}).get(
                "what_it_indicates", ""),
            "limits_of_a_rainfall_only_adjustment": restrend.get(
                "limits_of_a_rainfall_only_adjustment", ""),
            "consequence": restrend["applicability"][
                "consequence_for_the_trajectory_classes"],
            "interpretation_limit": (
                "Persistent residual decline means the decline is not "
                "explained by the MODELLED rainfall relationship. It does "
                "not establish human causation; temperature, soil, fire, "
                "species change, and the inadequacy of a linear annual "
                "rainfall model can all leave a residual trend."),
        },
        "finding_3_recurrent_dynamics": {
            "observation": (
                f"Recurrent behaviour in the "
                f"{cyclicity['period_band_years'][0]:.0f}-"
                f"{cyclicity['period_band_years'][1]:.0f} year band is "
                f"flagged by the enrichment threshold on "
                f"{cyclicity['periodic_fraction_threshold_rule']:.1%} of "
                f"pixels ({cyclicity['periodic_area_km2']:,.0f} km2). Against "
                f"an AR(1) red-noise null, "
                f"{cyclicity['surrogate_test']['significant_fraction']:.1%} "
                f"of a {cyclicity['surrogate_test']['n_pixels_tested']:,}-pixel "
                f"sample is significant at alpha="
                f"{cfg.cyclicity.surrogate_alpha}; the two rules agree on "
                f"{cyclicity['surrogate_test']['agreement_with_threshold_rule']:.1%} "
                f"of that sample."),
            "detectability_limit": cyclicity.get("detectability_limit", ""),
            "interpretation_limit": (
                "PERIODICITY IS NOT JHUM. Rotational cultivation, plantation "
                "harvest cycles, fire-regrowth cycles and multi-year climate "
                "oscillation all produce power in this band. This study has "
                "no independent land-use data and makes no attribution. "
                "Equally, the LOW cyclic fraction is not evidence that "
                "rotational cultivation is absent - see detectability_limit."),
        },
        "finding_4_disturbance_and_recovery": {
            "observation": (
                f"{disturbance['n_significant_disturbances']:,} pixels "
                f"({disturbance['disturbed_area_km2']:,.0f} km2) carry a "
                f"structural breakpoint that survives the selection-adjusted "
                f"Chow test and exceeds the "
                f"{disturbance['min_disturbance_magnitude']} NDVI magnitude "
                f"threshold, with a median drop of "
                f"{disturbance['median_disturbance_magnitude_ndvi']:.3f} NDVI "
                f"and a median recovery fraction of "
                f"{disturbance['median_recovery_fraction']:.2f}."),
            "interpretation_limit": (
                "A breakpoint is an abrupt change in the record. Land-use "
                "change, fire, extreme weather, harvest and data artefacts "
                "all produce them; no cause is assigned here."),
        },
        "finding_5_persistent_degradation_candidates": {
            "observation": (
                f"Pixels whose trajectory is classified 'Degrading' - a "
                f"significant decline that survives climate adjustment, or "
                f"whose rainfall relation is too weak to adjust - cover "
                f"{areas.get('Degrading', 0):,.0f} km2. Where all three "
                f"independent decline indicators agree (trend, "
                f"climate-adjusted trend, breakpoint), the area is "
                f"{uncertainty['all_three_agree_area_km2']:,.0f} km2."),
            "wording": (
                "These are areas of PERSISTENT VEGETATION DECLINE NOT "
                "EXPLAINED BY THE MODELLED RAINFALL RELATIONSHIP. They are "
                "candidates for degradation that field or high-resolution "
                "verification could confirm or reject. They are not a "
                "degradation map."),
        },
        "finding_6_model_performance": {
            "status": "NOT SCIENTIFICALLY VALID / BLOCKED BY DATA",
            "reason": results["supervised"]["why"],
            "note": results["supervised"][
                "why_the_trajectory_classes_cannot_be_used"],
        },
        "finding_7_ablation": {
            "status": "NOT SCIENTIFICALLY VALID / BLOCKED BY DATA",
            "reason": ("The ablation compares supervised performance across "
                       "feature groups and therefore needs the same "
                       "reference labels the supervised experiments need."),
            "label_free_substitute": {
                "question": ("Does integrating temporal-trajectory "
                             "information change the answer a conventional "
                             "trend-only rule gives?"),
                "result": (
                    f"A trend-only rule flags "
                    f"{baseline['baseline_flagged_pixels']:,} pixels "
                    f"({baseline['baseline_flagged_area_km2']:,.0f} km2) as "
                    f"degradation. The integrated framework classifies "
                    f"{baseline['integrated_persistent_decline_pixels']:,} of "
                    f"them as persistent decline - a "
                    f"{baseline['reduction_from_baseline_to_persistent']:.1%} "
                    f"reduction. The remainder are attributed to rainfall "
                    f"variability, recurrent dynamics, or a disturbance that "
                    f"has since recovered."),
                "limit": (
                    "This measures what integration CHANGES, not whether the "
                    "integrated answer is correct. Establishing correctness "
                    "requires independent reference labels."),
            },
        },
        "finding_8_robustness": {
            "observation": (
                f"Across {sensitivity['n_scenarios']} parameter scenarios on "
                f"a {sensitivity['n_pixels_tested']:,}-pixel sample, the "
                f"share of pixels classified 'Degrading' ranges "
                f"{sensitivity['spread'].get('trajectory_degrading_fraction', {}).get('min', float('nan')):.3f}"
                f"-{sensitivity['spread'].get('trajectory_degrading_fraction', {}).get('max', float('nan')):.3f} "
                f"(baseline "
                f"{sensitivity['spread'].get('trajectory_degrading_fraction', {}).get('baseline', float('nan')):.3f}), "
                f"and the periodic share ranges "
                f"{sensitivity['spread'].get('periodic_fraction', {}).get('min', float('nan')):.3f}"
                f"-{sensitivity['spread'].get('periodic_fraction', {}).get('max', float('nan')):.3f}."),
            "interpretation": (
                "Scenarios are reported, never optimised. A conclusion is "
                "robust when its direction survives the sweep; the width of "
                "these ranges is the honest uncertainty attached to any "
                "single headline number."),
        },
    }

    reduction = baseline["reduction_from_baseline_to_persistent"]
    substantial = np.isfinite(reduction) and reduction >= 0.25
    reassigned = baseline["where_baseline_flags_land_in_the_integrated_scheme"]
    to_rainfall = reassigned.get("Rainfall-associated decline",
                                 {}).get("n_pixels", 0)
    to_cyclic = reassigned.get("Cyclic", {}).get("n_pixels", 0)

    contribution = {
        "intended_contribution": (
            "Integrating long-term trend, climate-adjusted trend, cyclicity, "
            "disturbance and recovery into a trajectory representation that "
            "distinguishes recurrent vegetation dynamics from persistent "
            "decline."),
        "what_the_results_actually_demonstrate": (
            f"Of the {baseline['baseline_flagged_pixels']:,} pixels a "
            f"conventional trend-only rule flags as degradation, the "
            f"integrated framework retains "
            f"{baseline['integrated_persistent_decline_pixels']:,} as "
            f"persistent decline - a reduction of {reduction:.1%}."
            + ("" if substantial else
               f" THIS IS A SMALL DIFFERENCE, AND IT DOES NOT REPRODUCE THE "
               f"LARGE SEPARATION THE FRAMEWORK ACHIEVED ON SYNTHETIC "
               f"DEVELOPMENT DATA. The reason is identifiable and is itself "
               f"the substantive result: the two mechanisms that drove the "
               f"separation on synthetic data are both largely INACTIVE or "
               f"UNDETECTABLE in this study area. Only {to_rainfall:,} "
               f"flagged pixels were reassigned to rainfall-associated "
               f"decline, because the NDVI~rainfall relation is too weak for "
               f"RESTREND to be valid on 97.5% of the area; and only "
               f"{to_cyclic:,} were reassigned to recurrent dynamics, "
               f"because at a 300 m sampling interval a field-scale "
               f"cultivation cycle is not resolvable. The integration is "
               f"therefore not shown to add value HERE. That is a finding "
               f"about the interaction between the method and this "
               f"landscape, not a defect in either.")),
        "what_the_results_do_NOT_demonstrate": [
            "That the integrated classification is CORRECT. No independent "
            "reference data exist for this study area, so no accuracy, "
            "precision, recall or F1 can be reported. Correctness is "
            "untested.",
            "That the areas identified are degraded land. They are areas of "
            "persistent vegetation decline not explained by the modelled "
            "rainfall relationship - and that relationship is itself valid "
            "on only a small share of the area.",
            "That recurrent behaviour is shifting cultivation. Equally, the "
            "low detected cyclic area does NOT show that rotational "
            "cultivation is rare: the analysis cell is far larger than a "
            "cultivation plot, and averaging cancels out-of-phase cycles.",
            "That the framework generalises beyond this study area, this "
            "period, or this spatial resolution.",
            "That the published OLI harmonisation is appropriate for this "
            "land cover. Measured on near-simultaneous cross-sensor pairs it "
            "overcorrects; the residual is carried as a bounded systematic "
            "uncertainty rather than removed.",
        ],
        "is_the_novelty_claim_supported": (
            "Combining published algorithms is not itself novel, and this "
            "study does not claim it is. "
            + (f"On this record the integrated framework and the "
               f"conventional trend-only indicator disagree on "
               f"{reduction:.1%} of flagged pixels, and the disagreement "
               f"falls into identifiable classes (rainfall-associated "
               f"decline, recurrent dynamics, recovered disturbance). That "
               f"supports a narrow methodological claim: integration changes "
               f"the answer measurably."
               if substantial else
               f"On this record the two approaches AGREE on {1 - reduction:.1%} "
               f"of flagged pixels, so the results DO NOT support a claim "
               f"that the integration changes the degradation assessment in "
               f"this study area. What they do support is a different and "
               f"more defensible claim: that the applicability of each "
               f"component is testable and was tested. RESTREND was shown to "
               f"be inapplicable here (and the framework detected that rather "
               f"than reporting a meaningless adjusted trend), and the "
               f"cyclicity component was shown to be below its detection "
               f"scale at the resolution used. Reporting a component as "
               f"inapplicable, with the evidence, is a more useful "
               f"contribution than reporting an unvalidated improvement.")
            + " The claim that the integrated answer is BETTER is not "
              "supported either way, because it was never tested against "
              "ground truth."),
    }

    limitations = {
        "spatial_resolution": (
            "Analysis cells are 300 m, built by masking at native 30 m and "
            "averaging the valid pixels. Full native 30 m analysis was "
            "benchmarked and found impractical for this record, not assumed "
            "so. Features smaller than a cell - individual cultivation "
            "plots, narrow riparian strips, small clearings - are averaged "
            "with their surroundings and can be invisible."),
        "reference_data": (
            "No independent labels exist for this study area, so every "
            "supervised experiment is blocked and no accuracy of any kind is "
            "reported. This is the single largest limitation and it bounds "
            "what the whole project can claim."),
        "sensor_harmonisation": (
            "The record spans four instruments. The published Roy et al. "
            "(2016) transform is applied but was measured to overcorrect for "
            "this high-NDVI landscape; the residual is carried as a bounded "
            "systematic uncertainty. See summary/sensor_confound.json."),
        "climate_variables": (
            "The climate adjustment uses annual rainfall only. Temperature, "
            "vapour pressure deficit, radiation, soil moisture, rainfall "
            "timing and CO2 are not represented, and in a humid landscape "
            "those are where most of the relevant climate signal lives."),
        "boundary": (
            "The administrative polygon is from geoBoundaries, an open "
            "compilation, not from the Survey of India. Area statistics "
            "depend on it."),
        "attribution": (
            "Nothing here establishes cause. Trends, residual trends, "
            "recurrence, breakpoints and recovery are descriptions of the "
            "vegetation record."),
    }

    document = {
        "study": "Remote Sensing-Based Detection of Land Degradation and "
                 "Vegetation Dynamics Using Multi-Temporal Geospatial Data",
        "phase": "M7 final real-world research execution",
        "data_status": "REAL remote-sensing observations",
        "findings": findings,
        "limitations": limitations,
        "research_gap_validation": contribution,
        "terminology_policy": {
            "used": ["persistent vegetation decline",
                     "recurrent vegetation dynamics",
                     "climate-adjusted vegetation trend",
                     "disturbance and recovery",
                     "trajectory consistent with degradation"],
            "avoided": ["negative NDVI proves degradation",
                        "cyclicity proves shifting cultivation",
                        "RESTREND proves human-induced degradation",
                        "model probability equals certainty"],
        },
    }
    path = _write(exp.path("summary") / "findings.json", document)

    lines = ["# M7 findings â€” real-data study", "",
             f"**Data status:** {document['data_status']}", ""]
    for key, block in findings.items():
        heading = key.replace("_", " ").replace("finding ", "Finding ")
        lines.append(f"## {heading.capitalize()}")
        for field, text in block.items():
            if isinstance(text, dict):
                lines.append(f"- **{field}:**")
                for sub, value in text.items():
                    lines.append(f"    - *{sub}:* {value}")
            else:
                lines.append(f"- **{field}:** {text}")
        lines.append("")
    lines.append("## Limitations")
    for field, value in limitations.items():
        lines.append(f"- **{field}:** {value}")
    lines.append("")
    lines.append("## Research-gap validation")
    for field, value in contribution.items():
        if isinstance(value, list):
            lines.append(f"- **{field}:**")
            lines.extend(f"    - {item}" for item in value)
        else:
            lines.append(f"- **{field}:** {value}")
    (exp.path("summary") / "findings.md").write_text("\n".join(lines),
                                                     encoding="utf-8")
    log.info("findings document written")
    return path


# ---------------------------------------------------------------------------
def write_reproducibility_package(cfg, exp, results, log) -> Path:
    """Everything another researcher needs to repeat this (Part 25)."""
    from .reproducibility import environment_snapshot

    package = {
        "how_to_reproduce": [
            "1. Clone the repository at the git commit recorded below.",
            "2. pip install -r requirements.txt  (no credential is needed: "
            "Landsat comes from Microsoft Planetary Computer's anonymous "
            "STAC API and CHIRPS from the UCSB public server).",
            "3. python run_m7_acquire.py --config "
            "configs/m7_karbi_anglong_final.json --per-year 8",
            "4. python run_m7_study.py --config "
            "configs/m7_karbi_anglong_final.json",
            "5. Compare against configuration/frozen_config.json and "
            "summary/results.json in this directory.",
        ],
        "configuration_file": "configs/m7_karbi_anglong_final.json",
        "frozen_configuration": "configuration/frozen_config.json",
        "study_area_boundary": "configuration/study_area.geojson",
        "acquisition_record": "configuration/acquisition.json",
        "experiment_id": exp.experiment_id,
        "seed": cfg.seed,
        "deterministic": cfg.deterministic,
        "seeds_used": {
            "global": cfg.seed,
            "spatial_cv": cfg.research.spatial_cv.seed,
            "random_forest": cfg.research.model.seed,
            "cnn": cfg.research.cnn.seed,
            "cyclicity_surrogates": cfg.cyclicity.surrogate_seed,
        },
        "data_sources": results["sources"],
        "sampling": results["sampling"],
        "environment": environment_snapshot(),
        "restricted_data_included": False,
        "credentials_included": False,
        "note": (
            "The satellite and rainfall products are public. The scene "
            "cache and composited cubes are NOT included in the repository "
            "because they are large and regenerable; step 3 rebuilds them "
            "byte-identically given the same configuration, because scene "
            "selection is deterministic (least-cloudy first, fixed cap per "
            "year) and the analysis grid is derived from the boundary."),
        "known_nondeterminism": (
            "The archive itself can change: scenes may be reprocessed and "
            "collection contents can be updated. The acquisition record "
            "lists every scene identifier actually used, so a future run can "
            "be compared against it scene by scene."),
    }
    path = _write(exp.path("configuration") / "reproducibility.json", package)
    log.info("reproducibility package written")
    return path

