"""Compare two M7 runs side by side (M7 correction, Part 13).

The corrected run changes two things at once - the study-area geometry and
the way 30 m pixels become 300 m cells - so the differences below are the
COMBINED effect of both. That is stated rather than glossed: this table
shows what the corrections did to the numbers, not which correction did
what.

Areas are not directly comparable between runs whose extents differ, so both
the absolute area and the share of the analysed area are shown. The share is
the fair comparison; the absolute area is included because it is what a
report quotes.

    python tools/compare_m7_runs.py --before results/final_real_data/<a> \
                                    --after  results/final_real_data/<b>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(run: Path) -> dict:
    path = run / "summary" / "results.json"
    if not path.exists():
        raise SystemExit(f"no summary/results.json under {run}")
    return json.loads(path.read_text())


def _trend_area(results: dict, key: str) -> float:
    for row in results.get("trend", {}).get("areas", []):
        if row["class"] == key:
            return float(row["area_km2"])
    return float("nan")


def _restrend_area(results: dict, key: str) -> float:
    for row in results.get("restrend", {}).get("categories", []):
        if row["category"] == key:
            return float(row["area_km2"])
    return float("nan")


def rows(results: dict) -> dict:
    trend = results.get("trend", {})
    trajectories = results.get("trajectories", {})
    areas = trajectories.get("areas_km2", {})
    total = float(trend.get("analysed_area_km2", float("nan")))
    experiment = results.get("experiment", {})
    grid = experiment.get("grid", [None, None])

    def share(value):
        return value / total if total and total == total else float("nan")

    return {
        "study area": results.get("study_area", {}).get("name", "?"),
        "boundary kind": results.get("study_area", {}).get("attributes", {})
                                .get("geometry_kind", "?"),
        "analysis grid": f"{grid[0]} x {grid[1]}",
        "analysed area (km2)": total,
        "pixels analysed": experiment.get("n_analysed", float("nan")),
        "temporal observations": experiment.get("n_time_steps", float("nan")),
        "significant increase (km2)": _trend_area(results,
                                                  "significant_increase"),
        "significant decrease (km2)": _trend_area(results,
                                                  "significant_decrease"),
        "no significant trend (km2)": _trend_area(results,
                                                  "no_significant_trend"),
        "RESTREND-valid (km2)": results.get("restrend", {}).get(
            "restrend_valid_area_km2", float("nan")),
        "RESTREND-valid (share)": results.get("restrend", {}).get(
            "restrend_valid_fraction", float("nan")),
        "decline persisting after adjustment (km2)": _restrend_area(
            results, "decline_persists_after_climate_adjustment"),
        "cyclic (km2)": results.get("cyclicity", {}).get(
            "periodic_area_km2", float("nan")),
        "disturbed (km2)": results.get("disturbance_recovery", {}).get(
            "disturbed_area_km2", float("nan")),
        "trajectory Stable (km2)": areas.get("Stable", float("nan")),
        "trajectory Degrading (km2)": areas.get("Degrading", float("nan")),
        "trajectory Recovering (km2)": areas.get("Recovering", float("nan")),
        "trajectory Uncertain (km2)": areas.get("Uncertain / Other",
                                                float("nan")),
        "trend-only flags (pixels)": results.get(
            "baseline_comparison", {}).get("baseline_flagged_pixels",
                                           float("nan")),
        "integrated persistent decline (pixels)": results.get(
            "baseline_comparison", {}).get(
            "integrated_persistent_decline_pixels", float("nan")),
        "reduction from baseline": results.get("baseline_comparison", {}).get(
            "reduction_from_baseline_to_persistent", float("nan")),
        "sensor step (NDVI)": results.get("sensor_confound", {}).get(
            "step_ndvi", float("nan")),
        "paired cross-sensor residual (NDVI)": results.get(
            "sensor_confound", {}).get("paired_cross_sensor_residual_ndvi"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    before_run, after_run = Path(args.before), Path(args.after)
    before, after = rows(load(before_run)), rows(load(after_run))

    lines = [f"{'quantity':<45} {'BEFORE':>18} {'AFTER':>18}",
             "-" * 84]
    table = []
    for key in before:
        a, b = before[key], after.get(key)
        formatted = []
        for value in (a, b):
            if value is None:
                formatted.append("n/a")
            elif isinstance(value, float):
                formatted.append("n/a" if value != value else
                                 (f"{value:,.4f}" if abs(value) < 10
                                  else f"{value:,.0f}"))
            else:
                formatted.append(str(value))
        lines.append(f"{key:<45} {formatted[0]:>18} {formatted[1]:>18}")
        table.append({"quantity": key, "before": a, "after": b})
    report = "\n".join(lines)
    print(report)

    payload = {
        "before_run": str(before_run), "after_run": str(after_run),
        "rows": table,
        "caveat": (
            "The corrected run changes the study-area geometry AND the "
            "30 m -> 300 m coarsening method at the same time, so every "
            "difference is the combined effect of both. Absolute areas are "
            "not directly comparable between different extents; compare the "
            "shares."),
    }
    target = Path(args.out) if args.out else (after_run / "summary"
                                              / "comparison_with_previous.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str))
    (target.with_suffix(".txt")).write_text(report, encoding="utf-8")
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
