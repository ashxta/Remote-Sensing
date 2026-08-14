"""Does the framework separate degradation from its confounders? (M5)

THE RESEARCH QUESTION, MADE MEASURABLE
--------------------------------------
    Can multi-temporal remote-sensing analysis distinguish persistent land
    degradation from cyclic or environmentally driven vegetation dynamics?

Overall accuracy cannot answer that. A method can score 0.95 accuracy while
systematically mistaking every cyclic pixel for a degrading one, because the
confounder classes are a minority. The question is not "how often is the
method right" but "how often does it call a NON-degraded pixel degraded, and
which kind of non-degraded pixel does it confuse".

This module reports exactly that: for each method, the recall on genuinely
degrading pixels, and the FALSE-POSITIVE RATE SEPARATELY FOR EACH CONFOUNDER
class - cyclic vegetation, rainfall-driven variation, recovering pixels,
stable vegetation. A method that is good at the research question has high
recall AND a low false-positive rate on every confounder, not just a good
average.

The headline number is the `discrimination_margin`: recall on degradation
minus the WORST false-positive rate across the confounders. It is
deliberately pessimistic - a method is only as good as the confounder it
handles worst - and it is reported alongside the full table, never instead
of it.

INTERPRETATION LIMIT
--------------------
This measures separation against the REFERENCE LABELS of the dataset in
use. On synthetic data the reference classes are what the generator planted,
so the result characterises the method against that generator, not against a
landscape. It says nothing about whether a pixel is truly degraded on the
ground.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

__all__ = ["confounder_confusion", "discrimination_table",
           "run_discrimination_analysis", "DISCRIMINATION_LIMIT"]

DISCRIMINATION_LIMIT = (
    "Separation is measured against the reference labels of the dataset in "
    "use. On synthetic data those are the archetypes the generator planted, "
    "so the result characterises the method against the generator, not "
    "against any landscape.")


def confounder_confusion(reference, flagged, *, degradation_classes,
                         class_names: Mapping[int, str] | None = None
                         ) -> pd.DataFrame:
    """Per reference class: how often was it flagged as degradation?

    `flagged` is a boolean or 0/1 array of degradation calls. The result has
    one row per reference class with the count flagged, the rate, and
    whether that class is the degradation target or a confounder.
    """
    reference = np.asarray(reference)
    flagged = np.asarray(flagged).astype(bool)
    if reference.shape != flagged.shape:
        raise ValueError(f"reference {reference.shape} and flagged "
                         f"{flagged.shape} must describe the same samples")
    targets = set(int(c) for c in degradation_classes)
    rows = []
    for value in np.unique(reference):
        member = reference == value
        is_target = int(value) in targets
        rate = float(flagged[member].mean()) if member.any() else float("nan")
        rows.append({
            "reference_class": int(value),
            "class_name": (class_names or {}).get(int(value), str(value)),
            "role": "degradation target" if is_target else "confounder",
            "n": int(member.sum()),
            "n_flagged": int(flagged[member].sum()),
            # For the target this is recall; for a confounder it is the
            # false-positive rate. One column, two readings, stated in `role`.
            "flagged_rate": rate,
        })
    return pd.DataFrame(rows)


def discrimination_table(reference, predictions: Mapping[str, np.ndarray], *,
                         degradation_classes,
                         class_names: Mapping[int, str] | None = None,
                         sample_mask=None) -> tuple:
    """Confusion-with-confounders for several methods at once.

    `predictions` maps a method name to its degradation flag over the same
    samples. Returns (per-class table, per-method summary).
    """
    reference = np.asarray(reference)
    mask = np.ones(len(reference), bool) if sample_mask is None \
        else np.asarray(sample_mask, bool)
    targets = set(int(c) for c in degradation_classes)

    per_class, summary = [], []
    for method, flagged in predictions.items():
        flagged = np.asarray(flagged).astype(bool)
        table = confounder_confusion(reference[mask], flagged[mask],
                                     degradation_classes=targets,
                                     class_names=class_names)
        table.insert(0, "method", method)
        per_class.append(table)

        target_rows = table[table["role"] == "degradation target"]
        confounder_rows = table[table["role"] == "confounder"]
        recall = float(np.average(target_rows["flagged_rate"],
                                  weights=target_rows["n"])) \
            if len(target_rows) and target_rows["n"].sum() else float("nan")
        worst = confounder_rows.loc[confounder_rows["flagged_rate"].idxmax()] \
            if len(confounder_rows) else None
        overall_fp = (float(confounder_rows["n_flagged"].sum()
                            / max(confounder_rows["n"].sum(), 1))
                      if len(confounder_rows) else float("nan"))
        entry = {
            "method": method,
            "recall_on_degradation": recall,
            "false_positive_rate_overall": overall_fp,
            "worst_confounder": None if worst is None else worst["class_name"],
            "worst_confounder_false_positive_rate":
                float("nan") if worst is None else float(worst["flagged_rate"]),
            "discrimination_margin":
                float("nan") if worst is None
                else float(recall - worst["flagged_rate"]),
            "n_evaluated": int(mask.sum()),
        }
        for _, row in confounder_rows.iterrows():
            key = str(row["class_name"]).lower().replace(" ", "_").replace(
                "/", "_")
            entry[f"false_positive_rate_{key}"] = float(row["flagged_rate"])
        summary.append(entry)

    return (pd.concat(per_class, ignore_index=True) if per_class
            else pd.DataFrame()), pd.DataFrame(summary)


def run_discrimination_analysis(reference, predictions: Mapping[str, np.ndarray],
                                output_dir, *, degradation_classes,
                                class_names: Mapping[int, str] | None = None,
                                sample_mask=None, logger=None) -> dict:
    """Run the analysis, save it, and state what it does and does not show."""
    per_class, summary = discrimination_table(
        reference, predictions, degradation_classes=degradation_classes,
        class_names=class_names, sample_mask=sample_mask)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    per_class.to_csv(root / "confounder_confusion.csv", index=False)
    summary.to_csv(root / "discrimination_summary.csv", index=False)

    best = None
    if len(summary) and summary["discrimination_margin"].notna().any():
        best_row = summary.loc[summary["discrimination_margin"].idxmax()]
        best = {
            "method": str(best_row["method"]),
            "recall_on_degradation": float(best_row["recall_on_degradation"]),
            "worst_confounder": best_row["worst_confounder"],
            "worst_confounder_false_positive_rate":
                float(best_row["worst_confounder_false_positive_rate"]),
            "discrimination_margin":
                float(best_row["discrimination_margin"]),
        }

    report = {
        "question": ("Can multi-temporal remote-sensing analysis distinguish "
                     "persistent land degradation from cyclic or "
                     "environmentally driven vegetation dynamics?"),
        "how_it_is_measured": (
            "Recall on the reference degradation class, against the "
            "false-positive rate on EACH confounder class separately. The "
            "discrimination margin is recall minus the worst confounder "
            "false-positive rate, so a method is judged by the confounder it "
            "handles worst, not by an average that a minority class cannot "
            "move."),
        "degradation_classes": [int(c) for c in degradation_classes],
        "methods": summary.to_dict(orient="records"),
        "best_by_margin": best,
        "interpretation_limit": DISCRIMINATION_LIMIT,
    }
    (root / "discrimination.json").write_text(
        json.dumps(report, indent=2, default=str))

    if logger is not None:
        for row in summary.itertuples():
            logger.info("  discrimination %-22s recall %.3f | worst "
                        "confounder %s at %.3f | margin %.3f",
                        row.method, row.recall_on_degradation,
                        row.worst_confounder,
                        row.worst_confounder_false_positive_rate,
                        row.discrimination_margin)
    return {"per_class": per_class, "summary": summary, "report": report}
