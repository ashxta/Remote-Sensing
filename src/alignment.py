"""Spatial and temporal alignment (M6 Parts 9, 10).

RESTREND regresses each pixel's NDVI on that same pixel's rainfall. If the
two cubes are not on identical grids, the regression pairs one place's
vegetation with another place's rain and returns a confident, meaningless
answer. Nothing downstream can detect that: the arrays have the right shape,
the numbers are finite, the r-squared is a number. Misalignment is the
quietest way to invalidate this entire project, so it is checked
explicitly, it is checked before the data are used, and a failure raises.

    "Do not simply resize arrays."

Resizing an array changes its sample spacing without changing the
coordinates it claims to have. `reproject_to_grid` instead resamples through
the affine transforms and the CRS, so each output cell draws from the input
cells that actually cover its ground footprint.

RESAMPLING CHOICE
-----------------
CHIRPS is ~5.5 km; the NDVI grid is 30 m. Going from coarse to fine is
interpolation, and the choice matters:

* `bilinear` (the default) treats the rainfall field as the smooth,
  spatially autocorrelated surface it physically is, and avoids the blocky
  discontinuities that `nearest` would impose on the RESTREND covariate -
  artificial edges that would appear in the residual-trend map as if they
  were real boundaries.
* `nearest` is correct for categorical rasters (reference labels, land-cover
  classes) where an interpolated value would be a class that does not exist,
  and is the default for those.

Neither creates information. A 30 m rainfall grid derived from a 5.5 km
product carries 5.5 km of real spatial detail, and that limitation is
recorded in the dataset metadata and repeated in the M6 report.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .geo import GeoRef

__all__ = ["AlignmentError", "AlignmentReport", "check_grid_alignment",
           "require_alignment", "reproject_to_grid", "reproject_cube",
           "align_to_reference", "check_temporal_alignment",
           "RESAMPLING_FOR"]

#: Documented resampling choice per kind of layer. Continuous geophysical
#: fields interpolate; categorical layers must not.
RESAMPLING_FOR = {
    "rainfall": "bilinear",
    "index": "bilinear",
    "reflectance": "bilinear",
    "categorical": "nearest",
    "labels": "nearest",
    "count": "nearest",
}


class AlignmentError(ValueError):
    """Raised when two grids that must match do not."""


@dataclass
class AlignmentReport:
    """The outcome of comparing two grids, with every mismatch named."""
    aligned: bool
    mismatches: List[str]
    reference: Dict[str, Any]
    candidate: Dict[str, Any]
    tolerance: float

    def summary(self) -> dict:
        return {"aligned": self.aligned, "mismatches": list(self.mismatches),
                "reference_grid": self.reference,
                "candidate_grid": self.candidate,
                "tolerance": self.tolerance}


def _crs_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        from rasterio.crs import CRS
        return CRS.from_user_input(a) == CRS.from_user_input(b)
    except Exception:                                   # pragma: no cover
        return str(a) == str(b)


def check_grid_alignment(reference: GeoRef, candidate: GeoRef, *,
                         tolerance: float = 1e-6) -> AlignmentReport:
    """Compare two grids on every property RESTREND depends on.

    Checks, each reported separately so a failure says what is wrong:
    CRS, pixel width and height, raster dimensions, affine rotation terms,
    origin, and - the one that is easy to miss - sub-pixel grid offset. Two
    grids can share a CRS, a resolution and an extent and still be offset by
    half a pixel; every pixel would then be paired with a neighbour.
    """
    mismatches: List[str] = []
    a, b = reference.transform, candidate.transform

    if not _crs_equal(reference.crs, candidate.crs):
        mismatches.append(
            f"CRS differs: reference {reference.crs} vs {candidate.crs}")
    if abs(abs(a.a) - abs(b.a)) > tolerance \
            or abs(abs(a.e) - abs(b.e)) > tolerance:
        mismatches.append(
            f"resolution differs: reference {reference.resolution} vs "
            f"{candidate.resolution}")
    if reference.shape != candidate.shape:
        mismatches.append(
            f"raster dimensions differ: reference {reference.shape} vs "
            f"{candidate.shape}")
    if abs(a.b) > tolerance or abs(a.d) > tolerance \
            or abs(b.b) > tolerance or abs(b.d) > tolerance:
        mismatches.append(
            "affine transform is rotated or sheared; the pipeline assumes "
            "north-up, axis-aligned grids")
    if abs(a.c - b.c) > tolerance or abs(a.f - b.f) > tolerance:
        mismatches.append(
            f"origin differs: reference ({a.c}, {a.f}) vs ({b.c}, {b.f})")
        # An origin offset that is a whole number of pixels is a different
        # (recoverable) failure from one that is not; say which.
        if abs(a.a) > 0 and abs(a.e) > 0:
            offset_x = (a.c - b.c) / a.a
            offset_y = (a.f - b.f) / a.e
            if abs(offset_x - round(offset_x)) > 1e-3 \
                    or abs(offset_y - round(offset_y)) > 1e-3:
                mismatches.append(
                    f"grids are offset by a FRACTION of a pixel "
                    f"({offset_x:.4f}, {offset_y:.4f}); every pixel would be "
                    "paired with part of a neighbour")
    if not np.allclose(reference.bounds, candidate.bounds,
                       atol=max(tolerance, 1e-9), rtol=0):
        mismatches.append(
            f"extent differs: reference {tuple(round(v, 8) for v in reference.bounds)} "
            f"vs {tuple(round(v, 8) for v in candidate.bounds)}")

    return AlignmentReport(aligned=not mismatches, mismatches=mismatches,
                           reference=reference.to_dict(),
                           candidate=candidate.to_dict(), tolerance=tolerance)


def require_alignment(reference: GeoRef, candidate: GeoRef, *,
                      what: str = "layer", tolerance: float = 1e-6
                      ) -> AlignmentReport:
    """`check_grid_alignment`, but a mismatch raises."""
    report = check_grid_alignment(reference, candidate, tolerance=tolerance)
    if not report.aligned:
        raise AlignmentError(
            f"{what} is not aligned with the reference grid: "
            + "; ".join(report.mismatches)
            + ". Resample it onto the reference grid with "
              "alignment.reproject_to_grid rather than reshaping the array.")
    return report


# ---------------------------------------------------------------------------
def _resampling(name: str):
    from rasterio.enums import Resampling
    try:
        return getattr(Resampling, str(name).lower())
    except AttributeError:
        raise AlignmentError(
            f"unknown resampling method {name!r}; rasterio provides "
            f"{[r.name for r in Resampling]}") from None


def reproject_to_grid(array: np.ndarray, source: GeoRef, target: GeoRef, *,
                      resampling: str = "bilinear",
                      nodata: float = np.nan) -> np.ndarray:
    """Resample a 2-D array onto another grid through its geometry.

    NoData is preserved: input NaN does not contaminate neighbouring output
    cells, and output cells with no valid input coverage come back NaN
    rather than 0.
    """
    from rasterio.warp import reproject

    values = np.asarray(array, dtype="float64")
    if values.shape != source.shape:
        raise AlignmentError(f"array {values.shape} does not match its "
                             f"declared source grid {source.shape}")
    if source.crs is None or target.crs is None:
        raise AlignmentError("reprojection needs a CRS on both grids; one of "
                             "them is ungeoreferenced")
    destination = np.full(target.shape, np.nan, dtype="float64")
    reproject(source=values, destination=destination,
              src_transform=source.transform, src_crs=source.crs,
              dst_transform=target.transform, dst_crs=target.crs,
              src_nodata=nodata, dst_nodata=nodata,
              resampling=_resampling(resampling))
    return destination


def reproject_cube(cube: np.ndarray, source: GeoRef, target: GeoRef, *,
                   resampling: str = "bilinear",
                   nodata: float = np.nan) -> np.ndarray:
    """`reproject_to_grid` applied band by band to a (T, H, W) cube."""
    values = np.asarray(cube, dtype="float64")
    if values.ndim == 2:
        return reproject_to_grid(values, source, target,
                                 resampling=resampling, nodata=nodata)
    if values.ndim != 3:
        raise AlignmentError(f"expected a (T, H, W) cube, got {values.shape}")
    return np.stack([reproject_to_grid(band, source, target,
                                       resampling=resampling, nodata=nodata)
                     for band in values])


def align_to_reference(cube: np.ndarray, source: GeoRef, target: GeoRef, *,
                       kind: str = "rainfall",
                       resampling: Optional[str] = None,
                       nodata: float = np.nan) -> tuple:
    """Bring a layer onto the reference grid, and prove that it arrived.

    Returns (aligned_cube, report). The report records whether resampling
    was needed, which method was used and why, and the post-resampling
    alignment check - so the run's metadata can answer "was the rainfall
    actually on the NDVI grid?" without anyone re-deriving it.
    """
    method = resampling or RESAMPLING_FOR.get(str(kind).lower(), "bilinear")
    before = check_grid_alignment(target, source)
    if before.aligned:
        return np.asarray(cube, dtype="float64"), {
            "resampled": False, "kind": kind,
            "reason": "source already on the reference grid",
            "alignment_before": before.summary(),
            "alignment_after": before.summary()}

    aligned = reproject_cube(cube, source, target, resampling=method,
                             nodata=nodata)
    after = check_grid_alignment(target, target)
    return aligned, {
        "resampled": True,
        "kind": kind,
        "method": method,
        "reason": "; ".join(before.mismatches),
        "justification": (
            "bilinear interpolation for a continuous, spatially "
            "autocorrelated field; nearest neighbour for categorical layers "
            "where an interpolated value would not be a real class"
            if method in ("bilinear", "nearest") else
            f"configured resampling method {method}"),
        "source_grid": source.to_dict(),
        "target_grid": target.to_dict(),
        "native_resolution_retained": (
            "Resampling to a finer grid does not add spatial detail; the "
            f"layer still carries its native {source.resolution} resolution."),
        "nodata_policy": "NaN in, NaN out; uncovered cells are NaN",
        "alignment_before": before.summary(),
        "alignment_after": after.summary(),
    }


# ---------------------------------------------------------------------------
# Temporal alignment (Part 9)
# ---------------------------------------------------------------------------
def check_temporal_alignment(vegetation_times: Sequence[Any],
                             rainfall_times: Sequence[Any], *,
                             what: str = "rainfall") -> dict:
    """Verify that the two records describe the SAME periods, in order.

    A silent one-step shift between NDVI and rainfall is undetectable
    downstream and would make RESTREND regress each year's vegetation on the
    previous year's rain. Labels are compared as strings so an accidental
    reordering, truncation or offset is caught by identity rather than by
    length alone.
    """
    veg = [str(t) for t in vegetation_times]
    rain = [str(t) for t in rainfall_times]
    problems: List[str] = []
    if len(veg) != len(rain):
        problems.append(f"vegetation has {len(veg)} time steps but {what} "
                        f"has {len(rain)}")
    else:
        differing = [(i, a, b) for i, (a, b) in enumerate(zip(veg, rain))
                     if a != b]
        if differing:
            head = ", ".join(f"step {i}: {a!r} vs {b!r}"
                             for i, a, b in differing[:5])
            problems.append(f"{len(differing)} time labels differ ({head})")
            # Which way the offset runs matters for diagnosis. If the
            # rainfall label at each step is EARLIER than the vegetation
            # label, the covariate is stale - it lags. If it is later, the
            # covariate is from the future - it leads, which is look-ahead.
            if len(veg) > 1 and veg[:-1] == rain[1:]:
                problems.append("the two records are offset by one step; "
                                f"{what} lags vegetation (its values are "
                                "one period too early)")
            elif len(veg) > 1 and veg[1:] == rain[:-1]:
                problems.append("the two records are offset by one step; "
                                f"{what} leads vegetation (its values are "
                                "one period too late, which is look-ahead)")
    if veg != sorted(veg):
        problems.append("vegetation time labels are not in ascending order")

    if problems:
        raise AlignmentError(
            "temporal alignment failed: " + "; ".join(problems)
            + ". The two records must cover identical, identically ordered "
              "periods before RESTREND can pair them.")
    return {"aligned": True, "n_time_steps": len(veg),
            "first": veg[0] if veg else None, "last": veg[-1] if veg else None,
            "labels_identical": True,
            "checked": f"vegetation vs {what}"}
