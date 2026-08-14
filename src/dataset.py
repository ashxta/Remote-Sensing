"""Standardized-data contract and validity checks (M2 Part 10).

This module is the boundary between a DATA SOURCE (today the synthetic
generator, from M6 a real remote-sensing loader) and everything downstream.
Anything that reaches feature engineering must first pass through
`validate_dataset`.

Policy
------
Invalid data raise `DatasetValidationError` with a message naming the
failing check. Nothing here repairs data: no clipping of out-of-range NDVI
into range, no filling of NaN, no substitution of a default for missing
metadata. Silently converting invalid values into valid-looking ones is how
a pipeline produces confident nonsense.

The one deliberate exception is that a NaN is accepted everywhere as the
representation of a missing observation - that is the project's documented
missing-data convention, and per-pixel gating (`quality.assess`) decides
which pixels have enough valid observations to analyse.
"""
from __future__ import annotations

import numpy as np

from .config import Config

__all__ = ["DatasetValidationError", "validate_dataset",
           "describe_georeference"]


class DatasetValidationError(ValueError):
    """Raised when input data violate the standardized-data contract."""


def _flatten(stack: np.ndarray) -> np.ndarray:
    return stack.reshape(stack.shape[0], -1)


def describe_georeference(crs, transform) -> dict:
    """Summarise a spatial reference without assuming any study area."""
    info = {"crs": str(crs) if crs else None,
            "transform": list(transform)[:6] if transform is not None
            else None}
    if transform is not None:
        values = list(transform)[:6]
        info["pixel_width"] = abs(float(values[0]))
        info["pixel_height"] = abs(float(values[4]))
        info["origin_x"] = float(values[2])
        info["origin_y"] = float(values[5])
    return info


def _check_georeference(crs, transform, height, width, errors) -> None:
    """Geometry sanity only. No hard-coded region, extent or resolution."""
    if transform is None:
        errors.append("affine transform is missing")
        return
    values = [float(v) for v in list(transform)[:6]]
    if not all(np.isfinite(values)):
        errors.append("affine transform contains non-finite coefficients")
        return
    if abs(values[0]) <= 0 or abs(values[4]) <= 0:
        errors.append("affine transform has a zero pixel size")
        return
    if crs is None:
        return
    geographic = bool(getattr(crs, "is_geographic", False))
    if not geographic:
        return
    left, top = values[2], values[5]
    right = left + values[0] * width
    bottom = top + values[4] * height
    if not (-180.0 <= min(left, right) and max(left, right) <= 180.0):
        errors.append(f"longitude bounds {min(left, right):.4f}..."
                      f"{max(left, right):.4f} fall outside [-180, 180]")
    if not (-90.0 <= min(bottom, top) and max(bottom, top) <= 90.0):
        errors.append(f"latitude bounds {min(bottom, top):.4f}..."
                      f"{max(bottom, top):.4f} fall outside [-90, 90]")


def validate_dataset(ndvi, rain, *, cfg: Config | None = None, crs=None,
                     transform=None, require_georef: bool = False,
                     expected_time_steps: int | None = None,
                     strict: bool = True, rain_min: float = 0.0) -> dict:
    """Validate a standardized (T, N) or (T, H, W) NDVI/rainfall pair.

    Checks, in order:
      1. dimensions: 2-D or 3-D, identical shapes, non-empty spatial extent
      2. time dimension: at least 3 steps, and `expected_time_steps` when the
         caller declares one (e.g. `len(cfg.years)`)
      3. no infinities anywhere - missing data must be NaN, not +/-inf
      4. NDVI inside the physical range configured in `QualityConfig`
      5. rainfall not negative (a rainfall total cannot be below zero)
      6. at least one pixel with enough valid observations for the
         configured estimators
      7. georeference present when required, and geometrically sane

    With `strict=False` the value-range violations (4, 5) are reported in the
    returned metadata instead of raising, which is useful for inspecting a
    new data source before deciding how to gate it. Structural violations
    (1, 2, 3, 6, 7) always raise.
    """
    cfg = cfg or Config()
    errors: list[str] = []
    soft: list[str] = []

    ndvi = np.asarray(ndvi, dtype="float64")
    rain = np.asarray(rain, dtype="float64")
    if ndvi.ndim not in (2, 3) or rain.shape != ndvi.shape:
        raise DatasetValidationError(
            f"NDVI {ndvi.shape} and rainfall {rain.shape} must have "
            "identical (T,N) or (T,H,W) dimensions")
    if ndvi.shape[0] < 3 or any(s == 0 for s in ndvi.shape[1:]):
        raise DatasetValidationError(
            "dataset needs at least 3 time steps and non-empty spatial "
            f"dimensions, got {ndvi.shape}")
    if expected_time_steps is not None \
            and ndvi.shape[0] != int(expected_time_steps):
        raise DatasetValidationError(
            f"expected {int(expected_time_steps)} time steps, got "
            f"{ndvi.shape[0]}; the configured time axis and the data "
            "disagree")
    if np.isinf(ndvi).any() or np.isinf(rain).any():
        raise DatasetValidationError(
            "infinite NDVI/rainfall values are invalid; use NaN for missing "
            "observations")

    flat_ndvi, flat_rain = _flatten(ndvi), _flatten(rain)
    valid_counts = np.isfinite(flat_ndvi).sum(axis=0)
    finite_ndvi = ndvi[np.isfinite(ndvi)]
    finite_rain = rain[np.isfinite(rain)]
    if finite_ndvi.size == 0:
        raise DatasetValidationError("NDVI contains no valid observations")

    out_of_range = int(((ndvi < cfg.quality.ndvi_min)
                        | (ndvi > cfg.quality.ndvi_max)).sum())
    if out_of_range:
        (errors if strict else soft).append(
            f"{out_of_range} NDVI values fall outside the physical range "
            f"[{cfg.quality.ndvi_min}, {cfg.quality.ndvi_max}]")
    negative_rain = int((rain < rain_min).sum())
    if negative_rain:
        (errors if strict else soft).append(
            f"{negative_rain} rainfall values are below {rain_min}")

    n_sufficient = int((valid_counts >= cfg.quality.min_valid_obs).sum())
    if n_sufficient == 0:
        errors.append(
            f"no pixel has the {cfg.quality.min_valid_obs} valid "
            "observations required by QualityConfig.min_valid_obs")

    if require_georef and (not crs or transform is None):
        errors.append("geospatial experiment requires a CRS and a transform")
    if transform is not None or crs is not None:
        height, width = (ndvi.shape[1], ndvi.shape[2]) if ndvi.ndim == 3 \
            else (1, ndvi.shape[1])
        _check_georeference(crs, transform, height, width, errors)

    if errors:
        raise DatasetValidationError("; ".join(errors))

    return {
        "shape": list(ndvi.shape),
        "n_time_steps": int(ndvi.shape[0]),
        "n_pixels": int(flat_ndvi.shape[1]),
        "ndvi_min": float(np.min(finite_ndvi)),
        "ndvi_max": float(np.max(finite_ndvi)),
        "ndvi_out_of_physical_range": out_of_range,
        "rain_min": float(np.min(finite_rain)) if finite_rain.size else None,
        "rain_max": float(np.max(finite_rain)) if finite_rain.size else None,
        "rain_negative_values": negative_rain,
        "ndvi_nan": int(np.isnan(ndvi).sum()),
        "rain_nan": int(np.isnan(rain).sum()),
        "pixels_with_sufficient_observations": n_sufficient,
        "min_valid_obs_required": int(cfg.quality.min_valid_obs),
        "mean_valid_observations": float(np.mean(valid_counts)),
        "crs_present": bool(crs),
        "transform_present": transform is not None,
        "georeference": describe_georeference(crs, transform),
        "warnings": soft,
        "strict": bool(strict),
    }
