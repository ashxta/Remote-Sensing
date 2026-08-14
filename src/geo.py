"""Georeferenced raster I/O (M1 Part 7).

The original pipeline read GeoTIFFs, discarded their spatial reference and
emitted only PNGs. Nothing it produced could be opened in QGIS or overlaid
on other layers. This module fixes that: the spatial reference of the input
stack is captured once in a `GeoRef` and carried through to every raster
the pipeline writes.

Guarantees for outputs:
  * CRS preserved from the source
  * affine transform preserved (hence resolution, origin and bounds)
  * raster dimensions preserved
  * explicit NoData value written, never a silent 0
  * dtype chosen per layer (float32 for continuous, uint8/int16 for classes)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import array_bounds

NODATA_FLOAT = -9999.0
NODATA_INT = 255

__all__ = ["GeoRef", "NODATA_FLOAT", "NODATA_INT", "unflatten", "write_raster",
           "write_layer", "write_table", "write_point_geojson",
           "write_class_geojson", "verify_georeference"]


@dataclass
class GeoRef:
    """Spatial reference of a raster grid."""
    crs: object
    transform: object
    height: int
    width: int

    @classmethod
    def from_raster(cls, path) -> "GeoRef":
        with rasterio.open(path) as src:
            return cls(src.crs, src.transform, src.height, src.width)

    @property
    def shape(self):
        return (self.height, self.width)

    @property
    def bounds(self):
        return array_bounds(self.height, self.width, self.transform)

    @property
    def resolution(self):
        return (abs(self.transform.a), abs(self.transform.e))

    def to_dict(self) -> dict:
        left, bottom, right, top = self.bounds
        return {
            "crs": str(self.crs) if self.crs else None,
            "transform": list(self.transform)[:6],
            "height": int(self.height),
            "width": int(self.width),
            "resolution_x": float(self.resolution[0]),
            "resolution_y": float(self.resolution[1]),
            "bounds": {"left": float(left), "bottom": float(bottom),
                       "right": float(right), "top": float(top)},
        }

    def save(self, path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        return p

    def validate(self, array: np.ndarray) -> None:
        if array.shape[-2:] != self.shape:
            raise ValueError(
                f"array shape {array.shape[-2:]} does not match georeference "
                f"grid {self.shape}")


def unflatten(values: np.ndarray, mask: np.ndarray, shape,
              fill=np.nan) -> np.ndarray:
    """Scatter a (N,) vector of per-pixel results back onto the 2-D grid.

    `mask` is the boolean grid selecting the N analysed pixels.
    """
    values = np.asarray(values).reshape(-1)
    if mask.sum() != values.size:
        raise ValueError(f"mask selects {int(mask.sum())} pixels but got "
                         f"{values.size} values")
    grid = np.full(shape, fill, dtype="float64")
    grid[mask] = values
    return grid


def write_raster(path, array: np.ndarray, georef: GeoRef,
                 dtype: str = "float32", nodata=None,
                 band_names=None, description: str = "") -> Path:
    """Write a georeferenced GeoTIFF, preserving CRS/transform/nodata."""
    arr = np.asarray(array)
    if arr.ndim == 2:
        arr = arr[None, ...]
    georef.validate(arr)

    if nodata is None:
        nodata = NODATA_FLOAT if dtype.startswith("float") else NODATA_INT

    out = arr.astype("float64").copy()
    out[~np.isfinite(out)] = nodata
    out = out.astype(dtype)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff", "height": georef.height, "width": georef.width,
        "count": out.shape[0], "dtype": dtype, "crs": georef.crs,
        "transform": georef.transform, "nodata": nodata,
        "compress": "deflate", "tiled": False,
    }
    with rasterio.open(p, "w", **profile) as dst:
        dst.write(out)
        if band_names:
            for i, nm in enumerate(band_names[:out.shape[0]], start=1):
                dst.set_band_description(i, str(nm))
        if description:
            dst.update_tags(DESCRIPTION=description)
    return p


def write_layer(path, values: np.ndarray, mask: np.ndarray, georef: GeoRef,
                dtype: str = "float32", nodata=None,
                description: str = "") -> Path:
    """Convenience: scatter a per-pixel vector to the grid and write it."""
    grid = unflatten(values, mask, georef.shape)
    return write_raster(path, grid, georef, dtype=dtype, nodata=nodata,
                        description=description)


def write_table(path, df) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Vector exports (M4)
#
# GeoTIFF is the right format for a full grid, but per-sample results
# (predictions, confidence, fold membership) are point observations, and
# class maps are more useful to a GIS as polygons. Both are written in the
# GeoJSON standard's own CRS convention: RFC 7946 mandates WGS84 lon/lat, so
# coordinates are reprojected when the source grid is in another CRS, and
# the CRS actually used is recorded in the file's properties.
# ---------------------------------------------------------------------------
def _to_wgs84(xs, ys, crs):
    """Reproject coordinates to WGS84 lon/lat, as RFC 7946 requires."""
    if crs is None:
        return xs, ys, None
    try:
        from rasterio.crs import CRS
        from rasterio.warp import transform as warp_transform
        source = CRS.from_user_input(crs)
        if source.to_epsg() == 4326:
            return xs, ys, "EPSG:4326"
        lon, lat = warp_transform(source, CRS.from_epsg(4326), list(xs),
                                  list(ys))
        return np.asarray(lon), np.asarray(lat), "EPSG:4326"
    except Exception:                                   # pragma: no cover
        # Never lose the data because a reprojection failed; record that the
        # coordinates are still in the source CRS instead.
        return xs, ys, str(crs)


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_point_geojson(path, mask: np.ndarray, georef: GeoRef,
                        properties: dict, *, description: str = "") -> Path:
    """Write per-pixel results as GeoJSON points at pixel centres.

    `mask` selects the pixels on the grid; every array in `properties` must
    have one entry per selected pixel, in the same order the mask flattens.
    Non-finite values are written as null rather than as NaN, which is not
    valid JSON.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != georef.shape:
        raise ValueError(f"mask shape {mask.shape} does not match the "
                         f"georeference grid {georef.shape}")
    rows, cols = np.nonzero(mask)
    n = rows.size
    for name, values in properties.items():
        if len(np.asarray(values).reshape(-1)) != n:
            raise ValueError(
                f"property {name!r} has "
                f"{len(np.asarray(values).reshape(-1))} values but the mask "
                f"selects {n} pixels")
    # Pixel centres, not corners.
    xs, ys = rasterio.transform.xy(georef.transform, rows, cols, offset="center")
    xs, ys, crs_used = _to_wgs84(np.asarray(xs), np.asarray(ys), georef.crs)

    features = []
    for i in range(n):
        attributes = {name: _json_safe(np.asarray(values).reshape(-1)[i])
                      for name, values in properties.items()}
        attributes["row"] = int(rows[i])
        attributes["col"] = int(cols[i])
        features.append({"type": "Feature",
                         "geometry": {"type": "Point",
                                      "coordinates": [float(xs[i]),
                                                      float(ys[i])]},
                         "properties": attributes})
    payload = {"type": "FeatureCollection",
               "name": Path(path).stem,
               "crs_of_coordinates": crs_used,
               "source_crs": str(georef.crs) if georef.crs else None,
               "description": description,
               "features": features}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))
    return p


def write_class_geojson(path, class_grid: np.ndarray, georef: GeoRef, *,
                        class_names: dict | None = None,
                        ignore_values=(0,), max_features: int = 20000,
                        description: str = "") -> Path:
    """Vectorise a class grid into GeoJSON polygons.

    Contiguous runs of one class become one polygon, which is what a GIS
    wants for a trajectory or land-cover map. `max_features` caps the output
    so a noisy per-pixel map cannot produce a multi-hundred-megabyte file;
    the cap and the number of polygons dropped are recorded in the file.
    """
    from rasterio.features import shapes

    grid = np.asarray(class_grid)
    if grid.shape != georef.shape:
        raise ValueError(f"class grid {grid.shape} does not match the "
                         f"georeference grid {georef.shape}")
    values = np.where(np.isfinite(grid.astype("float64")), grid, 0
                      ).astype("int32")
    valid = ~np.isin(values, list(ignore_values))
    polygons = list(shapes(values, mask=valid, transform=georef.transform))

    truncated = max(len(polygons) - max_features, 0)
    features = []
    for geometry, value in polygons[:max_features]:
        coordinates = geometry["coordinates"]
        converted = []
        for ring in coordinates:
            xs = np.array([point[0] for point in ring])
            ys = np.array([point[1] for point in ring])
            xs, ys, crs_used = _to_wgs84(xs, ys, georef.crs)
            converted.append([[float(x), float(y)] for x, y in zip(xs, ys)])
        properties = {"class_value": int(value)}
        if class_names:
            properties["class_name"] = class_names.get(int(value),
                                                       str(int(value)))
        features.append({"type": "Feature",
                         "geometry": {"type": "Polygon",
                                      "coordinates": converted},
                         "properties": properties})
    crs_used = "EPSG:4326" if georef.crs else None
    payload = {"type": "FeatureCollection",
               "name": Path(path).stem,
               "crs_of_coordinates": crs_used,
               "source_crs": str(georef.crs) if georef.crs else None,
               "description": description,
               "n_polygons": len(features),
               "n_polygons_dropped_by_cap": int(truncated),
               "max_features": int(max_features),
               "features": features}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))
    return p


def verify_georeference(path) -> dict:
    """Reopen a written raster and report its spatial properties.

    Used by the test suite to prove round-trip preservation.
    """
    with rasterio.open(path) as src:
        left, bottom, right, top = src.bounds
        return {
            "crs": str(src.crs) if src.crs else None,
            "transform": list(src.transform)[:6],
            "height": src.height, "width": src.width,
            "count": src.count, "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "bounds": {"left": left, "bottom": bottom,
                       "right": right, "top": top},
            "resolution": src.res,
        }
