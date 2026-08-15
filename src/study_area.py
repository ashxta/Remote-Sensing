"""Configurable study-area boundary (M6 Part 2).

The analytical pipeline must not know where it is. Mann-Kendall, Sen's
slope, RESTREND, cyclicity, breakpoint detection, recovery, feature
engineering, the Random Forest, the CNN and the validation design contain no
reference to any region, and nothing in this module leaks one into them: a
`StudyArea` is data, supplied by configuration, and it is consumed only by
the ACQUISITION and EXPORT stages.

    boundary (configuration)
        -> acquisition / clipping        this module
        -> preprocessing
        -> StandardizedDataset
        -> existing M1-M5 pipeline       knows nothing about geography

Replacing one district with another is therefore a configuration change, not
a code change. That is a property of the architecture; it is NOT a claim
that the framework has been experimentally validated anywhere other than the
area actually processed.

DEPENDENCIES
------------
Deliberately none beyond rasterio. GeoJSON is parsed with the standard
library and geometries are reprojected, rasterised and bounded through
rasterio's own GDAL/PROJ bindings, so boundary handling works in an
environment without geopandas, shapely, fiona or pyproj. Where fiona IS
installed, shapefiles and GeoPackages are read through it as well.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .geo import GeoRef

__all__ = ["StudyArea", "StudyAreaError", "load_study_area", "geometry_bounds",
           "pixel_area_km2", "area_statistics", "EARTH_RADIUS_KM"]

#: Authalic (equal-area) mean radius of the WGS84 ellipsoid, in kilometres.
#: Used for geographic-CRS pixel areas; see `pixel_area_km2`.
EARTH_RADIUS_KM = 6371.0072

_POLYGONAL = ("Polygon", "MultiPolygon")


class StudyAreaError(ValueError):
    """Raised when a boundary is missing, malformed or unusable."""


# ---------------------------------------------------------------------------
# Geometry helpers
#
# A GeoJSON geometry is nested lists of coordinates; every operation below
# walks that structure rather than depending on a geometry library.
# ---------------------------------------------------------------------------
def _walk_coordinates(coordinates) -> List[Sequence[float]]:
    """Yield every [x, y] position in an arbitrarily nested coordinate list."""
    if not isinstance(coordinates, (list, tuple)) or not coordinates:
        return []
    head = coordinates[0]
    if isinstance(head, (int, float)):
        return [coordinates]
    positions: List[Sequence[float]] = []
    for item in coordinates:
        positions.extend(_walk_coordinates(item))
    return positions


def geometry_bounds(geometry: dict) -> tuple:
    """(west, south, east, north) of a GeoJSON geometry."""
    positions = _walk_coordinates(geometry.get("coordinates", []))
    if not positions:
        raise StudyAreaError("geometry has no coordinates")
    xs = np.array([float(p[0]) for p in positions])
    ys = np.array([float(p[1]) for p in positions])
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def _validate_geometry(geometry: Any, crs: str) -> dict:
    """Structural validation. Nothing here repairs a broken boundary."""
    if not isinstance(geometry, dict):
        raise StudyAreaError(f"geometry must be a GeoJSON mapping, got "
                             f"{type(geometry).__name__}")
    kind = geometry.get("type")
    if kind not in _POLYGONAL:
        raise StudyAreaError(
            f"study-area geometry must be Polygon or MultiPolygon, got "
            f"{kind!r}; a point or line cannot define an analysis extent")
    positions = _walk_coordinates(geometry.get("coordinates", []))
    if len(positions) < 4:
        raise StudyAreaError(
            f"{kind} needs at least 4 positions to close a ring, got "
            f"{len(positions)}")
    values = np.array([[float(p[0]), float(p[1])] for p in positions])
    if not np.isfinite(values).all():
        raise StudyAreaError("geometry contains non-finite coordinates")
    west, south, east, north = geometry_bounds(geometry)
    if east <= west or north <= south:
        raise StudyAreaError(
            f"geometry has an empty extent: {west}..{east} x {south}..{north}")
    if _is_geographic(crs):
        if not (-180.0 <= west and east <= 180.0):
            raise StudyAreaError(
                f"longitudes {west}..{east} fall outside [-180, 180]; the "
                f"geometry is probably not in {crs}")
        if not (-90.0 <= south and north <= 90.0):
            raise StudyAreaError(
                f"latitudes {south}..{north} fall outside [-90, 90]; the "
                f"geometry is probably not in {crs}")
    return geometry


def _is_geographic(crs) -> bool:
    try:
        from rasterio.crs import CRS
        return bool(CRS.from_user_input(crs).is_geographic)
    except Exception:                                   # pragma: no cover
        return str(crs).upper().endswith("4326")


def _same_crs(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        from rasterio.crs import CRS
        return CRS.from_user_input(a) == CRS.from_user_input(b)
    except Exception:                                   # pragma: no cover
        return str(a) == str(b)


# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StudyArea:
    """A named analysis extent, supplied as configuration.

    `geometry` is a GeoJSON Polygon/MultiPolygon expressed in `crs`. The
    object is immutable: `to_crs` returns a new instance rather than
    reprojecting in place, so a boundary cannot silently change CRS midway
    through a run.
    """

    name: str
    geometry: dict
    crs: str = "EPSG:4326"
    source: str = "unspecified"
    #: Free-form provenance: where the polygon came from, its vintage, its
    #: licence, and any caveat about how exactly it delineates the area.
    attributes: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _validate_geometry(self.geometry, self.crs)
        if not str(self.name).strip():
            raise StudyAreaError("study area needs a name")

    # ------------------------------------------------------------ factories
    @classmethod
    def from_bounds(cls, west: float, south: float, east: float, north: float,
                    *, name: str, crs: str = "EPSG:4326",
                    source: str = "bounding box",
                    attributes: dict | None = None) -> "StudyArea":
        """Rectangular extent. Honest about being rectangular.

        A bounding box is a legitimate analysis extent but it is NOT an
        administrative boundary; anything reported from it covers the
        rectangle, not the district.
        """
        geometry = {"type": "Polygon",
                    "coordinates": [[[west, south], [east, south],
                                     [east, north], [west, north],
                                     [west, south]]]}
        provenance = {"geometry_kind": "bounding box",
                      "is_administrative_boundary": False}
        provenance.update(attributes or {})
        return cls(name=name, geometry=geometry, crs=crs, source=source,
                   attributes=provenance)

    @classmethod
    def from_geojson(cls, path, *, name: Optional[str] = None,
                     name_property: Optional[str] = None,
                     select: Optional[dict] = None) -> "StudyArea":
        """Read a boundary from a GeoJSON file.

        Accepts a bare geometry, a Feature, or a FeatureCollection. When a
        collection holds several features, `select` filters them by property
        (e.g. `{"district": "Karbi Anglong"}`) and the survivors are merged
        into one MultiPolygon. Merging is a coordinate concatenation, not a
        topological union: overlapping inputs stay overlapping, which
        rasterisation handles correctly because a pixel is inside if it is
        inside any part.
        """
        path = Path(path)
        if not path.exists():
            raise StudyAreaError(
                f"study-area boundary not found: {path}. Set "
                "`study_area.boundary` in the configuration to a GeoJSON "
                "polygon, or use a bounding box via `study_area.bounds`.")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise StudyAreaError(f"{path} is not valid JSON: {error}") from error

        crs = _crs_of_geojson(payload)
        features = _features_of(payload)
        if select:
            features = [f for f in features
                        if all(str(f.get("properties", {}).get(k)) == str(v)
                               for k, v in select.items())]
            if not features:
                raise StudyAreaError(
                    f"no feature in {path.name} matches {select}")
        if not features:
            raise StudyAreaError(f"{path.name} contains no features")

        geometry = _merge_geometries([f["geometry"] for f in features])
        properties = dict(features[0].get("properties") or {})
        resolved = (name or (properties.get(name_property)
                             if name_property else None)
                    or payload.get("name") or path.stem)
        attributes = {"n_features_merged": len(features),
                      "properties": properties,
                      "file": str(path)}
        attributes.update({k: v for k, v in payload.items()
                           if k in ("description", "provenance", "licence",
                                    "license", "vintage", "accessed",
                                    "is_administrative_boundary",
                                    "geometry_kind")})
        return cls(name=str(resolved), geometry=geometry, crs=crs,
                   source=str(path), attributes=attributes)

    @classmethod
    def from_vector(cls, path, **kwargs) -> "StudyArea":
        """Read a shapefile/GeoPackage via fiona when it is installed.

        GeoJSON needs no optional dependency and is the documented format;
        this exists so an authoritative district shapefile can be used
        directly in M7 without a manual conversion step.
        """
        path = Path(path)
        if path.suffix.lower() in (".geojson", ".json"):
            return cls.from_geojson(path, **kwargs)
        try:
            import fiona
        except ImportError as error:                    # pragma: no cover
            raise StudyAreaError(
                f"reading {path.suffix} boundaries needs `fiona` "
                "(pip install fiona), or convert the file to GeoJSON, which "
                "requires no extra dependency") from error
        select = kwargs.pop("select", None)
        name = kwargs.pop("name", None)
        name_property = kwargs.pop("name_property", None)
        with fiona.open(path) as collection:             # pragma: no cover
            crs = collection.crs_wkt or "EPSG:4326"
            records = [dict(record) for record in collection]
        if select:                                       # pragma: no cover
            records = [r for r in records
                       if all(str(r.get("properties", {}).get(k)) == str(v)
                              for k, v in select.items())]
        if not records:                                  # pragma: no cover
            raise StudyAreaError(f"no feature in {path} matches {select}")
        geometry = _merge_geometries(                    # pragma: no cover
            [r["geometry"] for r in records])
        properties = dict(records[0].get("properties") or {})
        resolved = (name or (properties.get(name_property)  # pragma: no cover
                             if name_property else None) or path.stem)
        return cls(name=str(resolved), geometry=geometry,   # pragma: no cover
                   crs=crs, source=str(path),
                   attributes={"n_features_merged": len(records),
                               "properties": properties, "file": str(path)})

    # ------------------------------------------------------------- geometry
    @property
    def bounds(self) -> tuple:
        """(west, south, east, north) in `crs`."""
        return geometry_bounds(self.geometry)

    def to_crs(self, crs) -> "StudyArea":
        """Reproject the boundary. Returns a new StudyArea."""
        if _same_crs(self.crs, crs):
            return self
        from rasterio.warp import transform_geom
        moved = transform_geom(self.crs, crs, self.geometry, precision=-1)
        attributes = dict(self.attributes)
        attributes["reprojected_from"] = str(self.crs)
        return StudyArea(name=self.name, geometry=moved, crs=str(crs),
                         source=self.source, attributes=attributes)

    def bounds_in(self, crs) -> tuple:
        """Bounds expressed in another CRS, without keeping the geometry."""
        return self.to_crs(crs).bounds

    # -------------------------------------------------------------- rasters
    def mask(self, georef: GeoRef, *, all_touched: bool = False) -> np.ndarray:
        """Boolean (H, W) grid: True for pixels inside the boundary.

        The boundary is reprojected onto the raster's CRS first, so a
        boundary in WGS84 correctly masks a grid in UTM.
        """
        from rasterio.features import geometry_mask
        if georef.crs is None:
            raise StudyAreaError(
                "cannot mask an ungeoreferenced grid: the target raster has "
                "no CRS, so there is no way to place the boundary on it")
        local = self.to_crs(georef.crs)
        inside = geometry_mask([local.geometry], out_shape=georef.shape,
                               transform=georef.transform, invert=True,
                               all_touched=all_touched)
        return np.asarray(inside, dtype=bool)

    def window(self, georef: GeoRef, *, pad: int = 0) -> tuple:
        """(row_start, row_stop, col_start, col_stop) covering the boundary.

        Clamped to the grid. Raises if the boundary and the raster do not
        overlap at all - that is a configuration error, not something to
        paper over with an empty array.
        """
        from rasterio.transform import rowcol
        local = self.to_crs(georef.crs) if georef.crs else self
        west, south, east, north = local.bounds
        rows, cols = rowcol(georef.transform, [west, east, west, east],
                            [north, north, south, south], op=math.floor)
        row_start = max(int(min(rows)) - pad, 0)
        row_stop = min(int(max(rows)) + 1 + pad, georef.height)
        col_start = max(int(min(cols)) - pad, 0)
        col_stop = min(int(max(cols)) + 1 + pad, georef.width)
        if row_stop <= row_start or col_stop <= col_start:
            raise StudyAreaError(
                f"study area {self.name!r} (bounds {local.bounds}) does not "
                f"overlap the raster grid (bounds {georef.bounds}); check "
                "that the boundary and the data describe the same place")
        return row_start, row_stop, col_start, col_stop

    def clip(self, cube: np.ndarray, georef: GeoRef, *, pad: int = 0,
             all_touched: bool = False, fill=np.nan) -> tuple:
        """Crop a (T, H, W) or (H, W) array to the boundary and mask outside.

        Returns (clipped, clipped_georef, inside_mask). Pixels outside the
        boundary become `fill` (NaN for floats) rather than a sentinel, which
        is the project's missing-data convention; the returned mask records
        which pixels those were.
        """
        from rasterio.transform import Affine
        array = np.asarray(cube)
        if array.shape[-2:] != georef.shape:
            raise StudyAreaError(
                f"array grid {array.shape[-2:]} does not match the "
                f"georeference {georef.shape}")
        row_start, row_stop, col_start, col_stop = self.window(georef, pad=pad)
        window = (slice(row_start, row_stop), slice(col_start, col_stop))
        clipped = array[(...,) + window]

        transform = georef.transform * Affine.translation(col_start, row_start)
        out_ref = GeoRef(georef.crs, transform, row_stop - row_start,
                         col_stop - col_start)
        inside = self.mask(out_ref, all_touched=all_touched)
        if fill is not None:
            # NaN cannot be stored in an integer array. Promoting to float is
            # the only way to keep "outside the boundary" distinguishable
            # from a real value, and silently substituting a sentinel would
            # break the project's missing-data convention.
            if np.issubdtype(clipped.dtype, np.integer) \
                    and not np.isfinite(fill):
                clipped = clipped.astype("float64")
            else:
                clipped = clipped.copy()
            clipped[..., ~inside] = fill
        return clipped, out_ref, inside

    def grid(self, resolution: float, *, crs=None,
             snap: bool = True) -> GeoRef:
        """A GeoRef covering the boundary at `resolution` units per pixel.

        Used to define an export/analysis grid from the boundary alone. With
        `snap` the origin is aligned to a whole multiple of the resolution,
        so two runs at the same resolution produce the same pixel grid even
        if the boundary is edited slightly.
        """
        from rasterio.transform import from_origin
        target = crs or self.crs
        west, south, east, north = self.bounds_in(target)
        if resolution <= 0:
            raise StudyAreaError(f"resolution must be positive, got {resolution}")
        if snap:
            west = math.floor(west / resolution) * resolution
            south = math.floor(south / resolution) * resolution
            east = math.ceil(east / resolution) * resolution
            north = math.ceil(north / resolution) * resolution
        width = max(int(round((east - west) / resolution)), 1)
        height = max(int(round((north - south) / resolution)), 1)
        return GeoRef(_crs_object(target), from_origin(west, north,
                                                       resolution, resolution),
                      height, width)

    # ------------------------------------------------------------ reporting
    def describe(self) -> dict:
        """Provenance record, saved with every real-data run."""
        west, south, east, north = self.bounds
        return {
            "name": self.name,
            "crs": str(self.crs),
            "source": self.source,
            "geometry_type": self.geometry.get("type"),
            "bounds": {"west": west, "south": south,
                       "east": east, "north": north},
            "n_positions": len(_walk_coordinates(self.geometry["coordinates"])),
            "attributes": dict(self.attributes),
            "note": ("The analytical pipeline is independent of this "
                     "boundary; changing it changes WHERE the framework is "
                     "applied, not HOW. Results are only valid for the area "
                     "actually processed."),
        }

    #: Provenance that must survive a save/load round trip at the TOP level,
    #: because `from_geojson` reads these from there. Without it a saved
    #: administrative polygon reloads with `geometry_kind` unset, and every
    #: downstream record then reports its provenance as unknown.
    ROUND_TRIP_KEYS = ("is_administrative_boundary", "geometry_kind",
                       "licence", "vintage", "provenance", "description")

    def save(self, path) -> Path:
        """Write the boundary back out as GeoJSON, for archiving with a run."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        properties = dict(self.attributes)
        nested = properties.get("properties")
        payload = {"type": "FeatureCollection",
                   "name": self.name,
                   "crs_of_coordinates": str(self.crs),
                   "source": self.source,
                   "features": [{"type": "Feature", "geometry": self.geometry,
                                 "properties": properties}]}
        for key in self.ROUND_TRIP_KEYS:
            value = properties.get(key)
            if value is None and isinstance(nested, dict):
                value = nested.get(key)
            if value is not None:
                payload[key] = value
        target.write_text(json.dumps(payload, indent=2, default=str))
        return target


# ---------------------------------------------------------------------------
def _crs_object(crs):
    try:
        from rasterio.crs import CRS
        return CRS.from_user_input(crs)
    except Exception:                                   # pragma: no cover
        return crs


def _crs_of_geojson(payload: dict) -> str:
    """RFC 7946 fixes GeoJSON at WGS84; honour a legacy `crs` member anyway."""
    legacy = payload.get("crs")
    if isinstance(legacy, dict):
        name = (legacy.get("properties") or {}).get("name")
        if name:
            return str(name)
    declared = payload.get("crs_of_coordinates")
    return str(declared) if declared else "EPSG:4326"


def _features_of(payload: dict) -> List[dict]:
    kind = payload.get("type")
    if kind == "FeatureCollection":
        return [f for f in payload.get("features", [])
                if isinstance(f, dict) and f.get("geometry")]
    if kind == "Feature":
        return [payload] if payload.get("geometry") else []
    if kind in _POLYGONAL:
        return [{"type": "Feature", "geometry": payload, "properties": {}}]
    raise StudyAreaError(
        f"unsupported GeoJSON type {kind!r}; expected FeatureCollection, "
        "Feature, Polygon or MultiPolygon")


def _merge_geometries(geometries: Sequence[dict]) -> dict:
    """Concatenate polygons into one geometry (see `from_geojson`)."""
    polygons: List[Any] = []
    for geometry in geometries:
        kind = geometry.get("type")
        if kind == "Polygon":
            polygons.append(geometry["coordinates"])
        elif kind == "MultiPolygon":
            polygons.extend(geometry["coordinates"])
        else:
            raise StudyAreaError(
                f"cannot merge geometry of type {kind!r}; study-area "
                "boundaries must be polygonal")
    if not polygons:
        raise StudyAreaError("no polygonal geometry to merge")
    if len(polygons) == 1:
        return {"type": "Polygon", "coordinates": polygons[0]}
    return {"type": "MultiPolygon", "coordinates": polygons}


def load_study_area(cfg) -> StudyArea:
    """Build the study area a `StudyAreaConfig` describes.

    Boundary file wins over bounds; supplying neither is an error, because
    the alternative is an implicit study area and that is exactly what this
    module exists to prevent.
    """
    boundary = getattr(cfg, "boundary", "") or ""
    bounds = list(getattr(cfg, "bounds", []) or [])
    name = getattr(cfg, "name", "") or "study_area"
    if boundary:
        return StudyArea.from_vector(
            boundary, name=name or None,
            name_property=getattr(cfg, "name_property", None) or None,
            select=dict(getattr(cfg, "select", {}) or {}) or None)
    if len(bounds) == 4:
        return StudyArea.from_bounds(
            *[float(v) for v in bounds], name=name,
            crs=getattr(cfg, "crs", "EPSG:4326") or "EPSG:4326",
            source="StudyAreaConfig.bounds",
            attributes={"note": "rectangular extent from configuration, not "
                                "an administrative boundary"})
    raise StudyAreaError(
        "no study area configured: set `study_area.boundary` to a polygon "
        "file or `study_area.bounds` to [west, south, east, north]. The "
        "pipeline will not invent an extent.")


# ---------------------------------------------------------------------------
# Area statistics (Part 26)
#
# Counting pixels and multiplying by a nominal 30 m x 30 m is wrong on a
# geographic grid, where a pixel's ground area shrinks as |latitude| grows.
# Both cases are handled explicitly and the method used is reported.
# ---------------------------------------------------------------------------
def pixel_area_km2(georef: GeoRef) -> np.ndarray:
    """Ground area of every pixel, in square kilometres, as an (H, W) grid.

    Projected CRS: the transform's pixel size, converted through the CRS's
    linear unit. Constant across the grid.

    Geographic CRS: the exact area of the spherical quadrangle bounded by
    the pixel's edges,

        A = R^2 * dlon * (sin(lat_top) - sin(lat_bottom)),

    on a sphere of the WGS84 authalic radius. This varies by row, which is
    the entire reason the function returns a grid rather than a scalar. The
    spherical approximation differs from the ellipsoidal area by <0.2% at
    any latitude, far below the uncertainty in the class assignment being
    summed.
    """
    height, width = georef.shape
    transform = georef.transform
    dx, dy = abs(transform.a), abs(transform.e)
    if georef.crs is not None and not _is_geographic(georef.crs):
        metres = _linear_units_metres(georef.crs)
        return np.full((height, width), (dx * metres) * (dy * metres) / 1e6)

    top = transform.f
    lat_edges = np.deg2rad(top + transform.e * np.arange(height + 1))
    strip = (EARTH_RADIUS_KM ** 2) * np.deg2rad(dx) \
        * np.abs(np.sin(lat_edges[:-1]) - np.sin(lat_edges[1:]))
    return np.repeat(strip[:, None], width, axis=1)


def _linear_units_metres(crs) -> float:
    try:
        from rasterio.crs import CRS
        factor = CRS.from_user_input(crs).linear_units_factor[1]
        return float(factor) if factor else 1.0
    except Exception:                                   # pragma: no cover
        return 1.0


def area_statistics(class_grid: np.ndarray, georef: GeoRef, *,
                    class_names: Optional[Dict[int, str]] = None,
                    valid_mask: Optional[np.ndarray] = None,
                    ignore_values: Sequence[int] = ()) -> "Any":
    """Area per class, computed from real pixel geometry.

    Returns a DataFrame with the pixel count, the area in km2, and the share
    of the analysed area, plus a `method` column recording how the area was
    obtained so a reader never has to guess.
    """
    import pandas as pd

    grid = np.asarray(class_grid)
    if grid.shape != georef.shape:
        raise StudyAreaError(f"class grid {grid.shape} does not match the "
                             f"georeference grid {georef.shape}")
    areas = pixel_area_km2(georef)
    finite = np.isfinite(grid.astype("float64"))
    valid = finite if valid_mask is None else (finite & np.asarray(valid_mask,
                                                                  bool))
    for value in ignore_values:
        valid &= grid != value
    geographic = georef.crs is not None and _is_geographic(georef.crs)
    method = ("spherical quadrangle area per pixel row (geographic CRS, "
              f"R={EARTH_RADIUS_KM} km)" if geographic
              else "constant projected pixel area from the affine transform")

    total = float(areas[valid].sum())
    rows = []
    for value in np.unique(grid[valid]):
        member = valid & (grid == value)
        area = float(areas[member].sum())
        rows.append({
            "class_value": int(value),
            "class_name": (class_names or {}).get(int(value), str(int(value))),
            "n_pixels": int(member.sum()),
            "area_km2": area,
            "fraction_of_analysed_area": area / total if total else float("nan"),
            "method": method,
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.attrs["total_analysed_area_km2"] = total
        frame.attrs["method"] = method
    return frame
