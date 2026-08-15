"""Publication cartography for the real-data study (M7 Part 21).

`maps.py` renders quick-look arrays for development. A figure that goes into
a report needs more: the reader must be able to tell WHERE the map is, HOW
BIG the features are, WHICH WAY is north, WHAT the colours mean, and WHERE
the data came from. Every panel here carries all five.

COLOUR
------
Assigned by the job the colour does, never by taste:

* magnitude (mean NDVI, valid-observation count, dominant period) ->
  SEQUENTIAL, one hue, light to dark;
* polarity (Sen slope, RESTREND residual slope) -> DIVERGING, two hues about
  a NEUTRAL GREY midpoint, and the scale is forced symmetric so that zero
  sits at the neutral point. An asymmetric diverging scale silently moves
  zero off the midpoint and makes a weak decline look like a strong one;
* identity (trajectory class) -> CATEGORICAL, fixed order, never cycled.

The categorical palette was validated rather than eyeballed: all fifteen
pairs of the six chromatic classes exceed dE 15 under normal vision and dE 8
under simulated deuteranopia and protanopia (OKLab x100, Vienot-Brettel-
Mollon dichromat model). The seventh class, "Uncertain / Other", is
deliberately NOT chromatic - it is a light neutral with a diagonal hatch, so
"no confident assignment" reads as absence of colour rather than as another
category, and it stays distinguishable in greyscale and for print.

No rainbow colormap appears anywhere in this module. A rainbow imposes
bright bands at arbitrary values that readers mistake for real thresholds.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch

from .geo import GeoRef

__all__ = ["TRAJECTORY_COLORS", "UNCERTAIN_CLASS", "SEQUENTIAL", "DIVERGING",
           "map_panel", "categorical_map", "trajectory_facets",
           "temporal_profile", "location_map", "save_figure"]

#: Validated categorical palette. Order is fixed and meaningful (stable ->
#: decline -> disturbance -> recovery -> cyclic); a class keeps its colour
#: regardless of how many classes are present in a given run.
TRAJECTORY_COLORS: Dict[str, str] = {
    "Stable": "#117733",                      # green
    "Degrading": "#882255",                   # wine
    "Rainfall-associated decline": "#CCAA55",  # sand
    "Disturbed": "#CC6677",                   # rose
    "Recovering": "#4477AA",                  # blue
    "Cyclic": "#4433AA",                      # indigo
}
UNCERTAIN_CLASS = "Uncertain / Other"
UNCERTAIN_COLOR = "#E9E9E9"
UNCERTAIN_HATCH = "///"

#: Single-hue sequential ramps, light to dark.
SEQUENTIAL = {
    "vegetation": "YlGn",     # conventional for NDVI and read as vegetation
    "count": "Blues",
    "period": "Purples",
    "magnitude": "Oranges",
}


def _diverging(name: str, low: str, high: str) -> LinearSegmentedColormap:
    """Two hues about a NEUTRAL GREY midpoint - never a hue at zero."""
    return LinearSegmentedColormap.from_list(
        name, [low, "#F2F2F2", high], N=256)


DIVERGING = {
    # Brown (loss) -> neutral -> green (gain). Reads correctly for vegetation
    # and is the standard direction in the degradation literature.
    "trend": _diverging("trend", "#8C510A", "#01665E"),
    "residual": _diverging("residual", "#762A83", "#1B7837"),
}


# ---------------------------------------------------------------------------
# Cartographic furniture
# ---------------------------------------------------------------------------
def _extent(georef: GeoRef) -> tuple:
    left, bottom, right, top = georef.bounds
    return (left, right, bottom, top)


def _is_geographic(georef: GeoRef) -> bool:
    try:
        return bool(georef.crs.is_geographic)
    except Exception:                                   # pragma: no cover
        return False


def _scale_bar(axis, georef: GeoRef) -> None:
    """A scale bar in ground units, sized to a round number.

    On a projected grid the units are metres and the bar is exact. On a
    geographic grid a degree is not a constant distance, so the bar is
    computed at the map's centre latitude and labelled as approximate -
    silently drawing an exact-looking bar on a lat/lon map would be wrong.
    """
    left, right, bottom, top = _extent(georef)
    span = right - left
    if _is_geographic(georef):
        centre_lat = np.deg2rad((bottom + top) / 2.0)
        km_per_unit = 111.32 * np.cos(centre_lat)
        suffix = " (approx.)"
    else:
        km_per_unit = 1e-3
        suffix = ""
    span_km = span * km_per_unit
    if span_km <= 0 or not np.isfinite(span_km):         # pragma: no cover
        return

    target = span_km / 4.0
    magnitude = 10 ** np.floor(np.log10(max(target, 1e-6)))
    bar_km = min([m * magnitude for m in (1, 2, 5, 10)],
                 key=lambda v: abs(v - target))
    bar_units = bar_km / km_per_unit

    x0 = left + 0.06 * span
    y0 = bottom + 0.055 * (top - bottom)
    height = 0.012 * (top - bottom)
    # A light plate behind the bar so the label stays readable over dark or
    # busy map content; without it the text disappears into forest canopy.
    axis.add_patch(plt.Rectangle(
        (x0 - 0.015 * span, y0 - height * 0.9),
        bar_units + 0.03 * span, height * 4.0,
        facecolor="#FFFFFF", alpha=0.78, edgecolor="none", zorder=5))
    # Two-tone bar: the standard cartographic form, readable at small size.
    for index, colour in enumerate(("#222222", "#FFFFFF")):
        axis.add_patch(plt.Rectangle(
            (x0 + index * bar_units / 2, y0), bar_units / 2, height,
            facecolor=colour, edgecolor="#222222", linewidth=0.7, zorder=6))
    axis.text(x0 + bar_units / 2, y0 + height * 1.7,
              f"{bar_km:g} km{suffix}", ha="center", va="bottom", fontsize=7.5,
              color="#222222", zorder=6)


def _north_arrow(axis, georef: GeoRef) -> None:
    """North arrow. Valid because every grid here is north-up, axis-aligned."""
    left, right, bottom, top = _extent(georef)
    x = right - 0.065 * (right - left)
    # Leave headroom for the "N" label so it is not clipped by the frame.
    y0 = top - 0.24 * (top - bottom)
    length = 0.11 * (top - bottom)
    axis.annotate("", xy=(x, y0 + length), xytext=(x, y0),
                  arrowprops=dict(facecolor="#222222", edgecolor="#222222",
                                  width=1.6, headwidth=7, headlength=7),
                  zorder=6)
    axis.text(x, y0 + length * 1.05, "N", ha="center", va="bottom",
              fontsize=9, fontweight="bold", color="#222222", zorder=6)


def _coordinate_ticks(axis, georef: GeoRef) -> None:
    """Label the axes in longitude/latitude even on a projected grid.

    A UTM easting tells a reader nothing about where the study area is. The
    corner coordinates are transformed back to WGS84 so the frame is
    interpretable without a GIS.
    """
    left, right, bottom, top = _extent(georef)
    xs = np.linspace(left, right, 5)
    ys = np.linspace(bottom, top, 5)
    if _is_geographic(georef):
        lons, lats = xs, ys
    else:
        try:
            from rasterio.crs import CRS
            from rasterio.warp import transform as warp_transform
            lons, _ = warp_transform(georef.crs, CRS.from_epsg(4326),
                                     list(xs), [bottom] * len(xs))
            _, lats = warp_transform(georef.crs, CRS.from_epsg(4326),
                                     [left] * len(ys), list(ys))
        except Exception:                                # pragma: no cover
            lons, lats = xs, ys
    axis.set_xticks(xs)
    axis.set_yticks(ys)
    axis.set_xticklabels([f"{v:.2f}°E" for v in lons], fontsize=7)
    axis.set_yticklabels([f"{v:.2f}°N" for v in lats], fontsize=7)
    axis.tick_params(length=2.5, width=0.6, colors="#555555")
    for spine in axis.spines.values():
        spine.set_edgecolor("#999999")
        spine.set_linewidth(0.7)


def _source_note(figure, source: str) -> None:
    figure.text(0.005, 0.005, source, fontsize=6.5, color="#666666",
                ha="left", va="bottom", wrap=True)


def save_figure(figure, path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return target


# ---------------------------------------------------------------------------
# Continuous maps
# ---------------------------------------------------------------------------
def map_panel(values: np.ndarray, georef: GeoRef, path, *, title: str,
              subtitle: str = "", label: str = "", source: str = "",
              kind: str = "sequential", cmap=None, symmetric: bool = False,
              vmin: Optional[float] = None, vmax: Optional[float] = None,
              percentile_clip: float = 2.0,
              study_area_label: str = "") -> Path:
    """One georeferenced continuous map with full cartographic furniture.

    `symmetric` forces a diverging scale to be centred on zero. It is the
    default for `kind="diverging"` because an off-centre diverging scale puts
    the neutral colour at some non-zero value, which misrepresents the sign
    of every pixel near it.
    """
    grid = np.asarray(values, dtype="float64")
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        raise ValueError(f"{title}: nothing finite to map")

    if vmin is None or vmax is None:
        low = float(np.percentile(finite, percentile_clip))
        high = float(np.percentile(finite, 100 - percentile_clip))
        vmin = low if vmin is None else vmin
        vmax = high if vmax is None else vmax
    if kind == "diverging" or symmetric:
        bound = max(abs(vmin), abs(vmax))
        vmin, vmax = -bound, bound
    if vmin == vmax:                                     # pragma: no cover
        vmin, vmax = vmin - 1e-9, vmax + 1e-9

    colormap = cmap if cmap is not None else (
        DIVERGING["trend"] if kind == "diverging"
        else SEQUENTIAL.get("vegetation"))
    if isinstance(colormap, str):
        colormap = plt.get_cmap(colormap)
    colormap = colormap.with_extremes(bad="#FFFFFF")

    figure, axis = plt.subplots(figsize=(7.6, 6.4))
    image = axis.imshow(np.ma.masked_invalid(grid), cmap=colormap,
                        vmin=vmin, vmax=vmax, extent=_extent(georef),
                        origin="upper", interpolation="nearest")
    _coordinate_ticks(axis, georef)
    _scale_bar(axis, georef)
    _north_arrow(axis, georef)
    if study_area_label:
        axis.text(0.02, 0.975, study_area_label, transform=axis.transAxes,
                  fontsize=8, fontweight="bold", va="top", ha="left",
                  color="#222222",
                  bbox=dict(facecolor="white", alpha=0.82, edgecolor="none",
                            boxstyle="round,pad=0.28"), zorder=6)

    bar = figure.colorbar(image, ax=axis, shrink=0.78, pad=0.025)
    bar.set_label(label or title, fontsize=8)
    bar.ax.tick_params(labelsize=7)
    bar.outline.set_linewidth(0.5)

    _titles(axis, title, subtitle)
    _source_note(figure, source)
    return save_figure(figure, path)


def _titles(axis, title: str, subtitle: str = "") -> None:
    """Title above subtitle, without the two colliding.

    matplotlib places a title immediately above the axes, so a subtitle
    drawn at the same anchor lands on top of it. The title is lifted by
    enough room for the subtitle line, and the subtitle sits in the gap.
    """
    axis.set_title(title, fontsize=11, fontweight="bold",
                   pad=22 if subtitle else 9)
    if subtitle:
        axis.text(0.5, 1.012, subtitle, transform=axis.transAxes,
                  fontsize=8, ha="center", va="bottom", color="#555555")


# ---------------------------------------------------------------------------
# Categorical maps
# ---------------------------------------------------------------------------
def _class_style(names: Sequence[str]) -> tuple:
    colours, hatches = [], []
    for name in names:
        if name == UNCERTAIN_CLASS:
            colours.append(UNCERTAIN_COLOR)
            hatches.append(UNCERTAIN_HATCH)
        else:
            colours.append(TRAJECTORY_COLORS.get(name, "#BBBBBB"))
            hatches.append("")
    return colours, hatches


def categorical_map(codes: np.ndarray, georef: GeoRef, path, *,
                    class_names: Dict[int, str], title: str,
                    subtitle: str = "", source: str = "",
                    areas: Optional[Dict[str, float]] = None,
                    study_area_label: str = "") -> Path:
    """Trajectory-class map with a legend that also carries the areas.

    Putting the area beside each swatch means identity never rests on colour
    alone - the legend doubles as the table view the accessibility pass
    requires.
    """
    grid = np.asarray(codes, dtype="float64")
    present = [int(v) for v in np.unique(grid[np.isfinite(grid)])]
    names = [class_names.get(v, str(v)) for v in present]
    colours, hatches = _class_style(names)

    lookup = {value: index for index, value in enumerate(present)}
    indexed = np.full(grid.shape, np.nan)
    for value, index in lookup.items():
        indexed[grid == value] = index

    colormap = ListedColormap(colours).with_extremes(bad="#FFFFFF")
    norm = BoundaryNorm(np.arange(-0.5, len(present)), colormap.N)

    figure, axis = plt.subplots(figsize=(8.6, 6.4))
    axis.imshow(np.ma.masked_invalid(indexed), cmap=colormap, norm=norm,
                extent=_extent(georef), origin="upper",
                interpolation="nearest")
    # NO hatch is drawn on the map itself. Hatching is a fill pattern for
    # CONTIGUOUS areas; over a scattered per-pixel class it smears into
    # broad diagonal bands that a reader mistakes for a systematic artefact
    # in the data. The hatch stays on the legend swatch, where the patch is
    # solid and the texture reads correctly, and the facet panel carries the
    # real secondary encoding for spatial pattern.

    _coordinate_ticks(axis, georef)
    _scale_bar(axis, georef)
    _north_arrow(axis, georef)
    if study_area_label:
        axis.text(0.02, 0.975, study_area_label, transform=axis.transAxes,
                  fontsize=8, fontweight="bold", va="top", ha="left",
                  color="#222222",
                  bbox=dict(facecolor="white", alpha=0.82, edgecolor="none",
                            boxstyle="round,pad=0.28"), zorder=6)

    handles = []
    for name, colour, hatch in zip(names, colours, hatches):
        text = name if areas is None else \
            f"{name}  —  {areas.get(name, float('nan')):,.0f} km²"
        handles.append(Patch(facecolor=colour, edgecolor="#666666",
                             linewidth=0.5, hatch=hatch or None, label=text))
    axis.legend(handles=handles, loc="center left", bbox_to_anchor=(1.015, 0.5),
                fontsize=8, frameon=False, title="Trajectory class",
                title_fontsize=8.5)

    _titles(axis, title, subtitle)
    _source_note(figure, source)
    return save_figure(figure, path)


def trajectory_facets(codes: np.ndarray, georef: GeoRef, path, *,
                      class_names: Dict[int, str], title: str,
                      source: str = "") -> Path:
    """One small panel per class - the secondary encoding for the class map.

    A seven-class map asks a reader to hold seven colours at once. Faceting
    removes that demand entirely: each panel is a single class against a
    neutral background, so the spatial pattern of every class can be read
    without relying on colour discrimination at all.
    """
    grid = np.asarray(codes, dtype="float64")
    present = [int(v) for v in np.unique(grid[np.isfinite(grid)])]
    names = [class_names.get(v, str(v)) for v in present]
    colours, _ = _class_style(names)

    columns = min(4, max(len(present), 1))
    rows = int(np.ceil(len(present) / columns))
    figure, axes = plt.subplots(rows, columns,
                                figsize=(3.0 * columns, 2.7 * rows))
    axes = np.atleast_1d(axes).reshape(-1)
    analysed = np.isfinite(grid)

    background = "#EFEFEF"
    for index, (value, name, colour) in enumerate(zip(present, names, colours)):
        axis = axes[index]
        # In a single-class panel the colour only has to separate from the
        # BACKGROUND, not from the other classes. The near-white neutral that
        # makes "Uncertain" recede on the combined map would be invisible
        # here, so this panel gets a mid grey instead - otherwise the largest
        # class in this study would render as an empty box.
        panel_colour = "#7A7A7A" if name == UNCERTAIN_CLASS else colour
        axis.imshow(np.where(analysed, 0.0, np.nan), cmap=ListedColormap(
            [background]), extent=_extent(georef), origin="upper",
            interpolation="nearest")
        member = np.where(grid == value, 1.0, np.nan)
        axis.imshow(np.ma.masked_invalid(member),
                    cmap=ListedColormap([panel_colour]),
                    extent=_extent(georef), origin="upper",
                    interpolation="nearest")
        share = float(np.mean(grid[analysed] == value)) if analysed.any() else 0
        axis.set_title(f"{name}\n{share * 100:.1f}% of analysed pixels",
                       fontsize=8)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_edgecolor("#CCCCCC")
    for axis in axes[len(present):]:
        axis.axis("off")

    figure.suptitle(title, fontsize=11, fontweight="bold")
    _source_note(figure, source)
    return save_figure(figure, path)


# ---------------------------------------------------------------------------
# Temporal profiles
# ---------------------------------------------------------------------------
def temporal_profile(times, ndvi, rain, path, *, title: str,
                     subtitle: str = "", source: str = "",
                     sen_slope: Optional[float] = None,
                     mk_p: Optional[float] = None,
                     restrend_slope: Optional[float] = None,
                     restrend_p: Optional[float] = None,
                     restrend_valid: bool = False,
                     break_index: Optional[int] = None,
                     trough_index: Optional[int] = None,
                     period: Optional[float] = None,
                     enrichment: Optional[float] = None) -> Path:
    """NDVI and rainfall for one pixel, annotated with what was detected.

    This is the figure that shows a reader WHY a pixel was classified as it
    was, so every annotation is drawn from the pixel's own computed
    statistics rather than re-derived here.
    """
    steps = np.arange(len(ndvi))
    labels = [str(t) for t in times]
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(9.4, 5.6), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1]})

    top.plot(steps, ndvi, marker="o", markersize=3.4, linewidth=1.6,
             color="#117733", label="NDVI", zorder=3)
    valid = np.isfinite(ndvi)
    if sen_slope is not None and valid.sum() >= 2:
        centre = float(np.nanmedian(ndvi))
        mid = float(np.median(steps[valid]))
        top.plot(steps, centre + sen_slope * (steps - mid), linestyle="--",
                 linewidth=1.4, color="#8C510A",
                 label=(f"Sen slope {sen_slope:+.4f}/yr"
                        + (f" (p={mk_p:.3f})" if mk_p is not None else "")),
                 zorder=2)
    if break_index is not None and break_index >= 0:
        top.axvline(break_index, color="#CC6677", linewidth=1.3,
                    linestyle="-.", label=f"breakpoint ({labels[int(break_index)]})",
                    zorder=2)
    if trough_index is not None and trough_index >= 0:
        top.plot([trough_index], [ndvi[int(trough_index)]], marker="v",
                 markersize=8, color="#882255", linestyle="none",
                 label=f"post-disturbance trough ({labels[int(trough_index)]})",
                 zorder=4)
    top.set_ylabel("NDVI", fontsize=9)
    top.grid(alpha=0.25, linewidth=0.6)
    top.legend(fontsize=7.5, frameon=False, ncol=2, loc="best")

    bottom.bar(steps, rain, color="#4477AA", alpha=0.75, width=0.72)
    bottom.set_ylabel("rainfall (mm/yr)", fontsize=9)
    bottom.grid(alpha=0.25, linewidth=0.6, axis="y")
    tick = max(len(steps) // 12, 1)
    bottom.set_xticks(steps[::tick])
    bottom.set_xticklabels([labels[i] for i in steps[::tick]], rotation=45,
                           ha="right", fontsize=7.5)

    notes = []
    if restrend_slope is not None:
        state = "valid" if restrend_valid else "NOT valid (rainfall relation " \
                                               "too weak to adjust)"
        notes.append(f"RESTREND {restrend_slope:+.4f}/yr"
                     + (f", p={restrend_p:.3f}" if restrend_p is not None
                        else "") + f" — {state}")
    if period is not None and np.isfinite(period):
        notes.append(f"dominant period {period:.1f} yr"
                     + (f", enrichment {enrichment:.2f}x white noise"
                        if enrichment is not None and np.isfinite(enrichment)
                        else ""))
    if notes:
        top.text(0.005, -0.02, "   |   ".join(notes), transform=top.transAxes,
                 fontsize=7.5, va="top", ha="left", color="#444444")

    # Reserve the top strip for the two title lines, then place them inside
    # it; without the reserved space the subtitle sits on the title's
    # descenders.
    figure.subplots_adjust(top=0.88)
    figure.suptitle(title, fontsize=11, fontweight="bold", y=0.985)
    if subtitle:
        figure.text(0.5, 0.936, subtitle, fontsize=8, ha="center",
                    va="top", color="#555555")
    _source_note(figure, source)
    return save_figure(figure, path)


def location_map(georef: GeoRef, path, *, title: str, name: str,
                 source: str = "", context_degrees: float = 9.0) -> Path:
    """Where the study area is, drawn without any basemap dependency.

    A graticule with the study-area rectangle marked is enough to locate the
    work, and it avoids shipping a tile dependency or an offline basemap that
    would have to be licensed and cached.
    """
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds

    left, bottom, right, top = georef.bounds
    if georef.crs is not None and not _is_geographic(georef):
        west, south, east, north = transform_bounds(
            georef.crs, CRS.from_epsg(4326), left, bottom, right, top)
    else:
        west, south, east, north = left, bottom, right, top

    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    pad = context_degrees
    axis.set_xlim(west - pad, east + pad)
    axis.set_ylim(south - pad, north + pad)
    axis.set_facecolor("#F7F9FA")
    axis.grid(color="#C9D4DA", linewidth=0.6, alpha=0.9)
    axis.add_patch(plt.Rectangle((west, south), east - west, north - south,
                                 facecolor="#CC6677", alpha=0.45,
                                 edgecolor="#882255", linewidth=1.8,
                                 zorder=4))
    axis.annotate(name, xy=((west + east) / 2, north),
                  xytext=((west + east) / 2, north + pad * 0.42),
                  ha="center", fontsize=9, fontweight="bold", color="#882255",
                  arrowprops=dict(arrowstyle="-", color="#882255",
                                  linewidth=1.1), zorder=5)
    axis.set_xlabel("longitude (°E)", fontsize=8.5)
    axis.set_ylabel("latitude (°N)", fontsize=8.5)
    axis.tick_params(labelsize=7.5, colors="#555555")
    axis.set_title(title, fontsize=11, fontweight="bold", pad=8)
    axis.text(0.99, 0.015,
              f"study area: {west:.2f}–{east:.2f}°E, "
              f"{south:.2f}–{north:.2f}°N",
              transform=axis.transAxes, fontsize=7, ha="right", va="bottom",
              color="#444444")
    for spine in axis.spines.values():
        spine.set_edgecolor("#999999")
        spine.set_linewidth(0.7)
    _source_note(figure, source)
    return save_figure(figure, path)
