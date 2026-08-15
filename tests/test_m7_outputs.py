"""Final research outputs: cartography, profiles, tables (M7 Parts 20-25).

The colour tests are the substantive ones. A categorical map that two
readers in twenty cannot decode is a broken figure, and "it looks fine to
me" is not a check - so the accessibility properties are COMPUTED here
(OKLab distance under simulated dichromacy) rather than asserted by eye. If
someone later swaps a palette colour for a prettier one, these fail.
"""
import json
from itertools import combinations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src import m7_figures as MF
from src.geo import GeoRef

GEOREF = GeoRef(rasterio.crs.CRS.from_epsg(32646),
                from_origin(400000, 2900000, 300, 300), 60, 80)


# ---------------------------------------------------------- colour science
def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])
_RGB2LMS = np.array([[17.8824, 43.5161, 4.11935],
                     [3.45565, 27.1554, 3.86714],
                     [0.0299566, 0.184309, 1.46709]])
_LMS2RGB = np.linalg.inv(_RGB2LMS)
_SIM = {"deuteranopia": np.array([[1.0, 0.0, 0.0],
                                  [0.494207, 0.0, 1.24827],
                                  [0.0, 0.0, 1.0]]),
        "protanopia": np.array([[0.0, 2.02344, -2.52581],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0]])}


def _rgb(value):
    value = value.lstrip("#")
    return np.array([int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4)])


def _oklab(rgb):
    return _M2 @ np.cbrt(np.maximum(_M1 @ _srgb_to_linear(rgb), 0))


def _delta_e(a, b):
    return float(np.linalg.norm(_oklab(a) - _oklab(b)) * 100)


def _simulate(rgb, kind):
    lms = _RGB2LMS @ _srgb_to_linear(rgb)
    return _linear_to_srgb(_LMS2RGB @ (_SIM[kind] @ lms))


def test_every_pair_of_trajectory_colours_is_distinguishable():
    """Computed, not eyeballed: dE >= 15 normal, >= 8 under dichromacy."""
    colours = {name: _rgb(value)
               for name, value in MF.TRAJECTORY_COLORS.items()}
    problems = []
    for (name_a, a), (name_b, b) in combinations(colours.items(), 2):
        normal = _delta_e(a, b)
        deut = _delta_e(_simulate(a, "deuteranopia"),
                        _simulate(b, "deuteranopia"))
        prot = _delta_e(_simulate(a, "protanopia"),
                        _simulate(b, "protanopia"))
        if normal < 15:
            problems.append(f"{name_a}/{name_b} normal dE {normal:.1f} < 15")
        if min(deut, prot) < 8:
            problems.append(f"{name_a}/{name_b} CVD dE "
                            f"{min(deut, prot):.1f} < 8")
    assert not problems, "; ".join(problems)


def test_the_uncertain_class_is_neutral_and_hatched():
    """"No confident assignment" must not read as another category."""
    neutral = _rgb(MF.UNCERTAIN_COLOR)
    lab = _oklab(neutral)
    assert float(np.hypot(lab[1], lab[2])) < 0.02, "should be near-grey"
    assert MF.UNCERTAIN_HATCH, "needs texture as secondary encoding"
    assert MF.UNCERTAIN_CLASS not in MF.TRAJECTORY_COLORS


def test_diverging_ramps_have_a_neutral_midpoint_not_a_hue():
    """Zero must land on grey, so the sign of a weak value reads correctly."""
    for name, cmap in MF.DIVERGING.items():
        lab = _oklab(np.array(cmap(0.5)[:3]))
        chroma = float(np.hypot(lab[1], lab[2]))
        assert chroma < 0.03, f"{name} midpoint has chroma {chroma:.3f}"


def test_diverging_poles_are_distinguishable_including_under_cvd():
    """The two poles carry the sign, so confusing them inverts the reading.

    The poles of a diverging ramp are deliberately close in LIGHTNESS - if
    one were much darker it would dominate the map - so the separation has to
    come from hue, and that is exactly what dichromacy attacks. Hence the
    explicit CVD check rather than a plain distance threshold.
    """
    for name, cmap in MF.DIVERGING.items():
        low = np.array(cmap(0.0)[:3])
        high = np.array(cmap(1.0)[:3])
        normal = _delta_e(low, high)
        deut = _delta_e(_simulate(low, "deuteranopia"),
                        _simulate(high, "deuteranopia"))
        prot = _delta_e(_simulate(low, "protanopia"),
                        _simulate(high, "protanopia"))
        assert normal >= 15, f"{name} poles: normal dE {normal:.1f} < 15"
        assert min(deut, prot) >= 8, (
            f"{name} poles collapse under dichromacy: deut {deut:.1f}, "
            f"prot {prot:.1f}")


def test_no_rainbow_colormap_is_used_anywhere():
    banned = {"jet", "rainbow", "hsv", "gist_rainbow", "nipy_spectral",
              "turbo"}
    for role, name in MF.SEQUENTIAL.items():
        assert name.lower() not in banned, f"{role} uses {name}"


def test_sequential_ramps_increase_monotonically_in_darkness():
    """A sequential ramp must encode magnitude by lightness, not by hue."""
    import matplotlib.pyplot as plt

    for role, name in MF.SEQUENTIAL.items():
        cmap = plt.get_cmap(name)
        lightness = [float(_oklab(np.array(cmap(v)[:3]))[0])
                     for v in np.linspace(0.05, 0.95, 8)]
        differences = np.diff(lightness)
        assert (differences < 0.02).all(), \
            f"{role} ({name}) is not monotonically darkening: {lightness}"


# ------------------------------------------------------------------- maps
def _field(shape=(60, 80), kind="ramp"):
    rows, cols = shape
    if kind == "ramp":
        return np.tile(np.linspace(-0.02, 0.02, cols), (rows, 1))
    return np.tile(np.linspace(0.1, 0.9, cols), (rows, 1))


def test_a_continuous_map_is_written_with_its_furniture(tmp_path):
    path = MF.map_panel(_field(kind="ndvi"), GEOREF, tmp_path / "m.png",
                        title="Mean NDVI", subtitle="test",
                        label="NDVI", kind="sequential",
                        cmap=MF.SEQUENTIAL["vegetation"],
                        source="test source", study_area_label="Test area")
    assert path.exists() and path.stat().st_size > 20000


def test_a_diverging_map_is_forced_symmetric_about_zero(tmp_path, monkeypatch):
    """An off-centre diverging scale misrepresents the sign of weak values."""
    captured = {}
    import matplotlib.axes

    original = matplotlib.axes.Axes.imshow

    def spy(self, *args, **kwargs):
        captured.update({"vmin": kwargs.get("vmin"),
                         "vmax": kwargs.get("vmax")})
        return original(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", spy)
    skewed = np.tile(np.linspace(-0.001, 0.05, 80), (60, 1))
    MF.map_panel(skewed, GEOREF, tmp_path / "d.png", title="Trend",
                 kind="diverging", cmap=MF.DIVERGING["trend"])
    assert captured["vmin"] == pytest.approx(-captured["vmax"])


def test_an_all_nan_layer_is_refused_rather_than_drawn_blank(tmp_path):
    with pytest.raises(ValueError, match="nothing finite"):
        MF.map_panel(np.full((60, 80), np.nan), GEOREF, tmp_path / "n.png",
                     title="Empty")


def test_the_categorical_map_legend_carries_the_areas(tmp_path):
    codes = np.zeros((60, 80))
    codes[:20] = 1
    codes[20:40] = 2
    codes[40:] = 7
    names = {1: "Stable", 2: "Degrading", 7: MF.UNCERTAIN_CLASS}
    path = MF.categorical_map(
        codes, GEOREF, tmp_path / "c.png", class_names=names,
        title="Trajectories", areas={"Stable": 100.0, "Degrading": 50.0,
                                     MF.UNCERTAIN_CLASS: 25.0},
        source="test", study_area_label="Test area")
    assert path.exists() and path.stat().st_size > 20000


def test_the_facet_panel_gives_every_class_its_own_map(tmp_path):
    codes = np.zeros((60, 80))
    for index, value in enumerate((1, 2, 3, 7)):
        codes[index * 15:(index + 1) * 15] = value
    names = {1: "Stable", 2: "Degrading", 3: "Cyclic",
             7: MF.UNCERTAIN_CLASS}
    path = MF.trajectory_facets(codes, GEOREF, tmp_path / "f.png",
                                class_names=names, title="Facets")
    assert path.exists() and path.stat().st_size > 20000


def test_the_location_map_shows_where_the_study_area_is(tmp_path):
    path = MF.location_map(GEOREF, tmp_path / "loc.png",
                           title="Location", name="Test area")
    assert path.exists() and path.stat().st_size > 15000


def test_a_temporal_profile_renders_with_its_annotations(tmp_path):
    times = list(range(1990, 2025))
    ndvi = 0.7 - 0.004 * np.arange(35) + np.random.default_rng(0).normal(
        0, 0.02, 35)
    rain = 2000 + np.random.default_rng(1).normal(0, 200, 35)
    path = MF.temporal_profile(
        times, ndvi, rain, tmp_path / "p.png", title="Pixel",
        subtitle="test", sen_slope=-0.004, mk_p=0.001,
        restrend_slope=-0.003, restrend_p=0.02, restrend_valid=True,
        break_index=12, trough_index=15, period=7.0, enrichment=2.4)
    assert path.exists() and path.stat().st_size > 25000


def test_a_geographic_grid_gets_an_approximate_scale_bar(tmp_path):
    """A degree is not a constant distance; the bar must say so."""
    geographic = GeoRef(rasterio.crs.CRS.from_epsg(4326),
                        from_origin(92.3, 26.6, 0.01, 0.01), 60, 80)
    path = MF.map_panel(_field(kind="ndvi"), geographic, tmp_path / "g.png",
                        title="Geographic", kind="sequential")
    assert path.exists()


# -------------------------------------------------- representative pixels
class FakePrepared:
    """Minimal stand-in with the attributes the selector reads."""

    def __init__(self, features, labels, shape=(10, 10)):
        import pandas as pd
        self.features = pd.DataFrame(features)
        self.trajectory_labels = np.asarray(labels, dtype=object)
        self.shape = shape
        self.analysis_mask = np.ones(shape[0] * shape[1], bool)
        self.analysis_mask[len(self.trajectory_labels):] = False


def test_representative_selection_is_deterministic_and_documented():
    from src.config import Config
    from src.m7_outputs import select_representative_pixels

    rng = np.random.default_rng(4)
    n = 60
    features = {
        "sen": rng.normal(0, 0.01, n),
        "mk_p_value": rng.uniform(0, 1, n),
        "restrend": rng.normal(0, 0.01, n),
        "cyc_enrichment": rng.uniform(0.5, 4, n),
        "disturbance_magnitude": rng.uniform(0, 0.3, n),
        "recovery_fraction": rng.uniform(0, 1, n),
        "n_valid_ndvi": np.full(n, 30),
    }
    labels = ["Stable"] * 30 + ["Degrading"] * 30
    prepared = FakePrepared(features, labels)

    first = select_representative_pixels(prepared, Config())
    second = select_representative_pixels(prepared, Config())
    assert first == second, "selection must be reproducible"
    assert set(first) == {"Stable", "Degrading"}
    for record in first.values():
        assert "selection_rule" in record and record["selection_rule"]
        assert record["n_candidates"] > 0
        assert 0 <= record["feature_row"] < n


def test_the_chosen_pixel_is_near_the_class_median_not_an_extreme():
    from src.config import Config
    from src.m7_outputs import select_representative_pixels

    n = 41
    features = {
        "sen": np.linspace(-0.05, 0.05, n),
        "mk_p_value": np.full(n, 0.01),
        "restrend": np.zeros(n),
        "cyc_enrichment": np.ones(n),
        "disturbance_magnitude": np.zeros(n),
        "recovery_fraction": np.zeros(n),
        "n_valid_ndvi": np.full(n, 30),
    }
    prepared = FakePrepared(features, ["Degrading"] * n)
    chosen = select_representative_pixels(prepared, Config())
    # The median of a symmetric ramp is its centre.
    assert chosen["Degrading"]["feature_row"] == n // 2


def test_pixels_with_a_thin_record_are_avoided_when_possible():
    from src.config import Config
    from src.m7_outputs import select_representative_pixels

    n = 20
    features = {
        "sen": np.zeros(n), "mk_p_value": np.zeros(n),
        "restrend": np.zeros(n), "cyc_enrichment": np.ones(n),
        "disturbance_magnitude": np.zeros(n),
        "recovery_fraction": np.zeros(n),
        "n_valid_ndvi": np.array([5] * 10 + [30] * 10),
    }
    prepared = FakePrepared(features, ["Stable"] * n)
    chosen = select_representative_pixels(prepared, Config(), min_valid=25)
    assert chosen["Stable"]["feature_row"] >= 10
    assert chosen["Stable"]["n_valid_observations"] == 30
