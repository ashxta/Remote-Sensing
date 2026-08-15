"""Real-data acquisition over STAC (M7).

No test here touches the network. The archive interactions are driven
through stubs so the logic that decides WHICH scenes to fetch, at WHAT
resolution, and HOW their quality bands are interpreted is pinned down
offline.

The first test in the QA section is a regression test for a defect that
silently destroyed an entire 264-scene run: the out-of-footprint fill
substitution was applied to QA_RADSAT as well as QA_PIXEL, and because
QA_RADSAT uses the opposite convention (0 = good) that marked every pixel
saturated and produced a 100%-missing record.
"""
import datetime as dt
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src import stac_source
from src.geo import GeoRef
from src.stac_source import (CHIRPS, GDAL_HTTP_ENV, PLATFORM_TO_SENSOR,
                             StacError, StacItem, _choose_overview,
                             build_scene_cache, fetch_scene, search_landsat,
                             sign_href)
from src.study_area import StudyArea

AREA = StudyArea.from_bounds(92.3, 25.55, 93.85, 26.6, name="test_area")


def grid(resolution=300.0):
    return AREA.grid(resolution, crs="EPSG:32646")


# ------------------------------------------------------- overview selection
class FakeSource:
    def __init__(self, factors, resolution=30.0):
        self._factors = factors
        self.res = (resolution, resolution)

    def overviews(self, band):
        return self._factors


def test_the_coarsest_overview_still_finer_than_the_grid_is_chosen():
    """The decision that took a scene read from 155 s to 2.4 s."""
    source = FakeSource([2, 4, 8, 16, 32, 64])
    # 300 m target from 30 m native: 8x (240 m) qualifies, 16x (480 m) does not.
    assert _choose_overview(source, 300.0) == 2


def test_an_overview_coarser_than_the_target_is_never_selected():
    """Selecting one would upsample - inventing detail never retrieved."""
    factors = [2, 4, 8, 16]
    source = FakeSource(factors, resolution=30.0)
    level = _choose_overview(source, 100.0)
    # 2x = 60 m fits under 100 m; 4x = 120 m does not, so level 0 is the
    # coarsest admissible one.
    assert level == 0
    assert 30.0 * factors[level] <= 100.0
    assert 30.0 * factors[level + 1] > 100.0


def test_a_file_without_overviews_falls_back_to_full_resolution():
    assert _choose_overview(FakeSource([]), 300.0) is None


def test_a_target_finer_than_every_overview_uses_full_resolution():
    assert _choose_overview(FakeSource([2, 4]), 30.0) is None


# ------------------------------------------------------------------ signing
def test_a_non_archive_url_is_returned_untouched():
    assert sign_href("https://example.org/a.tif") == "https://example.org/a.tif"


def test_the_token_is_appended_and_cached(monkeypatch):
    calls = []

    def fake_request(url, payload=None, **kwargs):
        calls.append(url)
        return {"token": "sig=abc",
                "msft:expiry": (dt.datetime.now(dt.timezone.utc)
                                + dt.timedelta(hours=1)).isoformat()}

    monkeypatch.setattr(stac_source, "_request", fake_request)
    stac_source._TOKENS.clear()
    href = "https://landsateuwest.blob.core.windows.net/landsat-c2/x/y.tif"
    first = sign_href(href)
    second = sign_href(href)
    assert first.endswith("?sig=abc") and second == first
    assert len(calls) == 1, "the token should be cached, not re-requested"
    stac_source._TOKENS.clear()


# ------------------------------------------------------------------- search
def stac_feature(item_id, date, platform, cloud, *, with_qa=True,
                 with_red=True):
    assets = {}
    if with_red:
        assets["red"] = {"href": f"https://host/{item_id}_red.tif"}
    assets["nir08"] = {"href": f"https://host/{item_id}_nir.tif"}
    if with_qa:
        assets["qa_pixel"] = {"href": f"https://host/{item_id}_qa.tif"}
        assets["qa_radsat"] = {"href": f"https://host/{item_id}_sat.tif"}
    return {"id": item_id,
            "properties": {"datetime": f"{date}T04:00:00Z",
                           "platform": platform, "eo:cloud_cover": cloud},
            "assets": assets}


def stub_search(monkeypatch, features_by_year):
    def fake_request(url, payload=None, **kwargs):
        year = int(payload["datetime"][:4])
        return {"features": features_by_year.get(year, [])}

    monkeypatch.setattr(stac_source, "_request", fake_request)


def test_each_year_is_searched_separately(monkeypatch):
    """A busy year must not crowd a sparse one out of the page limit."""
    stub_search(monkeypatch, {
        2000: [stac_feature("a", "2000-11-01", "landsat-5", 10)],
        2001: [stac_feature("b", "2001-11-01", "landsat-7", 20)]})
    items = search_landsat(AREA, start_year=2000, end_year=2001)
    assert [i.item_id for i in items] == ["a", "b"]
    assert [i.sensor for i in items] == ["LANDSAT5_TM", "LANDSAT7_ETM"]


def test_the_per_year_cap_keeps_the_least_cloudy(monkeypatch):
    stub_search(monkeypatch, {2000: [
        stac_feature("cloudy", "2000-11-01", "landsat-5", 70),
        stac_feature("clear", "2000-11-02", "landsat-5", 3),
        stac_feature("middling", "2000-11-03", "landsat-5", 40)]})
    items = search_landsat(AREA, start_year=2000, end_year=2000, per_year=2)
    assert [i.item_id for i in items] == ["clear", "middling"]


def test_the_cap_is_applied_identically_to_every_year(monkeypatch):
    stub_search(monkeypatch, {
        2000: [stac_feature(f"a{i}", "2000-11-01", "landsat-5", i)
               for i in range(10)],
        2001: [stac_feature(f"b{i}", "2001-11-01", "landsat-5", i)
               for i in range(3)]})
    items = search_landsat(AREA, start_year=2000, end_year=2001, per_year=4)
    per_year = {}
    for item in items:
        per_year[item.date[:4]] = per_year.get(item.date[:4], 0) + 1
    assert per_year == {"2000": 4, "2001": 3}


def test_a_scene_without_a_quality_band_is_rejected(monkeypatch):
    """Compositing without a cloud mask would treat cloud tops as canopy."""
    stub_search(monkeypatch, {2000: [
        stac_feature("noqa", "2000-11-01", "landsat-5", 5, with_qa=False)]})
    assert search_landsat(AREA, start_year=2000, end_year=2000) == []


def test_a_scene_missing_a_required_band_is_rejected(monkeypatch):
    stub_search(monkeypatch, {2000: [
        stac_feature("nored", "2000-11-01", "landsat-5", 5, with_red=False)]})
    assert search_landsat(AREA, start_year=2000, end_year=2000) == []


def test_an_unknown_platform_is_skipped(monkeypatch):
    stub_search(monkeypatch, {2000: [
        stac_feature("modis", "2000-11-01", "terra", 5)]})
    assert search_landsat(AREA, start_year=2000, end_year=2000) == []


def test_the_platform_map_covers_every_landsat_in_the_study():
    for platform in ("landsat-5", "landsat-7", "landsat-8", "landsat-9"):
        assert platform in PLATFORM_TO_SENSOR


def test_sensor_filtering_is_honoured(monkeypatch):
    stub_search(monkeypatch, {2000: [
        stac_feature("tm", "2000-11-01", "landsat-5", 5),
        stac_feature("etm", "2000-11-02", "landsat-7", 5)]})
    items = search_landsat(AREA, start_year=2000, end_year=2000,
                           platforms=["LANDSAT7_ETM"])
    assert [i.item_id for i in items] == ["etm"]


# --------------------------------------------------- QA conventions (Part 6)
def test_only_qa_pixel_gets_the_fill_substitution(monkeypatch, tmp_path):
    """REGRESSION. QA_RADSAT's zero means "not saturated", i.e. GOOD.

    Applying QA_PIXEL's out-of-footprint fill substitution to it marks every
    pixel saturated, which masked an entire 264-scene real record to 100%
    missing before this was caught.
    """
    seen = {}

    def fake_read(href, target, *, mark_zero_as_fill, dtype="uint16"):
        name = href.rsplit("_", 1)[-1].split(".")[0]
        seen[name] = mark_zero_as_fill
        return np.ones(target.shape, dtype="uint16") * 5000

    monkeypatch.setattr(stac_source, "_read_onto_grid", fake_read)
    item = StacItem(item_id="scene", datetime="2000-11-01T00:00:00Z",
                    platform="landsat-5", sensor="LANDSAT5_TM",
                    cloud_cover=5.0,
                    assets={"red": "https://h/s_red.tif",
                            "nir": "https://h/s_nir.tif",
                            "qa": "https://h/s_qa.tif",
                            "saturation": "https://h/s_saturation.tif"})
    fetch_scene(item, grid(3000.0), tmp_path, overwrite=True)

    assert seen["qa"] is True, "QA_PIXEL needs the fill substitution"
    assert seen["saturation"] is False, (
        "QA_RADSAT must NOT get it: 0 means not saturated")
    assert seen["red"] is False and seen["nir"] is False


def test_a_saturation_band_of_zeros_masks_nothing():
    """The property the regression above violated, stated directly."""
    from src.sensors import LANDSAT_QA_BITS, landsat_qa_mask

    clear = np.full((4, 4), 1 << LANDSAT_QA_BITS["clear"], dtype="uint16")
    assert landsat_qa_mask(clear, saturation=np.zeros((4, 4))).all()
    assert not landsat_qa_mask(clear, saturation=np.ones((4, 4))).any()


# ------------------------------------------------------------ scene caching
def test_a_scene_that_misses_the_study_area_is_dropped(monkeypatch, tmp_path):
    monkeypatch.setattr(
        stac_source, "_read_onto_grid",
        lambda href, target, **kwargs: np.zeros(target.shape, dtype="uint16"))
    item = StacItem(item_id="empty", datetime="2000-11-01T00:00:00Z",
                    platform="landsat-5", sensor="LANDSAT5_TM",
                    cloud_cover=5.0,
                    assets={"red": "https://h/a_red.tif",
                            "nir": "https://h/a_nir.tif",
                            "qa": "https://h/a_qa.tif"})
    assert fetch_scene(item, grid(3000.0), tmp_path, overwrite=True) is None


def test_cached_scenes_are_reused_rather_than_refetched(monkeypatch, tmp_path):
    reads = []

    def fake_read(href, target, **kwargs):
        reads.append(href)
        return np.full(target.shape, 9000, dtype="uint16")

    monkeypatch.setattr(stac_source, "_read_onto_grid", fake_read)
    item = StacItem(item_id="scene", datetime="2000-11-01T00:00:00Z",
                    platform="landsat-5", sensor="LANDSAT5_TM",
                    cloud_cover=5.0,
                    assets={"red": "https://h/s_red.tif",
                            "nir": "https://h/s_nir.tif",
                            "qa": "https://h/s_qa.tif"})
    target = grid(3000.0)
    fetch_scene(item, target, tmp_path, overwrite=True)
    count = len(reads)
    record = fetch_scene(item, target, tmp_path)
    assert len(reads) == count, "a cached scene must not be refetched"
    assert record is not None and record["sensor"] == "LANDSAT5_TM"


def test_the_manifest_is_the_m6_format_and_declares_real_provenance(
        monkeypatch, tmp_path):
    monkeypatch.setattr(
        stac_source, "_read_onto_grid",
        lambda href, target, **kwargs: np.full(target.shape, 9000, "uint16"))
    items = [StacItem(item_id=f"s{i}", datetime=f"200{i}-11-01T00:00:00Z",
                      platform="landsat-5", sensor="LANDSAT5_TM",
                      cloud_cover=float(i),
                      assets={"red": f"https://h/s{i}_red.tif",
                              "nir": f"https://h/s{i}_nir.tif",
                              "qa": f"https://h/s{i}_qa.tif"})
             for i in range(3)]
    result = build_scene_cache(items, grid(3000.0), tmp_path, workers=2,
                               overwrite=True)
    payload = json.loads(result["manifest"].read_text())

    assert payload["metadata"]["synthetic"] is False
    assert payload["metadata"]["collection"] == "landsat-c2-l2"
    assert "USGS" in payload["metadata"]["provenance"]
    assert "subsampl" in payload["metadata"]["sampling"].lower()
    assert payload["metadata"]["n_cached"] == 3
    # The M6 loader must be able to read it unchanged.
    from src.real_data import load_manifest, manifest_metadata
    records = load_manifest(result["manifest"])
    assert len(records) == 3
    assert manifest_metadata(result["manifest"])["synthetic"] is False


def test_failures_are_recorded_rather_than_aborting_the_record(monkeypatch,
                                                              tmp_path):
    def flaky(href, target, **kwargs):
        if "s1" in href:
            raise RuntimeError("simulated timeout")
        return np.full(target.shape, 9000, dtype="uint16")

    monkeypatch.setattr(stac_source, "_read_onto_grid", flaky)
    items = [StacItem(item_id=f"s{i}", datetime=f"200{i}-11-01T00:00:00Z",
                      platform="landsat-5", sensor="LANDSAT5_TM",
                      cloud_cover=1.0,
                      assets={"red": f"https://h/s{i}_red.tif",
                              "nir": f"https://h/s{i}_nir.tif",
                              "qa": f"https://h/s{i}_qa.tif"})
             for i in range(3)]
    result = build_scene_cache(items, grid(3000.0), tmp_path, workers=1,
                               overwrite=True)
    assert result["n_failed"] == 1 and result["n_cached"] == 2
    payload = json.loads(result["manifest"].read_text())
    assert payload["metadata"]["failures"][0]["scene"] == "s1"


# ---------------------------------------------------------------- rainfall
def test_chirps_records_why_the_annual_product_was_used():
    assert "gzipped" in stac_source.CHIRPS_ANNUAL_RATIONALE
    assert "calendar" in stac_source.CHIRPS_ANNUAL_RATIONALE.lower()


def test_the_chirps_accumulation_caveat_states_the_ordering():
    caveat = stac_source.CHIRPS_ACCUMULATION_CAVEAT
    assert "monsoon" in caveat
    assert "not look-ahead" in caveat


def test_chirps_writes_the_m6_rainfall_manifest(monkeypatch, tmp_path):
    """The manifest must be exactly what `_read_rainfall` already reads."""
    class FakeReader:
        def __init__(self, *args, **kwargs):
            self.transform = from_origin(92.0, 27.0, 0.05, 0.05)
            self.crs = rasterio.crs.CRS.from_epsg(4326)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, band, window=None):
            return np.full((25, 35), 2000.0, dtype="float32")

        def window_transform(self, window):
            return from_origin(92.2, 26.7, 0.05, 0.05)

    real_open = rasterio.open

    def fake_open(path, mode="r", **kwargs):
        if str(path).startswith("/vsicurl/"):
            return FakeReader()
        return real_open(path, mode, **kwargs)

    monkeypatch.setattr(rasterio, "open", fake_open)
    result = stac_source.fetch_chirps_annual(AREA, tmp_path, start_year=2000,
                                             end_year=2002)
    payload = json.loads(result["manifest"].read_text())
    assert payload["file"].endswith(".tif")
    assert payload["dates"] == ["2000-07-01", "2001-07-01", "2002-07-01"]
    assert payload["metadata"]["synthetic"] is False
    assert payload["metadata"]["units"] == "mm"
    assert "Funk" in payload["metadata"]["citation"]

    from src.real_data import _read_rainfall
    cube, dates, georef = _read_rainfall(result["manifest"])
    assert cube.shape[0] == 3 and len(dates) == 3
    assert georef.crs is not None


def test_a_mid_year_date_falls_inside_the_calendar_accumulation_window():
    """The dates CHIRPS records must be usable by the M6 accumulator."""
    from src.compositing import annual_windows
    from src.real_data import rainfall_accumulation_windows

    windows = annual_windows(2000, 2000)
    (start, end), = rainfall_accumulation_windows(windows, "calendar_year")
    assert start <= dt.date(2000, 7, 1) <= end


def test_no_chirps_years_available_is_an_error(monkeypatch, tmp_path):
    def failing(path, *args, **kwargs):
        raise RuntimeError("404")

    monkeypatch.setattr(rasterio, "open", failing)
    with pytest.raises(StacError, match="1981"):
        stac_source.fetch_chirps_annual(AREA, tmp_path, start_year=1900,
                                        end_year=1901)


# ------------------------------------------------------------ configuration
def test_the_gdal_cache_option_is_an_int_not_a_string():
    """rasterio's config setter raises TypeError on a string here."""
    assert isinstance(GDAL_HTTP_ENV["GDAL_CACHEMAX"], int)
    with rasterio.Env(**GDAL_HTTP_ENV):
        pass


def test_the_subsampling_note_is_explicit_about_what_was_lost():
    note = stac_source.SUBSAMPLING_NOTE
    assert "NEAREST-NEIGHBOUR" in note
    assert "not by spatial averaging" in note
    assert "does not summarise" in note


def test_chirps_metadata_is_complete():
    for key in ("citation", "licence", "documentation", "units", "coverage"):
        assert CHIRPS[key]
