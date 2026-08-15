# Real-data setup (M6/M7)

How to get genuine satellite and rainfall data into the pipeline. Three
routes are supported and they run **the same preprocessing code**; only the
place the arithmetic happens differs.

| | STAC (M7) | Local | Earth Engine |
|---|---|---|---|
| Account needed | **no** | no | yes (free, non-commercial) |
| Download volume | small (windowed COG reads) | large (raw scenes) | small (finished cubes) |
| Reproducible by a reader | yes, from the config | needs the same scene list | yes, from the config |
| Status in this repo | **used for the real M7 study** | exercised on fixture scenes | **not executed** |

---

## Route 0 (recommended): STAC, no credentials — what M7 actually used

**Microsoft Planetary Computer** hosts the USGS Landsat Collection 2 Level-2
archive as Cloud-Optimized GeoTIFFs behind an **anonymous** STAC API, and
issues read tokens without an account. The pixels are the USGS product — same
scene identifiers, same scale factors, same `QA_PIXEL` band — only the
delivery differs. CHIRPS annual rainfall comes from the UCSB public server the
same way.

```bash
python run_m7_acquire.py --config configs/m7_karbi_anglong_final.json \
                         --per-year 8 --workers 8
python run_m7_study.py    --config configs/m7_karbi_anglong_final.json
```

This acquired **264 real scenes across 1990–2024** (Landsat 5/7/8/9) plus 35
years of CHIRPS in about six minutes, with zero failures. It writes exactly
the `scenes.json` / `rainfall.json` manifests that
`real_data.preprocess_real_data` already reads, so **none of the M6
preprocessing changed**.

### Two things that make it fast enough to matter

**COG overview selection.** Warping a full-resolution 8031×6981 scene onto a
300 m grid made GDAL fetch every tile it touched — measured at **155 s per
scene**, i.e. 33 hours for the record. Reading the coarsest overview still
finer than the analysis grid (8× = 240 m) fetches roughly 1/64 of the bytes:
**2.4 s per scene**. `_choose_overview` never picks an overview *coarser* than
the target, which would upsample and invent detail the read never retrieved.

**Nearest-neighbour subsampling, not averaging** — a scientific choice, not a
performance one. Averaging reflectance to 300 m *before* the cloud mask is
applied would blend clear and cloudy native pixels into a value no later
masking could separate, and a QA bitmask cannot be averaged at all. Each
analysis cell is therefore one genuine 30 m observation with its own exact QA
flags — and says nothing about the other ~99 in its footprint.

### A bug worth knowing about

`QA_PIXEL` and `QA_RADSAT` use **opposite conventions**. Outside a scene
footprint the warp fills 0; for `QA_PIXEL` that is not a valid quality word
(no fill bit, no reject bit set), so it must be marked as fill. For
`QA_RADSAT`, 0 means *not saturated* — i.e. good. Applying the same
substitution to both marked every pixel saturated and produced a
**100 %-missing record** from 264 perfectly good scenes.
`tests/test_m7_acquisition.py` locks this so it cannot return.

---

## Status of the other two routes

`earthengine-api` is **not installed** in this development environment and
no Earth Engine credential exists here. The Earth Engine path in
[`src/gee_export.py`](../src/gee_export.py) has therefore **never been run
against the live service**. Its request-building logic is unit-tested
against a recording stub (`tests/test_m6_gee.py`), which shows that the
collections, band names, QA bits, scale factors, harmonisation coefficients
and date windows it *would* request are the intended ones. That is not the
same as a successful export, and nothing in this repository claims it is.

The **local** path has been exercised end to end, from raw per-scene
reflectance and QA bitmasks through to a validated `StandardizedDataset` and
a completed analysis run — using fabricated fixture scenes
(`demo/make_scene_fixture.py`), so the *code path* is proven, not any
result.

---

## 1. Study-area boundary

Nothing runs without one; the pipeline refuses to invent an extent.

```json
"study_area": {
  "name": "Karbi_Anglong",
  "boundary": "data/boundaries/karbi_anglong.geojson",
  "name_property": "NAME_2",
  "select": {"NAME_2": "Karbi Anglong"}
}
```

GeoJSON needs no optional dependency. Shapefiles and GeoPackages work when
`fiona` is installed. A bounding box is accepted as a fallback:

```json
"study_area": {"name": "Karbi_Anglong_bbox",
               "bounds": [92.30, 25.55, 93.85, 26.60]}
```

⚠️ A bounding box is **not** an administrative boundary. Area statistics
computed from one describe the rectangle. See
[`data/boundaries/README.md`](../data/boundaries/README.md) for the
authoritative sources to obtain before M7.

---

## 2A. Local route (no account)

### Download

From [USGS EarthExplorer](https://earthexplorer.usgs.gov/) or
[M2M](https://m2m.cr.usgs.gov/), request **Landsat Collection 2 Level-2
Surface Reflectance**, Tier 1, for your path/row and date range. Karbi
Anglong falls in WRS-2 paths 135–137, rows 42–43 — confirm against the
[WRS-2 shapefile](https://www.usgs.gov/landsat-missions/landsat-shapefiles-and-kml-files)
for your exact boundary.

You need, per scene:

| Sensor | Red | NIR | Quality | Saturation |
|---|---|---|---|---|
| Landsat 4/5 TM, 7 ETM+ | `SR_B3` | `SR_B4` | `QA_PIXEL` | `QA_RADSAT` |
| Landsat 8/9 OLI | `SR_B4` | `SR_B5` | `QA_PIXEL` | `QA_RADSAT` |

### Write a scene manifest

`data/raw/scenes.json`:

```json
{
  "scenes": [
    {
      "date": "2015-11-14",
      "sensor": "LANDSAT8_OLI",
      "scene_id": "LC08_L2SP_136042_20151114_20200908_02_T1",
      "bands": {
        "red": "data/raw/LC08_.../LC08_..._SR_B4.TIF",
        "nir": "data/raw/LC08_.../LC08_..._SR_B5.TIF"
      },
      "qa": "data/raw/LC08_.../LC08_..._QA_PIXEL.TIF",
      "saturation": "data/raw/LC08_.../LC08_..._QA_RADSAT.TIF",
      "scene_cloud_cover": 12.4
    }
  ],
  "metadata": {"source": "USGS EarthExplorer", "downloaded": "2026-01-15"}
}
```

Sensor keys: `LANDSAT5_TM`, `LANDSAT7_ETM`, `LANDSAT8_OLI`,
`LANDSAT9_OLI2`, `SENTINEL2_MSI`. `scene_cloud_cover` comes from the scene
metadata (`CLOUD_COVER` in the MTL file) and drives the scene-level
prefilter; omit it and no scene is prefiltered.

A stacked export works too — one file, band indices instead of paths:

```json
{"date": "2015-11-14", "sensor": "LANDSAT8_OLI",
 "file": "data/raw/scene.tif",
 "band_index": {"red": 4, "nir": 5, "QA_PIXEL": 8}}
```

### Rainfall

[CHIRPS 2.0](https://www.chc.ucsb.edu/data/chirps) daily or monthly, from
the [UCSB data server](https://data.chc.ucsb.edu/products/CHIRPS-2.0/).
Monthly is sufficient: the pipeline accumulates to the composite period
anyway, and monthly files are ~1/30 the volume. Stack them into one
multi-band raster and write `data/raw/rainfall.json`:

```json
{"file": "data/raw/chirps_monthly_1990_2025.tif",
 "dates": ["1990-01-15", "1990-02-15", "..."],
 "metadata": {"product": "CHIRPS-2.0 monthly", "units": "mm"}}
```

One date per band, in band order. CHIRPS is ~0.05° (~5.5 km) in WGS84; the
pipeline reprojects it onto the analysis grid with bilinear resampling and
records that it did.

### Run

```bash
python run_real_data.py --prepare --config configs/karbi_anglong.json
```

`--prepare` composites the scenes into cubes under
`real_data.composite_dir` and writes a provenance record. Subsequent runs
reuse the cache (`reuse_cache: true`).

---

## 2B. Earth Engine route

### One-time setup

```bash
# 1. Register (free for non-commercial use) and create a Cloud project
#    https://code.earthengine.google.com/register
# 2. Install the client
pip install earthengine-api
# 3. Authenticate — this stores a credential in your USER PROFILE
earthengine authenticate
# 4. Put the project id in the configuration
#    "real_data": {"backend": "gee", "gee_project": "your-project-id"}
```

The credential lives in `~/.config/earthengine` (Linux/macOS) or
`%USERPROFILE%\.config\earthengine` (Windows). **It must never be copied
into this repository.** Those paths, plus `*credentials*.json`,
`*service-account*.json`, `.env` and friends, are in `.gitignore`. A
project id is not a secret; a token is.

### Check the request before spending export quota

```bash
python run_real_data.py --export-plan --config configs/karbi_anglong.json
```

This prints and saves every parameter an export would use — study area,
collections, band names, masked QA bits, scale factors, harmonisation
coefficients, composite windows, target CRS and resolution — without
contacting Google. Review it, then:

```python
from src.gee_export import export_composites
from src.study_area import load_study_area
from src.config import Config

cfg = Config.load("configs/karbi_anglong.json")
export_composites(load_study_area(cfg.study_area), cfg.real_data)
```

Exports land in your Drive folder (`export_folder`). When they finish,
point `real_data.ndvi_cube` / `real_data.rain_cube` at the downloaded
GeoTIFFs and run `python run_real_data.py`.

**Scale.** 36 years × 30 m over a ~9000 km² district is on the order of
10⁷ pixels per band. Expect a multi-hour export and check
`max_export_pixels`. Consider a coarser `target_resolution_m` for a first
pass.

---

## 3. Reference labels — the thing that is actually blocking

Satellite imagery does not come with land-degradation labels. Without
independent reference data the runner completes every statistical and
unsupervised stage on real data and reports the **supervised** stages as
`BLOCKED`, with the reason written to
`metrics/supervised_blocked.json`.

**It will not fall back on the analytical trajectory classes.** Those are
computed by `trajectory.classify_trajectories` from the same engineered
features a classifier would consume — the Mann-Kendall p-value, the Sen
slope, the RESTREND residual trend, the spectral enrichment, the breakpoint
and the recovery fraction. Training on them measures whether a Random
Forest can re-derive a deterministic rule from that rule's own inputs. It
would score near-perfectly and mean nothing. The loader rejects any
`reference.provenance` containing `trajectory`, `algorithmic`, `derived`,
`pipeline`, `pseudo`, `self`, `ndvi` or `model`.

To unblock, configure genuinely independent labels:

```json
"reference": {
  "path": "data/reference/degradation_labels.tif",
  "validation_path": "data/reference/independent_validation.tif",
  "classes": {"1": "stable", "2": "cropland", "3": "cyclic",
              "4": "degrading", "5": "recovering"},
  "degradation_classes": [4],
  "source": "…dataset name, vintage, citation…",
  "provenance": "expert_interpretation",
  "resampling": "nearest"
}
```

`provenance` must be stated; an empty one is refused. Candidate sources, in
descending order of strength:

1. **Field observations** at located points.
2. **Expert photo-interpretation** of high-resolution imagery (Google Earth
   / Planet time series) at a stratified sample, by an interpreter blind to
   the model output.
3. **A published shifting-cultivation or degradation dataset** whose
   classification scheme, resolution and vintage are compatible.
4. **An authoritative land-cover product** for the confounder classes only,
   with its scheme mapped explicitly onto this study's classes.

Keep `validation_path` spatially disjoint from `path` where possible, so
the evaluation set is independent of the training set as well as of the
features.

---

## 4. Configuration reference

Everything below is in `real_data`.

| Key | Default | Note |
|---|---|---|
| `sensors` | L5, L7, L8, L9 | Sentinel-2 excluded by default: its 2017 start would change the sensor mix partway through the record |
| `harmonisation_reference` | `LANDSAT7_ETM` | ETM+ overlaps both TM and OLI, so it is the only instrument tied empirically to both halves |
| `harmonisation_overrides` | `{}` | Required for any sensor without a built-in transform |
| `start_year` / `end_year` | 1990 / 2025 | |
| `temporal_unit` | `annual` | `seasonal`/`monthly` available — see `src/compositing.py` for why they break the cyclicity analysis |
| `window_start` / `window_end` | `10-15` / `12-31` | Post-monsoon: least cloudy, peak biomass, same phenological stage each year |
| `composite_statistic` | `median` | **not** max-value compositing — see below |
| `min_observations_per_composite` | 1 | A single clear observation is an observation |
| `mask_bits` | fill, dilated_cloud, cirrus, cloud, cloud_shadow, snow | Water is *not* masked: that is a study-design decision |
| `max_scene_cloud_cover` | 80 | Scene-level prefilter, percent |
| `rainfall_product` | `UCSB-CHG/CHIRPS/DAILY` | |
| `rainfall_accumulation` | `hydrological_year` | The 12 months **ending** at the composite — the rain that grew what is observed |
| `target_crs` | `auto` | UTM zone of the centroid, so CV blocks are a constant ground distance |
| `target_resolution_m` | 30 | |
| `allow_interpolation` | `false` | When on, only interior gaps ≤ `max_interpolation_gap`, every filled value recorded |

### Two choices worth defending in a viva

**Median, not maximum-value compositing.** MVC was designed for coarse
AVHRR data with an unreliable cloud mask, where the maximum *was* the cloud
filter. With a per-pixel QA mask already applied, the maximum instead picks
the extreme of the within-window distribution — a bias that grows as the
number of surviving observations shrinks. Observation counts vary
systematically over a 36-year record (Landsat 7's 2003 SLC failure, the
2012–13 mission gap, the post-2021 improvement), so that bias becomes a
spurious **trend**.

**Hydrological-year rainfall accumulation.** For a post-monsoon composite,
a calendar-year total includes rain that fell *after* the vegetation was
observed — look-ahead in the covariate. `calendar_year` is retained for
comparison with the repository's earlier convention.

---

## 5. Troubleshooting

| Message | Meaning |
|---|---|
| `no study area configured … will not invent an extent` | Set `study_area.boundary` or `study_area.bounds` |
| `rainfall cube is not aligned with the reference grid` | Grids differ; run `--prepare` rather than pairing cubes by hand |
| `grids are offset by a FRACTION of a pixel` | Same CRS and resolution, half-pixel shift — every pixel would be paired with a neighbour |
| `temporal alignment failed … lags vegetation` | Rainfall bands are one period early |
| `<sensor> has no NDVI harmonisation coefficients` | Supply `harmonisation_overrides` with a citation, or drop the sensor |
| `reference provenance … indicates labels derived from the vegetation series` | Circular labels; see §3 |
| `every scene exceeds max_scene_cloud_cover` | Loosen the prefilter or widen the composite window |
| `no pixels passed the quality gate` | Too few valid observations; check `min_valid_obs` against the availability figure |

---

## 6. Testing without any of this

```bash
# Fabricate raw scenes and run the whole ingestion path offline
python demo/make_scene_fixture.py --out data/raw/fixture
python -m pytest tests/test_m6_pipeline.py -q
```

Fixture output is marked `synthetic` in the manifest, that flag is written
into the cubes' GeoTIFF tags, and `RealRemoteSensingSource` reads it back
and labels the dataset **SYNTHETIC FIXTURE** — through to the figure titles
and the run notice. The ingestion path is deliberately identical for real
and fixture input, so that marker is the only thing preventing a
mislabelling, and `tests/test_m6_pipeline.py` checks that it survives.
