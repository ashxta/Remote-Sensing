# Study-area boundaries

Boundaries are **data, not code**. The analytical pipeline (Mann-Kendall,
Sen's slope, RESTREND, cyclicity, breakpoint detection, recovery, feature
engineering, Random Forest, CNN, validation, visualisation) contains no
reference to any region. Pointing `study_area.boundary` at a different file
changes *where* the framework is applied, not *how*.

Changing the boundary does **not** mean the framework has been validated in
the new area. Only the area actually processed has been studied.

## Files

| File | What it is | Authoritative? |
|---|---|---|
| `karbi_anglong_bbox.geojson` | Rectangular extent, 92.30–93.85 °E, 25.55–26.60 °N | **No** — a bounding box |

## `karbi_anglong_bbox.geojson` — read this before using it

This is the rectangular extent that the repository's own Earth Engine
scripts (`gee/01_ndvi_rainfall_timeseries.py`, `gee/02_landtrendr.py`) have
used since M1, written out as a GeoJSON polygon so the pipeline can consume
it through the standard boundary interface instead of a hard-coded literal.

It is a **bounding box, not the administrative district boundary**. The
rectangle:

* includes land outside Karbi Anglong district on every side;
* does not follow the district's actual, highly irregular outline;
* has no authoritative provenance — it is an approximation someone drew
  around the district.

Consequences, which must appear in any write-up that uses it:

* **Area statistics are for the rectangle**, not for the district. Never
  report "X km² of Karbi Anglong is degrading" from a bounding-box run.
* Pixels outside the district are analysed and included in every summary.
* Comparisons with published district-level figures are not like-for-like.

## Required before M7

Obtain the authoritative district polygon and place it here. Candidates,
in descending order of authority for an Indian district:

1. **Survey of India** administrative boundaries (the national mapping
   agency; definitive, licence terms apply).
2. **Census of India** district boundary files for the relevant census year
   — note that Karbi Anglong was split into East and West Karbi Anglong
   districts in 2016, so the vintage of the boundary matters and must be
   stated.
3. **GADM** (https://gadm.org) level-2 — convenient, widely used in the
   literature, but derived and not authoritative; acceptable if cited as
   such.
4. **FAO GAUL** / **geoBoundaries** — similar standing to GADM.

Then set:

```json
"study_area": {
  "name": "Karbi_Anglong",
  "boundary": "data/boundaries/karbi_anglong.geojson",
  "name_property": "NAME_2",
  "select": {"NAME_2": "Karbi Anglong"}
}
```

The loader accepts a bare geometry, a Feature, or a FeatureCollection, and
`select` filters a multi-district file down to the one wanted. Shapefiles
and GeoPackages are read too when `fiona` is installed; converting to
GeoJSON avoids that dependency entirely.

Record the source, vintage and licence in the file's `properties` — the
loader carries them into every run's provenance record.
