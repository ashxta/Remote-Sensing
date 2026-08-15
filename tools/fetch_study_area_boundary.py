"""Fetch an authoritative study-area boundary (M7 correction, Part 2).

The M7 study ran on a BOUNDING BOX, which is not the district. This tool
replaces it with a real administrative polygon obtained from a documented,
openly-licensed source. It does not draw, simplify, smooth or approximate
anything: the geometry written is the geometry downloaded.

SOURCE
------
geoBoundaries (https://www.geoboundaries.org), release gbOpen, India ADM2.
Open Data Commons Open Database License (ODbL) 1.0, 2021 vintage. The
project is maintained by the William & Mary geoLab and is a standard,
citable source for administrative boundaries in the remote-sensing
literature.

It is NOT the Survey of India product. Survey of India is the definitive
national authority for Indian administrative boundaries, and if a licensed
SoI or Census of India district file is available it should be preferred -
`--from-file` accepts one directly. geoBoundaries is used here because it is
open, versioned, citable and reachable without a licence negotiation, and
because the alternative was continuing to use a rectangle.

THE 2016 DISTRICT SPLIT
-----------------------
Karbi Anglong was divided into Karbi Anglong (East) and West Karbi Anglong
in 2016. The study period is 1990-2024, which spans the split. The tool
therefore MERGES both successor districts by default, reconstructing the
undivided district that existed for most of the record. Using only one
successor would silently change the study area partway through the period
the data cover. `--districts` overrides this.

    python tools/fetch_study_area_boundary.py
    python tools/fetch_study_area_boundary.py --from-file survey_of_india.geojson
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SOURCE = {
    "name": "geoBoundaries gbOpen, India ADM2",
    "url": ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/"
            "releaseData/gbOpen/IND/ADM2/geoBoundaries-IND-ADM2.geojson"),
    "api": "https://www.geoboundaries.org/api/current/gbOpen/IND/ADM2/",
    "licence": "Open Data Commons Open Database License (ODbL) 1.0",
    "vintage": "2021",
    "citation": ("Runfola, D. et al. (2020). geoBoundaries: A global "
                 "database of political administrative boundaries. "
                 "PLoS ONE 15(4): e0231866."),
    "authority_note": (
        "geoBoundaries is an open, versioned, citable compilation. It is NOT "
        "the Survey of India product, which is the definitive national "
        "authority for Indian administrative boundaries. Any published map "
        "should state which boundary source was used."),
}

DEFAULT_DISTRICTS = ("Karbi Anglong East", "Karbi Anglong West")


def download(url: str, cache: Path) -> dict:
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            url, headers={"User-Agent": "land-degradation-research/1.0"})
        with urllib.request.urlopen(request, timeout=900) as response:
            cache.write_bytes(response.read())
    return json.loads(cache.read_text(encoding="utf-8"))


def merge(features: list) -> dict:
    """Combine selected districts into one MultiPolygon.

    Coordinate concatenation, not a topological union: the shared internal
    border between the two successor districts is retained as coincident
    edges. Rasterisation is unaffected - a pixel is inside if it lies inside
    any part - and no vertex is moved, which a true union would do.
    """
    polygons = []
    for feature in features:
        geometry = feature["geometry"]
        if geometry["type"] == "Polygon":
            polygons.append(geometry["coordinates"])
        elif geometry["type"] == "MultiPolygon":
            polygons.extend(geometry["coordinates"])
        else:
            raise SystemExit(f"unexpected geometry {geometry['type']}")
    if len(polygons) == 1:
        return {"type": "Polygon", "coordinates": polygons[0]}
    return {"type": "MultiPolygon", "coordinates": polygons}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/boundaries/karbi_anglong.geojson")
    parser.add_argument("--cache", default="data/raw/geoBoundaries-IND-ADM2.geojson")
    parser.add_argument("--districts", nargs="*", default=list(DEFAULT_DISTRICTS))
    parser.add_argument("--name", default="Karbi_Anglong")
    parser.add_argument("--from-file", default=None,
                        help="use a locally supplied authoritative file "
                             "instead of downloading (e.g. Survey of India)")
    parser.add_argument("--name-field", default="shapeName")
    args = parser.parse_args()

    if args.from_file:
        payload = json.loads(Path(args.from_file).read_text(encoding="utf-8"))
        provenance = {"name": f"locally supplied file: {args.from_file}",
                      "licence": "as supplied", "vintage": "as supplied",
                      "citation": "supplied by the researcher",
                      "authority_note": "provenance is the researcher's to "
                                        "state"}
    else:
        payload = download(SOURCE["url"], Path(args.cache))
        provenance = dict(SOURCE)

    wanted = {d.lower() for d in args.districts}
    selected = [f for f in payload["features"]
                if str(f["properties"].get(args.name_field, "")).lower()
                in wanted]
    found = [f["properties"].get(args.name_field) for f in selected]
    missing = wanted - {str(n).lower() for n in found}
    if missing:
        available = sorted({str(f["properties"].get(args.name_field))
                            for f in payload["features"]
                            if "karbi" in str(f["properties"]).lower()})
        raise SystemExit(
            f"districts not found: {sorted(missing)}. Candidates containing "
            f"'karbi': {available}")

    geometry = merge(selected)

    from src.study_area import StudyArea, pixel_area_km2   # noqa: E402
    area = StudyArea(
        name=args.name, geometry=geometry, crs="EPSG:4326",
        source=provenance["name"],
        attributes={
            "is_administrative_boundary": True,
            "geometry_kind": "administrative polygon",
            "districts_merged": found,
            "n_districts_merged": len(found),
            "merge_rationale": (
                "Karbi Anglong was divided into East and West districts in "
                "2016. The study period 1990-2024 spans that split, so both "
                "successors are merged to reconstruct the undivided district "
                "that existed for most of the record. Using one successor "
                "would change the study area partway through the period."),
            "source": provenance["name"],
            "source_url": provenance.get("url", ""),
            "licence": provenance["licence"],
            "vintage": provenance["vintage"],
            "citation": provenance["citation"],
            "authority_note": provenance["authority_note"],
            "modifications": "none; vertices are exactly as published",
        })

    target = Path(args.out)
    area.save(target)

    west, south, east, north = area.bounds
    # Ground area, for a sanity check against published district figures.
    grid = area.grid(0.002, crs="EPSG:4326")
    inside = area.mask(grid)
    km2 = float(pixel_area_km2(grid)[inside].sum())

    print(f"wrote {target}")
    print(f"  districts : {found}")
    print(f"  geometry  : {geometry['type']}")
    print(f"  bounds    : {west:.4f},{south:.4f} .. {east:.4f},{north:.4f}")
    print(f"  area      : {km2:,.0f} km2 (from the polygon, ~200 m raster)")
    print(f"  licence   : {provenance['licence']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
