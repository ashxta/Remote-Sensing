"""Sensor definitions, quality masking and spectral indices (M6 Parts 6-7).

Every sensor-specific fact the real-data pipeline needs lives here as data:
which physical band each product calls what, how to turn its stored integers
into reflectance, which QA bits mean cloud, and what a cross-sensor
harmonisation costs. Nothing downstream of `real_data.py` knows that
Landsat 5 calls red `SR_B3` and Landsat 8 calls it `SR_B4`.

Writing it as data rather than as branches has two consequences that matter
for the research:

* the transformations are printable, so a run records exactly which scale
  factor, mask and harmonisation produced its NDVI;
* the transformations are testable offline, on small integer arrays, with
  no network and no Earth Engine account. Every function below is exercised
  by `tests/test_m6_sensors.py`.

WHY HARMONISATION IS NOT OPTIONAL
---------------------------------
A record spanning Landsat 5, 7, 8 and 9 changes instrument in 2013. TM/ETM+
and OLI have different red and near-infrared band passes, so the same ground
target yields systematically different NDVI. Left uncorrected that step
appears in the series as a level shift at a known date - which is precisely
the pattern the breakpoint detector is designed to find. It would be
reported as a disturbance. The correction is applied at the NDVI level, and
`SENSORS[...].harmonisation` records which published coefficients were used.

REFERENCES
----------
Landsat Collection 2 Level-2 scaling and QA_PIXEL bit definitions:
    USGS, Landsat Collection 2 Level-2 Science Product Guide (LSDS-1619).
OLI -> ETM+ NDVI transform:
    Roy, D.P. et al. (2016), "Characterization of Landsat-7 to Landsat-8
    reflective wavelength and normalized difference vegetation index
    continuity", Remote Sensing of Environment 185:57-70.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

__all__ = ["Sensor", "SENSORS", "SensorError", "get_sensor", "sensor_table",
           "apply_scale_factors", "landsat_qa_mask", "sentinel2_scl_mask",
           "compute_index", "harmonise_ndvi", "INDEX_DEFINITIONS",
           "index_table", "LANDSAT_QA_BITS"]


class SensorError(ValueError):
    """Raised for an unknown sensor or an impossible band request."""


# ---------------------------------------------------------------------------
# Landsat Collection 2 QA_PIXEL bit assignments (LSDS-1619).
# Named rather than inlined so a mask can be described in a report.
# ---------------------------------------------------------------------------
LANDSAT_QA_BITS: Dict[str, int] = {
    "fill": 0,
    "dilated_cloud": 1,
    "cirrus": 2,            # OLI/TIRS only; reserved and unset on TM/ETM+
    "cloud": 3,
    "cloud_shadow": 4,
    "snow": 5,
    "clear": 6,
    "water": 7,
}

#: Bits excluded by default. Cirrus is included because thin cirrus depresses
#: red reflectance more than NIR and therefore inflates NDVI; water is NOT
#: excluded here, because whether water belongs in a vegetation study is a
#: study-design decision, made in configuration, not a sensor fact.
DEFAULT_LANDSAT_MASK_BITS = ("fill", "dilated_cloud", "cirrus", "cloud",
                             "cloud_shadow", "snow")

#: Sentinel-2 L2A scene classification (SCL) values considered unusable.
#: 0 no data, 1 saturated/defective, 3 cloud shadow, 8/9 cloud medium/high
#: probability, 10 thin cirrus, 11 snow/ice.
DEFAULT_S2_SCL_EXCLUDE = (0, 1, 3, 8, 9, 10, 11)


@dataclass(frozen=True)
class Sensor:
    """Everything product-specific about one satellite sensor."""

    key: str
    platform: str
    instrument: str
    collection: str                 # Earth Engine / archive collection id
    processing_level: str
    #: physical band -> product band name
    bands: Mapping[str, str]
    scale: float                    # stored value * scale + offset = reflectance
    offset: float
    native_resolution_m: float
    temporal_resolution_days: float
    coverage: str                   # operational period of the archive
    quality_band: str               # band carrying the per-pixel QA
    quality_scheme: str             # "landsat_qa_pixel" | "sentinel2_scl"
    saturation_band: Optional[str] = None
    #: NDVI harmonisation applied to bring this sensor onto the reference
    #: sensor's scale: NDVI_ref = gain * NDVI_this + bias. None means "no
    #: published transform is configured", which the loader refuses to
    #: silently treat as an identity when sensors are mixed.
    harmonisation: Optional[Dict[str, Any]] = None
    documentation: str = ""
    licence: str = ""
    notes: str = ""

    def band(self, physical: str) -> str:
        """Product band name for a physical band (`red`, `nir`, ...)."""
        try:
            return self.bands[physical]
        except KeyError:
            raise SensorError(
                f"{self.key} has no {physical!r} band; it provides "
                f"{sorted(self.bands)}") from None

    def to_reflectance(self, values: np.ndarray) -> np.ndarray:
        return apply_scale_factors(values, self.scale, self.offset)

    def describe(self) -> dict:
        """The Part-3 dataset record, as data."""
        return {
            "key": self.key, "platform": self.platform,
            "instrument": self.instrument, "collection": self.collection,
            "processing_level": self.processing_level,
            "bands": dict(self.bands),
            "reflectance_scale": self.scale,
            "reflectance_offset": self.offset,
            "spatial_resolution_m": self.native_resolution_m,
            "temporal_resolution_days": self.temporal_resolution_days,
            "temporal_coverage": self.coverage,
            "quality_band": self.quality_band,
            "quality_scheme": self.quality_scheme,
            "saturation_band": self.saturation_band,
            "ndvi_harmonisation": dict(self.harmonisation)
            if self.harmonisation else None,
            "documentation": self.documentation,
            "licence": self.licence,
            "notes": self.notes,
        }


# Roy et al. (2016) Table 6, OLS regression, OLI NDVI -> ETM+ NDVI. ETM+ is
# the reference sensor because it overlaps both TM (1999-2011) and OLI
# (2013-), so it is the only instrument with a direct empirical tie to both
# halves of the record.
_ROY_2016_OLI_TO_ETM = {
    "gain": 0.9589, "bias": 0.0029, "target": "ETM+ NDVI",
    "reference": ("Roy et al. (2016), Remote Sensing of Environment "
                  "185:57-70, Table 6 (OLS)"),
    "method": "linear NDVI transform"}

_LANDSAT_TM_BANDS = {"blue": "SR_B1", "green": "SR_B2", "red": "SR_B3",
                     "nir": "SR_B4", "swir1": "SR_B5", "swir2": "SR_B7"}
_LANDSAT_OLI_BANDS = {"coastal": "SR_B1", "blue": "SR_B2", "green": "SR_B3",
                      "red": "SR_B4", "nir": "SR_B5", "swir1": "SR_B6",
                      "swir2": "SR_B7"}

_C2_DOC = ("https://www.usgs.gov/landsat-missions/"
           "landsat-collection-2-surface-reflectance")
_C2_LICENCE = ("USGS Landsat data are in the public domain; free "
               "redistribution permitted with attribution.")

SENSORS: Dict[str, Sensor] = {
    "LANDSAT5_TM": Sensor(
        key="LANDSAT5_TM", platform="Landsat 5", instrument="TM",
        collection="LANDSAT/LT05/C02/T1_L2", processing_level="Level-2 SR",
        bands=_LANDSAT_TM_BANDS, scale=2.75e-05, offset=-0.2,
        native_resolution_m=30.0, temporal_resolution_days=16.0,
        coverage="1984-03 to 2012-05", quality_band="QA_PIXEL",
        quality_scheme="landsat_qa_pixel", saturation_band="QA_RADSAT",
        harmonisation={"gain": 1.0, "bias": 0.0, "target": "ETM+ NDVI",
                       "reference": "TM and ETM+ band passes are near-"
                                    "identical; no transform applied",
                       "method": "identity"},
        documentation=_C2_DOC, licence=_C2_LICENCE,
        notes="Decommissioned 2013; the archive ends 2012-05."),
    "LANDSAT7_ETM": Sensor(
        key="LANDSAT7_ETM", platform="Landsat 7", instrument="ETM+",
        collection="LANDSAT/LE07/C02/T1_L2", processing_level="Level-2 SR",
        bands=_LANDSAT_TM_BANDS, scale=2.75e-05, offset=-0.2,
        native_resolution_m=30.0, temporal_resolution_days=16.0,
        coverage="1999-05 to 2024-01", quality_band="QA_PIXEL",
        quality_scheme="landsat_qa_pixel", saturation_band="QA_RADSAT",
        harmonisation={"gain": 1.0, "bias": 0.0, "target": "ETM+ NDVI",
                       "reference": "reference sensor",
                       "method": "identity"},
        documentation=_C2_DOC, licence=_C2_LICENCE,
        notes=("Scan Line Corrector failed 2003-05-31: roughly 22% of every "
               "later scene is a systematic wedge-shaped gap. Those pixels "
               "are NoData, become NaN, and are counted as missing "
               "observations rather than filled.")),
    "LANDSAT8_OLI": Sensor(
        key="LANDSAT8_OLI", platform="Landsat 8", instrument="OLI/TIRS",
        collection="LANDSAT/LC08/C02/T1_L2", processing_level="Level-2 SR",
        bands=_LANDSAT_OLI_BANDS, scale=2.75e-05, offset=-0.2,
        native_resolution_m=30.0, temporal_resolution_days=16.0,
        coverage="2013-03 to present", quality_band="QA_PIXEL",
        quality_scheme="landsat_qa_pixel", saturation_band="QA_RADSAT",
        harmonisation=dict(_ROY_2016_OLI_TO_ETM),
        documentation=_C2_DOC, licence=_C2_LICENCE,
        notes="Narrower red and NIR band passes than TM/ETM+; harmonised."),
    "LANDSAT9_OLI2": Sensor(
        key="LANDSAT9_OLI2", platform="Landsat 9", instrument="OLI-2/TIRS-2",
        collection="LANDSAT/LC09/C02/T1_L2", processing_level="Level-2 SR",
        bands=_LANDSAT_OLI_BANDS, scale=2.75e-05, offset=-0.2,
        native_resolution_m=30.0, temporal_resolution_days=16.0,
        coverage="2021-10 to present", quality_band="QA_PIXEL",
        quality_scheme="landsat_qa_pixel", saturation_band="QA_RADSAT",
        harmonisation=dict(_ROY_2016_OLI_TO_ETM),
        documentation=_C2_DOC, licence=_C2_LICENCE,
        notes="Band passes match OLI; the OLI transform is applied."),
    "SENTINEL2_MSI": Sensor(
        key="SENTINEL2_MSI", platform="Sentinel-2 A/B", instrument="MSI",
        collection="COPERNICUS/S2_SR_HARMONIZED",
        processing_level="Level-2A BOA reflectance",
        bands={"blue": "B2", "green": "B3", "red": "B4", "nir": "B8",
               "rededge": "B5", "swir1": "B11", "swir2": "B12"},
        scale=1e-4, offset=0.0, native_resolution_m=10.0,
        temporal_resolution_days=5.0, coverage="2017-03 to present",
        quality_band="SCL", quality_scheme="sentinel2_scl",
        saturation_band=None, harmonisation=None,
        documentation=("https://sentinels.copernicus.eu/web/sentinel/"
                       "user-guides/sentinel-2-msi"),
        licence="Copernicus open and free data policy.",
        notes=("NOT part of the default configuration. Its 2017 start covers "
               "under a quarter of the 1990-2025 record, so including it "
               "would change the sensor mix partway through the series - the "
               "same defect harmonisation exists to prevent - while adding "
               "no information to the early period the trend tests depend "
               "on. `harmonisation` is deliberately None: mixing it with "
               "Landsat requires a published coefficient supplied in "
               "configuration, and the loader refuses to assume identity.")),
}


def get_sensor(key: str) -> Sensor:
    try:
        return SENSORS[str(key).upper()]
    except KeyError:
        raise SensorError(
            f"unknown sensor {key!r}; configured sensors are "
            f"{sorted(SENSORS)}") from None


def sensor_table() -> "Any":
    """One row per sensor, for the dataset-provenance record."""
    import pandas as pd
    return pd.DataFrame([s.describe() for s in SENSORS.values()])


# ---------------------------------------------------------------------------
# Reflectance
# ---------------------------------------------------------------------------
def apply_scale_factors(values, scale: float, offset: float, *,
                        valid_min: float = -0.2,
                        valid_max: float = 1.6) -> np.ndarray:
    """Stored integers -> surface reflectance, with impossible values NaN.

    Collection-2 Level-2 stores reflectance as unsigned 16-bit integers with
    a fill value of 0. Applying the scale factor to fill would produce
    -0.2 - a perfectly plausible-looking reflectance - so fill is removed
    BEFORE scaling, and the result is bounded to a physically possible range
    afterwards. Values outside it indicate a scaling or masking error and
    become NaN rather than entering an index.
    """
    raw = np.asarray(values, dtype="float64")
    scaled = np.where(raw == 0, np.nan, raw) * float(scale) + float(offset)
    return np.where((scaled >= valid_min) & (scaled <= valid_max),
                    scaled, np.nan)


# ---------------------------------------------------------------------------
# Quality masking
# ---------------------------------------------------------------------------
def landsat_qa_mask(qa_pixel, *, exclude: Sequence[str] = None,
                    saturation=None, cloud_confidence_max: int = 2,
                    strict_confidence: bool = False) -> np.ndarray:
    """Boolean (…, H, W) grid: True where the observation is USABLE.

    `exclude` names bits from `LANDSAT_QA_BITS`; any set bit rejects the
    pixel. `saturation` is the QA_RADSAT band, where any non-zero value means
    at least one band saturated. With `strict_confidence`, pixels whose
    two-bit cloud-confidence field (bits 8-9) exceeds `cloud_confidence_max`
    are rejected even if the cloud bit itself is clear - useful over bright
    surfaces where the cloud test is unreliable.

    Returning "usable" rather than "masked" is deliberate: every caller then
    writes `np.where(usable, value, np.nan)`, and the sense of the mask
    cannot be confused at the call site.
    """
    qa = np.asarray(qa_pixel)
    if not np.issubdtype(qa.dtype, np.integer):
        if not np.all(np.isfinite(qa)):
            raise SensorError("QA_PIXEL contains non-finite values; it must "
                              "be an integer bitmask band")
        qa = qa.astype("int64")
    bits = DEFAULT_LANDSAT_MASK_BITS if exclude is None else tuple(exclude)
    unknown = [b for b in bits if b not in LANDSAT_QA_BITS]
    if unknown:
        raise SensorError(f"unknown QA_PIXEL bit name(s) {unknown}; known "
                          f"names are {sorted(LANDSAT_QA_BITS)}")
    rejected = np.zeros(qa.shape, dtype=bool)
    for name in bits:
        rejected |= (qa & (1 << LANDSAT_QA_BITS[name])) != 0
    if strict_confidence:
        rejected |= ((qa >> 8) & 0b11) > int(cloud_confidence_max)
    if saturation is not None:
        rejected |= np.asarray(saturation).astype("int64") != 0
    return ~rejected


def sentinel2_scl_mask(scl, *, exclude: Sequence[int] = None) -> np.ndarray:
    """Boolean grid: True where the Sentinel-2 SCL class is usable."""
    values = np.asarray(scl).astype("int64")
    bad = DEFAULT_S2_SCL_EXCLUDE if exclude is None else tuple(exclude)
    return ~np.isin(values, list(bad))


# ---------------------------------------------------------------------------
# Spectral indices
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IndexDefinition:
    name: str
    formula: str
    bands: Sequence[str]
    valid_min: float
    valid_max: float
    purpose: str
    reference: str = ""

    def describe(self) -> dict:
        return {"name": self.name, "formula": self.formula,
                "bands": list(self.bands),
                "valid_range": [self.valid_min, self.valid_max],
                "purpose": self.purpose, "reference": self.reference}


INDEX_DEFINITIONS: Dict[str, IndexDefinition] = {
    "ndvi": IndexDefinition(
        name="NDVI", formula="(nir - red) / (nir + red)",
        bands=("nir", "red"), valid_min=-1.0, valid_max=1.0,
        purpose=("Primary vegetation-dynamics signal for this project. Every "
                 "M1-M5 estimator, threshold and feature is defined on NDVI."),
        reference="Rouse et al. (1974), NASA SP-351, 309-317"),
    "evi": IndexDefinition(
        name="EVI",
        formula="2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1)",
        bands=("nir", "red", "blue"), valid_min=-1.0, valid_max=1.0,
        purpose=("Optional. Less prone to saturation over dense canopy than "
                 "NDVI, but needs the blue band, which is the noisiest in "
                 "the atmospherically corrected product."),
        reference="Huete et al. (2002), Remote Sens. Environ. 83:195-213"),
    "nbr": IndexDefinition(
        name="NBR", formula="(nir - swir2) / (nir + swir2)",
        bands=("nir", "swir2"), valid_min=-1.0, valid_max=1.0,
        purpose=("Optional. Sensitive to burn and to abrupt canopy removal, "
                 "so it is the natural cross-check on a detected "
                 "disturbance."),
        reference="Key & Benson (2006), USGS FIREMON LA-1-55"),
    "ndmi": IndexDefinition(
        name="NDMI", formula="(nir - swir1) / (nir + swir1)",
        bands=("nir", "swir1"), valid_min=-1.0, valid_max=1.0,
        purpose=("Optional. Canopy moisture; separates a drought response "
                 "from a structural loss."),
        reference="Gao (1996), Remote Sens. Environ. 58:257-266"),
}


def index_table() -> "Any":
    import pandas as pd
    return pd.DataFrame([d.describe() for d in INDEX_DEFINITIONS.values()])


def compute_index(name: str, bands: Mapping[str, np.ndarray], *,
                  clip_to_valid: bool = True) -> np.ndarray:
    """Compute a spectral index from reflectance arrays.

    `bands` maps physical band names (`nir`, `red`, ...) to reflectance
    arrays. Division by a vanishing denominator yields NaN, not an
    astronomically large index value; results outside the index's physical
    range become NaN too, because an NDVI of 3 is a masking failure, not an
    observation.
    """
    key = str(name).lower()
    if key not in INDEX_DEFINITIONS:
        raise SensorError(f"unknown index {name!r}; available: "
                          f"{sorted(INDEX_DEFINITIONS)}")
    definition = INDEX_DEFINITIONS[key]
    missing = [b for b in definition.bands if b not in bands]
    if missing:
        raise SensorError(
            f"{definition.name} needs bands {list(definition.bands)}; "
            f"missing {missing}")
    arrays = {b: np.asarray(bands[b], dtype="float64")
              for b in definition.bands}

    with np.errstate(invalid="ignore", divide="ignore"):
        if key == "ndvi":
            num, den = arrays["nir"] - arrays["red"], arrays["nir"] + arrays["red"]
        elif key == "nbr":
            num, den = (arrays["nir"] - arrays["swir2"],
                        arrays["nir"] + arrays["swir2"])
        elif key == "ndmi":
            num, den = (arrays["nir"] - arrays["swir1"],
                        arrays["nir"] + arrays["swir1"])
        else:                                            # evi
            num = 2.5 * (arrays["nir"] - arrays["red"])
            den = (arrays["nir"] + 6.0 * arrays["red"]
                   - 7.5 * arrays["blue"] + 1.0)
        value = np.where(np.abs(den) < 1e-10, np.nan, num / den)

    if clip_to_valid:
        value = np.where((value >= definition.valid_min)
                         & (value <= definition.valid_max), value, np.nan)
    return value


def harmonise_ndvi(ndvi, sensor: Sensor, *,
                   override: Optional[Mapping[str, float]] = None
                   ) -> np.ndarray:
    """Bring one sensor's NDVI onto the reference sensor's scale.

    Raises when the sensor has no configured transform and none is supplied.
    Assuming identity is exactly the silent failure this module exists to
    prevent, so it is not the default.
    """
    coefficients = dict(override) if override else (
        dict(sensor.harmonisation) if sensor.harmonisation else None)
    if coefficients is None:
        raise SensorError(
            f"{sensor.key} has no NDVI harmonisation coefficients. Supply "
            "them in configuration (real_data.harmonisation_overrides) with "
            "a published reference, or exclude the sensor. Stacking an "
            "unharmonised sensor produces a step change at the instrument "
            "transition that breakpoint detection reports as a real "
            "disturbance.")
    values = np.asarray(ndvi, dtype="float64")
    return float(coefficients["gain"]) * values + float(coefficients["bias"])
