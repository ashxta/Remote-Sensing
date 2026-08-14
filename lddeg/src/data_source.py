"""Data contract between a DATA SOURCE and the analysis pipeline (M4).

Everything downstream of this module - quality gating, feature engineering,
trajectory classes, models, validation, exports - consumes a
`StandardizedDataset` and nothing else. Swapping the synthetic generator for
real Landsat / Sentinel / CHIRPS data in M6 therefore means writing one new
`DataSource` subclass, not touching the analytical pipeline.

    DataSource.load() -> StandardizedDataset -> [ analysis pipeline ]

THE CONTRACT
------------
A `StandardizedDataset` guarantees, and `validate()` enforces:

* `ndvi` and `rain` are float64 (T, H, W) arrays of identical shape;
* NDVI is in physical units, bounded by `QualityConfig.ndvi_min/max`;
* rainfall is non-negative, in consistent units across the record;
* missing observations are NaN - never a sentinel like -9999 or 0;
* the time axis is ordered, of known length, one entry per band;
* `georef` carries CRS, affine transform and grid size, and every exported
  raster inherits them unchanged;
* `truth` is either absent or an (H, W) integer label grid on the same grid;
* `metadata` records where the data came from and how it was processed.

Nothing here resamples, reprojects, gap-fills or rescales. A source is
responsible for delivering an analysis-ready cube; the pipeline is
responsible for not silently repairing one that is not. That division is
deliberate - it is what keeps a real-data mistake loud instead of quiet.

WHAT M6 MUST PROVIDE - see `REAL_DATA_REQUIREMENTS` for the executable list.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import rasterio

from .config import Config
from .dataset import DatasetValidationError, validate_dataset
from .geo import GeoRef

__all__ = ["StandardizedDataset", "DataSource", "RasterStackSource",
           "InMemorySource", "load_dataset", "REAL_DATA_REQUIREMENTS"]


#: What a real-data loader must satisfy before M6 can replace the synthetic
#: source. Written as data so it can be printed, tested and checked off.
REAL_DATA_REQUIREMENTS: tuple = (
    {"id": "analysis_ready_cube",
     "requirement": "Deliver NDVI and rainfall as aligned (T, H, W) cubes on "
                    "one grid, one CRS and one time axis.",
     "why": "The pipeline never reprojects or resamples; misalignment would "
            "silently pair a pixel with another pixel's rainfall."},
    {"id": "nan_for_missing",
     "requirement": "Encode every masked observation (cloud, shadow, snow, "
                    "SLC-off gap, sensor outage, scene edge) as NaN.",
     "why": "Sentinel values such as -9999 or 0 are valid numbers to a "
            "regression and would be fitted as real vegetation."},
    {"id": "physical_units",
     "requirement": "Apply the sensor scale factor so NDVI is in [-1, 1]; "
                    "state rainfall units and keep them constant.",
     "why": "Scaled integers (e.g. NDVI x 10000) pass every shape check and "
            "silently corrupt every threshold in the configuration."},
    {"id": "sensor_harmonisation",
     "requirement": "Harmonise across sensors (Landsat 5/7/8/9, Sentinel-2) "
                    "before stacking, and record the method used.",
     "why": "An unharmonised sensor change is a step in the series that "
            "breakpoint detection will report as a real disturbance."},
    {"id": "consistent_compositing",
     "requirement": "Use one compositing rule (e.g. annual post-monsoon "
                    "maximum-NDVI composite) for the whole record.",
     "why": "A changing composite window changes the phenological stage "
            "being measured, which appears as a trend."},
    {"id": "declared_time_axis",
     "requirement": "Provide the observation times, one per band, ordered, "
                    "with the interval documented.",
     "why": "Cyclicity periods and Sen slopes are expressed in time steps; "
            "an irregular axis makes both meaningless."},
    {"id": "study_area_boundary",
     "requirement": "Supply the study-area boundary as configuration (vector "
                    "file or bounding box), not hard-coded in the analysis.",
     "why": "No analytical function may assume a region; the pipeline must "
            "run over any area the loader supplies."},
    {"id": "quality_mask_provenance",
     "requirement": "Record which QA/cloud mask was applied and its version.",
     "why": "Missingness is a covariate: a mask change alters valid-"
            "observation counts and therefore trend detectability."},
    {"id": "independent_reference_labels",
     "requirement": "Obtain reference labels from ground or independent "
                    "ancillary data, never from the same NDVI series.",
     "why": "Labels derived from the features are circular; the classifier "
            "would be scored on reproducing its own inputs."},
    {"id": "rainfall_alignment",
     "requirement": "Resample CHIRPS (or equivalent) onto the NDVI grid and "
                    "aggregate to the same period, recording the method.",
     "why": "RESTREND regresses NDVI on rainfall pixel by pixel; a mismatched "
            "grid or accumulation window invalidates the correction."},
)


@dataclass
class StandardizedDataset:
    """An analysis-ready NDVI/rainfall cube with its spatial reference."""
    ndvi: np.ndarray                     # (T, H, W) float64, NaN = missing
    rain: np.ndarray                     # (T, H, W) float64, NaN = missing
    georef: GeoRef
    times: List[Any] = field(default_factory=list)
    truth: Optional[np.ndarray] = None   # (H, W) integer reference labels
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- geometry
    @property
    def n_time(self) -> int:
        return int(self.ndvi.shape[0])

    @property
    def shape(self) -> tuple:
        return (int(self.ndvi.shape[1]), int(self.ndvi.shape[2]))

    @property
    def n_pixels(self) -> int:
        height, width = self.shape
        return height * width

    def flat(self) -> tuple:
        """(T, N) views of the cubes, as every estimator expects them."""
        return (self.ndvi.reshape(self.n_time, -1),
                self.rain.reshape(self.n_time, -1))

    def flat_truth(self) -> Optional[np.ndarray]:
        return None if self.truth is None else self.truth.reshape(-1)

    def has_labels(self) -> bool:
        return self.truth is not None

    # ----------------------------------------------------------- validation
    def validate(self, cfg: Config | None = None, *, strict: bool = False,
                 require_georef: bool = True,
                 expected_time_steps: int | None = None) -> dict:
        """Enforce the contract; returns the descriptive metadata."""
        cfg = cfg or Config()
        report = validate_dataset(
            self.ndvi, self.rain, cfg=cfg, crs=self.georef.crs,
            transform=self.georef.transform, require_georef=require_georef,
            expected_time_steps=expected_time_steps, strict=strict)
        if self.times and len(self.times) != self.n_time:
            raise DatasetValidationError(
                f"{len(self.times)} time labels for {self.n_time} bands; the "
                "time axis and the data disagree")
        if self.truth is not None and self.truth.shape != self.shape:
            raise DatasetValidationError(
                f"reference labels {self.truth.shape} do not match the grid "
                f"{self.shape}")
        report["source"] = dict(self.metadata)
        report["n_time_labels"] = len(self.times)
        report["has_reference_labels"] = self.has_labels()
        return report

    def describe(self) -> dict:
        """Provenance summary, saved with every experiment."""
        return {"shape": [self.n_time, *self.shape],
                "n_pixels": self.n_pixels,
                "times": [str(t) for t in self.times],
                "has_reference_labels": self.has_labels(),
                "georeference": self.georef.to_dict(),
                "metadata": dict(self.metadata)}


class DataSource(ABC):
    """A provider of analysis-ready data.

    Implement one subclass per acquisition path. M6's real-data loader is a
    new subclass; nothing downstream changes.
    """

    #: Short identifier recorded in the dataset metadata.
    name: str = "abstract"

    @abstractmethod
    def load(self) -> StandardizedDataset:
        """Return an analysis-ready dataset satisfying the contract."""

    def load_validated(self, cfg: Config | None = None, *,
                       strict: bool = False,
                       expected_time_steps: int | None = None
                       ) -> tuple:
        """Load and immediately enforce the contract.

        Returns (dataset, validation_report). Every runner uses this rather
        than `load`, so no unvalidated array can reach the analysis.
        """
        dataset = self.load()
        report = dataset.validate(cfg, strict=strict,
                                  expected_time_steps=expected_time_steps)
        return dataset, report


class RasterStackSource(DataSource):
    """Multi-band GeoTIFF stacks on disk: the current development source.

    Honours each file's declared NoData - it is missing data, not a value -
    and reads the band descriptions as the time axis when they are present.
    The real-data loader in M6 will produce exactly this structure from
    Landsat/Sentinel and CHIRPS instead of from the synthetic generator.
    """

    name = "raster_stack"

    def __init__(self, ndvi_path, rain_path, truth_path=None, *,
                 times: Sequence[Any] | None = None,
                 description: str = "GeoTIFF stacks"):
        self.ndvi_path = Path(ndvi_path)
        self.rain_path = Path(rain_path)
        self.truth_path = Path(truth_path) if truth_path else None
        self.times = list(times) if times is not None else None
        self.description = description

    @classmethod
    def from_config(cls, cfg: Config) -> "RasterStackSource":
        return cls(cfg.paths.ndvi_stack, cfg.paths.rain_stack,
                   cfg.paths.truth, times=list(cfg.years),
                   description="synthetic development stacks")

    @staticmethod
    def _read(path: Path) -> tuple:
        """Read a stack, converting the declared NoData to NaN."""
        if not path.exists():
            raise FileNotFoundError(
                f"raster stack not found: {path}. Generate the synthetic "
                "development data with `python demo/make_synthetic_data.py`, "
                "or point PathConfig at real stacks.")
        with rasterio.open(path) as source:
            data = source.read().astype("float64")
            nodata = source.nodata
            descriptions = [d for d in (source.descriptions or []) if d]
        if nodata is not None:
            data[data == nodata] = np.nan
        return data, nodata, descriptions

    def load(self) -> StandardizedDataset:
        # Read the stacks first: `_read` raises the actionable "generate the
        # synthetic data" message, which is more useful than rasterio's
        # driver error from opening a missing file for the georeference.
        ndvi, ndvi_nodata, band_names = self._read(self.ndvi_path)
        rain, rain_nodata, _ = self._read(self.rain_path)
        if rain.shape != ndvi.shape:
            raise ValueError(
                f"NDVI stack {ndvi.shape} and rainfall stack {rain.shape} "
                "have different shapes; a source must deliver them aligned "
                "on one grid and one time axis")
        georef = GeoRef.from_raster(self.ndvi_path)

        truth = None
        if self.truth_path is not None and self.truth_path.exists():
            with rasterio.open(self.truth_path) as source:
                truth = source.read(1)

        times = self.times
        if times is None:
            times = band_names if len(band_names) == ndvi.shape[0] \
                else list(range(ndvi.shape[0]))
        return StandardizedDataset(
            ndvi=ndvi, rain=rain, georef=georef, times=list(times),
            truth=truth,
            metadata={"source": self.name, "description": self.description,
                      "ndvi_path": str(self.ndvi_path),
                      "rain_path": str(self.rain_path),
                      "truth_path": str(self.truth_path)
                      if self.truth_path else None,
                      "ndvi_nodata": ndvi_nodata, "rain_nodata": rain_nodata,
                      "synthetic": "synthetic" in self.description.lower()})


class InMemorySource(DataSource):
    """Arrays supplied directly; used by tests and by notebook experiments."""

    name = "in_memory"

    def __init__(self, ndvi, rain, georef, *, truth=None, times=None,
                 description: str = "in-memory arrays"):
        self.ndvi = np.asarray(ndvi, dtype="float64")
        self.rain = np.asarray(rain, dtype="float64")
        self.georef = georef
        self.truth = truth
        self.times = list(times) if times is not None else []
        self.description = description

    def load(self) -> StandardizedDataset:
        return StandardizedDataset(
            ndvi=self.ndvi, rain=self.rain, georef=self.georef,
            times=self.times, truth=self.truth,
            metadata={"source": self.name, "description": self.description,
                      "synthetic": True})


def load_dataset(cfg: Config, source: DataSource | None = None, *,
                 strict: bool = False) -> tuple:
    """Load and validate the dataset a configuration points at.

    The single entry point every runner uses. Passing an explicit `source`
    is how M6 will substitute the real-data loader without touching any
    runner.
    """
    source = source or RasterStackSource.from_config(cfg)
    return source.load_validated(cfg, strict=strict,
                                 expected_time_steps=len(cfg.years))


def requirements_table() -> list:
    """The real-data requirements, for printing or saving with a run."""
    return [dict(item) for item in REAL_DATA_REQUIREMENTS]


def save_requirements(path) -> Path:
    """Write the real-data contract next to an experiment's outputs."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(
        {"contract": "StandardizedDataset",
         "requirements": requirements_table(),
         "note": ("What an M6 real-data loader must satisfy. The analytical "
                  "pipeline consumes StandardizedDataset only, so meeting "
                  "this contract is sufficient to replace the synthetic "
                  "source without changing the analysis.")}, indent=2))
    return target
