"""Centralized, serializable configuration.

Design notes (M1):
  * All tunable scientific thresholds live here, not in function defaults.
  * Config is a dataclass so it can be snapshotted to JSON alongside every
    experiment's results (reproducibility requirement).
  * No side effects on import. Directory creation is explicit, performed by
    `reproducibility.start_experiment`, so importing config in a test never
    touches the filesystem.

Backwards compatibility: module-level names (NDVI_STACK, YEARS, CLASSES,
SEED, ...) are preserved so existing callers keep working.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class PathConfig:
    root: str = str(ROOT)
    data: str = str(ROOT / "data")
    rasters: str = str(ROOT / "data" / "rasters")
    tables: str = str(ROOT / "data" / "tables")
    results: str = str(ROOT / "results")
    legacy_outputs: str = str(ROOT / "outputs")

    ndvi_stack: str = str(ROOT / "data" / "rasters" / "ndvi_stack.tif")
    rain_stack: str = str(ROOT / "data" / "rasters" / "rain_stack.tif")
    truth: str = str(ROOT / "data" / "rasters" / "truth_classes.tif")
    samples_csv: str = str(ROOT / "data" / "tables" / "training_samples.csv")


@dataclass
class QualityConfig:
    """Data-quality gating. Deliberately conservative: pixels failing these
    checks are FLAGGED and excluded, never silently interpolated."""
    min_valid_obs: int = 10
    max_missing_fraction: float = 0.4
    min_variance: float = 1e-8
    ndvi_min: float = -1.0
    ndvi_max: float = 1.0
    allow_interpolation: bool = False     # Part 5: off by default
    max_interpolation_gap: int = 2


@dataclass
class TrendConfig:
    alpha: float = 0.05
    apply_ties_correction: bool = True
    apply_autocorrelation_correction: bool = True
    hr_significance_level: float = 0.05
    hr_max_lag_fraction: float = 0.5
    min_obs: int = 10


@dataclass
class RestrendConfig:
    """RESTREND validity gating.

    RESTREND is interpretable as a climate-adjusted trend ONLY where the
    NDVI~rainfall regression is itself meaningful. Where it is not, the
    residual trend collapses toward the raw trend and must not be described
    as climate-corrected. We compute and expose an explicit validity flag.
    """
    alpha: float = 0.05
    min_r2: float = 0.10
    require_positive_beta: bool = True
    min_obs: int = 10


@dataclass
class CyclicityConfig:
    """Spectral periodicity detection.

    SCIENTIFIC FRAMING: this detects PERIODICITY in an NDVI series.
    Periodicity is consistent with, but is NOT proof of, any specific
    land-use practice (e.g. shifting cultivation). Attribution requires
    independent ground evidence.

    UNITS (corrected in M2): `periodicity_threshold` is compared against
    spectral ENRICHMENT (observed band-power fraction / the fraction
    expected under a flat-spectrum null), not against the raw band-power
    fraction. Enrichment is 1.0 for white noise regardless of band width or
    record length, so a threshold below 1.0 would flag noise as periodic.
    The previous default (0.35) was a raw-fraction value left over from the
    pre-enrichment scoring and labelled essentially every pixel periodic
    once `timeseries.cyclicity` switched to enrichment. 2.0 means "twice the
    band power expected under noise" and matches the default used by the
    estimator itself and by the M1/M2 test suites.
    """
    min_period: float = 4.0
    max_period: float = 12.0
    detrend: bool = True
    periodicity_threshold: float = 2.0
    min_obs: int = 12
    # Surrogate-data significance test (M3). Enrichment alone has no null
    # distribution attached to it: a threshold says "twice the flat-noise
    # level" but not "unlikely under a series with this autocorrelation".
    # Phase-randomised surrogates supply that null. Off by default because
    # it costs n_surrogates spectra per pixel; the research runner enables it
    # on the model sample.
    surrogate_test: bool = False
    n_surrogates: int = 199        # 199 -> p resolution of 0.005
    surrogate_alpha: float = 0.05
    surrogate_seed: int = 42


@dataclass
class BreakpointConfig:
    min_segment_length: int = 4
    min_variance_gain: float = 0.05
    significance_alpha: float = 0.05
    apply_significance_test: bool = True


@dataclass
class RecoveryConfig:
    """Post-disturbance recovery metrics (Part 4). Thresholds tunable."""
    recovery_threshold: float = 0.80
    min_disturbance_magnitude: float = 0.05
    min_post_obs: int = 3
    pre_window: int = 3


@dataclass
class ModelConfig:
    rf_trees: int = 500
    rf_min_samples_leaf: int = 2
    test_size: float = 0.30
    samples_per_class: int = 600
    cnn_epochs: int = 60
    cnn_lr: float = 1e-3
    cnn_batch: int = 64
    use_cnn: bool = True


# --------------------------------------------------------------------------
# M2 research configuration
#
# Everything M2 adds (feature groups, trajectory rules, spatial CV, model
# hyper-parameters, ablation design, sensitivity sweeps, temporal holdout)
# lives here so that it is snapshotted into results/<id>/config/config.json
# together with the M1 thresholds. No M2 default is hard-coded in a
# function signature.
# --------------------------------------------------------------------------
@dataclass
class FeatureConfig:
    """Which engineered feature groups a model may consume.

    Group names are defined once in `src.features.FEATURE_GROUPS`; the
    ordering here is the ordering of the produced design matrix.
    """
    groups: List[str] = field(default_factory=lambda: [
        "vegetation", "trend", "restrend", "cyclicity",
        "disturbance", "recovery", "rainfall"])
    require_finite: bool = False   # models impute inside the training fold


@dataclass
class TrajectoryConfig:
    """Rules for the analytical vegetation-trajectory categories.

    These are ANALYTICAL SIGNAL CATEGORIES, not verified land-cover
    classes. Every threshold below is a reporting convention and is
    documented in README.md; none of them is evidence of a land-use
    practice.

    SINGLE SOURCE OF TRUTH: `alpha`, `cyclicity_enrichment_threshold` and
    `min_disturbance_magnitude` mirror thresholds that already exist in the
    M1 sections (`trend.alpha`, `cyclicity.periodicity_threshold`,
    `recovery.min_disturbance_magnitude`). When a full `Config` is passed to
    `trajectory.classify_trajectories`, the M1 values win, so a parameter
    exists in exactly one place and a sensitivity sweep over it reaches
    every stage. The values here are used only when a bare
    `TrajectoryConfig` is passed directly.
    """
    alpha: float = 0.05                       # mirrors trend.alpha
    require_trend_significance: bool = True   # decline must be significant
    cyclicity_enrichment_threshold: float = 2.0   # x the white-noise level
    require_periodic_flag: bool = True        # AND the estimator's own flag
    recovery_fraction_threshold: float = 0.80  # share of the drop regained
    min_disturbance_magnitude: float = 0.05   # NDVI units, drop to qualify
    # A disturbance must survive the selection-adjusted Chow test before a
    # regained drop is called a recovery; otherwise ordinary noise in a
    # stable series is read as a disturbance-and-recovery cycle.
    require_significant_breakpoint: bool = True
    # Evaluated in this order; the first matching rule wins.
    priority: List[str] = field(default_factory=lambda: [
        "Recovering", "Degrading", "Cyclic", "Stable"])


@dataclass
class SpatialCVConfig:
    """Deterministic spatial block cross-validation.

    block_size    edge length of a square block, in pixels
    n_folds       number of geographically disjoint folds
    seed          block->fold assignment is a seeded permutation
    buffer_blocks blocks within this Chebyshev distance of a test block are
                  dropped from that fold's TRAINING set (0 = plain block CV;
                  >0 additionally separates train and test geographically)
    """
    block_size: int = 8
    n_folds: int = 5
    seed: int = 42
    buffer_blocks: int = 0


@dataclass
class RFExperimentConfig:
    """Random Forest research configuration (reproducible by construction)."""
    seed: int = 42
    n_estimators: int = 500
    min_samples_leaf: int = 2
    max_features: str = "sqrt"
    class_weight: str = "none"          # "none" or "balanced"
    imputation_strategy: str = "median"  # fitted on training folds only
    n_jobs: int = 1                      # 1 keeps tree order deterministic
    block_cv: SpatialCVConfig = field(default_factory=SpatialCVConfig)


@dataclass
class AblationExperiment:
    """One ablation cell: an id, a label and the feature groups it may use."""
    id: str
    name: str
    groups: List[str]


def _default_ablations() -> List[AblationExperiment]:
    """A--F: each experiment strictly contains the previous one."""
    veg = ["vegetation"]
    trend = veg + ["trend"]
    restrend = trend + ["restrend"]
    cyclic = restrend + ["cyclicity"]
    disturb = cyclic + ["disturbance", "recovery"]
    full = disturb + ["rainfall"]
    return [
        AblationExperiment("A", "A_basic_vegetation", veg),
        AblationExperiment("B", "B_basic_trend", trend),
        AblationExperiment("C", "C_add_restrend", restrend),
        AblationExperiment("D", "D_add_cyclicity", cyclic),
        AblationExperiment("E", "E_add_disturbance", disturb),
        AblationExperiment("F", "F_full", full),
    ]


@dataclass
class AblationConfig:
    experiments: List[AblationExperiment] = field(
        default_factory=_default_ablations)


@dataclass
class ParameterSweep:
    """One methodological parameter and the values to test it at."""
    path: str            # dotted path into Config, e.g. "trend.alpha"
    values: List[float]
    description: str = ""


def _default_sweeps() -> List[ParameterSweep]:
    return [
        ParameterSweep("trend.alpha", [0.01, 0.05, 0.10],
                       "Mann-Kendall significance level"),
        ParameterSweep("trend.min_obs", [8, 10, 14],
                       "minimum valid observations for a trend test"),
        ParameterSweep("restrend.min_r2", [0.05, 0.10, 0.20],
                       "minimum NDVI~rainfall partial r2 for a valid RESTREND"),
        ParameterSweep("cyclicity.min_period", [3.0, 4.0, 5.0],
                       "lower edge of the periodicity band (time steps)"),
        ParameterSweep("cyclicity.max_period", [10.0, 12.0, 14.0],
                       "upper edge of the periodicity band (time steps)"),
        ParameterSweep("cyclicity.periodicity_threshold", [1.5, 2.0, 3.0],
                       "spectral enrichment required to call a series cyclic"),
        ParameterSweep("breakpoint.min_variance_gain", [0.03, 0.05, 0.10],
                       "variance a two-segment fit must explain to be a break"),
        ParameterSweep("recovery.recovery_threshold", [0.70, 0.80, 0.90],
                       "share of pre-disturbance level counted as recovered"),
    ]


@dataclass
class SensitivityConfig:
    """Robustness sweeps over methodological parameters.

    Sweeps are reported, never optimised: the framework does not select a
    parameter value by comparing test-set metrics.
    """
    parameters: List[ParameterSweep] = field(default_factory=_default_sweeps)
    evaluate_model: bool = True
    n_estimators: int = 150      # cheaper model, sweeps are run many times
    enabled: bool = True


@dataclass
class UncertaintyConfig:
    """Thresholds for flagging a prediction as uncertain.

    Reporting conventions, not decision rules validated against ground
    truth. See `uncertainty.py` for what the measures do and do not mean.
    """
    confidence_threshold: float = 0.60   # max class probability
    margin_threshold: float = 0.15       # top probability minus runner-up


@dataclass
class CNNConfig:
    """1D CNN research experiment.

    The architecture stays deliberately small: the research question is
    whether a model learning directly from the raw sequence beats engineered
    temporal features, not whether a larger network wins. A bigger model
    would confound the comparison with capacity.
    """
    seed: int = 42
    channels: List[int] = field(default_factory=lambda: [32, 64])
    kernel_size: int = 5
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 80
    patience: int = 10                # early stopping on validation loss
    validation_folds: int = 1         # folds held out of training for early
    #                                   stopping, taken from the training
    #                                   folds only - never from the test fold
    spatial_cv: bool = True           # evaluate over every spatial fold
    max_folds: int = 0                # 0 = all folds; >0 caps the run length
    deterministic: bool = True
    enabled: bool = True


@dataclass
class ExplainConfig:
    """Explainability. Importance is association, never causation."""
    permutation_repeats: int = 5
    permutation_seed: int = 42
    top_n: int = 20
    shap: bool = True            # attempted only if the package is installed
    shap_max_samples: int = 200  # SHAP on trees is O(samples x trees)
    shap_seed: int = 42


@dataclass
class ExperimentMatrixConfig:
    """Which methods the final comparison runs, and on which task."""
    run_baseline: bool = True        # Mann-Kendall + Sen decision rule
    run_random_forest: bool = True
    run_cnn: bool = True
    run_proposed: bool = True        # full engineered feature framework
    #: Reference classes counted as degradation for the binary comparison.
    #: The conventional trend baseline can only answer "significant decline
    #: or not", so a like-for-like comparison has to be binary.
    degradation_classes: List[int] = field(default_factory=lambda: [4])
    buffer_sensitivity: List[int] = field(default_factory=lambda: [0, 1])


@dataclass
class TemporalHoldoutConfig:
    """Train on a historical window, evaluate on a later unseen window.

    cutoff_index wins if set; otherwise the cutoff is
    round(cutoff_fraction * T). Features for the historical experiment are
    built from the historical slice ONLY, so no future observation can
    enter a training feature.
    """
    cutoff_index: int = -1            # -1 = derive from cutoff_fraction
    cutoff_fraction: float = 0.65
    min_history: int = 12
    min_future: int = 12
    spatially_separated: bool = True  # also hold out whole spatial blocks
    enabled: bool = True


@dataclass
class ResearchConfig:
    """Everything the M2 research runner needs beyond the M1 estimators."""
    samples_per_class: int = 120
    #: Cap on analysed pixels (-1 = every pixel that passes quality gating).
    #: Thinning is a seeded, deterministic subsample, used for fast runs.
    max_analysis_pixels: int = -1
    features: FeatureConfig = field(default_factory=FeatureConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    spatial_cv: SpatialCVConfig = field(default_factory=SpatialCVConfig)
    model: RFExperimentConfig = field(default_factory=RFExperimentConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    holdout: TemporalHoldoutConfig = field(
        default_factory=TemporalHoldoutConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    matrix: ExperimentMatrixConfig = field(
        default_factory=ExperimentMatrixConfig)
    random_split_baseline: bool = True
    baseline_test_size: float = 0.30

    @classmethod
    def from_dict(cls, raw: dict) -> "ResearchConfig":
        raw = dict(raw)
        model = raw.get("model")
        if isinstance(model, dict):
            model = dict(model)
            if isinstance(model.get("block_cv"), dict):
                model["block_cv"] = SpatialCVConfig(**model["block_cv"])
            raw["model"] = RFExperimentConfig(**model)
        for key, klass in (("features", FeatureConfig),
                           ("trajectory", TrajectoryConfig),
                           ("spatial_cv", SpatialCVConfig),
                           ("holdout", TemporalHoldoutConfig),
                           ("uncertainty", UncertaintyConfig),
                           ("cnn", CNNConfig),
                           ("explain", ExplainConfig),
                           ("matrix", ExperimentMatrixConfig)):
            if isinstance(raw.get(key), dict):
                raw[key] = klass(**raw[key])
        if isinstance(raw.get("ablation"), dict):
            raw["ablation"] = AblationConfig(
                experiments=[AblationExperiment(**e)
                             for e in raw["ablation"]["experiments"]])
        if isinstance(raw.get("sensitivity"), dict):
            sens = dict(raw["sensitivity"])
            sens["parameters"] = [ParameterSweep(**p)
                                  for p in sens.get("parameters", [])]
            raw["sensitivity"] = SensitivityConfig(**sens)
        return cls(**raw)


@dataclass
class Config:
    experiment_name: str = "m1_baseline"
    seed: int = 42
    deterministic: bool = True
    years: List[int] = field(default_factory=lambda: list(range(1990, 2026)))

    paths: PathConfig = field(default_factory=PathConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    trend: TrendConfig = field(default_factory=TrendConfig)
    restrend: RestrendConfig = field(default_factory=RestrendConfig)
    cyclicity: CyclicityConfig = field(default_factory=CyclicityConfig)
    breakpoint: BreakpointConfig = field(default_factory=BreakpointConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    research: ResearchConfig = field(default_factory=ResearchConfig)

    classes: Dict[int, str] = field(default_factory=lambda: {
        1: "Stable high-NDVI",
        2: "Cropland / grassland",
        3: "Cyclic / periodic",
        4: "Declining",
        5: "Recovering",
    })
    class_colors: Dict[int, str] = field(default_factory=lambda: {
        1: "#1a7a3a", 2: "#c8b458", 3: "#e07b39",
        4: "#c0242b", 5: "#3a86c8",
    })

    def to_dict(self) -> dict:
        d = asdict(self)
        d["classes"] = {str(k): v for k, v in self.classes.items()}
        d["class_colors"] = {str(k): v for k, v in self.class_colors.items()}
        return d

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path

    @classmethod
    def load(cls, path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        sub = {"paths": PathConfig, "quality": QualityConfig,
               "trend": TrendConfig, "restrend": RestrendConfig,
               "cyclicity": CyclicityConfig, "breakpoint": BreakpointConfig,
               "recovery": RecoveryConfig, "model": ModelConfig}
        kw = {}
        for k, v in raw.items():
            if k == "research":
                kw[k] = ResearchConfig.from_dict(v)
            elif k in sub:
                kw[k] = sub[k](**v)
            elif k in ("classes", "class_colors"):
                kw[k] = {int(kk): vv for kk, vv in v.items()}
            else:
                kw[k] = v
        return cls(**kw)


DEFAULT = Config()

# --------------------------------------------------------------------------
# Backwards-compatible module-level aliases (existing callers rely on these)
# --------------------------------------------------------------------------
DATA = Path(DEFAULT.paths.data)
RASTERS = Path(DEFAULT.paths.rasters)
TABLES = Path(DEFAULT.paths.tables)
OUT = Path(DEFAULT.paths.legacy_outputs)
RESULTS = Path(DEFAULT.paths.results)

NDVI_STACK = Path(DEFAULT.paths.ndvi_stack)
RAIN_STACK = Path(DEFAULT.paths.rain_stack)
TRUTH = Path(DEFAULT.paths.truth)
SAMPLES_CSV = Path(DEFAULT.paths.samples_csv)

YEARS = DEFAULT.years
SEED = DEFAULT.seed
ALPHA = DEFAULT.trend.alpha
CLASSES = DEFAULT.classes
CLASS_COLORS = DEFAULT.class_colors
CYC_MIN_PERIOD = DEFAULT.cyclicity.min_period
CYC_MAX_PERIOD = DEFAULT.cyclicity.max_period
BREAK_MIN_IDX = DEFAULT.breakpoint.min_segment_length
BREAK_MAX_OFFSET = DEFAULT.breakpoint.min_segment_length
RF_TREES = DEFAULT.model.rf_trees
CNN_EPOCHS = DEFAULT.model.cnn_epochs
CNN_LR = DEFAULT.model.cnn_lr
