"""Reproducibility: seed control, experiment IDs, result directories,
logging, and configuration snapshots.

Every experiment writes:
    results/<experiment_id>/
        config/config.json        exact configuration used
        config/environment.json   package versions + platform
        metrics/                  JSON/CSV metrics
        figures/                  PNG figures
        predictions/              GeoTIFF / CSV predictions
        models/                   fitted model artefacts
        logs/run.log              full run log

so that any reported number can be traced to the configuration and code
state that produced it.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import Config

SUBDIRS = ("config", "metrics", "figures", "predictions", "models", "logs")


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG we may touch.

    Note: full bitwise determinism is achievable for numpy/random and for
    scikit-learn estimators given a fixed random_state. For PyTorch on GPU
    exact reproducibility is only guaranteed with deterministic algorithms
    enabled, which we request but which some ops do not support; this is
    documented rather than hidden.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # Only touch torch if it is ALREADY imported. Importing it here would
    # cost seconds on every call and pull a heavy dependency into tests that
    # do not need it.
    if "torch" not in sys.modules:
        return
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:
        pass


def get_rng(seed: int) -> np.random.Generator:
    """Preferred RNG for new code: an explicit Generator, not global state."""
    return np.random.default_rng(seed)


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "not-a-git-repo"
    except Exception:
        return "unavailable"


def environment_snapshot() -> dict:
    pkgs = {}
    for name in ("numpy", "scipy", "pandas", "sklearn", "rasterio",
                 "matplotlib", "torch", "pymannkendall"):
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pkgs[name] = "not installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "packages": pkgs,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


@dataclass
class Experiment:
    """Handle to one reproducible experiment run."""
    experiment_id: str
    root: Path
    config: Config
    logger: logging.Logger

    def path(self, subdir: str, filename: str = "") -> Path:
        if subdir not in SUBDIRS:
            raise ValueError(f"unknown subdir {subdir!r}; expected {SUBDIRS}")
        d = self.root / subdir
        d.mkdir(parents=True, exist_ok=True)
        return d / filename if filename else d

    def save_metrics(self, name: str, obj) -> Path:
        p = self.path("metrics", f"{name}.json")
        p.write_text(json.dumps(obj, indent=2, default=_json_default))
        self.logger.info("metrics written: %s", p.name)
        return p

    def save_table(self, name: str, df) -> Path:
        p = self.path("metrics", f"{name}.csv")
        df.to_csv(p)
        self.logger.info("table written: %s", p.name)
        return p

    def figure(self, name: str) -> Path:
        return self.path("figures", name)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def make_experiment_id(name: str, seed: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{name}_seed{seed}"


def _unique_root(results_root: Path, experiment_id: str) -> Path:
    """Never write two experiments into one directory.

    Timestamps have one-second resolution, so two runs started in the same
    second - which happens in tests and in scripted sweeps - would otherwise
    interleave their outputs. A suffix is appended instead, so each run's
    outputs stay isolated and attributable.
    """
    root = Path(results_root) / experiment_id
    if not root.exists():
        return root
    for suffix in range(2, 1000):
        candidate = Path(results_root) / f"{experiment_id}_run{suffix}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(                                 # pragma: no cover
        f"cannot allocate a unique result directory for {experiment_id}")


def start_experiment(config: Config | None = None,
                     experiment_id: str | None = None,
                     results_root: Path | None = None) -> Experiment:
    """Create the result tree, seed RNGs, snapshot config + environment."""
    config = config or Config()
    set_seed(config.seed, config.deterministic)

    eid = experiment_id or make_experiment_id(config.experiment_name,
                                              config.seed)
    root = _unique_root(Path(results_root or config.paths.results), eid)
    eid = root.name
    for sub in SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"lddeg.{eid}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                            datefmt="%H:%M:%S")
    fh = logging.FileHandler(root / "logs" / "run.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    config.save(root / "config" / "config.json")
    (root / "config" / "environment.json").write_text(
        json.dumps(environment_snapshot(), indent=2))

    logger.info("experiment %s started (seed=%d)", eid, config.seed)
    return Experiment(eid, root, config, logger)
