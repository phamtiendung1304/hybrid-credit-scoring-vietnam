"""
Configuration Loader
====================
Loads and validates the central config.yaml file.
Provides typed access to all project settings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


def load_config(path: Path = _CONFIG_PATH) -> dict[str, Any]:
    """Load the project configuration from config.yaml.

    Parameters
    ----------
    path : Path
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Nested configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist at the given path.
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    return cfg


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parents[2]


def resolve_path(relative_path: str) -> Path:
    """Resolve a relative path string against the project root.

    Parameters
    ----------
    relative_path : str
        A path relative to the project root (e.g. 'data/processed').

    Returns
    -------
    Path
        Absolute resolved path.
    """
    return get_project_root() / relative_path


# ── Singleton config instance ──────────────────────────────────────────────
CONFIG: dict[str, Any] = load_config()

RANDOM_SEED: int = CONFIG["project"]["random_seed"]
TARGET_COLUMN: str = CONFIG["data"]["target_column"]

BUREAU_FEATURES: list[str] = CONFIG["features"]["bureau_features"]
BEHAVIORAL_FEATURES: list[str] = CONFIG["features"]["behavioral_features"]
DEMOGRAPHIC_FEATURES: list[str] = CONFIG["features"]["demographic_features"]
ALL_FEATURES: list[str] = BUREAU_FEATURES + BEHAVIORAL_FEATURES + DEMOGRAPHIC_FEATURES

FEATURE_GROUPS: dict[str, list[str]] = {
    "bureau": BUREAU_FEATURES,
    "behavioral": BEHAVIORAL_FEATURES,
    "demographic": DEMOGRAPHIC_FEATURES,
}
