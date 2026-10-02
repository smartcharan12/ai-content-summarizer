"""Shared pytest fixtures and test-wide environment setup."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

# Keep test runs from writing log files or picking up a developer's .env values.
os.environ.setdefault("LOG_TO_FILE", "false")
os.environ.pop("SMOKE_TEST", None)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def real_config_paths() -> tuple[Path, Path]:
    """Paths to the project's real config.yaml and params.yaml."""
    return PROJECT_ROOT / "config" / "config.yaml", PROJECT_ROOT / "config" / "params.yaml"


@pytest.fixture
def write_yaml(tmp_path: Path):
    """Factory fixture: write a dict to a temp YAML file and return its path."""

    def _write(name: str, data: dict) -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return _write
