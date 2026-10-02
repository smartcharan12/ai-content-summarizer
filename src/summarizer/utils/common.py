"""Small, reusable helpers used by every stage.

Heavy libraries (torch, numpy) are imported *inside* the functions that need
them, so importing this module stays fast and works in lightweight test
environments.
"""

from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any

import yaml

from summarizer.constants import PROJECT_ROOT
from summarizer.exception import ConfigurationError
from summarizer.logger import get_logger

logger = get_logger(__name__)


def resolve_path(path: str | Path) -> Path:
    """Turn a path from a config file into an absolute path.

    Relative paths are interpreted relative to the project root, so the code
    behaves the same whether you run it from the repo root, ``notebooks/`` or
    inside a Docker container.
    """
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dictionary.

    Raises:
        ConfigurationError: if the file is missing, empty or not valid YAML.
    """
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as f:
            content = yaml.safe_load(f)
    except FileNotFoundError as err:
        raise ConfigurationError("YAML file not found", context={"path": str(path)}) from err
    except yaml.YAMLError as err:
        raise ConfigurationError("Invalid YAML", context={"path": str(path)}) from err

    if not isinstance(content, dict) or not content:
        raise ConfigurationError("YAML file is empty or not a mapping", context={"path": str(path)})
    logger.debug("Loaded YAML: %s", path)
    return content


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``.

    Nested dictionaries are merged key by key; every other value in
    ``override`` replaces the one in ``base``. Neither input is modified.

    >>> deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3}})
    {'a': {'x': 1, 'y': 3}}
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def create_directories(paths: list[str | Path]) -> None:
    """Create each directory (and its parents) if it does not exist yet."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists: %s", path)


def save_json(path: str | Path, data: Any) -> None:
    """Write ``data`` as pretty-printed UTF-8 JSON, creating parent folders."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Saved JSON: %s", path)


def load_json(path: str | Path) -> Any:
    """Read a JSON file."""
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so runs are reproducible."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is always installed with torch
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass
    logger.debug("Random seed set to %d", seed)


def get_device(preferred: str | None = None) -> str:
    """Pick the device to run the model on.

    Args:
        preferred: ``"cuda"``, ``"mps"``, ``"cpu"`` or ``None``/``"auto"`` to
            choose automatically (CUDA GPU > Apple Silicon GPU > CPU).

    Returns:
        A device string usable with ``model.to(device)``.
    """
    import torch

    preferred = (preferred or "auto").lower()
    if preferred == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU")
        return "cpu"
    if preferred in {"cuda", "mps", "cpu"}:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
