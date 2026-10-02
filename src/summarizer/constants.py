"""Project-wide constants.

Only things that are genuinely constant live here (file locations of the
config files). Tunable values belong in ``config/params.yaml``.
"""

import os
from pathlib import Path

# src/summarizer/constants.py -> parents[2] is the project root.
# PROJECT_ROOT can be overridden (e.g. inside Docker, where code lives in /app).
PROJECT_ROOT: Path = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()

CONFIG_FILE_PATH: Path = PROJECT_ROOT / "config" / "config.yaml"
PARAMS_FILE_PATH: Path = PROJECT_ROOT / "config" / "params.yaml"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
