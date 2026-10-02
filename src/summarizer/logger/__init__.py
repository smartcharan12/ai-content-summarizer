"""Centralised logging.

Usage everywhere in the project::

    from summarizer.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Loaded %d rows", n)

Behaviour is controlled by environment variables:

* ``LOG_LEVEL``  - DEBUG / INFO / WARNING / ERROR (default INFO)
* ``LOG_FORMAT`` - ``text`` for humans, ``json`` for log aggregators such as
  AWS CloudWatch (default text)
* ``LOG_TO_FILE`` - ``true`` to also write ``logs/summarizer.log`` (default true)

Logging is configured once, on the first call to :func:`get_logger`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler

from summarizer.constants import LOGS_DIR

_ROOT_LOGGER_NAME = "summarizer"
_configured = False


class JsonFormatter(logging.Formatter):
    """Format each record as a single JSON line (easy to search in CloudWatch)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


TEXT_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s:%(lineno)d - %(message)s"


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _configure() -> None:
    """Attach handlers to the project's root logger exactly once."""
    global _configured
    if _configured:
        return

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    formatter: logging.Formatter = (
        JsonFormatter()
        if os.getenv("LOG_FORMAT", "text").lower() == "json"
        else logging.Formatter(TEXT_FORMAT)
    )

    root = logging.getLogger(_ROOT_LOGGER_NAME)
    root.setLevel(level)
    root.propagate = False  # don't print twice via Python's global root logger

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if _env_flag("LOG_TO_FILE", default=True):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        # Rotate at 5 MB, keep 3 old files, so logs never fill the disk.
        file_handler = RotatingFileHandler(
            LOGS_DIR / "summarizer.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger that is a child of the project's root logger.

    Args:
        name: Usually ``__name__`` of the calling module.

    Returns:
        A configured :class:`logging.Logger`.
    """
    _configure()
    if not name.startswith(_ROOT_LOGGER_NAME):
        name = f"{_ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)
