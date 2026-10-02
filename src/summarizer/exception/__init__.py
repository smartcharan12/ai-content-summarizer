"""Custom exception hierarchy.

Why a hierarchy instead of raising plain ``Exception``?

* Callers can catch exactly what they care about. The API, for example,
  turns :class:`InputValidationError` into HTTP 422 and
  :class:`ModelNotReadyError` into HTTP 503, while anything else becomes 500.
* Every pipeline stage has its own type, so a log line immediately tells you
  *which* stage failed.

Always chain the original error with ``raise X(...) from err``. Python then
keeps the full original traceback (file, line number, cause), so we do not
need to re-implement that by hand.
"""

from __future__ import annotations

from typing import Any


class SummarizerError(Exception):
    """Base class for every error raised by this project.

    Args:
        message: Human-readable description of what went wrong.
        context: Optional structured details (paths, sizes, ids) that make the
            error easier to debug. They are appended to ``str(error)``.
    """

    def __init__(self, message: str, *, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if not self.context:
            return self.message
        details = ", ".join(f"{k}={v!r}" for k, v in self.context.items())
        return f"{self.message} ({details})"


# --- Configuration -----------------------------------------------------------
class ConfigurationError(SummarizerError):
    """A config/params file is missing, malformed or has an invalid value."""


# --- Training pipeline stages -------------------------------------------------
class DataIngestionError(SummarizerError):
    """Downloading or saving the raw dataset failed."""


class DataValidationError(SummarizerError):
    """The dataset failed a blocking validation check."""


class DataTransformationError(SummarizerError):
    """Cleaning or tokenizing the dataset failed."""


class ModelTrainingError(SummarizerError):
    """Fine-tuning the model failed."""


class ModelEvaluationError(SummarizerError):
    """Scoring the model failed."""


class ModelRegistryError(SummarizerError):
    """Unknown run id, invalid promotion, or corrupt registry file."""


# --- Inference ----------------------------------------------------------------
class ModelNotReadyError(SummarizerError):
    """No model is loaded, or the model could not be loaded."""


class InputValidationError(SummarizerError):
    """The text sent for summarization is invalid (empty, too long, ...)."""


class PredictionError(SummarizerError):
    """The model failed while generating a summary."""
