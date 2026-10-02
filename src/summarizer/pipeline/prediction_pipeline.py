"""Inference pipeline used by the API.

Request flow::

    text -> validate (empty? too short? too long?) -> clean (TextCleaner, same as training)
         -> tokenize (truncate to max_input_length) -> PEGASUS generate (decoding profile)
         -> post-process -> summary + metadata

Which model is served (``MODEL_SOURCE``):

* ``registry`` (default): the *production* model in ``artifacts/models/registry.json``.
  If nothing has been promoted yet, or its files are missing, it falls back to
  ``MODEL_NAME_OR_PATH`` and logs a warning.
* ``path``: always ``MODEL_NAME_OR_PATH`` (a local folder or a Hugging Face Hub id,
  e.g. your fine-tuned model pushed to the Hub).

Lengths in the response are **characters**, matching how the UI counts input;
word counts are included as well. ``compression_ratio = original_length / summary_length``.
"""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

from summarizer import __version__
from summarizer.components.model_registry import ModelRegistry
from summarizer.components.summary_generator import SummaryGenerator
from summarizer.components.text_cleaning import TextCleaner
from summarizer.config.configuration import ConfigurationManager
from summarizer.config.settings import AppSettings
from summarizer.exception import ConfigurationError, InputValidationError, ModelNotReadyError
from summarizer.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class SummaryResult:
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    processing_time_seconds: float
    original_word_count: int
    summary_word_count: int
    input_tokens: int
    truncated: bool
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PredictionPipeline:
    """Load the serving model once; summarise validated text on demand.

    Args:
        settings: Runtime settings (env vars).
        config: YAML configuration (decoding profiles, preprocessing).
        generator: Pre-built generator (tests inject a fake one).
    """

    def __init__(
        self,
        settings: AppSettings,
        config: ConfigurationManager | None = None,
        generator: SummaryGenerator | None = None,
    ) -> None:
        self.settings = settings
        config = config or ConfigurationManager()
        profiles = config.get_generation_profiles()
        if settings.generation_profile not in profiles:
            raise ConfigurationError(
                "GENERATION_PROFILE is not defined in params.yaml",
                context={"profile": settings.generation_profile, "available": sorted(profiles)},
            )
        self.generation_params = profiles[settings.generation_profile]
        preprocessing = config.params["preprocessing"]
        self.max_input_length = int(preprocessing["max_input_length"])
        self.model_path, self.model_version = self._resolve_model(config)
        self.generator = generator or SummaryGenerator(
            self.model_path,
            device=settings.device,
            max_input_length=self.max_input_length,
            cleaner=TextCleaner(
                normalize_whitespace=bool(preprocessing["normalize_whitespace"]),
                normalize_tags=bool(preprocessing["normalize_tags"]),
                lowercase=bool(preprocessing["lowercase"]),
            ),
        )
        # One generation at a time per process: a 570M-parameter model already
        # uses every CPU core, and parallel calls would multiply memory use.
        # Scale out with more containers instead (see docs/AWS_DEPLOYMENT.md).
        self._lock = threading.Lock()
        self.load_error: str | None = None

    # ------------------------------------------------------------ model
    def _resolve_model(self, config: ConfigurationManager) -> tuple[str, str]:
        fallback = self.settings.model_name_or_path
        if self.settings.model_source == "path":
            return fallback, fallback
        registry = ModelRegistry(config.registry_file)
        production = registry.get_production()
        if production is None:
            logger.warning("No production model in the registry; serving '%s'", fallback)
            return fallback, fallback
        path = registry.model_path(production)
        if not path.is_dir():
            logger.warning("Production model files missing at %s; serving '%s'", path, fallback)
            return fallback, fallback
        return str(path), f"{production['run_id']} (v{production['version']})"

    @property
    def is_ready(self) -> bool:
        return self.generator.is_loaded

    def load(self) -> PredictionPipeline:
        """Load model weights. Errors are kept in ``load_error`` for /health."""
        try:
            self.generator.load()
            self.load_error = None
        except ModelNotReadyError as err:
            self.load_error = str(err)
            raise
        return self

    def info(self) -> dict[str, Any]:
        return {
            "model": self.model_version,
            "model_loaded": self.is_ready,
            "device": self.generator.device,
            "generation_profile": self.settings.generation_profile,
            "generation_params": self.generation_params.to_dict(),
            "max_input_tokens": self.max_input_length,
            "max_input_chars": self.settings.max_input_chars,
            "min_input_words": self.settings.min_input_words,
            "max_batch_size": self.settings.max_batch_size,
            "load_seconds": self.generator.load_seconds,
            "load_error": self.load_error,
            "version": __version__,
        }

    # -------------------------------------------------------- validation
    def validate(self, text: str | None) -> str:
        """Return the stripped text or raise :class:`InputValidationError`."""
        if text is None or not text.strip():
            raise InputValidationError("Text is empty")
        text = text.strip()
        if len(text) > self.settings.max_input_chars:
            raise InputValidationError(
                f"Text is too long: {len(text)} characters "
                f"(maximum {self.settings.max_input_chars})",
                context={"chars": len(text), "max_chars": self.settings.max_input_chars},
            )
        words = len(text.split())
        if words < self.settings.min_input_words:
            raise InputValidationError(
                f"Text is too short to summarise: {words} words "
                f"(minimum {self.settings.min_input_words})",
                context={"words": words, "min_words": self.settings.min_input_words},
            )
        return text

    # -------------------------------------------------------- inference
    def _result(self, text: str, summary: str, seconds: float) -> SummaryResult:
        tokens = self.generator.count_tokens(text)
        return SummaryResult(
            summary=summary,
            original_length=len(text),
            summary_length=len(summary),
            compression_ratio=round(len(text) / max(len(summary), 1), 2),
            processing_time_seconds=round(seconds, 3),
            original_word_count=len(text.split()),
            summary_word_count=len(summary.split()),
            input_tokens=tokens,
            truncated=tokens > self.generator.max_input_length,
            model_version=self.model_version,
        )

    def _generate(self, texts: list[str]) -> list[str]:
        if not self.is_ready:
            raise ModelNotReadyError("Model is not loaded", context={"error": self.load_error})
        with self._lock:
            return self.generator.generate(
                texts, self.generation_params, batch_size=self.settings.max_batch_size
            )

    def summarize(self, text: str) -> SummaryResult:
        text = self.validate(text)
        start = time.perf_counter()
        summary = self._generate([text])[0]
        result = self._result(text, summary, time.perf_counter() - start)
        logger.info(
            "Summarised %d chars -> %d chars in %.2fs",
            result.original_length,
            result.summary_length,
            result.processing_time_seconds,
        )
        return result

    def summarize_batch(self, texts: list[str]) -> tuple[list[SummaryResult], float]:
        """Validate every text first (fail fast), then generate in one batch."""
        if not texts:
            raise InputValidationError("'texts' must contain at least one item")
        if len(texts) > self.settings.max_batch_size:
            raise InputValidationError(
                f"Too many texts: {len(texts)} (maximum {self.settings.max_batch_size})",
                context={"count": len(texts), "max_batch_size": self.settings.max_batch_size},
            )
        cleaned = []
        for index, text in enumerate(texts):
            try:
                cleaned.append(self.validate(text))
            except InputValidationError as err:
                raise InputValidationError(
                    f"texts[{index}]: {err.message}", context={"index": index, **err.context}
                ) from err

        start = time.perf_counter()
        summaries = self._generate(cleaned)
        total = time.perf_counter() - start
        per_item = total / len(cleaned)  # batched generation has no per-item clock
        results = [self._result(t, s, per_item) for t, s in zip(cleaned, summaries, strict=True)]
        logger.info("Summarised batch of %d in %.2fs", len(results), total)
        return results, round(total, 3)
