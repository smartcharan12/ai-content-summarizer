"""SummaryGenerator: load a seq2seq model once and turn text into summaries.

Used by BOTH the evaluation stage and the API's PredictionPipeline. Because
there is a single generation code path, the quality and latency measured in
evaluation are the quality and latency users actually get.

Steps for every input: clean (same TextCleaner as training) -> tokenize
(truncate to max_input_length) -> ``model.generate`` with a decoding profile
-> decode -> light post-processing.
"""

from __future__ import annotations

import re
import time
from typing import Any

from summarizer.components.text_cleaning import TextCleaner
from summarizer.entity.config_entity import GenerationParams
from summarizer.exception import ModelNotReadyError, PredictionError
from summarizer.logger import get_logger
from summarizer.utils.common import get_device

logger = get_logger(__name__)

_SPACES = re.compile(r"\s+")


def postprocess(summary: str) -> str:
    """Tidy model output: collapse whitespace, remove space before punctuation."""
    summary = _SPACES.sub(" ", summary).strip()
    return re.sub(r"\s+([.,!?;:])", r"\1", summary)


class SummaryGenerator:
    """Wraps a tokenizer + model.

    Args:
        model_name_or_path: Local folder (``artifacts/models/<run_id>``) or Hub id.
        device: ``auto``/``cpu``/``cuda``/``mps``.
        max_input_length: Inputs are truncated to this many tokens.
        cleaner: Text cleaner; must match the one used in training.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = "auto",
        max_input_length: int = 512,
        cleaner: TextCleaner | None = None,
    ) -> None:
        self.model_name_or_path = str(model_name_or_path)
        self.device = get_device(device)
        self.max_input_length = max_input_length
        self.cleaner = cleaner or TextCleaner()
        self.model: Any = None
        self.tokenizer: Any = None
        self.load_seconds: float | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> SummaryGenerator:
        """Load tokenizer and model weights (slow: do it once, at start-up)."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        start = time.perf_counter()
        logger.info("Loading model '%s' on %s", self.model_name_or_path, self.device)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name_or_path)
        except Exception as err:
            raise ModelNotReadyError(
                "Could not load model",
                context={"model": self.model_name_or_path, "error": str(err)},
            ) from err
        self.model = model.to(self.device).eval()  # eval(): disables dropout
        max_positions = getattr(self.model.config, "max_position_embeddings", None)
        if max_positions and self.max_input_length > max_positions:
            logger.warning(
                "Capping max_input_length %d -> %d", self.max_input_length, max_positions
            )
            self.max_input_length = max_positions
        self.load_seconds = round(time.perf_counter() - start, 2)
        logger.info("Model loaded in %.1fs", self.load_seconds)
        return self

    def count_tokens(self, text: str) -> int:
        """Tokens the model would see for ``text`` before truncation."""
        self._require_loaded()
        return len(self.tokenizer(self.cleaner.clean(text))["input_ids"])

    def generate(
        self, texts: list[str], params: GenerationParams, batch_size: int = 8
    ) -> list[str]:
        """Summarise ``texts`` in batches. Returns one summary per input, in order."""
        import torch

        self._require_loaded()
        cleaned = self.cleaner.clean_batch(texts)
        summaries: list[str] = []
        try:
            with torch.inference_mode():  # no gradient bookkeeping: faster, less memory
                for i in range(0, len(cleaned), batch_size):
                    batch = cleaned[i : i + batch_size]
                    inputs = self.tokenizer(
                        batch,
                        max_length=self.max_input_length,
                        truncation=True,
                        padding=True,
                        return_tensors="pt",
                    ).to(self.device)
                    output_ids = self.model.generate(**inputs, **params.to_dict())
                    decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    summaries.extend(postprocess(s) for s in decoded)
        except Exception as err:
            raise PredictionError("Generation failed", context={"error": str(err)}) from err
        return summaries

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise ModelNotReadyError("Model is not loaded - call load() first")
