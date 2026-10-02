"""Stage 3 - Data transformation.

Turns validated text into the tensors a seq2seq model trains on:

1. **Clean** dialogue and summary with :class:`TextCleaner` (same code as the API).
2. **Tokenize the dialogue** -> ``input_ids`` (sub-word ids) and
   ``attention_mask`` (1 = real token, 0 = padding), truncated to
   ``max_input_length``.
3. **Tokenize the summary** -> ``labels``, truncated to ``max_target_length``.
   The model learns to predict these ids one at a time ("teacher forcing").
4. **Padding.** With ``padding: dynamic`` (default) nothing is padded here;
   ``DataCollatorForSeq2Seq`` pads each *batch* to its longest member at
   training time, which wastes far less compute than padding everything to
   512. With ``padding: max_length`` we pad now and replace label padding with
   ``-100``, the index PyTorch's cross-entropy loss ignores, so the model is
   never trained to "predict padding".

The tokenized ``DatasetDict`` is saved to ``data/processed/`` in Arrow format,
together with ``transformation_meta.json``, which records the tokenizer and
lengths used. The trainer checks that file, so it can't train a model on ids
produced by a different tokenizer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from summarizer.components.text_cleaning import TextCleaner
from summarizer.entity.config_entity import DataTransformationConfig
from summarizer.exception import DataTransformationError
from summarizer.logger import get_logger
from summarizer.utils.common import save_json

logger = get_logger(__name__)

META_FILE = "transformation_meta.json"
LABEL_PAD_ID = -100


def _length_stats(lengths: list[int], limit: int) -> dict[str, Any]:
    s = pd.Series(lengths)
    return {
        "mean": round(float(s.mean()), 1),
        "median": float(s.median()),
        "p95": float(s.quantile(0.95)),
        "max": int(s.max()),
        "limit": limit,
        "truncated_rows": int((s > limit).sum()),
        "truncated_fraction": round(float((s > limit).mean()), 4),
    }


class DataTransformation:
    """Clean and tokenize the validated splits.

    Args:
        config: Stage configuration.
        tokenizer: Optional pre-loaded tokenizer (otherwise loaded from
            ``config.tokenizer_name``). Injecting one keeps tests fast.
    """

    def __init__(self, config: DataTransformationConfig, tokenizer: Any = None) -> None:
        self.config = config
        self.cleaner = TextCleaner(
            normalize_whitespace=config.normalize_whitespace,
            normalize_tags=config.normalize_tags,
            lowercase=config.lowercase,
        )
        self._tokenizer = tokenizer

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            logger.info("Loading tokenizer '%s'", self.config.tokenizer_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        return self._tokenizer

    # ------------------------------------------------------------- steps
    def load_and_clean(self, split: str) -> pd.DataFrame:
        path = self.config.validated_dir / f"{split}.csv"
        if not path.is_file():
            raise DataTransformationError(
                "Validated split not found - run the validation stage first",
                context={"path": str(path)},
            )
        df = pd.read_csv(path, dtype=object, keep_default_na=False)
        df["dialogue"] = self.cleaner.clean_batch(df["dialogue"].tolist())
        df["summary"] = self.cleaner.clean_batch(df["summary"].tolist())
        empty = (df["dialogue"] == "") | (df["summary"] == "")
        if empty.any():
            logger.warning("%s: %d rows empty after cleaning - dropped", split, int(empty.sum()))
        return df[~empty].reset_index(drop=True)

    def tokenize_batch(self, batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        """Tokenize a batch of rows (called by ``Dataset.map``)."""
        pad = "max_length" if self.config.padding == "max_length" else False
        model_inputs = self.tokenizer(
            batch["dialogue"],
            max_length=self.config.max_input_length,
            truncation=True,
            padding=pad,
        )
        # text_target= tokenizes the summary as a *target* (the modern
        # replacement for the removed `as_target_tokenizer()` context manager).
        labels = self.tokenizer(
            text_target=batch["summary"],
            max_length=self.config.max_target_length,
            truncation=True,
            padding=pad,
        )["input_ids"]
        if pad:
            pad_id = self.tokenizer.pad_token_id
            labels = [[(t if t != pad_id else LABEL_PAD_ID) for t in seq] for seq in labels]
        model_inputs["labels"] = labels
        return model_inputs

    def token_length_stats(self, df: pd.DataFrame) -> dict[str, Any]:
        """Untruncated token lengths -> how much truncation will cost."""
        from transformers.utils import logging as hf_logging

        # Measuring over-long inputs is the point here, so silence the
        # "sequence longer than the maximum" notice for these two calls.
        previous = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            dialogue = [len(x) for x in self.tokenizer(df["dialogue"].tolist())["input_ids"]]
            summary = [
                len(x) for x in self.tokenizer(text_target=df["summary"].tolist())["input_ids"]
            ]
        finally:
            hf_logging.set_verbosity(previous)
        return {
            "dialogue_tokens": _length_stats(dialogue, self.config.max_input_length),
            "summary_tokens": _length_stats(summary, self.config.max_target_length),
        }

    # --------------------------------------------------------------- run
    def run(self) -> dict[str, Any]:
        from datasets import Dataset, DatasetDict

        cfg = self.config
        splits: dict[str, Dataset] = {}
        report: dict[str, Any] = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tokenizer_name": cfg.tokenizer_name,
            "max_input_length": cfg.max_input_length,
            "max_target_length": cfg.max_target_length,
            "padding": cfg.padding,
            "cleaning": {
                "normalize_whitespace": cfg.normalize_whitespace,
                "normalize_tags": cfg.normalize_tags,
                "lowercase": cfg.lowercase,
            },
            "splits": {},
        }

        missing = [s for s in cfg.splits if not (cfg.validated_dir / f"{s}.csv").is_file()]
        if missing:  # fail fast, before spending time tokenizing the other splits
            raise DataTransformationError(
                "Validated split not found - run the validation stage first",
                context={"missing": missing, "dir": str(cfg.validated_dir)},
            )

        try:
            for split in cfg.splits:
                df = self.load_and_clean(split)
                dataset = Dataset.from_pandas(
                    df[["id", "dialogue", "summary"]], preserve_index=False
                )
                splits[split] = dataset.map(
                    self.tokenize_batch,
                    batched=True,
                    remove_columns=["dialogue", "summary"],  # keep "id" for traceability
                    desc=f"Tokenizing {split}",
                )
                report["splits"][split] = {"rows": len(df), **self.token_length_stats(df)}
                stats = report["splits"][split]["dialogue_tokens"]
                logger.info(
                    "%-10s %6d rows | dialogues truncated: %d (%.2f%%)",
                    split,
                    len(df),
                    stats["truncated_rows"],
                    100 * stats["truncated_fraction"],
                )
        except DataTransformationError:
            raise
        except Exception as err:
            raise DataTransformationError(
                "Tokenization failed", context={"error": str(err)}
            ) from err

        cfg.processed_dir.mkdir(parents=True, exist_ok=True)
        DatasetDict(splits).save_to_disk(str(cfg.processed_dir))
        save_json(cfg.processed_dir / META_FILE, report)
        save_json(cfg.report_file, report)
        logger.info("Saved tokenized dataset to %s", cfg.processed_dir)
        return report
