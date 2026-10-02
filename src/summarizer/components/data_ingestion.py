"""Stage 1 - Data ingestion.

Downloads SAMSum from the Hugging Face Hub and stores an untouched copy of
each split as CSV in ``data/raw/``. Nothing is cleaned or filtered here: raw
data must stay identical to the source so every later step can be re-run and
audited against it.

We keep SAMSum's official train/validation/test split rather than
re-splitting. Published SAMSum results use the same split, so our ROUGE
scores stay comparable to theirs.

Reproducibility: ``ingestion_stats.json`` records the dataset's Hub commit
hash and a SHA-256 checksum of each CSV, so you can prove which data a model
was trained on.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from summarizer.entity.config_entity import DataIngestionConfig
from summarizer.exception import DataIngestionError
from summarizer.logger import get_logger
from summarizer.utils.common import load_json, save_json
from summarizer.utils.text_stats import describe_lengths, word_counts

logger = get_logger(__name__)

# A loader takes (dataset_name, revision) and returns {split_name: DataFrame}.
# Injecting it lets tests run offline with fake data.
DatasetLoader = Callable[[str, str | None], dict[str, pd.DataFrame]]


def load_from_hub(name: str, revision: str | None) -> dict[str, pd.DataFrame]:
    """Default loader: download every split with the ``datasets`` library."""
    from datasets import load_dataset  # heavy import, only needed here

    dataset_dict = load_dataset(name, revision=revision)
    # astype(object): pandas 3 infers a new string dtype; plain Python strings
    # keep the CSV round-trip and downstream code simple.
    return {split: ds.to_pandas().astype(object) for split, ds in dataset_dict.items()}


def resolve_hub_commit(name: str, revision: str | None) -> str | None:
    """Return the Hub commit hash for the dataset, or None if offline."""
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(name, revision=revision).sha
    except Exception as err:  # noqa: BLE001 - metadata only, never fatal
        logger.warning("Could not resolve dataset commit hash: %s", err)
        return None


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DataIngestion:
    """Download the dataset and persist the raw splits.

    Args:
        config: Paths and dataset identifiers for this stage.
        loader: Function that fetches the splits (defaults to the HF Hub).
    """

    def __init__(self, config: DataIngestionConfig, loader: DatasetLoader = load_from_hub) -> None:
        self.config = config
        self.loader = loader

    def raw_path(self, split: str) -> Path:
        return self.config.raw_dir / f"{split}.csv"

    def _already_downloaded(self) -> bool:
        return all(self.raw_path(s).is_file() for s in self.config.splits)

    def download(self) -> dict[str, pd.DataFrame]:
        """Fetch the dataset and keep only the configured splits."""
        logger.info("Downloading dataset '%s'", self.config.hf_dataset_name)
        try:
            splits = self.loader(self.config.hf_dataset_name, self.config.hf_revision)
        except Exception as err:
            raise DataIngestionError(
                "Failed to download dataset",
                context={"dataset": self.config.hf_dataset_name, "error": str(err)},
            ) from err

        missing = [s for s in self.config.splits if s not in splits]
        if missing:
            raise DataIngestionError(
                "Dataset is missing expected splits",
                context={"missing": missing, "available": sorted(splits)},
            )
        return {s: splits[s] for s in self.config.splits}

    def save(self, splits: dict[str, pd.DataFrame]) -> None:
        """Write each split to ``data/raw/<split>.csv``."""
        self.config.raw_dir.mkdir(parents=True, exist_ok=True)
        for split, df in splits.items():
            path = self.raw_path(split)
            df.to_csv(path, index=False)
            logger.info("Saved %-10s %6d rows -> %s", split, len(df), path)

    def compute_stats(self, commit: str | None) -> dict[str, Any]:
        """Describe the saved raw files (row counts, columns, lengths, checksums)."""
        stats: dict[str, Any] = {
            "dataset": self.config.hf_dataset_name,
            "requested_revision": self.config.hf_revision,
            "resolved_commit": commit,
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "splits": {},
        }
        for split in self.config.splits:
            path = self.raw_path(split)
            df = pd.read_csv(path, dtype=str, keep_default_na=False)
            split_stats: dict[str, Any] = {
                "file": str(path.name),
                "sha256": sha256_of_file(path),
                "rows": len(df),
                "columns": list(df.columns),
            }
            for column in ("dialogue", "summary"):
                if column in df.columns:
                    split_stats[f"{column}_words"] = describe_lengths(word_counts(df[column]))
            stats["splits"][split] = split_stats
        stats["total_rows"] = sum(s["rows"] for s in stats["splits"].values())
        return stats

    def run(self, force: bool = False) -> dict[str, Any]:
        """Execute the stage.

        Args:
            force: Re-download even if raw files already exist.

        Returns:
            The statistics dictionary (also written to ``stats_file``).
        """
        commit = None
        if self._already_downloaded() and not force:
            logger.info("Raw data already present in %s - skipping download", self.config.raw_dir)
        else:
            self.save(self.download())
            commit = resolve_hub_commit(self.config.hf_dataset_name, self.config.hf_revision)

        try:
            stats = self.compute_stats(commit)
        except Exception as err:
            raise DataIngestionError(
                "Could not read back raw files", context={"error": str(err)}
            ) from err

        if commit is None and self.config.stats_file.is_file():
            # Skipped download: keep the commit recorded when the data was fetched.
            previous = load_json(self.config.stats_file)
            stats["resolved_commit"] = previous.get("resolved_commit")
            stats["ingested_at"] = previous.get("ingested_at", stats["ingested_at"])

        save_json(self.config.stats_file, stats)
        logger.info(
            "Ingestion complete: %d rows across %s", stats["total_rows"], self.config.splits
        )
        return stats
