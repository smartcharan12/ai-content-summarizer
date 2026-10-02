"""Tests for Stage 1 (DataIngestion). A fake loader keeps them offline."""

from __future__ import annotations

import pandas as pd
import pytest

from summarizer.components.data_ingestion import DataIngestion
from summarizer.entity.config_entity import DataIngestionConfig
from summarizer.exception import DataIngestionError
from summarizer.utils.common import load_json

SPLITS = ("train", "validation", "test")


def fake_splits() -> dict[str, pd.DataFrame]:
    row = {
        "id": "1",
        "dialogue": "Anna: Hi Bob!\nBob: Hi Anna, lunch?",
        "summary": "Bob asks Anna.",
    }
    return {s: pd.DataFrame([row] * n) for s, n in zip(SPLITS, (5, 2, 3), strict=True)}


@pytest.fixture
def config(tmp_path) -> DataIngestionConfig:
    return DataIngestionConfig(
        hf_dataset_name="fake/samsum",
        splits=SPLITS,
        raw_dir=tmp_path / "raw",
        stats_file=tmp_path / "reports" / "ingestion_stats.json",
    )


@pytest.fixture(autouse=True)
def no_hub_calls(monkeypatch):
    """Never contact the Hub for the commit hash during tests."""
    monkeypatch.setattr(
        "summarizer.components.data_ingestion.resolve_hub_commit", lambda *_: "abc123"
    )


def test_run_saves_raw_csvs_and_stats(config):
    stats = DataIngestion(config, loader=lambda *_: fake_splits()).run()

    for split in SPLITS:
        assert (config.raw_dir / f"{split}.csv").is_file()
    assert stats["total_rows"] == 10
    assert stats["resolved_commit"] == "abc123"
    assert stats["splits"]["train"]["columns"] == ["id", "dialogue", "summary"]
    assert len(stats["splits"]["train"]["sha256"]) == 64
    assert stats["splits"]["test"]["summary_words"]["max"] == 3
    assert load_json(config.stats_file) == stats


def test_raw_text_is_stored_unchanged(config):
    DataIngestion(config, loader=lambda *_: fake_splits()).run()
    saved = pd.read_csv(config.raw_dir / "train.csv", dtype=str)
    assert saved.loc[0, "dialogue"] == "Anna: Hi Bob!\nBob: Hi Anna, lunch?"


def test_second_run_skips_download_and_keeps_commit(config):
    DataIngestion(config, loader=lambda *_: fake_splits()).run()

    def must_not_be_called(*_):
        raise AssertionError("loader called although raw data exists")

    stats = DataIngestion(config, loader=must_not_be_called).run()
    assert stats["resolved_commit"] == "abc123"


def test_force_redownloads(config):
    DataIngestion(config, loader=lambda *_: fake_splits()).run()
    calls = []

    def loader(*args):
        calls.append(args)
        return fake_splits()

    DataIngestion(config, loader=loader).run(force=True)
    assert calls == [("fake/samsum", None)]


def test_missing_split_raises(config):
    partial = {k: v for k, v in fake_splits().items() if k != "test"}
    with pytest.raises(DataIngestionError, match="missing expected splits"):
        DataIngestion(config, loader=lambda *_: partial).run()


def test_network_error_is_wrapped(config):
    def broken(*_):
        raise ConnectionError("no internet")

    with pytest.raises(DataIngestionError, match="Failed to download") as info:
        DataIngestion(config, loader=broken).run()
    assert isinstance(info.value.__cause__, ConnectionError)
