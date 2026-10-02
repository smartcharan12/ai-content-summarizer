"""Tests for Stage 2 (DataValidation) using small hand-made CSV files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from summarizer.components.data_validation import DataValidation
from summarizer.entity.config_entity import DataValidationConfig
from summarizer.exception import DataValidationError
from summarizer.utils.common import load_json

GOOD_DIALOGUE = "Anna: Are we still meeting at noon?\nBob: Yes, see you at the cafe."
GOOD_SUMMARY = "Anna and Bob will meet at noon."


def make_config(tmp_path: Path, **overrides) -> DataValidationConfig:
    values = dict(
        raw_dir=tmp_path / "raw",
        validated_dir=tmp_path / "validated",
        report_file=tmp_path / "reports" / "validation_report.json",
        splits=("train", "validation", "test"),
        required_columns=("id", "dialogue", "summary"),
        min_dialogue_words=5,
        max_dialogue_words=50,
        min_summary_words=2,
        drop_duplicates=True,
        remove_cross_split_leakage=True,
        max_drop_fraction=0.9,
    )
    values.update(overrides)
    return DataValidationConfig(**values)


def good_rows(prefix: str, n: int) -> list[dict]:
    return [
        {
            "id": f"{prefix}{i}",
            "dialogue": f"{GOOD_DIALOGUE} ({prefix}{i})",
            "summary": GOOD_SUMMARY,
        }
        for i in range(n)
    ]


def write_raw(cfg: DataValidationConfig, splits: dict[str, list[dict] | pd.DataFrame]) -> None:
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
        df.to_csv(cfg.raw_dir / f"{split}.csv", index=False)


def check(report: dict, split: str, name: str) -> dict:
    return next(c for c in report["splits"][split]["checks"] if c["name"] == name)


@pytest.fixture
def bad_train() -> list[dict]:
    return good_rows("t", 4) + [
        {"id": "null_d", "dialogue": np.nan, "summary": GOOD_SUMMARY},
        {"id": "null_s", "dialogue": GOOD_DIALOGUE + " x", "summary": np.nan},
        {"id": "blank_d", "dialogue": "   ", "summary": GOOD_SUMMARY},
        {"id": "short_d", "dialogue": "A: hi", "summary": GOOD_SUMMARY},
        {"id": "long_d", "dialogue": "A: " + "word " * 60, "summary": GOOD_SUMMARY},
        {"id": "junk_s", "dialogue": GOOD_DIALOGUE + " y", "summary": "yyy"},
        {"id": "t0", "dialogue": GOOD_DIALOGUE + " z", "summary": GOOD_SUMMARY},  # dup id
        {"id": "dup_pair", "dialogue": f"{GOOD_DIALOGUE} (t1)", "summary": GOOD_SUMMARY},
        {"id": "leak", "dialogue": f"{GOOD_DIALOGUE} (v0)", "summary": "Other summary text."},
    ]


def test_every_bad_row_is_dropped_and_reported(tmp_path, bad_train):
    cfg = make_config(tmp_path)
    write_raw(cfg, {"train": bad_train, "validation": good_rows("v", 2), "test": good_rows("s", 2)})

    report = DataValidation(cfg).run()

    assert report["status"] == "passed"
    expected = {
        "missing_dialogue": "null_d",
        "missing_summary": "null_s",
        "empty_dialogue": "blank_d",
        "dialogue_too_short": "short_d",
        "dialogue_too_long": "long_d",
        "summary_too_short": "junk_s",
        "duplicate_id": "t0",
        "duplicate_pair": "dup_pair",
        "cross_split_leakage": "leak",
    }
    for name, row_id in expected.items():
        result = check(report, "train", name)
        assert result["count"] == 1, name
        assert result["example_ids"] == [row_id], name

    kept = pd.read_csv(cfg.validated_dir / "train.csv", dtype=str)
    assert sorted(kept["id"]) == ["t0", "t1", "t2", "t3"]
    assert report["splits"]["train"]["rows_dropped"] == 9


def test_whitespace_is_trimmed(tmp_path):
    cfg = make_config(tmp_path)
    rows = good_rows("t", 2)
    rows[0]["summary"] = "  " + GOOD_SUMMARY + "  "
    write_raw(cfg, {"train": rows, "validation": good_rows("v", 1), "test": good_rows("s", 1)})

    DataValidation(cfg).run()
    kept = pd.read_csv(cfg.validated_dir / "train.csv", dtype=str)
    assert kept.loc[0, "summary"] == GOOD_SUMMARY


def test_warnings_do_not_drop_rows(tmp_path):
    cfg = make_config(tmp_path)
    rows = good_rows("t", 3)
    rows[0]["dialogue"] = GOOD_DIALOGUE + "\nAnna: look <file_photo>"
    rows[1]["dialogue"] = GOOD_DIALOGUE + "\nAnna: read https://example.com"
    rows[2]["summary"] = " ".join(["word"] * 30)  # longer than the 15-word dialogue
    rows.append({"id": "t3", "dialogue": rows[1]["dialogue"], "summary": "Another annotator."})
    write_raw(cfg, {"train": rows, "validation": good_rows("v", 1), "test": good_rows("s", 1)})

    report = DataValidation(cfg).run()

    assert report["splits"]["train"]["rows_out"] == 4
    assert check(report, "train", "contains_media_placeholder")["example_ids"] == ["t0"]
    assert check(report, "train", "contains_url")["count"] == 2
    assert check(report, "train", "summary_longer_than_dialogue")["example_ids"] == ["t2"]
    assert check(report, "train", "shared_dialogue_multiple_summaries")["count"] == 2


def test_leakage_removal_can_be_disabled(tmp_path):
    cfg = make_config(tmp_path, remove_cross_split_leakage=False)
    train = good_rows("t", 2) + [
        {"id": "leak", "dialogue": f"{GOOD_DIALOGUE} (v0)", "summary": "X y"}
    ]
    write_raw(cfg, {"train": train, "validation": good_rows("v", 1), "test": good_rows("s", 1)})

    report = DataValidation(cfg).run()
    assert report["splits"]["train"]["rows_out"] == 3


def test_missing_column_blocks_and_still_writes_report(tmp_path):
    cfg = make_config(tmp_path)
    no_summary = pd.DataFrame(good_rows("t", 2)).drop(columns="summary")
    write_raw(
        cfg, {"train": no_summary, "validation": good_rows("v", 1), "test": good_rows("s", 1)}
    )

    with pytest.raises(DataValidationError, match="Data validation failed"):
        DataValidation(cfg).run()

    report = load_json(cfg.report_file)
    assert report["status"] == "failed"
    assert "missing required columns" in report["splits"]["train"]["blocking_errors"][0]
    assert not (cfg.validated_dir / "train.csv").exists()


def test_missing_file_blocks(tmp_path):
    cfg = make_config(tmp_path)
    write_raw(cfg, {"train": good_rows("t", 2), "validation": good_rows("v", 1)})

    with pytest.raises(DataValidationError):
        DataValidation(cfg).run()
    assert (
        "raw file not found" in load_json(cfg.report_file)["splits"]["test"]["blocking_errors"][0]
    )


def test_excessive_drop_rate_blocks(tmp_path, bad_train):
    cfg = make_config(tmp_path, max_drop_fraction=0.05)
    write_raw(cfg, {"train": bad_train, "validation": good_rows("v", 2), "test": good_rows("s", 2)})

    with pytest.raises(DataValidationError) as info:
        DataValidation(cfg).run()
    assert "has the source format changed" in str(info.value)


def test_empty_split_blocks(tmp_path):
    cfg = make_config(tmp_path)
    empty = pd.DataFrame(columns=["id", "dialogue", "summary"])
    write_raw(cfg, {"train": good_rows("t", 2), "validation": empty, "test": good_rows("s", 1)})

    with pytest.raises(DataValidationError, match="validation"):
        DataValidation(cfg).run()
