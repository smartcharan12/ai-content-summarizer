"""Stage 2 - Data validation.

Reads ``data/raw/*.csv`` and checks three levels of problems:

1. **Blocking checks** stop the pipeline: missing file, missing column,
   empty split, or too many rows dropped. Training on silently broken data
   is worse than not training at all.
2. **Row checks** drop the offending rows: missing/empty text, too-short or
   too-long dialogues, too-short summaries, duplicates, and train rows that
   leak into validation/test.
3. **Warnings** keep the rows but count them, e.g. summaries longer than
   the dialogue, attachment placeholders, URLs, and statistical outliers.
   A human should know about these, but they are legitimate data.

Rows that pass are written to ``data/validated/<split>.csv``. Whitespace is
trimmed; real text cleaning happens in Stage 3. Everything is summarised in
``validation_report.json``. The report is written *before* a blocking error
is raised, so you can always see why validation failed.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from summarizer.entity.config_entity import DataValidationConfig
from summarizer.exception import DataValidationError
from summarizer.logger import get_logger
from summarizer.utils.common import save_json
from summarizer.utils.text_stats import describe_lengths, word_counts

logger = get_logger(__name__)

TRAIN_SPLIT = "train"
MAX_EXAMPLE_IDS = 5

# SAMSum marks shared files with tags such as <file_photo>, <file_gif>, <photo>.
MEDIA_TAG_PATTERN = re.compile(
    r"<[a-z_:]*(?:file|photo|gif|video|image|picture|link)[a-z_:]*>", re.I
)
URL_PATTERN = re.compile(r"https?://|www\.", re.I)


@dataclass
class CheckResult:
    """Outcome of one row-level check or warning on one split."""

    name: str
    description: str
    severity: str  # "drop" or "warning"
    count: int
    example_ids: list[str] = field(default_factory=list)


@dataclass
class SplitReport:
    split: str
    rows_in: int = 0
    rows_out: int = 0
    rows_dropped: int = 0
    drop_fraction: float = 0.0
    checks: list[CheckResult] = field(default_factory=list)
    dialogue_words: dict[str, float] = field(default_factory=dict)
    summary_words: dict[str, float] = field(default_factory=dict)
    blocking_errors: list[str] = field(default_factory=list)


class DataValidation:
    """Validate raw splits and write the clean subset plus a JSON report."""

    def __init__(self, config: DataValidationConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ io
    def load_split(self, split: str) -> pd.DataFrame | None:
        path = self.config.raw_dir / f"{split}.csv"
        if not path.is_file():
            return None
        # dtype=object (not str): with pandas 3, dtype=str becomes the new
        # string dtype; plain objects keep NaN for empty cells, which we count.
        return pd.read_csv(path, dtype=object)

    # ------------------------------------------------------------- checks
    def check_schema(self, df: pd.DataFrame) -> list[str]:
        """Blocking structural checks for one split."""
        errors = []
        missing = [c for c in self.config.required_columns if c not in df.columns]
        if missing:
            errors.append(f"missing required columns: {missing}")
        if df.empty:
            errors.append("split has no rows")
        return errors

    @staticmethod
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Trim surrounding whitespace; keep NaN so missing values stay detectable."""
        out = df.copy()
        for column in ("id", "dialogue", "summary"):
            out[column] = out[column].map(lambda v: v.strip() if isinstance(v, str) else v)
        return out

    def row_checks(self, df: pd.DataFrame) -> dict[str, tuple[str, pd.Series]]:
        """Return ``{check_name: (description, mask_of_rows_to_drop)}``."""
        cfg = self.config
        dialogue_words = word_counts(df["dialogue"])
        summary_words = word_counts(df["summary"])
        checks: dict[str, tuple[str, pd.Series]] = {
            "missing_dialogue": ("dialogue is null", df["dialogue"].isna()),
            "missing_summary": ("summary is null", df["summary"].isna()),
            "empty_dialogue": (
                "dialogue is empty or whitespace only",
                df["dialogue"].notna() & (dialogue_words == 0),
            ),
            "empty_summary": (
                "summary is empty or whitespace only",
                df["summary"].notna() & (summary_words == 0),
            ),
            "dialogue_too_short": (
                f"dialogue has fewer than {cfg.min_dialogue_words} words",
                (dialogue_words > 0) & (dialogue_words < cfg.min_dialogue_words),
            ),
            "dialogue_too_long": (
                f"dialogue has more than {cfg.max_dialogue_words} words",
                dialogue_words > cfg.max_dialogue_words,
            ),
            "summary_too_short": (
                f"summary has fewer than {cfg.min_summary_words} words",
                (summary_words > 0) & (summary_words < cfg.min_summary_words),
            ),
        }
        if cfg.drop_duplicates:
            checks["duplicate_id"] = (
                "id already seen earlier in the split",
                df["id"].notna() & df["id"].duplicated(keep="first"),
            )
            checks["duplicate_pair"] = (
                "identical (dialogue, summary) pair already seen",
                df["dialogue"].notna() & df.duplicated(["dialogue", "summary"], keep="first"),
            )
        return checks

    @staticmethod
    def warning_checks(df: pd.DataFrame) -> dict[str, tuple[str, pd.Series]]:
        """Informational checks: rows are kept, but counted in the report."""
        dialogue = df["dialogue"].fillna("")
        dialogue_words = word_counts(df["dialogue"])
        summary_words = word_counts(df["summary"])
        q1, q3 = dialogue_words.quantile([0.25, 0.75])
        upper_fence = q3 + 1.5 * (q3 - q1)
        return {
            "summary_longer_than_dialogue": (
                "summary has at least as many words as the dialogue",
                (summary_words >= dialogue_words) & (dialogue_words > 0),
            ),
            "contains_media_placeholder": (
                "dialogue contains attachment tags such as <file_photo>",
                dialogue.str.contains(MEDIA_TAG_PATTERN),
            ),
            "contains_url": ("dialogue contains a URL", dialogue.str.contains(URL_PATTERN)),
            "dialogue_length_outlier": (
                f"dialogue words above the IQR fence ({upper_fence:.0f})",
                dialogue_words > upper_fence,
            ),
            "shared_dialogue_multiple_summaries": (
                "dialogue appears more than once with different summaries "
                "(multiple human annotations - kept on purpose)",
                df["dialogue"].notna()
                & df.duplicated("dialogue", keep=False)
                & ~df.duplicated(["dialogue", "summary"], keep=False),
            ),
        }

    @staticmethod
    def _result(name: str, description: str, severity: str, mask: pd.Series, df: pd.DataFrame):
        ids = df.loc[mask, "id"].astype(str).head(MAX_EXAMPLE_IDS).tolist()
        return CheckResult(name, description, severity, int(mask.sum()), ids)

    # ------------------------------------------------------------ per split
    def validate_split(self, split: str, df: pd.DataFrame) -> tuple[pd.DataFrame, SplitReport]:
        report = SplitReport(split=split, rows_in=len(df))
        report.blocking_errors = self.check_schema(df)
        if report.blocking_errors:
            return df.iloc[0:0], report

        df = self.normalize(df[list(self.config.required_columns)])
        drop_mask = pd.Series(False, index=df.index)
        for name, (description, mask) in self.row_checks(df).items():
            report.checks.append(self._result(name, description, "drop", mask, df))
            drop_mask |= mask
        for name, (description, mask) in self.warning_checks(df).items():
            report.checks.append(self._result(name, description, "warning", mask, df))

        return df[~drop_mask].reset_index(drop=True), report

    def remove_leakage(self, validated: dict[str, pd.DataFrame], reports: dict[str, SplitReport]):
        """Drop train rows whose dialogue also appears in any other split."""
        if TRAIN_SPLIT not in validated or len(validated) < 2:
            return
        train = validated[TRAIN_SPLIT]
        held_out = set().union(
            *(set(df["dialogue"]) for s, df in validated.items() if s != TRAIN_SPLIT)
        )
        mask = train["dialogue"].isin(held_out)
        reports[TRAIN_SPLIT].checks.append(
            self._result(
                "cross_split_leakage",
                "train dialogue also appears in validation/test",
                "drop",
                mask,
                train,
            )
        )
        validated[TRAIN_SPLIT] = train[~mask].reset_index(drop=True)
        logger.info("Removed %d leaked dialogues from train", int(mask.sum()))

    # ----------------------------------------------------------------- run
    def run(self) -> dict[str, Any]:
        """Validate every split, write outputs and the report.

        Raises:
            DataValidationError: if any blocking check fails (after the report
                has been written).
        """
        validated: dict[str, pd.DataFrame] = {}
        reports: dict[str, SplitReport] = {}

        for split in self.config.splits:
            raw = self.load_split(split)
            if raw is None:
                reports[split] = SplitReport(
                    split=split, blocking_errors=[f"raw file not found: {split}.csv"]
                )
                continue
            validated[split], reports[split] = self.validate_split(split, raw)

        if self.config.remove_cross_split_leakage and not any(
            r.blocking_errors for r in reports.values()
        ):
            self.remove_leakage(validated, reports)

        for split, report in reports.items():
            df = validated.get(split)
            if df is None or report.blocking_errors:
                continue
            report.rows_out = len(df)
            report.rows_dropped = report.rows_in - report.rows_out
            report.drop_fraction = round(report.rows_dropped / max(report.rows_in, 1), 4)
            report.dialogue_words = describe_lengths(word_counts(df["dialogue"]))
            report.summary_words = describe_lengths(word_counts(df["summary"]))
            if report.drop_fraction > self.config.max_drop_fraction:
                report.blocking_errors.append(
                    f"dropped {report.drop_fraction:.1%} of rows "
                    f"(limit {self.config.max_drop_fraction:.1%}) - has the source format changed?"
                )
            if report.rows_out == 0:
                report.blocking_errors.append("no rows left after validation")

        passed = not any(r.blocking_errors for r in reports.values())
        summary = self._build_report(reports, passed)
        save_json(self.config.report_file, summary)

        if not passed:
            errors = {s: r.blocking_errors for s, r in reports.items() if r.blocking_errors}
            raise DataValidationError("Data validation failed", context={"errors": errors})

        self.config.validated_dir.mkdir(parents=True, exist_ok=True)
        for split, df in validated.items():
            path = self.config.validated_dir / f"{split}.csv"
            df.to_csv(path, index=False)
            logger.info(
                "Validated %-10s kept %6d / %6d rows -> %s",
                split,
                reports[split].rows_out,
                reports[split].rows_in,
                path,
            )
        return summary

    def _build_report(self, reports: dict[str, SplitReport], passed: bool) -> dict[str, Any]:
        return {
            "status": "passed" if passed else "failed",
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "rules": {
                "required_columns": list(self.config.required_columns),
                "min_dialogue_words": self.config.min_dialogue_words,
                "max_dialogue_words": self.config.max_dialogue_words,
                "min_summary_words": self.config.min_summary_words,
                "drop_duplicates": self.config.drop_duplicates,
                "remove_cross_split_leakage": self.config.remove_cross_split_leakage,
                "max_drop_fraction": self.config.max_drop_fraction,
            },
            "totals": {
                "rows_in": sum(r.rows_in for r in reports.values()),
                "rows_out": sum(r.rows_out for r in reports.values()),
            },
            "splits": {split: asdict(r) for split, r in reports.items()},
        }
