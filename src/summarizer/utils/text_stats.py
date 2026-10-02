"""Text-length helpers shared by ingestion stats, validation and the EDA notebook.

"Words" here means whitespace-separated tokens. That is cheap, independent of
any tokenizer, and good enough for data-quality checks. Model token counts
(which drive truncation) are measured separately in the EDA notebook.
"""

from __future__ import annotations

import pandas as pd


def word_counts(texts: pd.Series) -> pd.Series:
    """Number of whitespace-separated words per row (missing text counts as 0)."""
    return texts.fillna("").astype(str).str.split().str.len().astype(int)


def describe_lengths(counts: pd.Series) -> dict[str, float]:
    """Summary statistics of a length distribution, rounded for reports."""
    if counts.empty:
        return {"count": 0}
    q = counts.quantile([0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "count": int(counts.count()),
        "mean": round(float(counts.mean()), 2),
        "std": round(float(counts.std(ddof=0)), 2),
        "min": int(counts.min()),
        "p25": float(q[0.25]),
        "median": float(q[0.5]),
        "p75": float(q[0.75]),
        "p95": float(q[0.95]),
        "p99": float(q[0.99]),
        "max": int(counts.max()),
    }
