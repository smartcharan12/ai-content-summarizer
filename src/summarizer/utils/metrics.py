"""Summarization metrics.

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) compares a
generated summary with a human reference by counting overlapping units.
We report the F1 score (harmonic mean of precision and recall):

* **ROUGE-1**: overlap of single words. Does the summary mention the right content?
* **ROUGE-2**: overlap of word pairs (bigrams). Does it phrase things the
  way the reference does? Always much lower than ROUGE-1 on SAMSum, because
  78% of reference bigrams don't even occur in the dialogue (see EDA §9).
* **ROUGE-L**: longest common subsequence: words in the same order, gaps
  allowed. Rewards correct sentence structure.
* **ROUGE-Lsum**: ROUGE-L computed per sentence and unioned. This is the
  variant most summarization papers report.

Stemming is on ("meets" matches "meet"), as in the original ROUGE script.

**BLEU** (from machine translation) is precision-oriented n-gram overlap with a
brevity penalty. It is reported for completeness; ROUGE is the standard
metric for summarization.

ROUGE measures word overlap, not truth. A fluent summary that attributes an
action to the wrong speaker can still score well, so always read the saved
samples alongside the numbers.
"""

from __future__ import annotations

import re

import numpy as np

ROUGE_TYPES = ("rouge1", "rouge2", "rougeL", "rougeLsum")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> str:
    """ROUGE-Lsum expects one sentence per line."""
    return "\n".join(s for s in _SENTENCE_END.split(text.strip()) if s)


def rouge_scores(predictions: list[str], references: list[str]) -> dict[str, list[float]]:
    """Per-example ROUGE F1 scores (0-1) for each ROUGE type."""
    from rouge_score import rouge_scorer

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    scorer = rouge_scorer.RougeScorer(list(ROUGE_TYPES), use_stemmer=True)
    scores: dict[str, list[float]] = {t: [] for t in ROUGE_TYPES}
    for pred, ref in zip(predictions, references, strict=True):
        result = scorer.score(_sentences(ref), _sentences(pred))
        for t in ROUGE_TYPES:
            scores[t].append(result[t].fmeasure)
    return scores


def bleu_score(predictions: list[str], references: list[str]) -> float:
    """Corpus-level SacreBLEU (0-100)."""
    import sacrebleu

    return float(sacrebleu.corpus_bleu(predictions, [references]).score)


def bootstrap_ci(
    values: list[float], resamples: int = 1000, seed: int = 42, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    With 100-200 test dialogues, two decoding settings can differ by 0.5 ROUGE
    just by chance. If their intervals overlap heavily, don't claim one is better.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or resamples == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(resamples, arr.size), replace=True).mean(axis=1)
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def has_repeated_ngram(text: str, n: int = 3) -> bool:
    """True if any word n-gram occurs twice, e.g. "will meet at 5 will meet at 5"."""
    words = text.lower().split()
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return len(grams) != len(set(grams))


def word_count(text: str) -> int:
    return len(text.split())


def compression_ratio(source: str, summary: str) -> float:
    """How many times shorter the summary is than the source (in words)."""
    return word_count(source) / max(word_count(summary), 1)
