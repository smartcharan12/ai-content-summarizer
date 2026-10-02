"""Tests for summarization metrics."""

from __future__ import annotations

import math

import pytest

from summarizer.utils.metrics import (
    bleu_score,
    bootstrap_ci,
    compression_ratio,
    has_repeated_ngram,
    rouge_scores,
)


def test_identical_texts_score_one():
    scores = rouge_scores(["Anna will meet Bob at noon."], ["Anna will meet Bob at noon."])
    assert all(v == [pytest.approx(1.0)] for v in scores.values())


def test_disjoint_texts_score_zero():
    scores = rouge_scores(["completely different words"], ["Anna meets Bob"])
    assert scores["rouge1"] == [0.0] and scores["rouge2"] == [0.0]


def test_rouge1_is_f1_of_word_overlap():
    # prediction covers 3 of the reference's 6 words with precision 1.0 -> F1 = 2/3
    scores = rouge_scores(["the cat sat"], ["the cat sat on the mat"])
    assert scores["rouge1"][0] == pytest.approx(2 / 3, abs=1e-3)


def test_rouge_uses_stemming():
    scores = rouge_scores(["Anna meets Bob"], ["Anna meet Bob"])
    assert scores["rouge1"][0] == pytest.approx(1.0)


def test_rouge2_is_stricter_than_rouge1():
    scores = rouge_scores(["Bob and Anna will meet"], ["Anna and Bob will meet"])
    assert scores["rouge2"][0] < scores["rouge1"][0] == pytest.approx(1.0)


def test_rouge_length_mismatch():
    with pytest.raises(ValueError):
        rouge_scores(["a"], ["a", "b"])


def test_bleu_identical_is_100():
    assert bleu_score(["Anna will call Bob tomorrow ."], ["Anna will call Bob tomorrow ."]) == (
        pytest.approx(100.0)
    )


def test_bootstrap_ci():
    low, high = bootstrap_ci([0.2, 0.4, 0.6, 0.8] * 25, resamples=500)
    assert low < 0.5 < high
    assert bootstrap_ci([0.3] * 10, resamples=100) == (pytest.approx(0.3), pytest.approx(0.3))
    assert all(math.isnan(x) for x in bootstrap_ci([]))


def test_repeated_ngram_detection():
    assert has_repeated_ngram("they will meet at five and they will meet at five")
    assert not has_repeated_ngram("Anna will meet Bob at five.")


def test_compression_ratio_never_divides_by_zero():
    assert compression_ratio("one two three four", "one two") == 2.0
    assert compression_ratio("one two", "") == 2.0
