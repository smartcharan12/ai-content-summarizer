"""Tests for TextCleaner (shared by training and inference)."""

from __future__ import annotations

import pytest

from summarizer.components.text_cleaning import TextCleaner

cleaner = TextCleaner()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<file_photo>", "[photo]"),
        ("<photo_file>", "[photo]"),
        ("<file:photo>", "[photo]"),
        ("<file-photo>", "[photo]"),
        ("<picutre>", "[photo]"),  # typo in the dataset
        ("<file_foto>", "[photo]"),
        ("<link_photo>", "[photo]"),  # photo wins over link
        ("<file_gif>", "[gif]"),
        ("<fil_gif>", "[gif]"),
        ("<file_movie>", "[video]"),
        ("<file_audio>", "[audio]"),
        ("<file_link>", "[link]"),
        ("<file_other>", "[file]"),
        ("<othre_file>", "[file]"),
        ("<file_docx>", "[file]"),
        ("<FILE_PHOTO>", "[photo]"),
    ],
)
def test_attachment_tags_are_canonicalised(raw, expected):
    assert cleaner.clean(f"Anna: look {raw}") == f"Anna: look {expected}"


@pytest.mark.parametrize("tag", ["<lol>", "<emoticon_smile>", "<m(__)m>"])
def test_unknown_tags_are_left_alone(tag):
    assert cleaner.clean(f"Bob: {tag}") == f"Bob: {tag}"


def test_heart_emoticon():
    assert cleaner.clean("Anna: love you <3 <333") == "Anna: love you (heart) (heart)"


def test_whitespace_normalised_but_turns_kept():
    raw = "  Anna:   hi \t there  \r\n\r\n   Bob:  hey\n"
    assert cleaner.clean(raw) == "Anna: hi there\nBob: hey"


def test_unicode_and_html_entities():
    # full-width letters and a non-breaking space become plain ASCII; &amp; -> &
    assert cleaner.clean("Ｈｅｌｌｏ Tom &amp; Jerry") == "Hello Tom & Jerry"


def test_case_preserved_by_default_and_lowercase_flag():
    assert cleaner.clean("Amanda: Hi") == "Amanda: Hi"
    assert TextCleaner(lowercase=True).clean("Amanda: Hi") == "amanda: hi"


def test_flags_can_disable_steps():
    raw = "Anna:  look <file_photo>"
    assert TextCleaner(normalize_tags=False).clean(raw) == "Anna: look <file_photo>"
    assert TextCleaner(normalize_whitespace=False).clean(raw) == "Anna:  look [photo]"


@pytest.mark.parametrize("empty", [None, "", "   ", "\n\n"])
def test_empty_input(empty):
    assert cleaner.clean(empty) == ""


def test_cleaning_is_idempotent():
    raw = "Anna:  <file_photo> <3\r\nBob: ok &amp; bye"
    once = cleaner.clean(raw)
    assert cleaner.clean(once) == once


def test_clean_batch():
    assert cleaner.clean_batch(["a  b", None]) == ["a b", ""]
