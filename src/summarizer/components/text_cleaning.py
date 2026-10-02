"""Text cleaning shared by training (Stage 3) and inference (the API).

Using ONE function in both places prevents *train/serve skew*: if the model
was trained on text where ``<file_photo>`` became ``[photo]``, user input at
inference time must be transformed in exactly the same way.

What the EDA (notebooks/EDA.ipynb) and a tokenizer check told us:

* The PEGASUS SentencePiece vocabulary has no ``<`` character. A tag such as
  ``<file_photo>`` becomes ``<unk> file _ photo >``, and the heart emoticon
  ``<3`` (589 occurrences) becomes ``<unk> 3``. We rewrite both into forms the
  tokenizer knows: ``[photo]`` and ``(heart)``.
* SAMSum has ~40 spellings of attachment tags, including typos such as
  ``<fil_gif>``, ``<picutre>`` and ``<othre_file>``. They are mapped onto six
  canonical tags. Tags we don't recognise (``<lol>``) are left alone.
* Newlines are dropped by the tokenizer anyway. Turns stay recognisable
  because every line starts with ``Name:``, so we keep the newlines and only
  tidy whitespace within lines.
* No lowercasing by default: PEGASUS is cased, and 97% of reference
  summaries mention speakers by name.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

# Order matters: the first matching rule wins (e.g. <link_photo> -> photo).
_TAG_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("photo", re.compile(r"photo|pic|image|foto")),
    ("gif", re.compile(r"gif")),
    ("video", re.compile(r"video|movie")),
    ("audio", re.compile(r"audio|voice")),
    ("link", re.compile(r"link|url")),
    ("file", re.compile(r"fil|doc|other|othre|excel|pdf|assignment|paper")),
]
# A "tag" is <...> containing only letters, digits, _ : - (so "<3" or "<`∀´>" don't match).
_TAG_PATTERN = re.compile(r"<([a-z0-9_:\-]{2,40})>", re.IGNORECASE)
_HEART_PATTERN = re.compile(r"<3+")
_SPACES_PATTERN = re.compile(r"[^\S\n]+")  # whitespace except newlines


def _canonical_tag(match: re.Match[str]) -> str:
    name = match.group(1).lower()
    for canonical, rule in _TAG_RULES:
        if rule.search(name):
            return f"[{canonical}]"
    return match.group(0)  # unknown tag such as <lol>: leave untouched


@dataclass(frozen=True)
class TextCleaner:
    """Configurable, deterministic text normaliser.

    Args:
        normalize_whitespace: Collapse repeated spaces/tabs, strip each line,
            drop blank lines.
        normalize_tags: Rewrite attachment tags and ``<3`` into
            tokenizer-friendly forms.
        lowercase: Lowercase everything (off for PEGASUS, which is cased).
    """

    normalize_whitespace: bool = True
    normalize_tags: bool = True
    lowercase: bool = False

    def clean(self, text: str | None) -> str:
        """Return the cleaned version of ``text`` (``None`` becomes ``""``)."""
        if not text:
            return ""
        # NFKC folds look-alike characters (full-width letters, non-breaking
        # spaces) into their plain forms; html.unescape fixes "&amp;" etc.
        text = unicodedata.normalize("NFKC", html.unescape(str(text)))
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        if self.normalize_tags:
            text = _TAG_PATTERN.sub(_canonical_tag, text)
            text = _HEART_PATTERN.sub("(heart)", text)

        if self.normalize_whitespace:
            lines = (_SPACES_PATTERN.sub(" ", line).strip() for line in text.split("\n"))
            text = "\n".join(line for line in lines if line)
        else:
            text = text.strip()

        if self.lowercase:
            text = text.lower()
        return text

    def clean_batch(self, texts: list[str | None]) -> list[str]:
        return [self.clean(t) for t in texts]
