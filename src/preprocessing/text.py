"""Text normalisation for model input and for duplicate matching.

Everything here is a pure function of its arguments: same input, same output, no
I/O and no configuration lookups. That keeps cleaning testable in isolation and
keeps it safe to apply before splitting — normalisation never learns anything
from the corpus, so unlike vectoriser fitting it cannot leak information across
the train/test boundary (master spec §9).

Two normalisation strengths exist on purpose, and conflating them would be a bug
in either direction:

* :func:`clean_text` prepares **model input**. It is conservative, because
  casing, punctuation, and numerals carry signal a classifier can use.
* :func:`normalize_for_matching` prepares **duplicate-detection keys**. It is
  aggressive, because "Attention Is All You Need." and "attention is all you
  need" are the same paper and must hash identically.
"""

from __future__ import annotations

import re
import unicodedata

__all__ = [
    "clean_text",
    "collapse_whitespace",
    "normalize_for_matching",
    "normalize_unicode",
    "strip_html",
    "strip_urls",
    "truncate_chars",
]

# Matches http(s)/ftp URLs, bare www hosts, and DOI strings. Academic abstracts
# routinely contain all three, and none of them help a topic classifier.
_URL_RE = re.compile(
    r"""(?ix)
    \b(
        (?:https?|ftp)://[^\s<>"')\]]+     # scheme-qualified URL
      | www\.[^\s<>"')\]]+                 # bare www host
      | doi:\s*10\.\d{4,9}/[^\s<>"')\]]+   # doi: prefixed identifier
      | 10\.\d{4,9}/[^\s<>"')\]]+          # bare DOI
    )
    """
)

# Deliberately narrow: matches a tag only when the name looks like a tag name,
# so mathematical text such as "for n < 10 and m > 2" survives untouched.
_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^<>]*?)?/?>")

# HTML/JATS entities that survive in abstracts from publisher feeds.
_ENTITIES = {
    "&amp;": "&",
    "&lt;": "<",
    "&gt;": ">",
    "&quot;": '"',
    "&apos;": "'",
    "&nbsp;": " ",
    "&#x2009;": " ",
    "&#8201;": " ",
}

_WHITESPACE_RE = re.compile(r"\s+")

# Retained for matching keys: word characters and single spaces only.
_NON_ALNUM_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

# Zero-width characters survive NFKC but split tokens invisibly.
_ZERO_WIDTH = ("​", "‌", "‍", "﻿")


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Apply Unicode normalisation and strip zero-width characters.

    NFKC is chosen so that compatibility variants collapse onto their canonical
    forms: ligatures, full-width Latin, and superscript digits all appear in
    publisher-supplied abstracts and would otherwise produce distinct tokens for
    identical words.

    Args:
        text: Input text.
        form: Any normal form accepted by :func:`unicodedata.normalize`.

    Returns:
        The normalised text.
    """
    normalized = unicodedata.normalize(form, text)
    for char in _ZERO_WIDTH:
        normalized = normalized.replace(char, "")
    return normalized


def strip_urls(text: str) -> str:
    """Remove URLs and DOI strings, leaving a single space in their place."""
    return _URL_RE.sub(" ", text)


def strip_html(text: str) -> str:
    """Remove HTML/JATS tags and decode the entities that commonly survive.

    Abstracts sourced from publisher feeds arrive with markup fragments such as
    ``<jats:italic>`` or ``&amp;``. Entities are decoded *after* tag removal so
    that a decoded ``&lt;`` cannot be mistaken for the start of a tag.
    """
    without_tags = _TAG_RE.sub(" ", text)
    for entity, replacement in _ENTITIES.items():
        without_tags = without_tags.replace(entity, replacement)
    return without_tags


def collapse_whitespace(text: str) -> str:
    """Collapse every whitespace run to one space and strip the ends."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def truncate_chars(text: str, max_chars: int | None) -> str:
    """Truncate to ``max_chars``, preferring a word boundary.

    Args:
        text: Input text.
        max_chars: Character budget. ``None`` or non-positive returns ``text``
            unchanged.

    Returns:
        The possibly truncated text, with no trailing partial word.
    """
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    pivot = clipped.rfind(" ")
    # Only back off to the word boundary when that retains most of the budget;
    # one pathologically long token should still be cut rather than kept whole.
    if pivot > max_chars * 0.8:
        clipped = clipped[:pivot]
    return clipped.rstrip()


def clean_text(
    text: str | None,
    *,
    lowercase: bool = False,
    remove_urls: bool = True,
    apply_nfkc: bool = True,
    squeeze_whitespace: bool = True,
    max_chars: int | None = None,
) -> str:
    """Clean text for model input.

    Markup stripping is unconditional rather than a flag: leftover ``<jats:p>``
    tags are never desirable model input, so there is no meaningful choice to
    expose. The remaining steps map onto the ``text`` block of
    ``configs/config.yaml``; :mod:`src.data_pipeline.labels` performs that
    mapping, keeping this module free of any configuration dependency.

    Args:
        text: Input text, possibly ``None``.
        lowercase: Lowercase the result. Off by default because casing is usable
            signal for a linear model, and ``TfidfVectorizer`` lowercases anyway.
        remove_urls: Drop URLs and DOIs (``text.strip_urls``).
        apply_nfkc: Apply NFKC normalisation (``text.normalize_unicode``).
        squeeze_whitespace: Collapse whitespace runs (``text.collapse_whitespace``).
        max_chars: Optional character cap, applied last.

    Returns:
        The cleaned text; ``""`` when ``text`` is ``None`` or blank.
    """
    if not text:
        return ""

    result = normalize_unicode(text) if apply_nfkc else text
    result = strip_html(result)
    if remove_urls:
        result = strip_urls(result)
    if squeeze_whitespace:
        result = collapse_whitespace(result)
    if lowercase:
        result = result.lower()
    return truncate_chars(result, max_chars)


def normalize_for_matching(text: str | None) -> str:
    """Aggressively normalise text into a duplicate-matching key.

    Applies NFKC normalisation, markup and URL removal, case folding,
    punctuation removal, and whitespace collapsing. The result is unsuitable as
    model input and is only ever hashed or shingled.

    Case folding uses :meth:`str.casefold` rather than :meth:`str.lower` so that
    non-English forms compare correctly.

    Args:
        text: Input text, possibly ``None``.

    Returns:
        The matching key; ``""`` when there is nothing to match on.
    """
    if not text:
        return ""
    result = normalize_unicode(text)
    result = strip_html(result)
    result = strip_urls(result)
    result = result.casefold()
    result = _NON_ALNUM_RE.sub(" ", result)
    return collapse_whitespace(result)
