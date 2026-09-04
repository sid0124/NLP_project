"""Sentence segmentation for academic paper text (spec §7/§8).

The HAN input contract is sentences: sections are decomposed into labelled
sentences before the Transformer touches them. Naive splitting on ``[.!?]``
mis-splits academic prose — abbreviations (``et al.``, ``Fig.``, ``e.g.``),
decimals, figure references, and citation markers all end in periods. This
module is a deterministic, dependency-free splitter tuned for that text.

Two entry points:

* :func:`split_sentences` — a list of sentence strings (what most consumers
  need; used by the hierarchical encoder and the RAG chunker).
* :func:`split_sentences_with_spans` — character spans into the **original**
  text, so callers can attribute each sentence to a source page or section.

A sentence is ended only by ``.``, ``!`` or ``?`` followed by whitespace (or
end-of-text), unless the preceding token is a known abbreviation, a single
capital initial (``J. Smith``), or a decimal/version number (``0.5``,
``Section 2.1``). Line breaks inside a paragraph are treated as spaces
(:func:`split_sentences` flattens them; the span variant keeps them).
"""

from __future__ import annotations

import re

__all__ = [
    "looks_like_abbreviation",
    "split_sentences",
    "split_sentences_with_spans",
]

#: Common abbreviations whose trailing period does not end a sentence.
_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        # Academic / technical
        "al",  # et al.
        "approx", "appendix", "cf", "chap", "ed", "eds", "eq", "eqs",
        "et", "etc", "fig", "figs", "i.e", "e.g", "ibid", "no", "nos",
        "p", "pp", "ref", "refs", "sec", "secs", "st", "trans", "vol",
        "vols", "vs",
        # Titles / honorifics
        "dr", "prof", "mr", "mrs", "ms", "jr", "sr", "rev", "hon",
        "gen", "col", "capt", "lt", "sgt",
        # Units / institutions
        "inc", "ltd", "co", "corp", "dept", "univ", "assoc", "est",
        "min", "max", "avg", "std",
    }
)

#: A single capital initial between words ("J. Smith", "M. I. Jordan").
_INITIAL_RE = re.compile(r"^[A-Z]$")
#: A decimal number ("0.5", "3.14").
_DECIMAL_RE = re.compile(r"^\d+\.\d+$")
#: Versioned numbers ("2.1", "3.2.1").
_VERSIONED_RE = re.compile(r"^\d+(?:\.\d+)+$")
#: Acronyms with internal periods ("U.S.A.") and bracketed figure refs ("Fig.").
_ACRONYM_RE = re.compile(r"^(?:[A-Z]\.)+[A-Z]$")


def looks_like_abbreviation(candidate: str) -> bool:
    """True when ``candidate`` ends with a period that is not a boundary.

    Inspects the final whitespace-delimited token of the candidate, stripping
    trailing closing punctuation, and checks it against the abbreviation
    table, single-capital initials, decimals, versioned numbers, and
    internal-period acronyms such as ``U.S.A.``
    """
    cleaned = candidate.strip()
    if not cleaned:
        return False
    core = cleaned.rstrip(".!?")
    if not core:
        return False
    last_word = core.split()[-1].rstrip(")]}\"'")
    key = last_word.lower()
    if key in _ABBREVIATIONS:
        return True
    return bool(
        _INITIAL_RE.match(last_word)
        or _DECIMAL_RE.match(last_word)
        or _VERSIONED_RE.match(last_word)
        or _ACRONYM_RE.match(last_word)
    )

_TERMINATORS = (".", "!", "?")


def split_sentences_with_spans(text: str | None) -> list[tuple[str, int, int]]:
    """Split ``text`` into sentences, returning ``(sentence, start, end)`` tuples.

    Spans index into the **original** string (preserving line breaks) so
    callers can trace each sentence back to its source page or section.
    ``start`` is the offset of the first non-space character and ``end`` the
    offset of the last character of the sentence. Trailing quotes or closing
    parens are kept inside the sentence and counted in ``end``.

    Args:
        text: Raw academic text, possibly ``None``.

    Returns:
        ``(sentence, start, end)`` tuples in reading order; an empty list for
        ``None`` or blank input.
    """
    if not text or not text.strip():
        return []

    results: list[tuple[str, int, int]] = []
    buffer: list[str] = []
    start: int | None = None

    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if start is None:
            if char.isspace():
                index += 1
                continue
            start = index
        buffer.append(char)

        if char in _TERMINATORS:
            # Include trailing quotes / closing characters in this sentence.
            end = index
            probe = index + 1
            while probe < length and text[probe] in ")]}\"'”’":
                end = probe
                probe += 1
            # A terminator splits only when followed by whitespace or EOT.
            is_boundary = probe >= length or text[probe].isspace()
            if is_boundary:
                for j in range(index + 1, probe):
                    buffer.append(text[j])
                candidate = "".join(buffer).strip()
                if looks_like_abbreviation(candidate):
                    # The period here is an abbreviation's, not a boundary.
                    index = probe
                    continue
                results.append((candidate, start, end))
                buffer = []
                start = None
                index = probe
                continue
            index = probe
            continue

        index += 1

    if buffer:
        tail = "".join(buffer).strip()
        if tail and start is not None:
            results.append((tail, start, length - 1))

    return results


def split_sentences(text: str | None) -> list[str]:
    """Split ``text`` into a list of sentence strings.

    Normalises internal line breaks to spaces first — PDF-extracted and
    LaTeX-wrapped text arrive that way and the HAN input contract treats a
    sentence as one logical line of prose — then splits exactly like
    :func:`split_sentences_with_spans`.

    Args:
        text: Raw academic text, possibly ``None``.

    Returns:
        Non-empty sentences in reading order.
    """
    if not text or not text.strip():
        return []

    flattened = re.sub(r"[ \t\f\v]+| *\r?\n *", " ", text).strip()
    return [sentence for sentence, _, _ in split_sentences_with_spans(flattened)]