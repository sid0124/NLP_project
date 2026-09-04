"""Configurable text preprocessor for dataset pipelines (spec §3).

Wraps the pure cleaning functions from :mod:`src.preprocessing.text` behind a
single configurable entry point so the loaders and the hierarchical encoder
apply identical normalisation without re-deriving keyword flags.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.preprocessing.text import clean_text

__all__ = ["TextPreprocessor", "make_preprocessor"]


@dataclass(frozen=True, slots=True)
class TextPreprocessor:
    """A frozen set of cleaning flags applied to text.

    Attributes match :func:`src.preprocessing.text.clean_text` keywords so a
    preprocessor is a named bundle of normalisation choices.
    """

    lowercase: bool = False
    remove_urls: bool = True
    apply_nfkc: bool = True
    squeeze_whitespace: bool = True
    max_chars: int | None = None

    def __call__(self, text: str | None) -> str:
        """Clean ``text`` with this preprocessor's flags."""
        return clean_text(
            text,
            lowercase=self.lowercase,
            remove_urls=self.remove_urls,
            apply_nfkc=self.apply_nfkc,
            squeeze_whitespace=self.squeeze_whitespace,
            max_chars=self.max_chars,
        )


def make_preprocessor(*, lowercase: bool = False, max_chars: int | None = None) -> TextPreprocessor:
    """Build a :class:`TextPreprocessor` from common dataset settings.

    Args:
        lowercase: Lowercase model input.
        max_chars: Optional character cap applied last.

    Returns:
        A ready-to-call :class:`TextPreprocessor`.
    """
    return TextPreprocessor(lowercase=lowercase, max_chars=max_chars)