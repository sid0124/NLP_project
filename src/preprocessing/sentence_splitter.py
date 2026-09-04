"""Sentence segmentation facade (spec §3 / §7).

The evaluation is implemented in :mod:`src.preprocessing.sentences`; this
module exposes it under the canonical ``sentence_splitter`` name so the dataset
pipeline, the hierarchical encoder, and the RAG chunker share one import path.
"""

from __future__ import annotations

from src.preprocessing.sentences import (
    looks_like_abbreviation,
    split_sentences,
    split_sentences_with_spans,
)

__all__ = [
    "looks_like_abbreviation",
    "split_sentences",
    "split_sentences_with_spans",
]