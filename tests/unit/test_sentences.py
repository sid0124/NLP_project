"""Unit tests for sentence segmentation (spec §7/§8)."""

import pytest

from src.preprocessing.sentences import (
    looks_like_abbreviation,
    split_sentences,
    split_sentences_with_spans,
)


class TestLooksLikeAbbreviation:
    def test_common_abbreviations(self):
        assert looks_like_abbreviation("et al.")
        assert looks_like_abbreviation("Fig.")
        assert looks_like_abbreviation("e.g.")
        assert looks_like_abbreviation("i.e.")
        assert looks_like_abbreviation("vs.")

    def test_initial_decimal_acronym(self):
        assert looks_like_abbreviation("J.")
        assert looks_like_abbreviation("0.5")
        assert looks_like_abbreviation("U.S.A.")

    def test_full_sentence_is_not_abbreviation(self):
        assert not looks_like_abbreviation("This is a complete sentence.")
        assert not looks_like_abbreviation("It works well.")


class TestSplitSentences:
    def test_basic_split(self):
        text = "First sentence. Second sentence! Third question? Fourth."
        result = split_sentences(text)
        assert result == [
            "First sentence.",
            "Second sentence!",
            "Third question?",
            "Fourth.",
        ]

    def test_handles_none_and_empty(self):
        assert split_sentences(None) == []
        assert split_sentences("") == []
        assert split_sentences("   \n  ") == []

    def test_abbreviation_not_a_boundary(self):
        text = "This builds on Smith et al. work. The model is shown in Fig. 2."
        result = split_sentences(text)
        assert result == [
            "This builds on Smith et al. work.",
            "The model is shown in Fig. 2.",
        ]

    def test_decimal_and_acronym(self):
        text = "We reach 94.2% accuracy on U.S.A. benchmarks. It is strong."
        result = split_sentences(text)
        assert result == [
            "We reach 94.2% accuracy on U.S.A. benchmarks.",
            "It is strong.",
        ]

    def test_newlines_flattened(self):
        text = "Line one continues here.\nAnd this is a second sentence."
        result = split_sentences(text)
        assert result == ["Line one continues here.", "And this is a second sentence."]

    def test_trailing_quote(self):
        text = 'He said "run it." Then we left.'
        result = split_sentences(text)
        assert result == ['He said "run it."', "Then we left."]


class TestSplitSentencesWithSpans:
    def test_spans_cover_input_without_overlap(self):
        text = "Alpha sentence. Beta sentence."
        items = split_sentences_with_spans(text)
        assert [(s, start, end) for s, start, end in items] == [
            ("Alpha sentence.", 0, 14),
            ("Beta sentence.", 16, 29),
        ]

    def test_spans_empty(self):
        assert split_sentences_with_spans(None) == []
        assert split_sentences_with_spans("") == []