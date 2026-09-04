"""Unit tests for sections-to-sentence-units (spec §3 / §7)."""

from src.preprocessing.section_parser import sections_to_sentence_units
from src.preprocessing.sections import parse_text_into_sections
from src.preprocessing.text_preprocessor import TextPreprocessor


def test_sentence_units_from_plain_text():
    raw = """1. Introduction
We study attention. It works well.

2. Methodology
We propose a model. We evaluate it."""

    sections = parse_text_into_sections(raw, title="Sample", abstract="An abstract.")
    units = sections_to_sentence_units(sections)
    assert len(units) >= 4
    by_name = {u.canonical_name for u in units}
    assert "introduction" in by_name
    assert "methodology" in by_name
    assert all(u.text.strip() for u in units)


def test_max_sentences_per_section():
    raw = """1. Intro
One sentence. Two sentence. Three sentence. Four sentence."""
    sections = parse_text_into_sections(raw)
    capped = sections_to_sentence_units(sections, max_sentences_per_section=2)
    intro_units = [u for u in capped if u.canonical_name in ("introduction", "title")]
    assert len(intro_units) <= 2


def test_text_preprocessor_callable():
    proc = TextPreprocessor(lowercase=True, remove_urls=True)
    cleaned = proc("Hello WORLD.  See https://example.com")
    assert cleaned == "hello world. see"
    assert "https://" not in cleaned


def test_preprocessor_handles_none():
    proc = TextPreprocessor()
    assert proc(None) == ""
    assert proc("") == ""