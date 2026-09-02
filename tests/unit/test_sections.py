"""Unit tests for section parsing and canonical mapping."""

from src.preprocessing.sections import map_to_canonical_section, parse_text_into_sections
from src.schemas.paper import PaperSection, Paragraph


def test_map_to_canonical_section():
    assert map_to_canonical_section("1. Introduction") == "introduction"
    assert map_to_canonical_section("Abstract") == "abstract"
    assert map_to_canonical_section("3. Methodology & Approach") == "methodology"
    assert map_to_canonical_section("4. Experimental Results") == "results"
    assert map_to_canonical_section("Conclusion and Future Work") == "conclusion"
    assert map_to_canonical_section("Random Unrecognized Heading") == "other"


def test_parse_text_into_sections_from_plain_text():
    raw_text = """1. Introduction
This paper introduces a novel deep learning model for classification.

2. Methodology
We propose a hierarchical attention mechanism over sections and paragraphs.

3. Results
Our model achieves state of the art accuracy on the benchmark dataset."""

    sections = parse_text_into_sections(raw_text, title="Sample Paper", abstract="An abstract summary.")
    assert len(sections) >= 4
    canonical_names = [s.canonical_name for s in sections]
    assert "title" in canonical_names
    assert "abstract" in canonical_names
    assert "introduction" in canonical_names
    assert "methodology" in canonical_names


def test_parse_text_into_sections_with_existing():
    existing = [
        PaperSection(
            section_id="sec-1",
            section_name="Background & Prior Work",
            section_order=0,
            canonical_name="related_work",
            paragraphs=[Paragraph(paragraph_id="p-1", paragraph_order=0, text="Related work content.")],
        )
    ]
    sections = parse_text_into_sections(text="", title="Title", abstract="Abstract", existing_sections=existing)
    assert len(sections) == 3
    assert sections[2].canonical_name == "related_work"

