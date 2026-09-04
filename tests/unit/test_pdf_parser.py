"""Unit tests for PDFPaperParser (spec §30: corrupted/scanned/empty inputs)."""

import pytest

from src.ingestion.pdf_parser import PDFPaperParser, PDFParseError, is_scanned_pdf


def test_parse_text():
    sample_text = """1. Introduction
This paper introduces deep learning methods for natural language processing.

2. Methodology
We propose a transformer model evaluated on the SQuAD dataset using F1 score.

3. Results
Our model achieves competitive performance.

References
[1] Vaswani et al. Attention Is All You Need. 2017."""

    parser = PDFPaperParser()
    doc = parser.parse_text(sample_text, filename="sample_paper.pdf")

    assert doc.paper_id == "sample_paper"
    assert doc.title == "1. Introduction"
    assert len(doc.sections) > 0
    assert len(doc.references) > 0


def test_parse_bytes_plain_text():
    parser = PDFPaperParser()
    doc = parser.parse_bytes(b"Title line\n\nSome abstract body text here.", filename="note.txt")
    assert doc.paper_id == "note"
    assert doc.source == "pdf_parser"


def test_parse_empty_bytes_raises():
    parser = PDFPaperParser()
    with pytest.raises(PDFParseError):
        parser.parse_bytes(b"", filename="empty.pdf")


def test_parse_binary_non_pdf_raises():
    parser = PDFPaperParser()
    with pytest.raises(PDFParseError):
        parser.parse_bytes(b"\x00\x01\x02\xff\xfe binary junk", filename="bogus.bin")


def test_scanned_pdf_detection():
    assert is_scanned_pdf("")
    assert is_scanned_pdf("   ")
    assert is_scanned_pdf("a" * 10)
    real = (
        "This is a real paragraph with more than sixty characters of "
        "extractable text, which is a normal length for one sentence."
    )
    assert not is_scanned_pdf(real)


def test_parse_missing_file_raises(tmp_path):
    parser = PDFPaperParser()
    with pytest.raises(PDFParseError):
        parser.parse_file(tmp_path / "does_not_exist.pdf")

