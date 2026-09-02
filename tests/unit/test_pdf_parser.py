"""Unit tests for PDFPaperParser."""

from src.ingestion.pdf_parser import PDFPaperParser


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

