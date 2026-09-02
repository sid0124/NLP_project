"""PDF document parser for extracting structured paper documents (master spec §10).

Parses raw PDF bytes or file paths into canonical `PaperDocument` objects with titled
sections, paragraphs, references, and page numbers.
"""

from __future__ import annotations

import io
import re
from datetime import date
from pathlib import Path
from typing import BinaryIO

from src.preprocessing.sections import parse_text_into_sections
from src.schemas.paper import PaperDocument, PaperSection, Paragraph

__all__ = ["PDFPaperParser"]


class PDFPaperParser:
    """Parser for converting PDF files into structured PaperDocument objects."""

    def __init__(self) -> None:
        pass

    def parse_bytes(self, content: bytes, filename: str = "uploaded.pdf") -> PaperDocument:
        """Parse raw PDF or text bytes into a PaperDocument.

        Args:
            content: Raw byte payload.
            filename: Original file name.

        Returns:
            A populated :class:`src.schemas.paper.PaperDocument`.
        """
        text = ""
        # Try extracting text using PyPDF or fallback layout parsing
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(content))
            pages_text: list[str] = []
            for i, page in enumerate(reader.pages):
                page_str = page.extract_text() or ""
                pages_text.append(f"--- Page {i + 1} ---\n" + page_str)
            text = "\n\n".join(pages_text)
        except Exception:
            # Fallback: decode raw text content if not a binary PDF
            text = content.decode("utf-8", errors="replace")

        return self.parse_text(text, filename=filename)

    def parse_file(self, file_path: str | Path) -> PaperDocument:
        """Parse a PDF file path into a PaperDocument."""
        path = Path(file_path)
        content = path.read_bytes()
        return self.parse_bytes(content, filename=path.name)

    def parse_text(self, text: str, filename: str = "document.pdf") -> PaperDocument:
        """Parse extracted plain text into a structured PaperDocument."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        # Extract title (first substantial line or filename without extension)
        paper_id = re.sub(r"[^\w\-]", "_", Path(filename).stem)
        title = lines[0] if lines else Path(filename).stem.replace("_", " ").title()

        # Extract abstract excerpt if present
        abstract = None
        abstract_match = re.search(r"\bAbstract\b[:\s\n]*(.*?)(?=\n\n|\b1\.\s+|\bIntroduction\b|$)", text, re.IGNORECASE | re.DOTALL)
        if abstract_match:
            abstract = abstract_match.group(1).strip()[:2000]

        sections = parse_text_into_sections(text, title=title, abstract=abstract)

        # Extract reference section strings if present
        references: list[str] = []
        ref_match = re.search(r"\bReferences\b[:\s\n]*(.*)", text, re.IGNORECASE | re.DOTALL)
        if ref_match:
            ref_text = ref_match.group(1).strip()
            references = [r.strip() for r in re.split(r"\n\n|\[\d+\]", ref_text) if len(r.strip()) > 10][:50]

        return PaperDocument(
            paper_id=paper_id,
            source="pdf_parser",
            title=title,
            abstract=abstract,
            sections=sections,
            references=references,
            publication_year=date.today().year,
        )

