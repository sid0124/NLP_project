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
from src.utils.logging import get_logger

__all__ = ["PDFPaperParser", "PDFParseError", "is_scanned_pdf"]

logger = get_logger(__name__)

#: A file smaller than this cannot be a real PDF (the header alone is 5 bytes).
_MIN_PDF_BYTES = 32
#: Extractable-text threshold below which a PDF is treated as scanned/empty.
_MIN_EXTRACTED_CHARS = 60


class PDFParseError(ValueError):
    """Raised when a PDF cannot be parsed into a usable PaperDocument.

    ``message`` is safe for end users (no stack traces, no file internals);
    the original exception, when any, is available as ``__cause__`` for logs.
    """


def _is_pdf_header(content: bytes) -> bool:
    """Return True when ``content`` starts with the ``%PDF-`` magic bytes."""
    return content[:5] == b"%PDF-"


def is_scanned_pdf(text: str, *, min_chars: int = _MIN_EXTRACTED_CHARS) -> bool:
    """Heuristic: a PDF produced no meaningful embedded text.

    Scanned documents (image-only pages) typically yield a handful of stray
    characters; anything under ``min_chars`` of extractable text is treated as
    scanned or image-based.
    """
    return len(text.strip()) < min_chars


class PDFPaperParser:
    """Parser for converting PDF files into structured PaperDocument objects."""

    def parse_bytes(
        self, content: bytes, filename: str = "uploaded.pdf", *, encoding: str = "utf-8"
    ) -> PaperDocument:
        """Parse raw PDF or plain-text bytes into a PaperDocument.

        Args:
            content: Raw byte payload. A ``%PDF-`` payload is parsed with
                pypdf; anything else is decoded as plain text and parsed with
                :meth:`parse_text` (so ``.txt`` uploads keep working).
            filename: Original file name (affects the paper id/title).
            encoding: Text encoding used for non-PDF content.

        Returns:
            A populated :class:`src.schemas.paper.PaperDocument`.

        Raises:
            PDFParseError: If the payload is empty, a corrupt/unreadable PDF,
                a scanned PDF with no extractable text, or a non-PDF binary
                file that cannot be decoded as text.
        """
        if not content:
            raise PDFParseError("The uploaded file is empty.")

        if not _is_pdf_header(content):
            # Plain-text (or unknown) content: only accept decodable text.
            try:
                text = content.decode(encoding)
            except (UnicodeDecodeError, LookupError):
                raise PDFParseError(
                    f"'{filename}' is not a PDF and could not be read as text "
                    f"({encoding} decoding failed)."
                ) from None
            if not text.strip():
                raise PDFParseError(f"'{filename}' contains no readable text.")
            return self.parse_text(text, filename=filename)

        return self._parse_pdf_bytes(content, filename)

    def parse_file(self, file_path: str | Path) -> PaperDocument:
        """Parse a PDF file path into a PaperDocument.

        Raises:
            PDFParseError: If the file is missing, unreadable, or unparseable.
            OSError: For permission errors reading the file.
        """
        path = Path(file_path)
        if not path.is_file():
            raise PDFParseError(f"PDF file not found: {path}")
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise PDFParseError(f"Could not read PDF file '{path.name}': {exc}") from exc
        return self.parse_bytes(content, filename=path.name)

    def _parse_pdf_bytes(self, content: bytes, filename: str) -> PaperDocument:
        """Extract page text with pypdf and build the document hierarchy."""
        if len(content) < _MIN_PDF_BYTES:
            raise PDFParseError(
                f"'{filename}' is too small to be a valid PDF ({len(content)} bytes). "
                "The file may be truncated or corrupted."
            )

        try:
            from pypdf import PdfReader
        except ImportError:  # pragma: no cover - environment dependent
            raise PDFParseError(
                "PDF parsing requires the 'pypdf' package, which is not installed. "
                "Install it with: pip install -r requirements.txt"
            ) from None

        try:
            reader = PdfReader(io.BytesIO(content), strict=False)
            pages: list[str] = []
            for page in reader.pages:
                try:
                    page_text = page.extract_text() or ""
                except Exception as exc:  # noqa: BLE001 - per-page robustness
                    logger.warning("pdf_parser | page extraction failed: %s", exc)
                    page_text = ""
                pages.append(page_text)
        except Exception as exc:  # noqa: BLE001 - pypdf raises broadly
            raise PDFParseError(
                f"'{filename}' could not be read as a PDF. The file may be "
                "corrupted, encrypted, or not actually a PDF."
            ) from exc

        if is_scanned_pdf("\n".join(pages)):
            raise PDFParseError(
                f"'{filename}' appears to be a scanned or image-only PDF — no "
                "embedded text was extracted. OCR is not supported yet."
            )

        return self._build_document(pages, filename)

    def parse_text(self, text: str, filename: str = "document.pdf") -> PaperDocument:
        """Parse extracted plain text into a structured PaperDocument.

        Used both for plain-text uploads and as the text path for non-PDF
        bytes. Paragraphs carry no page numbers here — none are known.
        """
        if not text or not text.strip():
            raise PDFParseError(f"'{filename}' contains no readable text.")
        return self._build_document([text], filename)

    def _build_document(self, pages: Sequence[str], filename: str) -> PaperDocument:
        """Build a PaperDocument from ordered page texts.

        The pages are joined and passed through the shared section parser, then
        page numbers are attributed to each paragraph via
        :meth:`_assign_page_numbers`.
        """
        paper_id = re.sub(r"[^\w\-]", "_", Path(filename).stem)
        full_text = "\n\n".join(pages)

        lines = [line.strip() for line in full_text.splitlines() if line.strip()]
        title = lines[0] if lines else Path(filename).stem.replace("_", " ").title()

        abstract = None
        abstract_match = re.search(
            r"\bAbstract\b[:\s\n]*(.*?)(?=\n\n|\b1\.\s+|\bIntroduction\b|$)",
            full_text,
            re.IGNORECASE | re.DOTALL,
        )
        if abstract_match:
            abstract = abstract_match.group(1).strip()[:2000]

        sections = parse_text_into_sections(full_text, title=title, abstract=abstract)

        references: list[str] = []
        ref_match = re.search(r"\bReferences\b[:\s\n]*(.*)", full_text, re.IGNORECASE | re.DOTALL)
        if ref_match:
            ref_text = ref_match.group(1).strip()
            references = [
                r.strip()
                for r in re.split(r"\n\s*\n|\[\d+\]\s*", ref_text)
                if len(r.strip()) > 10
            ][:50]

        self._assign_page_numbers(sections, pages)

        return PaperDocument(
            paper_id=paper_id,
            source="pdf_parser",
            title=title,
            abstract=abstract,
            sections=sections,
            references=references,
            publication_year=date.today().year,
        )

    @staticmethod
    def _assign_page_numbers(sections: list[PaperSection], pages: Sequence[str]) -> None:
        """Populate ``Paragraph.page_number`` by locating text in pages.

        Walks pages monotonically and assigns each paragraph the first page
        containing a distinguishing fragment of its start, so paragraph page
        numbers never go backwards even when a page break splits a paragraph.
        """
        if not pages:
            return

        page_index = 0
        n_pages = len(pages)
        for section in sections:
            for para in section.paragraphs:
                fragment = re.sub(r"\s+", " ", para.text)[:40]
                if not fragment:
                    continue
                while page_index < n_pages:
                    haystack = re.sub(r"\s+", " ", pages[page_index])
                    if fragment[:20] in haystack:
                        para.page_number = page_index + 1
                        break
                    page_index += 1
                # Give up silently: the paragraph may sit exactly on a page
                # boundary; page attribution is best-effort, never fatal.

