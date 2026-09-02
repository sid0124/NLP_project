"""Section parsing and canonical mapping for structured paper documents (master spec §7).

Maps raw or noisy section headings onto canonical section names
(:data:`src.schemas.paper.CANONICAL_SECTIONS`) and parses unstructured text into
ordered section and paragraph hierarchies.
"""

from __future__ import annotations

import re
from typing import Sequence

from src.schemas.paper import CANONICAL_SECTIONS, PaperSection, Paragraph

__all__ = [
    "map_to_canonical_section",
    "parse_text_into_sections",
]

_CANONICAL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("abstract", re.compile(r"\babstract\b", re.IGNORECASE)),
    ("introduction", re.compile(r"\b(introduction|overview|background)\b", re.IGNORECASE)),
    ("related_work", re.compile(r"\b(related\s+work|prior\s+work|literature\s+review)\b", re.IGNORECASE)),
    ("methodology", re.compile(r"\b(method(ology)?|approach|proposed\s+method|model|framework|system\s+architecture)\b", re.IGNORECASE)),
    ("experiments", re.compile(r"\b(experiments?|experimental\s+setup|evaluations?|baseline(s)?)\b", re.IGNORECASE)),
    ("results", re.compile(r"\b(results?|findings|performance|experimental\s+results)\b", re.IGNORECASE)),
    ("discussion", re.compile(r"\b(discussion|analysis|limitation(s)?|ablation\s+study)\b", re.IGNORECASE)),
    ("conclusion", re.compile(r"\b(conclusion(s)?|future\s+work|concluding\s+remarks)\b", re.IGNORECASE)),
    ("references", re.compile(r"\b(references?|bibliography|citations)\b", re.IGNORECASE)),
    ("appendix", re.compile(r"\b(appendix|supplementary|supplemental)\b", re.IGNORECASE)),
)

_HEADER_PREFIX_RE = re.compile(r"^(?:(?:\d+\.)+\d*|[A-Z]\.|\d+)\s*")


def map_to_canonical_section(heading: str | None) -> str:
    """Map a raw section heading to a canonical section name.

    Args:
        heading: The raw header text (e.g. ``"1. Introduction"`` or ``"3. Experimental Setup"``).

    Returns:
        One of :data:`src.schemas.paper.CANONICAL_SECTIONS`. Defaults to ``"other"``.
    """
    if not heading:
        return "other"

    cleaned = _HEADER_PREFIX_RE.sub("", heading.strip()).strip()
    if not cleaned:
        return "other"

    for name, pattern in _CANONICAL_PATTERNS:
        if pattern.search(cleaned):
            return name

    return "other"


def parse_text_into_sections(
    text: str | None,
    title: str | None = None,
    abstract: str | None = None,
    existing_sections: Sequence[PaperSection] | None = None,
) -> list[PaperSection]:
    """Parse raw text or metadata into structured `PaperSection` instances.

    If `existing_sections` are provided and non-empty, they are updated with canonical names.
    Otherwise, section headers are inferred from the text body using double newlines and header heuristics.

    Args:
        text: Raw text body of the paper.
        title: Optional paper title.
        abstract: Optional paper abstract.
        existing_sections: Optional pre-parsed sections.

    Returns:
        List of :class:`src.schemas.paper.PaperSection` objects.
    """
    sections: list[PaperSection] = []
    order = 0

    if title:
        sections.append(
            PaperSection(
                section_id=f"sec-{order}",
                section_name="Title",
                section_order=order,
                canonical_name="title",
                paragraphs=[Paragraph(paragraph_id=f"p-{order}-0", paragraph_order=0, text=title.strip())],
            )
        )
        order += 1

    if abstract:
        sections.append(
            PaperSection(
                section_id=f"sec-{order}",
                section_name="Abstract",
                section_order=order,
                canonical_name="abstract",
                paragraphs=[Paragraph(paragraph_id=f"p-{order}-0", paragraph_order=0, text=abstract.strip())],
            )
        )
        order += 1

    if existing_sections:
        for sec in existing_sections:
            canonical = map_to_canonical_section(sec.section_name) if not sec.canonical_name else sec.canonical_name
            sections.append(
                PaperSection(
                    section_id=sec.section_id or f"sec-{order}",
                    section_name=sec.section_name,
                    section_order=order,
                    canonical_name=canonical,
                    paragraphs=sec.paragraphs,
                )
            )
            order += 1
        return sections

    if not text:
        return sections

    # Parse plain text by double newlines into paragraphs / sections
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

    current_name = "Body"
    current_canonical = "introduction"
    current_paragraphs: list[Paragraph] = []

    for block in blocks:
        lines = block.splitlines()
        first_line = lines[0].strip()

        # Check if first line looks like a heading (short, no ending period, matches canonical pattern)
        is_heading = len(first_line) < 80 and not first_line.endswith(".") and map_to_canonical_section(first_line) != "other"

        if is_heading:
            # Flush previous section if it had paragraphs
            if current_paragraphs:
                sections.append(
                    PaperSection(
                        section_id=f"sec-{order}",
                        section_name=current_name,
                        section_order=order,
                        canonical_name=current_canonical,
                        paragraphs=current_paragraphs,
                    )
                )
                order += 1
                current_paragraphs = []

            current_name = first_line
            current_canonical = map_to_canonical_section(first_line)
            body_lines = "\n".join(lines[1:]).strip()
            if body_lines:
                current_paragraphs.append(
                    Paragraph(
                        paragraph_id=f"p-{order}-{len(current_paragraphs)}",
                        paragraph_order=len(current_paragraphs),
                        text=body_lines,
                    )
                )
        else:
            current_paragraphs.append(
                Paragraph(
                    paragraph_id=f"p-{order}-{len(current_paragraphs)}",
                    paragraph_order=len(current_paragraphs),
                    text=block,
                )
            )

    if current_paragraphs:
        sections.append(
            PaperSection(
                section_id=f"sec-{order}",
                section_name=current_name,
                section_order=order,
                canonical_name=current_canonical,
                paragraphs=current_paragraphs,
            )
        )

    return sections

