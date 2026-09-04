"""Section parsing facade for the hierarchical model (spec §3 / §7/§8).

Re-exports the canonical-mapping machinery from :mod:`src.preprocessing.sections`
and adds :func:`sections_to_sentence_units`, which turns a parsed
``Paper -> Section -> Paragraph`` document into the **sentence units** the
Hierarchical Attention Network consumes:

    section_name, canonical_name, order, sentences

The long-document bounds from ``configs/model.yaml -> longdoc`` are applied here
so downstream encoder code never has to re-implement truncation.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.preprocessing.sections import map_to_canonical_section, parse_text_into_sections
from src.preprocessing.sentences import split_sentences
from src.schemas.paper import PaperSection

__all__ = [
    "SentenceUnit",
    "map_to_canonical_section",
    "parse_text_into_sections",
    "sections_to_sentence_units",
]

_SENTENCE_UNIT = ConfigDict(extra="forbid")


class SentenceUnit(BaseModel):
    """One sentence with its section provenance, for HAN input."""

    model_config = _SENTENCE_UNIT

    section_name: str
    canonical_name: str
    section_order: int = Field(ge=0)
    sentence_index: int = Field(ge=0)
    text: str

    @property
    def key(self) -> str:
        """Stable unique key for embedding-cache lookups."""
        return f"{self.section_order}:{self.sentence_index}"


def sections_to_sentence_units(
    sections: Sequence[PaperSection],
    *,
    max_sentences_per_section: int | None = None,
    include_empty: bool = False,
) -> list[SentenceUnit]:
    """Flatten sections into ordered sentence units.

    Each section contributes at most ``max_sentences_per_section`` sentences
    (the earlier, usually more informative ones). Sections without paragraphs
    are skipped unless ``include_empty`` is set.

    Args:
        sections: Parsed sections, ordered by ``section_order``.
        max_sentences_per_section: Cap per section. ``None`` disables the cap.
        include_empty: Keep units for empty sentences/sections (rarely useful;
            defaults to dropping them).

    Returns:
        Sentence units in document reading order.
    """
    units: list[SentenceUnit] = []
    for section in sorted(sections, key=lambda s: s.section_order):
        ordered = sorted(section.paragraphs, key=lambda p: p.paragraph_order)
        text = "\n".join(p.text for p in ordered if p.text).strip()
        if not text and not include_empty:
            continue

        sentences = split_sentences(text)
        if max_sentences_per_section is not None:
            sentences = sentences[:max_sentences_per_section]

        for sentence_index, sentence in enumerate(sentences):
            if not sentence.strip() and not include_empty:
                continue
            units.append(
                SentenceUnit(
                    section_name=section.section_name,
                    canonical_name=section.canonical_name or "other",
                    section_order=section.section_order,
                    sentence_index=sentence_index,
                    text=sentence.strip(),
                )
            )
    return units


def iter_sentence_units(
    sections: Sequence[PaperSection],
    *,
    max_sentences_per_section: int | None = None,
) -> Iterator[SentenceUnit]:
    """Generator form of :func:`sections_to_sentence_units`.

    Lets streaming callers build embeddings one section at a time without
    materialising the whole document.
    """
    yield from sections_to_sentence_units(
        sections, max_sentences_per_section=max_sentences_per_section
    )