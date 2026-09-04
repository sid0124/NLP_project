"""Domain models describing a structured academic paper (master spec §10).

The hierarchy mirrors how a paper is actually organised::

    PaperDocument -> PaperSection -> Paragraph -> (sentences, later) -> tokens

Milestone 1 populates only the metadata layer (title, abstract, topics), so
``sections`` is normally empty. The field exists now, and downstream code reads
text through :meth:`PaperDocument.text_for`, so the PDF parser in a later phase
can fill the hierarchy in without changing any consumer. That hierarchy is also
what the Hierarchical Attention Network consumes in Milestone 3, which is why
the paper is never flattened into a single string at the schema level.

Two distinct models are kept deliberately separate:

* :class:`PaperDocument` — the rich source-of-truth record. Label-free by
  design, because PDF uploads have no ground-truth labels.
* :class:`DatasetRecord` — the flat, ML-facing view that training consumes.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "Author",
    "DatasetRecord",
    "PaperDocument",
    "PaperSection",
    "Paragraph",
    "SplitName",
    "TopicAssignment",
]

_STRICT = ConfigDict(extra="forbid", validate_assignment=True)

SplitName = Literal["train", "val", "test"]

#: Canonical section names for research papers (master spec §7). The section
#: detector maps noisy real-world headings onto these, and the section-attention
#: layer in Milestone 3 learns one importance weight per canonical name.
CANONICAL_SECTIONS: tuple[str, ...] = (
    "title",
    "abstract",
    "introduction",
    "related_work",
    "methodology",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "references",
    "appendix",
    "other",
)


class Paragraph(BaseModel):
    """A single paragraph of body text."""

    model_config = _STRICT

    paragraph_id: str
    paragraph_order: int = Field(ge=0)
    text: str
    #: Page provenance for PDF-extracted text (1-based). Populated by the PDF
    #: parser so the RAG layer and the UI can cite "Section 4.2, Page 6".
    page_number: int | None = None

    @property
    def char_count(self) -> int:
        """Length of the paragraph in characters."""
        return len(self.text)


class PaperSection(BaseModel):
    """A titled section, holding ordered paragraphs."""

    model_config = _STRICT

    section_id: str
    section_name: str
    section_order: int = Field(ge=0)
    paragraphs: list[Paragraph] = Field(default_factory=list)
    #: ``section_name`` mapped onto :data:`CANONICAL_SECTIONS`. Kept alongside
    #: the raw heading so the original text is never lost.
    canonical_name: str | None = None

    @property
    def text(self) -> str:
        """Section body, paragraphs joined in order by blank lines."""
        ordered = sorted(self.paragraphs, key=lambda p: p.paragraph_order)
        return "\n\n".join(p.text for p in ordered if p.text)


class Author(BaseModel):
    """A paper author."""

    model_config = _STRICT

    name: str
    author_id: str | None = None
    affiliations: list[str] = Field(default_factory=list)
    #: Source-reported ordinal role, e.g. ``"first"``/``"middle"``/``"last"``.
    position: str | None = None
    orcid: str | None = None


class TopicAssignment(BaseModel):
    """A scored topic from the source taxonomy.

    Mirrors an OpenAlex topic, which carries the full
    ``topic -> subfield -> field -> domain`` chain. Retaining every level means
    the label space can be re-cut at a different granularity from cached data,
    with no re-fetch.
    """

    model_config = _STRICT

    display_name: str
    score: float = Field(ge=0.0, le=1.0)
    topic_id: str | None = None
    subfield: str | None = None
    field: str | None = None
    domain: str | None = None

    def level(self, taxonomy_level: str) -> str | None:
        """Return this assignment's name at the requested taxonomy level.

        Args:
            taxonomy_level: One of ``"topic"``, ``"subfield"``, ``"field"``,
                or ``"domain"``.

        Raises:
            ValueError: If ``taxonomy_level`` is not a known level.
        """
        match taxonomy_level:
            case "topic":
                return self.display_name
            case "subfield":
                return self.subfield
            case "field":
                return self.field
            case "domain":
                return self.domain
            case _:
                raise ValueError(
                    f"Unknown taxonomy level '{taxonomy_level}'; "
                    "expected one of: topic, subfield, field, domain"
                )


class PaperDocument(BaseModel):
    """A structured academic paper, independent of how it was obtained.

    Produced by every ingestion source (OpenAlex today, a PDF parser later), so
    downstream stages depend on this shape rather than on any source's schema.
    """

    model_config = _STRICT

    # -- identity ----------------------------------------------------------
    paper_id: str
    source: str
    doi: str | None = None

    # -- bibliographic metadata -------------------------------------------
    title: str
    abstract: str | None = None
    authors: list[Author] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    publication_date: date | None = None
    publication_year: int | None = None
    venue: str | None = None
    language: str | None = None
    document_type: str | None = None

    # -- hierarchical body text (populated by the PDF parser in a later phase)
    sections: list[PaperSection] = Field(default_factory=list)

    # -- relationships (feeds the optional citation-graph module) ----------
    references: list[str] = Field(default_factory=list)

    # -- source taxonomy assignments --------------------------------------
    primary_topic: TopicAssignment | None = None
    topics: list[TopicAssignment] = Field(default_factory=list)

    @field_validator("title", mode="before")
    @classmethod
    def _title_not_blank(cls, value: Any) -> Any:
        """Normalise a missing or whitespace-only title to an empty string.

        Validation, not ingestion, is where such records get rejected, so this
        keeps parsing tolerant while leaving the quality gate authoritative.
        """
        if value is None:
            return ""
        return value.strip() if isinstance(value, str) else value

    @property
    def has_sections(self) -> bool:
        """True once hierarchical body text has been parsed."""
        return bool(self.sections)

    @property
    def reference_count(self) -> int:
        """Number of outbound references recorded for this paper."""
        return len(self.references)

    @property
    def full_text(self) -> str:
        """Title, abstract, and every parsed section, concatenated in order."""
        return self.text_for(("title", "abstract", "sections"))

    def text_for(self, fields: tuple[str, ...] | list[str]) -> str:
        """Assemble model input text from the named fields.

        Driven by ``text.fields`` in ``configs/config.yaml``, so the input
        composition is a configuration choice rather than a code change. When
        the PDF parser lands, adding ``"sections"`` to that list is the only
        edit needed for models to start consuming full text.

        Args:
            fields: Field names in the order they should be concatenated.
                Recognised values are ``"title"``, ``"abstract"``,
                ``"keywords"``, and ``"sections"``; unknown names are ignored
                so config can reference fields a given source cannot supply.

        Returns:
            The joined text, with empty parts omitted.
        """
        parts: list[str] = []
        for name in fields:
            match name:
                case "title":
                    if self.title:
                        parts.append(self.title)
                case "abstract":
                    if self.abstract:
                        parts.append(self.abstract)
                case "keywords":
                    if self.keywords:
                        parts.append(" ".join(self.keywords))
                case "sections":
                    ordered = sorted(self.sections, key=lambda s: s.section_order)
                    parts.extend(text for s in ordered if (text := s.text))
                case _:
                    continue
        return "\n\n".join(parts)

    def label_at(self, taxonomy_level: str) -> str | None:
        """Return the single-label target from ``primary_topic``.

        Args:
            taxonomy_level: Level of the taxonomy to read, e.g. ``"subfield"``.

        Returns:
            The class name, or ``None`` when the paper has no primary topic.
        """
        if self.primary_topic is None:
            return None
        return self.primary_topic.level(taxonomy_level)

    def labels_at(
        self,
        taxonomy_level: str,
        *,
        min_score: float = 0.0,
        max_labels: int | None = None,
    ) -> list[str]:
        """Return the multi-label target set, highest-scoring first.

        Deduplicates while preserving order, because several distinct topics
        commonly roll up to the same subfield.

        Args:
            taxonomy_level: Level of the taxonomy to read.
            min_score: Discard topic assignments scoring below this.
            max_labels: Optional cap on the number of labels returned.

        Returns:
            Ordered, deduplicated class names.
        """
        ranked = sorted(self.topics, key=lambda t: t.score, reverse=True)
        seen: dict[str, None] = {}
        for topic in ranked:
            if topic.score < min_score:
                continue
            name = topic.level(taxonomy_level)
            if name:
                seen.setdefault(name, None)
        names = list(seen)
        return names[:max_labels] if max_labels is not None else names


class DatasetRecord(BaseModel):
    """One flat training example: text plus target(s).

    The ML-facing projection of a :class:`PaperDocument`. ``label`` carries the
    multi-class target and ``labels`` the multi-label set; which one is
    authoritative follows ``labels.mode`` in configuration. ``title`` is
    retained purely so error-analysis output is human-readable.

    The richer fields (``abstract``, ``full_text``, ``sections``, ``domain``,
    ``authors``, ``year``, ``source``) are optional and mirror the dataset
    contract in ``dataset_loader.py``: CSV/JSON/JSONL exports carry them, and
    the hierarchical model consumes ``sections`` directly instead of
    re-flattening the document.
    """

    model_config = _STRICT

    paper_id: str
    text: str
    title: str = ""
    label: str | None = None
    labels: list[str] = Field(default_factory=list)
    split: SplitName | None = None
    #: Small, opt-in provenance bag (e.g. publication year) for slicing metrics
    #: during error analysis. Kept narrow on purpose: the spec warns against
    #: storing everything as one opaque JSON blob (§28).
    meta: dict[str, Any] = Field(default_factory=dict)

    # --- Dataset-pipeline fields (spec §3) -----------------------------------
    abstract: str = ""
    full_text: str = ""
    sections: list[PaperSection] = Field(default_factory=list)
    domain: str | None = None
    authors: list[Author] = Field(default_factory=list)
    year: int | None = None
    source: str = ""

    @property
    def char_count(self) -> int:
        """Length of the model input text in characters."""
        return len(self.text)

    @property
    def word_count(self) -> int:
        """Whitespace-delimited token count of the model input text."""
        return len(self.text.split())

    @property
    def section_text(self) -> str:
        """Section bodies joined in reading order ('' when no sections)."""
        ordered = sorted(self.sections, key=lambda s: s.section_order)
        return "\n\n".join(s.text for s in ordered if s.text)
