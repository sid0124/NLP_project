"""Corpus quality assessment (master spec §36).

Validation is a **reporting and filtering gate**, not a parsing step. The
ingestion adapters are deliberately tolerant — they reject only payloads with no
identifier — so that every quality rule lives here, in one place, driven by
``dataset.validation`` in configuration.

The module answers two separate questions and keeps them separate:

* *Is this individual record usable?* — :func:`validate_paper`, which returns the
  reasons it is not.
* *Is the corpus as a whole trustworthy?* — :func:`validate_corpus`, which
  aggregates per-record outcomes into a :class:`DataQualityReport` covering
  missing fields, length distributions, language, and class imbalance.

A high invalid fraction is treated as a hard failure rather than a warning: it
almost always means the query or the taxonomy configuration is wrong, and
training on the remainder would produce a misleading result.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.config.settings import ValidationConfig
from src.schemas.paper import PaperDocument
from src.utils.logging import get_logger

__all__ = [
    "DataQualityReport",
    "PaperIssue",
    "ValidationError",
    "validate_corpus",
    "validate_paper",
]

logger = get_logger(__name__)


class ValidationError(RuntimeError):
    """Raised when a corpus is too degraded to build a dataset from."""


class PaperIssue(BaseModel):
    """One record's validation outcome."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    #: Machine-readable reason codes, e.g. ``["abstract_too_short"]``.
    reasons: list[str] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when no rule was violated."""
        return not self.reasons


class DataQualityReport(BaseModel):
    """Aggregate corpus statistics, persisted alongside every dataset build.

    Written to disk so a reviewer can see what the corpus looked like *before*
    filtering, which is otherwise unrecoverable once records are dropped.
    """

    model_config = ConfigDict(extra="forbid")

    total_records: int
    valid_records: int
    invalid_records: int
    #: Reason code -> number of records exhibiting it. Records may have several.
    reason_counts: dict[str, int] = Field(default_factory=dict)
    #: Label -> count, over valid records only, at the active taxonomy level.
    label_counts: dict[str, int] = Field(default_factory=dict)
    unlabeled_records: int = 0
    language_counts: dict[str, int] = Field(default_factory=dict)
    year_counts: dict[str, int] = Field(default_factory=dict)
    #: Abstract-length summary over valid records, in characters.
    abstract_chars: dict[str, float] = Field(default_factory=dict)
    duplicate_paper_ids: int = 0
    #: Largest class size divided by smallest, over valid labelled records.
    imbalance_ratio: float | None = None
    warnings: list[str] = Field(default_factory=list)
    thresholds: dict[str, Any] = Field(default_factory=dict)

    @property
    def invalid_fraction(self) -> float:
        """Share of records that failed at least one rule."""
        return self.invalid_records / self.total_records if self.total_records else 0.0

    def summary_lines(self) -> list[str]:
        """Render a short human-readable summary for logs and Markdown."""
        lines = [
            f"records            : {self.total_records}",
            f"valid              : {self.valid_records} "
            f"({1 - self.invalid_fraction:.1%} of total)",
            f"invalid            : {self.invalid_records} ({self.invalid_fraction:.1%})",
            f"duplicate ids      : {self.duplicate_paper_ids}",
            f"unlabeled (valid)  : {self.unlabeled_records}",
            f"classes            : {len(self.label_counts)}",
        ]
        if self.imbalance_ratio is not None:
            lines.append(f"imbalance ratio    : {self.imbalance_ratio:.2f}x")
        if self.abstract_chars:
            lines.append(
                "abstract chars     : "
                f"min={self.abstract_chars.get('min', 0):.0f} "
                f"median={self.abstract_chars.get('median', 0):.0f} "
                f"mean={self.abstract_chars.get('mean', 0):.0f} "
                f"max={self.abstract_chars.get('max', 0):.0f}"
            )
        if self.reason_counts:
            top = sorted(self.reason_counts.items(), key=lambda kv: -kv[1])
            lines.append("rejection reasons  : " + ", ".join(f"{k}={v}" for k, v in top))
        return lines


def validate_paper(paper: PaperDocument, config: ValidationConfig) -> PaperIssue:
    """Check one paper against the configured quality rules.

    Every violated rule is reported, not just the first, so the quality report
    can show how failure modes overlap.

    Args:
        paper: The parsed document to check.
        config: Thresholds from ``dataset.validation``.

    Returns:
        The record's issue list; empty when the record is usable.
    """
    reasons: list[str] = []

    title = paper.title.strip()
    if not title:
        reasons.append("title_missing")
    elif len(title) < config.min_title_chars:
        reasons.append("title_too_short")

    abstract = (paper.abstract or "").strip()
    if not abstract:
        reasons.append("abstract_missing")
    else:
        if len(abstract) < config.min_abstract_chars:
            reasons.append("abstract_too_short")
        if len(abstract) > config.max_abstract_chars:
            reasons.append("abstract_too_long")

    # An empty allowed_languages list disables the language rule entirely,
    # which is how a multilingual corpus would be configured.
    if config.allowed_languages:
        language = (paper.language or "").lower()
        if not language:
            reasons.append("language_missing")
        elif language not in {code.lower() for code in config.allowed_languages}:
            reasons.append("language_not_allowed")

    return PaperIssue(paper_id=paper.paper_id, reasons=reasons)


def _length_stats(lengths: list[int]) -> dict[str, float]:
    """Summarise a list of lengths without pulling in numpy."""
    if not lengths:
        return {}
    ordered = sorted(lengths)
    count = len(ordered)
    middle = count // 2
    median = (
        float(ordered[middle])
        if count % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return {
        "min": float(ordered[0]),
        "p05": float(ordered[max(0, int(count * 0.05) - 1)]),
        "median": median,
        "mean": sum(ordered) / count,
        "p95": float(ordered[min(count - 1, int(count * 0.95))]),
        "max": float(ordered[-1]),
    }


def validate_corpus(
    papers: list[PaperDocument],
    config: ValidationConfig,
    *,
    taxonomy_level: str,
    raise_on_threshold: bool = True,
) -> tuple[list[PaperDocument], DataQualityReport]:
    """Validate a corpus, returning the usable records and a quality report.

    Note that the returned records are *not* deduplicated; duplicate ids are
    counted here for reporting only, and removal is
    :mod:`src.data_pipeline.dedup`'s job. Keeping the two stages separate means
    the report describes the corpus as fetched.

    Args:
        papers: Parsed documents to validate.
        config: Thresholds from ``dataset.validation``.
        taxonomy_level: Level at which to tally the label distribution.
        raise_on_threshold: When ``True``, exceeding
            ``max_invalid_fraction`` raises instead of warning.

    Returns:
        The valid records, and the aggregate report.

    Raises:
        ValidationError: If ``papers`` is empty, or the invalid fraction exceeds
            ``max_invalid_fraction`` while ``raise_on_threshold`` is set.
    """
    if not papers:
        raise ValidationError(
            "Cannot validate an empty corpus. Check that the ingestion step "
            "produced records and that the source path is correct."
        )

    valid: list[PaperDocument] = []
    reason_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    year_counts: Counter[str] = Counter()
    abstract_lengths: list[int] = []
    seen_ids: Counter[str] = Counter()
    unlabeled = 0

    for paper in papers:
        seen_ids[paper.paper_id] += 1
        language_counts[(paper.language or "unknown").lower()] += 1

        issue = validate_paper(paper, config)
        if not issue.is_valid:
            reason_counts.update(issue.reasons)
            continue

        valid.append(paper)
        abstract_lengths.append(len((paper.abstract or "").strip()))
        year_counts[str(paper.publication_year) if paper.publication_year else "unknown"] += 1

        label = paper.label_at(taxonomy_level)
        if label:
            label_counts[label] += 1
        else:
            unlabeled += 1

    duplicate_ids = sum(count - 1 for count in seen_ids.values() if count > 1)

    warnings: list[str] = []
    imbalance_ratio: float | None = None
    if label_counts:
        largest, smallest = max(label_counts.values()), min(label_counts.values())
        imbalance_ratio = largest / smallest
        if imbalance_ratio > config.imbalance_warn_ratio:
            warnings.append(
                f"Class imbalance {imbalance_ratio:.1f}x exceeds the configured warning "
                f"ratio of {config.imbalance_warn_ratio:.1f}x "
                f"(largest={largest}, smallest={smallest}). Macro-averaged metrics and "
                "class_weight='balanced' are already in use; treat per-class recall on "
                "the smallest classes with caution."
            )
    if unlabeled:
        warnings.append(
            f"{unlabeled} valid record(s) carry no label at taxonomy level "
            f"'{taxonomy_level}' and will be dropped during labelling."
        )
    if duplicate_ids:
        warnings.append(
            f"{duplicate_ids} duplicate paper id(s) present; deduplication runs next "
            "and removes them before splitting."
        )

    report = DataQualityReport(
        total_records=len(papers),
        valid_records=len(valid),
        invalid_records=len(papers) - len(valid),
        reason_counts=dict(reason_counts),
        label_counts=dict(label_counts.most_common()),
        unlabeled_records=unlabeled,
        language_counts=dict(language_counts.most_common()),
        year_counts=dict(sorted(year_counts.items())),
        abstract_chars=_length_stats(abstract_lengths),
        duplicate_paper_ids=duplicate_ids,
        imbalance_ratio=imbalance_ratio,
        warnings=warnings,
        thresholds=config.model_dump(),
    )

    for line in report.summary_lines():
        logger.info("validation | %s", line)
    for warning in warnings:
        logger.warning("validation | %s", warning)

    if report.invalid_fraction > config.max_invalid_fraction:
        message = (
            f"Invalid record fraction {report.invalid_fraction:.1%} exceeds the configured "
            f"maximum of {config.max_invalid_fraction:.1%} "
            f"({report.invalid_records}/{report.total_records} records rejected). "
            f"Top reasons: {dict(reason_counts.most_common(5))}. "
            "This usually indicates a misconfigured query rather than bad luck."
        )
        if raise_on_threshold:
            raise ValidationError(message)
        logger.warning("validation | %s", message)
        report.warnings.append(message)

    if not valid:
        raise ValidationError(
            "Every record failed validation; there is nothing to build a dataset from. "
            f"Reasons: {dict(reason_counts.most_common())}"
        )

    return valid, report
