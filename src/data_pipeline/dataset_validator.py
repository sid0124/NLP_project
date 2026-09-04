"""Dataset validation and preprocessing report generation (spec §3).

Unlike :mod:`src.data_pipeline.validation` — which validates **OpenAlex
corpus** payloads before labelling — this validator operates on the flat
:class:`~src.schemas.paper.DatasetRecord` view produced by
:mod:`src.data_pipeline.dataset_loader` (or any labelled corpus).

Checks (spec §3): missing labels, duplicate paper ids, empty documents,
invalid categories, class imbalance (reporting + warning), extremely short
papers, and malformed/placeholder text.

Nothing is silently discarded: every rejected record is listed with its
reasons, and :func:`write_preprocessing_report` persists a JSON + Markdown
report beside the cleaned dataset so a human can audit the drop.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.paper import DatasetRecord
from src.utils.logging import get_logger

__all__ = [
    "DatasetValidationReport",
    "RejectedRecord",
    "validate_dataset",
    "write_preprocessing_report",
]

logger = get_logger(__name__)

#: Terms that recur in placeholder or "Lorem Ipsum" style filler text.
_PLACEHOLDER_TERMS = frozenset(
    {"lorem ipsum", "todo", "tbd", "placeholder", "filler text", "xxx"}
)
#: Suspicious control characters indicating mojibake or corrupted encoding.
_MOJIBARE = ("\ufffd", "\x00", "\x01")


class RejectedRecord(BaseModel):
    """One dataset record that failed validation, with its reasons."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    title: str = ""
    reasons: list[str] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True when no rule was violated."""
        return not self.reasons


class DatasetValidationReport(BaseModel):
    """Aggregate outcome of validating a dataset before training."""

    model_config = ConfigDict(extra="forbid")

    total_records: int
    valid_records: int
    invalid_records: int
    #: Reason code -> number of records exhibiting it. Records may fail several.
    reason_counts: dict[str, int] = Field(default_factory=dict)
    #: Label -> count over valid records.
    label_counts: dict[str, int] = Field(default_factory=dict)
    #: Labels present in the data but not in the allowed vocabulary.
    unknown_labels: list[str] = Field(default_factory=list)
    #: Largest class size divided by the smallest, over valid labelled records.
    imbalance_ratio: float | None = None
    #: Character-length statistics of model text over valid records.
    text_chars: dict[str, float] = Field(default_factory=dict)
    rejected: list[RejectedRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def invalid_fraction(self) -> float:
        """Share of records that failed at least one check."""
        return self.invalid_records / self.total_records if self.total_records else 0.0

    def summary_lines(self) -> list[str]:
        """Render a short human-readable summary for logs and Markdown."""
        lines = [
            f"total records   : {self.total_records}",
            f"valid records   : {self.valid_records}",
            f"rejected records: {self.invalid_records}",
            f"classes         : {len(self.label_counts)}",
        ]
        if self.reason_counts:
            lines.append(
                f"top reasons     : {dict(Counter(self.reason_counts).most_common(5))}"
            )
        return lines


def _is_placeholder(text: str) -> bool:
    """True when the text looks like boilerplate rather than real content."""
    lowered = text.lower()
    return any(term in lowered for term in _PLACEHOLDER_TERMS)


def _is_mojibake(text: str) -> bool:
    """True when the text carries corruption markers."""
    return any(char in text for char in _MOJIBARE)


def _length_stats(lengths: Sequence[int]) -> dict[str, float]:
    """Compute a compact summary of a list of lengths."""
    if not lengths:
        return {}
    ordered = sorted(lengths)
    return {
        "mean": sum(lengths) / len(lengths),
        "median": ordered[len(ordered) // 2],
        "min": ordered[0],
        "max": ordered[-1],
    }


def validate_dataset(
    records: Sequence[DatasetRecord],
    *,
    label_vocabulary: Sequence[str] | None = None,
    min_text_chars: int = 100,
    min_class_count: int = 1,
    imbalance_warn_ratio: float = 10.0,
) -> tuple[list[DatasetRecord], DatasetValidationReport]:
    """Validate a dataset, returning the valid subset and a report.

    Args:
        records: The dataset to validate.
        label_vocabulary: Allowed labels. When given, any label outside it is
            reported as ``invalid_category`` and the record rejected. When
            ``None``, only missing-label records are rejected.
        min_text_chars: Records whose model text is shorter than this are
            rejected as ``too_short``. Below this, a hierarchical model has
            too few sentences for attention to mean anything.
        min_class_count: Smallest class size used in the imbalance ratio even
            when a class is empty.
        imbalance_warn_ratio: Warn when the largest class is this many times
            the smallest.

    Returns:
        ``(valid_records, report)``.

    Raises:
        ValueError: If ``records`` is empty — there is nothing to validate.
    """
    if not records:
        raise ValueError("validate_dataset called with an empty record set.")

    report = DatasetValidationReport(
        total_records=len(records),
        valid_records=0,
        invalid_records=0,
    )

    allowed = set(label_vocabulary) if label_vocabulary is not None else None
    seen_ids: Counter[str] = Counter()
    valid: list[DatasetRecord] = []
    rejected: list[RejectedRecord] = []
    lengths: list[int] = []

    for record in records:
        reasons: list[str] = []

        if not (record.label or record.labels):
            reasons.append("missing_label")
        elif allowed is not None:
            for label in [record.label, *record.labels]:
                if label and label not in allowed:
                    reasons.append("invalid_category")
                    report.unknown_labels.append(label)
                    break

        if not record.text or not record.text.strip():
            reasons.append("empty_document")
        elif _is_placeholder(record.text):
            reasons.append("malformed_text")
        elif _is_mojibake(record.text):
            reasons.append("malformed_text")
        elif len(record.text.strip()) < min_text_chars:
            reasons.append("too_short")

        seen_ids[record.paper_id] += 1
        if seen_ids[record.paper_id] > 1:
            reasons.append("duplicate_paper_id")

        entry = RejectedRecord(paper_id=record.paper_id, title=record.title, reasons=reasons)
        if entry.is_valid:
            valid.append(record)
            lengths.append(len(record.text.strip()))
            if record.label:
                report.label_counts[record.label] = report.label_counts.get(record.label, 0) + 1
        else:
            rejected.append(entry)
            for reason in reasons:
                report.reason_counts[reason] = report.reason_counts.get(reason, 0) + 1

    report.valid_records = len(valid)
    report.invalid_records = len(rejected)
    report.rejected = rejected
    report.text_chars = _length_stats(lengths)

    if report.label_counts:
        largest = max(report.label_counts.values())
        smallest = min([count for count in report.label_counts.values()] + [min_class_count])
        report.imbalance_ratio = largest / smallest
        if report.imbalance_ratio > imbalance_warn_ratio:
            report.warnings.append(
                f"Class imbalance of {report.imbalance_ratio:.1f}x exceeds "
                f"{imbalance_warn_ratio:.1f}x. Treat per-class metrics on the "
                "smallest classes with caution; macro-F1 and class weights are "
                "already the project defaults."
            )

    if report.unknown_labels:
        report.warnings.append(
            f"Labels outside the vocabulary: {sorted(set(report.unknown_labels))}. "
            "Those records were rejected as invalid_category."
        )

    if not valid:
        logger.warning(
            "dataset_validator | every record was rejected "
            "(%d attempted). See the report for reasons.", report.total_records
        )

    for line in report.summary_lines():
        logger.info("dataset_validator | %s", line)
    for warning in report.warnings:
        logger.warning("dataset_validator | %s", warning)

    return valid, report


def write_preprocessing_report(report: DatasetValidationReport, out_dir: str | Path) -> Path:
    """Persist a preprocessing report (JSON + Markdown) beside the dataset.

    Args:
        report: The report to persist.
        out_dir: Directory to write ``preprocessing_report.json`` and
            ``preprocessing_report.md`` into.

    Returns:
        The Markdown report path.
    """
    from src.utils.io import ensure_dir, write_json, write_text

    directory = ensure_dir(out_dir)
    write_json(
        directory / "preprocessing_report.json",
        report.model_dump(mode="json"),
        sort_keys=False,
    )
    path = directory / "preprocessing_report.md"
    lines = [
        "# Dataset Preprocessing Report",
        "",
        f"- Total records: **{report.total_records}**",
        f"- Valid records: **{report.valid_records}**",
        f"- Rejected records: **{report.invalid_records}** "
        f"({report.invalid_fraction:.1%})",
        f"- Classes (valid): **{len(report.label_counts)}**",
        "",
        "## Rejection reasons",
        "",
    ]
    if report.reason_counts:
        lines += [
            f"- `{reason}`: **{count}**"
            for reason, count in Counter(report.reason_counts).most_common()
        ]
    else:
        lines.append("- None")
    lines += ["", "## Warnings", ""]
    lines += [f"- {warning}" for warning in report.warnings] or ["- None"]
    lines += ["", "## Rejected records", ""]
    if report.rejected:
        lines += [
            f"- `{entry.paper_id}`: {', '.join(entry.reasons) or 'n/a'}"
            for entry in report.rejected[:200]
        ]
        if len(report.rejected) > 200:
            lines.append(f"- ... and {len(report.rejected) - 200} more")
    else:
        lines.append("- None")
    lines.append("")
    write_text(path, "\n".join(lines))
    return path