"""Target construction: papers to labelled dataset records.

This stage turns source taxonomy assignments into supervised targets and is the
one place that knows about ``labels.mode``. Both modes are built for **every**
record regardless of the active mode, because they are cheap to compute and
having both on disk means switching modes never requires a rebuild:

* **multiclass** — one label per paper, from ``primary_topic`` at the configured
  taxonomy level. Softmax targets.
* **multilabel** — a score-thresholded set of labels, from all topic
  assignments. Independent sigmoid targets, which is what the dashboard's
  "Top Predicted Domains" panel displays (see ``docs/ui-target.md``).

``labels.mode`` decides which field is *authoritative* for training and for the
small-class filter, not which fields exist.

Filtering order matters and is deliberate: classes below ``min_class_count`` are
dropped **before** splitting, because a class with one member cannot be
stratified across three splits, and a class with three members yields a test
fold whose per-class recall is either 0.0 or 1.0 — a number that looks like a
result but carries no information.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.config.settings import LabelsConfig, TextConfig
from src.preprocessing.text import clean_text
from src.schemas.paper import DatasetRecord, PaperDocument
from src.utils.logging import get_logger

__all__ = [
    "LabelReport",
    "LabelingError",
    "build_label_vocabulary",
    "build_records",
    "paper_to_record",
]

logger = get_logger(__name__)


class LabelingError(RuntimeError):
    """Raised when no usable labelled records survive target construction."""


class LabelReport(BaseModel):
    """Outcome of target construction, persisted with the dataset build."""

    model_config = ConfigDict(extra="forbid")

    mode: str
    taxonomy_level: str
    input_records: int
    output_records: int
    #: Class -> record count, over retained records, under the active mode.
    label_counts: dict[str, int] = Field(default_factory=dict)
    #: Ordered class vocabulary; the canonical class index for the whole run.
    label_vocabulary: list[str] = Field(default_factory=list)
    dropped_no_label: int = 0
    dropped_empty_text: int = 0
    dropped_excluded_class: int = 0
    #: Class -> count for classes removed by ``min_class_count``.
    dropped_small_classes: dict[str, int] = Field(default_factory=dict)
    #: Multi-label only: mean number of labels per retained record.
    mean_labels_per_record: float | None = None
    label_cardinality: dict[str, int] = Field(default_factory=dict)
    settings: dict[str, object] = Field(default_factory=dict)

    def summary_lines(self) -> list[str]:
        """Render a short human-readable summary for logs and Markdown."""
        lines = [
            f"mode               : {self.mode} @ {self.taxonomy_level}",
            f"input records      : {self.input_records}",
            f"output records     : {self.output_records}",
            f"classes            : {len(self.label_vocabulary)}",
            f"dropped no label   : {self.dropped_no_label}",
            f"dropped empty text : {self.dropped_empty_text}",
        ]
        if self.dropped_excluded_class:
            lines.append(f"dropped excluded   : {self.dropped_excluded_class}")
        if self.dropped_small_classes:
            lines.append(f"dropped small cls  : {self.dropped_small_classes}")
        if self.mean_labels_per_record is not None:
            lines.append(f"labels per record  : {self.mean_labels_per_record:.2f} mean")
            lines.append(f"label cardinality  : {self.label_cardinality}")
        return lines


def paper_to_record(
    paper: PaperDocument,
    labels_config: LabelsConfig,
    text_config: TextConfig,
) -> DatasetRecord:
    """Project one paper into a flat dataset record.

    Both ``label`` and ``labels`` are populated regardless of the active mode.
    Text is assembled from ``text.fields`` and cleaned with the ``text`` flags,
    so input composition is a configuration choice (master spec §32).

    Args:
        paper: A validated, deduplicated document.
        labels_config: ``labels`` settings, supplying the taxonomy level and the
            multi-label thresholds.
        text_config: ``text`` settings, supplying fields and cleaning flags.

    Returns:
        The dataset record, with ``split`` left unset for the split stage.
    """
    level = labels_config.taxonomy_level
    multilabel = labels_config.multilabel

    raw_text = paper.text_for(text_config.fields)
    text = clean_text(
        raw_text,
        lowercase=text_config.lowercase,
        remove_urls=text_config.strip_urls,
        apply_nfkc=text_config.normalize_unicode,
        squeeze_whitespace=text_config.collapse_whitespace,
    )

    label = paper.label_at(level)
    labels = paper.labels_at(
        level,
        min_score=multilabel.min_topic_score,
        max_labels=multilabel.max_labels_per_paper,
    )
    # The primary topic is authoritative, so it must appear in the label set even
    # if its own score fell below the multi-label threshold. Without this a
    # record can end up with a multi-class label absent from its own label set.
    if label and label not in labels:
        labels = [label, *labels][: multilabel.max_labels_per_paper]

    return DatasetRecord(
        paper_id=paper.paper_id,
        text=text,
        title=paper.title,
        label=label,
        labels=labels,
        meta={
            "year": paper.publication_year,
            "venue": paper.venue,
            "n_references": paper.reference_count,
            "n_authors": len(paper.authors),
            # First author only. Enough for a byline ("Zhang, Y. et al.") without
            # copying the full author list into every record, and the count above
            # already carries "how many others" (master spec §9).
            "first_author": paper.authors[0].name if paper.authors else None,
        },
    )


def build_label_vocabulary(records: Sequence[DatasetRecord], *, multilabel: bool) -> list[str]:
    """Return the sorted class vocabulary implied by ``records``.

    Sorted alphabetically rather than by frequency so that the class index is
    stable across runs and corpora: a frequency ordering would silently permute
    every saved confusion matrix when class counts shift.
    """
    if multilabel:
        return sorted({label for record in records for label in record.labels})
    return sorted({record.label for record in records if record.label})


def _active_labels(record: DatasetRecord, *, multilabel: bool) -> list[str]:
    """Return the labels that count under the active mode."""
    if multilabel:
        return record.labels
    return [record.label] if record.label else []


def build_records(
    papers: Iterable[PaperDocument],
    labels_config: LabelsConfig,
    text_config: TextConfig,
) -> tuple[list[DatasetRecord], LabelReport]:
    """Build labelled dataset records, filtering unusable ones.

    Records are dropped when they have no label at the configured taxonomy
    level, when their model-input text is empty after cleaning, when their class
    is excluded by configuration, or when their class has fewer than
    ``min_class_count`` members.

    Args:
        papers: Validated, deduplicated documents.
        labels_config: ``labels`` settings.
        text_config: ``text`` settings.

    Returns:
        The retained records and a report describing every filter's effect.

    Raises:
        LabelingError: If no records survive, or if fewer than two classes
            remain — a single-class corpus cannot be a classification task.
    """
    multilabel = labels_config.is_multilabel
    excluded = set(labels_config.exclude_classes)

    candidates: list[DatasetRecord] = []
    dropped_no_label = dropped_empty_text = dropped_excluded = 0
    input_count = 0

    for paper in papers:
        input_count += 1
        record = paper_to_record(paper, labels_config, text_config)

        if not record.text:
            dropped_empty_text += 1
            continue

        if excluded:
            # Exclusion applies to both target views so the two stay consistent.
            record.labels = [label for label in record.labels if label not in excluded]
            if record.label in excluded:
                record.label = None

        if not _active_labels(record, multilabel=multilabel):
            if excluded:
                dropped_excluded += 1
            else:
                dropped_no_label += 1
            continue

        candidates.append(record)

    if not candidates:
        raise LabelingError(
            f"No records carry a usable label at taxonomy level "
            f"'{labels_config.taxonomy_level}' in {labels_config.mode} mode "
            f"({dropped_no_label} without labels, {dropped_empty_text} with empty text). "
            "Check labels.taxonomy_level against the ingested topic hierarchy."
        )

    # -- small-class filter, before splitting ------------------------------
    counts: Counter[str] = Counter()
    for record in candidates:
        counts.update(_active_labels(record, multilabel=multilabel))

    minimum = labels_config.min_class_count
    small = {name: count for name, count in counts.items() if count < minimum}
    retained: list[DatasetRecord] = []

    for record in candidates:
        if multilabel:
            # Drop only the rare labels; the record survives if any label remains,
            # since its other labels are still valid supervision.
            record.labels = [label for label in record.labels if label not in small]
            if record.label in small:
                record.label = None
            if record.labels:
                retained.append(record)
        elif record.label not in small:
            retained.append(record)

    if not retained:
        raise LabelingError(
            f"Every class has fewer than labels.min_class_count={minimum} members "
            f"(largest class has {max(counts.values())}). Lower min_class_count, or "
            "fetch more records per class."
        )

    vocabulary = build_label_vocabulary(retained, multilabel=multilabel)
    if len(vocabulary) < 2:
        raise LabelingError(
            f"Only {len(vocabulary)} class(es) remain after filtering ({vocabulary}); "
            "classification requires at least two. Lower labels.min_class_count or "
            "widen the query in configs/dataset.yaml."
        )

    final_counts: Counter[str] = Counter()
    cardinality: Counter[str] = Counter()
    for record in retained:
        active = _active_labels(record, multilabel=multilabel)
        final_counts.update(active)
        cardinality[str(len(active))] += 1

    mean_labels = (
        sum(len(r.labels) for r in retained) / len(retained) if multilabel else None
    )

    report = LabelReport(
        mode=labels_config.mode,
        taxonomy_level=labels_config.taxonomy_level,
        input_records=input_count,
        output_records=len(retained),
        label_counts=dict(final_counts.most_common()),
        label_vocabulary=vocabulary,
        dropped_no_label=dropped_no_label,
        dropped_empty_text=dropped_empty_text,
        dropped_excluded_class=dropped_excluded,
        dropped_small_classes=dict(sorted(small.items())),
        mean_labels_per_record=mean_labels,
        label_cardinality=dict(sorted(cardinality.items())),
        settings=labels_config.model_dump(),
    )

    for line in report.summary_lines():
        logger.info("labels | %s", line)
    if small:
        logger.warning(
            "labels | dropped %d class(es) below min_class_count=%d: %s",
            len(small),
            minimum,
            dict(sorted(small.items())),
        )

    return retained, report
