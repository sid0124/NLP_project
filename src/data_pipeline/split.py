"""Leakage-safe stratified train/validation/test splitting (master spec §9).

Splitting is the last stage before anything is fitted, and the guarantees it
makes are what make every downstream number trustworthy:

1. **Deterministic.** Records are sorted by ``paper_id`` before splitting, so the
   assignment depends only on the record set and the seed — never on the order
   shards happened to be read in.
2. **Stratified.** Class proportions are preserved across splits, so a small
   class is not concentrated in one fold.
3. **Audited for leakage.** After splitting, :func:`audit_leakage` re-checks that
   no ``paper_id``, no identical normalised text, and no *near-duplicate* text
   spans two splits. The near-duplicate check reuses
   :func:`~src.data_pipeline.dedup.near_duplicate_pairs`, so the audit and the
   deduplicator cannot disagree about what a duplicate is.

The audit is deliberately not trusting: deduplication runs earlier in the
pipeline and should have removed everything, so a non-empty audit finding means
a real bug, and the build fails rather than reporting an inflated score.

**Multi-label stratification is approximate.** scikit-learn cannot stratify on
label *sets*, and exact multi-label stratification needs an iterative algorithm
from an extra dependency. Records are therefore stratified on their primary
label, which balances the dominant class well and secondary labels only
approximately. This is recorded in the manifest rather than hidden.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import train_test_split

from src.config.settings import DedupConfig, SplitConfig
from src.data_pipeline.dedup import near_duplicate_pairs
from src.preprocessing.text import normalize_for_matching
from src.schemas.paper import DatasetRecord, SplitName
from src.utils.io import sha256_text
from src.utils.logging import get_logger

__all__ = [
    "DATASET_MANIFEST_NAME",
    "LABEL_VOCABULARY_NAME",
    "SPLIT_NAMES",
    "LeakageError",
    "SplitManifest",
    "SplitStats",
    "audit_leakage",
    "split_file_name",
    "split_records",
]

logger = get_logger(__name__)

SPLIT_NAMES: tuple[SplitName, ...] = ("train", "val", "test")

# --------------------------------------------------------------------------
# On-disk contract for ``data/processed``
# --------------------------------------------------------------------------
# A build writes one ``<split>.jsonl`` per entry in SPLIT_NAMES, plus these two
# sidecar files. The producer (``scripts/build_dataset.py``) and the consumer
# (:mod:`src.training.dataset`) both take the names from here, so a rename
# cannot leave one side reading a file the other stopped writing.
SPLIT_FILE_SUFFIX = ".jsonl"
DATASET_MANIFEST_NAME = "dataset_manifest.json"
LABEL_VOCABULARY_NAME = "label_vocabulary.json"


def split_file_name(split: str) -> str:
    """Return the JSONL file name holding one split's records."""
    return f"{split}{SPLIT_FILE_SUFFIX}"


class LeakageError(RuntimeError):
    """Raised when the same paper, or near-identical text, spans two splits."""


class SplitStats(BaseModel):
    """Per-split composition."""

    model_config = ConfigDict(extra="forbid")

    name: str
    n_records: int
    fraction: float
    label_counts: dict[str, int] = Field(default_factory=dict)
    #: Hash of the sorted paper ids in this split, for cheap build comparison.
    content_hash: str


class SplitManifest(BaseModel):
    """Reproducibility record for one split (master spec §51/§52)."""

    model_config = ConfigDict(extra="forbid")

    created_at: datetime
    seed: int
    mode: str
    stratified: bool
    #: Present when stratification was requested but had to be abandoned.
    stratification_note: str | None = None
    ratios: dict[str, float] = Field(default_factory=dict)
    total_records: int
    splits: list[SplitStats] = Field(default_factory=list)
    label_vocabulary: list[str] = Field(default_factory=list)
    leakage_checks: dict[str, int] = Field(default_factory=dict)

    def summary_lines(self) -> list[str]:
        """Render a short human-readable summary for logs and Markdown."""
        lines = [f"seed               : {self.seed}", f"total records      : {self.total_records}"]
        for split in self.splits:
            lines.append(
                f"{split.name:<19}: {split.n_records} ({split.fraction:.1%}), "
                f"{len(split.label_counts)} classes"
            )
        if self.stratification_note:
            lines.append(f"stratification     : {self.stratification_note}")
        lines.append(f"leakage checks     : {self.leakage_checks}")
        return lines


def _stratify_key(record: DatasetRecord, *, multilabel: bool) -> str:
    """Return the value a record is stratified on.

    In multi-label mode the primary label is used, falling back to the
    lexicographically first label so that a record whose primary topic was
    filtered out is still stratified on something meaningful.
    """
    if record.label:
        return record.label
    if multilabel and record.labels:
        return sorted(record.labels)[0]
    return "<unlabeled>"


def _can_stratify(keys: Sequence[str]) -> tuple[bool, str | None]:
    """Decide whether a three-way stratified split is possible.

    A stratified three-way split needs at least one member of every class in
    every fold, so a class with fewer than three members makes it impossible.
    Falling back with a warning beats raising: the small-class filter in
    :mod:`src.data_pipeline.labels` is the intended place to prevent this, and a
    deliberately tiny test corpus should still split.
    """
    counts = Counter(keys)
    if len(counts) < 2:
        return False, f"only {len(counts)} distinct stratification class(es)"
    too_small = {name: n for name, n in counts.items() if n < len(SPLIT_NAMES)}
    if too_small:
        return False, (
            f"{len(too_small)} class(es) have fewer than {len(SPLIT_NAMES)} members "
            f"({dict(sorted(too_small.items()))}); fell back to a random split"
        )
    return True, None


def _split_indices(
    keys: Sequence[str], config: SplitConfig, seed: int
) -> tuple[list[int], list[int], list[int], bool, str | None]:
    """Compute train/val/test index lists.

    Two successive binary splits: first train against the rest, then that
    remainder into validation and test. The second split's proportion is
    renormalised so the final ratios match the configured ones.
    """
    indices = list(range(len(keys)))
    stratify_ok, note = (True, None)
    if config.stratify:
        stratify_ok, note = _can_stratify(keys)
        if note:
            logger.warning("split | stratification disabled: %s", note)
    else:
        stratify_ok, note = False, "stratification disabled in configuration"

    strata = list(keys) if stratify_ok else None
    train_idx, rest_idx = train_test_split(
        indices,
        train_size=config.train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=strata,
    )

    # val_ratio and test_ratio are fractions of the whole corpus; convert
    # val_ratio into its share of the held-out remainder.
    remainder = config.val_ratio + config.test_ratio
    val_share = config.val_ratio / remainder if remainder else 0.5
    rest_strata = [keys[i] for i in rest_idx] if stratify_ok else None

    if len(rest_idx) < 2:
        # Nothing left to divide; give it all to validation and record why.
        return train_idx, rest_idx, [], stratify_ok, note

    if stratify_ok:
        # The remainder can be too small to stratify even when the whole corpus
        # was not, so re-check rather than letting scikit-learn raise.
        rest_ok, rest_note = _can_stratify(rest_strata or [])
        if not rest_ok:
            rest_strata = None
            note = (note or "") + (
                f" val/test split unstratified: {rest_note}." if rest_note else ""
            )
            note = note.strip() or None

    val_idx, test_idx = train_test_split(
        rest_idx,
        train_size=val_share,
        random_state=seed,
        shuffle=True,
        stratify=rest_strata,
    )
    return train_idx, val_idx, test_idx, stratify_ok, note


def audit_leakage(
    records: Sequence[DatasetRecord],
    dedup_config: DedupConfig,
    *,
    raise_on_finding: bool = True,
) -> dict[str, int]:
    """Verify that no paper, text, or near-duplicate text spans two splits.

    Args:
        records: Records with ``split`` already assigned.
        dedup_config: ``dataset.dedup`` settings, so the near-duplicate notion
            matches the deduplicator's exactly.
        raise_on_finding: When ``True``, any finding raises.

    Returns:
        Counts keyed ``duplicate_ids_across_splits``,
        ``identical_texts_across_splits``, and
        ``near_duplicate_texts_across_splits``.

    Raises:
        LeakageError: If any check finds a cross-split collision and
            ``raise_on_finding`` is set.
    """
    findings: list[str] = []

    # 1. The same paper id in two splits.
    id_to_splits: dict[str, set[str]] = {}
    for record in records:
        id_to_splits.setdefault(record.paper_id, set()).add(record.split or "<unassigned>")
    cross_ids = {pid: splits for pid, splits in id_to_splits.items() if len(splits) > 1}
    if cross_ids:
        findings.append(
            f"{len(cross_ids)} paper id(s) appear in more than one split, e.g. "
            f"{dict(list(cross_ids.items())[:3])}"
        )

    # 2. Byte-identical text in two splits.
    normalized = [normalize_for_matching(record.text) for record in records]
    text_to_splits: dict[str, set[str]] = {}
    for text, record in zip(normalized, records, strict=True):
        if text:
            text_to_splits.setdefault(text, set()).add(record.split or "<unassigned>")
    cross_texts = sum(1 for splits in text_to_splits.values() if len(splits) > 1)
    if cross_texts:
        findings.append(f"{cross_texts} identical text(s) appear in more than one split")

    # 3. Near-identical text in two splits — the case exact keys cannot catch.
    near = dedup_config.near_duplicate
    cross_near = 0
    if near.enabled:
        pairs, _ = near_duplicate_pairs(
            normalized,
            shingle_size=near.shingle_size,
            threshold=near.jaccard_threshold,
            sketch_size=near.sketch_size,
            max_bucket_size=near.max_bucket_size,
        )
        offenders = [
            (records[i].paper_id, records[j].paper_id, round(score, 4))
            for i, j, score in pairs
            if records[i].split != records[j].split
        ]
        cross_near = len(offenders)
        if offenders:
            findings.append(
                f"{cross_near} near-duplicate pair(s) span two splits at Jaccard "
                f">= {near.jaccard_threshold}, e.g. {offenders[:3]}"
            )

    checks = {
        "duplicate_ids_across_splits": len(cross_ids),
        "identical_texts_across_splits": cross_texts,
        "near_duplicate_texts_across_splits": cross_near,
    }

    if findings:
        message = (
            "Train/test leakage detected after splitting:\n  - "
            + "\n  - ".join(findings)
            + "\nDeduplication runs before splitting and should have removed these, "
            "so this indicates a bug rather than a configuration problem. Metrics "
            "from this split would be inflated."
        )
        if raise_on_finding:
            raise LeakageError(message)
        logger.error("split | %s", message)
    else:
        logger.info("split | leakage audit clean: %s", checks)

    return checks


def split_records(
    records: Sequence[DatasetRecord],
    split_config: SplitConfig,
    dedup_config: DedupConfig,
    *,
    seed: int,
    multilabel: bool,
    audit: bool = True,
) -> tuple[list[DatasetRecord], SplitManifest]:
    """Assign every record to a split and return them with a manifest.

    Args:
        records: Labelled records with ``split`` unset.
        split_config: ``split`` settings (ratios and stratification flag).
        dedup_config: ``dataset.dedup`` settings, used by the leakage audit.
        seed: Random seed, recorded in the manifest.
        multilabel: Whether the active mode is multi-label, which selects the
            stratification key and the label tallies.
        audit: Run :func:`audit_leakage` after splitting.

    Returns:
        New records carrying ``split``, sorted by ``paper_id``, and the manifest.

    Raises:
        ValueError: If ``records`` is empty.
        LeakageError: If the audit finds a cross-split collision.
    """
    if not records:
        raise ValueError("Cannot split an empty record set.")

    # Sorting first makes the assignment a function of the record set and the
    # seed alone, independent of the order shards were read in.
    ordered = sorted(records, key=lambda r: r.paper_id)
    keys = [_stratify_key(record, multilabel=multilabel) for record in ordered]

    train_idx, val_idx, test_idx, stratified, note = _split_indices(keys, split_config, seed)

    assigned: list[DatasetRecord] = []
    for split_name, indices in zip(SPLIT_NAMES, (train_idx, val_idx, test_idx), strict=True):
        for index in indices:
            # model_copy rather than mutation: the caller's records stay unsplit,
            # so a build can be re-split under a different seed without surprise.
            assigned.append(ordered[index].model_copy(update={"split": split_name}))
    assigned.sort(key=lambda r: r.paper_id)

    total = len(assigned)
    stats: list[SplitStats] = []
    for split_name in SPLIT_NAMES:
        members = [r for r in assigned if r.split == split_name]
        counts: Counter[str] = Counter()
        for record in members:
            counts.update(record.labels if multilabel else ([record.label] if record.label else []))
        stats.append(
            SplitStats(
                name=split_name,
                n_records=len(members),
                fraction=len(members) / total if total else 0.0,
                label_counts=dict(counts.most_common()),
                content_hash=sha256_text("\n".join(sorted(r.paper_id for r in members))),
            )
        )

    checks = audit_leakage(assigned, dedup_config) if audit else {}

    vocabulary = sorted(
        {label for r in assigned for label in (r.labels if multilabel else [r.label]) if label}
    )
    manifest = SplitManifest(
        created_at=datetime.now(UTC),
        seed=seed,
        mode="multilabel" if multilabel else "multiclass",
        stratified=stratified,
        stratification_note=note,
        ratios={
            "train": split_config.train_ratio,
            "val": split_config.val_ratio,
            "test": split_config.test_ratio,
        },
        total_records=total,
        splits=stats,
        label_vocabulary=vocabulary,
        leakage_checks=checks,
    )

    for line in manifest.summary_lines():
        logger.info("split | %s", line)

    empty = [s.name for s in stats if s.n_records == 0]
    if empty:
        logger.warning(
            "split | %s split(s) are empty; the corpus is too small for the configured ratios",
            ", ".join(empty),
        )

    return assigned, manifest
