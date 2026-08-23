"""Exact and near-duplicate removal (master spec §9).

Deduplication runs on the **whole corpus, before splitting**. That ordering is
the point of the module: if the same paper appears twice and the copies land in
different splits, the model is evaluated on text it memorised during training,
and every reported metric is inflated. Removing duplicates after splitting, or
per split, would not fix it.

Two mechanisms, in order of increasing cost:

1. **Exact** — identical source id, identical DOI, or identical hash of
   normalised title plus abstract. Catches the same record fetched twice and the
   preprint/published pairs that share a DOI.
2. **Near** — word-shingle Jaccard similarity above a configured threshold.
   Catches retitled preprints and lightly edited resubmissions, which exact
   keys miss entirely.

Near-duplicate detection is quadratic in the number of documents, which is
intractable at corpus scale (16k documents is 135 million pairs). Candidate
generation therefore uses a **bottom-k sketch**: each document contributes its
*k* smallest shingle hashes to an inverted index, and only documents sharing at
least one such hash are compared exactly. For the high similarity thresholds
this module targets, the probability that a genuine near-duplicate pair shares
no bottom-k hash is negligible, while the candidate set stays near-linear.
Setting ``sketch_size: 0`` disables sketching and compares all pairs, which the
test suite uses to confirm the two paths agree.

Within a duplicate group the **first occurrence wins**. Input order is
deterministic (shards are read in sorted order, records in file order), so
repeated builds drop the same records. For a preprint/published pair this keeps
whichever the source returned first; the difference is immaterial for topic
classification, and the discarded ids are recorded in the report either way.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from src.config.settings import DedupConfig
from src.preprocessing.text import normalize_for_matching
from src.schemas.paper import PaperDocument
from src.utils.logging import get_logger

__all__ = [
    "DedupReport",
    "DuplicatePair",
    "deduplicate",
    "jaccard",
    "near_duplicate_pairs",
    "shingle_hashes",
]

logger = get_logger(__name__)

#: Recognised values for ``dedup.exact.keys``.
EXACT_KEYS = ("source_id", "doi", "title_abstract_hash")


class DuplicatePair(BaseModel):
    """One removal: ``duplicate_id`` was dropped in favour of ``kept_id``."""

    model_config = ConfigDict(extra="forbid")

    kept_id: str
    duplicate_id: str
    #: ``"source_id"``, ``"doi"``, ``"title_abstract_hash"``, or ``"near"``.
    method: str
    #: Populated for near-duplicates only.
    similarity: float | None = None


class DedupReport(BaseModel):
    """What deduplication removed, and how much work it took to find it."""

    model_config = ConfigDict(extra="forbid")

    input_records: int
    output_records: int
    exact_removed: int = 0
    near_removed: int = 0
    removals: list[DuplicatePair] = Field(default_factory=list)
    #: Per-key exact-match removal counts.
    exact_by_key: dict[str, int] = Field(default_factory=dict)
    #: Pairs actually compared during near-duplicate detection. Contrasted with
    #: ``candidate_pairs_exhaustive`` to show what sketching saved.
    candidate_pairs_compared: int = 0
    candidate_pairs_exhaustive: int = 0
    settings: dict[str, object] = Field(default_factory=dict)

    @property
    def total_removed(self) -> int:
        """Records dropped by both mechanisms combined."""
        return self.exact_removed + self.near_removed

    def summary_lines(self) -> list[str]:
        """Render a short human-readable summary for logs and Markdown."""
        lines = [
            f"input records      : {self.input_records}",
            f"exact removed      : {self.exact_removed} {self.exact_by_key or ''}".rstrip(),
            f"near removed       : {self.near_removed}",
            f"output records     : {self.output_records}",
        ]
        if self.candidate_pairs_exhaustive:
            saved = 1 - (self.candidate_pairs_compared / self.candidate_pairs_exhaustive)
            lines.append(
                f"pairs compared     : {self.candidate_pairs_compared} of "
                f"{self.candidate_pairs_exhaustive} exhaustive ({saved:.1%} skipped)"
            )
        return lines


def _stable_hash(value: str) -> int:
    """Hash a string to a 64-bit int, stably across processes.

    ``hash()`` is salted per process for strings, so it cannot be used: the
    bottom-k sketch must select the same shingles on every run for builds to be
    reproducible.
    """
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def shingle_hashes(text: str, size: int) -> frozenset[int]:
    """Hash the word shingles of ``text`` into a set of 64-bit integers.

    Hashes rather than tuples are stored because a corpus-wide shingle index
    over raw strings costs an order of magnitude more memory for no benefit —
    set intersection is all that is ever needed.

    Args:
        text: Text already normalised by
            :func:`~src.preprocessing.text.normalize_for_matching`.
        size: Shingle width in words.

    Returns:
        The shingle hash set. Texts shorter than ``size`` words yield a single
        shingle covering the whole text, so short abstracts still compare
        sensibly rather than producing an empty set that matches nothing.
    """
    tokens = text.split()
    if not tokens:
        return frozenset()
    if len(tokens) <= size:
        return frozenset({_stable_hash(" ".join(tokens))})
    return frozenset(
        _stable_hash(" ".join(tokens[i : i + size])) for i in range(len(tokens) - size + 1)
    )


def jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    """Jaccard similarity of two shingle-hash sets, ``0.0`` when both are empty."""
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    # |A ∪ B| = |A| + |B| - |A ∩ B|, avoiding materialising the union.
    union = len(left) + len(right) - intersection
    return intersection / union if union else 0.0


def _matching_text(paper: PaperDocument) -> str:
    """Build the normalised text used for near-duplicate comparison."""
    return normalize_for_matching(f"{paper.title} {paper.abstract or ''}")


def _exact_keys_for(paper: PaperDocument, keys: Sequence[str]) -> list[tuple[str, str]]:
    """Return the ``(key_name, key_value)`` pairs this paper is indexed under.

    Unrecognised key names are skipped with a warning rather than raising, so a
    configuration typo degrades detection instead of aborting a long build.
    Empty values are skipped: a missing DOI must not match another missing DOI.
    """
    pairs: list[tuple[str, str]] = []
    for key in keys:
        match key:
            case "source_id" | "openalex_id":
                # 'openalex_id' accepted as a source-specific alias.
                value = paper.paper_id.strip()
            case "doi":
                value = (paper.doi or "").strip().casefold()
            case "title_abstract_hash":
                normalized = _matching_text(paper)
                value = (
                    hashlib.sha256(normalized.encode("utf-8")).hexdigest() if normalized else ""
                )
            case _:
                logger.warning(
                    "Ignoring unknown dedup key '%s'; recognised keys are %s", key, EXACT_KEYS
                )
                continue
        if value:
            pairs.append((key, value))
    return pairs


def _remove_exact(
    papers: Sequence[PaperDocument], keys: Sequence[str]
) -> tuple[list[PaperDocument], list[DuplicatePair], dict[str, int]]:
    """Drop records sharing any configured exact key with an earlier record."""
    seen: dict[tuple[str, str], str] = {}
    kept: list[PaperDocument] = []
    removals: list[DuplicatePair] = []
    by_key: defaultdict[str, int] = defaultdict(int)

    for paper in papers:
        key_pairs = _exact_keys_for(paper, keys)
        match: tuple[str, str] | None = next((kp for kp in key_pairs if kp in seen), None)

        if match is not None:
            removals.append(
                DuplicatePair(kept_id=seen[match], duplicate_id=paper.paper_id, method=match[0])
            )
            by_key[match[0]] += 1
            continue

        kept.append(paper)
        for key_pair in key_pairs:
            seen[key_pair] = paper.paper_id

    return kept, removals, dict(by_key)


def _candidate_pairs(
    sketches: Sequence[frozenset[int]], max_bucket_size: int
) -> set[tuple[int, int]]:
    """Generate candidate index pairs from an inverted bottom-k sketch index.

    Buckets larger than ``max_bucket_size`` are skipped: a shingle shared by
    hundreds of documents is boilerplate ("in this paper we propose a novel")
    and contributes a quadratic blow-up of pairs that are almost never
    duplicates. Genuine near-duplicates share many sketch hashes, so they are
    still recovered through their rarer ones.
    """
    index: defaultdict[int, list[int]] = defaultdict(list)
    for position, sketch in enumerate(sketches):
        for value in sketch:
            index[value].append(position)

    pairs: set[tuple[int, int]] = set()
    skipped_buckets = 0
    for members in index.values():
        if len(members) < 2:
            continue
        if len(members) > max_bucket_size:
            skipped_buckets += 1
            continue
        for i, left in enumerate(members):
            for right in members[i + 1 :]:
                pairs.add((left, right))

    if skipped_buckets:
        logger.debug(
            "Skipped %d oversized sketch bucket(s) during candidate generation", skipped_buckets
        )
    return pairs


def near_duplicate_pairs(
    texts: Sequence[str],
    *,
    shingle_size: int,
    threshold: float,
    sketch_size: int,
    max_bucket_size: int,
) -> tuple[list[tuple[int, int, float]], int]:
    """Find every index pair in ``texts`` whose shingle similarity meets ``threshold``.

    Shared by deduplication and by the split stage's leakage audit, so both use
    exactly the same notion of "near-duplicate" — a second, subtly different
    implementation in the audit could certify a leaky split as clean.

    Args:
        texts: Texts already normalised for matching.
        shingle_size: Shingle width in words.
        threshold: Minimum Jaccard similarity to report.
        sketch_size: Bottom-k sketch size for blocking; ``0`` compares all pairs.
        max_bucket_size: Sketch buckets larger than this are skipped.

    Returns:
        ``(pairs, compared)`` where ``pairs`` holds ``(lower_index,
        higher_index, similarity)`` sorted by index, and ``compared`` is how many
        pairs were actually scored.
    """
    shingles = [shingle_hashes(text, shingle_size) for text in texts]

    if sketch_size > 0:
        # Bottom-k: the k smallest hashes form a consistent random sample of the
        # shingle set, so similar documents select overlapping samples.
        sketches = [frozenset(sorted(s)[:sketch_size]) for s in shingles]
        candidates = _candidate_pairs(sketches, max_bucket_size)
    else:
        candidates = {(i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))}

    found: list[tuple[int, int, float]] = []
    for left, right in sorted(candidates):
        score = jaccard(shingles[left], shingles[right])
        if score >= threshold:
            found.append((left, right, score))
    return found, len(candidates)


def _remove_near(
    papers: Sequence[PaperDocument],
    *,
    shingle_size: int,
    threshold: float,
    sketch_size: int,
    max_bucket_size: int,
) -> tuple[list[PaperDocument], list[DuplicatePair], int]:
    """Drop records whose shingle similarity to an earlier record exceeds ``threshold``.

    Returns:
        The retained records, the removals, and the number of pairs compared.
    """
    pairs, compared = near_duplicate_pairs(
        [_matching_text(paper) for paper in papers],
        shingle_size=shingle_size,
        threshold=threshold,
        sketch_size=sketch_size,
        max_bucket_size=max_bucket_size,
    )

    # Pairs arrive sorted with the lower index first, so the earlier record is
    # always the survivor and removals never depend on set iteration order.
    duplicate_of: dict[int, tuple[int, float]] = {}
    for left, right, score in pairs:
        if right in duplicate_of:
            continue  # already removed by an earlier, closer match
        # Attribute the removal to the earliest surviving record in the chain, so
        # a run of three near-identical papers reports two removals against one
        # anchor rather than a chain of pairwise references.
        anchor = left
        while anchor in duplicate_of:
            anchor = duplicate_of[anchor][0]
        if anchor == right:
            continue
        duplicate_of[right] = (anchor, score)

    kept = [paper for index, paper in enumerate(papers) if index not in duplicate_of]
    removals = [
        DuplicatePair(
            kept_id=papers[anchor].paper_id,
            duplicate_id=papers[index].paper_id,
            method="near",
            similarity=round(score, 4),
        )
        for index, (anchor, score) in sorted(duplicate_of.items())
    ]
    return kept, removals, compared


def deduplicate(
    papers: Iterable[PaperDocument], config: DedupConfig
) -> tuple[list[PaperDocument], DedupReport]:
    """Remove exact and near-duplicate records from a corpus.

    Args:
        papers: Validated documents, in a deterministic order.
        config: ``dataset.dedup`` settings.

    Returns:
        The deduplicated records in their original relative order, and a report
        listing every removal.
    """
    ordered = list(papers)
    total = len(ordered)
    removals: list[DuplicatePair] = []
    exact_by_key: dict[str, int] = {}
    exact_removed = near_removed = compared = 0

    if config.exact.enabled:
        ordered, exact_removals, exact_by_key = _remove_exact(ordered, config.exact.keys)
        removals.extend(exact_removals)
        exact_removed = len(exact_removals)

    if config.near_duplicate.enabled and len(ordered) > 1:
        near = config.near_duplicate
        ordered, near_removals, compared = _remove_near(
            ordered,
            shingle_size=near.shingle_size,
            threshold=near.jaccard_threshold,
            sketch_size=near.sketch_size,
            max_bucket_size=near.max_bucket_size,
        )
        removals.extend(near_removals)
        near_removed = len(near_removals)

    remaining = len(ordered)
    report = DedupReport(
        input_records=total,
        output_records=remaining,
        exact_removed=exact_removed,
        near_removed=near_removed,
        removals=removals,
        exact_by_key=exact_by_key,
        candidate_pairs_compared=compared,
        candidate_pairs_exhaustive=(total - exact_removed) * (total - exact_removed - 1) // 2,
        settings=config.model_dump(),
    )

    for line in report.summary_lines():
        logger.info("dedup | %s", line)
    if report.total_removed:
        logger.info(
            "dedup | removed %d duplicate(s) before splitting, preventing train/test leakage",
            report.total_removed,
        )
    return ordered, report
