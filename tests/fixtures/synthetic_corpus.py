"""Seeded synthetic corpus for split and training mechanics.

Two fixtures serve two different purposes, and mixing them would weaken both:

* ``openalex_sample.jsonl`` (built by ``generate_sample.py``) is a small,
  hand-designed corpus of **messy payloads** — it exercises parsing, validation,
  and deduplication edge cases and is deliberately unsuitable for training.
* This module generates a **clean, bulk, separable** corpus for the mechanics
  that need volume: stratified splitting, vectorizer fitting, and end-to-end
  training. It is generated rather than committed because the tests care about
  its *properties* (enough records per class, linearly separable, no duplicates),
  not about specific wording.

Every draw comes from a :class:`random.Random` seeded per call, so two
invocations with the same seed produce byte-identical output and a failing test
can be reproduced exactly.

Separability is engineered, not hoped for: each class owns a private marker
vocabulary that appears in no other class, over a shared filler vocabulary that
appears in all of them. A linear model on TF-IDF features should reach near-1.0
macro F1, which means a test asserting "training works" fails on a real bug
rather than on an unlearnable task.

The written dataset goes through the **real** :func:`build_records` and
:func:`split_records`, leakage audit included, so a fixture that would violate
the pipeline's own guarantees cannot be produced.
"""

from __future__ import annotations

import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import LabelsConfig, Settings, SplitConfig  # noqa: E402
from src.data_pipeline.labels import build_records  # noqa: E402
from src.data_pipeline.split import (  # noqa: E402
    DATASET_MANIFEST_NAME,
    LABEL_VOCABULARY_NAME,
    SPLIT_NAMES,
    split_file_name,
    split_records,
)
from src.schemas.paper import PaperDocument, TopicAssignment  # noqa: E402
from src.utils.io import git_commit_sha, sha256_text, write_json, write_jsonl  # noqa: E402

__all__ = [
    "CLASS_MARKERS",
    "FILLER",
    "SYNTHETIC_SEED",
    "build_synthetic_papers",
    "write_synthetic_dataset",
]

#: Default seed. Distinct from the project seed so a fixture change cannot be
#: mistaken for a training-seed effect.
SYNTHETIC_SEED = 20260824

#: Real subfield names from ``configs/dataset.yaml``, so a class name in a test
#: failure is recognisable and the label space matches production shape.
#: Each class's markers are unique to it — that is what makes the task learnable.
CLASS_MARKERS: dict[str, tuple[str, ...]] = {
    "Artificial Intelligence": (
        "reinforcement", "policy", "reward", "agent", "bandit", "planner",
    ),
    "Computer Vision and Pattern Recognition": (
        "segmentation", "occlusion", "pixel", "convolutional", "viewpoint", "keypoint",
    ),
    "Computer Networks and Communications": (
        "throughput", "router", "congestion", "latency", "packet", "topology",
    ),
    "Software": (
        "refactoring", "compiler", "regression", "codebase", "linter", "assertion",
    ),
}

#: Vocabulary shared by every class, so classes are not separable on length or
#: on trivially disjoint token sets alone.
FILLER: tuple[str, ...] = (
    "we", "propose", "a", "novel", "approach", "that", "improves", "over",
    "prior", "work", "on", "several", "standard", "benchmarks", "our",
    "experiments", "show", "consistent", "gains", "and", "we", "release",
    "code", "for", "reproducibility", "the", "method", "is", "simple", "to",
    "implement", "requires", "no", "additional", "supervision",
)


def _abstract(rng: random.Random, markers: tuple[str, ...]) -> str:
    """Compose one abstract from filler text sprinkled with class markers."""
    words = [rng.choice(FILLER) for _ in range(rng.randint(45, 70))]
    # Enough marker occurrences to clear a min_df of 2 across the class, spread
    # through the text rather than clustered at one end.
    for _ in range(rng.randint(6, 10)):
        words.insert(rng.randrange(len(words) + 1), rng.choice(markers))
    return " ".join(words) + "."


def build_synthetic_papers(
    *,
    per_class: int = 20,
    classes: dict[str, tuple[str, ...]] | None = None,
    seed: int = SYNTHETIC_SEED,
) -> list[PaperDocument]:
    """Generate a separable, duplicate-free corpus of papers.

    Args:
        per_class: Records to generate per class. The default clears a
            three-way stratified split comfortably.
        classes: Class name to marker vocabulary. Defaults to
            :data:`CLASS_MARKERS`.
        seed: Seed for the local generator.

    Returns:
        Papers in deterministic order, each with a ``primary_topic`` at the
        subfield level and one secondary topic, so both label modes have
        something to read.
    """
    rng = random.Random(seed)
    vocabularies = classes or CLASS_MARKERS
    names = list(vocabularies)

    papers: list[PaperDocument] = []
    for class_index, (class_name, markers) in enumerate(vocabularies.items()):
        for record_index in range(per_class):
            # A unique marker per paper guarantees no two abstracts collide, so
            # the deduplicator never removes a record the test is counting on.
            unique = f"synthid{class_index:02d}{record_index:03d}"
            title = f"{markers[record_index % len(markers)].capitalize()} study {unique}"
            secondary = names[(class_index + 1) % len(names)]
            papers.append(
                PaperDocument(
                    paper_id=f"S{class_index:02d}{record_index:03d}",
                    source="synthetic",
                    title=title,
                    abstract=f"{unique} {_abstract(rng, markers)}",
                    publication_year=2020 + (record_index % 5),
                    language="en",
                    document_type="article",
                    primary_topic=TopicAssignment(
                        display_name=f"{class_name} topic",
                        score=round(rng.uniform(0.70, 0.99), 4),
                        subfield=class_name,
                        field="Computer Science",
                        domain="Physical Sciences",
                    ),
                    topics=[
                        TopicAssignment(
                            display_name=f"{class_name} topic",
                            score=round(rng.uniform(0.70, 0.99), 4),
                            subfield=class_name,
                            field="Computer Science",
                            domain="Physical Sciences",
                        ),
                        TopicAssignment(
                            display_name=f"{secondary} topic",
                            score=round(rng.uniform(0.31, 0.55), 4),
                            subfield=secondary,
                            field="Computer Science",
                            domain="Physical Sciences",
                        ),
                    ],
                )
            )
    return papers


def write_synthetic_dataset(
    out_dir: Path,
    settings: Settings,
    *,
    per_class: int = 20,
    seed: int = SYNTHETIC_SEED,
    min_class_count: int = 5,
) -> dict[str, Any]:
    """Build and write a processed dataset in the same layout as a real build.

    Runs the production labelling and splitting stages — leakage audit included —
    then writes the split files, the label vocabulary, and a manifest carrying the
    same keys ``scripts/build_dataset.py`` writes. Training therefore cannot tell
    this dataset from a real one, which is what makes an end-to-end test
    meaningful.

    Args:
        out_dir: Destination directory.
        settings: Project settings, supplying split ratios, text options, and the
            dedup configuration used by the leakage audit.
        per_class: Records per class before splitting.
        seed: Seed for corpus generation and for splitting.
        min_class_count: Override for ``labels.min_class_count``. The production
            default (40) would discard a fixture this small, so it is relaxed
            here rather than in the committed configuration.

    Returns:
        The manifest that was written.
    """
    labels_config: LabelsConfig = settings.labels.model_copy(
        update={"min_class_count": min_class_count}
    )
    # Ratios large enough that every class lands in every split at this size.
    split_config: SplitConfig = settings.app.split

    papers = build_synthetic_papers(per_class=per_class, seed=seed)
    records, label_report = build_records(papers, labels_config, settings.app.text)
    assigned, split_manifest = split_records(
        records,
        split_config,
        settings.dataset.dedup,
        seed=seed,
        multilabel=labels_config.is_multilabel,
    )

    files: dict[str, dict[str, Any]] = {}
    for split_name in SPLIT_NAMES:
        members = [record for record in assigned if record.split == split_name]
        destination = out_dir / split_file_name(split_name)
        count = write_jsonl(destination, (r.model_dump(mode="json") for r in members))
        files[split_name] = {
            "path": destination.name,
            "records": count,
            # Hashed exactly as the real build hashes, so the loader's integrity
            # check is exercised for real rather than bypassed.
            "sha256": sha256_text(destination.read_text(encoding="utf-8")),
        }

    write_json(
        out_dir / LABEL_VOCABULARY_NAME,
        {
            "mode": labels_config.mode,
            "taxonomy_level": labels_config.taxonomy_level,
            "classes": label_report.label_vocabulary,
        },
    )

    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "tests/fixtures/synthetic_corpus.py",
        "source_fetch_manifest": None,
        "git_commit": git_commit_sha(),
        "seed": seed,
        "config": {
            "labels": labels_config.model_dump(mode="json"),
            "split": split_config.model_dump(mode="json"),
            "text": settings.app.text.model_dump(mode="json"),
            "validation": settings.dataset.validation.model_dump(mode="json"),
            "dedup": settings.dataset.dedup.model_dump(mode="json"),
        },
        "files": files,
        "stages": {
            "labels": label_report.model_dump(mode="json"),
            "split": split_manifest.model_dump(mode="json"),
        },
        "synthetic": True,
    }
    write_json(out_dir / DATASET_MANIFEST_NAME, manifest)
    return manifest


def main() -> int:
    """Write a synthetic dataset to a directory given on the command line."""
    from src.config.settings import load_settings

    default_target = PROJECT_ROOT / "data" / "interim" / "synthetic"
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else default_target
    manifest = write_synthetic_dataset(target, load_settings())
    counts = {name: info["records"] for name, info in manifest["files"].items()}
    print(f"Wrote synthetic dataset to {target}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
