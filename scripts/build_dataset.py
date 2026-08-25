"""Build train/validation/test splits from a cached corpus.

Runs the leakage-safe pipeline in the one order that keeps metrics honest
(master spec §9)::

    load -> validate -> deduplicate -> label -> split -> write

Deduplication runs over the whole corpus *before* splitting, and the split stage
re-audits its own output, so no paper or near-copy of one can straddle train and
test. Nothing here fits a vectorizer or a model: feature extraction happens after
the split, on the training fold only.

Reads cached payloads only — never the network — so a build is reproducible from
committed data.

Examples:
    Build from the committed sample snapshot::

        python scripts/build_dataset.py

    Build a multi-label dataset from a full fetch::

        python scripts/build_dataset.py --source data/raw/openalex --mode multilabel
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401

from src.config.settings import Settings, load_settings
from src.data_pipeline.dedup import deduplicate
from src.data_pipeline.labels import LabelingError, build_records
from src.data_pipeline.split import (
    DATASET_MANIFEST_NAME,
    LABEL_VOCABULARY_NAME,
    SPLIT_NAMES,
    LeakageError,
    split_file_name,
    split_records,
)
from src.data_pipeline.validation import ValidationError, validate_corpus
from src.ingestion.base import IngestionError
from src.ingestion.loader import load_manifest, load_papers
from src.schemas.paper import DatasetRecord
from src.utils.io import (
    PROJECT_ROOT,
    git_commit_sha,
    sha256_text,
    write_json,
    write_jsonl,
    write_text,
)
from src.utils.logging import get_logger, setup_logging
from src.utils.seed import set_seed

logger = get_logger(__name__)

# The two shared names live in src.data_pipeline.split, which the training
# loader also reads them from; only this build report is written nowhere else.
QUALITY_REPORT_NAME = "data_quality_report.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default=None,
        help="JSONL shard or directory of shards. Defaults to paths.sample_dir.",
    )
    parser.add_argument(
        "--config", default="configs", help="Configuration directory (default: configs)."
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for splits and manifest. Defaults to paths.processed_dir.",
    )
    parser.add_argument(
        "--mode",
        choices=("multiclass", "multilabel"),
        default=None,
        help="Override labels.mode without editing configuration.",
    )
    parser.add_argument(
        "--taxonomy-level",
        choices=("field", "subfield"),
        default=None,
        help="Override labels.taxonomy_level.",
    )
    parser.add_argument(
        "--min-class-count",
        type=int,
        default=None,
        help="Override labels.min_class_count. Lower it for small corpora.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override project.seed.")
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override logging.level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def _overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI flags into a nested override mapping for the config loader.

    Overrides are passed to the loader rather than written back to YAML, so a
    one-off run never mutates the committed configuration.
    """
    labels: dict[str, Any] = {}
    if args.mode is not None:
        labels["mode"] = args.mode
    if args.taxonomy_level is not None:
        labels["taxonomy_level"] = args.taxonomy_level
    if args.min_class_count is not None:
        labels["min_class_count"] = args.min_class_count

    overrides: dict[str, Any] = {}
    if labels:
        overrides["labels"] = labels
    if args.seed is not None:
        overrides["project"] = {"seed": args.seed}
    if args.log_level is not None:
        overrides["logging"] = {"level": args.log_level.upper()}
    return overrides


def _relative_to_root(path: Path) -> str:
    """Render a path relative to the project root, for a portable manifest.

    Absolute paths would make a manifest machine-specific, so a build on another
    checkout could not be compared against this one. Paths outside the project
    (an ``--out`` pointing at another drive) are recorded verbatim.
    """
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _write_splits(records: list[DatasetRecord], out_dir: Path) -> dict[str, dict[str, Any]]:
    """Write one JSONL file per split and return per-file provenance.

    Returns:
        Split name -> ``{path, records, sha256}``. The hash covers the file's
        exact bytes, so a later run can prove it consumed the same data.
    """
    written: dict[str, dict[str, Any]] = {}
    for split_name in SPLIT_NAMES:
        members = [r for r in records if r.split == split_name]
        destination = out_dir / split_file_name(split_name)
        count = write_jsonl(destination, (r.model_dump(mode="json") for r in members))
        written[split_name] = {
            "path": _relative_to_root(destination),
            "records": count,
            "sha256": sha256_text(destination.read_text(encoding="utf-8")),
        }
        logger.info("build | wrote %d record(s) to %s", count, destination.name)
    return written


def _write_quality_markdown(path: Path, sections: list[tuple[str, list[str]]]) -> None:
    """Render the stage reports as a single readable Markdown document."""
    lines = [
        "# Dataset Build Report",
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')}.",
        "",
    ]
    for title, body in sections:
        lines += [f"## {title}", "", "```", *body, "```", ""]
    write_text(path, "\n".join(lines))


def build(settings: Settings, source: Path | str, out_dir: Path) -> dict[str, Any]:
    """Run the full build and persist its outputs.

    Args:
        settings: Resolved configuration.
        source: Cached corpus: a ``.jsonl`` shard or a directory of them.
        out_dir: Destination for splits, vocabulary, manifest, and report.

    Returns:
        The manifest that was written.

    Raises:
        ValidationError: If corpus quality is below the configured threshold.
        LabelingError: If no usable labelled records survive.
        LeakageError: If the split audit finds a cross-split collision.
    """
    labels_config = settings.labels
    multilabel = labels_config.is_multilabel

    papers = list(load_papers(source, settings.app.ingestion.source))
    fetch_manifest = load_manifest(source)

    valid, quality = validate_corpus(
        papers,
        settings.dataset.validation,
        taxonomy_level=labels_config.taxonomy_level,
    )
    deduped, dedup_report = deduplicate(valid, settings.dataset.dedup)
    records, label_report = build_records(deduped, labels_config, settings.app.text)
    assigned, split_manifest = split_records(
        records,
        settings.app.split,
        settings.dataset.dedup,
        seed=settings.seed,
        multilabel=multilabel,
    )

    files = _write_splits(assigned, out_dir)
    write_json(
        out_dir / LABEL_VOCABULARY_NAME,
        {
            "mode": labels_config.mode,
            "taxonomy_level": labels_config.taxonomy_level,
            # Index position is the class index used by every model and every
            # saved confusion matrix, so this ordering is part of the contract.
            "classes": label_report.label_vocabulary,
        },
        sort_keys=False,
    )

    manifest: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": str(source),
        "source_fetch_manifest": fetch_manifest.model_dump(mode="json") if fetch_manifest else None,
        "git_commit": git_commit_sha(),
        "seed": settings.seed,
        "config": {
            "labels": labels_config.model_dump(mode="json"),
            "split": settings.app.split.model_dump(mode="json"),
            "text": settings.app.text.model_dump(mode="json"),
            "validation": settings.dataset.validation.model_dump(mode="json"),
            "dedup": settings.dataset.dedup.model_dump(mode="json"),
        },
        "files": files,
        "stages": {
            "validation": quality.model_dump(mode="json"),
            "dedup": dedup_report.model_dump(mode="json"),
            "labels": label_report.model_dump(mode="json"),
            "split": split_manifest.model_dump(mode="json"),
        },
    }
    write_json(out_dir / DATASET_MANIFEST_NAME, manifest)

    _write_quality_markdown(
        out_dir / QUALITY_REPORT_NAME,
        [
            ("Validation", quality.summary_lines()),
            ("Deduplication", dedup_report.summary_lines()),
            ("Labelling", label_report.summary_lines()),
            ("Split", split_manifest.summary_lines()),
        ],
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    settings = load_settings(args.config, overrides=_overrides_from_args(args))
    setup_logging(level=settings.log_level)
    set_seed(settings.seed)

    source = Path(args.source) if args.source else settings.paths.resolved("sample_dir")
    out_dir = Path(args.out) if args.out else settings.paths.resolved("processed_dir")

    logger.info(
        "build | source=%s out=%s mode=%s level=%s seed=%d",
        source,
        out_dir,
        settings.labels.mode,
        settings.labels.taxonomy_level,
        settings.seed,
    )

    try:
        manifest = build(settings, source, out_dir)
    except (FileNotFoundError, IngestionError) as exc:
        logger.error("build | cannot read corpus: %s", exc)
        return 2
    except ValidationError as exc:
        logger.error("build | corpus failed quality gate: %s", exc)
        return 3
    except LabelingError as exc:
        logger.error("build | no usable labels: %s", exc)
        return 4
    except LeakageError as exc:
        # Distinct exit code: this one means a bug, not bad input.
        logger.error("build | LEAKAGE: %s", exc)
        return 5

    totals = {name: info["records"] for name, info in manifest["files"].items()}
    logger.info("build | done: %s -> %s", totals, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
