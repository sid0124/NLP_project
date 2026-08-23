"""Offline loaders for cached ingestion payloads.

Reading cached records is deliberately separate from fetching them. Everything
here is pure filesystem I/O plus the source's static ``parse_record``, so the
dataset build, the tests, and the integration suite all run with no network
access and produce identical results on every invocation.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from src.ingestion.base import FetchManifest, IngestionError, get_source
from src.schemas.paper import PaperDocument
from src.utils.io import read_json, read_jsonl, resolve_path
from src.utils.logging import get_logger

__all__ = ["discover_shards", "iter_raw_records", "load_manifest", "load_papers"]

logger = get_logger(__name__)

MANIFEST_FILENAME = "manifest.json"


def discover_shards(path: Path | str) -> list[Path]:
    """Return the JSONL shards at ``path``, sorted for deterministic ordering.

    Accepts either a single ``.jsonl`` file or a directory of them, so the same
    call site works for a one-file fixture and a multi-class fetch directory.

    Args:
        path: A ``.jsonl`` file or a directory containing them.

    Returns:
        Sorted shard paths.

    Raises:
        FileNotFoundError: If ``path`` does not exist, or holds no ``.jsonl``
            files. Silence here would surface much later as a confusing
            "empty dataset" error.
    """
    resolved = resolve_path(path)
    if resolved.is_file():
        return [resolved]
    if not resolved.is_dir():
        raise FileNotFoundError(f"No such file or directory: {resolved}")

    shards = sorted(resolved.glob("*.jsonl"))
    if not shards:
        raise FileNotFoundError(
            f"No .jsonl shards found in {resolved}. "
            "Run scripts/fetch_data.py first, or point --source at data/sample."
        )
    return shards


def iter_raw_records(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream raw records from every shard at ``path``.

    Yields:
        Decoded record mappings, shard by shard. Non-mapping lines are skipped
        with a warning rather than aborting the load.
    """
    for shard in discover_shards(path):
        count = 0
        for record in read_jsonl(shard, skip_malformed=True):
            if not isinstance(record, dict):
                logger.warning("Skipping non-object record in %s", shard.name)
                continue
            count += 1
            yield record
        logger.debug("Read %d record(s) from %s", count, shard.name)


def load_papers(path: Path | str, source: str) -> Iterator[PaperDocument]:
    """Parse cached payloads at ``path`` into documents using ``source``.

    ``parse_record`` is part of the :class:`~src.ingestion.base.IngestionSource`
    contract as a static method, so no source instance — and therefore no
    network configuration — is needed to read a cached corpus.

    Args:
        path: A ``.jsonl`` shard or a directory of them.
        source: Registered source name, e.g. ``"openalex"``.

    Yields:
        Parsed documents, skipping payloads the source cannot represent.

    Raises:
        KeyError: If ``source`` is not registered.
        FileNotFoundError: If no shards are found.
    """
    source_cls = get_source(source)
    parsed = skipped = 0
    for raw in iter_raw_records(path):
        document = source_cls.parse_record(raw)
        if document is None:
            skipped += 1
            continue
        parsed += 1
        yield document

    logger.info("Parsed %d document(s) from %s (%d unusable payload(s))", parsed, path, skipped)
    if parsed == 0:
        raise IngestionError(
            f"Parsed zero documents from {path}. The cached payloads may be empty "
            f"or in a format '{source}' does not recognise."
        )


def load_manifest(path: Path | str) -> FetchManifest | None:
    """Load the fetch manifest beside a cached corpus.

    Args:
        path: The corpus directory, or the manifest file itself.

    Returns:
        The parsed manifest, or ``None`` when absent or unreadable. A corpus
        without a manifest is still usable — provenance is simply unknown — so
        this degrades to a warning rather than an error.
    """
    resolved = resolve_path(path)
    manifest_path = resolved if resolved.is_file() else resolved / MANIFEST_FILENAME
    if not manifest_path.is_file():
        logger.warning("No fetch manifest at %s; corpus provenance is unknown", manifest_path)
        return None
    try:
        return FetchManifest(**read_json(manifest_path))
    except (ValueError, TypeError) as exc:
        logger.warning("Could not parse fetch manifest %s: %s", manifest_path, exc)
        return None
