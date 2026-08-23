"""Ingestion source interface and registry.

Every academic-metadata source implements :class:`IngestionSource`, so the rest
of the pipeline depends on this contract rather than on any provider's schema.
Adding a source (Semantic Scholar, arXiv, a PDF parser) means writing an adapter
and registering it — no edits to the dataset build, training, or evaluation
code (master spec §11).

The interface splits fetching from parsing on purpose:

* :meth:`IngestionSource.fetch` performs network I/O and caches raw payloads.
* :meth:`IngestionSource.parse_record` is a pure function over one cached
  payload, so the entire pipeline can be exercised offline and deterministically
  from committed fixtures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from src.schemas.paper import PaperDocument

__all__ = [
    "ClassFetchStat",
    "FetchManifest",
    "IngestionError",
    "IngestionSource",
    "available_sources",
    "get_source",
    "register_source",
]


class IngestionError(RuntimeError):
    """Raised when a source cannot fetch or cache records.

    Parse failures on individual records do *not* raise: a single malformed
    payload should be counted and skipped, not abort a corpus-wide fetch.
    """


class ClassFetchStat(BaseModel):
    """Per-class outcome of a fetch, recorded for auditability."""

    model_config = ConfigDict(extra="forbid")

    class_name: str
    target_id: str
    requested: int
    retrieved: int
    #: Total matching records the source reported, before the per-class cap.
    available: int | None = None
    file: str
    sha256: str


class FetchManifest(BaseModel):
    """Provenance record written beside every cached fetch.

    Live API results drift over time, so reproducibility comes from the cached
    payloads plus this manifest — the exact query, counts, and content hashes
    that produced a given corpus (master spec §51/§52).
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    run_id: str
    fetched_at: datetime
    taxonomy_level: str
    query: dict[str, Any]
    total_records: int
    classes: list[ClassFetchStat] = Field(default_factory=list)
    #: Subset of resolved configuration that shaped this fetch.
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class IngestionSource(ABC):
    """Abstract source of academic paper metadata."""

    #: Registry key, matching ``ingestion.source`` in ``configs/config.yaml``.
    name: ClassVar[str] = "base"

    @abstractmethod
    def fetch(self, destination: Path, *, run_id: str) -> FetchManifest:
        """Retrieve raw records and cache them to disk.

        Args:
            destination: Directory to write cached payloads and the manifest to.
            run_id: Identifier correlating this fetch with pipeline logs.

        Returns:
            The manifest describing what was fetched.

        Raises:
            IngestionError: If the source is unreachable or returns no records.
        """

    @staticmethod
    @abstractmethod
    def parse_record(raw: dict[str, Any]) -> PaperDocument | None:
        """Convert one raw payload into a :class:`PaperDocument`.

        Pure and offline: given the same payload it must always return the same
        document, so tests and dataset rebuilds are deterministic.

        Args:
            raw: A single record as cached by :meth:`fetch`.

        Returns:
            The parsed document, or ``None`` if the payload is too incomplete to
            represent a paper at all (for example, a missing identifier).
        """

    def parse_records(self, raws: Iterator[dict[str, Any]]) -> Iterator[PaperDocument]:
        """Parse a stream of payloads, skipping unusable ones.

        Yields:
            Successfully parsed documents.
        """
        for raw in raws:
            document = self.parse_record(raw)
            if document is not None:
                yield document


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, type[IngestionSource]] = {}


def register_source(cls: type[IngestionSource]) -> type[IngestionSource]:
    """Register an ingestion source under its ``name``.

    Intended as a class decorator.

    Raises:
        ValueError: If ``name`` is unset or already registered by another class.
    """
    key = cls.name
    if not key or key == "base":
        raise ValueError(f"{cls.__name__} must define a unique 'name' class attribute")
    existing = _REGISTRY.get(key)
    if existing is not None and existing is not cls:
        raise ValueError(f"Ingestion source '{key}' is already registered to {existing.__name__}")
    _REGISTRY[key] = cls
    return cls


def get_source(name: str) -> type[IngestionSource]:
    """Look up a registered ingestion source class.

    Raises:
        KeyError: If no source is registered under ``name``.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown ingestion source '{name}'. Registered sources: {sorted(_REGISTRY)}"
        ) from None


def available_sources() -> list[str]:
    """Return the sorted names of all registered sources."""
    return sorted(_REGISTRY)
