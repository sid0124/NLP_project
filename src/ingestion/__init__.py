"""Pluggable academic-metadata sources and cached-payload loaders.

Concrete sources are imported here so that importing :mod:`src.ingestion` is
enough to populate the registry. Without this, ``get_source("openalex")`` would
raise unless the caller happened to import the adapter module first.
"""

from src.ingestion.base import (
    FetchManifest,
    IngestionError,
    IngestionSource,
    available_sources,
    get_source,
    register_source,
)
from src.ingestion.loader import iter_raw_records, load_manifest, load_papers
from src.ingestion.openalex import OpenAlexSource

__all__ = [
    "FetchManifest",
    "IngestionError",
    "IngestionSource",
    "OpenAlexSource",
    "available_sources",
    "get_source",
    "iter_raw_records",
    "load_manifest",
    "load_papers",
    "register_source",
]
