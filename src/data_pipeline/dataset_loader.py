"""Universal dataset loading: CSV / JSON / JSONL.

Loads corpora into the project's :class:`src.schemas.paper.DatasetRecord`
contract, supporting the fields:

    paper_id, title, abstract, full_text, sections, label, domain,
    authors, year, source

This complements the OpenAlex ingestion pipeline (``src.ingestion``): that
pipeline fetches and caches a labelled corpus; :func:`load_dataset` exists so a
researcher can hand-curate a corpus in a spreadsheet or JSON export and feed it
straight into the same validation-split-train pipeline.

The loader is deliberately permissive — records missing optional fields load
fine. *Integrity checks* (missing labels, empty documents, duplicates) are the
job of :mod:`src.data_pipeline.dataset_validator`, which reports rather than
silently discarding.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Literal

from src.schemas.paper import Author, DatasetRecord
from src.utils.io import read_jsonl
from src.utils.logging import get_logger

__all__ = [
    "DatasetLoadError",
    "load_csv",
    "load_dataset",
    "load_json",
    "load_jsonl",
]

logger = get_logger(__name__)

#: Supported file formats, dispatched by loader.
Format = Literal["csv", "json", "jsonl"]

#: Column aliases for the label so spreadsheets can use either legal name.
_LABEL_ALIASES = ("label", "domain")
#: Column aliases for the unique identifier.
_ID_ALIASES = ("paper_id", "id", "paperid")


class DatasetLoadError(ValueError):
    """Raised when a dataset file is unreadable, unsupported, or empty."""


def _read_table(path: Path) -> list[list[str]]:
    """Read a CSV file into rows, handling the BOM and CRLF robustly."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise DatasetLoadError(f"CSV file is empty: {path}")
    return rows


def _rows_to_records(rows: list[list[str]], source_name: str) -> list[dict[str, Any]]:
    """Convert CSV rows (header + body) into a list of field dicts."""
    header = [cell.strip().lower() for cell in rows[0]]
    records: list[dict[str, Any]] = []
    for row in rows[1:]:
        if len(row) < len(header):
            # Pad short rows so a misaligned CSV never silently shifts columns.
            row = list(row) + [""] * (len(header) - len(row))
        records.append(dict(zip(header, row, strict=False)))
    if not records:
        raise DatasetLoadError(f"CSV file has a header but no data rows: {source_name}")
    return records


def _resolve_label(row: dict[str, Any]) -> str | None:
    """Return the label/domain column value, preferring the first present."""
    for key in _LABEL_ALIASES:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _resolve_paper_id(row: dict[str, Any], index: int) -> str:
    """Return a stable paper id from the row's id columns or a fallback."""
    for key in _ID_ALIASES:
        value = row.get(key)
        if value is not None:
            cleaned = str(value).strip()
            if cleaned and cleaned.lower() not in ("nan", "none", "null"):
                return cleaned
    return f"paper_{index:06d}"


def _text_field(row: dict[str, Any], *keys: str) -> str:
    """Return the first non-empty string among ``keys`` in ``row``."""
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _compose_model_text(row: dict[str, Any]) -> str:
    """Build the model-input text for a raw record.

    Preference order keeps the richest source first: full text when present,
    otherwise abstract with title, otherwise whatever text the corpus has.
    """
    full_text = _text_field(row, "full_text", "body", "text")
    abstract = _text_field(row, "abstract", "summary")
    title = _text_field(row, "title")
    if full_text:
        return full_text
    parts = [part for part in (title, abstract) if part]
    return "\n\n".join(parts)


def _parse_authors(value: Any) -> list[Author]:
    """Parse an authors column of any supported shape."""
    if isinstance(value, list):
        names = [
            item.get("name") if isinstance(item, dict) else str(item) for item in value
        ]
    elif isinstance(value, str):
        names = [
            name.strip()
            for name in value.replace(";", ",").split(",")
            if name.strip()
        ]
    else:
        return []
    return [Author(name=name) for name in names if name]


def _record_from_row(row: dict[str, Any], source: str, index: int) -> DatasetRecord:
    """Project one raw record dict onto a :class:`DatasetRecord`."""
    title = _text_field(row, "title")
    abstract = _text_field(row, "abstract", "summary")
    full_text = _text_field(row, "full_text", "body", "text")
    model_text = _compose_model_text(row)

    year_raw = row.get("year") or row.get("publication_year")
    year: int | None = None
    if year_raw is not None:
        try:
            year = int(str(year_raw).strip())
        except (TypeError, ValueError):
            year = None

    # The unified label column wins; otherwise fall back to the domain column.
    label = _resolve_label(row)

    return DatasetRecord(
        paper_id=_resolve_paper_id(row, index),
        title=title,
        abstract=abstract,
        full_text=full_text,
        text=model_text,
        label=label,
        meta={"source_row_index": index},
        authors=_parse_authors(row.get("authors", row.get("author", ""))),
        year=year,
        source=source,
    )


def _load_dicts(
    path: Path, fmt: Format, *, source: str | None
) -> list[dict[str, Any]]:
    """Load a file into a list of record dicts, dispatching on ``fmt``."""
    if fmt == "csv":
        return _rows_to_records(_read_table(path), source_name=source or str(path))
    if fmt == "json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            # Accept {records: [...]} and bare mappings alike.
            records = payload.get("records") or payload.get("papers")
            if isinstance(records, list):
                return [r for r in records if isinstance(r, dict)]
            return [payload]
        if isinstance(payload, list):
            return [r for r in payload if isinstance(r, dict)]
        raise DatasetLoadError(f"JSON dataset must be a list or a mapping: {source}")
    if fmt == "jsonl":
        return list(read_jsonl(path))
    raise DatasetLoadError(f"Unsupported dataset format: {fmt!r}")


def load_dataset(
    path: str | Path,
    *,
    fmt: Format | None = None,
) -> list[DatasetRecord]:
    """Load a CSV, JSON, or JSONL dataset into :class:`DatasetRecord` objects.

    Args:
        path: Dataset file. The format is inferred from the file suffix
            unless ``fmt`` overrides it.
        fmt: Explicit format among ``csv``, ``json``, ``jsonl``. When ``None``
            the suffix decides; unknown suffixes raise :class:`DatasetLoadError`.

    Returns:
        The loaded records. Optional fields may be absent; validation of what a
        model can train on happens downstream in
        :mod:`src.data_pipeline.dataset_validator`.

    Raises:
        DatasetLoadError: If the file is unreadable, unsupported, or empty.
        FileNotFoundError: If ``path`` does not exist.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Dataset file not found: {resolved}")

    inferred = (fmt or resolved.suffix.lstrip(".").lower())
    if inferred not in {"csv", "json", "jsonl"}:
        raise DatasetLoadError(
            f"Cannot infer dataset format from '{resolved.suffix}'. "
            "Pass fmt='csv' | 'json' | 'jsonl' explicitly."
        )

    rows = _load_dicts(resolved, inferred, source=str(resolved))
    records = [
        _record_from_row(row, source=str(resolved), index=i)
        for i, row in enumerate(rows)
    ]
    if not records:
        raise DatasetLoadError(f"Dataset contains no usable records: {resolved}")
    logger.info("dataset_loader | loaded %d records from %s", len(records), resolved)
    return records


def load_csv(path: str | Path) -> list[DatasetRecord]:
    """Convenience for :func:`load_dataset` with an explicit CSV format."""
    return load_dataset(path, fmt="csv")


def load_json(path: str | Path) -> list[DatasetRecord]:
    """Convenience for :func:`load_dataset` with an explicit JSON format."""
    return load_dataset(path, fmt="json")


def load_jsonl(path: str | Path) -> list[DatasetRecord]:
    """Convenience for :func:`load_dataset` with an explicit JSONL format."""
    return load_dataset(path, fmt="jsonl")