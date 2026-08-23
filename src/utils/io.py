"""Filesystem, serialisation, and provenance helpers.

Every text file in this project is read and written as **explicit UTF-8**.
The default encoding on Windows is cp1252, which cannot represent the
non-ASCII author names, titles, and abstracts that academic metadata is full
of; relying on the platform default silently corrupts data or raises
``UnicodeEncodeError``. JSON is likewise written with ``ensure_ascii=False`` so
that stored text stays human-readable rather than escaped.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import yaml

from src.utils.logging import get_logger

__all__ = [
    "PROJECT_ROOT",
    "atomic_write_text",
    "ensure_dir",
    "git_commit_sha",
    "read_json",
    "read_jsonl",
    "read_yaml",
    "resolve_path",
    "sha256_file",
    "sha256_text",
    "write_json",
    "write_jsonl",
    "write_text",
]

logger = get_logger(__name__)

ENCODING = "utf-8"

#: Repository root, derived from this file's location (src/utils/io.py).
#: Lets configured relative paths resolve identically no matter which
#: directory a script is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: Path | str) -> Path:
    """Resolve ``path`` against the project root unless already absolute."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


def ensure_dir(path: Path | str) -> Path:
    """Create a directory (including parents) if absent and return it."""
    directory = resolve_path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# ---------------------------------------------------------------------------
# YAML / JSON
# ---------------------------------------------------------------------------
def read_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML mapping.

    Args:
        path: File to read, absolute or project-root-relative.

    Returns:
        The parsed mapping; an empty file yields an empty dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the document is not a mapping, or the YAML is invalid.
    """
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"YAML config not found: {resolved}")
    try:
        loaded = yaml.safe_load(resolved.read_text(encoding=ENCODING))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {resolved}: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a YAML mapping at the top level of {resolved}, got {type(loaded).__name__}")
    return loaded


def read_json(path: Path | str) -> Any:
    """Load a JSON document.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the document is not valid JSON.
    """
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"JSON file not found: {resolved}")
    try:
        return json.loads(resolved.read_text(encoding=ENCODING))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {resolved}: {exc}") from exc


def write_json(path: Path | str, obj: Any, *, indent: int = 2, sort_keys: bool = False) -> Path:
    """Serialise ``obj`` to JSON atomically.

    Args:
        path: Destination file; parent directories are created as needed.
        obj: Any JSON-serialisable object.
        indent: Indentation width; ``0`` or ``None`` yields compact output.
        sort_keys: Sort mapping keys, useful for stable content hashes.

    Returns:
        The resolved destination path.
    """
    resolved = resolve_path(path)
    ensure_dir(resolved.parent)
    payload = json.dumps(
        obj,
        indent=indent or None,
        sort_keys=sort_keys,
        ensure_ascii=False,
        default=_json_default,
    )
    return atomic_write_text(resolved, payload + "\n")


def _json_default(value: Any) -> Any:
    """Coerce common non-JSON-native types encountered in run metadata."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set | frozenset):
        return sorted(value)
    if hasattr(value, "item"):  # numpy scalar -> Python scalar
        return value.item()
    if hasattr(value, "isoformat"):  # date / datetime
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


# ---------------------------------------------------------------------------
# JSON Lines — the on-disk format for paper records
# ---------------------------------------------------------------------------
def read_jsonl(path: Path | str, *, skip_malformed: bool = False) -> Iterator[dict[str, Any]]:
    """Stream records from a JSON Lines file.

    Generator-based so that corpora larger than memory can be processed.

    Args:
        path: File to read.
        skip_malformed: Log and skip unparseable lines instead of raising. Use
            when tolerating a partially written file is preferable to failing.

    Yields:
        One decoded record per non-blank line.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: On a malformed line, unless ``skip_malformed`` is set.
    """
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"JSONL file not found: {resolved}")

    with resolved.open("r", encoding=ENCODING) as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                if skip_malformed:
                    logger.warning("Skipping malformed JSONL line %d in %s: %s", line_number, resolved, exc)
                    continue
                raise ValueError(f"Invalid JSON on line {line_number} of {resolved}: {exc}") from exc


def write_jsonl(path: Path | str, records: Iterable[dict[str, Any]]) -> int:
    """Write records as JSON Lines and return the number written.

    Streams rather than buffering the whole corpus, and writes via a temporary
    file so an interrupted run cannot leave a half-written dataset in place.
    """
    resolved = resolve_path(path)
    ensure_dir(resolved.parent)

    count = 0
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(resolved.parent), suffix=".jsonl.tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "w", encoding=ENCODING, newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, default=_json_default))
                handle.write("\n")
                count += 1
        tmp_path.replace(resolved)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return count


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------
def write_text(path: Path | str, text: str) -> Path:
    """Write UTF-8 text, creating parent directories as needed."""
    resolved = resolve_path(path)
    ensure_dir(resolved.parent)
    resolved.write_text(text, encoding=ENCODING, newline="\n")
    return resolved


def atomic_write_text(path: Path | str, text: str) -> Path:
    """Write UTF-8 text via a temporary file, then rename into place.

    Prevents readers from ever observing a partially written manifest or report
    if the process is interrupted mid-write.
    """
    resolved = resolve_path(path)
    ensure_dir(resolved.parent)
    tmp_fd, tmp_name = tempfile.mkstemp(dir=str(resolved.parent), suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "w", encoding=ENCODING, newline="\n") as handle:
            handle.write(text)
        tmp_path.replace(resolved)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return resolved


# ---------------------------------------------------------------------------
# Hashing & provenance — underpins dataset versioning (spec §51/§52)
# ---------------------------------------------------------------------------
def sha256_text(text: str) -> str:
    """Return the hex SHA-256 of ``text`` encoded as UTF-8."""
    return hashlib.sha256(text.encode(ENCODING)).hexdigest()


def sha256_file(path: Path | str, *, chunk_size: int = 1 << 20) -> str:
    """Return the hex SHA-256 of a file, read in chunks.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Cannot hash missing file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_sha(short: bool = True) -> str | None:
    """Return the current git commit SHA, or ``None`` when unavailable.

    Returns ``None`` rather than raising when git is not installed, the project
    is not a repository, or ``HEAD`` is unborn (no commits yet) — all of which
    are legitimate states in which a run should still be allowed to proceed.
    """
    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            args,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            encoding=ENCODING,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
