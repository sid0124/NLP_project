"""Structured, run-scoped logging.

Two concerns are handled here that the ``configs/logging.yaml`` dictConfig
cannot express on its own:

1. **UTF-8 streams.** Windows consoles default to cp1252, while academic
   metadata routinely contains non-ASCII author names and titles. Logging a
   single such record raises ``UnicodeEncodeError`` and aborts the pipeline, so
   ``stdout``/``stderr`` are reconfigured before any handler is attached.
2. **Run correlation.** Every record carries a ``run_id`` so one pipeline run
   can be traced end to end across ingestion, dataset build, training, and
   evaluation (master spec §41).
"""

from __future__ import annotations

import logging
import logging.config
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "RunContextFilter",
    "force_utf8_streams",
    "get_logger",
    "get_run_id",
    "new_run_id",
    "set_run_id",
    "setup_logging",
]

# ContextVar rather than a module-level global: safe under threads and asyncio,
# which the background workers in a later milestone will rely on.
_RUN_ID: ContextVar[str] = ContextVar("run_id", default="-")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_LOGGING_CONFIG = _PROJECT_ROOT / "configs" / "logging.yaml"

# Fallback used when configs/logging.yaml is absent or malformed, so that a
# logging misconfiguration degrades to plain output instead of killing the run.
_FALLBACK_FORMAT = "%(asctime)s | %(levelname)-7s | run=%(run_id)s | %(name)-28s | %(message)s"


class RunContextFilter(logging.Filter):
    """Attach the active run identifier to every log record.

    Referenced by ``configs/logging.yaml`` via its ``()`` factory key. Always
    returns ``True``: this filter enriches records, it never drops them.
    """

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        """Set ``record.run_id``, then admit the record."""
        if not hasattr(record, "run_id"):
            record.run_id = _RUN_ID.get()
        return True


def new_run_id(prefix: str = "") -> str:
    """Build a sortable, unique run identifier.

    Format is ``YYYYmmddTHHMMSSZ-<6 hex chars>``, optionally prefixed. The
    timestamp leads so that run directories sort chronologically on disk; the
    random suffix keeps runs started in the same second distinct.

    Args:
        prefix: Optional label prepended to the identifier, e.g. a model name.

    Returns:
        The generated run identifier.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:6]
    return f"{prefix}-{stamp}-{suffix}" if prefix else f"{stamp}-{suffix}"


def set_run_id(run_id: str) -> None:
    """Make ``run_id`` the identifier stamped onto subsequent log records."""
    _RUN_ID.set(run_id)


def get_run_id() -> str:
    """Return the active run identifier, or ``"-"`` if none has been set."""
    return _RUN_ID.get()


def force_utf8_streams() -> None:
    """Reconfigure stdout/stderr to UTF-8, replacing unencodable characters.

    Public because it is needed *before* and *independently of* logging setup:
    a script that merely ``print``s a non-ASCII author name on Windows raises
    ``UnicodeEncodeError`` even when no logger is involved. Entry points call
    this first; :func:`setup_logging` also calls it.

    Idempotent and defensive: streams may be redirected or already closed (as
    under some test runners and CI capture layers), so failures are ignored
    rather than allowed to break logging setup.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - stream already closed
            pass


def _apply_fallback_config(level: str) -> None:
    """Install a minimal stdout logging configuration."""
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_FALLBACK_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S"))
    handler.addFilter(RunContextFilter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def setup_logging(
    config_path: Path | str | None = None,
    level: str | None = None,
    run_id: str | None = None,
) -> str:
    """Configure process-wide logging and bind a run identifier.

    Safe to call more than once; the previous handler set is replaced.

    Args:
        config_path: dictConfig YAML to load. Defaults to
            ``configs/logging.yaml``; a fallback config is used if unreadable.
        level: Root log level override, e.g. ``"DEBUG"``.
        run_id: Identifier to stamp on records. Generated when omitted.

    Returns:
        The active run identifier.
    """
    force_utf8_streams()

    active_run_id = run_id or new_run_id()
    set_run_id(active_run_id)

    path = Path(config_path) if config_path is not None else _DEFAULT_LOGGING_CONFIG
    resolved_level = (level or "INFO").upper()

    try:
        raw = path.read_text(encoding="utf-8")
        config: dict[str, Any] = yaml.safe_load(raw) or {}
        if level is not None:
            config.setdefault("root", {})["level"] = resolved_level
        logging.config.dictConfig(config)
    except (OSError, ValueError, TypeError, KeyError, ImportError) as exc:
        _apply_fallback_config(resolved_level)
        logging.getLogger(__name__).warning(
            "Could not apply logging config from %s (%s); using fallback console config.",
            path,
            exc,
        )

    if level is not None:
        logging.getLogger().setLevel(resolved_level)

    return active_run_id


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger.

    Thin wrapper over :func:`logging.getLogger`, used so modules import a
    single logging entry point rather than reaching for ``logging`` directly.
    """
    return logging.getLogger(name)
