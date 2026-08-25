"""HTTP layer over a completed training run.

The API is read-only by design. Training is a batch job that writes a seeded,
manifested run directory; serving is a separate process that reads one. Keeping
them apart is what makes a number in the dashboard traceable to the run that
produced it.

Build the app with :func:`src.api.app.create_app`.
"""

from __future__ import annotations

__all__ = ["create_app"]


def create_app(*args: object, **kwargs: object) -> object:
    """Re-export :func:`src.api.app.create_app` without importing it eagerly.

    A plain ``from src.api.app import create_app`` here would pull FastAPI in
    whenever anything under ``src.api`` is imported — including
    :mod:`src.api.runstore`, which a test or script may want on its own.
    """
    from src.api.app import create_app as factory

    return factory(*args, **kwargs)  # type: ignore[arg-type]
