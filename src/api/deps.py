"""Shared dependencies for the API routers.

Three things live here, all of them about doing expensive work exactly once.

:func:`get_settings` and :func:`get_store` are cached for the process lifetime.
Loading four YAML files and scanning the results directory per request would be
wasteful; more importantly, the run store holds a fitted pipeline and a
transformed corpus matrix, and rebuilding those per request would make the
similarity endpoint unusable.

:func:`get_active_run` converts a missing or broken run into an HTTP 503 with the
command that would fix it. That is the failure mode a fresh clone hits — no run
trained yet — and a stack trace in the browser console is a poor way to learn
that ``scripts/train_baseline.py`` has not been run.

The cache is process-wide, which means a test that changes configuration must
call :func:`reset_caches`. Tests do; :mod:`src.api.app` calls it at startup.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, Query, Request, status

from src.api.runstore import LoadedRun, RunStore, RunUnavailableError
from src.api.security import check_api_key
from src.config.settings import Settings, load_settings
from src.utils.logging import get_logger

__all__ = [
    "ActiveRun",
    "Pagination",
    "PaginationParams",
    "SettingsDep",
    "StoreDep",
    "get_active_run",
    "get_settings",
    "get_store",
    "require_api_key",
    "reset_caches",
]

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide resolved settings."""
    return load_settings()


@lru_cache(maxsize=1)
def get_store() -> RunStore:
    """Return the process-wide run store."""
    return RunStore(get_settings())


def reset_caches() -> None:
    """Clear the settings and store caches.

    Needed whenever configuration or the results directory changes underneath a
    live process — which in practice means tests, and the app factory at startup
    so a second factory call in the same interpreter does not inherit the first
    one's settings.
    """
    get_settings.cache_clear()
    get_store.cache_clear()


SettingsDep = Annotated[Settings, Depends(get_settings)]
StoreDep = Annotated[RunStore, Depends(get_store)]


async def require_api_key(request: Request, settings: SettingsDep) -> None:
    """Enforce the API key when one is configured (master spec §40).

    Applied as a router-level dependency rather than per route, so a new endpoint
    is protected by default. ``/api/health`` is mounted outside that router
    precisely so a monitoring probe does not need the secret.

    Raises:
        HTTPException: 401 when the key is required and absent or wrong.
    """
    header = settings.api.security.api_key_header
    check_api_key(settings, request.headers.get(header))


def get_active_run(store: StoreDep) -> LoadedRun:
    """Return the run the dashboard should read.

    Raises:
        HTTPException: 503 when no usable run exists. 503 rather than 404: the
            endpoint is correct and the client is not at fault — the server has
            nothing loaded yet, and that is a temporary state a training run
            resolves.
    """
    try:
        return store.active()
    except RunUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


ActiveRun = Annotated[LoadedRun, Depends(get_active_run)]


@dataclass(frozen=True)
class PageWindow:
    """A resolved slice of a result list."""

    limit: int
    offset: int


def pagination(
    settings: SettingsDep,
    limit: Annotated[int | None, Query(ge=1, description="Rows per page.")] = None,
    offset: Annotated[int, Query(ge=0, description="Rows to skip.")] = 0,
) -> PageWindow:
    """Resolve a requested page against the configured bounds.

    A function rather than a class-based dependency: FastAPI resolves annotations
    from the callable's ``__globals__``, which a class does not have, so with
    ``from __future__ import annotations`` a class dependency cannot see the
    aliases defined above it.

    An over-large ``limit`` is clamped rather than rejected. Pagination is a hint,
    the client still receives a valid page, and the ``limit`` echoed in the
    response body reports what it actually got.
    """
    config = settings.api.pagination
    return PageWindow(
        limit=min(limit or config.default_page_size, config.max_page_size),
        offset=offset,
    )


Pagination = Annotated[PageWindow, Depends(pagination)]
