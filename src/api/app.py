"""FastAPI application factory.

Assembled here rather than at module scope so that tests can build an app against
a temporary results directory, and so a second app in the same interpreter does
not inherit the first one's cached settings.

Two decisions worth stating.

**The dashboard is served from the API's own origin.** ``server.serve_frontend``
mounts ``frontend/`` at ``/``. A same-origin ``fetch`` is not a cross-origin
request, so in the default configuration CORS never enters the picture — which
removes the opportunity to make the usual mistake of opening the API to ``*``
while debugging and shipping it that way (master spec §40). The CORS middleware is
still installed for the separate-dev-server case, but only with explicitly listed
origins.

**Middleware order is deliberate.** Starlette runs middleware outermost-first in
the order added, so the size limit is added last and therefore runs first: an
oversized body is refused before CORS or header logic touches it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import FileResponse, JSONResponse

from src.api.deps import get_settings, get_store, require_api_key, reset_caches
from src.api.routers import dashboard, papers, system
from src.api.runstore import RunUnavailableError
from src.api.security import BodySizeLimitMiddleware, SecurityHeadersMiddleware
from src.config.settings import Settings
from src.utils.io import resolve_path
from src.utils.logging import get_logger, setup_logging

__all__ = ["API_PREFIX", "create_app"]

logger = get_logger(__name__)

#: Every JSON route lives under this prefix, which keeps the static mount at
#: ``/`` from ever shadowing an endpoint.
API_PREFIX = "/api"

#: No module-level ``app`` object. Building one at import time would read
#: ``configs/`` as a side effect of importing the module, which makes the import
#: fail on a misconfigured checkout and prevents a test from supplying its own
#: settings. Serve it with ``python scripts/serve_api.py``, or point uvicorn at
#: the factory: ``uvicorn --factory src.api.app:create_app``.

_DESCRIPTION = """
Read-only API over a completed training run.

It trains nothing: `scripts/train_baseline.py` writes a run to `results/<run_id>/`
and this service reads it. Endpoints that need a model load the run's own
`model.joblib`, so an ad-hoc classification uses exactly the estimator the
reported metrics describe.

Features that are not built report `available: false` with a reason rather than
returning approximations. `GET /api/meta` lists them.
"""


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warn loudly at startup about the two states that matter.

    An open API and an absent run are both survivable and both worth one line in
    the log, because each is invisible until a request fails.
    """
    settings: Settings = app.state.settings
    if not settings.api.security.require_api_key:
        logger.warning(
            "api | no API key required — correct for %s, wrong for anything reachable "
            "from another host. Set ARIS_API_KEY and security.require_api_key to enable.",
            settings.api.server.host,
        )
    if settings.api.cors.allows_any_origin:
        logger.warning("api | CORS allows any origin. Replace '*' with explicit origins.")

    try:
        run = get_store().active()
        logger.info(
            "api | serving run %s (%s, %d classes)",
            run.run_id,
            run.model_display_name,
            len(run.classes),
        )
        if run.is_synthetic_corpus:
            logger.warning(
                "api | run %s was trained on the generated test corpus, not real papers. "
                "The dashboard labels this; do not read its metrics as results.",
                run.run_id,
            )
    except RunUnavailableError as exc:
        logger.warning("api | starting with no usable run: %s", exc)

    yield


def _install_error_handlers(app: FastAPI) -> None:
    """Give every deliberate failure one body shape.

    Without this, a client has to cope with FastAPI's ``detail`` string, its
    validation-error list, and whatever an unhandled exception produces. One
    shape means one error path in the frontend.
    """

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Registered against Starlette's class, not FastAPI's subclass, so it also
        # covers the router's own 404 and 405. Otherwise an unmatched path returns
        # Starlette's bare ``{"detail": ...}`` and the client needs two error paths.
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": _slug(exc.status_code),
                "detail": exc.detail if isinstance(exc.detail, str) else str(exc.detail),
            },
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "validation_error",
                "detail": "The request did not match the expected shape.",
                # Field paths and messages only. The submitted values are omitted:
                # echoing input back into an error body is how a payload ends up
                # in a log aggregator that was never meant to hold it.
                "hint": "; ".join(
                    f"{'.'.join(str(part) for part in item.get('loc', ()))}: "
                    f"{item.get('msg', 'invalid')}"
                    for item in exc.errors()
                )
                or None,
            },
        )

    @app.exception_handler(RunUnavailableError)
    async def _run_error(request: Request, exc: RunUnavailableError) -> JSONResponse:
        # 503, not 500: the request was well-formed and the code is correct — the
        # server simply has no run loaded, which training resolves.
        logger.warning("api | run unavailable on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=503,
            content={
                "error": "run_unavailable",
                "detail": str(exc),
                "hint": "Train a run: python scripts/train_baseline.py --model tfidf_logreg",
            },
        )


def _slug(status_code: int) -> str:
    """Map an HTTP status to a stable machine-readable error key.

    Every status the API can produce deliberately is listed, including the two
    that arrive by accident rather than by a raise: 405 from a method typo against
    a real path, and 500 from ``require_api_key`` with no key configured. Falling
    through to ``"error"`` for those would give the client a key it cannot branch
    on for the two failures most likely to appear during setup.
    """
    return {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        413: "payload_too_large",
        422: "validation_error",
        500: "internal_error",
        501: "not_implemented",
        503: "run_unavailable",
    }.get(status_code, "error")


def _mount_frontend(app: FastAPI, settings: Settings) -> None:
    """Serve the dashboard from the API's own origin.

    Mounted last, at ``/``, so it can never shadow an ``/api`` route. Missing
    directory is a warning rather than an error: the API is useful on its own, and
    a deployment may serve the static files from a CDN.
    """
    directory = resolve_path(settings.api.server.frontend_dir)
    if not directory.is_dir():
        logger.warning(
            "api | frontend_dir '%s' does not exist; serving JSON only.", directory
        )
        return

    index = directory / "index.html"

    @app.get("/", include_in_schema=False)
    async def _index() -> FileResponse:
        """Serve the dashboard shell."""
        return FileResponse(index)

    # html=True makes the mount serve index.html for a bare directory request,
    # which is what a reload of a sub-path needs.
    app.mount("/", StaticFiles(directory=directory, html=True), name="frontend")
    logger.info("api | serving dashboard from %s", directory)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application.

    Args:
        settings: Resolved configuration. Defaults to the process-wide settings,
            reloaded from ``configs/`` so a factory call picks up an edited file.

    Returns:
        The configured application.
    """
    if settings is None:
        reset_caches()
        settings = get_settings()

    setup_logging(level=settings.log_level)

    app = FastAPI(
        title=f"{settings.app.project.name} API",
        version=settings.app.project.version,
        description=_DESCRIPTION,
        lifespan=_lifespan,
        docs_url=f"{API_PREFIX}/docs",
        openapi_url=f"{API_PREFIX}/openapi.json",
        redoc_url=None,
    )
    app.state.settings = settings

    # Added in reverse order of execution: Starlette runs the last-added
    # middleware first, and the size limit must run before anything reads a body.
    if settings.api.security.send_security_headers:
        app.add_middleware(SecurityHeadersMiddleware)
    if settings.api.cors.allow_origins:
        cors = settings.api.cors
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors.allow_origins,
            allow_credentials=cors.allow_credentials,
            allow_methods=cors.allow_methods,
            allow_headers=cors.allow_headers,
        )
    app.add_middleware(
        BodySizeLimitMiddleware, max_bytes=settings.api.security.max_request_bytes
    )

    _install_error_handlers(app)

    # Health is mounted without the auth dependency so a monitoring probe keeps
    # working when the key rotates. Everything else is protected by a
    # router-level dependency, so a new endpoint is covered by default rather
    # than by remembering to annotate it.
    app.include_router(system.public_router, prefix=API_PREFIX)
    guarded = [Depends(require_api_key)]
    app.include_router(system.router, prefix=API_PREFIX, dependencies=guarded)
    app.include_router(dashboard.router, prefix=API_PREFIX, dependencies=guarded)
    app.include_router(papers.router, prefix=API_PREFIX, dependencies=guarded)

    if settings.api.server.serve_frontend:
        _mount_frontend(app, settings)

    return app
