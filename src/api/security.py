"""Request-level protections for the API (master spec §40).

Four independent measures, each covering something the others do not:

* **API key** — an optional shared secret, checked in constant time. Optional is
  the honest design for a loopback dev server; what matters is that turning it on
  is a single environment variable rather than a code change, so the deployment
  path does not require inventing an auth layer under time pressure.
* **Body size limit** — enforced from ``Content-Length`` *before* the body is
  read, so an oversized request costs a header parse rather than memory.
* **Security headers** — a small fixed set, aimed at what this app actually does:
  it serves its own static dashboard and returns JSON.
* **Filename sanitising** — unused today, because there is no upload endpoint.
  Kept here, tested, so that when the PDF phase lands the helper already exists
  and the temptation to inline ``Path(filename).name`` and call it done does not
  arise.

There is deliberately **no upload endpoint**. Master spec §40 requires validated,
size-limited, isolated document handling and "never trust uploaded PDF content";
none of that is buildable before the parser exists, and an endpoint that accepts
a file it cannot safely process is worse than a button that says "not yet".
"""

from __future__ import annotations

import re
import secrets
import unicodedata
from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response

from src.config.settings import Settings
from src.utils.logging import get_logger

__all__ = [
    "SECURITY_HEADERS",
    "BodySizeLimitMiddleware",
    "SecurityHeadersMiddleware",
    "api_key_error",
    "check_api_key",
    "safe_filename",
]

logger = get_logger(__name__)

#: Response headers applied to every response.
#:
#: The CSP is tight because this app's own needs are narrow: it serves one HTML
#: page, its own CSS and ES modules, and inline SVG. ``'unsafe-inline'`` appears
#: for styles only — the dashboard sets computed bar widths and chart geometry
#: through the ``style`` attribute, which a style-src nonce cannot cover. Scripts
#: get no such exemption, which is the half that matters for XSS.
SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "font-src 'self'; "
        "object-src 'none'; "
        "base-uri 'none'; "
        "form-action 'none'; "
        "frame-ancestors 'none'"
    ),
}

#: Characters permitted in a sanitised filename.
_FILENAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

#: Longest filename produced by :func:`safe_filename`, leaving room for a
#: directory prefix inside common path length limits.
_FILENAME_MAX = 96


def api_key_error() -> HTTPException:
    """Return the 401 raised for a missing or wrong API key.

    The message names the header but never echoes the submitted value, which
    would put a candidate secret into the logs of every proxy in the path.
    """
    from src.api.deps import get_settings

    header = get_settings().api.security.api_key_header
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=f"Missing or invalid API key. Send it in the {header} header.",
        headers={"WWW-Authenticate": header},
    )


def check_api_key(settings: Settings, supplied: str | None) -> None:
    """Verify a supplied API key against the configured secret.

    Enforcement requires *both* that a key is configured and that
    ``security.require_api_key`` is set — either alone is ambiguous. A configured
    key with the flag off is a key staged for later; the flag on with no key
    configured is a misconfiguration that would otherwise lock everyone out with
    a 401 that no value can satisfy, so it fails as a 500 naming the cause
    instead.

    Args:
        settings: Resolved configuration.
        supplied: Value of the API-key header, or ``None`` if absent.

    Raises:
        HTTPException: 401 when a key is required and the supplied value does not
            match; 500 when enforcement is on but no key is configured.
    """
    if not settings.api.security.require_api_key:
        return

    expected = settings.env.aris_api_key
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "security.require_api_key is true but ARIS_API_KEY is not set in the "
                "environment, so no request can be authenticated. Set it in .env."
            ),
        )

    # compare_digest, not ==: a short-circuiting comparison leaks the length of
    # the matching prefix through response timing.
    if supplied is None or not secrets.compare_digest(supplied, expected):
        raise api_key_error()


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject request bodies larger than the configured ceiling.

    Checked against the declared ``Content-Length`` before the handler runs, so
    an oversized body is refused for the cost of a header read. A request that
    omits the header and streams its body is not covered here — uvicorn's own
    limits apply to that case, and this API has no streaming endpoint.
    """

    def __init__(self, app: Callable[..., Awaitable[None]], *, max_bytes: int) -> None:
        """Bind the middleware to a byte ceiling.

        Args:
            app: The wrapped ASGI application.
            max_bytes: Largest permitted request body.
        """
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Refuse oversized requests, otherwise pass through."""
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                length = int(declared)
            except ValueError:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "error": "bad_request",
                        "detail": "Content-Length is not an integer.",
                    },
                )
            if length > self.max_bytes:
                logger.warning(
                    "api | rejected %d-byte body on %s (limit %d)",
                    length,
                    request.url.path,
                    self.max_bytes,
                )
                return JSONResponse(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "error": "payload_too_large",
                        "detail": (
                            f"Request body is {length} bytes; the limit is "
                            f"{self.max_bytes}."
                        ),
                        "hint": "Raise security.max_request_bytes in configs/api.yaml.",
                    },
                )
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Attach :data:`SECURITY_HEADERS` to every response."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Add the header set, without overwriting anything already present."""
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        return response


def safe_filename(raw: str, *, fallback: str = "upload") -> str:
    """Reduce a client-supplied filename to a safe basename.

    Not currently called: there is no upload endpoint yet. It exists because
    master spec §40 requires filename sanitising for the PDF phase, and the
    correct implementation is easy to get subtly wrong — ``Path(x).name`` alone
    leaves Windows device names, right-to-left override characters, and
    all-punctuation names intact.

    Defence order matters. Unicode is normalised first so that a decomposed or
    homoglyph form cannot smuggle a separator past the character filter; then
    every path separator is dropped by taking only the final segment of both
    conventions; then anything outside a conservative allowlist is collapsed.

    Args:
        raw: The client-supplied name.
        fallback: Stem used when nothing usable survives.

    Returns:
        A basename containing only ASCII letters, digits, ``.``, ``_``, and
        ``-``, never empty, never starting with a dot, and length-capped.
    """
    normalised = unicodedata.normalize("NFKC", raw)
    # Strip control and format characters, including the bidi overrides used to
    # disguise an extension ("evil‮gnp.exe" displays as "evilexe.png").
    normalised = "".join(ch for ch in normalised if unicodedata.category(ch) not in {"Cc", "Cf"})

    # Both separators, because a name arriving from a Windows client may use
    # backslashes that POSIX ``PurePath`` would treat as ordinary characters.
    basename = normalised.replace("\\", "/").rsplit("/", 1)[-1]

    cleaned = _FILENAME_SAFE.sub("_", basename).strip("._-")
    if not cleaned:
        return fallback

    # Preserve the extension when truncating: a length cap that eats the suffix
    # turns a validated ".pdf" into an extensionless file.
    if len(cleaned) > _FILENAME_MAX:
        stem, _, suffix = cleaned.rpartition(".")
        if stem and len(suffix) <= 8:
            keep = _FILENAME_MAX - len(suffix) - 1
            cleaned = f"{stem[:keep]}.{suffix}"
        else:
            cleaned = cleaned[:_FILENAME_MAX]
    return cleaned
