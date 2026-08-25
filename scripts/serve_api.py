"""Serve the dashboard and its API.

Reads a completed training run out of ``results/<run_id>/`` and serves both the
JSON API and the static dashboard from one origin. It trains nothing and writes
nothing, so it is safe to restart at any time.

Serving both from one origin is the point: a same-origin ``fetch`` is not a
cross-origin request, so the default configuration needs no CORS allowances at
all (master spec §40).

Examples:
    Serve on the configured host and port::

        python scripts/serve_api.py

    Pin a specific run and open the browser automatically::

        python scripts/serve_api.py --run-id m1-tfidf_logreg --open

    Expose it on the network, which requires an API key::

        ARIS_API_KEY=... python scripts/serve_api.py --host 0.0.0.0 --require-api-key
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser

import _bootstrap  # noqa: F401

from src.api.runstore import RunUnavailableError, is_valid_run_id
from src.config.settings import load_settings
from src.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Bind address. Defaults to api.yaml server.host (loopback).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Bind port. Defaults to api.yaml server.port.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Serve a specific run instead of the most recently finished one.",
    )
    parser.add_argument(
        "--require-api-key",
        action="store_true",
        help=(
            "Require the API-key header on every route except /api/health. "
            "ARIS_API_KEY must be set in the environment."
        ),
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Restart on source changes. Development only.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        dest="open_browser",
        help="Open the dashboard in the default browser once the server is up.",
    )
    parser.add_argument("--log-level", default=None, help="Override the root log level.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Returns:
        Process exit status: 0 on a clean shutdown, 2 for a bad ``--run-id``,
        3 when ``--require-api-key`` is given without a key in the environment.
    """
    args = parse_args(argv)

    settings = load_settings()
    setup_logging(level=args.log_level or settings.log_level)

    # Applied to the settings object rather than passed down separately, so the
    # app and every dependency observe one resolved configuration.
    if args.host:
        settings.api.server.host = args.host
    if args.port:
        settings.api.server.port = args.port
    if args.require_api_key:
        settings.api.security.require_api_key = True
    if args.run_id:
        if not is_valid_run_id(args.run_id):
            logger.error("serve | '%s' is not a valid run id", args.run_id)
            return 2
        settings.api.runs.default_run_id = args.run_id

    if settings.api.security.require_api_key and not settings.env.aris_api_key:
        logger.error(
            "serve | --require-api-key was given but ARIS_API_KEY is not set, so no "
            "request could be authenticated. Add it to .env and retry."
        )
        return 3

    if settings.api.server.host not in {"127.0.0.1", "localhost", "::1"} and not (
        settings.api.security.require_api_key
    ):
        logger.warning(
            "serve | binding %s with no API key required — anything that can reach this "
            "host can read the API. Pass --require-api-key.",
            settings.api.server.host,
        )

    # Imported here, after settings are resolved, so a configuration error is
    # reported by this script rather than surfacing as an import-time traceback.
    import uvicorn

    from src.api.app import create_app
    from src.api.runstore import RunStore

    try:
        run = RunStore(settings).active()
    except RunUnavailableError as exc:
        run = None
        logger.warning("serve | no usable run yet: %s", exc)

    # 0.0.0.0 is a bind address, not an address to visit; print loopback instead.
    visitable = "127.0.0.1" if settings.api.server.host == "0.0.0.0" else settings.api.server.host  # noqa: S104
    url = f"http://{visitable}:{settings.api.server.port}"
    print(f"\nDashboard:  {url}")
    print(f"API docs:   {url}/api/docs")
    print(f"Health:     {url}/api/health")
    if run is not None:
        print(f"Serving run: {run.run_id} ({run.model_display_name})")
    else:
        print("Serving run: none — train one with scripts/train_baseline.py")
    print("\nCtrl+C to stop.\n")

    if args.open_browser:
        # Delayed so the browser requests a socket that is already listening.
        threading.Timer(1.2, webbrowser.open, args=(url,)).start()

    if args.reload:
        # --reload needs an import string, because the reloader re-imports in a
        # fresh process where an already-built app object cannot be handed over.
        # That process calls create_app() with no arguments, so it reads
        # configs/ directly — the flags above do not reach it.
        if any([args.host, args.port, args.run_id, args.require_api_key]):
            logger.warning(
                "serve | --reload re-reads configs/ in the worker process, so --host, "
                "--port, --run-id, and --require-api-key are ignored. Edit "
                "configs/api.yaml or drop --reload."
            )
        uvicorn.run(
            "src.api.app:create_app",
            factory=True,
            host=settings.api.server.host,
            port=settings.api.server.port,
            reload=True,
            log_level=settings.log_level.lower(),
        )
    else:
        uvicorn.run(
            create_app(settings),
            host=settings.api.server.host,
            port=settings.api.server.port,
            log_level=settings.log_level.lower(),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
