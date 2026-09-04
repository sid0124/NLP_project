"""Train and evaluate SciBERT + HAN on the processed dataset.

Writes the same self-describing run directory as scripts/train_baseline.py
(metrics, figures, the fitted model, per-paper predictions, the resolved
configuration, and a provenance manifest) under results/<run_id>/.

The first run downloads the SciBERT weights (~440 MB) from huggingface.co;
later runs load them from the local cache and work offline.

Examples:
    Train the final model on the committed sample corpus::

        python scripts/train_han.py

    Give the run a stable id (used by configs/api.yaml -> runs.default_run_id)::

        python scripts/train_han.py --run-id scibert_han_v1
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import _bootstrap  # noqa: F401

from src.config.settings import Settings, load_settings
from src.training.dataset import DatasetNotFoundError
from src.training.train_baseline import RunExistsError
from src.training.train_han import train_han
from src.utils.logging import get_logger, setup_logging
from src.utils.seed import set_seed

logger = get_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs", help="Configuration directory (default: configs)."
    )
    parser.add_argument(
        "--data-dir", default=None, help="Processed dataset directory (default: processed_dir)."
    )
    parser.add_argument(
        "--results-dir", default=None, help="Parent directory for run directories."
    )
    parser.add_argument("--run-id", default=None, help="Explicit run id.")
    parser.add_argument("--seed", type=int, default=None, help="Override project.seed.")
    parser.add_argument("--log-level", default=None, help="Override logging.level.")
    return parser.parse_args(argv)


def _overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI flags into a nested override mapping."""
    overrides: dict[str, Any] = {}
    if args.seed is not None:
        overrides["project"] = {"seed": args.seed}
    if args.log_level is not None:
        overrides["logging"] = {"level": args.log_level.upper()}
    return overrides


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    settings = load_settings(args.config, overrides=_overrides_from_args(args))
    setup_logging(level=settings.log_level)
    set_seed(settings.seed)

    logger.info(
        "train_han | encoder=%s data=%s seed=%d",
        settings.model.encoder.model_name,
        args.data_dir or settings.paths.resolved("processed_dir"),
        settings.seed,
    )

    try:
        result = train_han(
            settings,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            run_id=args.run_id,
        )
    except DatasetNotFoundError as exc:
        logger.error("train_han | dataset not ready: %s", exc)
        return 2
    except RuntimeError as exc:
        logger.error("train_han | encoder unavailable: %s", exc)
        return 4
    except RunExistsError as exc:
        logger.error("train_han | %s", exc)
        return 6

    primary = result.primary_metric
    print(f"\nRun complete: {result.run_dir}")
    print(f"  {primary['name']} ({result.primary_split}): {primary['value']:.4f}")
    print(f"  artifacts: {', '.join(result.artifacts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())