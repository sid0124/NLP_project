"""Train and evaluate one classical baseline on the processed dataset.

Reads the leakage-safe splits under ``data/processed`` (or ``--data-dir``), fits
the selected baseline on the training split alone, evaluates it on the configured
split (and on test when configured), and writes a self-describing run directory
under ``results/<run_id>/`` — metrics, figures, the fitted model, per-paper
predictions, the resolved configuration, and a provenance manifest.

Nothing here touches the network or fits a model outside the pipeline, so a run
is fully reproducible from committed data and configuration.

Examples:
    Train the TF-IDF + Logistic Regression baseline::

        python scripts/train_baseline.py --model tfidf_logreg

    Train the SVM baseline against a specific dataset with a fixed seed::

        python scripts/train_baseline.py --model tfidf_svm --data-dir data/processed --seed 7
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import _bootstrap  # noqa: F401

from src.config.settings import Settings, load_settings
from src.training.dataset import DatasetNotFoundError
from src.training.train_baseline import RunExistsError, train_baseline
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
        "--model",
        required=True,
        help="Baseline key from configs/model.yaml, e.g. tfidf_logreg or tfidf_svm.",
    )
    parser.add_argument(
        "--config", default="configs", help="Configuration directory (default: configs)."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Processed dataset directory. Defaults to paths.processed_dir.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Parent directory for run directories. Defaults to paths.results_dir.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Explicit run id (results/<run_id>/). Defaults to a timestamped id.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override project.seed.")
    parser.add_argument(
        "--log-level",
        default=None,
        help="Override logging.level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args(argv)


def _overrides_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI flags into a nested override mapping for the config loader.

    Overrides are passed to the loader rather than written back to YAML, so a
    one-off run never mutates the committed configuration.
    """
    overrides: dict[str, Any] = {}
    if args.seed is not None:
        overrides["project"] = {"seed": args.seed}
    if args.log_level is not None:
        overrides["logging"] = {"level": args.log_level.upper()}
    return overrides


def _validate_model_name(settings: Settings, model_name: str) -> None:
    """Fail early with a helpful list if the model is not configured.

    Raises:
        SystemExit: If ``model_name`` names no configured baseline.
    """
    available = sorted(settings.model.baselines)
    if model_name not in available:
        raise SystemExit(
            f"Unknown --model '{model_name}'. Configured baselines: {available}"
        )


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = parse_args(argv)
    settings = load_settings(args.config, overrides=_overrides_from_args(args))
    setup_logging(level=settings.log_level)
    set_seed(settings.seed)

    _validate_model_name(settings, args.model)

    logger.info(
        "train | model=%s data=%s seed=%d mode=%s",
        args.model,
        args.data_dir or settings.paths.resolved("processed_dir"),
        settings.seed,
        settings.labels.mode,
    )

    try:
        result = train_baseline(
            settings,
            args.model,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            run_id=args.run_id,
        )
    except DatasetNotFoundError as exc:
        logger.error("train | dataset not ready: %s", exc)
        return 2
    except NotImplementedError as exc:
        logger.error("train | unsupported configuration: %s", exc)
        return 4
    except RunExistsError as exc:
        logger.error("train | %s", exc)
        return 6

    primary = result.primary_metric
    logger.info(
        "train | done: %s=%.4f on %s -> %s",
        primary["name"],
        primary["value"],
        result.primary_split,
        result.run_dir,
    )
    # A concise, copy-pasteable pointer to the artifacts, distinct from the log.
    print(f"\nRun complete: {result.run_dir}")
    print(f"  {primary['name']} ({result.primary_split}): {primary['value']:.4f}")
    print(f"  artifacts: {', '.join(result.artifacts)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
