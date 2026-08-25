"""Orchestration for one baseline training run.

This module owns the sequence — load, fit, evaluate, persist — and nothing else.
Model construction lives in :mod:`src.models.baselines`, metric computation in
:mod:`src.evaluation.metrics`, and artifact layout in
:mod:`src.evaluation.report`. Keeping orchestration separate is what lets the
same run be driven from the CLI, from a test, or later from a job queue.

Two invariants hold regardless of caller:

* **Fitting sees the training split only.** ``pipeline.fit(train.texts, ...)`` is
  the single fit call in the module; validation and test text reach the estimator
  only through ``predict`` (master spec §9).
* **A run never overwrites another.** Each run gets a fresh ``results/<run_id>/``
  and refuses to start if that directory already holds files, so a repeated run
  id can never silently replace an earlier result (master spec §51/§52).
"""

from __future__ import annotations

import importlib.metadata
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config.settings import Settings
from src.evaluation.metrics import classification_report_dict, evaluate_predictions
from src.evaluation.plots import plot_class_distribution, plot_confusion_matrix
from src.evaluation.report import (
    CLASS_DISTRIBUTION_STEM,
    CONFUSION_MATRIX_STEM,
    MODEL_NAME,
    predictions_file_name,
    render_report,
    write_classification_report,
    write_metrics,
    write_predictions,
    write_report,
    write_run_config,
    write_run_manifest,
)
from src.models.baselines import (
    VECTORIZER_STEP,
    build_baseline,
    prediction_scores,
    resolved_params,
)
from src.training.dataset import ProcessedDataset, SplitData, load_processed_dataset
from src.utils.io import ensure_dir, git_commit_sha, resolve_path
from src.utils.logging import get_logger, new_run_id
from src.utils.seed import set_seed

__all__ = [
    "RunExistsError",
    "TrainingResult",
    "train_baseline",
]

logger = get_logger(__name__)

#: Distributions whose versions are recorded, so a metric can be traced to the
#: exact library that produced it. Import names differ from distribution names
#: (``sklearn`` vs ``scikit-learn``), so distributions are named here.
_TRACKED_DISTRIBUTIONS: tuple[str, ...] = (
    "scikit-learn",
    "numpy",
    "scipy",
    "joblib",
    "matplotlib",
    "pydantic",
)


class RunExistsError(FileExistsError):
    """Raised when the target run directory already contains files."""


@dataclass
class TrainingResult:
    """Everything one training run produced.

    Attributes:
        run_id: Identifier, also the results directory name.
        run_dir: Directory holding every artifact.
        model_name: Baseline key from ``configs/model.yaml``.
        pipeline: The fitted pipeline.
        metrics: Metrics keyed by split name.
        manifest: The run provenance manifest.
        artifacts: File names written into ``run_dir``.
        primary_split: Split the headline metric describes.
    """

    run_id: str
    run_dir: Path
    model_name: str
    pipeline: Any
    metrics: dict[str, dict[str, Any]]
    manifest: dict[str, Any]
    artifacts: list[str]
    primary_split: str

    @property
    def primary_metric(self) -> dict[str, Any]:
        """The configured model-selection metric and its value."""
        return self.metrics[self.primary_split]["primary_metric"]


def _package_versions() -> dict[str, str]:
    """Record the versions of libraries that influence the numbers."""
    versions: dict[str, str] = {"python": platform.python_version()}
    for distribution in _TRACKED_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            # A missing optional distribution should not abort a run; the gap is
            # itself worth recording.
            versions[distribution] = "not installed"
    return versions


def _prepare_run_dir(results_dir: Path, run_id: str) -> Path:
    """Create the run directory, refusing to reuse a populated one.

    Raises:
        RunExistsError: If the directory exists and is not empty.
    """
    run_dir = resolve_path(results_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise RunExistsError(
            f"Run directory already contains files: {run_dir}. Runs are never "
            f"overwritten; use a different run id."
        )
    return ensure_dir(run_dir)


def _score_split(
    pipeline: Any,
    split: SplitData,
    *,
    classes: Sequence[str],
    settings: Settings,
) -> tuple[dict[str, Any], list[str], np.ndarray | None, str]:
    """Predict on one split and compute its metrics.

    Returns:
        ``(metrics, y_pred, scores, score_kind)``.
    """
    evaluation = settings.model.evaluation
    y_pred = [str(label) for label in pipeline.predict(split.texts)]
    scores, score_kind = prediction_scores(pipeline, split.texts)

    metrics = evaluate_predictions(
        split.labels,
        y_pred,
        classes=classes,
        averages=evaluation.averages,
        primary_metric=evaluation.primary_metric,
        per_class=evaluation.per_class_metrics,
        normalize_confusion_matrix=evaluation.plots.normalize_confusion_matrix,
        scores=scores,
        score_kind=score_kind,
        hamming_loss_requested=evaluation.hamming_loss,
        multilabel=settings.labels.is_multilabel,
    )
    logger.info(
        "eval | %s: %s=%.4f accuracy=%.4f (n=%d)",
        split.name,
        metrics["primary_metric"]["name"],
        metrics["primary_metric"]["value"],
        metrics["accuracy"],
        metrics["n_samples"],
    )
    return metrics, y_pred, scores, score_kind


def _write_plots(
    run_dir: Path,
    dataset: ProcessedDataset,
    metrics_by_split: dict[str, dict[str, Any]],
    *,
    primary_split: str,
    settings: Settings,
) -> list[str]:
    """Render the configured figures and return the file names written."""
    plots = settings.model.evaluation.plots
    written: list[str] = []
    suffix = plots.figure_format.lstrip(".")

    if plots.confusion_matrix:
        for split, metrics in metrics_by_split.items():
            matrix = metrics["confusion_matrix"]
            normalized = matrix.get("normalized") is not None
            # One file per scored split; the evaluation split keeps the bare name
            # so a reader never has to guess which figure is the headline.
            stem = (
                CONFUSION_MATRIX_STEM
                if split == primary_split
                else f"{CONFUSION_MATRIX_STEM}_{split}"
            )
            path = plot_confusion_matrix(
                matrix["normalized"] if normalized else matrix["counts"],
                matrix["labels"],
                run_dir / f"{stem}.{suffix}",
                title=f"Confusion matrix — {split}",
                normalized=normalized,
                dpi=plots.dpi,
            )
            written.append(path.name)

    if plots.class_distribution:
        path = plot_class_distribution(
            {name: split.class_counts for name, split in dataset.splits.items()},
            run_dir / f"{CLASS_DISTRIBUTION_STEM}.{suffix}",
            classes=dataset.classes or None,
            dpi=plots.dpi,
        )
        written.append(path.name)

    return written


def train_baseline(
    settings: Settings,
    model_name: str,
    *,
    data_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
    run_id: str | None = None,
) -> TrainingResult:
    """Train, evaluate, and persist one configured baseline.

    Args:
        settings: Resolved configuration; supplies the seed, the label mode, the
            model definition, and every evaluation option.
        model_name: Baseline key from ``configs/model.yaml``.
        data_dir: Processed dataset directory. Defaults to
            ``paths.processed_dir``.
        results_dir: Parent directory for run directories. Defaults to
            ``paths.results_dir``.
        run_id: Explicit run id. Defaults to a timestamped id prefixed with the
            model name.

    Returns:
        A :class:`TrainingResult` describing the run and its artifacts.

    Raises:
        KeyError: If ``model_name`` is not configured.
        DatasetNotFoundError: If the processed dataset is missing or incomplete.
        NotImplementedError: If ``labels.mode`` is ``multilabel``.
        RunExistsError: If the run directory already holds files.
    """
    dataset_dir = resolve_path(data_dir) if data_dir else settings.paths.resolved("processed_dir")
    results_root = (
        resolve_path(results_dir) if results_dir else settings.paths.resolved("results_dir")
    )
    baseline_config = settings.model.baseline(model_name)  # raises early on a bad name
    training_config = settings.model.training

    seed = set_seed(settings.seed)
    run_id = run_id or new_run_id(prefix=model_name)
    run_dir = _prepare_run_dir(results_root, run_id)
    started = datetime.now(UTC)

    logger.info(
        "train | run=%s model=%s data=%s seed=%d", run_id, model_name, dataset_dir, seed
    )

    dataset = load_processed_dataset(
        dataset_dir, expected_text_fields=settings.app.text.fields
    )
    train = dataset["train"]

    # The vocabulary file fixes class order; fall back to labels observed in the
    # data only if the build wrote no vocabulary at all.
    classes = dataset.classes or sorted(
        {label for split in dataset.splits.values() for label in split.labels}
    )

    pipeline = build_baseline(
        settings.model, model_name, seed=seed, multilabel=settings.labels.is_multilabel
    )

    # ---- the one and only fit call: training split, nothing else -----------
    fit_started = time.perf_counter()
    pipeline.fit(train.texts, train.labels)
    fit_seconds = round(time.perf_counter() - fit_started, 3)

    vectorizer = pipeline.named_steps.get(VECTORIZER_STEP)
    vocabulary_size = len(getattr(vectorizer, "vocabulary_", {}) or {})
    logger.info(
        "train | fitted in %.3fs on %d training document(s); %d features",
        fit_seconds,
        len(train),
        vocabulary_size,
    )

    # Score the configured evaluation split, plus test when requested and present.
    scored: list[str] = [training_config.eval_split]
    if training_config.score_test and training_config.eval_split != "test":
        scored.append("test")

    metrics_by_split: dict[str, dict[str, Any]] = {}
    artifacts: list[str] = []
    primary_split = training_config.eval_split
    report_json: dict[str, Any] | None = None

    for split_name in scored:
        split = dataset.get(split_name)
        if split is None:
            logger.warning(
                "eval | split '%s' is absent or empty in %s; skipping it",
                split_name,
                dataset_dir,
            )
            continue

        metrics, y_pred, scores, score_kind = _score_split(
            pipeline, split, classes=classes, settings=settings
        )
        metrics_by_split[split_name] = metrics
        if split_name == primary_split:
            report_json = classification_report_dict(split.labels, y_pred, classes=classes)

        write_predictions(
            run_dir,
            split_name,
            split.records,
            y_pred,
            classes=classes,
            scores=scores,
            score_kind=score_kind,
        )
        artifacts.append(predictions_file_name(split_name))

    if not metrics_by_split:
        raise ValueError(
            f"No split could be evaluated. Requested {scored}, but none is present and "
            f"non-empty in {dataset_dir}."
        )
    if primary_split not in metrics_by_split:
        # The configured split was missing; report against whatever was scored
        # rather than pretending the headline describes the intended split.
        primary_split = next(iter(metrics_by_split))
        logger.warning("eval | headline metrics fall back to split '%s'", primary_split)

    if training_config.save_model:
        joblib.dump(pipeline, run_dir / MODEL_NAME)
        artifacts.append(MODEL_NAME)
        logger.info("train | saved fitted pipeline to %s", MODEL_NAME)

    artifacts.append(write_metrics(run_dir, metrics_by_split).name)
    if report_json is not None:
        artifacts.append(write_classification_report(run_dir, report_json).name)
    artifacts += [path.name for path in write_run_config(run_dir, settings)]
    artifacts += _write_plots(
        run_dir, dataset, metrics_by_split, primary_split=primary_split, settings=settings
    )

    finished = datetime.now(UTC)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "seed": seed,
        "git_commit": git_commit_sha(),
        "model": {
            "name": model_name,
            "display_name": baseline_config.display_name,
            "vectorizer": baseline_config.vectorizer,
            "classifier_type": baseline_config.classifier.type,
            "configured_params": {
                "vectorizer": settings.model.vectorizer_for(model_name).params,
                "classifier": baseline_config.classifier.params,
            },
            "resolved_params": resolved_params(pipeline),
            "n_features": vocabulary_size,
        },
        "labels": {
            "mode": settings.labels.mode,
            "taxonomy_level": settings.labels.taxonomy_level,
            "n_classes": len(classes),
            "classes": classes,
        },
        "text": {
            "configured_fields": list(settings.app.text.fields),
            "built_with_fields": (
                ((dataset.manifest.get("config") or {}).get("text") or {}).get("fields")
            ),
            "note": (
                "Record text was composed at dataset build time; training consumes "
                "it verbatim and does not re-derive it from paper fields."
            ),
        },
        "dataset": {
            "directory": str(dataset_dir),
            "split_sizes": dataset.split_sizes,
            "file_hashes": dataset.dataset_hashes,
            "build_created_at": dataset.manifest.get("created_at"),
            "build_git_commit": dataset.manifest.get("git_commit"),
            "build_seed": dataset.manifest.get("seed"),
            "source": dataset.manifest.get("source"),
            "integrity_findings": dataset.findings_as_dicts(),
        },
        "evaluation": {
            "splits_scored": list(metrics_by_split),
            "primary_split": primary_split,
            "primary_metric": metrics_by_split[primary_split]["primary_metric"],
            "confidence_kind": metrics_by_split[primary_split]["confidence"].get("kind"),
        },
        "timing": {
            "fit_seconds": fit_seconds,
            "total_seconds": round((finished - started).total_seconds(), 3),
        },
        "versions": _package_versions(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
    }

    artifacts.append(write_run_manifest(run_dir, manifest).name)
    report_text = render_report(
        manifest, metrics_by_split, primary_split=primary_split, artifacts=artifacts
    )
    artifacts.append(write_report(run_dir, report_text).name)

    logger.info("train | run complete: %s", run_dir)
    return TrainingResult(
        run_id=run_id,
        run_dir=run_dir,
        model_name=model_name,
        pipeline=pipeline,
        metrics=metrics_by_split,
        manifest=manifest,
        artifacts=sorted(set(artifacts)),
        primary_split=primary_split,
    )
