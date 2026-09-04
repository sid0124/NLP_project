"""Orchestration for one SciBERT + HAN training run (spec §9).

Mirrors :mod:`src.training.train_baseline` exactly — same run-directory layout,
same metrics payload, same manifest contract — so the API's run store, the
dashboard's Model Performance view, and the experiment comparison table treat a
HAN run like any other run. What differs:

* the model is a :class:`~src.models.han_classifier.HANClassifier`;
* the input is the full document text structured into sections and sentences
  (the hierarchical contract, spec §8), not a flattened TF-IDF string;
* the encoder must be downloaded/cached once (first run needs network).

Invariants identical to the baseline path:

* **Fitting sees the training split only.** Validation/test text reaches the
  model only through inference (master spec §9).
* **A run never overwrites another.** Fresh ``results/<run_id>/`` or refusal.
"""

from __future__ import annotations

import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config.settings import Settings
from src.evaluation.metrics import classification_report_dict, evaluate_predictions
from src.evaluation.plots import plot_class_distribution, plot_confusion_matrix
from src.evaluation.report import (
    CLASS_DISTRIBUTION_STEM,
    CONFUSION_MATRIX_STEM,
    predictions_file_name,
    render_report,
    write_classification_report,
    write_metrics,
    write_predictions,
    write_report,
    write_run_config,
    write_run_manifest,
)
from src.models.han_classifier import HANClassifier
from src.training.dataset import load_processed_dataset
from src.training.train_baseline import MODEL_NAME, _prepare_run_dir
from src.utils.io import git_commit_sha, resolve_path
from src.utils.logging import get_logger, new_run_id
from src.utils.seed import set_seed

__all__ = ["HANTrainingResult", "train_han"]

logger = get_logger(__name__)

_TRACKED_DISTRIBUTIONS: tuple[str, ...] = (
    "torch",
    "transformers",
    "scikit-learn",
    "numpy",
    "scipy",
    "joblib",
    "pydantic",
)


def _package_versions() -> dict[str, str]:
    """Record versions of the libraries that produced this run."""
    import importlib.metadata

    versions: dict[str, str] = {}
    for distribution in _TRACKED_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


class HANTrainingResult:
    """Everything one HAN training run produced."""

    def __init__(
        self,
        *,
        run_id: str,
        run_dir: Path,
        model_name: str,
        model: HANClassifier,
        metrics: dict[str, dict[str, Any]],
        manifest: dict[str, Any],
        artifacts: list[str],
        primary_split: str,
    ) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.model_name = model_name
        self.model = model
        self.metrics = metrics
        self.manifest = manifest
        self.artifacts = artifacts
        self.primary_split = primary_split

    @property
    def primary_metric(self) -> dict[str, Any]:
        """The configured model-selection metric on the primary split."""
        return self.metrics[self.primary_split]["primary_metric"]


def _score_split_han(
    model: HANClassifier,
    split,
    *,
    classes: list[str],
    settings: Settings,
) -> tuple[dict[str, Any], list[str], np.ndarray | None, str]:
    """Predict on one split with the HAN and compute its metrics."""
    evaluation = settings.model.evaluation
    y_pred = model.predict(split.texts)
    scores, score_kind = model.predict_proba(split.texts), "probability"

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
        hamming_loss_requested=False,
        multilabel=False,
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


def train_han(
    settings: Settings,
    *,
    data_dir: Path | str | None = None,
    results_dir: Path | str | None = None,
    run_id: str | None = None,
) -> HANTrainingResult:
    """Train, evaluate, and persist the SciBERT + HAN model.

    Args:
        settings: Resolved configuration.
        data_dir: Processed dataset directory; defaults to ``processed_dir``.
        results_dir: Parent directory for run directories.
        run_id: Explicit run id; defaults to a timestamped id.

    Returns:
        An :class:`HANTrainingResult`.

    Raises:
        DatasetNotFoundError: If the processed dataset is missing.
        RunExistsError: If the run directory already holds files.
        RuntimeError: If the encoder stack is unavailable.
    """
    dataset_dir = resolve_path(data_dir) if data_dir else settings.paths.resolved("processed_dir")
    results_root = (
        resolve_path(results_dir) if results_dir else settings.paths.resolved("results_dir")
    )

    seed = set_seed(settings.seed)
    run_id = run_id or new_run_id(prefix="scibert_han")
    run_dir = _prepare_run_dir(results_root, run_id)
    started = datetime.now(UTC)

    logger.info("train_han | run=%s data=%s seed=%d", run_id, dataset_dir, seed)

    dataset = load_processed_dataset(
        dataset_dir, expected_text_fields=settings.app.text.fields
    )
    train = dataset["train"]
    classes = dataset.classes or sorted(
        {label for split in dataset.splits.values() for label in split.labels}
    )

    # ---- build + fit: training split only ---------------------------------
    model = HANClassifier.from_config(settings)
    fit_started = time.perf_counter()
    model.fit(train.texts, train.labels)
    fit_seconds = round(time.perf_counter() - fit_started, 3)
    logger.info("train_han | fitted on %d training document(s) in %.3fs", len(train), fit_seconds)

    # ---- evaluate on held-out splits ---------------------------------------
    training_config = settings.model.training
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
        metrics, y_pred, scores, score_kind = _score_split_han(
            model, split, classes=classes, settings=settings
        )
        metrics_by_split[split_name] = metrics
        if split_name == primary_split:
            report_json = classification_report_dict(
                split.labels, y_pred, classes=classes, multilabel=False
            )
        write_predictions(
            run_dir,
            split_name,
            split.records,
            y_pred,
            classes=classes,
            scores=scores,
            score_kind=score_kind,
            multilabel=False,
        )
        artifacts.append(predictions_file_name(split_name))

    if not metrics_by_split:
        raise ValueError(
            f"No split could be evaluated. Requested {scored}, but none is present "
            f"and non-empty in {dataset_dir}."
        )
    if primary_split not in metrics_by_split:
        primary_split = next(iter(metrics_by_split))
        logger.warning("eval | headline metrics fall back to split '%s'", primary_split)

    if training_config.save_model:
        artifacts.append(model.save(run_dir / MODEL_NAME).name)
        logger.info("train | saved HAN model to %s", MODEL_NAME)

    artifacts.append(write_metrics(run_dir, metrics_by_split).name)
    if report_json is not None:
        artifacts.append(write_classification_report(run_dir, report_json).name)
    artifacts += [path.name for path in write_run_config(run_dir, settings)]

    # ---- plots -------------------------------------------------------------
    plots = settings.model.evaluation.plots
    suffix = plots.figure_format.lstrip(".")
    if plots.confusion_matrix:
        for split_name, metrics in metrics_by_split.items():
            stem = (
                CONFUSION_MATRIX_STEM
                if split_name == primary_split
                else f"{CONFUSION_MATRIX_STEM}_{split_name}"
            )
            normalized = metrics["confusion_matrix"].get("normalized") is not None
            path = plot_confusion_matrix(
                metrics["confusion_matrix"]["normalized" if normalized else "counts"],
                metrics["confusion_matrix"]["labels"],
                run_dir / f"{stem}.{suffix}",
                title=f"Confusion matrix — {split_name}",
                normalized=normalized,
                dpi=plots.dpi,
            )
            artifacts.append(path.name)
    if plots.class_distribution:
        path = plot_class_distribution(
            {name: split.class_counts for name, split in dataset.splits.items()},
            run_dir / f"{CLASS_DISTRIBUTION_STEM}.{suffix}",
            dpi=plots.dpi,
        )
        artifacts.append(path.name)

    # ---- manifest (same contract as baseline runs) --------------------------
    finished = datetime.now(UTC)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "created_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "seed": seed,
        "git_commit": git_commit_sha(),
        "model": {
            "name": "scibert_han",
            "display_name": "SciBERT + Hierarchical Attention Network",
            "encoder": settings.model.encoder.model_name,
            "architecture": "HAN (sentence + section attention over SciBERT embeddings)",
            "hyperparameters": {
                "han": settings.model.han.model_dump(mode="json"),
                "longdoc": settings.model.longdoc.model_dump(mode="json"),
                "neural_training": settings.model.neural_training.model_dump(mode="json"),
            },
        },
        "labels": {
            "mode": settings.labels.mode,
            "taxonomy_level": settings.labels.taxonomy_level,
            "n_classes": len(classes),
            "classes": classes,
        },
        "dataset": {
            "directory": str(dataset_dir),
            "split_sizes": dataset.split_sizes,
            "file_hashes": dataset.dataset_hashes,
            "build_created_at": dataset.manifest.get("created_at"),
            "source": dataset.manifest.get("source"),
            "integrity_findings": dataset.findings_as_dicts(),
        },
        "evaluation": {
            "splits_scored": list(metrics_by_split),
            "primary_split": primary_split,
            "primary_metric": metrics_by_split[primary_split]["primary_metric"],
            "confidence_kind": "probability",
            "attention_available": True,
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

    logger.info("train_han | run complete: %s", run_dir)
    return HANTrainingResult(
        run_id=run_id,
        run_dir=run_dir,
        model_name="scibert_han",
        model=model,
        metrics=metrics_by_split,
        manifest=manifest,
        artifacts=sorted(set(artifacts)),
        primary_split=primary_split,
    )