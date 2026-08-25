"""Classification metrics (master spec §13).

Pure functions over label sequences: nothing here reads configuration, touches
the filesystem, or holds a model. That keeps the numbers testable against
hand-computed values and keeps the reporting layer free to decide *what* to
persist.

Three deliberate choices shape the output:

* **The class list comes from the dataset vocabulary, not from the labels
  present in a split.** Index position is part of the on-disk contract, so a rare
  class absent from validation still occupies its row in the confusion matrix and
  its entry in the per-class table. Metrics for such a class read 0, which is the
  truth, rather than vanishing.
* **``zero_division=0``** everywhere. A class with no predictions has undefined
  precision; scikit-learn's default emits a warning and substitutes 0 anyway, so
  this makes the substitution explicit and the run quiet and deterministic.
* **Confidence is labelled by provenance.** A probability and an SVM decision
  margin are both "confidence" colloquially, but only one is on a [0, 1] scale
  with a distributional meaning. The score kind travels with the numbers
  (master spec §14/§15) so no report can present a margin as a probability.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    hamming_loss,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)

__all__ = [
    "AVERAGE_NAMES",
    "METRIC_NAMES",
    "classification_report_dict",
    "confidence_summary",
    "confusion_matrix_data",
    "evaluate_predictions",
    "multilabel_confusion_matrix_data",
    "parse_primary_metric",
    "primary_metric_value",
    "required_averages",
]

#: Averaging strategies this module can compute, in reporting order.
AVERAGE_NAMES: tuple[str, ...] = ("macro", "micro", "weighted")

#: Metrics computed for every averaging strategy and for every class.
METRIC_NAMES: tuple[str, ...] = ("precision", "recall", "f1")

#: Metrics that stand alone rather than being averaged over classes.
_GLOBAL_METRICS: tuple[str, ...] = ("accuracy", "balanced_accuracy")


def parse_primary_metric(name: str) -> tuple[str | None, str]:
    """Split a metric identifier such as ``"macro_f1"`` into its parts.

    Args:
        name: Metric identifier from ``evaluation.primary_metric``.

    Returns:
        ``(average, metric)``, where ``average`` is ``None`` for a global metric
        such as ``"accuracy"``.

    Raises:
        ValueError: If the identifier names an unsupported average or metric. The
            message lists every valid option, since this is almost always a
            configuration typo.
    """
    if name in _GLOBAL_METRICS:
        return None, name

    average, _, metric = name.rpartition("_")
    if average in AVERAGE_NAMES and metric in METRIC_NAMES:
        return average, metric

    valid = [*_GLOBAL_METRICS, *(f"{a}_{m}" for a in AVERAGE_NAMES for m in METRIC_NAMES)]
    raise ValueError(f"Unsupported primary_metric '{name}'. Valid options: {valid}")


def required_averages(configured: Sequence[str], primary_metric: str) -> list[str]:
    """Return the averages to compute, honouring configuration and the selector.

    The average the primary metric depends on is always included, even if it was
    omitted from ``evaluation.averages`` — otherwise model selection would have no
    number to select on.

    Raises:
        ValueError: If a configured average is not supported, or if
            ``primary_metric`` is malformed.
    """
    unknown = [name for name in configured if name not in AVERAGE_NAMES]
    if unknown:
        raise ValueError(
            f"Unsupported entries in evaluation.averages: {unknown}. "
            f"Valid options: {list(AVERAGE_NAMES)}"
        )

    needed = list(dict.fromkeys(configured))
    average, _ = parse_primary_metric(primary_metric)
    if average is not None and average not in needed:
        needed.append(average)
    # Report in canonical order so two runs' metrics.json files diff cleanly.
    return [name for name in AVERAGE_NAMES if name in needed]


def _averaged_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    classes: Sequence[str],
    averages: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Compute precision/recall/F1 under each requested averaging strategy."""
    result: dict[str, dict[str, float]] = {}
    for average in averages:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(classes),
            average=average,
            zero_division=0,
        )
        result[average] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return result


def _multilabel_matrix(values: Sequence[Any], *, classes: Sequence[str]) -> np.ndarray:
    """Convert label sets or indicator rows into an ``(n, n_classes)`` matrix."""
    if isinstance(values, np.ndarray):
        matrix = np.asarray(values, dtype=int)
        if matrix.ndim != 2 or matrix.shape[1] != len(classes):
            raise ValueError(
                f"multi-label indicator matrix has shape {matrix.shape}, "
                f"expected (n_samples, {len(classes)})"
            )
        return matrix

    class_index = {label: index for index, label in enumerate(classes)}
    rows: list[list[int]] = []
    for item in values:
        row = [0] * len(classes)
        labels = [item] if isinstance(item, str) else list(item)
        for label in labels:
            if label in class_index:
                row[class_index[label]] = 1
        rows.append(row)
    return np.asarray(rows, dtype=int)


def _per_class_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], *, classes: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Compute precision/recall/F1/support for every class in the vocabulary."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(classes),
        average=None,
        zero_division=0,
    )
    return {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(classes)
    }


def _multilabel_averaged_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    averages: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Compute multi-label precision/recall/F1 under each averaging strategy."""
    result: dict[str, dict[str, float]] = {}
    for average in averages:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=average,
            zero_division=0,
        )
        result[average] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }
    return result


def _multilabel_per_class_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, *, classes: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Compute one-vs-rest precision/recall/F1/support for every label."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        average=None,
        zero_division=0,
    )
    return {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(classes)
    }


def confusion_matrix_data(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    classes: Sequence[str],
    normalize: bool = True,
) -> dict[str, Any]:
    """Build a confusion matrix keyed to the full class vocabulary.

    Args:
        y_true: Gold labels.
        y_pred: Predicted labels.
        classes: Full vocabulary, defining row/column order.
        normalize: Also return rows normalised by true-class support, i.e. a
            per-class recall breakdown.

    Returns:
        ``{"labels", "counts", "normalized"}``; ``normalized`` is ``None`` when
        not requested. Rows with no support normalise to zeros rather than NaN so
        the result stays JSON-serialisable.
    """
    counts = confusion_matrix(y_true, y_pred, labels=list(classes))
    normalized: list[list[float]] | None = None
    if normalize:
        row_totals = counts.sum(axis=1, keepdims=True)
        safe = np.divide(
            counts,
            row_totals,
            out=np.zeros(counts.shape, dtype=float),
            where=row_totals > 0,
        )
        normalized = [[float(value) for value in row] for row in safe]

    return {
        "labels": list(classes),
        "counts": [[int(value) for value in row] for row in counts],
        "normalized": normalized,
    }


def multilabel_confusion_matrix_data(
    y_true: np.ndarray, y_pred: np.ndarray, *, classes: Sequence[str]
) -> dict[str, Any]:
    """Build a per-label one-vs-rest confusion matrix payload."""
    matrices = multilabel_confusion_matrix(y_true, y_pred)
    per_label = {}
    for index, label in enumerate(classes):
        tn, fp, fn, tp = matrices[index].ravel()
        per_label[label] = {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }
    return {
        "labels": list(classes),
        "counts": None,
        "normalized": None,
        "per_label": per_label,
        "note": "Multi-label evaluation uses one-vs-rest confusion counts per label.",
    }


def classification_report_dict(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    classes: Sequence[str],
    multilabel: bool = False,
) -> dict[str, Any]:
    """Return scikit-learn's classification report as a nested mapping.

    Persisted verbatim as ``classification_report.json`` so results can be
    compared against any other scikit-learn project without re-deriving the
    layout.
    """
    if multilabel:
        report = classification_report(
            _multilabel_matrix(y_true, classes=classes),
            _multilabel_matrix(y_pred, classes=classes),
            target_names=list(classes),
            output_dict=True,
            zero_division=0,
        )
    else:
        report = classification_report(
            y_true,
            y_pred,
            labels=list(classes),
            output_dict=True,
            zero_division=0,
        )
    # numpy scalars would serialise only via a custom encoder; flatten here so the
    # object is plain-JSON regardless of who writes it.
    return {
        key: (
            {inner: float(number) for inner, number in value.items()}
            if isinstance(value, dict)
            else float(value)
        )
        for key, value in report.items()
    }


def confidence_summary(
    scores: np.ndarray | None,
    kind: str,
    *,
    correct: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Summarise per-prediction confidence, labelled by what it actually is.

    For probabilities, confidence is the highest class probability. For decision
    margins it is the gap between the best and second-best decision values — a
    monotone but **uncalibrated** stand-in that must never be printed as a
    percentage (master spec §15).

    Args:
        scores: ``(n_samples, n_classes)`` scores, or ``None`` when unavailable.
        kind: ``"probability"``, ``"decision"``, or ``"unavailable"``, as returned
            by :func:`src.models.baselines.prediction_scores`.
        correct: Per-sample correctness, used to contrast confident hits with
            confident misses.

    Returns:
        A mapping always carrying ``kind`` and ``available``; distribution
        statistics and a ``caveat`` are present only when scores exist.
    """
    if scores is None or kind == "unavailable" or scores.size == 0:
        return {
            "kind": kind,
            "available": False,
            "reason": (
                "The estimator exposes neither predict_proba nor decision_function, "
                "so no per-prediction confidence could be recorded."
            ),
        }

    ordered = np.sort(scores, axis=1)
    if kind == "probability":
        values = ordered[:, -1]
        caveat = (
            "Maximum class probability. Uncalibrated: these are the model's raw "
            "outputs, not validated frequencies (master spec §15)."
        )
    else:
        # A one-class problem cannot have a runner-up; fall back to the raw value.
        values = (
            ordered[:, -1] - ordered[:, -2] if ordered.shape[1] >= 2 else np.abs(ordered[:, -1])
        )
        caveat = (
            "Top-1 minus top-2 decision-function margin, NOT a probability. It is "
            "unbounded and uncalibrated; do not render it as a percentage."
        )

    summary: dict[str, Any] = {
        "kind": kind,
        "available": True,
        "statistic": "max_probability" if kind == "probability" else "top1_minus_top2_margin",
        "caveat": caveat,
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }

    if correct is not None and len(correct) == len(values):
        flags = np.asarray(correct, dtype=bool)
        # Confidence that is higher on hits than on misses is the property that
        # makes a review threshold meaningful at all; report it rather than a
        # single mean that hides it.
        summary["mean_when_correct"] = float(np.mean(values[flags])) if flags.any() else None
        summary["mean_when_incorrect"] = (
            float(np.mean(values[~flags])) if (~flags).any() else None
        )

    return summary


def primary_metric_value(metrics: dict[str, Any], primary_metric: str) -> float:
    """Extract the model-selection metric from an evaluation payload.

    Raises:
        ValueError: If the identifier is unsupported.
        KeyError: If the payload lacks the average the identifier needs.
    """
    average, metric = parse_primary_metric(primary_metric)
    if average is None:
        return float(metrics[metric])
    try:
        return float(metrics["averages"][average][metric])
    except KeyError:
        raise KeyError(
            f"primary_metric '{primary_metric}' needs the '{average}' average, which "
            f"is absent from these metrics. Available: {sorted(metrics.get('averages', {}))}"
        ) from None


def evaluate_predictions(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    classes: Sequence[str],
    averages: Sequence[str] = AVERAGE_NAMES,
    primary_metric: str = "macro_f1",
    per_class: bool = True,
    normalize_confusion_matrix: bool = True,
    scores: np.ndarray | None = None,
    score_kind: str = "unavailable",
    hamming_loss_requested: bool = False,
    multilabel: bool = False,
) -> dict[str, Any]:
    """Score one split and return every configured metric in one payload.

    Args:
        y_true: Gold labels.
        y_pred: Predicted labels, aligned with ``y_true``.
        classes: Full class vocabulary, fixing row and column order.
        averages: Averaging strategies to compute.
        primary_metric: Identifier of the model-selection metric; its average is
            computed even if omitted from ``averages``.
        per_class: Include the per-class table.
        normalize_confusion_matrix: Include row-normalised counts.
        scores: Optional per-class scores for confidence reporting.
        score_kind: What ``scores`` contains.
        hamming_loss_requested: Whether ``evaluation.hamming_loss`` is enabled.
        multilabel: Whether the run is multi-label.

    Returns:
        A JSON-serialisable mapping with ``n_samples``, global metrics,
        ``averages``, ``per_class``, ``confusion_matrix``, ``confidence``,
        ``hamming_loss``, and ``primary_metric``.

    Raises:
        ValueError: If the label sequences differ in length or are empty, or if a
            configured average or primary metric is unsupported.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} labels but y_pred has {len(y_pred)}")
    if not len(y_true):
        raise ValueError("Cannot evaluate an empty split.")
    if not classes:
        raise ValueError("An empty class vocabulary cannot be evaluated against.")

    wanted = required_averages(averages, primary_metric)

    if multilabel:
        true_matrix = _multilabel_matrix(y_true, classes=classes)
        pred_matrix = _multilabel_matrix(y_pred, classes=classes)
        averaged = _multilabel_averaged_metrics(true_matrix, pred_matrix, averages=wanted)
        metrics: dict[str, Any] = {
            "n_samples": len(y_true),
            "n_classes": len(classes),
            # In multi-label mode sklearn's accuracy is exact-set match.
            "accuracy": float(accuracy_score(true_matrix, pred_matrix)),
            "balanced_accuracy": averaged.get("macro", {}).get("recall", 0.0),
            "averages": averaged,
            "per_class": (
                _multilabel_per_class_metrics(true_matrix, pred_matrix, classes=classes)
                if per_class
                else {}
            ),
            "confusion_matrix": multilabel_confusion_matrix_data(
                true_matrix, pred_matrix, classes=classes
            ),
            "confidence": confidence_summary(
                scores,
                score_kind,
                correct=[
                    bool(np.array_equal(t, p))
                    for t, p in zip(true_matrix, pred_matrix, strict=True)
                ],
            ),
        }
        metrics["hamming_loss"] = {
            "requested": hamming_loss_requested,
            "value": float(hamming_loss(true_matrix, pred_matrix))
            if hamming_loss_requested
            else None,
            "note": (
                "Fraction of label decisions that are wrong across all samples "
                "and classes."
                if hamming_loss_requested
                else "Not requested."
            ),
        }
        metrics["primary_metric"] = {
            "name": primary_metric,
            "value": primary_metric_value(metrics, primary_metric),
        }
        return metrics

    metrics: dict[str, Any] = {
        "n_samples": len(y_true),
        "n_classes": len(classes),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # Mean per-class recall: the honest headline on an imbalanced corpus,
        # where plain accuracy flatters a model that ignores small classes.
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "averages": _averaged_metrics(y_true, y_pred, classes=classes, averages=wanted),
        "per_class": (
            _per_class_metrics(y_true, y_pred, classes=classes) if per_class else {}
        ),
        "confusion_matrix": confusion_matrix_data(
            y_true, y_pred, classes=classes, normalize=normalize_confusion_matrix
        ),
        "confidence": confidence_summary(
            scores,
            score_kind,
            correct=[t == p for t, p in zip(y_true, y_pred, strict=True)],
        ),
    }

    # Hamming loss is a multi-label metric. In multi-class it degenerates to
    # 1 - accuracy, so reporting it would duplicate an existing number under a
    # name implying something else. The request is acknowledged and explained.
    metrics["hamming_loss"] = {
        "requested": hamming_loss_requested,
        "value": None,
        "note": (
            "Not computed: Hamming loss is a multi-label metric and reduces to "
            "1 - accuracy in multi-class mode."
            if not multilabel
            else "Multi-label evaluation is not implemented yet."
        ),
    }

    metrics["primary_metric"] = {
        "name": primary_metric,
        "value": primary_metric_value(metrics, primary_metric),
    }
    return metrics
