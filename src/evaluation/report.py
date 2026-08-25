"""Persisting a training run as a self-describing directory.

Every run writes to its own ``results/<run_id>/`` and nothing else, so results
accumulate instead of overwriting each other (master spec §51/§52). The set of
files is fixed and named by the constants below, so the CLI, the tests, and a
future results browser all agree on what a run directory contains:

============================== ==============================================
File                           Contents
============================== ==============================================
``model.joblib``               Fitted pipeline: vectorizer + classifier
``metrics.json``               Every configured metric, per split
``classification_report.json`` scikit-learn's report, verbatim
``confusion_matrix.<fmt>``     Heatmap for the evaluation split
``class_distribution.<fmt>``   Per-class support across splits
``run_config.yaml`` / ``.json`` Resolved configuration that produced the run
``run_manifest.json``          Provenance: ids, hashes, versions, git commit
``report.md``                  Human-readable summary
``predictions_<split>.jsonl``  Per-paper predictions with confidence
============================== ==============================================

``run_config`` deliberately **excludes** ``settings.env``, which carries the
contact address used for the OpenAlex polite pool and is the natural place for a
future API key. Secrets must never be written into an artifact (master spec §32),
and a results directory is exactly the kind of thing that gets zipped and shared.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.config.settings import Settings
from src.schemas.paper import DatasetRecord
from src.utils.io import atomic_write_text, write_json, write_jsonl, write_yaml
from src.utils.logging import get_logger

__all__ = [
    "CLASSIFICATION_REPORT_NAME",
    "CLASS_DISTRIBUTION_STEM",
    "CONFUSION_MATRIX_STEM",
    "METRICS_NAME",
    "MODEL_NAME",
    "REPORT_NAME",
    "RUN_CONFIG_STEM",
    "RUN_MANIFEST_NAME",
    "predictions_file_name",
    "render_report",
    "resolved_config_payload",
    "write_classification_report",
    "write_metrics",
    "write_predictions",
    "write_report",
    "write_run_config",
    "write_run_manifest",
]

logger = get_logger(__name__)

MODEL_NAME = "model.joblib"
METRICS_NAME = "metrics.json"
CLASSIFICATION_REPORT_NAME = "classification_report.json"
RUN_MANIFEST_NAME = "run_manifest.json"
RUN_CONFIG_STEM = "run_config"
REPORT_NAME = "report.md"
CONFUSION_MATRIX_STEM = "confusion_matrix"
CLASS_DISTRIBUTION_STEM = "class_distribution"

#: How many per-class score entries to keep in the predictions file. The full
#: vector is redundant for inspection and grows with the class count.
_TOP_K_SCORES = 3

#: How many off-diagonal confusion cells the Markdown report lists.
_TOP_CONFUSIONS = 8


def predictions_file_name(split: str) -> str:
    """Return the per-paper predictions file name for one split."""
    return f"predictions_{split}.jsonl"


def resolved_config_payload(settings: Settings) -> dict[str, Any]:
    """Build the configuration snapshot for a run directory.

    Environment settings are omitted on purpose: they hold the OpenAlex contact
    address today and are where a credential would live tomorrow.

    Args:
        settings: The resolved settings the run used.

    Returns:
        A JSON- and YAML-serialisable mapping of every configuration layer that
        is safe to publish.
    """
    return {
        "app": settings.app.model_dump(mode="json"),
        "dataset": settings.dataset.model_dump(mode="json"),
        "model": settings.model.model_dump(mode="json"),
        "config_dir": str(settings.config_dir),
        "_note": (
            "Environment settings are intentionally excluded: they may contain "
            "contact addresses or credentials, which must never be written into a "
            "results artifact."
        ),
    }


def write_run_config(run_dir: Path, settings: Settings) -> list[Path]:
    """Write the resolved configuration as both YAML and JSON.

    YAML for reading, JSON for programmatic comparison between runs.
    """
    payload = resolved_config_payload(settings)
    return [
        write_yaml(run_dir / f"{RUN_CONFIG_STEM}.yaml", payload),
        write_json(run_dir / f"{RUN_CONFIG_STEM}.json", payload),
    ]


def write_metrics(run_dir: Path, metrics: dict[str, Any]) -> Path:
    """Write the full per-split metrics payload."""
    return write_json(run_dir / METRICS_NAME, metrics)


def write_classification_report(run_dir: Path, report: dict[str, Any]) -> Path:
    """Write scikit-learn's classification report."""
    return write_json(run_dir / CLASSIFICATION_REPORT_NAME, report)


def write_run_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    """Write the run provenance manifest."""
    return write_json(run_dir / RUN_MANIFEST_NAME, manifest)


def write_predictions(
    run_dir: Path,
    split: str,
    records: Sequence[DatasetRecord],
    y_pred: Sequence[Any],
    *,
    classes: Sequence[str],
    scores: np.ndarray | None = None,
    score_kind: str = "unavailable",
    multilabel: bool = False,
) -> Path:
    """Write per-paper predictions so individual errors can be inspected.

    Aggregate metrics say a model is 82 % accurate; this file says *which* papers
    it got wrong, which is what error analysis needs.

    Args:
        run_dir: Destination run directory.
        split: Split name, used in the file name.
        records: Records in prediction order.
        y_pred: Predicted labels or label sets, aligned with ``records``.
        classes: Class order matching the columns of ``scores``.
        scores: Optional ``(n_samples, n_classes)`` score matrix.
        score_kind: What ``scores`` holds — ``"probability"`` or ``"decision"``.

    Returns:
        The path written.

    Raises:
        ValueError: If ``records`` and ``y_pred`` differ in length.
    """
    if len(records) != len(y_pred):
        raise ValueError(f"{len(records)} records but {len(y_pred)} predictions for split {split}")

    class_index = {label: index for index, label in enumerate(classes)}

    def _row(position: int, record: DatasetRecord, predicted: Any) -> dict[str, Any]:
        predicted_labels = [str(label) for label in predicted] if multilabel else [str(predicted)]
        exact_match = set(record.labels) == set(predicted_labels) if multilabel else (
            record.label == predicted
        )
        payload: dict[str, Any] = {
            "paper_id": record.paper_id,
            "title": record.title,
            "true_label": record.label,
            "predicted_label": predicted_labels[0] if predicted_labels else None,
            "true_labels": list(record.labels),
            "predicted_labels": predicted_labels if multilabel else [],
            "correct": exact_match,
            "confidence_kind": score_kind,
        }
        if scores is None:
            return payload

        row = np.asarray(scores[position], dtype=float)
        ordered = np.argsort(-row)
        best = float(row[ordered[0]])
        if score_kind == "probability" or row.size < 2:
            confidence = best
        else:
            # Margin to the runner-up: how decisively this class won, which is the
            # only confidence-like reading a decision function supports.
            confidence = best - float(row[ordered[1]])
        payload["confidence"] = confidence
        payload["top_scores"] = [
            {"label": classes[index], "score": float(row[index])}
            for index in ordered[:_TOP_K_SCORES]
            if index < len(classes)
        ]
        if record.label in class_index:
            payload["true_label_score"] = float(row[class_index[record.label]])
        return payload

    rows = (
        _row(position, record, predicted)
        for position, (record, predicted) in enumerate(zip(records, y_pred, strict=True))
    )
    path = run_dir / predictions_file_name(split)
    count = write_jsonl(path, rows)
    logger.info("report | wrote %d prediction(s) to %s", count, path.name)
    return path


def _metrics_table(metrics: dict[str, Any]) -> list[str]:
    """Render headline and averaged metrics as a Markdown table."""
    lines = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Samples | {metrics['n_samples']} |",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Balanced accuracy | {metrics['balanced_accuracy']:.4f} |",
    ]
    for average, values in metrics.get("averages", {}).items():
        lines.append(
            f"| {average.capitalize()} P / R / F1 | "
            f"{values['precision']:.4f} / {values['recall']:.4f} / {values['f1']:.4f} |"
        )
    return lines


def _per_class_table(metrics: dict[str, Any]) -> list[str]:
    """Render the per-class table, worst F1 first."""
    per_class = metrics.get("per_class") or {}
    if not per_class:
        return ["_Per-class metrics were disabled for this run._"]

    rows = sorted(per_class.items(), key=lambda kv: (kv[1]["f1"], kv[0]))
    lines = ["| Class | Precision | Recall | F1 | Support |", "| --- | --- | --- | --- | --- |"]
    lines += [
        f"| {label} | {values['precision']:.3f} | {values['recall']:.3f} | "
        f"{values['f1']:.3f} | {values['support']} |"
        for label, values in rows
    ]
    return lines


def _top_confusions(metrics: dict[str, Any]) -> list[str]:
    """List the largest off-diagonal confusion cells."""
    matrix = metrics.get("confusion_matrix") or {}
    labels = matrix.get("labels") or []
    counts = matrix.get("counts")
    if counts is None:
        per_label = matrix.get("per_label") or {}
        if not per_label:
            return ["No confusion matrix is available for this split."]
        rows = sorted(
            (
                label,
                values.get("false_positive", 0),
                values.get("false_negative", 0),
            )
            for label, values in per_label.items()
        )
        lines = ["| Label | False positives | False negatives |", "| --- | --- | --- |"]
        lines += [f"| {label} | {fp} | {fn} |" for label, fp, fn in rows]
        return lines
    counts = counts or []
    offenders = [
        (counts[row][column], labels[row], labels[column])
        for row in range(len(labels))
        for column in range(len(labels))
        if row != column and counts[row][column] > 0
    ]
    if not offenders:
        return ["No off-diagonal errors on this split."]

    offenders.sort(key=lambda item: (-item[0], item[1], item[2]))
    lines = ["| True class | Predicted as | Count |", "| --- | --- | --- |"]
    lines += [
        f"| {true} | {predicted} | {count} |"
        for count, true, predicted in offenders[:_TOP_CONFUSIONS]
    ]
    return lines


def _confidence_lines(metrics: dict[str, Any]) -> list[str]:
    """Summarise the confidence block in prose, labelled by provenance."""
    confidence = metrics.get("confidence") or {}
    if not confidence.get("available"):
        return [
            "Not available for this model: "
            + str(confidence.get("reason", "no score function exposed."))
        ]

    label = (
        "Maximum class probability"
        if confidence["kind"] == "probability"
        else "Top-1 minus top-2 decision margin"
    )
    lines = [
        f"- **Statistic:** {label} (`{confidence['statistic']}`)",
        f"- **Mean / median / p10:** {confidence['mean']:.4f} / "
        f"{confidence['median']:.4f} / {confidence['p10']:.4f}",
    ]
    if confidence.get("mean_when_correct") is not None:
        lines.append(f"- **Mean on correct predictions:** {confidence['mean_when_correct']:.4f}")
    if confidence.get("mean_when_incorrect") is not None:
        lines.append(
            f"- **Mean on incorrect predictions:** {confidence['mean_when_incorrect']:.4f}"
        )
    lines.append(f"- **Caveat:** {confidence['caveat']}")
    return lines


def render_report(
    manifest: dict[str, Any],
    metrics_by_split: dict[str, dict[str, Any]],
    *,
    primary_split: str,
    artifacts: Sequence[str] = (),
) -> str:
    """Render the human-readable run summary.

    Args:
        manifest: The run manifest, read for identity and provenance.
        metrics_by_split: Metrics keyed by split name.
        primary_split: Split the headline metric and figures describe.
        artifacts: File names present in the run directory.

    Returns:
        Markdown text.
    """
    primary = metrics_by_split[primary_split]
    selection = primary.get("primary_metric", {})
    dataset = manifest.get("dataset", {})

    lines: list[str] = [
        f"# Run `{manifest.get('run_id', 'unknown')}`",
        "",
        f"**{manifest.get('model', {}).get('display_name', 'model')}** "
        f"(`{manifest.get('model', {}).get('name', '?')}`) — "
        f"{manifest.get('created_at', 'unknown time')}",
        "",
        "## Headline",
        "",
        f"- **{selection.get('name', 'primary metric')}** on `{primary_split}`: "
        f"**{selection.get('value', float('nan')):.4f}** — the configured "
        f"model-selection metric.",
        f"- Accuracy on `{primary_split}`: {primary['accuracy']:.4f} "
        f"(balanced: {primary['balanced_accuracy']:.4f})",
        "",
        "## Run provenance",
        "",
        "| Field | Value |",
        "| --- | --- |",
        f"| Run id | `{manifest.get('run_id', '?')}` |",
        f"| Seed | {manifest.get('seed', '?')} |",
        f"| Git commit | `{manifest.get('git_commit') or 'unavailable'}` |",
        f"| Label mode | {manifest.get('labels', {}).get('mode', '?')} "
        f"({manifest.get('labels', {}).get('taxonomy_level', '?')} level) |",
        f"| Classes | {manifest.get('labels', {}).get('n_classes', '?')} |",
        f"| Dataset | `{dataset.get('directory', '?')}` |",
        f"| Split sizes | {dataset.get('split_sizes', {})} |",
        f"| Train duration | {manifest.get('timing', {}).get('fit_seconds', '?')} s |",
        "",
    ]

    for split, metrics in metrics_by_split.items():
        lines += [
            f"## Metrics — `{split}`",
            "",
            *_metrics_table(metrics),
            "",
            f"### Per class — `{split}`",
            "",
            *_per_class_table(metrics),
            "",
            f"### Largest confusions — `{split}`",
            "",
            *_top_confusions(metrics),
            "",
            f"### Confidence — `{split}`",
            "",
            *_confidence_lines(metrics),
            "",
        ]

    findings = dataset.get("integrity_findings") or []
    if findings:
        lines += ["## Dataset integrity findings", ""]
        lines += [
            f"- **{item['check']}** ({item['severity']}): {item['detail']}" for item in findings
        ]
        lines.append("")

    if artifacts:
        lines += ["## Artifacts", "", *(f"- `{name}`" for name in sorted(artifacts)), ""]

    lines += [
        "## How to read this",
        "",
        "- These are **classical baselines** (TF-IDF features, linear classifier). They",
        "  exist to establish a floor that later transformer and section-attention",
        "  models must beat, not as the final system.",
        "- The vectorizer was fitted on the **training split only**, inside the",
        "  pipeline, so validation and test scores are not inflated by leakage.",
        "- Confidence values are **uncalibrated**. Nothing here has been validated as",
        "  a frequency, and an SVM margin is not a probability at all — treat both as",
        "  ranking signals only.",
        "- Metrics come from a single split, not cross-validation, so small",
        "  differences between runs are within noise.",
        "",
    ]
    return "\n".join(lines)


def write_report(run_dir: Path, text: str) -> Path:
    """Write the Markdown run summary."""
    return atomic_write_text(run_dir / REPORT_NAME, text)
